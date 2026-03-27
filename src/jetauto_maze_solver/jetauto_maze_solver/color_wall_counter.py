#!/usr/bin/env python3
"""
Colored wall counting for Gazebo + maze_navigator stack (no RViz required).

Uses the same camera rules as maze_navigator (camera_color_config): HSV bands,
minimum area fraction, winner selection. LiDAR + centroid bearing estimates wall
pose for deduplication (front/back).

Extra gating: stable frames, strict max range, centroid + LiDAR bearing (reject
side glances when passing a wall), front LiDAR must show a nearby obstacle (reject
long straight corridors), enlarged back-face dedupe radius for same wall.

Optional: publish /jetauto/camera/color_debug for rqt_image_view (off by default).
"""

from __future__ import annotations

import math
import os
from datetime import datetime

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan

from jetauto_maze_solver.camera_color_config import (
    COLOR_MIN_AREA_FRAC,
    dominant_wall_color_from_bgr,
)


def yaw_from_quat(q) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def range_at(
    ranges,
    angle_min: float,
    angle_inc: float,
    deg: float,
    window_deg: float = 15.0,
    max_r: float = 3.0,
) -> float:
    rad = math.radians(deg)
    hw = math.radians(window_deg)
    vals = [
        r
        for i, r in enumerate(ranges)
        if abs(angle_min + i * angle_inc - rad) <= hw and math.isfinite(r) and r > 0.05
    ]
    return min(vals) if vals else max_r


_PT_ORDER = ('vermelho', 'verde', 'azul')
_COLOR_LABELS = {'azul': 'A', 'verde': 'V', 'vermelho': 'R'}

_COLOR_BGR = {
    'vermelho': (0, 0, 255),
    'verde': (0, 255, 0),
    'azul': (255, 0, 0),
}


class ColorWallCounter(Node):
    # Same-side duplicate detections (odom noise / wobble)
    WALL_DEDUPE_DIST = 2.4
    # Opposite approach to same thin wall — must exceed wall thickness + both projections
    WALL_DEDUPE_DIST_BACK = 5.5
    # Cap for LiDAR projection; keep >= longest corridor sightline you expect
    WALL_MAX_PROJ = 4.0
    # UFPB depth_camera.urdf.xacro horizontal_fov
    CAMERA_HFOV = 1.2

    def __init__(self):
        super().__init__('color_wall_counter')

        self.declare_parameter('image_topic', '/jetauto/camera/image_raw')
        self.declare_parameter('odom_topic', '/odometry/filtered')
        self.declare_parameter('scan_topic', '/jetauto/lidar/scan')
        self.declare_parameter('wall_dedupe_dist', self.WALL_DEDUPE_DIST)
        self.declare_parameter('wall_dedupe_dist_back', self.WALL_DEDUPE_DIST_BACK)
        self.declare_parameter('wall_max_proj', self.WALL_MAX_PROJ)
        self.declare_parameter('camera_hfov', self.CAMERA_HFOV)
        self.declare_parameter('color_min_area_fraction', COLOR_MIN_AREA_FRAC)
        self.declare_parameter('publish_debug_image', False)
        # Require N consecutive camera frames with same dominant color (reduces flicker / false triggers)
        self.declare_parameter('stable_color_frames', 3)
        # Max range along color bearing to accept a count (stay strict — avoids “far” counts)
        self.declare_parameter('max_wall_distance_to_count', 1.65)
        # Centroid must stay near image centre (reject corners / walls glimpsed from the side)
        self.declare_parameter('centroid_max_abs', 0.32)
        # |bearing from colour blob| must be below this (deg); side walls while passing exceed it
        self.declare_parameter('max_lidar_bearing_deg', 20.0)
        # Min range straight ahead must be <= this (m); open corridor ahead => do not count
        self.declare_parameter('max_front_range_for_count', 2.05)
        self.declare_parameter('front_window_deg', 14.0)
        # Optional: |dist(bearing) - dist(straight ahead)| must be <= this (m) when counting
        self.declare_parameter('max_range_vs_front_disagree', 0.75)

        img_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        scan_topic = self.get_parameter('scan_topic').get_parameter_value().string_value
        self.WALL_DEDUPE_DIST = float(self.get_parameter('wall_dedupe_dist').value)
        self.WALL_DEDUPE_DIST_BACK = float(self.get_parameter('wall_dedupe_dist_back').value)
        self.WALL_MAX_PROJ = float(self.get_parameter('wall_max_proj').value)
        self.CAMERA_HFOV = float(self.get_parameter('camera_hfov').value)
        self._min_area_frac = float(self.get_parameter('color_min_area_fraction').value)
        self._publish_debug = self.get_parameter('publish_debug_image').value
        self._stable_frames = max(1, int(self.get_parameter('stable_color_frames').value))
        self._max_dist_count = float(self.get_parameter('max_wall_distance_to_count').value)
        self._centroid_max_abs = float(self.get_parameter('centroid_max_abs').value)
        self._max_bearing_deg = float(self.get_parameter('max_lidar_bearing_deg').value)
        self._max_front_range = float(self.get_parameter('max_front_range_for_count').value)
        self._front_window_deg = float(self.get_parameter('front_window_deg').value)
        self._max_range_disagree = float(self.get_parameter('max_range_vs_front_disagree').value)

        self.bridge = CvBridge()
        self.create_subscription(Image, img_topic, self._image_cb, 10)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)
        self.create_subscription(LaserScan, scan_topic, self._scan_cb, 10)

        self.debug_pub = self.create_publisher(Image, '/jetauto/camera/color_debug', 10)

        self.current_pose = None
        self.latest_scan = None
        self._prev_color = None
        self._streak_color = None
        self._streak_count = 0
        self._seen_walls = []

        self.counts = {k: 0 for k in _PT_ORDER}
        self.detection_positions = {k: [] for k in _PT_ORDER}

        self.get_logger().info(
            f'color_wall_counter — max_dist={self._max_dist_count}m bearing<={self._max_bearing_deg}° '
            f'front<={self._max_front_range}m dedupe_back={self.WALL_DEDUPE_DIST_BACK}m'
        )

    def _odom_cb(self, msg: Odometry):
        self.current_pose = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            yaw_from_quat(msg.pose.pose.orientation),
        )

    def _scan_cb(self, msg: LaserScan):
        self.latest_scan = msg

    def _publish_debug_image(self, bgr: np.ndarray, masks: dict, detected: str | None):
        if not self._publish_debug:
            return
        overlay = bgr.copy()
        h, w = bgr.shape[:2]
        cv2.line(overlay, (w // 2, 0), (w // 2, h), (80, 80, 80), 1)

        if detected is not None and detected in masks:
            mask = masks[detected]
            clr = _COLOR_BGR[detected]
            tint = np.zeros_like(bgr)
            tint[mask > 0] = clr
            cv2.addWeighted(tint, 0.5, overlay, 0.5, 0, overlay)
            cv2.putText(
                overlay,
                detected.upper(),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                clr,
                2,
            )
            M = cv2.moments(mask)
            if M['m00'] > 0:
                cx_px = int(M['m10'] / M['m00'])
                cy_px = int(M['m01'] / M['m00'])
                norm_x = (cx_px - w / 2.0) / w
                lidar_deg = -norm_x * math.degrees(self.CAMERA_HFOV)
                cv2.drawMarker(overlay, (cx_px, cy_px), clr, cv2.MARKER_CROSS, 20, 2)
                cv2.line(overlay, (w // 2, cy_px), (cx_px, cy_px), clr, 1)
                cv2.putText(
                    overlay,
                    f'lidar {lidar_deg:+.1f}deg',
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    clr,
                    2,
                )
        else:
            cv2.putText(
                overlay,
                'Sem cor',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (200, 200, 200),
                2,
            )
        try:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(overlay, 'bgr8'))
        except Exception:
            pass

    def _image_cb(self, msg: Image):
        if self.current_pose is None:
            return

        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as exc:
            self.get_logger().warn(f'cv_bridge error: {exc}')
            return

        color, masks, centroid_x = dominant_wall_color_from_bgr(
            bgr, min_area_fraction=self._min_area_frac
        )
        self._publish_debug_image(bgr, masks, color)

        if color is None:
            self._prev_color = None
            self._streak_color = None
            self._streak_count = 0
            return

        if (
            centroid_x is not None
            and self._centroid_max_abs < 1.0
            and abs(centroid_x) > self._centroid_max_abs
        ):
            self._streak_color = None
            self._streak_count = 0
            return

        if color != self._streak_color:
            self._streak_color = color
            self._streak_count = 1
        else:
            self._streak_count += 1

        if self._streak_count < self._stable_frames:
            return

        if color == self._prev_color:
            return

        lidar_deg = 0.0
        if centroid_x is not None:
            lidar_rad = -centroid_x * self.CAMERA_HFOV
            lidar_deg = math.degrees(lidar_rad)

        if abs(lidar_deg) > self._max_bearing_deg:
            return

        rx, ry, ryaw = self.current_pose
        if self.latest_scan is not None:
            scan = self.latest_scan
            cap = min(self.WALL_MAX_PROJ, float(scan.range_max))
            dist = range_at(
                scan.ranges,
                scan.angle_min,
                scan.angle_increment,
                lidar_deg,
                window_deg=8.0,
                max_r=cap,
            )
            dist = min(dist, cap)
            dist_front = range_at(
                scan.ranges,
                scan.angle_min,
                scan.angle_increment,
                0.0,
                window_deg=self._front_window_deg,
                max_r=cap,
            )
            dist_front = min(dist_front, cap)
        else:
            dist = self.WALL_MAX_PROJ
            dist_front = self.WALL_MAX_PROJ

        if dist > self._max_dist_count:
            return

        # Not facing a nearby obstacle ahead — typical when glancing along a wall in motion
        if dist_front > self._max_front_range:
            return

        if abs(dist - dist_front) > self._max_range_disagree:
            return

        wall_angle = ryaw + math.radians(lidar_deg)
        wall_x = rx + dist * math.cos(wall_angle)
        wall_y = ry + dist * math.sin(wall_angle)

        already = False
        for wx, wy, wyaw, wc in self._seen_walls:
            if wc != color:
                continue
            dist_to_known = math.hypot(wall_x - wx, wall_y - wy)
            angle_diff = abs(
                math.atan2(math.sin(ryaw - wyaw), math.cos(ryaw - wyaw))
            )
            threshold = (
                self.WALL_DEDUPE_DIST_BACK
                if angle_diff > math.pi / 2
                else self.WALL_DEDUPE_DIST
            )
            if dist_to_known < threshold:
                already = True
                break

        if already:
            # Same physical wall re-seen: hold _prev_color so we do not re-run dedupe every frame.
            self._prev_color = color
            return

        self._seen_walls.append((wall_x, wall_y, ryaw, color))
        self.counts[color] += 1
        self.detection_positions[color].append((wall_x, wall_y))
        self._prev_color = color

        placar = self._format_placar()
        self.get_logger().info(
            f'[VISAO] Parede {color.upper()} — '
            f'wall≈({wall_x:.2f},{wall_y:.2f}) dist={dist:.2f}m | {placar}'
        )

    def _format_placar(self) -> str:
        return ', '.join(f'{_COLOR_LABELS[k]}:{self.counts[k]}' for k in _PT_ORDER)

    def _save_report(self):
        report_path = os.path.join(os.getcwd(), 'color_wall_report.txt')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        total = sum(self.counts.values())
        lines = [
            '=== Relatorio de Paredes Coloridas ===',
            f'Data/hora: {timestamp}',
            '',
            'Contagem por cor:',
        ]
        for k in _PT_ORDER:
            lines.append(f'  {k.capitalize():10s}: {self.counts[k]}')
            for i, (wx, wy) in enumerate(self.detection_positions[k], 1):
                lines.append(f'    #{i} parede estimada em ({wx:.1f}, {wy:.1f})')
        lines += ['', f'Total de paredes detectadas: {total}']
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        self.get_logger().info(f'[VISAO] Relatorio salvo em {report_path}')


def main(args=None):
    rclpy.init(args=args)
    node = ColorWallCounter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._save_report()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
