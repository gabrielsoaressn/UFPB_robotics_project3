#!/usr/bin/env python3
"""
Vision — colored wall detection and counting (passive).

Replicates the wall deduplication strategy from robotics_subject color_detector_node
(robotica-main): wall position is estimated by projecting LiDAR range at the bearing
that matches the color blob centroid in the camera frame, then deduplicating with
a larger distance threshold when the robot faces the opposite direction (~same
physical wall, other face).

This node never publishes cmd_vel.

Subscriptions (defaults match UFPB Gazebo URDF):
  image_topic: /jetauto/camera/image_raw
  odom_topic:  /odometry/filtered
  scan_topic:  /jetauto/lidar/scan
"""

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
    """Smallest valid LiDAR reading in a window centered at deg (robot frame)."""
    rad = math.radians(deg)
    hw = math.radians(window_deg)
    vals = [
        r
        for i, r in enumerate(ranges)
        if abs(angle_min + i * angle_inc - rad) <= hw and math.isfinite(r) and r > 0.05
    ]
    return min(vals) if vals else max_r


class ColorWallCounter(Node):
    # Same spirit as robotica-main color_detector_node
    WALL_DEDUPE_DIST = 2.0
    WALL_DEDUPE_DIST_BACK = 3.0
    WALL_MAX_PROJ = 1.925
    MIN_COLOR_RATIO = 0.06
    # depth_camera.urdf.xacro (UFPB): horizontal_fov 1.2 rad
    CAMERA_HFOV = 1.2

    MAX_SPEED = 0.05
    MAX_ANGULAR_SPEED = 0.1
    # Optional: require blob centroid near image center (reduces side-wall triggers)
    CENTER_TOL = 0.25

    COLOR_RANGES = {
        'vermelho': [
            (np.array([0, 120, 80]), np.array([10, 255, 255])),
            (np.array([168, 120, 80]), np.array([180, 255, 255])),
        ],
        'verde': [
            (np.array([40, 90, 80]), np.array([85, 255, 255])),
        ],
        'azul': [
            (np.array([100, 90, 80]), np.array([135, 255, 255])),
        ],
    }

    COLOR_LABELS = {
        'azul': 'A',
        'verde': 'V',
        'vermelho': 'R',
    }

    def __init__(self):
        super().__init__('color_wall_counter')

        self.declare_parameter('image_topic', '/jetauto/camera/image_raw')
        self.declare_parameter('odom_topic', '/odometry/filtered')
        self.declare_parameter('scan_topic', '/jetauto/lidar/scan')
        self.declare_parameter('wall_dedupe_dist', self.WALL_DEDUPE_DIST)
        self.declare_parameter('wall_dedupe_dist_back', self.WALL_DEDUPE_DIST_BACK)
        self.declare_parameter('wall_max_proj', self.WALL_MAX_PROJ)
        self.declare_parameter('min_color_ratio', self.MIN_COLOR_RATIO)
        self.declare_parameter('camera_hfov', self.CAMERA_HFOV)
        self.declare_parameter('lidar_window_deg', 8.0)

        img_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        scan_topic = self.get_parameter('scan_topic').get_parameter_value().string_value
        self.WALL_DEDUPE_DIST = self.get_parameter('wall_dedupe_dist').value
        self.WALL_DEDUPE_DIST_BACK = self.get_parameter('wall_dedupe_dist_back').value
        self.WALL_MAX_PROJ = self.get_parameter('wall_max_proj').value
        self.MIN_COLOR_RATIO = self.get_parameter('min_color_ratio').value
        self.CAMERA_HFOV = self.get_parameter('camera_hfov').value
        self._lidar_window_deg = self.get_parameter('lidar_window_deg').value

        self.bridge = CvBridge()
        self.create_subscription(Image, img_topic, self._image_cb, 10)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)
        self.create_subscription(LaserScan, scan_topic, self._scan_cb, 10)

        self.counts = {c: 0 for c in self.COLOR_RANGES}
        self.detection_positions = {c: [] for c in self.COLOR_RANGES}

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.robot_speed = 0.0
        self.robot_angular_speed = 0.0
        self.odom_ready = False
        self.latest_scan = None

        # Edge-trigger: first frame a color appears (avoids counting every image)
        self._prev_color = None
        # (wall_x, wall_y, robot_yaw_at_detection, color_key)
        self._seen_walls = []

        self.get_logger().info(
            '[VISAO] Color wall counter — LiDAR+camera wall pose + angle-aware dedupe'
        )
        self.get_logger().info(
            f'[VISAO] image={img_topic} scan={scan_topic} | '
            f'dedupe={self.WALL_DEDUPE_DIST}m / back={self.WALL_DEDUPE_DIST_BACK}m'
        )

    def _odom_cb(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.robot_yaw = yaw_from_quat(msg.pose.pose.orientation)
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.robot_speed = math.hypot(vx, vy)
        self.robot_angular_speed = abs(msg.twist.twist.angular.z)
        self.odom_ready = True

    def _scan_cb(self, msg: LaserScan):
        self.latest_scan = msg

    def _detect_dominant_color(self, bgr: np.ndarray):
        """
        Returns (color_key | None, masks dict, centroid_x normalized [-0.5, 0.5]).
        """
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        total = bgr.shape[0] * bgr.shape[1]
        h, w = bgr.shape[:2]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        masks = {}
        ratios = {}
        for name, ranges in self.COLOR_RANGES.items():
            m = np.zeros((h, w), dtype=np.uint8)
            for lo, hi in ranges:
                m = cv2.bitwise_or(m, cv2.inRange(hsv, lo, hi))
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
            masks[name] = m
            ratios[name] = cv2.countNonZero(m) / total

        best = max(ratios, key=ratios.get)
        if ratios[best] < self.MIN_COLOR_RATIO:
            return None, masks, None

        M = cv2.moments(masks[best])
        if M['m00'] <= 0:
            return None, masks, None
        cx_px = M['m10'] / M['m00']
        if abs(cx_px - w / 2.0) > self.CENTER_TOL * w:
            return None, masks, None

        centroid_x = (cx_px - w / 2.0) / w
        return best, masks, centroid_x

    def _image_cb(self, msg: Image):
        if not self.odom_ready:
            return
        if self.robot_speed > self.MAX_SPEED or self.robot_angular_speed > self.MAX_ANGULAR_SPEED:
            return

        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'[VISAO] cv_bridge: {e}')
            return

        color, _masks, centroid_x = self._detect_dominant_color(bgr)

        if color is None:
            self._prev_color = None
            return

        if color == self._prev_color:
            return
        self._prev_color = color

        lidar_deg = 0.0
        if centroid_x is not None:
            lidar_rad = -centroid_x * self.CAMERA_HFOV
            lidar_deg = math.degrees(lidar_rad)

        rx, ry, ryaw = self.robot_x, self.robot_y, self.robot_yaw
        if self.latest_scan is not None:
            scan = self.latest_scan
            dist = range_at(
                scan.ranges,
                scan.angle_min,
                scan.angle_increment,
                lidar_deg,
                window_deg=self._lidar_window_deg,
                max_r=self.WALL_MAX_PROJ,
            )
            dist = min(dist, self.WALL_MAX_PROJ)
        else:
            dist = self.WALL_MAX_PROJ
            self.get_logger().warn('[VISAO] No LiDAR yet — using wall_max_proj fallback')

        wall_angle = ryaw + math.radians(lidar_deg)
        wall_x = rx + dist * math.cos(wall_angle)
        wall_y = ry + dist * math.sin(wall_angle)

        if self._is_duplicate_wall(color, wall_x, wall_y, ryaw):
            self.get_logger().debug(
                f'[VISAO] {color}: duplicate wall at ({wall_x:.2f},{wall_y:.2f}) — skip'
            )
            return

        self._seen_walls.append((wall_x, wall_y, ryaw, color))
        self.counts[color] += 1
        self.detection_positions[color].append((wall_x, wall_y))

        placar = self._format_placar()
        self.get_logger().info(
            f'[VISAO] Parede {color.upper()} — '
            f'robo=({rx:.2f},{ry:.2f}) parede≈({wall_x:.2f},{wall_y:.2f}) '
            f'dist_lidar={dist:.2f}m | Placar: {placar}'
        )

    def _is_duplicate_wall(self, color: str, wall_x: float, wall_y: float, ryaw: float) -> bool:
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
                return True
        return False

    def _format_placar(self) -> str:
        parts = []
        for c in self.COLOR_RANGES:
            parts.append(f'{self.COLOR_LABELS[c]}:{self.counts[c]}')
        return ', '.join(parts)

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
        for c in self.COLOR_RANGES:
            lines.append(f'  {c.capitalize():10s}: {self.counts[c]}')
            for i, (wx, wy) in enumerate(self.detection_positions[c], 1):
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
