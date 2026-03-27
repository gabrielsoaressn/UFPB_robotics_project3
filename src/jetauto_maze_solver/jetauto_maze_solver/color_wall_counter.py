#!/usr/bin/env python3
"""
Sistema de Visão — Detecção e Contagem de Paredes Coloridas
Solução Equilibrada: Conta em movimento a direito, ignora em curva.
"""

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
from datetime import datetime

class ColorWallCounter(Node):

    # Reduzido para 10%. Permite contar a parede enquanto o robô se aproxima.
    MIN_AREA_FRACTION = 0.10
    COOLDOWN_SECS = 4.0
    CENTER_TOL = 0.35   # A cor deve estar nos 70% centrais da imagem
    MIN_DISTANCE = 2.0  # Raio de 2m para fundir a frente e o verso da mesma parede

    COLOR_RANGES = {
        'azul': [(np.array([100, 80, 50]), np.array([130, 255, 255]))],
        'verde': [(np.array([40, 80, 50]), np.array([85, 255, 255]))],
        'vermelho': [
            (np.array([0, 80, 50]),   np.array([10, 255, 255])),
            (np.array([170, 80, 50]), np.array([180, 255, 255]))
        ],
    }

    COLOR_LABELS = {'azul': 'A', 'verde': 'V', 'vermelho': 'R'}

    def __init__(self):
        super().__init__('color_wall_counter')
        self.bridge = CvBridge()

        self.create_subscription(Image, '/camera/image_raw', self._image_cb, 10)
        self.create_subscription(Odometry, '/odometry/filtered', self._odom_cb, 10)

        self.counts = {color: 0 for color in self.COLOR_RANGES}
        self.last_detection_time = {color: None for color in self.COLOR_RANGES}
        self.detection_positions = {color: [] for color in self.COLOR_RANGES}

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.angular_z = 0.0
        self.odom_ready = False
        
        self.last_frame_time = self.get_clock().now()

        self.get_logger().info('[VISÃO] Visão Equilibrada Iniciada (Permite contagem em aproximação)')

    def _odom_cb(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.robot_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        # Lê apenas a velocidade de rotação
        self.angular_z = msg.twist.twist.angular.z
        self.odom_ready = True

    def _image_cb(self, msg: Image):
        if not self.odom_ready:
            return

        now = self.get_clock().now()
        
        # Limita a 5 FPS para não sobrecarregar
        if (now - self.last_frame_time).nanoseconds / 1e9 < 0.2:
            return
        self.last_frame_time = now

        # ── REGRA RELAXADA: APENAS NÃO ESTAR A RODAR ──
        # Se a velocidade angular for maior que 0.15, o robô está a fazer uma curva.
        # Se for menor, ele está a ir a direito para a parede (ou parado).
        if abs(self.angular_z) > 0.15:
            return
        # ──────────────────────────────────────────────

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            return

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        min_area = self.MIN_AREA_FRACTION * h * w

        for color, ranges in self.COLOR_RANGES.items():
            mask = np.zeros((h, w), dtype=np.uint8)
            for lower, upper in ranges:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) < min_area:
                continue

            M = cv2.moments(largest)
            if M['m00'] == 0: continue
            cx = M['m10'] / M['m00']
            
            # A cor deve estar nos 70% centrais do ecrã
            if abs(cx - w / 2.0) > self.CENTER_TOL * w:
                continue

            # ── PROJECÇÃO FIXA DA PAREDE ──
            # O robô regista a parede 0.8 metros à sua frente.
            wall_x = self.robot_x + 0.8 * math.cos(self.robot_yaw)
            wall_y = self.robot_y + 0.8 * math.sin(self.robot_yaw)
            # ──────────────────────────────

            if not self._cooldown_ok(color, now):
                continue

            if not self._position_ok(color, wall_x, wall_y):
                continue

            self.counts[color] += 1
            self.last_detection_time[color] = now
            self.detection_positions[color].append((wall_x, wall_y))

            self.get_logger().info(
                f'[VISÃO] Parede {color.upper()} -> Robô:({self.robot_x:.1f}, {self.robot_y:.1f}) | '
                f'Parede:({wall_x:.1f}, {wall_y:.1f}) | Placar: {self._format_placar()}')

    def _cooldown_ok(self, color: str, now) -> bool:
        last = self.last_detection_time[color]
        if last is None: return True
        return (now - last).nanoseconds / 1e9 >= self.COOLDOWN_SECS

    def _position_ok(self, color: str, wall_x: float, wall_y: float) -> bool:
        for wx, wy in self.detection_positions[color]:
            dist = math.sqrt((wall_x - wx) ** 2 + (wall_y - wy) ** 2)
            if dist < self.MIN_DISTANCE:
                return False
        return True

    def _position_ok(self, color: str) -> bool:
        """Retorna True se a posicao atual esta longe de todas as deteccoes anteriores."""
        for px, py in self.detection_positions[color]:
            dist = math.sqrt((self.robot_x - px) ** 2 + (self.robot_y - py) ** 2)
            if dist < self.MIN_DISTANCE:
                return False
        return True

    def _format_placar(self) -> str:
        return ', '.join([f"{self.COLOR_LABELS[c]}:{self.counts[c]}" for c in self.COLOR_RANGES])

    def _save_report(self):
        report_path = os.path.join(os.getcwd(), 'color_wall_report.txt')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        total = sum(self.counts.values())

        lines = [
            '=== Relatório de Paredes Coloridas ===',
            f'Data/hora: {timestamp}',
            '',
            'Contagem por cor:',
        ]
        for color in self.COLOR_RANGES:
            lines.append(f'  {color.capitalize():10s}: {self.counts[color]}')
            for i, (wx, wy) in enumerate(self.detection_positions[color], 1):
                lines.append(f'    #{i} estimada em ({wx:.1f}, {wy:.1f})')
        lines += ['', f'Total de paredes detectadas: {total}']

        with open(report_path, 'w') as f: f.write('\n'.join(lines) + '\n')
        self.get_logger().info(f'[VISÃO] Relatório salvo em {report_path}')

def main(args=None):
    rclpy.init(args=args)
    node = ColorWallCounter()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node._save_report()
        node.destroy_node()
        try: rclpy.shutdown()
        except: pass

if __name__ == '__main__':
    main()