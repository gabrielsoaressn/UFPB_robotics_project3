#!/usr/bin/env python3
"""
Sistema de Visao — Deteccao e Contagem de Paredes Coloridas (Isolado e Passivo)

Detecta paredes Azuis, Verdes e Vermelhas usando camera RGB.
So detecta uma parede quando:
  1. A regiao colorida ocupa >= MIN_AREA_FRACTION da imagem (robo esta proximo)
  2. O centroide da regiao esta no centro horizontal +/- CENTER_TOL (parede na frente do robo)
Deduplicacao por posicao estimada da parede no mundo (yaw + distancia proporcional a area):
  frente e verso produzem estimativas convergentes, bloqueadas pelo WALL_CLUSTER_DIST

*** Este no NUNCA publica comandos de velocidade. ***

Topicos assinados: /jetauto/camera/image_raw, /odometry/filtered
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

    # Fracao minima da imagem que a parede deve ocupar (robo proximo)
    MIN_AREA_FRACTION = 0.15
    COOLDOWN_SECS = 5.0
    # Tolerancia horizontal: centroide da cor deve estar no centro +/- (CENTER_TOL * largura)
    # 0.25 = aceita somente no quarto central da imagem (parede na frente do robo)
    CENTER_TOL = 0.25
    # Velocidade linear maxima (m/s) para considerar que o robo esta parado
    MAX_SPEED = 0.05
    # Velocidade angular maxima (rad/s) — exclui curvas/rotacoes
    MAX_ANGULAR_SPEED = 0.1
    # Distancia estimada (m) ate a parede quando area_fraction == MIN_AREA_FRACTION
    MIN_DET_DIST = 1.5
    # Distancia maxima (m) entre estimativas de posicao para ser considerada a mesma parede
    WALL_CLUSTER_DIST = 2.5

    COLOR_RANGES = {
        'azul': [
            (np.array([100, 80, 50]), np.array([130, 255, 255])),
        ],
        'verde': [
            (np.array([40, 80, 50]), np.array([85, 255, 255])),
        ],
        'vermelho': [
            (np.array([0, 80, 50]),   np.array([10, 255, 255])),
            (np.array([170, 80, 50]), np.array([180, 255, 255])),
        ],
    }

    COLOR_LABELS = {
        'azul': 'A',
        'verde': 'V',
        'vermelho': 'R',
    }

    def __init__(self):
        super().__init__('color_wall_counter')

        self.bridge = CvBridge()

        self.create_subscription(
            Image, '/camera/image_raw', self._image_cb, 10)
        self.create_subscription(
            Odometry, '/odometry/filtered', self._odom_cb, 10)

        # Contagens por cor
        self.counts = {color: 0 for color in self.COLOR_RANGES}

        # Cooldown temporal por cor
        self.last_detection_time = {color: None for color in self.COLOR_RANGES}

        # Posicoes estimadas (wall_x, wall_y) de cada parede detectada
        self.detection_positions = {color: [] for color in self.COLOR_RANGES}

        # Posicao e velocidade atual do robo
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.robot_speed = 0.0
        self.robot_angular_speed = 0.0
        self.odom_ready = False

        self.get_logger().info(
            '[VISAO] Sistema de Visao iniciado — detectando paredes coloridas')
        self.get_logger().info(
            f'[VISAO] Cooldown: {self.COOLDOWN_SECS}s  '
            f'Vel max: {self.MAX_SPEED}m/s  '
            f'Area minima: {self.MIN_AREA_FRACTION * 100:.0f}%  '
            f'Centro +/-{self.CENTER_TOL * 100:.0f}%  '
            f'Cluster parede: {self.WALL_CLUSTER_DIST}m')

    def _odom_cb(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.robot_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.robot_speed = math.sqrt(vx * vx + vy * vy)
        self.robot_angular_speed = abs(msg.twist.twist.angular.z)
        self.odom_ready = True

    def _image_cb(self, msg: Image):
        if not self.odom_ready:
            return

        # So processa imagem quando o robo esta completamente parado
        if self.robot_speed > self.MAX_SPEED or self.robot_angular_speed > self.MAX_ANGULAR_SPEED:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'[VISAO] Erro cv_bridge: {e}')
            return

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        min_area = self.MIN_AREA_FRACTION * h * w

        now = self.get_clock().now()

        for color, ranges in self.COLOR_RANGES.items():
            mask = np.zeros((h, w), dtype=np.uint8)
            for lower, upper in ranges:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area < min_area:
                continue

            # Verificar se a parede esta na frente do robo (centroide horizontalmente centrado)
            M = cv2.moments(largest)
            if M['m00'] == 0:
                continue
            cx = M['m10'] / M['m00']
            if abs(cx - w / 2.0) > self.CENTER_TOL * w:
                continue

            # Estimar posicao da parede no mundo:
            #   quanto maior a area aparente, mais perto a parede esta.
            #   dist_aprox = MIN_DET_DIST * sqrt(MIN_AREA_FRACTION / area_fraction)
            area_fraction = area / (h * w)
            approx_dist = self.MIN_DET_DIST * math.sqrt(
                self.MIN_AREA_FRACTION / area_fraction)
            wall_x = self.robot_x + approx_dist * math.cos(self.robot_yaw)
            wall_y = self.robot_y + approx_dist * math.sin(self.robot_yaw)

            # Cooldown temporal
            if not self._cooldown_ok(color, now):
                continue

            # Deduplicacao pela posicao estimada da parede
            if not self._position_ok(color, wall_x, wall_y):
                continue

            # Nova parede detectada!
            self.counts[color] += 1
            self.last_detection_time[color] = now
            self.detection_positions[color].append((wall_x, wall_y))

            placar = self._format_placar()
            self.get_logger().info(
                f'[VISAO] Parede {color.upper()} — '
                f'robo=({self.robot_x:.1f},{self.robot_y:.1f}) '
                f'parede≈({wall_x:.1f},{wall_y:.1f}) dist≈{approx_dist:.1f}m | '
                f'Placar: {placar}')

    def _cooldown_ok(self, color: str, now) -> bool:
        last = self.last_detection_time[color]
        if last is None:
            return True
        elapsed = (now - last).nanoseconds / 1e9
        return elapsed >= self.COOLDOWN_SECS

    def _position_ok(self, color: str, wall_x: float, wall_y: float) -> bool:
        """Retorna True se a parede estimada e diferente de todas as anteriores.

        Compara posicoes estimadas das paredes — nao do robo.
        Frente e verso da mesma parede produzem estimativas proximas,
        portanto sao bloqueadas pelo WALL_CLUSTER_DIST independentemente
        da direcao de aproximacao.
        """
        for wx, wy in self.detection_positions[color]:
            dist = math.sqrt((wall_x - wx) ** 2 + (wall_y - wy) ** 2)
            if dist < self.WALL_CLUSTER_DIST:
                return False
        return True

    def _format_placar(self) -> str:
        parts = []
        for color in self.COLOR_RANGES:
            label = self.COLOR_LABELS[color]
            count = self.counts[color]
            parts.append(f'{label}:{count}')
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
        for color in self.COLOR_RANGES:
            lines.append(f'  {color.capitalize():10s}: {self.counts[color]}')
            for i, (wx, wy) in enumerate(self.detection_positions[color], 1):
                lines.append(f'    #{i} parede estimada em ({wx:.1f}, {wy:.1f})')
        lines += [
            '',
            f'Total de paredes detectadas: {total}',
        ]

        with open(report_path, 'w') as f:
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