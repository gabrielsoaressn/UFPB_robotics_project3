#!/usr/bin/env python3
"""
Sistema de Visão — Detecção e Contagem de Paredes Coloridas

Detecta paredes Azuis, Verdes e Vermelhas (e Amarelas) usando câmera RGB.

Deduplicação por posição:
  Quando uma parede é detectada, estima-se a posição da parede no mundo:
    wall_x = robot_x + front_dist * cos(yaw)
    wall_y = robot_y + front_dist * sin(yaw)
  Se já existe uma parede da mesma cor a menos de DEDUP_DIST metros,
  assume-se que é a mesma parede (ou o seu verso) e não é recontada.

Este nó NUNCA publica comandos de velocidade.

Tópicos assinados:
  /jetauto/camera/image_raw   — frame RGB da câmera
  /odometry/filtered          — posição e orientação do robô
  /jetauto/lidar/scan         — distância frontal até a parede
"""

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import numpy as np


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class ColorWallCounter(Node):

    # ── Área mínima do contorno (fração da imagem) ─────────────────
    # A cor deve ocupar pelo menos 3% da imagem para ser considerada
    # uma parede próxima/frontal (filtra reflexos e objetos distantes).
    MIN_AREA_FRACTION = 0.03

    # ── Deduplicação por posição ───────────────────────────────────
    # Duas detecções da mesma cor a menos de DEDUP_DIST metros
    # são tratadas como a mesma parede (ou o seu verso).
    DEDUP_DIST = 0.8    # metros

    # ── Cooldown mínimo (segundos) ─────────────────────────────────
    # Barreira temporal extra para evitar dupla contagem rápida
    # no mesmo local (ex.: câmera tremendo).
    COOLDOWN_SECS = 2.0

    # ── LiDAR: setor frontal para estimar distância à parede ───────
    FRONT_LO = math.radians(-20)
    FRONT_HI = math.radians(20)
    RANGE_CAP = 3.0
    LIDAR_MIN_VALID = 0.15

    # ── Faixas HSV para cada cor ───────────────────────────────────
    COLOR_RANGES = {
        'azul': [
            (np.array([100, 80, 50]), np.array([130, 255, 255])),
        ],
        'verde': [
            (np.array([40, 80, 50]), np.array([85, 255, 255])),
        ],
        'vermelho': [
            (np.array([0,   80, 50]), np.array([10,  255, 255])),
            (np.array([170, 80, 50]), np.array([180, 255, 255])),
        ],
        'amarelo': [
            (np.array([20, 80, 50]), np.array([40, 255, 255])),
        ],
    }

    COLOR_LABELS = {'azul': 'A', 'verde': 'V', 'vermelho': 'R', 'amarelo': 'Y'}

    def __init__(self):
        super().__init__('color_wall_counter')

        self.bridge = CvBridge()

        self.create_subscription(
            Image, '/jetauto/camera/image_raw', self._image_cb, 10)
        self.create_subscription(
            Odometry, '/odometry/filtered', self._odom_cb, 10)
        self.create_subscription(
            LaserScan, '/jetauto/lidar/scan', self._scan_cb, 10)

        # Posição e orientação do robô
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.current_yaw = 0.0

        # Distância frontal (estimativa da distância à parede à frente)
        self.front_dist = self.RANGE_CAP

        # Contagens por cor
        self.counts = {color: 0 for color in self.COLOR_RANGES}

        # Lista de posições de paredes já contadas por cor: [(x, y), ...]
        self.wall_positions = {color: [] for color in self.COLOR_RANGES}

        # Timestamp da última detecção por cor (cooldown mínimo)
        self.last_detection_time = {color: None for color in self.COLOR_RANGES}

        self.get_logger().info(
            '[VISÃO] Sistema de Visão iniciado — deduplicação por posição  '
            f'DEDUP_DIST={self.DEDUP_DIST}m  COOLDOWN={self.COOLDOWN_SECS}s')

    # ══════════════════════════════════════════════════════════════════
    # Callbacks dos sensores
    # ══════════════════════════════════════════════════════════════════

    def _odom_cb(self, msg: Odometry):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        self.current_yaw = yaw_from_quaternion(msg.pose.pose.orientation)

    def _scan_cb(self, msg: LaserScan):
        """Extrai distância mínima no setor frontal."""
        ranges = msg.ranges
        n = len(ranges)
        if n == 0:
            return
        a_min = msg.angle_min
        a_inc = msg.angle_increment

        def idx(a):
            return max(0, min(n - 1, round((a - a_min) / a_inc)))

        best = self.RANGE_CAP
        for i in range(min(idx(self.FRONT_LO), idx(self.FRONT_HI)),
                       max(idx(self.FRONT_LO), idx(self.FRONT_HI)) + 1):
            r = ranges[i]
            if math.isfinite(r) and r >= self.LIDAR_MIN_VALID:
                best = min(best, min(r, self.RANGE_CAP))
        self.front_dist = best

    def _image_cb(self, msg: Image):
        """Processa cada frame: detecta cores e contabiliza paredes novas."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'[VISÃO] Erro cv_bridge: {e}')
            return

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        min_area = self.MIN_AREA_FRACTION * h * w
        now = self.get_clock().now()

        for color, ranges in self.COLOR_RANGES.items():
            # Máscara combinada para a cor
            mask = np.zeros((h, w), dtype=np.uint8)
            for lower, upper in ranges:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            if cv2.contourArea(max(contours, key=cv2.contourArea)) < min_area:
                continue

            # ── Cooldown mínimo ──
            if not self._cooldown_ok(color, now):
                continue

            # ── Estima posição da parede no mundo ──
            wall_x = self.odom_x + self.front_dist * math.cos(self.current_yaw)
            wall_y = self.odom_y + self.front_dist * math.sin(self.current_yaw)

            # ── Deduplicação por posição ──
            if self._already_seen(color, wall_x, wall_y):
                continue

            # ── Nova parede! Registra e contabiliza ──
            self.wall_positions[color].append((wall_x, wall_y))
            self.counts[color] += 1
            self.last_detection_time[color] = now

            placar = self._format_placar()
            self.get_logger().info(
                f'[VISÃO] Parede {color.upper()} detectada!  '
                f'pos_parede=({wall_x:.2f}, {wall_y:.2f})  '
                f'Placar: {placar}')

    # ══════════════════════════════════════════════════════════════════
    # Auxiliares
    # ══════════════════════════════════════════════════════════════════

    def _cooldown_ok(self, color: str, now) -> bool:
        last = self.last_detection_time[color]
        if last is None:
            return True
        return (now - last).nanoseconds / 1e9 >= self.COOLDOWN_SECS

    def _already_seen(self, color: str, wx: float, wy: float) -> bool:
        """
        Retorna True se já existe uma parede da mesma cor registrada
        a menos de DEDUP_DIST metros de (wx, wy).
        """
        for (px, py) in self.wall_positions[color]:
            if math.hypot(wx - px, wy - py) < self.DEDUP_DIST:
                return True
        return False

    def _format_placar(self) -> str:
        parts = [f'{self.COLOR_LABELS[c]}:{self.counts[c]}'
                 for c in self.COLOR_RANGES]
        return ', '.join(parts)


# ═══════════════════════════════════════════════════════════════════════
# Ponto de entrada
# ═══════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = ColorWallCounter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
