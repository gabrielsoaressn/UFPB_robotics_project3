#!/usr/bin/env python3
"""
Maze Navigator — Centro do Corredor (Robô Omnidirecional)

O robô usa o LiDAR para medir as distâncias às paredes esquerda e direita
e emprega linear.y para se manter centrado no corredor enquanto avança.
Quando a frente está bloqueada, a câmera determina a direção do giro:
  Vermelho → esquerda  |  Verde → direita  |  Sem cor → lado com mais espaço

Estados:
  FOLLOW_CORRIDOR → avança, centraliza lateralmente, corrige heading
  TURNING         → giro no lugar controlado por odometria

Tópicos:
  Assina: /jetauto/lidar/scan, /odometry/filtered, /jetauto/camera/image_raw
  Publica: /jetauto/cmd_vel
"""

import math
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge


# ═══════════════════════════════════════════════════════════════════════
# Utilitários de ângulo
# ═══════════════════════════════════════════════════════════════════════

def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


# ═══════════════════════════════════════════════════════════════════════
# Nó principal
# ═══════════════════════════════════════════════════════════════════════

class MazeNavigator(Node):

    # ── Setores do LiDAR (radianos) ────────────────────────────────
    FRONT_LO = math.radians(-20)
    FRONT_HI = math.radians(20)
    LEFT_LO  = math.radians(70)
    LEFT_HI  = math.radians(110)
    RIGHT_LO = math.radians(-110)
    RIGHT_HI = math.radians(-70)

    # Raios individuais para cálculo de heading (dois por parede)
    R_SIDE_ANGLE = math.radians(-90)   # perpendicular à parede direita
    R_DIAG_ANGLE = math.radians(-45)   # diagonal frente-direita
    L_SIDE_ANGLE = math.radians(90)    # perpendicular à parede esquerda
    L_DIAG_ANGLE = math.radians(45)    # diagonal frente-esquerda

    # ── Limiares de distância (metros) ─────────────────────────────
    FRONT_BLOCKED = 0.5     # frente bloqueada → escolhe direção e gira
    WALL_DETECT   = 1.2     # distância máxima para considerar parede presente
    TARGET_SIDE   = 0.4     # distância desejada à parede (um só lado)

    # ── Controle lateral — linear.y ─────────────────────────────────
    # Positivo = mover para a esquerda (convenção ROS, y aponta para esquerda)
    KP_LAT    = 0.3     # ganho proporcional
    LAT_CLAMP = 0.15    # velocidade lateral máxima (m/s)

    # ── Correção de heading — angular.z ─────────────────────────────
    ALIGN_KP    = 1.5
    ALIGN_CLAMP = 0.1   # rad/s máximo de correção

    # ── Velocidades ─────────────────────────────────────────────────
    FORWARD_SPEED = 0.15    # m/s para frente
    TURN_SPEED    = 0.4     # rad/s durante giro no estado TURNING
    YAW_TOLERANCE = 0.05    # rad (~2.9°)

    # ── LiDAR ───────────────────────────────────────────────────────
    RANGE_CAP       = 3.0
    LIDAR_MIN_VALID = 0.15

    # ── Câmera — faixas HSV ─────────────────────────────────────────
    COLOR_MIN_AREA_FRAC = 0.05  # área mínima relativa da cor na imagem
    RED_LOWER1  = np.array([0,   80,  50])
    RED_UPPER1  = np.array([10,  255, 255])
    RED_LOWER2  = np.array([170, 80,  50])
    RED_UPPER2  = np.array([180, 255, 255])
    GREEN_LOWER = np.array([40,  80,  50])
    GREEN_UPPER = np.array([85,  255, 255])

    def __init__(self):
        super().__init__('maze_navigator')

        self.create_subscription(LaserScan, '/jetauto/lidar/scan', self._scan_cb, 10)
        self.create_subscription(Odometry, '/odometry/filtered', self._odom_cb, 10)
        self.create_subscription(Image, '/jetauto/camera/image_raw', self._image_cb, 10)

        self.cmd_pub = self.create_publisher(Twist, '/jetauto/cmd_vel', 10)
        self.bridge = CvBridge()
        self.timer = self.create_timer(0.1, self._control_loop)

        # ── LiDAR ──
        self.front_dist = self.RANGE_CAP
        self.left_dist  = self.RANGE_CAP
        self.right_dist = self.RANGE_CAP
        self.r_side = self.RANGE_CAP  # raio a -90°
        self.r_diag = self.RANGE_CAP  # raio a -45°
        self.l_side = self.RANGE_CAP  # raio a +90°
        self.l_diag = self.RANGE_CAP  # raio a +45°
        self.scan_ready = False

        # ── Odometria ──
        self.current_yaw = 0.0
        self.odom_ready = False

        # ── Câmera ──
        self.detected_color = None   # 'vermelho', 'verde' ou None

        # ── Máquina de estados ──
        self.state = 'FOLLOW_CORRIDOR'
        self.target_yaw     = 0.0
        self.turn_direction = 0.0   # +1 = esquerda, -1 = direita
        self.loop_count = 0

        self.get_logger().info(
            '[NAV] Maze Navigator iniciado — Centro do Corredor + Paredes Coloridas')

    # ══════════════════════════════════════════════════════════════════
    # Callbacks dos sensores
    # ══════════════════════════════════════════════════════════════════

    def _odom_cb(self, msg: Odometry):
        self.current_yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self.odom_ready = True

    def _scan_cb(self, msg: LaserScan):
        ranges = msg.ranges
        n = len(ranges)
        if n == 0:
            return

        a_min = msg.angle_min
        a_inc = msg.angle_increment

        def idx(a):
            return max(0, min(n - 1, round((a - a_min) / a_inc)))

        def sector_min(lo, hi):
            best = self.RANGE_CAP
            for i in range(min(idx(lo), idx(hi)), max(idx(lo), idx(hi)) + 1):
                r = ranges[i]
                if math.isfinite(r) and r >= self.LIDAR_MIN_VALID:
                    best = min(best, min(r, self.RANGE_CAP))
            return best

        def ray_at(a):
            r = ranges[idx(a)]
            if math.isfinite(r) and r >= self.LIDAR_MIN_VALID:
                return min(r, self.RANGE_CAP)
            return self.RANGE_CAP

        self.front_dist = sector_min(self.FRONT_LO, self.FRONT_HI)
        self.left_dist  = sector_min(self.LEFT_LO,  self.LEFT_HI)
        self.right_dist = sector_min(self.RIGHT_LO, self.RIGHT_HI)
        self.r_side = ray_at(self.R_SIDE_ANGLE)
        self.r_diag = ray_at(self.R_DIAG_ANGLE)
        self.l_side = ray_at(self.L_SIDE_ANGLE)
        self.l_diag = ray_at(self.L_DIAG_ANGLE)
        self.scan_ready = True

    def _image_cb(self, msg: Image):
        """Detecta cor dominante (vermelho/verde) no frame atual da câmera."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'[NAV] cv_bridge: {e}')
            return

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        min_area = self.COLOR_MIN_AREA_FRAC * h * w

        mask_r = cv2.bitwise_or(
            cv2.inRange(hsv, self.RED_LOWER1, self.RED_UPPER1),
            cv2.inRange(hsv, self.RED_LOWER2, self.RED_UPPER2))
        red_area   = cv2.countNonZero(mask_r)
        green_area = cv2.countNonZero(
            cv2.inRange(hsv, self.GREEN_LOWER, self.GREEN_UPPER))

        if red_area >= min_area and red_area >= green_area:
            self.detected_color = 'vermelho'
        elif green_area >= min_area and green_area > red_area:
            self.detected_color = 'verde'
        else:
            self.detected_color = None

    # ══════════════════════════════════════════════════════════════════
    # Cálculos de controle
    # ══════════════════════════════════════════════════════════════════

    def _choose_turn(self):
        """
        Decide a direção do giro quando a frente está bloqueada.

        Prioridade:
          1. Câmera detectou VERMELHO → esquerda
          2. Câmera detectou VERDE    → direita
          3. Sem cor: vira para o lado com maior espaço lateral
        """
        color = self.detected_color
        if color == 'vermelho':
            target = normalize_angle(self.current_yaw + math.pi / 2.0)
            return target, 1.0, 'esquerda (parede VERMELHA)'
        elif color == 'verde':
            target = normalize_angle(self.current_yaw - math.pi / 2.0)
            return target, -1.0, 'direita (parede VERDE)'
        else:
            # Sem indicação de cor: prefere o lado com mais espaço
            if self.left_dist >= self.right_dist:
                target = normalize_angle(self.current_yaw + math.pi / 2.0)
                return target, 1.0, 'esquerda (mais espaço)'
            else:
                target = normalize_angle(self.current_yaw - math.pi / 2.0)
                return target, -1.0, 'direita (mais espaço)'

    def _lateral_correction(self) -> float:
        """
        Calcula linear.y para centralizar o robô no corredor.

        Convenção ROS: linear.y > 0 → move para esquerda.

        Casos:
          Ambas as paredes: centraliza entre elas.
          Só parede direita: mantém TARGET_SIDE de distância.
          Só parede esquerda: mantém TARGET_SIDE de distância.
          Nenhuma parede: sem correção lateral.
        """
        L = self.left_dist
        R = self.right_dist
        left_wall  = L < self.WALL_DETECT
        right_wall = R < self.WALL_DETECT

        if left_wall and right_wall:
            # (L - R) / 2 > 0 → mais próximo da direita → mover esquerda ✓
            error = (L - R) / 2.0
        elif right_wall:
            # TARGET_SIDE - R > 0 quando muito perto da direita → mover esquerda ✓
            error = self.TARGET_SIDE - R
        elif left_wall:
            # L - TARGET_SIDE < 0 quando muito perto da esquerda → mover direita ✓
            error = L - self.TARGET_SIDE
        else:
            error = 0.0

        return max(-self.LAT_CLAMP, min(self.KP_LAT * error, self.LAT_CLAMP))

    def _heading_correction(self) -> float:
        """
        Calcula angular.z para manter o robô alinhado com o corredor.

        Usa geometria de dois raios por parede para derivar o ângulo real
        do robô em relação à superfície. Combina as correções de ambas
        as paredes presentes (média).

        Geometria (frame do robô: x=frente, y=esquerda):
          Parede direita — raios a -90° (a) e -45° (b):
            Vetor da parede: dx = b·cos45, dy = a − b·sin45
            Erro de heading = atan2(dy, dx)
          Parede esquerda — raios a +90° (c) e +45° (d):
            Vetor da parede: dx = d·cos45, dy = d·sin45 − c
            Erro de heading = atan2(dy, dx)   ← mesmo sinal, geometria simétrica
        """
        corrections = []

        a, b = self.r_side, self.r_diag
        if a < self.WALL_DETECT and b < self.RANGE_CAP - 0.1:
            dy = a - b * 0.7071
            dx = b * 0.7071
            corrections.append(math.atan2(dy, dx))

        c, d = self.l_side, self.l_diag
        if c < self.WALL_DETECT and d < self.RANGE_CAP - 0.1:
            dy = d * 0.7071 - c
            dx = d * 0.7071
            corrections.append(math.atan2(dy, dx))

        if not corrections:
            return 0.0

        heading_error = sum(corrections) / len(corrections)
        az = self.ALIGN_KP * heading_error
        return max(-self.ALIGN_CLAMP, min(az, self.ALIGN_CLAMP))

    # ══════════════════════════════════════════════════════════════════
    # Loop de controle principal (10 Hz)
    # ══════════════════════════════════════════════════════════════════

    def _control_loop(self):
        if not self.scan_ready or not self.odom_ready:
            return

        self.loop_count += 1
        twist = Twist()

        if self.state == 'TURNING':
            self._state_turning(twist)
        else:
            self._state_follow_corridor(twist)

        self.cmd_pub.publish(twist)

    # ──────────────────────────────────────────────────────────────────
    # ESTADO: FOLLOW_CORRIDOR
    # ──────────────────────────────────────────────────────────────────

    def _state_follow_corridor(self, twist: Twist):
        """
        Avança reto enquanto:
          1. linear.y centraliza lateralmente no corredor
          2. angular.z mantém o heading paralelo às paredes

        Transição: frente bloqueada → consulta câmera → TURNING
        """
        if self.front_dist <= self.FRONT_BLOCKED:
            self.target_yaw, self.turn_direction, desc = self._choose_turn()
            self.state = 'TURNING'
            self.get_logger().info(
                f'[NAV] FOLLOW_CORRIDOR → TURNING {desc}  '
                f'F={self.front_dist:.2f}m  '
                f'target={math.degrees(self.target_yaw):.1f}°')
            twist.linear.x  = 0.0
            twist.linear.y  = 0.0
            twist.angular.z = 0.0
            return

        twist.linear.x  = self.FORWARD_SPEED
        twist.linear.y  = self._lateral_correction()
        twist.angular.z = self._heading_correction()

        if self.loop_count % 20 == 0:
            self.get_logger().info(
                f'[NAV] FOLLOW_CORRIDOR  '
                f'F={self.front_dist:.2f}m  '
                f'L={self.left_dist:.2f}m  R={self.right_dist:.2f}m  '
                f'vy={twist.linear.y:+.3f}  az={twist.angular.z:+.3f}  '
                f'cor={self.detected_color}')

    # ──────────────────────────────────────────────────────────────────
    # ESTADO: TURNING
    # ──────────────────────────────────────────────────────────────────

    def _state_turning(self, twist: Twist):
        """Gira no lugar até atingir target_yaw com tolerância YAW_TOLERANCE."""
        error = normalize_angle(self.target_yaw - self.current_yaw)

        if abs(error) < self.YAW_TOLERANCE:
            self.state = 'FOLLOW_CORRIDOR'
            self.get_logger().info(
                f'[NAV] TURNING concluído → FOLLOW_CORRIDOR  '
                f'yaw={math.degrees(self.current_yaw):.1f}°  '
                f'erro_final={math.degrees(error):.1f}°')
            twist.linear.x  = 0.0
            twist.linear.y  = 0.0
            twist.angular.z = 0.0
        else:
            twist.linear.x  = 0.0
            twist.linear.y  = 0.0
            twist.angular.z = self.TURN_SPEED * self.turn_direction
            if self.loop_count % 5 == 0:
                self.get_logger().info(
                    f'[NAV] TURNING  erro={math.degrees(error):.1f}°  '
                    f'az={twist.angular.z:+.2f}  '
                    f'yaw={math.degrees(self.current_yaw):.1f}°  '
                    f'target={math.degrees(self.target_yaw):.1f}°')

    # ──────────────────────────────────────────────────────────────────
    # Cleanup
    # ──────────────────────────────────────────────────────────────────

    def destroy_node(self):
        self.cmd_pub.publish(Twist())
        super().destroy_node()


# ═══════════════════════════════════════════════════════════════════════
# Ponto de entrada
# ═══════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = MazeNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
