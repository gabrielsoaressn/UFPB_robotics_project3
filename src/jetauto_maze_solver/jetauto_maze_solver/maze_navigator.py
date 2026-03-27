#!/usr/bin/env python3
"""
Maze Navigator — Regra da Mao Direita (Robo Omnidirecional)

Resolve o labirinto usando a regra da mao direita (right-hand rule):
  1. Se o corredor abre a direita -> vira a direita
  2. Se a frente esta livre -> segue em frente
  3. Se a frente esta bloqueada -> vira a esquerda (ou 180 se beco sem saida)

Cores da camera podem sobrescrever a decisao quando a frente esta bloqueada:
  Vermelho / Azul -> esquerda  |  Verde -> direita

Estados:
  FOLLOW_CORRIDOR -> avanca, centraliza lateralmente, corrige heading
  TURNING         -> giro no lugar controlado por odometria
  ADVANCE         -> avanca brevemente apos curva para entrar no novo corredor

Topicos:
  Assina: /jetauto/lidar/scan, /odometry/filtered, /jetauto/camera/image_raw (Gazebo URDF)
  Publica: /jetauto/cmd_vel

Seguranca frontal (heuristica, nao garantia formal):
  - Se frente <= FRONT_BLOCKED: para e inicia giro.
  - Velocidade linear.x reduz entre FRONT_BLOCKED e FRONT_SLOW_FULL (freio suave).
"""

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

from jetauto_maze_solver.camera_color_config import (
    COLOR_MIN_AREA_FRAC as _CAM_COLOR_MIN_AREA_FRAC,
    dominant_wall_color_from_bgr,
)


# =====================================================================
# Utilitarios de angulo
# =====================================================================

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


# =====================================================================
# No principal
# =====================================================================

class MazeNavigator(Node):

    # -- Setores do LiDAR (radianos) --
    FRONT_LO = math.radians(-20)
    FRONT_HI = math.radians(20)
    LEFT_LO  = math.radians(70)
    LEFT_HI  = math.radians(110)
    RIGHT_LO = math.radians(-110)
    RIGHT_HI = math.radians(-70)

    # Raios individuais para calculo de heading (dois por parede)
    R_SIDE_ANGLE = math.radians(-90)
    R_DIAG_ANGLE = math.radians(-45)
    L_SIDE_ANGLE = math.radians(90)
    L_DIAG_ANGLE = math.radians(45)

    # -- Limiares de distancia (metros) --
    # Below FRONT_BLOCKED: stop forward motion and plan turn (collision avoidance).
    FRONT_BLOCKED = 0.5
    # Between FRONT_BLOCKED and FRONT_SLOW_FULL: scale linear.x down (soft braking).
    FRONT_SLOW_FULL = 0.95
    WALL_DETECT    = 1.2
    SIDE_OPEN      = 1.0    # distancia acima da qual o lado esta "aberto"
    TARGET_SIDE    = 0.4

    # -- Controle lateral -- linear.y --
    KP_LAT    = 0.3
    LAT_CLAMP = 0.15

    # -- Correcao de heading -- angular.z --
    ALIGN_KP    = 1.5
    ALIGN_CLAMP = 0.1

    # -- Velocidades --
    FORWARD_SPEED = 0.15
    TURN_SPEED    = 0.4
    YAW_TOLERANCE = 0.05    # rad (~2.9 graus)

    # -- LiDAR --
    RANGE_CAP       = 3.0
    LIDAR_MIN_VALID = 0.15

    # -- Temporizacoes --
    ADVANCE_SECS       = 1.5   # segundos avancando apos curva
    TURN_COOLDOWN_SECS = 2.0   # cooldown antes de checar parede aberta de novo

    # -- Camera (same HSV / area rules as color_wall_counter via camera_color_config) --
    COLOR_MIN_AREA_FRAC = _CAM_COLOR_MIN_AREA_FRAC

    def __init__(self):
        super().__init__('maze_navigator')

        self.create_subscription(LaserScan, '/jetauto/lidar/scan', self._scan_cb, 10)
        self.create_subscription(Odometry, '/odometry/filtered', self._odom_cb, 10)
        self.create_subscription(Image, '/jetauto/camera/image_raw', self._image_cb, 10)

        self.cmd_pub = self.create_publisher(Twist, '/jetauto/cmd_vel', 10)
        self.bridge = CvBridge()
        self.timer = self.create_timer(0.1, self._control_loop)

        # -- LiDAR --
        self.front_dist = self.RANGE_CAP
        self.left_dist  = self.RANGE_CAP
        self.right_dist = self.RANGE_CAP
        self.r_side = self.RANGE_CAP
        self.r_diag = self.RANGE_CAP
        self.l_side = self.RANGE_CAP
        self.l_diag = self.RANGE_CAP
        self.scan_ready = False

        # -- Odometria --
        self.current_yaw = 0.0
        self.odom_ready = False

        # -- Camera --
        self.detected_color = None   # 'vermelho', 'verde', 'azul' ou None

        # -- Maquina de estados --
        self.state = 'FOLLOW_CORRIDOR'
        self.target_yaw      = 0.0
        self.turn_direction   = 0.0
        self.loop_count       = 0

        # -- Regra da mao direita --
        self._had_right_wall      = False
        self._turn_cooldown_end   = 0.0
        self._advance_start       = 0.0

        self.get_logger().info(
            '[NAV] Maze Navigator iniciado — Regra da Mao Direita')

    # =================================================================
    # Callbacks dos sensores
    # =================================================================

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
        """Detecta cor dominante (vermelho/verde/azul) — mesma logica que color_wall_counter."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'[NAV] cv_bridge: {e}')
            return

        c, _, _ = dominant_wall_color_from_bgr(
            cv_image, min_area_fraction=self.COLOR_MIN_AREA_FRAC
        )
        self.detected_color = c

    # =================================================================
    # Calculos de controle
    # =================================================================

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _forward_speed_safe(self) -> float:
        """
        Forward speed scaled by frontal LiDAR distance. No mathematical guarantee of
        no contact (sensor rate, noise, inertia), but avoids driving at full speed
        into the last half-metre in front of the robot.
        """
        fd = self.front_dist
        if fd <= self.FRONT_BLOCKED:
            return 0.0
        if fd >= self.FRONT_SLOW_FULL:
            return self.FORWARD_SPEED
        span = self.FRONT_SLOW_FULL - self.FRONT_BLOCKED
        if span <= 1e-6:
            return self.FORWARD_SPEED
        t = (fd - self.FRONT_BLOCKED) / span
        # Keep a small minimum so the robot can still creep in very tight geometry
        return self.FORWARD_SPEED * max(0.12, min(1.0, t))

    def _start_turn(self, target_yaw, direction, desc):
        """Inicia um giro para target_yaw."""
        self.target_yaw = target_yaw
        self.turn_direction = direction
        self.state = 'TURNING'
        self.get_logger().info(
            f'[NAV] -> TURNING {desc}  '
            f'F={self.front_dist:.2f} Lray={self.l_side:.2f} Rray={self.r_side:.2f}  '
            f'target={math.degrees(target_yaw):.1f} graus')

    def _choose_turn_front_blocked(self):
        """
        Decide a direcao do giro quando a frente esta bloqueada.

        Prioridade:
          1. Camera detectou VERMELHO ou AZUL -> esquerda
          2. Camera detectou VERDE            -> direita
          3. Sem cor -> regra da mao direita:
             direita livre -> direita
             esquerda livre -> esquerda
             tudo bloqueado -> meia-volta (180 graus)
        """
        color = self.detected_color

        if color == 'vermelho' or color == 'azul':
            target = normalize_angle(self.current_yaw + math.pi / 2.0)
            return target, 1.0, f'esquerda (parede {color.upper()})'

        if color == 'verde':
            target = normalize_angle(self.current_yaw - math.pi / 2.0)
            return target, -1.0, 'direita (parede VERDE)'

        # Sem cor: regra da mao direita
        # Usa raios perpendiculares (90/-90 graus) para detectar aberturas
        # Threshold = WALL_DETECT (1.2m) — paredes do corredor ficam a ~0.75m
        if self.r_side > self.WALL_DETECT:
            target = normalize_angle(self.current_yaw - math.pi / 2.0)
            return target, -1.0, 'direita (mao direita)'

        if self.l_side > self.WALL_DETECT:
            target = normalize_angle(self.current_yaw + math.pi / 2.0)
            return target, 1.0, 'esquerda (unica saida)'

        # Beco sem saida -> meia-volta
        target = normalize_angle(self.current_yaw + math.pi)
        return target, 1.0, 'MEIA-VOLTA (beco sem saida)'

    def _lateral_correction(self) -> float:
        """Calcula linear.y para centralizar o robo no corredor."""
        L = self.left_dist
        R = self.right_dist
        left_wall  = L < self.WALL_DETECT
        right_wall = R < self.WALL_DETECT

        if left_wall and right_wall:
            error = (L - R) / 2.0
        elif right_wall:
            error = self.TARGET_SIDE - R
        elif left_wall:
            error = L - self.TARGET_SIDE
        else:
            error = 0.0

        return max(-self.LAT_CLAMP, min(self.KP_LAT * error, self.LAT_CLAMP))

    def _heading_correction(self) -> float:
        """Calcula angular.z para manter o robo alinhado com o corredor."""
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

    # =================================================================
    # Loop de controle principal (10 Hz)
    # =================================================================

    def _control_loop(self):
        if not self.scan_ready or not self.odom_ready:
            return

        self.loop_count += 1
        twist = Twist()

        if self.state == 'TURNING':
            self._state_turning(twist)
        elif self.state == 'ADVANCE':
            self._state_advance(twist)
        else:
            self._state_follow_corridor(twist)

        self.cmd_pub.publish(twist)

    # -----------------------------------------------------------------
    # ESTADO: FOLLOW_CORRIDOR
    # -----------------------------------------------------------------

    def _state_follow_corridor(self, twist: Twist):
        """
        Avanca reto, centralizando e corrigindo heading.

        Transicoes (regra da mao direita):
          1. Frente bloqueada -> decide via cor ou mao-direita
          2. Parede direita sumiu -> vira a direita (proativo)
        """
        now = self._now_sec()

        # -- Prioridade 1: frente bloqueada (seguranca) --
        if self.front_dist <= self.FRONT_BLOCKED:
            target, direction, desc = self._choose_turn_front_blocked()
            self._start_turn(target, direction, desc)
            twist.linear.x = 0.0
            twist.linear.y = 0.0
            twist.angular.z = 0.0
            return

        # -- Prioridade 2: regra da mao direita -- corredor abriu a direita --
        # Usa raio perpendicular (-90 graus) para detectar abertura
        right_open = self.r_side > self.SIDE_OPEN
        if right_open and self._had_right_wall and now > self._turn_cooldown_end:
            target = normalize_angle(self.current_yaw - math.pi / 2.0)
            self._start_turn(target, -1.0, 'direita (corredor abriu)')
            twist.linear.x = 0.0
            twist.linear.y = 0.0
            twist.angular.z = 0.0
            return

        # -- Atualiza rastreamento da parede direita --
        if self.r_side < self.WALL_DETECT:
            self._had_right_wall = True

        # -- Movimento normal --
        twist.linear.x  = self._forward_speed_safe()
        twist.linear.y  = self._lateral_correction()
        twist.angular.z = self._heading_correction()

        if self.loop_count % 20 == 0:
            self.get_logger().info(
                f'[NAV] FOLLOW  '
                f'F={self.front_dist:.2f} Lray={self.l_side:.2f} Rray={self.r_side:.2f}  '
                f'vy={twist.linear.y:+.3f} az={twist.angular.z:+.3f}  '
                f'rwall={self._had_right_wall} cor={self.detected_color}')

    # -----------------------------------------------------------------
    # ESTADO: TURNING
    # -----------------------------------------------------------------

    def _state_turning(self, twist: Twist):
        """Gira no lugar ate atingir target_yaw."""
        error = normalize_angle(self.target_yaw - self.current_yaw)

        if abs(error) < self.YAW_TOLERANCE:
            self.state = 'ADVANCE'
            self._advance_start = self._now_sec()
            self.get_logger().info(
                f'[NAV] TURNING concluido -> ADVANCE  '
                f'yaw={math.degrees(self.current_yaw):.1f} graus')
            twist.linear.x  = 0.0
            twist.linear.y  = 0.0
            twist.angular.z = 0.0
        else:
            twist.linear.x  = 0.0
            twist.linear.y  = 0.0
            twist.angular.z = self.TURN_SPEED * self.turn_direction
            if self.loop_count % 5 == 0:
                self.get_logger().info(
                    f'[NAV] TURNING  erro={math.degrees(error):.1f} graus  '
                    f'yaw={math.degrees(self.current_yaw):.1f}  '
                    f'target={math.degrees(self.target_yaw):.1f}')

    # -----------------------------------------------------------------
    # ESTADO: ADVANCE
    # -----------------------------------------------------------------

    def _state_advance(self, twist: Twist):
        """
        Avanca brevemente apos uma curva para entrar no novo corredor.
        Usa correcao lateral e de heading para nao bater nas paredes.
        """
        elapsed = self._now_sec() - self._advance_start

        if elapsed >= self.ADVANCE_SECS or self.front_dist <= 0.3:
            self.state = 'FOLLOW_CORRIDOR'
            self._had_right_wall = self.r_side < self.WALL_DETECT
            self._turn_cooldown_end = self._now_sec() + self.TURN_COOLDOWN_SECS
            self.get_logger().info(
                f'[NAV] ADVANCE concluido -> FOLLOW_CORRIDOR  '
                f'rwall={self._had_right_wall}')
            twist.linear.x  = 0.0
            twist.linear.y  = 0.0
            twist.angular.z = 0.0
            return

        twist.linear.x  = self._forward_speed_safe()
        twist.linear.y  = self._lateral_correction()
        twist.angular.z = self._heading_correction()

    # -----------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------

    def destroy_node(self):
        try:
            self.cmd_pub.publish(Twist())
        except Exception:
            pass
        super().destroy_node()


# =====================================================================
# Ponto de entrada
# =====================================================================

def main(args=None):
    rclpy.init(args=args)
    node = MazeNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
