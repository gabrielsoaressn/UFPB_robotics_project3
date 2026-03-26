#!/usr/bin/env python3
"""
Maze Navigator — Máquina de Estados para Regra da Mão Direita (Robô Diferencial)

Restrição absoluta: linear.y = 0.0 SEMPRE.

Estados:
  INITIAL_STRAIGHT → Largada segura: vai reto até frente bloqueada
  FOLLOW_WALL      → Andar reto com alinhamento paralelo à parede direita
  INNER_CORNER     → Frente bloqueada → parar e girar 90° à esquerda (via TURNING)
  OUTER_CORNER     → Parede direita sumiu → avançar 0.3m reto, depois girar 90° à direita
  TURNING          → Giro controlado por odometria até atingir target_yaw

Tópicos:
  Assina: /jetauto/lidar/scan, /odometry/filtered
  Publica: /jetauto/cmd_vel
"""

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist


# ═══════════════════════════════════════════════════════════════════════
# Utilidades de ângulo
# ═══════════════════════════════════════════════════════════════════════

def yaw_from_quaternion(q):
    """Extrai yaw (rotação em Z) de um quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle):
    """Normaliza ângulo para o intervalo [-π, π]."""
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
    # O LiDAR cobre -135° a +135° com 180 amostras.
    # Frente: cone estreito de ±20° ao redor de 0°
    FRONT_LO = math.radians(-20)
    FRONT_HI = math.radians(20)
    # Direita: setor de -100° a -50° (ângulos negativos = lado direito)
    RIGHT_LO = math.radians(-100)
    RIGHT_HI = math.radians(-50)

    # ── Limiares de distância (metros) ─────────────────────────────
    FRONT_BLOCKED = 0.6     # frente bloqueada → INNER_CORNER
    RIGHT_PRESENT = 0.8     # considera parede presente à direita
    RIGHT_LOST    = 0.85    # parede sumiu → OUTER_CORNER

    # ── Alinhamento paralelo (dois raios à direita) ──────────────
    ALIGN_SIDE_ANGLE = math.radians(-90)   # raio perpendicular à parede
    ALIGN_DIAG_ANGLE = math.radians(-45)   # raio diagonal frontal-direita
    ALIGN_KP    = 1.5    # ganho proporcional de alinhamento
    ALIGN_CLAMP = 0.1    # correção angular máxima (rad/s)

    # ── Velocidades ────────────────────────────────────────────────
    FORWARD_SPEED = 0.15    # m/s linear para frente
    TURN_SPEED    = 0.4     # rad/s durante giro no estado TURNING

    # ── OUTER_CORNER: distância de avanço antes de girar ───────────
    ADVANCE_DIST = 0.3      # metros

    # ── TURNING: tolerância angular ────────────────────────────────
    YAW_TOLERANCE = 0.05    # radianos (~2.9°)

    # ── Processamento do LiDAR ─────────────────────────────────────
    RANGE_CAP       = 3.0   # distância máxima considerada (ignora inf)
    LIDAR_MIN_VALID = 0.15  # leituras abaixo disto são ruído do sensor

    def __init__(self):
        super().__init__('maze_navigator')

        # ── Subscribers ──
        self.create_subscription(
            LaserScan, '/jetauto/lidar/scan', self._scan_cb, 10)
        self.create_subscription(
            Odometry, '/odometry/filtered', self._odom_cb, 10)

        # ── Publisher ──
        self.cmd_pub = self.create_publisher(Twist, '/jetauto/cmd_vel', 10)

        # ── Timer de controle (10 Hz) ──
        self.timer = self.create_timer(0.1, self._control_loop)

        # ── Dados do LiDAR ──
        self.front_dist = self.RANGE_CAP
        self.right_dist = self.RANGE_CAP
        self.align_side = self.RANGE_CAP   # raio a -90°
        self.align_diag = self.RANGE_CAP   # raio a -45°
        self.scan_ready = False

        # ── Dados da odometria ──
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.current_yaw = 0.0
        self.odom_ready = False

        # ── Máquina de Estados ──
        self.state = 'INITIAL_STRAIGHT'
        self.target_yaw = 0.0         # yaw alvo para o estado TURNING
        self.turn_direction = 0.0     # +1.0 = esquerda, -1.0 = direita
        self.advance_start_x = 0.0   # snapshot de odom para OUTER_CORNER
        self.advance_start_y = 0.0

        # Só ativa OUTER_CORNER após encontrar parede pela primeira vez,
        # evitando falso trigger na entrada do labirinto.
        self.wall_found = False

        # Contador para throttle de logs
        self.loop_count = 0

        self.get_logger().info(
            '[NAV] ═══ Maze Navigator Iniciado ═══ '
            'Máquina de Estados — Regra da Mão Direita')

    # ══════════════════════════════════════════════════════════════════
    # Callbacks dos sensores
    # ══════════════════════════════════════════════════════════════════

    def _odom_cb(self, msg: Odometry):
        """Atualiza posição (x, y) e yaw a partir da odometria filtrada (EKF)."""
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        self.current_yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self.odom_ready = True

    def _scan_cb(self, msg: LaserScan):
        """Processa LiDAR: extrai distância mínima nos setores frontal e direito."""
        ranges = msg.ranges
        n = len(ranges)
        if n == 0:
            return

        a_min = msg.angle_min
        a_inc = msg.angle_increment

        def idx(angle_rad):
            """Converte ângulo (rad) em índice do array de ranges."""
            return max(0, min(n - 1, round((angle_rad - a_min) / a_inc)))

        def sector_min(lo_rad, hi_rad):
            """Retorna a menor distância válida dentro de um setor angular."""
            i_lo = idx(lo_rad)
            i_hi = idx(hi_rad)
            best = self.RANGE_CAP
            for i in range(min(i_lo, i_hi), max(i_lo, i_hi) + 1):
                r = ranges[i]
                if not math.isfinite(r) or r < self.LIDAR_MIN_VALID:
                    continue
                best = min(best, min(r, self.RANGE_CAP))
            return best

        def ray_at(angle_rad):
            """Retorna a distância de um raio específico."""
            i = idx(angle_rad)
            r = ranges[i]
            if not math.isfinite(r) or r < self.LIDAR_MIN_VALID:
                return self.RANGE_CAP
            return min(r, self.RANGE_CAP)

        self.front_dist = sector_min(self.FRONT_LO, self.FRONT_HI)
        self.right_dist = sector_min(self.RIGHT_LO, self.RIGHT_HI)
        self.align_side = ray_at(self.ALIGN_SIDE_ANGLE)
        self.align_diag = ray_at(self.ALIGN_DIAG_ANGLE)
        self.scan_ready = True

    # ══════════════════════════════════════════════════════════════════
    # Máquina de Estados — Loop de Controle Principal (10 Hz)
    # ══════════════════════════════════════════════════════════════════

    def _control_loop(self):
        """Despacha o controle para o estado atual da máquina."""
        if not self.scan_ready or not self.odom_ready:
            return

        self.loop_count += 1
        F = self.front_dist
        R = self.right_dist

        # Cria mensagem de velocidade — linear.y = 0.0 SEMPRE
        twist = Twist()
        twist.linear.y = 0.0

        # Despacha para o estado atual
        if self.state == 'INITIAL_STRAIGHT':
            self._state_initial_straight(twist, F)
        elif self.state == 'TURNING':
            self._state_turning(twist)
        elif self.state == 'OUTER_CORNER':
            self._state_outer_corner(twist, F)
        else:
            # FOLLOW_WALL é o estado padrão
            self._state_follow_wall(twist, F, R)

        # Publica o comando de velocidade
        self.cmd_pub.publish(twist)

    # ──────────────────────────────────────────────────────────────────
    # ESTADO: INITIAL_STRAIGHT (Largada Segura)
    # ──────────────────────────────────────────────────────────────────

    def _state_initial_straight(self, twist: Twist, F: float):
        """
        Largada: vai puramente reto ignorando paredes laterais.
        Quando a frente bloqueia (< 0.6m), entra em INNER_CORNER (giro esquerda).
        """

        if F <= self.FRONT_BLOCKED:
            self.target_yaw = normalize_angle(
                self.current_yaw + math.pi / 2.0)
            self.turn_direction = 1.0
            self.state = 'TURNING'
            self.get_logger().info(
                f'[NAV] INITIAL_STRAIGHT → TURNING esquerda  '
                f'F={F:.2f}m  target_yaw={math.degrees(self.target_yaw):.1f}°')
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            return

        twist.linear.x = self.FORWARD_SPEED
        twist.angular.z = 0.0

        if self.loop_count % 20 == 0:
            self.get_logger().info(
                f'[NAV] INITIAL_STRAIGHT  F={F:.2f}m  (indo reto)')

    # ──────────────────────────────────────────────────────────────────
    # ESTADO: FOLLOW_WALL (Padrão)
    # ──────────────────────────────────────────────────────────────────

    def _state_follow_wall(self, twist: Twist, F: float, R: float):
        """
        Anda reto com correção de alinhamento paralelo à parede direita.

        Usa dois raios (-90° e -45°) para calcular o ângulo do robô em
        relação à parede. Se há drift, aplica correção proporcional suave.
        Sem parede visível, vai puramente reto.

        Transições:
          - Frente bloqueada (F <= 0.6m) → TURNING esquerda 90°
          - Parede sumiu (R > 0.85m)     → OUTER_CORNER → avanço + TURNING direita 90°
        """

        # ── PRIORIDADE MÁXIMA: Frente bloqueada → giro 90° esquerda ──
        if F <= self.FRONT_BLOCKED:
            self.target_yaw = normalize_angle(
                self.current_yaw + math.pi / 2.0)
            self.turn_direction = 1.0
            self.state = 'TURNING'
            self.get_logger().info(
                f'[NAV] FOLLOW_WALL → TURNING esquerda  '
                f'F={F:.2f}m  R={R:.2f}m  '
                f'target_yaw={math.degrees(self.target_yaw):.1f}°')
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            return

        # ── Parede direita sumiu → OUTER_CORNER ──
        if self.wall_found and R > self.RIGHT_LOST:
            self.state = 'OUTER_CORNER'
            self.advance_start_x = self.odom_x
            self.advance_start_y = self.odom_y
            self.get_logger().info(
                f'[NAV] FOLLOW_WALL → OUTER_CORNER  '
                f'R={R:.2f}m  F={F:.2f}m')
            twist.linear.x = self.FORWARD_SPEED
            twist.angular.z = 0.0
            return

        # ── Marca parede detectada (habilita OUTER_CORNER no futuro) ──
        if not self.wall_found and R < self.RIGHT_PRESENT:
            self.wall_found = True
            self.get_logger().info(
                f'[NAV] Parede direita confirmada (R={R:.2f}m)')

        # ── Andar reto com alinhamento paralelo ──
        twist.linear.x = self.FORWARD_SPEED

        a = self.align_side  # raio a -90° (perpendicular)
        b = self.align_diag  # raio a -45° (diagonal)

        if a < self.RIGHT_PRESENT and b < self.RANGE_CAP - 0.1:
            # Pontos onde os raios atingem a parede:
            #   P_side = (0, -a)                          (raio a -90°)
            #   P_diag = (b*cos45, -b*sin45)              (raio a -45°)
            # Vetor da parede: P_diag - P_side
            #   dx = b*cos45,  dy = a - b*sin45
            # Se paralelo → dy = 0. Desvio → heading_error = atan2(dy, dx)
            dx = b * 0.7071
            dy = a - b * 0.7071
            heading_error = math.atan2(dy, dx)
            az = self.ALIGN_KP * heading_error
            twist.angular.z = max(-self.ALIGN_CLAMP, min(az, self.ALIGN_CLAMP))
        else:
            twist.angular.z = 0.0

        if self.loop_count % 20 == 0:
            self.get_logger().info(
                f'[NAV] FOLLOW_WALL  F={F:.2f}m  R={R:.2f}m  '
                f'side={a:.2f}m  diag={b:.2f}m  az={twist.angular.z:+.3f}')

    # ──────────────────────────────────────────────────────────────────
    # ESTADO: OUTER_CORNER (Quina à Direita)
    # ──────────────────────────────────────────────────────────────────

    def _state_outer_corner(self, twist: Twist, F: float):
        """
        A parede direita sumiu — estamos numa quina externa.
        1) Avança reto 0.3m para que o eixo traseiro passe da quina.
        2) Depois do avanço, calcula target_yaw = -90° e vai para TURNING.

        Segurança: se frente bloquear durante o avanço, desvia à esquerda.
        """

        # ── Segurança: frente bloqueada durante avanço → virar esquerda ──
        if F <= self.FRONT_BLOCKED:
            self.target_yaw = normalize_angle(
                self.current_yaw + math.pi / 2.0)
            self.turn_direction = 1.0
            self.state = 'TURNING'
            self.get_logger().info(
                f'[NAV] OUTER_CORNER interrompido (frente bloqueada F={F:.2f}m) '
                f'→ TURNING esquerda')
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            return

        # Calcula distância percorrida desde o início do avanço
        dx = self.odom_x - self.advance_start_x
        dy = self.odom_y - self.advance_start_y
        dist = math.hypot(dx, dy)

        if dist >= self.ADVANCE_DIST:
            # ── Avanço concluído → girar 90° à direita ──
            self.target_yaw = normalize_angle(
                self.current_yaw - math.pi / 2.0)
            self.turn_direction = -1.0  # direita (angular.z negativo)
            self.state = 'TURNING'
            self.get_logger().info(
                f'[NAV] OUTER_CORNER avanço concluído ({dist:.2f}m) '
                f'→ TURNING direita  '
                f'target_yaw={math.degrees(self.target_yaw):.1f}°')
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        else:
            # Ainda avançando reto (sem girar)
            twist.linear.x = self.FORWARD_SPEED
            twist.angular.z = 0.0

    # ──────────────────────────────────────────────────────────────────
    # ESTADO: TURNING (Giro Controlado por Odometria)
    # ──────────────────────────────────────────────────────────────────

    def _state_turning(self, twist: Twist):
        """
        Gira no lugar (linear.x = 0) na direção planejada até atingir
        target_yaw com tolerância de ±0.05 rad.
        """
        error = normalize_angle(self.target_yaw - self.current_yaw)

        if abs(error) < self.YAW_TOLERANCE:
            # ── Giro concluído → voltar para FOLLOW_WALL ──
            self.state = 'FOLLOW_WALL'
            self.wall_found = False  # re-detectar parede antes de habilitar OUTER_CORNER
            self.get_logger().info(
                f'[NAV] TURNING concluído → FOLLOW_WALL  '
                f'yaw={math.degrees(self.current_yaw):.1f}°  '
                f'erro_final={math.degrees(error):.1f}°')
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        else:
            # ── Continua girando na direção planejada ──
            twist.linear.x = 0.0
            twist.angular.z = self.TURN_SPEED * self.turn_direction
            # Log periódico durante giro
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
        """Para o robô (velocidade zero) antes de destruir o nó."""
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
