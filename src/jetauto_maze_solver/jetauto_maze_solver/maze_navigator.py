#!/usr/bin/env python3
"""
Maze Navigator — Centro do Corredor (Robô Omnidirecional)

O robô usa o LiDAR para medir as distâncias às paredes esquerda e direita
e emprega linear.y para se manter centrado no corredor enquanto avança.
Quando a frente está bloqueada, entra em COLOR_CHECK (para 0.5 s, amostra a cor)
e depois gira na direção indicada pela parede:
  Vermelho → esquerda  |  Verde → direita  |  Sem cor → célula menos visitada

Anti-backtracking: mantém um mapa de células visitadas (grid 0.4 m)
e, quando não há cor, prefere a direção com menos células visitadas à frente.

Estados:
  FOLLOW_CORRIDOR → avança, centraliza lateralmente, corrige heading
  COLOR_CHECK     → para o robô e amostra a cor por 0.5 s para decisão fiável
  TURNING         → giro no lugar controlado por odometria

Tópicos:
  Assina: /jetauto/lidar/scan, /odometry/filtered, /jetauto/camera/image_raw
  Publica: /jetauto/cmd_vel
"""

import math
from collections import Counter
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

    # Raios individuais para cálculo de heading
    R_SIDE_ANGLE = math.radians(-90)
    R_DIAG_ANGLE = math.radians(-45)
    L_SIDE_ANGLE = math.radians(90)
    L_DIAG_ANGLE = math.radians(45)

    # ── Limiares de distância (metros) ─────────────────────────────
    FRONT_BLOCKED = 0.7     # frente bloqueada → inicia COLOR_CHECK (antecipado)
    SIDE_BLOCKED  = 0.7     # lateral bloqueada → critério de beco sem saída
    WALL_DETECT   = 1.2     # considera parede presente se < este valor
    TARGET_SIDE   = 0.4     # distância desejada à parede (só um lado)

    # ── Desaceleração proporcional à distância frontal ───────────────
    # O robô começa a reduzir a velocidade quando front_dist < SLOW_DIST,
    # chegando ao mínimo MIN_SPEED próximo de FRONT_BLOCKED.
    SLOW_DIST  = 1.2    # metros — começa a desacelerar aqui
    MIN_SPEED  = 0.05   # m/s — velocidade mínima antes de parar

    # ── Controle lateral — linear.y ─────────────────────────────────
    KP_LAT    = 0.15    # ganho proporcional (reduzido para suavidade)
    LAT_CLAMP = 0.10    # velocidade lateral máxima (m/s)
    LAT_ALPHA = 0.25    # coef. de suavização EMA (menor = mais suave)

    # ── Correção de heading — angular.z ─────────────────────────────
    ALIGN_KP    = 1.5
    ALIGN_CLAMP = 0.1

    # ── Velocidades ─────────────────────────────────────────────────
    FORWARD_SPEED = 0.15
    TURN_SPEED    = 0.4
    YAW_TOLERANCE = 0.05    # rad (~2.9°)

    # ── LiDAR ───────────────────────────────────────────────────────
    RANGE_CAP       = 3.0
    LIDAR_MIN_VALID = 0.15

    # ── COLOR_CHECK ─────────────────────────────────────────────────
    # Número de ciclos (10 Hz) em que o robô fica parado amostrado a cor
    # antes de decidir a direção do giro. 5 ciclos = 0.5 s.
    COLOR_CHECK_CYCLES = 5

    # ── Anti-backtracking (células visitadas) ───────────────────────
    CELL_SIZE       = 0.4   # tamanho da célula do grid (metros)
    BACKTRACK_DISTS = (1.0, 1.5, 2.0)  # distâncias de projeção para avaliação

    # ── Câmera — faixas HSV ─────────────────────────────────────────
    COLOR_MIN_AREA_FRAC = 0.05
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
        self.left_max   = self.RANGE_CAP
        self.right_max  = self.RANGE_CAP
        self.r_side = self.RANGE_CAP
        self.r_diag = self.RANGE_CAP
        self.l_side = self.RANGE_CAP
        self.l_diag = self.RANGE_CAP
        self.scan_ready = False

        # ── Odometria ──
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.current_yaw = 0.0
        self.odom_ready = False

        # ── Câmera ──
        self.detected_color = None   # 'vermelho', 'verde' ou None

        # ── Controle lateral suavizado ──
        self.lat_cmd = 0.0   # valor filtrado por EMA

        # ── Anti-backtracking ──
        self.visited_cells: set = set()

        # ── Máquina de estados ──
        self.state = 'FOLLOW_CORRIDOR'
        self.target_yaw      = 0.0
        self.turn_direction  = 0.0   # +1 = esquerda, -1 = direita
        self.color_check_count   = 0
        self.color_check_samples = []
        self.consecutive_blocks  = 0   # giros seguidos com frente ainda bloqueada
        self.loop_count = 0

        self.get_logger().info(
            '[NAV] Maze Navigator iniciado — Centro do Corredor + Paredes Coloridas')

    # ══════════════════════════════════════════════════════════════════
    # Callbacks dos sensores
    # ══════════════════════════════════════════════════════════════════

    def _odom_cb(self, msg: Odometry):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
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

        def sector_max(lo, hi):
            best = 0.0
            for i in range(min(idx(lo), idx(hi)), max(idx(lo), idx(hi)) + 1):
                r = ranges[i]
                if math.isfinite(r) and r >= self.LIDAR_MIN_VALID:
                    best = max(best, min(r, self.RANGE_CAP))
            return best

        def ray_at(a):
            r = ranges[idx(a)]
            if math.isfinite(r) and r >= self.LIDAR_MIN_VALID:
                return min(r, self.RANGE_CAP)
            return self.RANGE_CAP

        self.front_dist = sector_min(self.FRONT_LO, self.FRONT_HI)
        self.left_dist  = sector_min(self.LEFT_LO,  self.LEFT_HI)
        self.right_dist = sector_min(self.RIGHT_LO, self.RIGHT_HI)
        self.left_max   = sector_max(self.LEFT_LO,  self.LEFT_HI)
        self.right_max  = sector_max(self.RIGHT_LO, self.RIGHT_HI)
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

    def _pos_to_cell(self, x: float, y: float):
        return (math.floor(x / self.CELL_SIZE), math.floor(y / self.CELL_SIZE))

    def _visited_score(self, new_yaw: float) -> int:
        """
        Conta quantas das células projetadas à frente (na direção new_yaw)
        já foram visitadas. Quanto menor, menos explorada é essa direção.
        """
        score = 0
        for dist in self.BACKTRACK_DISTS:
            px = self.odom_x + dist * math.cos(new_yaw)
            py = self.odom_y + dist * math.sin(new_yaw)
            if self._pos_to_cell(px, py) in self.visited_cells:
                score += 1
        return score

    def _choose_turn(self, color_decision: str):
        """
        Decide a direção do giro.

        Prioridade:
          0. Beco sem saída → U-turn 180°
             Critério A (geométrico): nenhum raio lateral vê além de SIDE_BLOCKED
             Critério B (contador):   frente bloqueada 2x consecutivas sem sair
          1. VERMELHO → esquerda (+90°)
          2. VERDE    → direita  (-90°)
          3. Sem cor  → direção com menor score de células visitadas.
                        Empate: esquerda.
        """
        # ── Beco sem saída ──
        geo_deadend   = (self.left_max  <= self.SIDE_BLOCKED
                         and self.right_max <= self.SIDE_BLOCKED)
        self.consecutive_blocks += 1
        count_deadend = self.consecutive_blocks >= 2

        if geo_deadend or count_deadend:
            self.consecutive_blocks = 0
            target = normalize_angle(self.current_yaw + math.pi)
            reason = 'geométrico' if geo_deadend else 'bloqueio consecutivo'
            self.get_logger().warn(f'[NAV] Beco detectado ({reason}) → U-turn')
            return target, 1.0, f'U-turn 180° ({reason})'

        # ── Cor da câmera ──
        effective_color = color_decision

        if effective_color == 'vermelho':
            target = normalize_angle(self.current_yaw + math.pi / 2.0)
            return target, 1.0, 'esquerda (parede VERMELHA)'
        elif effective_color == 'verde':
            target = normalize_angle(self.current_yaw - math.pi / 2.0)
            return target, -1.0, 'direita (parede VERDE)'
        else:
            yaw_L = normalize_angle(self.current_yaw + math.pi / 2.0)
            yaw_R = normalize_angle(self.current_yaw - math.pi / 2.0)

            left_open  = self.left_max  > self.WALL_DETECT
            right_open = self.right_max > self.WALL_DETECT

            if left_open and not right_open:
                return yaw_L, 1.0, f'esquerda (LiDAR: L aberto={self.left_max:.2f}m, R parede={self.right_max:.2f}m)'
            elif right_open and not left_open:
                return yaw_R, -1.0, f'direita (LiDAR: R aberto={self.right_max:.2f}m, L parede={self.left_max:.2f}m)'

            # Ambos abertos ou ambos fechados → usa anti-backtracking
            score_L = self._visited_score(yaw_L)
            score_R = self._visited_score(yaw_R)
            if score_L <= score_R:
                return yaw_L, 1.0, f'esquerda (visitadas: L={score_L} R={score_R})'
            else:
                return yaw_R, -1.0, f'direita (visitadas: L={score_L} R={score_R})'

    def _lateral_correction(self) -> float:
        """
        Calcula o erro lateral bruto para centralizar no corredor.
        A suavização EMA é aplicada no loop de controle.

        Convenção ROS: linear.y > 0 → move para esquerda.
        """
        L = self.left_dist
        R = self.right_dist
        left_wall  = L < self.WALL_DETECT
        right_wall = R < self.WALL_DETECT

        if left_wall and right_wall:
            error = (L - R) / 2.0
        elif right_wall:
            error = self.TARGET_SIDE - R   # positivo = muito perto da direita → move esq.
        elif left_wall:
            error = L - self.TARGET_SIDE   # negativo = muito perto da esquerda → move dir.
        else:
            error = 0.0

        raw = max(-self.LAT_CLAMP, min(self.KP_LAT * error, self.LAT_CLAMP))
        return raw

    def _heading_correction(self) -> float:
        """
        Calcula angular.z para manter o robô alinhado com o corredor.
        Usa geometria de dois raios por parede; combina ambas se presentes.

        Parede direita (raios -90° e -45°):
          Vetor da parede: dx = b·cos45, dy = a − b·sin45
        Parede esquerda (raios +90° e +45°):
          Vetor da parede: dx = d·cos45, dy = d·sin45 − c
        """
        corrections = []

        a, b = self.r_side, self.r_diag
        if a < self.WALL_DETECT and b < self.RANGE_CAP - 0.1:
            corrections.append(math.atan2(a - b * 0.7071, b * 0.7071))

        c, d = self.l_side, self.l_diag
        if c < self.WALL_DETECT and d < self.RANGE_CAP - 0.1:
            corrections.append(math.atan2(d * 0.7071 - c, d * 0.7071))

        if not corrections:
            return 0.0

        az = self.ALIGN_KP * (sum(corrections) / len(corrections))
        return max(-self.ALIGN_CLAMP, min(az, self.ALIGN_CLAMP))

    # ══════════════════════════════════════════════════════════════════
    # Loop de controle principal (10 Hz)
    # ══════════════════════════════════════════════════════════════════

    def _control_loop(self):
        if not self.scan_ready or not self.odom_ready:
            return

        self.loop_count += 1

        # Registra célula atual como visitada
        self.visited_cells.add(self._pos_to_cell(self.odom_x, self.odom_y))

        twist = Twist()

        if self.state == 'COLOR_CHECK':
            self._state_color_check(twist)
        elif self.state == 'TURNING':
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
          linear.y (suavizado) centraliza no corredor
          angular.z mantém heading paralelo às paredes

        Transição: frente bloqueada → COLOR_CHECK
        """
        if self.front_dist <= self.FRONT_BLOCKED:
            self.state = 'COLOR_CHECK'
            self.color_check_count   = 0
            self.color_check_samples = []
            self.lat_cmd = 0.0  # zera suavização ao parar
            self.get_logger().info(
                f'[NAV] FOLLOW_CORRIDOR → COLOR_CHECK  F={self.front_dist:.2f}m')
            twist.linear.x  = 0.0
            twist.linear.y  = 0.0
            twist.angular.z = 0.0
            return

        raw_lat = self._lateral_correction()
        # Filtro EMA: suaviza oscilações laterais
        self.lat_cmd = (1.0 - self.LAT_ALPHA) * self.lat_cmd + self.LAT_ALPHA * raw_lat

        # Velocidade proporcional: reduz à medida que se aproxima da parede frontal
        if self.front_dist < self.SLOW_DIST:
            t = (self.front_dist - self.FRONT_BLOCKED) / (self.SLOW_DIST - self.FRONT_BLOCKED)
            t = max(0.0, min(1.0, t))
            speed = self.MIN_SPEED + (self.FORWARD_SPEED - self.MIN_SPEED) * t
        else:
            speed = self.FORWARD_SPEED

        twist.linear.x  = speed
        twist.linear.y  = self.lat_cmd
        twist.angular.z = self._heading_correction()

        if self.loop_count % 20 == 0:
            self.get_logger().info(
                f'[NAV] FOLLOW_CORRIDOR  '
                f'F={self.front_dist:.2f}m  '
                f'L={self.left_dist:.2f}m  R={self.right_dist:.2f}m  '
                f'vy={twist.linear.y:+.3f}  az={twist.angular.z:+.3f}  '
                f'cor={self.detected_color}')

    # ──────────────────────────────────────────────────────────────────
    # ESTADO: COLOR_CHECK
    # ──────────────────────────────────────────────────────────────────

    def _state_color_check(self, twist: Twist):
        """
        Para o robô e amostra a cor da câmera por COLOR_CHECK_CYCLES ciclos.
        Ao final usa a cor mais frequente para decidir a direção do giro.

        Caso a frente se abra novamente (falso positivo), cancela e volta
        para FOLLOW_CORRIDOR.
        """
        # Segurança: frente abriu → cancelar
        if self.front_dist > self.FRONT_BLOCKED + 0.1:
            self.state = 'FOLLOW_CORRIDOR'
            self.get_logger().info('[NAV] COLOR_CHECK cancelado (frente abriu)')
            twist.linear.x = twist.linear.y = twist.angular.z = 0.0
            return

        # Mantém robô parado
        twist.linear.x  = 0.0
        twist.linear.y  = 0.0
        twist.angular.z = 0.0

        # Coleta amostra de cor
        self.color_check_samples.append(self.detected_color)
        self.color_check_count += 1

        if self.color_check_count < self.COLOR_CHECK_CYCLES:
            return  # ainda coletando amostras

        # ── Decisão: cor mais frequente nas amostras ──
        counts = Counter(self.color_check_samples)
        color_decision = counts.most_common(1)[0][0]  # None também conta

        self.target_yaw, self.turn_direction, desc = self._choose_turn(color_decision)
        self.state = 'TURNING'
        self.get_logger().info(
            f'[NAV] COLOR_CHECK → TURNING {desc}  '
            f'amostras={dict(counts)}  '
            f'target={math.degrees(self.target_yaw):.1f}°')

    # ──────────────────────────────────────────────────────────────────
    # ESTADO: TURNING
    # ──────────────────────────────────────────────────────────────────

    def _state_turning(self, twist: Twist):
        """Gira no lugar até atingir target_yaw com tolerância YAW_TOLERANCE."""
        error = normalize_angle(self.target_yaw - self.current_yaw)

        if abs(error) < self.YAW_TOLERANCE:
            self.lat_cmd = 0.0  # zera suavização ao sair do giro
            twist.linear.x  = 0.0
            twist.linear.y  = 0.0
            twist.angular.z = 0.0

            # Verificação pós-giro: frente ainda bloqueada?
            # Pode indicar beco sem saída (precisa de +90°) ou drift acumulado.
            # Volta ao COLOR_CHECK para reavaliar em vez de travar.
            if self.front_dist <= self.FRONT_BLOCKED:
                self.state = 'COLOR_CHECK'
                self.color_check_count   = 0
                self.color_check_samples = []
                self.lat_cmd = 0.0
                self.get_logger().warn(
                    f'[NAV] TURNING concluído mas frente ainda bloqueada '
                    f'(F={self.front_dist:.2f}m) → COLOR_CHECK novamente')
            else:
                # Saiu do bloqueio com sucesso → zera contador de becos
                self.consecutive_blocks = 0
                self.state = 'FOLLOW_CORRIDOR'
                self.get_logger().info(
                    f'[NAV] TURNING concluído → FOLLOW_CORRIDOR  '
                    f'yaw={math.degrees(self.current_yaw):.1f}°  '
                    f'erro_final={math.degrees(error):.1f}°')
        else:
            twist.linear.x  = 0.0
            twist.linear.y  = 0.0
            twist.angular.z = self.TURN_SPEED * math.copysign(1.0, error)
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
