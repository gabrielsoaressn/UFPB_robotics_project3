#!/usr/bin/env python3
"""
Maze Navigator — Centro do Corredor com Memória Ativa e Visão (Robô Omnidirecional)

Lógica:
- Avança mantendo o centro do corredor via controle proporcional lateral (linear.y) e heading.
- Ao bloquear a frente, para por 0.5s para amostrar a cor da parede.
- Gira 90° baseado na cor. Se não houver cor, cruza a viabilidade do LiDAR (onde tem espaço livre)
  com a grade de memória, escolhendo a direção menos visitada.
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
    while angle > math.pi: angle -= 2.0 * math.pi
    while angle < -math.pi: angle += 2.0 * math.pi
    return angle

# ═══════════════════════════════════════════════════════════════════════
# Nó principal
# ═══════════════════════════════════════════════════════════════════════

class MazeNavigator(Node):

    # Setores do LiDAR
    FRONT_LO, FRONT_HI = math.radians(-20), math.radians(20)
    LEFT_LO,  LEFT_HI  = math.radians(70),  math.radians(110)
    RIGHT_LO, RIGHT_HI = math.radians(-110), math.radians(-70)

    # Raios individuais para cálculo de heading
    R_SIDE_ANGLE, R_DIAG_ANGLE = math.radians(-90), math.radians(-45)
    L_SIDE_ANGLE, L_DIAG_ANGLE = math.radians(90), math.radians(45)

    # Limiares de distância
    FRONT_BLOCKED = 0.60    # Espaço seguro para o chassi girar sem colisões
    WALL_DETECT   = 1.2     
    TARGET_SIDE   = 0.40    

    # Controle lateral — linear.y (Suavizado)
    KP_LAT    = 0.15
    LAT_CLAMP = 0.10
    LAT_ALPHA = 0.25

    # Correção de heading — angular.z
    ALIGN_KP    = 1.5
    ALIGN_CLAMP = 0.15

    # Velocidades
    FORWARD_SPEED = 0.15
    ARC_SPEED     = 0.08     # Velocidade linear durante o arco de curva
    TURN_SPEED    = 0.4
    TURN_KP       = 2.5      # Ganho proporcional do giro (suaviza a chegada ao alvo)
    TURN_MIN      = 0.10     # Velocidade mínima para não travar perto do alvo
    YAW_TOLERANCE = 0.05

    RANGE_CAP       = 3.0
    LIDAR_MIN_VALID = 0.15

    # COLOR_CHECK
    COLOR_CHECK_CYCLES = 5

    # Anti-backtracking
    CELL_SIZE       = 0.8   # Aumentado para tolerar drift de odometria a longo prazo
    BACKTRACK_DISTS = (1.0, 1.5, 2.0)

    # Câmera — faixas HSV
    COLOR_MIN_AREA_FRAC = 0.05
    RED_LOWER1, RED_UPPER1 = np.array([0, 80, 50]), np.array([10, 255, 255])
    RED_LOWER2, RED_UPPER2 = np.array([170, 80, 50]), np.array([180, 255, 255])
    GREEN_LOWER, GREEN_UPPER = np.array([40, 80, 50]), np.array([85, 255, 255])

    def __init__(self):
        super().__init__('maze_navigator')

        self.create_subscription(LaserScan, '/jetauto/lidar/scan', self._scan_cb, 10)
        self.create_subscription(Odometry, '/odometry/filtered', self._odom_cb, 10)
        self.create_subscription(Image, '/jetauto/camera/image_raw', self._image_cb, 10)

        self.cmd_pub = self.create_publisher(Twist, '/jetauto/cmd_vel', 10)
        self.bridge = CvBridge()
        self.timer = self.create_timer(0.1, self._control_loop)

        self.front_dist, self.left_dist, self.right_dist = self.RANGE_CAP, self.RANGE_CAP, self.RANGE_CAP
        self.r_side, self.r_diag = self.RANGE_CAP, self.RANGE_CAP
        self.l_side, self.l_diag = self.RANGE_CAP, self.RANGE_CAP
        self.scan_ready, self.odom_ready = False, False

        self.odom_x, self.odom_y, self.current_yaw = 0.0, 0.0, 0.0

        self.detected_color = None
        self.lat_cmd = 0.0
        self.visited_cells: set = set()

        self.state = 'FOLLOW_CORRIDOR'
        self.target_yaw = 0.0
        self.turn_direction = 0.0
        self.color_check_count = 0
        self.color_check_samples = []
        self.loop_count = 0

        self.get_logger().info('[NAV] Maze Navigator Híbrido: Centralização OMNI + Memória')

    def _odom_cb(self, msg: Odometry):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        self.current_yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self.odom_ready = True

    def _scan_cb(self, msg: LaserScan):
        ranges = msg.ranges
        if not ranges: return

        def idx(a): return max(0, min(len(ranges) - 1, round((a - msg.angle_min) / msg.angle_increment)))
        def sector_min(lo, hi):
            best = self.RANGE_CAP
            for i in range(min(idx(lo), idx(hi)), max(idx(lo), idx(hi)) + 1):
                r = ranges[i]
                if math.isfinite(r) and r >= self.LIDAR_MIN_VALID: best = min(best, min(r, self.RANGE_CAP))
            return best
        def ray_at(a):
            r = ranges[idx(a)]
            return min(r, self.RANGE_CAP) if math.isfinite(r) and r >= self.LIDAR_MIN_VALID else self.RANGE_CAP

        self.front_dist = sector_min(self.FRONT_LO, self.FRONT_HI)
        self.left_dist  = sector_min(self.LEFT_LO,  self.LEFT_HI)
        self.right_dist = sector_min(self.RIGHT_LO, self.RIGHT_HI)
        self.r_side, self.r_diag = ray_at(self.R_SIDE_ANGLE), ray_at(self.R_DIAG_ANGLE)
        self.l_side, self.l_diag = ray_at(self.L_SIDE_ANGLE), ray_at(self.L_DIAG_ANGLE)
        self.scan_ready = True

    def _image_cb(self, msg: Image):
        try: cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except: return

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        min_area = self.COLOR_MIN_AREA_FRAC * hsv.shape[0] * hsv.shape[1]

        mask_r = cv2.bitwise_or(cv2.inRange(hsv, self.RED_LOWER1, self.RED_UPPER1), cv2.inRange(hsv, self.RED_LOWER2, self.RED_UPPER2))
        red_area = cv2.countNonZero(mask_r)
        green_area = cv2.countNonZero(cv2.inRange(hsv, self.GREEN_LOWER, self.GREEN_UPPER))

        if red_area >= min_area and red_area >= green_area: self.detected_color = 'vermelho'
        elif green_area >= min_area and green_area > red_area: self.detected_color = 'verde'
        else: self.detected_color = None

    def _pos_to_cell(self, x: float, y: float):
        return (math.floor(x / self.CELL_SIZE), math.floor(y / self.CELL_SIZE))

    def _visited_score(self, new_yaw: float) -> int:
        score = 0
        for dist in self.BACKTRACK_DISTS:
            px, py = self.odom_x + dist * math.cos(new_yaw), self.odom_y + dist * math.sin(new_yaw)
            if self._pos_to_cell(px, py) in self.visited_cells: score += 1
        return score

    def _choose_turn(self, color_decision: str):
        yaw_L = normalize_angle(self.current_yaw + math.pi / 2.0)
        yaw_R = normalize_angle(self.current_yaw - math.pi / 2.0)

        # 1. Hardware Override: Cores ditam a regra
        if color_decision == 'vermelho': return yaw_L, 1.0, 'esquerda (parede VERMELHA)'
        if color_decision == 'verde': return yaw_R, -1.0, 'direita (parede VERDE)'

        # 2. Avaliação de Viabilidade (Não virar para paredes sólidas)
        can_go_left  = self.left_dist > 0.8
        can_go_right = self.right_dist > 0.8

        score_L, score_R = self._visited_score(yaw_L), self._visited_score(yaw_R)

        # 3. Decisão baseada em Memória + Viabilidade
        if can_go_left and can_go_right:
            if score_L <= score_R: return yaw_L, 1.0, f'esquerda (L={score_L} vs R={score_R})'
            else: return yaw_R, -1.0, f'direita (L={score_L} vs R={score_R})'
        elif can_go_left:
            return yaw_L, 1.0, 'esquerda (forçada, direita bloqueada)'
        elif can_go_right:
            return yaw_R, -1.0, 'direita (forçada, esquerda bloqueada)'
        else:
            # Beco sem saída: vira 90°, na próxima iteração vira mais 90° (Meia-volta)
            return yaw_L, 1.0, 'meia-volta parcial (beco sem saída)'

    def _lateral_correction(self) -> float:
        L, R = self.left_dist, self.right_dist
        left_wall, right_wall = L < self.WALL_DETECT, R < self.WALL_DETECT

        if left_wall and right_wall: error = (L - R) / 2.0
        elif right_wall: error = self.TARGET_SIDE - R
        elif left_wall: error = L - self.TARGET_SIDE
        else: error = 0.0

        return max(-self.LAT_CLAMP, min(self.KP_LAT * error, self.LAT_CLAMP))

    def _get_wall_alignment_error(self):
        """Calcula o erro angular real em relação às paredes usando o LiDAR."""
        corrections = []

        a, b = self.r_side, self.r_diag
        if a < self.WALL_DETECT and b < self.RANGE_CAP - 0.1:
            corrections.append(math.atan2(a - b * 0.7071, b * 0.7071))

        c, d = self.l_side, self.l_diag
        if c < self.WALL_DETECT and d < self.RANGE_CAP - 0.1:
            corrections.append(math.atan2(d * 0.7071 - c, d * 0.7071))

        if not corrections:
            return None
            
        return sum(corrections) / len(corrections)

    def _heading_correction(self) -> float:
        """Usa o erro de alinhamento para gerar o comando angular.z no corredor."""
        wall_error = self._get_wall_alignment_error()
        if wall_error is None:
            return 0.0
            
        az = self.ALIGN_KP * wall_error
        return max(-self.ALIGN_CLAMP, min(az, self.ALIGN_CLAMP))

    def _control_loop(self):
        if not self.scan_ready or not self.odom_ready: return

        self.loop_count += 1
        self.visited_cells.add(self._pos_to_cell(self.odom_x, self.odom_y))
        twist = Twist()

        if self.state == 'COLOR_CHECK': self._state_color_check(twist)
        elif self.state == 'TURNING': self._state_turning(twist)
        else: self._state_follow_corridor(twist)

        self.cmd_pub.publish(twist)

    def _state_follow_corridor(self, twist: Twist):
        if self.front_dist <= self.FRONT_BLOCKED:
            self.state = 'COLOR_CHECK'
            self.color_check_count, self.color_check_samples = 0, []
            self.lat_cmd = 0.0
            twist.linear.x = twist.linear.y = twist.angular.z = 0.0
            return

        self.lat_cmd = (1.0 - self.LAT_ALPHA) * self.lat_cmd + self.LAT_ALPHA * self._lateral_correction()
        twist.linear.x, twist.linear.y, twist.angular.z = self.FORWARD_SPEED, self.lat_cmd, self._heading_correction()

    def _state_color_check(self, twist: Twist):
        if self.front_dist > self.FRONT_BLOCKED + 0.1:
            self.state = 'FOLLOW_CORRIDOR'
            twist.linear.x = twist.linear.y = twist.angular.z = 0.0
            return

        twist.linear.x = twist.linear.y = twist.angular.z = 0.0
        self.color_check_samples.append(self.detected_color)
        self.color_check_count += 1

        if self.color_check_count >= self.COLOR_CHECK_CYCLES:
            counts = Counter(self.color_check_samples)
            color_decision = counts.most_common(1)[0][0]
            self.target_yaw, self.turn_direction, desc = self._choose_turn(color_decision)
            self.state = 'TURNING'
            self.get_logger().info(f'[NAV] Decisão: {desc}')

    def _state_turning(self, twist: Twist):
        """Gira pela odometria até atingir target_yaw."""
        odom_error = normalize_angle(self.target_yaw - self.current_yaw)

        if abs(odom_error) < self.YAW_TOLERANCE:
            self.state = 'FOLLOW_CORRIDOR'
            self.lat_cmd = 0.0
            twist.linear.x = twist.linear.y = twist.angular.z = 0.0
            self.get_logger().info(
                f'[NAV] Giro finalizado. Erro: {math.degrees(odom_error):.1f}°')
        else:
            raw = self.TURN_KP * odom_error
            clamped = max(-self.TURN_SPEED, min(raw, self.TURN_SPEED))
            if abs(clamped) < self.TURN_MIN:
                clamped = math.copysign(self.TURN_MIN, odom_error)
            twist.angular.z = clamped
            twist.linear.x = self.ARC_SPEED   # avança durante a curva, fazendo arco
            twist.linear.y = 0.0

    def destroy_node(self):
        self.cmd_pub.publish(Twist())
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    try: rclpy.spin(MazeNavigator())
    except KeyboardInterrupt: pass
    finally: rclpy.shutdown()

if __name__ == '__main__':
    main()