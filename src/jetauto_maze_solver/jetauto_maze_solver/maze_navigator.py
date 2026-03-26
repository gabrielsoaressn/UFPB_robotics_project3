#!/usr/bin/env python3
"""
Maze Navigator — Exploração de Fronteiras com Desvio Reativo de Obstáculos

Estratégia:
  1. Detecta fronteiras no mapa SLAM (células livres adjacentes a células desconhecidas).
  2. Agrupa fronteiras próximas e seleciona o centroide mais distante da origem.
  3. Navega em direção à fronteira usando controle proporcional de heading.
  4. Evita paredes reativamente via LiDAR (frente, esquerda, direita).
  5. Atualiza a fronteira-alvo periodicamente ou ao chegar perto do alvo.

Estados:
  INITIAL_STRAIGHT → Largada segura até o mapa SLAM estar disponível
  SEEK_FRONTIER    → Navega em direção à fronteira mais distante da origem
  TURNING          → Giro controlado por odometria (usado quando frente bloqueada)

Tópicos:
  Assina: /jetauto/lidar/scan, /odometry/filtered, /map
  Publica: /jetauto/cmd_vel
"""

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
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
    FRONT_LO = math.radians(-20)
    FRONT_HI = math.radians(20)
    RIGHT_LO  = math.radians(-100)
    RIGHT_HI  = math.radians(-50)
    LEFT_LO   = math.radians(50)
    LEFT_HI   = math.radians(100)

    # ── Limiares de distância (metros) ─────────────────────────────
    FRONT_BLOCKED = 0.55    # frente bloqueada → girar
    WALL_DANGER   = 0.30    # distância mínima lateral antes de desviar

    # ── Velocidades ────────────────────────────────────────────────
    FORWARD_SPEED = 0.15    # m/s
    TURN_SPEED    = 0.5     # rad/s durante giro no estado TURNING
    KP_HEADING       = 1.0   # ganho proporcional para heading em direção à fronteira
    KP_AVOID         = 1.5   # ganho para desvio lateral de paredes
    AZ_CLAMP         = 0.55  # limite total de velocidade angular
    HEADING_DEADBAND = 0.08  # radianos (~4.6°): abaixo disso az_heading = 0 (cruise)

    # ── TURNING: tolerância angular ────────────────────────────────
    YAW_TOLERANCE = 0.06    # radianos (~3.4°)

    # ── Processamento do LiDAR ─────────────────────────────────────
    RANGE_CAP       = 3.0
    LIDAR_MIN_VALID = 0.15

    # ── Exploração de Fronteiras ───────────────────────────────────
    FRONTIER_UPDATE_CYCLES   = 40   # atualiza fronteira a cada 40 ciclos (4 s a 10 Hz)
    FRONTIER_CLUSTER_SIZE    = 0.8  # raio de agrupamento de fronteiras (m)
    FRONTIER_MIN_CLUSTER     = 3    # pontos mínimos por cluster para ser válido
    FRONTIER_ARRIVAL_DIST    = 0.5  # considera chegado quando a menos de 0.5 m do alvo
    FRONTIER_MIN_DIST        = 1.2  # ignora fronteiras a menos de 1.2 m do robô
    MAP_SCAN_STEP            = 3    # passo de varredura do mapa (células) — performance

    def __init__(self):
        super().__init__('maze_navigator')

        # ── Subscribers ──
        self.create_subscription(
            LaserScan, '/jetauto/lidar/scan', self._scan_cb, 10)
        self.create_subscription(
            Odometry, '/odometry/filtered', self._odom_cb, 10)
        self.create_subscription(
            OccupancyGrid, '/map', self._map_cb, 10)

        # ── Publisher ──
        self.cmd_pub = self.create_publisher(Twist, '/jetauto/cmd_vel', 10)

        # ── Timer de controle (10 Hz) ──
        self.timer = self.create_timer(0.1, self._control_loop)

        # ── Dados do LiDAR ──
        self.front_dist = self.RANGE_CAP
        self.right_dist = self.RANGE_CAP
        self.left_dist  = self.RANGE_CAP
        self.scan_ready = False

        # ── Dados da odometria ──
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.current_yaw = 0.0
        self.odom_ready = False

        # ── Mapa SLAM ──
        self.slam_map = None

        # ── Máquina de Estados ──
        self.state = 'INITIAL_STRAIGHT'
        self.target_yaw   = 0.0
        self.turn_direction = 0.0

        # ── Fronteira atual ──
        self.frontier_goal = None          # (wx, wy) da fronteira alvo
        self.frontier_update_counter = self.FRONTIER_UPDATE_CYCLES  # força update inicial

        # Contador para throttle de logs
        self.loop_count = 0

        self.get_logger().info(
            '[NAV] ═══ Maze Navigator Iniciado ═══ Exploração de Fronteiras')

    # ══════════════════════════════════════════════════════════════════
    # Callbacks dos sensores
    # ══════════════════════════════════════════════════════════════════

    def _odom_cb(self, msg: Odometry):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        self.current_yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self.odom_ready = True

    def _map_cb(self, msg: OccupancyGrid):
        self.slam_map = msg

    def _scan_cb(self, msg: LaserScan):
        ranges = msg.ranges
        n = len(ranges)
        if n == 0:
            return

        a_min = msg.angle_min
        a_inc = msg.angle_increment

        def idx(angle_rad):
            return max(0, min(n - 1, round((angle_rad - a_min) / a_inc)))

        def sector_min(lo_rad, hi_rad):
            i_lo = idx(lo_rad)
            i_hi = idx(hi_rad)
            best = self.RANGE_CAP
            for i in range(min(i_lo, i_hi), max(i_lo, i_hi) + 1):
                r = ranges[i]
                if not math.isfinite(r) or r < self.LIDAR_MIN_VALID:
                    continue
                best = min(best, min(r, self.RANGE_CAP))
            return best

        self.front_dist = sector_min(self.FRONT_LO, self.FRONT_HI)
        self.right_dist = sector_min(self.RIGHT_LO, self.RIGHT_HI)
        self.left_dist  = sector_min(self.LEFT_LO,  self.LEFT_HI)
        self.scan_ready = True

    # ══════════════════════════════════════════════════════════════════
    # Detecção e seleção de fronteiras
    # ══════════════════════════════════════════════════════════════════

    def _find_best_frontier(self):
        """
        Varre o mapa SLAM em busca de fronteiras:
          - Célula livre (0..49) adjacente a célula desconhecida (-1)
        Agrupa fronteiras próximas e retorna o centroide do cluster
        mais distante da origem (0, 0) do mapa.
        Retorna (wx, wy) ou None se não houver fronteira.
        """
        if self.slam_map is None:
            return None

        data  = self.slam_map.data
        info  = self.slam_map.info
        w, h  = info.width, info.height
        res   = info.resolution
        ox    = info.origin.position.x
        oy    = info.origin.position.y
        step  = self.MAP_SCAN_STEP

        frontier_pts = []

        for row in range(1, h - 1, step):
            for col in range(1, w - 1, step):
                idx = row * w + col
                if not (0 <= data[idx] < 50):
                    continue  # célula não é livre

                # Verifica se algum vizinho de 4-conectividade é desconhecido
                has_unknown = (
                    data[(row - 1) * w + col] == -1 or
                    data[(row + 1) * w + col] == -1 or
                    data[row * w + (col - 1)] == -1 or
                    data[row * w + (col + 1)] == -1
                )
                if has_unknown:
                    wx = ox + (col + 0.5) * res
                    wy = oy + (row + 0.5) * res
                    frontier_pts.append((wx, wy))

        if not frontier_pts:
            return None

        # ── Agrupamento por grade (grid clustering) ─────────────────
        cs = self.FRONTIER_CLUSTER_SIZE
        cluster_map: dict = {}
        for wx, wy in frontier_pts:
            key = (int(wx / cs), int(wy / cs))
            cluster_map.setdefault(key, []).append((wx, wy))

        # Centroides dos clusters com pontos suficientes
        centroids = []
        for pts in cluster_map.values():
            if len(pts) >= self.FRONTIER_MIN_CLUSTER:
                cx = sum(p[0] for p in pts) / len(pts)
                cy = sum(p[1] for p in pts) / len(pts)
                centroids.append((cx, cy))

        if not centroids:
            return None

        # ── Filtra fronteiras muito próximas do robô ─────────────────
        # Evita selecionar fronteiras que o robô já está em cima,
        # o que causaria o loop "atingida → busca → mesma fronteira".
        far = [c for c in centroids
               if math.hypot(c[0] - self.odom_x,
                              c[1] - self.odom_y) > self.FRONTIER_MIN_DIST]

        # Se não há fronteiras distantes o suficiente, usa todas como fallback
        candidates = far if far else centroids

        # ── Seleciona o centroide mais distante da origem (0, 0) ────
        best = max(candidates, key=lambda c: math.hypot(c[0], c[1]))
        return best

    # ══════════════════════════════════════════════════════════════════
    # Loop de Controle Principal (10 Hz)
    # ══════════════════════════════════════════════════════════════════

    def _control_loop(self):
        if not self.scan_ready or not self.odom_ready:
            return

        self.loop_count += 1
        F = self.front_dist
        R = self.right_dist
        L = self.left_dist

        twist = Twist()
        twist.linear.y = 0.0

        if self.state == 'INITIAL_STRAIGHT':
            self._state_initial_straight(twist, F)
        elif self.state == 'TURNING':
            self._state_turning(twist)
        else:
            self._state_seek_frontier(twist, F, R, L)

        self.cmd_pub.publish(twist)

    # ──────────────────────────────────────────────────────────────────
    # ESTADO: INITIAL_STRAIGHT
    # ──────────────────────────────────────────────────────────────────

    def _state_initial_straight(self, twist: Twist, F: float):
        """
        Vai reto até o mapa SLAM estar disponível e ter fronteiras.
        Assim que a primeira fronteira for encontrada, vai para SEEK_FRONTIER.
        """
        if self.slam_map is not None:
            frontier = self._find_best_frontier()
            if frontier is not None:
                self.frontier_goal = frontier
                self.frontier_update_counter = 0
                self.state = 'SEEK_FRONTIER'
                self.get_logger().info(
                    f'[NAV] Mapa disponível → SEEK_FRONTIER  '
                    f'primeiro alvo=({frontier[0]:.2f}, {frontier[1]:.2f})')
                return

        if F <= self.FRONT_BLOCKED:
            # Bloqueado antes do mapa chegar — gira à esquerda
            self.target_yaw = normalize_angle(self.current_yaw + math.pi / 2.0)
            self.turn_direction = 1.0
            self.state = 'TURNING'
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            return

        twist.linear.x = self.FORWARD_SPEED
        twist.angular.z = 0.0

        if self.loop_count % 20 == 0:
            self.get_logger().info(
                f'[NAV] INITIAL_STRAIGHT — aguardando mapa SLAM  F={F:.2f}m')

    # ──────────────────────────────────────────────────────────────────
    # ESTADO: SEEK_FRONTIER
    # ──────────────────────────────────────────────────────────────────

    def _state_seek_frontier(self, twist: Twist, F: float, R: float, L: float):
        """
        Navega em direção à fronteira mais distante da origem.

        Atualiza o alvo periodicamente (FRONTIER_UPDATE_CYCLES ciclos).
        Controle de movimento:
          - Frente bloqueada → gira para o lado com mais espaço (preferindo
            a direção mais próxima do heading para o alvo)
          - Livre → heading proporcional ao alvo + desvio lateral de paredes
        """

        # ── Atualiza fronteira periodicamente ──
        self.frontier_update_counter += 1
        if self.frontier_update_counter >= self.FRONTIER_UPDATE_CYCLES:
            self.frontier_update_counter = 0
            new_goal = self._find_best_frontier()
            if new_goal is not None:
                old = self.frontier_goal
                self.frontier_goal = new_goal
                if old is None or math.hypot(
                        new_goal[0] - old[0], new_goal[1] - old[1]) > 0.5:
                    self.get_logger().info(
                        f'[NAV] Fronteira atualizada → '
                        f'({new_goal[0]:.2f}, {new_goal[1]:.2f})  '
                        f'dist_origem={math.hypot(*new_goal):.2f}m')
            else:
                self.get_logger().info(
                    '[NAV] Nenhuma fronteira encontrada — labirinto explorado?')

        # ── Sem alvo: para o robô ──
        if self.frontier_goal is None:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            return

        gx, gy = self.frontier_goal
        dx = gx - self.odom_x
        dy = gy - self.odom_y
        dist_to_goal = math.hypot(dx, dy)

        # ── Chegou na fronteira → força nova busca ──
        if dist_to_goal < self.FRONTIER_ARRIVAL_DIST:
            self.frontier_goal = None
            self.frontier_update_counter = self.FRONTIER_UPDATE_CYCLES
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.get_logger().info(
                f'[NAV] Fronteira atingida! Buscando próxima...')
            return

        goal_yaw       = math.atan2(dy, dx)
        heading_error  = normalize_angle(goal_yaw - self.current_yaw)

        # ── Frente bloqueada → girar para o lado mais próximo do alvo ──
        if F <= self.FRONT_BLOCKED:
            if heading_error >= 0:
                direction = 1.0   # esquerda (mais próxima do alvo)
            else:
                direction = -1.0  # direita
            twist.linear.x  = 0.0
            twist.angular.z = self.TURN_SPEED * direction
            if self.loop_count % 10 == 0:
                self.get_logger().info(
                    f'[NAV] Obstáculo frontal (F={F:.2f}m) → '
                    f'girando {"esquerda" if direction > 0 else "direita"}')
            return

        # ── Livre: heading ao alvo + desvio lateral de paredes ──

        # Zona morta de heading: se o robô já está bem alinhado com o alvo
        # e não há paredes perigosas, manda angular.z = 0.0 puro (cruise),
        # evitando micro-correções que fazem o robô andar inclinado.
        if abs(heading_error) < self.HEADING_DEADBAND:
            az_heading = 0.0
        else:
            az_heading = max(-self.AZ_CLAMP,
                             min(heading_error * self.KP_HEADING, self.AZ_CLAMP))

        az_avoid = 0.0
        if R < self.WALL_DANGER:
            az_avoid += (self.WALL_DANGER - R) * self.KP_AVOID   # empurra à esq.
        if L < self.WALL_DANGER:
            az_avoid -= (self.WALL_DANGER - L) * self.KP_AVOID   # empurra à dir.

        twist.linear.x  = self.FORWARD_SPEED
        twist.angular.z = max(-self.AZ_CLAMP,
                              min(az_heading + az_avoid, self.AZ_CLAMP))

        if self.loop_count % 20 == 0:
            self.get_logger().info(
                f'[NAV] SEEK  goal=({gx:.1f},{gy:.1f})  dist={dist_to_goal:.2f}m  '
                f'heading_err={math.degrees(heading_error):.1f}°  '
                f'F={F:.2f}  R={R:.2f}  L={L:.2f}  '
                f'az={twist.angular.z:+.3f}')

    # ──────────────────────────────────────────────────────────────────
    # ESTADO: TURNING (Giro Controlado por Odometria)
    # ──────────────────────────────────────────────────────────────────

    def _state_turning(self, twist: Twist):
        """
        Usado apenas na largada (INITIAL_STRAIGHT) caso a frente bloqueie
        antes do mapa SLAM estar disponível.
        """
        error = normalize_angle(self.target_yaw - self.current_yaw)

        if abs(error) < self.YAW_TOLERANCE:
            self.state = 'INITIAL_STRAIGHT'
            twist.linear.x  = 0.0
            twist.angular.z = 0.0
            self.get_logger().info(
                f'[NAV] TURNING concluído → INITIAL_STRAIGHT  '
                f'yaw={math.degrees(self.current_yaw):.1f}°')
        else:
            twist.linear.x  = 0.0
            twist.angular.z = self.TURN_SPEED * self.turn_direction

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
