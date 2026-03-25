#!/usr/bin/env python3
"""
Sistema de Visão — Detecção e Contagem de Paredes Coloridas (Isolado e Passivo)

Detecta paredes Azuis, Verdes, Vermelhas e Amarelas usando câmera RGB.
Converte para HSV, aplica máscaras de cor e busca contornos com área mínima.
Cooldown temporal de 8 segundos por cor para evitar contagem duplicada.

*** Este nó NUNCA publica comandos de velocidade. ***

Tópico assinado: /jetauto/camera/image_raw
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class ColorWallCounter(Node):

    # ── Área mínima do contorno (fração da imagem) ─────────────────
    # Só conta se o maior contorno da cor ocupar pelo menos 3% da imagem,
    # indicando que a parede está próxima / na frente do robô.
    MIN_AREA_FRACTION = 0.03

    # ── Cooldown temporal (segundos) ───────────────────────────────
    # Após detectar uma cor, bloqueia nova contagem dessa MESMA cor
    # por 8 segundos, impedindo que a mesma parede seja contada
    # múltiplas vezes enquanto o robô passa por ela.
    COOLDOWN_SECS = 8.0

    # ── Faixas HSV para cada cor ───────────────────────────────────
    # Gazebo renderiza cores bem saturadas, então as faixas podem ser
    # razoavelmente estreitas. Cada cor pode ter múltiplas faixas
    # (vermelho precisa de duas por causa do wrap-around do Hue).
    COLOR_RANGES = {
        'azul': [
            (np.array([100, 80, 50]), np.array([130, 255, 255])),
        ],
        'verde': [
            (np.array([40, 80, 50]), np.array([85, 255, 255])),
        ],
        'vermelho': [
            # Vermelho faz wrap-around no canal Hue (0 e 180)
            (np.array([0, 80, 50]),   np.array([10, 255, 255])),
            (np.array([170, 80, 50]), np.array([180, 255, 255])),
        ],
        'amarelo': [
            (np.array([20, 80, 50]), np.array([40, 255, 255])),
        ],
    }

    # Letras abreviadas para o placar
    COLOR_LABELS = {
        'azul': 'A',
        'verde': 'V',
        'vermelho': 'R',
        'amarelo': 'Y',
    }

    def __init__(self):
        super().__init__('color_wall_counter')

        self.bridge = CvBridge()

        # ── Subscriber: apenas câmera RGB (NUNCA publica cmd_vel) ──
        self.create_subscription(
            Image, '/jetauto/camera/image_raw', self._image_cb, 10)

        # ── Contagens por cor ──
        self.counts = {color: 0 for color in self.COLOR_RANGES}

        # ── Timestamp da última detecção de cada cor (para cooldown) ──
        self.last_detection_time = {color: None for color in self.COLOR_RANGES}

        self.get_logger().info(
            '[VISÃO] Sistema de Visão iniciado — detectando paredes coloridas')
        self.get_logger().info(
            f'[VISÃO] Cooldown: {self.COOLDOWN_SECS}s  '
            f'Área mínima: {self.MIN_AREA_FRACTION * 100:.0f}% da imagem')

    # ══════════════════════════════════════════════════════════════════
    # Callback da câmera
    # ══════════════════════════════════════════════════════════════════

    def _image_cb(self, msg: Image):
        """Processa cada frame: converte para HSV, busca contornos coloridos."""

        # Converte ROS Image → OpenCV BGR → HSV
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
            # Constrói máscara combinada (união de todas as faixas da cor)
            mask = np.zeros((h, w), dtype=np.uint8)
            for lower, upper in ranges:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

            # Encontra contornos externos na máscara
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            # Seleciona o maior contorno e verifica área mínima
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area < min_area:
                continue

            # Verifica cooldown temporal (8 segundos)
            if not self._cooldown_ok(color, now):
                continue

            # ── Nova parede detectada! ──
            self.counts[color] += 1
            self.last_detection_time[color] = now

            placar = self._format_placar()
            self.get_logger().info(
                f'[VISÃO] Parede {color.upper()} detectada! Placar: {placar}')

    # ══════════════════════════════════════════════════════════════════
    # Auxiliares
    # ══════════════════════════════════════════════════════════════════

    def _cooldown_ok(self, color: str, now) -> bool:
        """Retorna True se o cooldown da cor já expirou (>= 8 segundos)."""
        last = self.last_detection_time[color]
        if last is None:
            return True  # Nunca detectou esta cor ainda
        elapsed = (now - last).nanoseconds / 1e9
        return elapsed >= self.COOLDOWN_SECS

    def _format_placar(self) -> str:
        """Formata o placar no formato: A:0, V:0, R:0, Y:0"""
        parts = []
        for color in self.COLOR_RANGES:
            label = self.COLOR_LABELS[color]
            count = self.counts[color]
            parts.append(f'{label}:{count}')
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
