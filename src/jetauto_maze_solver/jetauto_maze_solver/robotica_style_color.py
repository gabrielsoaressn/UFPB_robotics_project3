"""
Color detection copied from robotics_subject/scripts/color_detector_node.py (robotica-main).

Uses HSV + morphological open + MIN_COLOR_RATIO (image fraction), not raw pixel count.
Internal labels: RED, GREEN, BLUE — map to vermelho/verde/azul at the counter.
"""

from __future__ import annotations

import cv2
import numpy as np

RGB_TO_PT = {'RED': 'vermelho', 'GREEN': 'verde', 'BLUE': 'azul'}


def detect_dominant_color_robotica(bgr: np.ndarray, min_color_ratio: float):
    """
    Returns (color_key | None, masks dict keyed RED/GREEN/BLUE, centroid_x normalized).
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    total = bgr.shape[0] * bgr.shape[1]

    m_red1 = cv2.inRange(hsv, np.array([0, 120, 80]), np.array([10, 255, 255]))
    m_red2 = cv2.inRange(hsv, np.array([168, 120, 80]), np.array([180, 255, 255]))
    m_red = cv2.bitwise_or(m_red1, m_red2)
    m_green = cv2.inRange(hsv, np.array([40, 90, 80]), np.array([85, 255, 255]))
    m_blue = cv2.inRange(hsv, np.array([100, 90, 80]), np.array([135, 255, 255]))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m_red = cv2.morphologyEx(m_red, cv2.MORPH_OPEN, kernel)
    m_green = cv2.morphologyEx(m_green, cv2.MORPH_OPEN, kernel)
    m_blue = cv2.morphologyEx(m_blue, cv2.MORPH_OPEN, kernel)

    ratios = {
        'RED': cv2.countNonZero(m_red) / total,
        'GREEN': cv2.countNonZero(m_green) / total,
        'BLUE': cv2.countNonZero(m_blue) / total,
    }
    masks = {'RED': m_red, 'GREEN': m_green, 'BLUE': m_blue}

    best = max(ratios, key=ratios.get)
    if ratios[best] < min_color_ratio:
        return None, masks, None

    M = cv2.moments(masks[best])
    if M['m00'] > 0:
        cx_px = M['m10'] / M['m00']
    else:
        cx_px = bgr.shape[1] / 2.0
    centroid_x = (cx_px - bgr.shape[1] / 2.0) / bgr.shape[1]
    return best, masks, centroid_x
