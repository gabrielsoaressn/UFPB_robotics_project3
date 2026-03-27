"""
Shared HSV color detection for maze_navigator and color_wall_counter.

Matches maze_navigator.py camera logic: same thresholds, same winner rule
(max pixel count; red before green before blue on ties), no morphology.
"""

from __future__ import annotations

import numpy as np
import cv2

# Same as MazeNavigator.COLOR_MIN_AREA_FRAC
COLOR_MIN_AREA_FRAC = 0.05

RED_LOWER1 = np.array([0, 80, 50])
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([170, 80, 50])
RED_UPPER2 = np.array([180, 255, 255])
GREEN_LOWER = np.array([40, 80, 50])
GREEN_UPPER = np.array([85, 255, 255])
BLUE_LOWER = np.array([100, 80, 50])
BLUE_UPPER = np.array([130, 255, 255])

COLOR_ORDER = ('vermelho', 'verde', 'azul')


def build_color_masks(hsv: np.ndarray) -> dict[str, np.ndarray]:
    mask_r = cv2.bitwise_or(
        cv2.inRange(hsv, RED_LOWER1, RED_UPPER1),
        cv2.inRange(hsv, RED_LOWER2, RED_UPPER2),
    )
    mask_g = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    mask_b = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
    return {'vermelho': mask_r, 'verde': mask_g, 'azul': mask_b}


def dominant_wall_color_from_bgr(
    bgr: np.ndarray,
    min_area_fraction: float | None = None,
):
    """
    Returns (color_name | None, masks dict, centroid_x | None).
    centroid_x is normalized like color_detector: (cx - w/2) / w in [-0.5, 0.5].
    """
    frac = COLOR_MIN_AREA_FRAC if min_area_fraction is None else min_area_fraction
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    min_area = frac * h * w

    masks = build_color_masks(hsv)
    red_area = cv2.countNonZero(masks['vermelho'])
    green_area = cv2.countNonZero(masks['verde'])
    blue_area = cv2.countNonZero(masks['azul'])

    best_area = max(red_area, green_area, blue_area)
    if best_area < min_area:
        return None, masks, None

    if best_area == red_area:
        name = 'vermelho'
    elif best_area == green_area:
        name = 'verde'
    else:
        name = 'azul'

    mask = masks[name]
    M = cv2.moments(mask)
    if M['m00'] <= 0:
        centroid_x = 0.0
    else:
        cx_px = M['m10'] / M['m00']
        centroid_x = (cx_px - bgr.shape[1] / 2.0) / bgr.shape[1]

    return name, masks, centroid_x
