"""perception.color: cheap HSV attribute check for a detection box.

Verification (PLAN Phase 5) only needs "is the thing I am looking at the colour
the user asked for", so there is no model here. We count HSV pixels inside the
box against ranges for a handful of colours and return the dominant one if it
clears a floor. Hue ranges are disjoint; ``red`` wraps hue 0 so it has two.
"""

from __future__ import annotations

import cv2
import numpy as np

HSV = tuple[int, int, int]
COLOR_RANGES: dict[str, list[tuple[HSV, HSV]]] = {
    "red": [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (180, 255, 255))],
    "orange": [((10, 120, 70), (25, 255, 255))],
    "yellow": [((25, 120, 70), (35, 255, 255))],
    "green": [((35, 60, 50), (85, 255, 255))],
    "blue": [((85, 60, 50), (130, 255, 255))],
    "purple": [((130, 60, 50), (170, 255, 255))],
    "white": [((0, 0, 200), (180, 45, 255))],
    "gray": [((0, 0, 60), (180, 45, 200))],
    "black": [((0, 0, 0), (180, 255, 60))],
}


def clamp_box(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int]:
    """Clamp a float box to integer pixel bounds; returns a possibly-empty box."""
    x1, y1, x2, y2 = (int(round(value)) for value in box)
    x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
    y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
    return x1, y1, x2, y2


def color_fractions(
    frame: np.ndarray,
    box: tuple[float, float, float, float],
    ranges: dict[str, list[tuple[HSV, HSV]]] = COLOR_RANGES,
) -> dict[str, float]:
    """Fraction of the box's pixels inside each colour range (empty if no box)."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = clamp_box(box, width, height)
    if x2 <= x1 or y2 <= y1:
        return {}
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    total = float(hsv.shape[0] * hsv.shape[1])
    fractions: dict[str, float] = {}
    for name, bands in ranges.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in bands:
            mask |= cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
        fractions[name] = float(np.count_nonzero(mask)) / total
    return fractions


def dominant_color(
    frame: np.ndarray,
    box: tuple[float, float, float, float],
    min_fraction: float = 0.15,
    ranges: dict[str, list[tuple[HSV, HSV]]] = COLOR_RANGES,
) -> tuple[str, float] | None:
    """Return ``(colour, fraction)`` of the dominant colour, if it clears the floor."""
    fractions = color_fractions(frame, box, ranges)
    if not fractions:
        return None
    name = max(fractions, key=fractions.__getitem__)
    if fractions[name] < min_fraction:
        return None
    return name, fractions[name]


def matches_color(
    frame: np.ndarray,
    box: tuple[float, float, float, float],
    name: str,
    min_fraction: float = 0.15,
    ranges: dict[str, list[tuple[HSV, HSV]]] = COLOR_RANGES,
) -> bool:
    """True if ``name`` covers at least ``min_fraction`` of the box."""
    if name not in ranges:
        return False
    fractions = color_fractions(frame, box, {name: ranges[name]})
    return fractions.get(name, 0.0) >= min_fraction
