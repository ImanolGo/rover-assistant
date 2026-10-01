"""Laptop-only tests for perception.color: HSV attribute check."""

from __future__ import annotations

import numpy as np
import pytest

from rover.perception.color import clamp_box, color_fractions, dominant_color, matches_color


def _solid(bgr: tuple[int, int, int], size: int = 40) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:] = bgr
    return image


@pytest.mark.parametrize(
    ("bgr", "name"),
    [
        ((0, 0, 255), "red"),
        ((0, 255, 0), "green"),
        ((255, 0, 0), "blue"),
        ((0, 255, 255), "yellow"),
        ((0, 165, 255), "orange"),
        ((128, 0, 128), "purple"),
        ((255, 255, 255), "white"),
        ((128, 128, 128), "gray"),
        ((0, 0, 0), "black"),
    ],
)
def test_dominant_color_on_solid_images(bgr, name):
    result = dominant_color(_solid(bgr), (0, 0, 40, 40))
    assert result is not None
    assert result[0] == name


def test_box_isolates_the_region():
    image = _solid((255, 0, 0))  # blue frame
    image[10:30, 10:30] = (0, 0, 255)  # red square
    assert dominant_color(image, (10, 10, 30, 30))[0] == "red"
    assert dominant_color(image, (0, 0, 10, 10))[0] == "blue"


def test_degenerate_box_is_empty():
    assert dominant_color(_solid((0, 0, 255)), (5, 5, 5, 5)) is None
    assert color_fractions(_solid((0, 0, 255)), (5, 5, 5, 5)) == {}


def test_matches_color():
    image = _solid((0, 0, 255))
    assert matches_color(image, (0, 0, 40, 40), "red")
    assert not matches_color(image, (0, 0, 40, 40), "blue")
    assert not matches_color(image, (0, 0, 40, 40), "chartreuse")


def test_clamp_box_to_frame():
    assert clamp_box((-5, -5, 100, 100), 40, 40) == (0, 0, 40, 40)
    assert clamp_box((40, 40, 10, 10), 40, 40) == (10, 10, 40, 40)
