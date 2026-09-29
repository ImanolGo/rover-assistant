"""Laptop-only tests for perception.geometry: calibration, bearings, remap."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rover.perception.geometry import CameraGeometry, load_calibration, scale_intrinsics

CALIBRATION = Path(__file__).resolve().parents[1] / "config" / "camera_calibration.yaml"


def _pinhole(fx: float = 100.0, fy: float = 100.0, cx: float = 50.0, cy: float = 50.0):
    matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return matrix, np.zeros(5, dtype=np.float32), (100, 100)


def test_load_real_calibration():
    matrix, coeffs, size = load_calibration(CALIBRATION)
    assert matrix.shape == (3, 3)
    assert coeffs.shape == (5,)
    assert size == (1640, 1232)
    assert matrix[0, 0] == pytest.approx(786.24, abs=0.1)
    assert matrix[0, 2] == pytest.approx(791.76, abs=0.1)


def test_scale_intrinsics_halves_center_and_focal():
    matrix, _, _ = _pinhole()
    scaled = scale_intrinsics(matrix, 0.5)
    assert scaled[0, 0] == pytest.approx(50.0)
    assert scaled[0, 2] == pytest.approx(25.0)
    assert scaled[1, 2] == pytest.approx(25.0)


def test_bearing_exact_for_zero_distortion():
    matrix, coeffs, size = _pinhole()
    geometry = CameraGeometry(matrix, coeffs, size, output_size=size)
    assert geometry.bearing_deg(50, 50) == pytest.approx(0.0, abs=1e-4)
    # u = cx + fx -> the ray at 45 degrees
    assert geometry.bearing_deg(150, 50) == pytest.approx(45.0, abs=1e-3)
    assert geometry.bearing_deg(-50, 50) == pytest.approx(-45.0, abs=1e-3)


def test_elevation_positive_is_down():
    matrix, coeffs, size = _pinhole()
    geometry = CameraGeometry(matrix, coeffs, size, output_size=size)
    assert geometry.elevation_deg(50, 150) == pytest.approx(45.0, abs=1e-3)


def test_half_resolution_bearings_match_full_resolution():
    matrix, coeffs, size = _pinhole()
    full = CameraGeometry(matrix, coeffs, size, output_size=size)
    half = CameraGeometry(matrix, coeffs, size, output_size=(50, 50))
    # The same physical ray appears at (u,v) full-res and (u/2, v/2) half-res.
    assert full.bearing_deg(120, 40) == pytest.approx(half.bearing_deg(60, 20), abs=1e-3)


def test_real_calibration_center_and_sign():
    geometry = CameraGeometry.from_yaml(CALIBRATION, output_size=(820, 616))
    assert geometry.bearing_deg(geometry.cx, geometry.cy) == pytest.approx(0.0, abs=1e-3)
    assert geometry.bearing_deg(geometry.cx + 150, geometry.cy) > 0
    assert geometry.bearing_deg(geometry.cx - 150, geometry.cy) < 0


def test_fov_h_degrees():
    matrix, coeffs, size = _pinhole(fx=50.0)
    geometry = CameraGeometry(matrix, coeffs, size, output_size=size)
    # 2*atan2(50, 50) = 90 degrees
    assert geometry.fov_h_deg() == pytest.approx(90.0, abs=1e-3)


def test_undistort_preserves_shape_and_caches_maps():
    geometry = CameraGeometry.from_yaml(CALIBRATION, output_size=(820, 616))
    frame = np.zeros((616, 820, 3), dtype=np.uint8)
    frame[300, 400] = 255
    out = geometry.undistort(frame)
    assert out.shape == frame.shape
    assert out.dtype == np.uint8
    assert (820, 616) in geometry._maps  # maps cached
    assert geometry.undistort(frame).shape == frame.shape  # second call reuses the maps
