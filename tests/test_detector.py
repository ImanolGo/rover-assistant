"""Laptop-only tests for perception.detector (no torch/ultralytics needed)."""

from __future__ import annotations

import numpy as np
import pytest

from rover.perception.detector import Detection, FakeDetector, resolve_weights
from rover.perception.geometry import CameraGeometry


def _geometry() -> CameraGeometry:
    matrix = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return CameraGeometry(matrix, np.zeros(5, dtype=np.float32), (100, 100))


def test_from_box_computes_bearing_and_height_fraction():
    detection = Detection.from_box(
        "cup", 0.9, (40, 20, 60, 60), geometry=_geometry(), frame_size=(100, 100), track_id=7
    )
    assert detection.center == (50.0, 40.0)
    assert detection.bearing_deg == pytest.approx(0.0, abs=1e-3)
    assert detection.h_frac == pytest.approx(0.40)
    assert detection.track_id == 7
    assert detection.box == (40.0, 20.0, 60.0, 60.0)


def test_object_on_the_left_has_negative_bearing():
    geometry = _geometry()
    left = Detection.from_box("cup", 0.9, (0, 20, 20, 60), geometry=geometry, frame_size=(100, 100))
    right = Detection.from_box(
        "cup", 0.9, (80, 20, 100, 60), geometry=geometry, frame_size=(100, 100)
    )
    assert left.bearing_deg < 0.0 < right.bearing_deg


def test_resolve_weights_prefers_the_existing_engine(tmp_path):
    engine = tmp_path / "model.engine"
    weights = tmp_path / "model.pt"
    weights.write_text("pt")
    assert resolve_weights(engine, weights) == str(weights)  # engine absent -> fallback
    engine.write_text("engine")
    assert resolve_weights(engine, weights) == str(engine)


def test_fake_detector_returns_scripted_detections():
    geometry = _geometry()
    detection = Detection.from_box(
        "person", 0.8, (10, 10, 30, 90), geometry=geometry, frame_size=(100, 100), track_id=1
    )
    detector = FakeDetector([detection], geometry=geometry)
    out = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
    assert out == [detection]
    detector.set([])
    assert detector.detect(np.zeros((100, 100, 3), dtype=np.uint8)) == []
