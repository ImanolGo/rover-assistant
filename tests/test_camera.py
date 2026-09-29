"""Laptop-only tests for hal.camera: pipeline string, BGRx conversion, latest-frame."""

from __future__ import annotations

import time

import numpy as np
import pytest

from rover.hal.camera import (
    Camera,
    CameraConfig,
    FakeSource,
    FileSource,
    bgrx_to_bgr,
    build_csi_pipeline,
    open_camera,
)


def test_csi_pipeline_has_hw_downscale_and_no_videoconvert():
    pipeline = build_csi_pipeline(CameraConfig())
    assert "nvarguscamerasrc sensor-id=0" in pipeline
    assert "width=1640,height=1232" in pipeline
    assert "nvvidconv flip-method=0" in pipeline
    assert "width=820,height=616,format=BGRx" in pipeline
    assert "max-buffers=1 drop=true sync=false" in pipeline
    assert "videoconvert" not in pipeline


def test_bgrx_to_bgr_tight_buffer():
    pixels = np.arange(4 * 5 * 4, dtype=np.uint8).reshape(4, 5, 4)
    result = bgrx_to_bgr(pixels.tobytes(), 5, 4)
    assert result.shape == (4, 5, 3)
    assert np.array_equal(result, pixels[:, :, :3])


def test_bgrx_to_bgr_handles_row_padding():
    height, width = 3, 5
    stride = width * 4 + 8  # padded rows
    pixels = np.arange(height * width * 4, dtype=np.uint8).reshape(height, width, 4)
    padded = np.zeros((height, stride), dtype=np.uint8)
    padded[:, : width * 4] = pixels.reshape(height, width * 4)
    result = bgrx_to_bgr(padded.tobytes(), width, height)
    assert np.array_equal(result, pixels[:, :, :3])


def test_bgrx_to_bgr_rejects_bad_size():
    with pytest.raises(ValueError):
        bgrx_to_bgr(b"\x00" * 7, 5, 4)


def test_camera_returns_latest_frame_from_source():
    frames = [np.full((2, 2, 3), value, dtype=np.uint8) for value in (10, 20, 30)]
    camera = Camera(FakeSource(frames))
    try:
        frame, t_capture = camera.latest()
        assert frame.shape == (2, 2, 3)
        assert t_capture <= time.time()
    finally:
        camera.close()


def test_camera_times_out_without_frames():
    camera = Camera(FakeSource([]))
    try:
        with pytest.raises(TimeoutError):
            camera.latest(timeout=0.1)
    finally:
        camera.close()


def test_camera_close_closes_source():
    source = FakeSource([np.zeros((1, 1, 3), dtype=np.uint8)])
    camera = Camera(source)
    camera.close()
    assert source.closed


def test_open_camera_with_injected_source():
    camera = open_camera(source=FakeSource([np.ones((2, 2, 3), dtype=np.uint8)]))
    try:
        assert camera.latest()[0].shape == (2, 2, 3)
    finally:
        camera.close()


def test_file_source_loops_images(tmp_path):
    cv2 = pytest.importorskip("cv2")
    for index in range(2):
        cv2.imwrite(str(tmp_path / f"frame_{index}.png"), np.full((4, 6, 3), index * 50, np.uint8))
    source = FileSource(CameraConfig(backend="file", file_path=str(tmp_path), fps=1000))
    try:
        first = source.read()
        second = source.read()
        third = source.read()
    finally:
        source.close()
    assert first is not None and second is not None and third is not None
    assert third[0].shape == (4, 6, 3)
    # Loops back to the first image.
    assert np.array_equal(third[0][0, 0], first[0][0, 0])
