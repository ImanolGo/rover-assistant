"""Laptop-only tests for api.server: endpoints, overlay, MJPEG, thermals."""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from rover.api.server import (
    VideoSnapshot,
    annotate,
    create_app,
    encode_jpeg,
    mjpeg_stream,
    read_thermal_zones,
)
from rover.perception.detector import Detection
from rover.perception.geometry import CameraGeometry


def _geometry() -> CameraGeometry:
    matrix = np.array([[400.0, 0.0, 410.0], [0.0, 400.0, 308.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return CameraGeometry(matrix, np.zeros(5, dtype=np.float32), (820, 616))


class FakeContext:
    """Implements ApiContext with no camera, for endpoint tests."""

    def __init__(self) -> None:
        self.commands: list[str] = []
        self.stopped = 0
        self._detection = Detection.from_box(
            "cup", 0.9, (10, 10, 100, 200), geometry=_geometry(), frame_size=(820, 616), track_id=3
        )

    def status(self) -> dict:
        return {"state": "APPROACH", "fps": 15.0, "objects": 1, "left": 0.1, "right": -0.1}

    def snapshot(self) -> VideoSnapshot:
        return VideoSnapshot(
            frame=np.zeros((616, 820, 3), dtype=np.uint8),
            detections=[self._detection],
            state="APPROACH",
            left=0.1,
            right=-0.1,
            fps=15.0,
        )

    def submit_command(self, text: str) -> bool:
        self.commands.append(text)
        return True

    def stop(self) -> None:
        self.stopped += 1


def test_health_and_status_include_resources():
    client = TestClient(create_app(FakeContext()))
    assert client.get("/health").json() == {"status": "ok"}
    status = client.get("/status").json()
    assert status["state"] == "APPROACH"
    assert status["fps"] == 15.0
    assert "memory" in status["resources"]
    assert "cpu_percent" in client.get("/api/resources").json()


def test_cmd_forwards_and_stop_calls_context():
    context = FakeContext()
    client = TestClient(create_app(context))
    response = client.post("/cmd", json={"text": "go to the cup"})
    assert response.json()["ok"] is True
    assert context.commands == ["go to the cup"]
    assert client.post("/stop").json() == {"ok": True}
    assert context.stopped == 1


def test_dashboard_is_served():
    html = TestClient(create_app(FakeContext())).get("/").text
    assert "<title>Rover Assistant</title>" in html
    assert "/video" in html


def test_annotate_draws_box_and_handles_missing_frame():
    frame = annotate(FakeContext().snapshot())
    assert frame.shape == (616, 820, 3)
    blank = annotate(VideoSnapshot())
    assert blank.shape == (616, 820, 3)


def test_encode_jpeg_and_mjpeg_stream():
    assert encode_jpeg(np.zeros((8, 8, 3), dtype=np.uint8)).startswith(b"\xff\xd8")
    part = next(mjpeg_stream(FakeContext(), max_frames=1))
    assert part.startswith(b"--frame")
    assert b"\xff\xd8" in part


def test_read_thermal_zones(tmp_path):
    zone = tmp_path / "thermal_zone0"
    zone.mkdir()
    (zone / "type").write_text("cpu-thermal")
    (zone / "temp").write_text("47500")
    assert read_thermal_zones(str(tmp_path)) == {"cpu-thermal": 47.5}
