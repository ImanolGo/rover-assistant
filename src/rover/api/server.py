"""rover.api.server: FastAPI debug surface for the brain (PLAN Phase 3).

Endpoints:
    GET  /health          liveness probe
    GET  /status          state + resources (CPU/RAM/disk/load/thermals)
    GET  /api/resources   host metrics only
    GET  /video           MJPEG: boxes, track ids, state, commanded L/R bars
    GET  /                HTML dashboard
    POST /cmd             inject a text command (debug)
    POST /stop            immediate stop (same path as the "stop" regex)

The server is deliberately thin: everything it renders comes from an
:class:`ApiContext` implemented by the app (``rover.main``). That keeps it
testable with a fake context and no camera.
"""

from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Protocol

import cv2
import numpy as np
import psutil
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

_THERMAL_ZONE_DIR = "/sys/devices/virtual/thermal"
_JPEG_QUALITY = 80
_DASHBOARD = Path(__file__).with_name("dashboard.html")


def read_thermal_zones(base_dir: str = _THERMAL_ZONE_DIR) -> dict[str, float]:
    """Best-effort ``thermal_zone*`` temperatures in Celsius (ported from legacy)."""
    temps: dict[str, float] = {}
    for zone in sorted(glob.glob(os.path.join(base_dir, "thermal_zone*"))):
        try:
            with open(os.path.join(zone, "type"), "r", encoding="utf-8") as handle:
                name = handle.read().strip()
            with open(os.path.join(zone, "temp"), "r", encoding="utf-8") as handle:
                millidegrees = int(handle.read().strip())
        except (OSError, ValueError, TypeError, AttributeError):
            continue
        temps[name or os.path.basename(zone)] = round(millidegrees / 1000.0, 1)
    return temps


def resource_snapshot() -> dict[str, Any]:
    """Host CPU / memory / disk / load / temperatures (legacy ``resource_snapshot``)."""
    memory = psutil.virtual_memory()
    try:
        disk = psutil.disk_usage("/")
        disk_data: dict[str, Any] = {
            "total_gb": round(disk.total / 1024**3, 1),
            "used_percent": disk.percent,
        }
    except (OSError, AttributeError):
        disk_data = {}
    try:
        load_avg: list[float] = [round(value, 2) for value in psutil.getloadavg()]
    except (AttributeError, OSError):
        load_avg = []
    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "cpu_count": psutil.cpu_count() or 0,
        "load_avg": load_avg,
        "memory": {
            "total_mb": round(memory.total / 1024 / 1024, 1),
            "available_mb": round(memory.available / 1024 / 1024, 1),
            "used_mb": round((memory.total - memory.available) / 1024 / 1024, 1),
            "percent": memory.percent,
        },
        "disk": disk_data,
        "temperatures_c": read_thermal_zones(),
    }


@dataclass
class VideoSnapshot:
    """Everything needed to render one video frame and the status overlay."""

    frame: np.ndarray | None = None
    detections: list[Any] = field(default_factory=list)
    state: str = "IDLE"
    left: float = 0.0
    right: float = 0.0
    fps: float = 0.0


class ApiContext(Protocol):
    """What ``rover.main`` must expose to the API."""

    def status(self) -> dict[str, Any]: ...

    def snapshot(self) -> VideoSnapshot: ...

    def submit_command(self, text: str) -> bool: ...

    def stop(self) -> None: ...


class CommandIn(BaseModel):
    """Body of ``POST /cmd``."""

    text: str


def encode_jpeg(frame: np.ndarray, quality: int = _JPEG_QUALITY) -> bytes:
    """Encode a BGR frame as JPEG bytes (empty on failure)."""
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buffer.tobytes() if ok else b""


def _draw_bar(
    canvas: np.ndarray, y: int, label: str, value: float, color: tuple[int, int, int]
) -> None:
    """Draw a signed L/R bar: 0 in the middle, value in [-1, 1] mapped to half-width."""
    height, width = canvas.shape[:2]
    mid = width // 2
    span = int((width // 2 - 20) * max(-1.0, min(1.0, value)))
    cv2.line(canvas, (mid, y), (mid, y), (255, 255, 255), 1)
    if span:
        cv2.rectangle(canvas, (mid, y - 6), (mid + span, y + 6), color, -1)
    cv2.putText(canvas, label, (10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"{value:+.2f}",
        (width - 70, y + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def annotate(snapshot: VideoSnapshot, size: tuple[int, int] = (820, 616)) -> np.ndarray:
    """Render boxes, ids, state and the commanded L/R bars onto a copy of the frame."""
    if snapshot.frame is None:
        canvas = np.full((size[1], size[0], 3), 30, dtype=np.uint8)
    else:
        canvas = snapshot.frame.copy()
    height, width = canvas.shape[:2]

    for detection in snapshot.detections:
        x1, y1, x2, y2 = (int(round(v)) for v in detection.box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        track = "" if detection.track_id is None else f"#{detection.track_id} "
        label = f"{track}{detection.cls} {detection.conf:.2f} b{detection.bearing_deg:+.0f}"
        cv2.putText(
            canvas,
            label,
            (x1, max(12, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        f"{snapshot.state}  {snapshot.fps:4.1f} fps  {len(snapshot.detections)} obj",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    _draw_bar(canvas, height - 40, "L", snapshot.left, (0, 200, 255))
    _draw_bar(canvas, height - 16, "R", snapshot.right, (255, 120, 0))
    return canvas


def mjpeg_stream(
    context: ApiContext, fps: float = 10.0, max_frames: int | None = None
) -> Iterator[bytes]:
    """Yield MJPEG parts until the client disconnects or ``max_frames`` is reached."""
    period = 1.0 / fps if fps > 0 else 0.0
    produced = 0
    while max_frames is None or produced < max_frames:
        started = time.time()
        frame = annotate(context.snapshot())
        jpeg = encode_jpeg(frame)
        if jpeg:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        produced += 1
        if max_frames is None:
            delay = period - (time.time() - started)
            if delay > 0:
                time.sleep(delay)


def create_app(context: ApiContext) -> FastAPI:
    """Build the FastAPI app bound to ``context``."""
    app = FastAPI(title="rover-assistant", version="0.1.0")
    started = time.time()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/status")
    def status() -> dict[str, Any]:
        data: dict[str, Any] = {
            "status": "ok",
            "uptime_s": round(time.time() - started, 1),
            "resources": resource_snapshot(),
        }
        data.update(context.status())
        return data

    @app.get("/api/resources")
    def resources() -> dict[str, Any]:
        return resource_snapshot()

    @app.get("/video")
    def video() -> StreamingResponse:
        return StreamingResponse(
            mjpeg_stream(context), media_type="multipart/x-mixed-replace; boundary=frame"
        )

    @app.get("/", response_class=HTMLResponse)
    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard() -> str:
        return _DASHBOARD.read_text(encoding="utf-8")

    @app.post("/cmd")
    def command(payload: CommandIn) -> dict[str, Any]:
        return {"ok": context.submit_command(payload.text), "text": payload.text}

    @app.post("/stop")
    def stop() -> dict[str, bool]:
        context.stop()
        return {"ok": True}

    return app


def run_server(context: ApiContext, host: str, port: int, **kwargs: Any) -> None:
    """Blocking uvicorn entry point (used by ``rover.main`` in a thread/task)."""
    import uvicorn

    uvicorn.run(create_app(context), host=host, port=port, log_level="warning", **kwargs)
