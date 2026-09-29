"""hal.camera: camera HAL with csi | usb | file backends.

OpenCV has no GStreamer on this Jetson, so the CSI backend uses ``gi``/``Gst``
directly. The pipeline downscales in hardware (nvvidconv) to 820x616 and hands
pandas ``BGRx`` straight to the appsink — no CPU ``videoconvert`` (bench 1.1:
50% -> 23% of a core). A background thread keeps only the newest frame, so
perception always sees the freshest image instead of a queue.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class CameraConfig:
    """Tunables for the camera HAL (defaults mirror config/robot.yaml)."""

    backend: str = "csi"
    sensor_id: int = 0
    capture: tuple[int, int] = (1640, 1232)
    output: tuple[int, int] = (820, 616)
    fps: int = 30
    flip_method: int = 0
    usb_index: int = 0
    file_path: str = "bench/data/frames"


def build_csi_pipeline(config: CameraConfig | None = None) -> str:
    """GStreamer launch string: NVMM capture, HW downscale, BGRx appsink."""
    config = config or CameraConfig()
    width, height = config.capture
    out_width, out_height = config.output
    return (
        f"nvarguscamerasrc sensor-id={config.sensor_id} sensor-mode=-1 do-timestamp=true ! "
        f"video/x-raw(memory:NVMM),width={width},height={height},"
        f"framerate={config.fps}/1,format=NV12 ! "
        f"nvvidconv flip-method={config.flip_method} ! "
        f"video/x-raw,width={out_width},height={out_height},format=BGRx ! "
        "appsink name=appsink emit-signals=true max-buffers=1 drop=true sync=false"
    )


def bgrx_to_bgr(data: Any, width: int, height: int) -> np.ndarray:
    """Convert packed/padded BGRx bytes to a tight ``(h, w, 3)`` uint8 array."""
    flat = np.frombuffer(data, dtype=np.uint8)
    if flat.size == width * height * 4:
        return flat.reshape(height, width, 4)[:, :, :3].copy()
    if height and flat.size % height == 0:
        stride = flat.size // height
        if stride >= width * 4:
            padded = flat.reshape(height, stride)[:, : width * 4]
            return padded.reshape(height, width, 4)[:, :, :3].copy()
    raise ValueError(f"unexpected BGRx buffer size {flat.size} for {width}x{height}")


class FrameSource(Protocol):
    """Blocking frame source: returns ``(bgr_frame, capture_time)`` or None."""

    def read(self) -> tuple[np.ndarray, float] | None: ...

    def close(self) -> None: ...


class CsiSource:
    """CSI camera via ``nvarguscamerasrc`` and a gi/Gst appsink."""

    def __init__(self, config: CameraConfig | None = None):
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        self.config = config or CameraConfig()
        self._gst = Gst
        Gst.init(None)
        self._pipeline = Gst.parse_launch(build_csi_pipeline(self.config))
        self._sink = self._pipeline.get_by_name("appsink")
        self._pipeline.set_state(Gst.State.PLAYING)

    def read(self) -> tuple[np.ndarray, float] | None:
        gst = self._gst
        sample = self._sink.emit("try-pull-sample", gst.SECOND)
        if not sample:
            return None
        buffer = sample.get_buffer()
        ok, info = buffer.map(gst.MapFlags.READ)
        if not ok:
            return None
        try:
            structure = sample.get_caps().get_structure(0)
            width = structure.get_int("width")[1]
            height = structure.get_int("height")[1]
            frame = bgrx_to_bgr(bytes(info.data), width, height)
        finally:
            buffer.unmap(info)
        return frame, time.time()

    def close(self) -> None:
        self._pipeline.set_state(self._gst.State.NULL)


class UsbSource:
    """USB webcam via ``cv2.VideoCapture``."""

    def __init__(self, config: CameraConfig | None = None):
        import cv2

        self.config = config or CameraConfig()
        self._cv2 = cv2
        self._capture = cv2.VideoCapture(self.config.usb_index)

    def read(self) -> tuple[np.ndarray, float] | None:
        ok, frame = self._capture.read()
        if not ok:
            return None
        return frame, time.time()

    def close(self) -> None:
        self._capture.release()


class FileSource:
    """Loops over a folder of images (or a single video file) via OpenCV."""

    def __init__(self, config: CameraConfig | None = None):
        import cv2

        self._cv2 = cv2
        self.config = config or CameraConfig()
        root = Path(self.config.file_path)
        self._images = (
            sorted(p for p in root.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)
            if root.is_dir()
            else []
        )
        self._video = cv2.VideoCapture(str(root)) if root.is_file() else None
        if not self._images and self._video is None:
            raise FileNotFoundError(f"no images or video at {root}")
        self._index = 0

    def read(self) -> tuple[np.ndarray, float] | None:
        if self._video is not None:
            ok, frame = self._video.read()
            if not ok:
                self._video.set(self._cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._video.read()
            return (frame, time.time()) if ok else None
        frame = self._cv2.imread(str(self._images[self._index % len(self._images)]))
        self._index += 1
        time.sleep(1.0 / self.config.fps)  # pace a folder like a camera
        return (frame, time.time()) if frame is not None else None

    def close(self) -> None:
        if self._video is not None:
            self._video.release()


class FakeSource:
    """Replays in-memory frames; used by the laptop tests / sim."""

    def __init__(self, frames: list[np.ndarray]):
        self._frames = frames
        self._index = 0
        self.closed = False

    def read(self) -> tuple[np.ndarray, float] | None:
        if not self._frames:
            return None
        frame = self._frames[self._index % len(self._frames)]
        self._index += 1
        return frame, time.time()

    def close(self) -> None:
        self.closed = True


class Camera:
    """Latest-frame camera: a thread grabs, :meth:`latest` never queues."""

    def __init__(self, source: FrameSource, name: str = "camera"):
        self.name = name
        self._source = source
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._t_capture = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            got = self._source.read()
            if got is None:
                time.sleep(0.005)
                continue
            frame, t_capture = got
            with self._lock:
                self._frame = frame
                self._t_capture = t_capture

    def latest(self, timeout: float = 2.0) -> tuple[np.ndarray, float]:
        """Newest ``(frame, t_capture)``; raise TimeoutError if none arrives."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._frame is not None:
                    return self._frame, self._t_capture
            time.sleep(0.005)
        raise TimeoutError(f"{self.name}: no frame within {timeout}s")

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._source.close()


def make_source(config: CameraConfig | None = None) -> FrameSource:
    """Build the frame source for the configured backend."""
    config = config or CameraConfig()
    if config.backend == "csi":
        return CsiSource(config)
    if config.backend == "usb":
        return UsbSource(config)
    if config.backend == "file":
        return FileSource(config)
    raise ValueError(f"unknown camera backend {config.backend!r}")


def open_camera(config: CameraConfig | None = None, source: FrameSource | None = None) -> Camera:
    """Build a :class:`Camera` (optionally wrapping an injected source)."""
    return Camera(source if source is not None else make_source(config))
