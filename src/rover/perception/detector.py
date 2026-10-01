"""perception.detector: YOLO11n + ByteTrack, with a deterministic fake.

On the Jetson the model is a TensorRT FP16 engine exported on-device; on a
laptop it falls back to ``.pt`` weights. Ultralytics is imported lazily, so this
module (and the whole brain) imports without torch on a laptop. Each
:class:`Detection` carries the bearing of its box centre and its height
fraction, which is what the control loop actually uses (no depth model).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class Detection:
    """One tracked object in one frame.

    ``box`` is ``(x1, y1, x2, y2)`` in pixels, ``bearing_deg`` is positive to the
    right of the image centre (ARCHITECTURE.md §5), and ``h_frac`` is the box
    height divided by the frame height (the distance proxy).
    """

    cls: str
    conf: float
    box: tuple[float, float, float, float]
    track_id: int | None
    bearing_deg: float
    h_frac: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @classmethod
    def from_box(
        cls,
        cls_name: str,
        conf: float,
        box: tuple[float, float, float, float],
        *,
        geometry: Any,
        frame_size: tuple[int, int],
        track_id: int | None = None,
    ) -> "Detection":
        """Build a detection, computing bearing and height fraction from the box."""
        x1, y1, x2, y2 = (float(value) for value in box)
        _, height = frame_size
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        return cls(
            cls=cls_name,
            conf=float(conf),
            box=(x1, y1, x2, y2),
            track_id=track_id,
            bearing_deg=geometry.bearing_deg(center_x, center_y),
            h_frac=(y2 - y1) / float(height) if height else 0.0,
        )


class Detector(Protocol):
    """Frame -> detections interface (YOLO on the robot, fake on a laptop)."""

    def detect(self, frame: np.ndarray) -> list[Detection]: ...

    def close(self) -> None: ...


def resolve_weights(primary: str | Path, fallback: str | Path | None) -> str:
    """Prefer the on-device engine; fall back to ``.pt`` weights if it is absent."""
    if Path(primary).exists():
        return str(primary)
    if fallback and Path(fallback).exists():
        return str(fallback)
    return str(primary)


class YoloDetector:
    """Ultralytics YOLO with ByteTrack persistence (``track(persist=True)``)."""

    def __init__(
        self,
        geometry: Any,
        model_path: str | Path,
        weights_fallback: str | Path | None = None,
        *,
        imgsz: int = 640,
        conf: float = 0.35,
        tracker: str = "bytetrack.yaml",
        device: str | int | None = None,
        threads: int = 1,
    ):
        from ultralytics import YOLO  # lazy: keeps laptops torch-free

        # Cap torch's CPU thread pool: inference is on the GPU, and the default
        # (one worker per core) burned ~425% CPU in the single brain process.
        try:
            import torch

            torch.set_num_threads(max(1, int(threads)))
        except (ImportError, RuntimeError):  # pragma: no cover - torch details
            pass
        self.geometry = geometry
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.tracker = tracker
        self.device = device
        self.model_path = resolve_weights(model_path, weights_fallback)
        self._model = YOLO(self.model_path)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        results = self._model.track(
            frame,
            persist=True,
            tracker=self.tracker,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            names = getattr(result, "names", {}) or {}
            track_ids = boxes.id.tolist() if boxes.id is not None else None
            classes = boxes.cls.tolist()
            confidences = boxes.conf.tolist()
            coords = boxes.xyxy.tolist()
            for index, box in enumerate(coords):
                class_id = int(classes[index])
                name = (
                    names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
                )
                track_id = int(track_ids[index]) if track_ids is not None else None
                detections.append(
                    Detection.from_box(
                        name,
                        confidences[index],
                        box,
                        geometry=self.geometry,
                        frame_size=(width, height),
                        track_id=track_id,
                    )
                )
        return detections

    def close(self) -> None:
        self._model = None


class FakeDetector:
    """Returns a scripted list of detections (laptop sim / tests, no torch)."""

    def __init__(self, detections: list[Detection] | None = None, geometry: Any = None):
        self._detections = list(detections or [])
        self.geometry = geometry

    def set(self, detections: list[Detection]) -> None:
        self._detections = list(detections)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return list(self._detections)

    def close(self) -> None: ...


def make_detector(perception: Any, geometry: Any, *, sim: bool = False) -> Detector:
    """Build the configured detector; in sim, fall back to a fake if unavailable.

    On real hardware a missing engine or missing ultralytics must fail loudly, so
    the fallback only applies when ``sim`` is set (laptops have no torch/TRT).
    """
    try:
        import ultralytics  # noqa: F401

        return YoloDetector(
            geometry,
            perception.engine,
            perception.weights_fallback,
            imgsz=perception.imgsz,
            conf=perception.conf,
            tracker=perception.tracker,
            threads=getattr(perception, "torch_threads", 1),
        )
    except Exception:  # noqa: BLE001 - sim must boot without a model
        if not sim:
            raise
        return FakeDetector(geometry=geometry)
