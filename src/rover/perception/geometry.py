"""perception.geometry: camera intrinsics, bearings and one-shot undistortion.

Loads the legacy calibration (calibrated at 1640x1232, plumb_bob / 5-coefficient
model) and scales K for the actual capture size (820x616 = 0.5x). The control
loop only calls :meth:`CameraGeometry.bearing_deg` on a bbox centre
(``cv2.undistortPoints``, microseconds); the full-frame remap in
:meth:`CameraGeometry.undistort` is used only on the frames sent to Gemma, with
cached maps. OpenCV has no CUDA here, so nothing remaps every frame.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


def load_calibration(path: str | Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Load ``camera_matrix`` / ``distortion_coefficients`` / image size."""
    with open(path, "r", encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle)
    matrix = np.array(data["camera_matrix"], dtype=np.float32).reshape(3, 3)
    coeffs = np.array(data["distortion_coefficients"], dtype=np.float32).flatten()
    size = (int(data.get("image_width", 0)), int(data.get("image_height", 0)))
    return matrix, coeffs, size


def scale_intrinsics(matrix: np.ndarray, scale: float) -> np.ndarray:
    """Scale fx, fy, cx, cy by ``scale`` (distortion coefficients are unchanged)."""
    scaled = np.array(matrix, dtype=np.float32)
    scaled[0, 0] *= scale
    scaled[1, 1] *= scale
    scaled[0, 2] *= scale
    scaled[1, 2] *= scale
    return scaled


class CameraGeometry:
    """Pinhole intrinsics with bearings from distorted pixels and cached remaps."""

    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        image_size: tuple[int, int],
        output_size: tuple[int, int] | None = None,
        alpha: float = 1.0,
    ):
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float32).reshape(3, 3)
        self.dist_coeffs = np.asarray(dist_coeffs, dtype=np.float32).flatten()
        self.calibration_size = (int(image_size[0]), int(image_size[1]))
        self.size = (
            (int(output_size[0]), int(output_size[1])) if output_size else self.calibration_size
        )
        self.alpha = float(alpha)
        if self.size[0] != self.calibration_size[0]:
            self.scale = self.size[0] / self.calibration_size[0]
        else:
            self.scale = 1.0
        self._k = scale_intrinsics(self.camera_matrix, self.scale)
        self._new_camera_matrix: np.ndarray | None = None
        self._roi: tuple[int, int, int, int] = (0, 0, self.size[0], self.size[1])
        self._maps: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        output_size: tuple[int, int] | None = None,
        alpha: float = 1.0,
    ) -> "CameraGeometry":
        matrix, coeffs, size = load_calibration(path)
        return cls(matrix, coeffs, size, output_size=output_size, alpha=alpha)

    @property
    def fx(self) -> float:
        return float(self._k[0, 0])

    @property
    def fy(self) -> float:
        return float(self._k[1, 1])

    @property
    def cx(self) -> float:
        return float(self._k[0, 2])

    @property
    def cy(self) -> float:
        return float(self._k[1, 2])

    def fov_h_deg(self) -> float:
        """Horizontal field of view in degrees (from the scaled focal length)."""
        return math.degrees(2.0 * math.atan2(self.size[0] / 2.0, self.fx))

    def undistort_points(self, points: np.ndarray) -> np.ndarray:
        """Undistort ``(N, 2)`` pixel coordinates in *this* image size to rays."""
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.undistortPoints(pts, self._k, self.dist_coeffs).reshape(-1, 2)

    def bearing_deg(self, u: float, v: float) -> float:
        """Horizontal bearing of pixel ``(u, v)``; positive = right of centre."""
        x, _ = self.undistort_points(np.array([[u, v]], dtype=np.float32))[0]
        return math.degrees(math.atan2(float(x), 1.0))

    def elevation_deg(self, u: float, v: float) -> float:
        """Vertical angle of pixel ``(u, v)``; positive = below centre (ROS optical)."""
        _, y = self.undistort_points(np.array([[u, v]], dtype=np.float32))[0]
        return math.degrees(math.atan2(float(y), 1.0))

    def _ensure_new_camera_matrix(self) -> None:
        if self._new_camera_matrix is None:
            new_k, roi = cv2.getOptimalNewCameraMatrix(
                self._k, self.dist_coeffs, self.size, self.alpha, self.size
            )
            self._new_camera_matrix = new_k
            self._roi = tuple(int(v) for v in roi)

    def _ensure_maps(self, frame_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        if frame_size not in self._maps:
            self._ensure_new_camera_matrix()
            self._maps[frame_size] = cv2.initUndistortRectifyMap(
                self._k,
                self.dist_coeffs,
                None,
                self._new_camera_matrix,
                frame_size,
                cv2.CV_16SC2,
            )
        return self._maps[frame_size]

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        """Full-frame remap (cached maps) for frames sent to Gemma."""
        height, width = frame.shape[:2]
        map_x, map_y = self._ensure_maps((width, height))
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
