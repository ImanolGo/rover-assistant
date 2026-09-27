#!/usr/bin/env python3
"""Phase 1.0: measure cv2.remap undistort cost per frame (cause #2 in ARCHITECTURE §10).

Legacy undistorted every frame at 1640x1232 on the CPU (OpenCV has no CUDA here).
New design undistorts only bbox centres; full remap only for frames sent to Gemma.
"""

from __future__ import annotations

import sys
import time

import cv2
import numpy as np
import yaml


def load_calibration(path: str):
    with open(path) as f:
        cal = yaml.safe_load(f)
    K = np.array(cal["camera_matrix"], dtype=np.float64).reshape(3, 3)
    D = np.array(cal["distortion_coefficients"], dtype=np.float64).ravel()
    return K, D


def remap_cost(w: int, h: int, K, D, alpha: float, n: int = 60) -> dict:
    newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha, (w, h))
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, newK, (w, h), cv2.CV_32FC1)
    frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    # warmup
    for _ in range(5):
        cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
    t0 = time.perf_counter()
    for _ in range(n):
        out = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
    dt = (time.perf_counter() - t0) / n
    fps = 1.0 / dt
    return {
        "resolution": f"{w}x{h}",
        "alpha": alpha,
        "ms_per_frame": round(dt * 1000, 2),
        "max_fps_if_every_frame": round(fps, 1),
        "output_shape": list(out.shape),
    }


if __name__ == "__main__":
    sys.path.insert(0, "bench")
    from common import write_result

    K, D = load_calibration("config/camera_calibration.yaml")
    results = {
        "full_1640x1232": remap_cost(1640, 1232, K, D, 1.0),
        "half_820x616": remap_cost(820, 616, K, D, 1.0),
    }
    for name, r in results.items():
        print(name, r)
    ms = results["half_820x616"]["ms_per_frame"]
    print(f"\nLegacy remapped EVERY frame at full res; new design: {ms} ms, Gemma queries only.")
    print(write_result("p10_undistort_cost", results))
