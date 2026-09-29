#!/usr/bin/env python3
"""Manual hardware test: CSI capture rate, resolution and a saved frame.

Run on the Jetson:

    .venv/bin/python hardware_tests/test_camera.py --frames 120

It warms up the sensor (auto-exposure), then reports capture FPS for the
configured 1640x1232 -> 820x616 nvvidconv downscale and writes one PNG.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from rover.hal.camera import Camera, CameraConfig, CsiSource


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--out", default="bench/results/raw/hw_camera_frame.png")
    parser.add_argument("--warmup", type=float, default=2.0)
    args = parser.parse_args()

    config = CameraConfig()
    camera = Camera(CsiSource(config))
    try:
        time.sleep(args.warmup)
        times = []
        frame = None
        for _ in range(args.frames):
            frame, t_capture = camera.latest()
            times.append(t_capture)
        intervals = [b - a for a, b in zip(times, times[1:]) if b > a]
        fps = 1.0 / (sum(intervals) / len(intervals)) if intervals else 0.0
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), frame)
        print(f"captured {len(times)} frames from {config.capture} -> {frame.shape}")
        print(f"mean fps: {fps:.1f}; wrote {out}")
    finally:
        camera.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
