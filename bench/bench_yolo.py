#!/usr/bin/env python3
"""bench 1.2: YOLO11n TensorRT FP16 imgsz 640 — model-only and end-to-end FPS, with/without ByteTrack.

End-to-end = pre (letterbox/normalize) + inference + post (NMS). RSS recorded.
"""

from __future__ import annotations

import sys
import time

import numpy as np


def bench(model_path: str, name: str, n: int = 300, track: bool = False) -> dict:
    from ultralytics import YOLO

    model = YOLO(model_path, task="detect")
    frame = np.random.randint(0, 255, (616, 820, 3), dtype=np.uint8)

    # warmup 3 runs (plan: warm up 3, then measure N)
    for _ in range(3):
        if track:
            model.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=640, verbose=False)
        else:
            model(frame, imgsz=640, verbose=False)

    t0 = time.perf_counter()
    for _ in range(n):
        if track:
            model.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=640, verbose=False)
        else:
            model(frame, imgsz=640, verbose=False)
    dt = time.perf_counter() - t0

    import psutil

    rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    fps = n / dt
    print(f"{name}: {fps:.1f} FPS ({dt * 1000 / n:.2f} ms/frame), RSS {rss_mb:.0f} MB")
    return {"fps": round(fps, 1), "ms_per_frame": round(dt * 1000 / n, 2), "rss_mb": round(rss_mb)}


if __name__ == "__main__":
    sys.path.insert(0, "bench")
    from common import write_result

    results = {}
    results["model_only"] = bench("models/yolo_trt/yolo11n_fp16.engine", "model-only (predict)")
    results["with_bytetrack"] = bench(
        "models/yolo_trt/yolo11n_fp16.engine", "end-to-end + ByteTrack", track=True
    )
    print(write_result("p12_yolo11n_trt_fp16", results))
