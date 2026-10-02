#!/usr/bin/env python3
"""Background contention load for the STT gate (B3).

Camera + YOLO (TensorRT) resident, plus a continuous loop of Gemma vision
requests to llama-server — the same load G1 uses. Runs for ``DURATION`` seconds
(or until killed) and records min MemAvailable / swap via MemProbe and YOLO FPS.

    DURATION=240 .venv/bin/python bench/contention_load.py
"""

from __future__ import annotations

import base64
import json
import os
import statistics
import sys
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, "bench")
from bench_coexist import YoloLoop, vision_query  # noqa: E402
from common import MemProbe  # noqa: E402


def unique_b64(seed: int) -> str:
    img = np.random.default_rng(seed).integers(0, 255, (616, 820, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return base64.b64encode(buf.tobytes()).decode()


def main() -> int:
    duration = float(os.environ.get("DURATION", "0") or 0)
    probe = MemProbe("stt_contention", interval_s=0.5)
    probe.start()

    yolo = YoloLoop()
    stop = threading.Event()
    stats: dict = {"vision_ok": 0, "vision_fail": 0, "latency": [], "errors": 0}

    def vision_loop() -> None:
        seed = 0
        while not stop.is_set():
            seed += 1
            ok, wall, _ = vision_query(unique_b64(seed))
            if ok:
                stats["vision_ok"] += 1
                stats["latency"].append(wall)
            else:
                stats["vision_fail"] += 1

    thread = threading.Thread(target=vision_loop, daemon=True)
    thread.start()
    print("contention load up (camera + YOLO + vision requests)", flush=True)

    deadline = time.time() + duration if duration else None
    try:
        while deadline is None or time.time() < deadline:
            time.sleep(1.0)
    finally:
        stop.set()
        thread.join(timeout=5)
        yolo.close()
        mem = probe.stop()

    latency = sorted(stats["latency"])
    results = {
        "duration_s": duration,
        "vision_ok": stats["vision_ok"],
        "vision_fail": stats["vision_fail"],
        "vision_p50_s": round(statistics.median(latency), 2) if latency else None,
        "yolo_fps_median": (
            round(statistics.median(yolo.fps_samples), 1) if yolo.fps_samples else None
        ),
        **mem,
    }
    print(json.dumps(results, indent=2))
    from common import write_result

    print(write_result("p4b_stt_contention_load", results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
