#!/usr/bin/env python3
"""Vision-only llama load (no camera) for contention runs.

Use when the brain app already owns the camera + YOLO: this only drives Gemma
vision requests (llama's single slot) and samples memory, so an STT bench can be
run against the same load G1 uses.

    .venv/bin/python bench/vision_load.py --duration 150
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time

sys.path.insert(0, "bench")
from bench_coexist import vision_query  # noqa: E402
from common import MemProbe, write_result  # noqa: E402
from soak import unique_b64  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=150.0)
    parser.add_argument("--out", default="p4b_vision_load")
    args = parser.parse_args()

    probe = MemProbe(args.out, interval_s=0.5)
    probe.start()

    stop = threading.Event()
    stats = {"ok": 0, "fail": 0, "lat": []}

    def loop() -> None:
        seed = 0
        while not stop.is_set():
            seed += 1
            good, wall, _ = vision_query(unique_b64(seed))
            if good:
                stats["ok"] += 1
                stats["lat"].append(wall)
            else:
                stats["fail"] += 1

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    time.sleep(args.duration)
    stop.set()
    thread.join(timeout=10)
    mem = probe.stop()

    lat = sorted(stats["lat"])
    results = {
        "duration_s": args.duration,
        "vision_ok": stats["ok"],
        "vision_fail": stats["fail"],
        "vision_p50_s": round(statistics.median(lat), 2) if lat else None,
        **mem,
    }
    print(json.dumps(results, indent=2))
    print(write_result(args.out, results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
