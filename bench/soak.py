#!/usr/bin/env python3
"""Phase 6 baseline soak: full stack running, one Gemma vision query at a time.

Assumes llama-server is healthy and the brain app (`rover`, real) is running so
camera + YOLO + voice-idle are resident. This loop only issues vision queries and
samples memory. Records MemAvailable min/p5, swap growth, vision ok/fail, and the
app's reported YOLO FPS.

    .venv/bin/python bench/soak.py --minutes 30
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import statistics
import sys
import time

import cv2
import httpx
import numpy as np

sys.path.insert(0, "bench")
from bench_coexist import vision_query  # noqa: E402
from common import MemProbe, write_result  # noqa: E402

URL = "http://127.0.0.1:8080"
APP = "http://127.0.0.1:8000"


def unique_b64(seed: int) -> str:
    img = np.random.default_rng(seed).integers(0, 255, (616, 820, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return base64.b64encode(buf.tobytes()).decode()


def app_status() -> dict:
    try:
        return httpx.get(f"{APP}/status", timeout=3).json()
    except Exception:  # noqa: BLE001
        return {}


def available_series(path: str) -> list[float]:
    """Read the MemAvailable column from the MemProbe CSV."""
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            key = next((k for k in reader.fieldnames or [] if "avail" in k.lower()), None)
            return [float(row[key]) for row in reader if key and row.get(key)]
    except (OSError, ValueError):
        return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=float, default=30.0)
    parser.add_argument("--vision-every", type=float, default=15.0)
    parser.add_argument("--out", default="p6_soak_baseline")
    args = parser.parse_args()

    probe = MemProbe(args.out, interval_s=0.5)
    probe.start()

    seed = 0
    ok = fail = 0
    latencies: list[float] = []
    fps: list[float] = []
    deadline = time.time() + args.minutes * 60
    while time.time() < deadline:
        seed += 1
        good, wall, _ = vision_query(unique_b64(seed))
        if good:
            ok += 1
            latencies.append(wall)
        else:
            fail += 1
        status = app_status()
        if isinstance(status.get("fps"), (int, float)):
            fps.append(float(status["fps"]))
        time.sleep(max(0.0, args.vision_every - wall))

    mem = probe.stop()
    series = available_series(mem.get("csv", ""))
    p5 = sorted(series)[int(len(series) * 0.05)] if series else None
    lat = sorted(latencies)

    results = {
        "minutes": args.minutes,
        "vision_ok": ok,
        "vision_fail": fail,
        "vision_p50_s": round(statistics.median(lat), 2) if lat else None,
        "vision_p90_s": round(lat[int(len(lat) * 0.9)], 2) if lat else None,
        "app_fps_p50": round(statistics.median(fps), 1) if fps else None,
        "app_fps_min": round(min(fps), 1) if fps else None,
        "mem_available_p5_mb": round(p5, 1) if p5 is not None else None,
        **mem,
    }
    print(json.dumps(results, indent=2))
    print(write_result(args.out, results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
