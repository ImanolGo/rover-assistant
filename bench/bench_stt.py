#!/usr/bin/env python3
"""bench 1.3: whisper.cpp base.en (CUDA) latency + RSS on test WAVs.

Runs whisper-cli as a persistent-style subprocess per clip (cold each time, as the
voice pipeline would spawn it per utterance) and reports per-clip latency.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import psutil

WHISPER = Path.home() / "whisper.cpp/build/bin/whisper-cli"
MODEL = "models/whisper/ggml-base.en.bin"
CLIPS = [
    "assets/audio/TheRainInSpain.wav",
    "assets/audio/HeyRover.wav",
    "assets/audio/HeyJarvis.wav",
]


def run_clip(clip: str, n: int = 5) -> dict:
    latencies = []
    text = ""
    for _ in range(n):
        t0 = time.perf_counter()
        proc = subprocess.run(
            [str(WHISPER), "-m", MODEL, "-f", clip, "-nt", "--no-prints", "-np"],
            capture_output=True,
            text=True,
            check=True,
        )
        latencies.append(time.perf_counter() - t0)
        text = proc.stdout.strip()
    lat_sorted = sorted(latencies)
    return {
        "clip": clip,
        "text": text[:60],
        "p50_s": round(lat_sorted[len(lat_sorted) // 2], 3),
        "p90_s": round(lat_sorted[int(len(lat_sorted) * 0.9)], 3),
    }


if __name__ == "__main__":
    sys.path.insert(0, "bench")
    from common import write_result

    # RSS of one cold run (peak while transcribing)
    proc = subprocess.Popen(
        [str(WHISPER), "-m", MODEL, "-f", CLIPS[0], "-nt", "--no-prints", "-np"],
        stdout=subprocess.DEVNULL,
    )
    peak_rss = 0
    p = psutil.Process(proc.pid)
    while proc.poll() is None:
        try:
            peak_rss = max(peak_rss, p.memory_info().rss / (1024 * 1024))
        except psutil.NoSuchProcess:
            break
        time.sleep(0.05)
    proc.wait()

    results = {"clips": [run_clip(c) for c in CLIPS], "peak_rss_mb": round(peak_rss)}
    for c in results["clips"]:
        print(c)
    print(f"peak RSS: {peak_rss:.0f} MB")
    print(write_result("p13_stt_whisper_base_en", results))
