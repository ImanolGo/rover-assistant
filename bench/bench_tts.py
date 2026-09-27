#!/usr/bin/env python3
"""bench 1.4: Piper TTS — time-to-first-audio and real-time factor for 5 sentences.

Uses the piper CLI (persistent-process equivalent: one load, many sentences via --output-raw
would stream; here we measure per-sentence synthesis cost after warm-up, which the
persistent-process design in voice/tts.py will reuse).
"""

from __future__ import annotations

import subprocess
import sys
import time
import wave
from pathlib import Path

import psutil

SENTENCES = [
    "Looking for the red cup.",
    "I found the cup and I am on my way.",
    "I have arrived at the cup.",
    "Sorry, I lost you. Say follow me when you are ready.",
    "Hey Rover is at your service. What do you see?",
]
VOICE = "models/piper/en_US-lessac-medium.onnx"
OUT_DIR = Path("bench/results/raw/piper")


def synth(sentence: str, idx: int) -> tuple[float, float]:
    """Returns (ttfa_s, rtf) — TTFA approximated by load+first sentence; RTF = synth_time / audio_time."""
    out = OUT_DIR / f"s{idx}.wav"
    t0 = time.perf_counter()
    subprocess.run(
        [".venv/bin/piper", "-m", VOICE, "-f", str(out)],
        input=sentence,
        text=True,
        capture_output=True,
        check=True,
    )
    elapsed = time.perf_counter() - t0
    with wave.open(str(out)) as w:
        audio_s = w.getnframes() / w.getframerate()
    return elapsed, elapsed / audio_s, audio_s


if __name__ == "__main__":
    sys.path.insert(0, "bench")
    from common import write_result

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, s in enumerate(SENTENCES):
        elapsed, rtf, audio_s = synth(s, i)
        rows.append(
            {"sentence": s[:40], "synth_s": round(elapsed, 2), "audio_s": round(audio_s, 2), "rtf": round(rtf, 2)}
        )
        print(rows[-1])

    # RSS of a running piper
    proc = subprocess.Popen(
        [".venv/bin/piper", "-m", VOICE, "--output-raw"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    proc.stdin.write(b"warm up\n")
    proc.stdin.flush()
    time.sleep(4)
    rss_mb = psutil.Process(proc.pid).memory_info().rss / (1024 * 1024)
    proc.stdin.close()
    proc.wait(timeout=30)

    import statistics

    results = {
        "sentences": rows,
        "p50_synth_s": round(statistics.median(r["synth_s"] for r in rows), 2),
        "p50_rtf": round(statistics.median(r["rtf"] for r in rows), 2),
        "rss_mb": round(rss_mb),
    }
    print(results)
    print(write_result("p14_tts_piper", results))
