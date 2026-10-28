#!/usr/bin/env python3
"""Playback timing: does ``PiperTts.say()`` return when the robot has gone quiet?

Records the real mic while the robot speaks one sentence and compares the moment
``say()`` returns with the last mic frame that still hears the speaker (RMS above
3x the room-noise p90). Positive ``return_minus_heard_s`` = say() returned after
the sound ended (safe for the barge-in ``speaking`` flag); negative = too early
(STATUS known issue 15). Speaks out loud; no motors.

    .venv/bin/python bench/bench_playback.py --runs 3
"""

from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time

import numpy as np

sys.path.insert(0, "bench")
from common import write_result  # noqa: E402

from rover.config import load_config  # noqa: E402
from rover.hal.audio import AlsaCapture  # noqa: E402
from rover.voice.tts import PiperTts  # noqa: E402

SENTENCE = "Testing playback timing. One, two, three, four."
FRAME_S = 0.08


def run_once(capture: AlsaCapture, tts: PiperTts) -> dict:
    frames: list[tuple[float, float]] = []
    stop = threading.Event()

    def grab() -> None:
        while not stop.is_set():
            frame = capture.read_frame()
            if frame is not None:
                rms = float(np.sqrt(np.mean(frame.astype(float) ** 2)))
                frames.append((time.monotonic(), rms))

    thread = threading.Thread(target=grab, daemon=True)
    thread.start()
    time.sleep(1.5)  # room-noise baseline
    started = time.monotonic()
    tts.say(SENTENCE)
    returned = time.monotonic()
    time.sleep(2.0)
    stop.set()
    thread.join(timeout=1.0)

    noise = float(np.percentile([r for t, r in frames if t < started], 90))
    threshold = max(3 * noise, 300.0)
    loud = [t for t, r in frames if t >= started and r > threshold]
    heard_end = loud[-1] + FRAME_S if loud else None
    return {
        "noise_p90": round(noise, 1),
        "say_returned_s": round(returned - started, 3),
        "heard_end_s": round(heard_end - started, 3) if heard_end else None,
        "return_minus_heard_s": round(returned - heard_end, 3) if heard_end else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    config = load_config()
    capture = AlsaCapture(config.audio)
    tts = PiperTts(config.voice.tts_voice, config.audio)
    try:
        runs = [run_once(capture, tts) for _ in range(args.runs)]
    finally:
        capture.close()
        tts.close()
    deltas = [r["return_minus_heard_s"] for r in runs if r["return_minus_heard_s"] is not None]
    result = {
        "sentence": SENTENCE,
        "speaker_latency_s": config.audio.speaker_latency_s,
        "runs": runs,
        "return_minus_heard_min_s": min(deltas) if deltas else None,
        "return_minus_heard_median_s": round(statistics.median(deltas), 3) if deltas else None,
        "early_runs": sum(d < 0 for d in deltas),
    }
    for r in runs:
        print(r)
    print(write_result("p4c_playback_timing", result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
