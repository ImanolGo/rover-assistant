#!/usr/bin/env python3
"""bench 1.5: openWakeWord 'hey_roe_ver.onnx' — CPU% while idle-listening + detection on clips.

Also checks the model fires on HeyRover.wav and (ideally) not on HeyJarvis.wav.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import soundfile as sf


def load_model():
    from openwakeword.model import Model

    return Model(wakeword_models=["models/wake_word/hey_roe_ver.onnx"], inference_framework="onnx")


def score_clip(model, path: str) -> dict:
    data, sr = sf.read(path, dtype="int16")
    if data.ndim > 1:
        data = data[:, 0]
    if sr != 16000:
        import scipy.signal

        n = int(len(data) * 16000 / sr)
        data = scipy.signal.resample(data, n).astype(np.int16)
    chunk = 1280  # 80 ms
    scores = []
    for i in range(0, len(data) - chunk, chunk):
        pred = model.predict(data[i : i + chunk])
        scores.append(max(pred.values()))
    model.reset()
    return {"clip": path, "max_score": round(float(max(scores)), 3), "frames_over_0.5": int(np.sum(np.array(scores) > 0.5))}


def idle_cpu(model, seconds: float = 15.0) -> float:
    """Stream silence at 80 ms frames, measuring CPU% of this process."""
    import os

    def jiffies():
        parts = open("/proc/self/stat").read().rsplit(") ", 1)[1].split()
        return int(parts[11]) + int(parts[12])

    j0 = jiffies()
    t0 = time.monotonic()
    silence = np.zeros(1280, dtype=np.int16)
    n = int(seconds / 0.08)
    for _ in range(n):
        model.predict(silence)
        time.sleep(0.08 - 0.006)  # roughly the real cadence
    dt = time.monotonic() - t0
    return 100.0 * (jiffies() - j0) / 100.0 / dt


if __name__ == "__main__":
    sys.path.insert(0, "bench")
    from common import write_result

    m = load_model()
    scores = [score_clip(m, p) for p in ("assets/audio/HeyRover.wav", "assets/audio/HeyJarvis.wav")]
    print(scores)
    cpu = idle_cpu(m)
    print(f"idle-listening CPU: {cpu:.0f}% of one core")
    print(write_result("p15_wake_hey_roe_ver", {"clips": scores, "idle_cpu_percent": round(cpu, 1)}))
