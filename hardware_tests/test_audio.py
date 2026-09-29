#!/usr/bin/env python3
"""Manual hardware test: select mic/speaker by name, record, and play back.

Run on the Jetson:

    .venv/bin/python hardware_tests/test_audio.py --seconds 3

Records through the USB mic at 16 kHz mono, reports the peak level, and plays
the clip back through the UACDemoV1.0 speaker (resampled to 48 kHz stereo).
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np

from rover.hal.audio import (
    AlsaCapture,
    AlsaSpeaker,
    AudioConfig,
    list_alsa_devices,
    normalize_int16,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--out", default="bench/results/raw/hw_audio.wav")
    parser.add_argument("--record-only", action="store_true")
    args = parser.parse_args()

    config = AudioConfig()
    devices = list_alsa_devices("-l")
    print(devices or "(arecord -l returned nothing)")

    capture = AlsaCapture(config)
    frames = []
    needed = int(args.seconds * 1000 / config.frame_ms)
    try:
        for _ in range(needed):
            frame = capture.read_frame()
            if frame is None:
                break
            frames.append(frame)
    finally:
        capture.close()

    if not frames:
        raise SystemExit("no audio captured — check the mic name in config/robot.yaml")
    clip = normalize_int16(np.concatenate(frames))
    print(f"captured {len(frames)} frames, peak={int(np.abs(clip).max())}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(config.mic_rate)
        handle.writeframes(clip.tobytes())
    print(f"wrote {out}")

    if not args.record_only:
        speaker = AlsaSpeaker(config)
        try:
            speaker.play(clip, config.mic_rate)
        finally:
            speaker.close()
        print("played back")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
