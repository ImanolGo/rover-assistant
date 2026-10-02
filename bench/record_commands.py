#!/usr/bin/env python3
"""PLAN 1.10 / B1: record real-mic commands (watts up, motors running).

Records through the *real* audio HAL (same device/route as runtime) into
``bench/data/commands/`` with a ``labels.jsonl`` of ``{file, text, intent,
target}``. A human prompts and speaks each line; takes can be redone.

    .venv/bin/python bench/record_commands.py --yes          # motors on, wheels up
    .venv/bin/python bench/record_commands.py --no-motors    # quiet, for comparison

Motors spin the wheels in place (wheels OFF the ground) so the recordings carry
the same noise the runtime sees. ``--yes`` confirms wheels are raised.

Keys per line: Enter = accept, r = redo, s = skip, q = quit.
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

# Reuse the synthesized-command list so the two datasets stay aligned.
sys.path.insert(0, "bench")
from bench_voice import COMMANDS as BENCH_COMMANDS  # noqa: E402

from rover.config import load_config  # noqa: E402
from rover.hal.audio import AlsaCapture, normalize_int16  # noqa: E402
from rover.hal.rover import RoverConfig, SerialRover  # noqa: E402
from rover.voice.intents import classify  # noqa: E402

OUT_DIR = Path("bench/data/commands")

EXTRA: list[tuple[str, str, str | None]] = [
    ("freeze", "stop", None),
    ("cancel", "stop", None),
    ("abort", "stop", None),
    ("what time is it", "unknown", None),
    ("tell me a joke", "unknown", None),
    ("how are you today", "unknown", None),
    ("play some music", "unknown", None),
    ("thank you", "unknown", None),
]


def build_lines() -> list[dict[str, str | None]]:
    lines: list[dict[str, str | None]] = []
    seen: set[str] = set()
    for text, intent, target in BENCH_COMMANDS + EXTRA:
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append({"text": text, "intent": intent, "target": target})
    return lines


def slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text.lower()).strip("_")[:40]


def record(capture: AlsaCapture, seconds: float) -> np.ndarray:
    needed = int(seconds * 1000 / capture.config.frame_ms)
    frames = []
    for _ in range(needed):
        frame = capture.read_frame()
        if frame is None:
            break
        frames.append(frame)
    if not frames:
        return np.zeros(0, dtype=np.int16)
    return normalize_int16(np.concatenate(frames))


def write_wav(path: Path, clip: np.ndarray, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(clip.tobytes())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--yes", action="store_true", help="confirm wheels are off the ground")
    parser.add_argument(
        "--no-motors", action="store_true", help="record without spinning the wheels"
    )
    parser.add_argument("--speed", type=float, default=0.15)
    args = parser.parse_args()

    motors = not args.no_motors
    if motors and not args.yes:
        raise SystemExit(
            "Refusing to run motors: pass --yes only with the wheels raised "
            "(or use --no-motors)."
        )

    config = load_config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    capture = AlsaCapture(config.audio)
    rover = None
    if motors:
        rover = SerialRover(RoverConfig(port=config.rover.port, speed_cap=args.speed))

    lines = build_lines()
    labels_path = OUT_DIR / "labels.jsonl"
    records: list[dict] = []

    def save() -> None:
        labels_path.write_text("\n".join(json.dumps(row) for row in records) + "\n")

    try:
        for index, line in enumerate(lines):
            text = str(line["text"])
            target = f"{index:02d}_{slug(text)}.wav"
            while True:
                if rover is not None:
                    rover.drive(-args.speed, args.speed)  # spin in place (wheels up)
                print(f"[{index + 1}/{len(lines)}] record '{text}': ", end="", flush=True)
                clip = record(capture, args.seconds)
                if rover is not None:
                    rover.stop()
                peak = int(np.abs(clip).max()) if clip.size else 0
                choice = input(f"peak={peak} [Enter=accept r=redo s=skip q=quit] ").strip().lower()
                if choice == "q":
                    save()
                    print(f"saved {len(records)} recordings")
                    return 0
                if choice == "s":
                    break
                if choice == "r":
                    continue
                write_wav(OUT_DIR / target, clip, config.audio.mic_rate)
                intent = classify(text)
                records.append(
                    {
                        "file": target,
                        "text": text,
                        "intent": line["intent"] or intent.name,
                        "target": line["target"] or intent.target,
                        "peak": peak,
                    }
                )
                break
    finally:
        capture.close()
        if rover is not None:
            rover.close()

    save()
    print(f"wrote {len(records)} recordings + {labels_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
