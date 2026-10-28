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

``--guided``: the robot speaks each prompt, then plays the wake chime; speak
right after the chime (the take starts there).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
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
from rover.voice.tts import play_wav  # noqa: E402

OUT_DIR = Path("bench/data/commands")
PROMPT_GAP_S = 0.3  # silence between the spoken prompt and the chime
MOTOR_DELAY_S = 0.7  # wheels start after the chime is heard, so they cannot mask it

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


def record_take(capture: AlsaCapture, rover, seconds: float, speed: float) -> np.ndarray:
    # Keep refreshing the command: the rover writer zeroes the wheels after 0.5 s
    # with no fresh command, so a single drive() would only twitch.
    stop = threading.Event()
    spinner = None
    if rover is not None:

        def _spin() -> None:
            if stop.wait(MOTOR_DELAY_S):
                return
            while not stop.is_set():
                rover.drive(-speed, speed)  # spin in place (wheels up)
                time.sleep(0.1)

        spinner = threading.Thread(target=_spin, daemon=True)
        spinner.start()
    try:
        return record(capture, seconds)
    finally:
        stop.set()
        if spinner is not None:
            spinner.join(timeout=0.5)
        if rover is not None:
            rover.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--yes", action="store_true", help="confirm wheels are off the ground")
    parser.add_argument(
        "--no-motors", action="store_true", help="record without spinning the wheels"
    )
    parser.add_argument("--speed", type=float, default=0.30)
    parser.add_argument(
        "--guided", action="store_true", help="speak each prompt via TTS; auto-accept"
    )
    parser.add_argument("--repeat-stop", type=int, default=0, help="extra stop takes (guided)")
    parser.add_argument(
        "--noise-only",
        action="store_true",
        help="record --seconds of motor noise only into motor_noise.wav (false-wake test)",
    )
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

    if args.noise_only:
        print(f"recording {args.seconds:.0f}s of motor noise (wheels up)...", flush=True)
        clip = record_take(capture, rover, args.seconds, args.speed)
        write_wav(OUT_DIR / "motor_noise.wav", clip, config.audio.mic_rate)
        capture.close()
        if rover is not None:
            rover.close()
        print(f"wrote {OUT_DIR / 'motor_noise.wav'} ({clip.size / config.audio.mic_rate:.0f}s)")
        return 0

    lines = build_lines()
    labels_path = OUT_DIR / "labels.jsonl"
    records: list[dict] = []
    tts = None
    if args.guided:
        from rover.voice.tts import PiperTts

        tts = PiperTts(config.voice.tts_voice, config.audio)

    def save() -> None:
        labels_path.write_text("\n".join(json.dumps(row) for row in records) + "\n")

    def accept(text, clip, target, want_intent, want_target) -> None:
        peak = int(np.abs(clip).max()) if clip.size else 0
        write_wav(OUT_DIR / target, clip, config.audio.mic_rate)
        intent = classify(text)
        records.append(
            {
                "file": target,
                "text": text,
                "intent": want_intent or intent.name,
                "target": want_target or intent.target,
                "peak": peak,
            }
        )

    def guided_prompt(spoken: str, shown: str) -> None:
        # Like the runtime turn: drop the prompt audio, queue the chime without
        # waiting and start the take at once. Speaker delay + chime is ~0.6 s and a
        # quick answer starts right after the beep, so waiting for it clipped the
        # first word; the take keeps the chime (as runtime turns do).
        # Keep reading the mic while the robot talks: an unread arecord pipe fills
        # (~2 s) and arecord then backs up its own buffer, which flush() cannot
        # reach, so a long prompt's tail leaked into the take ("...further away").
        # The runtime's CapturePump always reads, so it never had this backlog.
        talking = threading.Event()
        talking.set()

        def discard() -> None:
            while talking.is_set():
                capture.read_frame()

        reader = threading.Thread(target=discard, daemon=True)
        reader.start()
        try:
            tts.say(spoken)
            time.sleep(PROMPT_GAP_S)
        finally:
            talking.clear()
            reader.join(timeout=1.0)
        capture.flush()
        play_wav(tts.speaker, config.voice.wake_sound, wait=False)
        print(shown, flush=True)

    try:
        for index, line in enumerate(lines):
            text = str(line["text"])
            target = f"{index:02d}_{slug(text)}.wav"
            if args.guided:
                guided_prompt(f"Say, {text}", f"[{index + 1}/{len(lines)}] {text}")
                clip = record_take(capture, rover, args.seconds, args.speed)
                accept(text, clip, target, line["intent"], line["target"])
                continue
            while True:
                print(f"[{index + 1}/{len(lines)}] record '{text}': ", end="", flush=True)
                clip = record_take(capture, rover, args.seconds, args.speed)
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
                accept(text, clip, target, line["intent"], line["target"])
                break

        for i in range(args.repeat_stop):
            target = f"stop_{i:02d}.wav"
            if tts is not None:
                spoken = "Say stop" if i == 0 else "Say stop again, a little further away"
                guided_prompt(spoken, f"[stop {i + 1}/{args.repeat_stop}]")
            else:
                print(f"[stop {i + 1}/{args.repeat_stop}] say stop: ", end="", flush=True)
            clip = record_take(capture, rover, args.seconds, args.speed)
            accept("stop", clip, target, "stop", None)
    finally:
        capture.close()
        if rover is not None:
            rover.close()
        if tts is not None:
            tts.close()

    save()
    print(f"wrote {len(records)} recordings + {labels_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
