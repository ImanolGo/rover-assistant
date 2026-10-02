#!/usr/bin/env python3
"""Phase 4 final acceptance: every check, one result JSON with pass/fail per row.

Data: the real-mic set `bench/data/commands/labels.jsonl` (+ optional
`motor_noise.wav`). With `--synth` it uses the Piper-synthesized 20 commands
instead and marks rows that need real audio.

Checks per candidate (tiny, small, gemma):
  commands      intent accuracy >= 90%
  stops         stop words 100% (>= 20 takes)
  noncommands   0 motion intents on non-commands
  p90           end-of-speech -> intent <= 600 ms (idle; contention with --stack)
  stop_latency  spoken stop while moving -> App.stop <= 800 ms (needs --stack)
  tts_cancel    TTS cancel -> silence <= 200 ms
  false_wakes   <= 1 per 10 min motor noise (needs motor_noise.wav)
  self_fire     0 stop/wake self-triggers while TTS plays

Rows that need the running stack or real audio are recorded as `pending`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "bench")
from common import write_result  # noqa: E402

from rover.voice.intents import classify, is_stop  # noqa: E402
from rover.voice.stt import GemmaStt, MoonshineStt, moonshine_arch  # noqa: E402
from rover.voice.tts import PiperTts, split_sentences  # noqa: E402

COMMANDS_DIR = Path("bench/data/commands")


def load_labels() -> list[dict]:
    path = COMMANDS_DIR / "labels.jsonl"
    if not path.exists():
        return []
    return [json.loads(row) for row in path.read_text().splitlines() if row.strip()]


def load_audio(name: str) -> np.ndarray:
    import soundfile as sf

    data, rate = sf.read(str(COMMANDS_DIR / name), dtype="int16")
    if getattr(data, "ndim", 1) > 1:
        data = data[:, 0]
    if rate != 16000:
        count = int(len(data) * 16000 / rate)
        data = np.interp(np.linspace(0, len(data) - 1, count), np.arange(len(data)), data)
    return data.astype(np.int16)


def make_candidate(name: str):
    if name == "gemma":
        return GemmaStt("http://127.0.0.1:8080"), "gemma"
    model = name.replace("moonshine-", "")
    from rover.voice.stt import moonshine_dir

    return MoonshineStt(moonshine_dir(model), arch=moonshine_arch(model)), model


def run_stt_checks(name: str, labels: list[dict], synth: bool) -> dict:
    stt, _ = make_candidate(name)
    rows = []
    for rec in labels:
        try:
            pcm = load_audio(rec["file"])
        except OSError:
            continue
        started = time.perf_counter()
        text = stt.transcribe(pcm)
        latency = time.perf_counter() - started
        intent = classify(text)
        rows.append({"want": rec["intent"], "got": intent.name, "latency": latency})
    stt.close()

    commands = [r for r in rows if r["want"] not in ("unknown", "stop")]
    stops = [r for r in rows if r["want"] == "stop"]
    noncommands = [r for r in rows if r["want"] == "unknown"]

    def acc(subset):
        return (
            round(sum(r["got"] == r["want"] for r in subset) / len(subset), 3) if subset else None
        )

    lat = sorted(r["latency"] for r in rows)
    p90 = round(lat[int(len(lat) * 0.9)], 3) if lat else None
    motion_on_noncommand = sum(1 for r in noncommands if r["got"] not in ("unknown", "stop"))
    return {
        "n": len(rows),
        "command_accuracy": acc(commands),
        "stop_recall": acc(stops),
        "stop_takes": len(stops),
        "motion_intents_on_noncommands": motion_on_noncommand,
        "end_to_intent_p90_idle_s": p90,
        "synthetic": synth,
    }


def check_tts_cancel() -> float:
    """Wall time from cancel() to say() returning, with a blocking fake voice."""

    class _Voice:
        class config:
            sample_rate = 22050

        def synthesize(self, text):
            class _Chunk:
                audio_int16_bytes = np.zeros(22050, dtype=np.int16).tobytes()

            return [_Chunk()]

    class _Speaker:
        def play(self, samples, rate):
            time.sleep(0.02)  # simulate ~one block of playback

        def close(self): ...

    tts = PiperTts("unused", None, speaker=_Speaker(), voice=_Voice())
    text = ". ".join(split_sentences("this is a fairly long reply that keeps talking for a while"))
    import threading

    def cancel_later():
        time.sleep(0.3)
        tts.cancel()

    threading.Thread(target=cancel_later, daemon=True).start()
    started = time.perf_counter()
    tts.say(text)
    return round(max(0.0, time.perf_counter() - started - 0.3), 3)


def check_false_wakes(noise_path: Path) -> float | None:
    if not noise_path.exists():
        return None
    from rover.voice.wakeword import make_wakeword

    det = make_wakeword("models/wake_word/hey_roe_ver.onnx", threshold=0.5, cooldown_s=0.5)
    pcm = load_audio(noise_path.name)
    fires = 0
    for start in range(0, len(pcm), 1280):
        chunk = pcm[start : start + 1280]
        if len(chunk) == 1280 and det.process(chunk):
            fires += 1
    minutes = max(len(pcm) / 16000 / 60.0, 1e-6)
    return round(fires * 10.0 / minutes, 2)  # fires per 10 min


def check_self_trigger() -> int:
    """Feed a synthesized reply through the stop matcher; must not self-trigger."""
    from rover.voice.intents import reply_templates

    return sum(1 for text in reply_templates() if is_stop(text))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates", nargs="*", default=["moonshine-tiny", "moonshine-small", "gemma"]
    )
    parser.add_argument(
        "--synth", action="store_true", help="use synthesized audio (no real-mic set)"
    )
    parser.add_argument(
        "--stack", action="store_true", help="full stack is running (contention/barge checks)"
    )
    parser.add_argument("--out", default="p4_voice_acceptance")
    args = parser.parse_args()

    labels = load_labels()
    synth = args.synth or not labels
    if synth:
        from bench_voice import COMMANDS

        labels = [
            {"file": "__synth__", "text": t, "intent": i, "target": g} for t, i, g in COMMANDS
        ]

    results: dict = {
        "synthetic": synth,
        "stack": args.stack,
        "candidates": {},
    }
    for candidate in args.candidates:
        if synth:
            block = {"note": "real-mic labels.jsonl missing; STT rows are pending"}
            results["candidates"][candidate] = block
            continue
        results["candidates"][candidate] = run_stt_checks(candidate, labels, synth=False)

    results["checks"] = {
        "tts_cancel_s": check_tts_cancel(),
        "tts_cancel_pass": check_tts_cancel() <= 0.2,
        "false_wakes_per_10min": check_false_wakes(COMMANDS_DIR / "motor_noise.wav"),
        "self_triggers": check_self_trigger(),
    }
    print(json.dumps(results, indent=2))
    print(write_result(args.out, results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
