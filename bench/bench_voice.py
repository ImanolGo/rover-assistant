#!/usr/bin/env python3
"""Phase 4 bench: 20 commands through Silero VAD + Gemma STT + intents.

Each command is synthesized with Piper, resampled to 16 kHz, split into 80 ms
frames and pushed through the *real* :class:`SpeechSegmenter` (Silero) and
:class:`GemmaStt`; the resulting transcript is classified by
:func:`rover.voice.intents.classify`. Reports intent accuracy and the
end-of-speech -> intent latency (transcribe + classify).

Synthesized audio is used because the 20 real mic recordings (PLAN 1.10) are
still pending; replace ``COMMANDS`` audio with recordings when they exist.
"""

from __future__ import annotations

import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

from rover.voice.intents import classify
from rover.voice.stt import GemmaStt
from rover.voice.vad import SpeechSegmenter, load_silero

URL = "http://127.0.0.1:8080"
VOICE = "models/piper/en_US-lessac-medium.onnx"
PIPER = Path(sys.executable).with_name("piper")
FRAME = 1280
SAMPLE_RATE = 16000

# (spoken text, expected intent name, expected target or None)
COMMANDS: list[tuple[str, str, str | None]] = [
    ("stop", "stop", None),
    ("halt", "stop", None),
    ("go forward", "forward", None),
    ("move forward please", "forward", None),
    ("reverse", "backward", None),
    ("turn left", "turn_left", None),
    ("turn right", "turn_right", None),
    ("turn around", "turn_around", None),
    ("come here", "forward", None),
    ("go home", "return_home", None),
    ("what do you see", "describe", None),
    ("describe the scene", "describe", None),
    ("look around", "describe", None),
    ("follow me", "follow", None),
    ("is there a blue bus", "is_there", "bus"),
    ("is there a person", "is_there", "person"),
    ("go to the red cup", "go_to", "cup"),
    ("go to the blue bottle", "go_to", "bottle"),
    ("find my green backpack", "go_to", "backpack"),
    ("look for the chair", "go_to", "chair"),
]


def synthesize(text: str, out: Path) -> np.ndarray:
    """Piper -> 16 kHz int16 mono array."""
    if out.exists():
        out.unlink()
    subprocess.run(
        [str(PIPER), "-m", VOICE, "-f", str(out)],
        input=text,
        text=True,
        capture_output=True,
        check=True,
    )
    data, rate = sf.read(str(out), dtype="int16")
    if data.ndim > 1:
        data = data[:, 0]
    if rate != SAMPLE_RATE:
        count = int(len(data) * SAMPLE_RATE / rate)
        data = np.interp(np.linspace(0, len(data) - 1, count), np.arange(len(data)), data).astype(
            np.int16
        )
    return data.astype(np.int16)


def frames(samples: np.ndarray):
    for start in range(0, len(samples), FRAME):
        chunk = samples[start : start + FRAME]
        if len(chunk) < FRAME:
            chunk = np.pad(chunk, (0, FRAME - len(chunk)))
        yield chunk


def capture_turn(segmenter: SpeechSegmenter, audio: np.ndarray, silence_s: float = 1.5) -> bool:
    """Feed speech then trailing silence; True if VAD found a start and an end."""
    started = ended = False
    for frame in frames(audio):
        event = segmenter.process(frame)
        started = started or event == "start"
        if event == "end":
            ended = True
            break
    if not ended:
        for _ in range(int(silence_s * SAMPLE_RATE / FRAME)):
            if segmenter.process(np.zeros(FRAME, dtype=np.int16)) == "end":
                ended = True
                break
    return started and ended


def main() -> int:
    assert httpx.get(f"{URL}/health", timeout=10).json()["status"] == "ok"
    out_dir = Path("bench/results/raw/voice")
    out_dir.mkdir(parents=True, exist_ok=True)

    scratch = Path("bench/results/raw/voice")
    scratch.mkdir(parents=True, exist_ok=True)
    stem = scratch / "voice_bench_tmp"
    segmenter = SpeechSegmenter(load_silero(end_silence_ms=800), max_utterance_s=10.0)
    stt = GemmaStt(URL)

    rows = []
    for index, (text, want_name, want_target) in enumerate(COMMANDS):
        audio = synthesize(text, stem.with_suffix(".wav"))
        segmenter.reset()
        captured = capture_turn(segmenter, audio)
        turn = segmenter.audio()
        t0 = time.perf_counter()
        transcript = stt.transcribe(turn)
        intent = classify(transcript)
        latency = time.perf_counter() - t0
        name_ok = intent.name == want_name
        target_ok = want_target is None or intent.target == want_target
        rows.append(
            {
                "said": text,
                "transcript": transcript[:50],
                "intent": intent.name,
                "target": intent.target,
                "ok": bool(captured and name_ok and target_ok),
                "latency_s": round(latency, 3),
            }
        )
        print(rows[-1], flush=True)

    correct = sum(row["ok"] for row in rows)
    latencies = sorted(row["latency_s"] for row in rows)

    def pct(p: float) -> float:
        return latencies[min(len(latencies) - 1, int(p * len(latencies)))]

    results = {
        "commands": rows,
        "correct": correct,
        "total": len(rows),
        "accuracy": round(correct / len(rows), 3),
        "latency_p50_s": round(statistics.median(latencies), 3),
        "latency_p90_s": round(pct(0.9), 3),
        "note": "audio synthesized with Piper (mic recordings pending, PLAN 1.10)",
    }
    print(results)
    sys.path.insert(0, "bench")
    from common import write_result

    print(write_result("p4_voice_commands", results))
    stt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
