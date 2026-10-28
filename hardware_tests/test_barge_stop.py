#!/usr/bin/env python3
"""Manual hardware test: spoken "stop" (barge-in, no wake word) while the wheels spin.

Run on the Jetson with the robot ON A BOX (wheels off the ground) and ``rover``
stopped (it owns the serial port):

    .venv/bin/python hardware_tests/test_barge_stop.py --yes --trials 5

Each trial spins the wheels in place, plays the wake chime as the cue, and waits for
the operator to say "stop". The same pieces as ``rover`` (Silero segmenter, Moonshine
stream, ``BargeInStop``) stop the wheels. Reports per trial what was heard and the
time from the chime to the stop; a trial with no stop within ``--timeout`` is a miss
(the wheels stop anyway at the timeout).
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import wave
from pathlib import Path

import numpy as np

from rover.config import load_config
from rover.hal.audio import AlsaCapture, AlsaSpeaker
from rover.hal.rover import RoverConfig, SerialRover
from rover.voice.intents import vocabulary
from rover.voice.pipeline import BargeInStop
from rover.voice.stt import make_stt
from rover.voice.tts import play_wav
from rover.voice.vad import SpeechSegmenter, load_silero


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--speed", type=float, default=0.30)
    parser.add_argument("--timeout", type=float, default=6.0, help="seconds after the cue")
    parser.add_argument("--yes", action="store_true", help="confirm wheels are off the ground")
    parser.add_argument("--save", default=None, help="dir for per-trial WAV + event timeline")
    args = parser.parse_args()
    if not args.yes:
        raise SystemExit("Refusing to drive: pass --yes only with the wheels raised.")

    config = load_config()
    voice = config.voice
    capture = AlsaCapture(config.audio)
    speaker = AlsaSpeaker(config.audio)
    rover = SerialRover(RoverConfig(port=config.rover.port, speed_cap=args.speed))
    stt = make_stt(
        voice, config.planner.url, keyterms=vocabulary(config.perception.target_classes_h_stop)
    )
    streams: list = []

    def new_stream():
        streams.append(stt.new_stream())
        return streams[-1]

    stopped = threading.Event()
    barge = BargeInStop(
        SpeechSegmenter(
            load_silero(end_silence_ms=voice.end_silence_ms, threshold=voice.vad_threshold),
            max_utterance_s=voice.max_utterance_s,
            onset_timeout_s=voice.no_speech_timeout_s,
        ),
        new_stream,
        on_stop=lambda: (rover.stop(), stopped.set()),
    )

    results = []
    try:
        time.sleep(1.0)
        for trial in range(1, args.trials + 1):
            stopped.clear()
            barge.reset()
            streams.clear()
            spinning = threading.Event()
            spinning.set()

            def spin() -> None:
                # Refresh within the 0.5 s watchdog; the stop callback wins the next write.
                while spinning.is_set() and not stopped.is_set():
                    rover.drive(-args.speed, args.speed)
                    time.sleep(0.1)

            spinner = threading.Thread(target=spin, daemon=True)
            spinner.start()
            time.sleep(1.5)  # wheels up to speed, motor noise present
            capture.flush()
            play_wav(speaker, voice.wake_sound, wait=False)
            cue = time.monotonic()
            print(f"[{trial}/{args.trials}] chime -> say 'stop'", flush=True)
            audio, events = [], []
            while not stopped.is_set() and time.monotonic() - cue < args.timeout:
                frame = capture.read_frame()
                if frame is not None:
                    audio.append(frame)
                    had = len(streams)
                    barge.process(frame)
                    t = round(time.monotonic() - cue, 3)
                    if len(streams) > had:
                        events.append((t, "vad_start", ""))
                    if (
                        streams
                        and (text := streams[-1].partial())
                        and (not events or events[-1][2] != text)
                    ):
                        events.append((t, "partial", text))
            elapsed = time.monotonic() - cue
            spinning.clear()
            spinner.join(timeout=0.5)
            rover.stop()
            heard = streams[-1].partial() if streams else ""
            ok = stopped.is_set()
            results.append((ok, elapsed if ok else None, heard))
            if args.save:
                out = Path(args.save)
                out.mkdir(parents=True, exist_ok=True)
                with wave.open(str(out / f"trial_{trial:02d}.wav"), "wb") as handle:
                    handle.setnchannels(1)
                    handle.setsampwidth(2)
                    handle.setframerate(config.audio.mic_rate)
                    handle.writeframes(np.concatenate(audio).astype(np.int16).tobytes())
                row = {"trial": trial, "stopped": ok, "elapsed_s": round(elapsed, 3)}
                (out / f"trial_{trial:02d}.json").write_text(json.dumps({**row, "events": events}))
            outcome = f"STOP  chime->stop {elapsed:.2f}s" if ok else f"miss  ({args.timeout:.0f}s)"
            print(f"   {outcome}  heard={heard!r}", flush=True)
            time.sleep(2.0)
    finally:
        rover.stop()
        rover.close()
        capture.close()
        speaker.close()
        stt.close()

    hits = [r for r in results if r[0]]
    times = sorted(r[1] for r in hits)
    median = times[len(times) // 2] if times else None
    print(f"stopped {len(hits)}/{len(results)}; chime->stop median {median}")
    return 0 if len(hits) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
