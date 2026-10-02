#!/usr/bin/env python3
"""Manual hardware test: wake word -> transcript -> intent, with spoken ack.

Run on the Jetson with llama-server up (Gemma audio does the STT):

    .venv/bin/python hardware_tests/test_voice.py --seconds 60

Say **"Hey Rover, go to the red cup"**. A notify sound plays, the turn is
transcribed, the intent is printed, and the acknowledgement is spoken. This uses
the real mic, wake model, Silero VAD, Gemma STT and Piper — no camera/YOLO — so
it is a pure voice check.
"""

from __future__ import annotations

import argparse
import time

from rover.config import load_config
from rover.hal.audio import AlsaCapture, AlsaSpeaker
from rover.voice.pipeline import VoiceLoop
from rover.voice.stt import GemmaStt
from rover.voice.tts import PiperTts
from rover.voice.vad import SpeechSegmenter, load_silero
from rover.voice.wakeword import make_wakeword


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--verbose", action="store_true", help="print the live wake-word score")
    args = parser.parse_args()

    config = load_config()
    capture = AlsaCapture(config.audio)
    speaker = AlsaSpeaker(config.audio)
    wakeword = make_wakeword(config.voice.wakeword_model, threshold=config.voice.wakeword_threshold)
    segmenter = SpeechSegmenter(
        load_silero(
            end_silence_ms=config.voice.end_silence_ms, threshold=config.voice.vad_threshold
        ),
        max_utterance_s=config.voice.max_utterance_s,
    )
    stt = GemmaStt(config.planner.url)
    tts = PiperTts(config.voice.tts_voice, config.audio)

    def on_transcript(text: str) -> None:
        print(f"  transcript: {text!r}", flush=True)

    loop = VoiceLoop(
        wakeword,
        segmenter,
        stt,
        tts=tts,
        speaker=speaker,
        wake_sound=config.voice.wake_sound,
        on_transcript=on_transcript,
        flush=capture.flush,
        max_turn_s=config.voice.max_turn_s,
    )

    print("listening — say 'Hey Rover, go to the red cup'")
    deadline = time.time() + args.seconds
    turns = 0
    last_status = 0.0
    try:
        while time.time() < deadline:
            frame = capture.read_frame()
            if frame is None:
                break
            started = time.perf_counter()
            intent = loop.process_frame(frame)
            if args.verbose and time.time() - last_status > 1.0:
                last_status = time.time()
                print(
                    f"  [state={loop.state} wake_score={loop.wakeword.last_score:.2f}]",
                    flush=True,
                )
            if intent is not None:
                turns += 1
                elapsed = time.perf_counter() - started
                print(
                    f"  intent: {intent.name} target={intent.target} "
                    f"attrs={intent.attributes} (turn done, +{elapsed:.2f}s)",
                    flush=True,
                )
    finally:
        capture.close()
        tts.close()
        stt.close()
    print(f"done: {turns} turn(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
