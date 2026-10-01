"""Laptop-only tests for voice.pipeline: the wake -> VAD -> STT -> intent loop."""

from __future__ import annotations

import wave

import numpy as np

from rover.hal.audio import NullSpeaker
from rover.voice.pipeline import LISTENING, RECORDING, VoiceLoop
from rover.voice.stt import FakeStt
from rover.voice.tts import FakeTts
from rover.voice.vad import FakeVad, SpeechSegmenter
from rover.voice.wakeword import FakeWakeWord

ZERO = np.zeros(1280, dtype=np.int16)
ONE = np.ones(1280, dtype=np.int16)


def _segmenter(end_after: int = 3) -> SpeechSegmenter:
    return SpeechSegmenter(FakeVad(start_after=0, end_after=end_after), max_utterance_s=10.0)


def _drive_to_intent(loop: VoiceLoop, frames: int = 8):
    for _ in range(frames):
        intent = loop.process_frame(ONE)
        if intent is not None:
            return intent
    return None


def test_turn_produces_go_to_intent_with_attributes():
    loop = VoiceLoop(FakeWakeWord([0.0, 0.9]), _segmenter(), FakeStt("go to the red cup"))
    assert loop.process_frame(ZERO) is None  # score 0 -> still listening
    assert loop.process_frame(ZERO) is None  # wake -> open the turn
    assert loop.state == RECORDING

    intent = _drive_to_intent(loop)
    assert intent is not None
    assert intent.name == "go_to"
    assert intent.target == "cup"
    assert intent.attributes == ("red",)
    assert loop.state == LISTENING


def test_wake_plays_notify_sound_and_ack_is_spoken(tmp_path):
    sound = tmp_path / "notify.wav"
    with wave.open(str(sound), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(np.zeros(100, dtype=np.int16).tobytes())

    speaker = NullSpeaker()
    tts = FakeTts()
    loop = VoiceLoop(
        FakeWakeWord([0.9]),
        _segmenter(),
        FakeStt("turn left"),
        tts=tts,
        speaker=speaker,
        wake_sound=str(sound),
    )
    assert loop.process_frame(ZERO) is None  # wake fires immediately
    assert len(speaker.played) == 1  # notify sound

    intent = _drive_to_intent(loop)
    assert intent is not None and intent.name == "turn_left"
    assert tts.spoken == ["Turning left."]


def test_transcript_callback_receives_text():
    seen: list[str] = []
    loop = VoiceLoop(FakeWakeWord([0.9]), _segmenter(), FakeStt("stop"), on_transcript=seen.append)
    loop.process_frame(ZERO)
    intent = _drive_to_intent(loop)
    assert intent is not None and intent.name == "stop"
    assert seen == ["stop"]
