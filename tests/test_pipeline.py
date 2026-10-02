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


class SpyWake(FakeWakeWord):
    """FakeWakeWord that counts how often it is reset."""

    def __init__(self, scores, **kwargs):
        super().__init__(scores, **kwargs)
        self.resets = 0

    def reset(self) -> None:
        super().reset()
        self.resets += 1


def test_wakeword_is_reset_when_a_turn_finishes():
    wake = SpyWake([0.9])
    loop = VoiceLoop(wake, _segmenter(), FakeStt("turn left"))
    loop.process_frame(ZERO)  # wake -> open turn
    intent = _drive_to_intent(loop)
    assert intent is not None and intent.name == "turn_left"
    assert wake.resets >= 1


def test_wake_can_fire_again_after_a_turn():
    loop = VoiceLoop(FakeWakeWord([0.9], cooldown_s=0.0), _segmenter(), FakeStt("stop"))
    loop.process_frame(ZERO)
    _drive_to_intent(loop)
    assert loop.state == LISTENING
    assert loop.process_frame(ZERO) is None  # a fresh wake re-opens the turn
    assert loop.state == RECORDING


class ScriptedStt:
    """Returns one transcript per turn, repeating the last if exhausted."""

    def __init__(self, texts: list[str]):
        self.texts = list(texts) or [""]
        self.index = 0
        self.calls = 0

    def transcribe(self, pcm, sample_rate: int = 16000) -> str:
        self.calls += 1
        text = self.texts[min(self.index, len(self.texts) - 1)]
        self.index += 1
        return text


def test_turn_continues_after_a_lone_wake_word():
    flushes: list[int] = []
    loop = VoiceLoop(
        FakeWakeWord([0.9]),
        SpeechSegmenter(FakeVad(start_after=0, end_after=2), max_utterance_s=10.0),
        ScriptedStt(["Hey Rover", "go to the red cup"]),
        flush=lambda: flushes.append(1),
        max_turn_s=10.0,
    )
    assert loop.process_frame(ZERO) is None  # wake -> open turn
    assert loop.process_frame(ZERO) is None  # "Hey Rover" only -> keep waiting
    assert loop.state == RECORDING
    assert flushes  # the mic buffer was flushed between attempts

    intent = loop.process_frame(ZERO)  # the command itself
    assert intent is not None
    assert intent.name == "go_to"
    assert intent.target == "cup"
    assert loop.state == LISTENING


def test_turn_abandoned_if_no_command_before_deadline():
    stt = ScriptedStt([""])
    loop = VoiceLoop(
        FakeWakeWord([0.9]),
        SpeechSegmenter(FakeVad(start_after=999), onset_timeout_s=0.0),
        stt,
        max_turn_s=0.0,
    )
    assert loop.process_frame(ZERO) is None  # wake -> open turn (deadline already passed)
    assert loop.process_frame(ZERO) is None
    assert loop.state == LISTENING
    assert stt.calls == 0
    assert loop.false_wakes == 1


def test_no_speech_after_wake_counts_a_false_wake():
    loop = VoiceLoop(
        FakeWakeWord([0.9]),
        SpeechSegmenter(FakeVad(start_after=999), onset_timeout_s=0.0),
        ScriptedStt([""]),
    )
    assert loop.process_frame(ZERO) is None  # wake -> open turn
    assert loop.process_frame(ZERO) is None  # no speech -> timeout
    assert loop.state == LISTENING
    assert loop.false_wakes == 1


def test_transcript_callback_receives_text():
    seen: list[str] = []
    loop = VoiceLoop(FakeWakeWord([0.9]), _segmenter(), FakeStt("stop"), on_transcript=seen.append)
    loop.process_frame(ZERO)
    intent = _drive_to_intent(loop)
    assert intent is not None and intent.name == "stop"
    assert seen == ["stop"]
