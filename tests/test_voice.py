"""Laptop-only tests for voice: wakeword cooldown, VAD turn, STT, TTS helpers."""

from __future__ import annotations

import json
import os
import wave

import httpx
import numpy as np

from rover.hal.audio import NullSpeaker, drain_fd
from rover.voice.stt import FakeStt, GemmaStt, pcm16_to_wav_bytes
from rover.voice.tts import (
    FakeTts,
    default_piper_binary,
    play_wav,
    read_pcm_until_idle,
    split_sentences,
)
from rover.voice.vad import FakeVad, SpeechSegmenter
from rover.voice.wakeword import FakeWakeWord

# --- wakeword ---------------------------------------------------------------


def test_wakeword_fires_once_per_cooldown():
    detector = FakeWakeWord([0.1, 0.9, 0.9, 0.1, 0.95], threshold=0.5, cooldown_s=100.0)
    results = [detector.process(np.zeros(1280, dtype=np.int16)) for _ in range(5)]
    assert results == [False, True, False, False, False]


def test_wakeword_fires_again_after_cooldown_disabled():
    detector = FakeWakeWord([0.9, 0.9], threshold=0.5, cooldown_s=0.0)
    assert detector.process(np.zeros(1280, dtype=np.int16)) is True
    assert detector.process(np.zeros(1280, dtype=np.int16)) is True


# --- VAD --------------------------------------------------------------------


def test_vad_segmenter_reports_start_and_end_and_accumulates_audio():
    segmenter = SpeechSegmenter(FakeVad(start_after=0, end_after=3), max_utterance_s=10.0)
    events = []
    for _ in range(4):
        event = segmenter.process(np.ones(1280, dtype=np.int16))
        if event:
            events.append(event)
            if event == "end":
                break
    assert events == ["start", "end"]
    assert segmenter.audio().dtype == np.int16
    assert segmenter.audio().size >= 2 * 1280


def test_vad_segmenter_hard_cap_ends_the_turn():
    segmenter = SpeechSegmenter(FakeVad(start_after=0, end_after=None), max_utterance_s=0.0)
    assert segmenter.process(np.ones(1280, dtype=np.int16)) == "start"
    assert segmenter.process(np.ones(1280, dtype=np.int16)) == "end"


def test_vad_onset_timeout_when_no_speech_arrives():
    segmenter = SpeechSegmenter(FakeVad(start_after=999), onset_timeout_s=0.0)
    assert segmenter.process(np.ones(1280, dtype=np.int16)) == "timeout"
    assert segmenter.audio().size == 0


# --- STT --------------------------------------------------------------------


def test_pcm16_to_wav_bytes_is_a_valid_wav():
    pcm = (np.sin(np.linspace(0, 6.28, 1600)) * 1000).astype(np.int16)
    import io

    with wave.open(io.BytesIO(pcm16_to_wav_bytes(pcm, 16000))) as handle:
        assert handle.getframerate() == 16000
        assert handle.getnchannels() == 1
        assert handle.getnframes() == 1600


def test_gemma_stt_parses_content_and_strips_quotes():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        content = body["messages"][0]["content"]
        assert content[0]["type"] == "input_audio"
        return httpx.Response(
            200, json={"choices": [{"message": {"content": ' "go to the red cup" '}}]}
        )

    stt = GemmaStt("http://test", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert stt.transcribe(np.zeros(1600, dtype=np.int16)) == "go to the red cup"


def test_gemma_stt_falls_back_to_reasoning_content():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "", "reasoning_content": "hello"}}]}
        )

    stt = GemmaStt("http://test", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert stt.transcribe(np.zeros(1600, dtype=np.int16)) == "hello"


def test_fake_stt_counts_calls():
    stt = FakeStt("turn left")
    assert stt.transcribe(np.zeros(10, dtype=np.int16)) == "turn left"
    assert stt.calls == 1


# --- TTS --------------------------------------------------------------------


def test_split_sentences():
    assert split_sentences("Looking for the cup. I am on my way!") == [
        "Looking for the cup.",
        "I am on my way!",
    ]
    assert split_sentences("no punctuation here") == ["no punctuation here"]
    assert split_sentences("   ") == []


def test_read_pcm_until_idle_returns_available_bytes_then_stops():
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, b"\x01\x02\x03\x04")
        stream = os.fdopen(read_fd, "rb", buffering=0)
        try:
            data = read_pcm_until_idle(stream, timeout_s=0.2)
        finally:
            stream.close()
    finally:
        os.close(write_fd)
    assert data == b"\x01\x02\x03\x04"


def test_play_wav_uses_the_wav_sample_rate(tmp_path):
    path = tmp_path / "tone.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(np.zeros(100, dtype=np.int16).tobytes())
    speaker = NullSpeaker()
    play_wav(speaker, str(path))
    assert len(speaker.played) == 1
    assert speaker.played[0].shape[0] == 100


def test_drain_fd_discards_buffered_bytes_without_blocking():
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, b"x" * 300)
        assert drain_fd(read_fd) == 300
        assert drain_fd(read_fd) == 0  # nothing left, does not block
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_default_piper_binary_points_at_the_venv_cli():
    assert default_piper_binary().endswith("piper")


def test_fake_tts_records_speech():
    tts = FakeTts()
    tts.say("Hello there.")
    assert tts.spoken == ["Hello there."]
    tts.close()
