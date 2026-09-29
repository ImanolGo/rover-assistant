"""Laptop-only tests for hal.audio: device parsing, normalize, resample, WAV fake."""

from __future__ import annotations

import wave

import numpy as np
import pytest

from rover.hal.audio import (
    AudioConfig,
    NullSpeaker,
    WavSource,
    find_device,
    normalize_int16,
    parse_alsa_devices,
    to_speaker_format,
)

ARECORD = """**** List of CAPTURE Hardware Devices ****
card 1: Device [USB PnP Sound Device], device 0: USB Audio [USB Audio]
  Subdevices: 1/1
"""

APLAY = """**** List of PLAYBACK Hardware Devices ****
card 0: UACDemoV1 [UACDemoV1.0], device 0: USB Audio [USB Audio]
  Subdevices: 1/1
"""


def test_parse_and_find_mic():
    devices = parse_alsa_devices(ARECORD)
    assert devices == [
        {"card": 1, "device": 0, "name": "USB PnP Sound Device", "subdevice": "USB Audio"}
    ]
    assert find_device(ARECORD, "USB PnP Sound Device") == "plughw:1,0"


def test_find_speaker_by_name():
    assert find_device(APLAY, "UACDemoV1.0") == "plughw:0,0"
    assert find_device(APLAY, "nonexistent") is None


def test_normalize_boosts_quiet_but_caps_gain():
    quiet = np.array([0, 1000, -1000], dtype=np.int16)
    boosted = normalize_int16(quiet)
    assert boosted.max() == 15000  # clipped by max_gain=15


def test_normalize_leaves_silence_and_loud_untouched():
    silence = np.zeros(4, dtype=np.int16)
    assert np.array_equal(normalize_int16(silence), silence)

    loud = np.array([0, 20000, -20000], dtype=np.int16)
    assert np.array_equal(normalize_int16(loud), loud)


def test_to_speaker_format_resamples_and_fans_out_stereo():
    config = AudioConfig()
    samples = (np.sin(np.linspace(0, 20, 1600)) * 20000).astype(np.int16)
    out = to_speaker_format(samples, 16000, config)
    assert out.dtype == np.float32
    assert out.shape == (4800, 2)  # 16k -> 48k, mono -> stereo
    assert float(np.abs(out).max()) <= 0.95


def test_to_speaker_format_peak_limits():
    config = AudioConfig()
    loud = np.ones(160, dtype=np.float32) * 2.0
    out = to_speaker_format(loud, 16000, config)
    assert float(np.abs(out).max()) == pytest.approx(0.95)


def _write_wav(path, samples: np.ndarray, rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.astype(np.int16).tobytes())


def test_wav_source_reads_full_frames_and_loops(tmp_path):
    path = tmp_path / "clip.wav"
    _write_wav(path, np.arange(1280, dtype=np.int16), rate=16000)

    source = WavSource(path, frame_samples=1280, loop=True)
    first = source.read_frame()
    second = source.read_frame()
    source.close()
    assert first is not None and second is not None
    assert first.shape == (1280,)
    assert np.array_equal(first, second)  # looped back to the start


def test_wav_source_without_loop_returns_none_at_end(tmp_path):
    path = tmp_path / "short.wav"
    _write_wav(path, np.arange(640, dtype=np.int16), rate=16000)

    source = WavSource(path, frame_samples=1280, loop=False)
    assert source.read_frame() is None
    source.close()


def test_null_speaker_records():
    speaker = NullSpeaker()
    speaker.play(np.ones(10, dtype=np.float32), 16000)
    assert len(speaker.played) == 1
    speaker.close()
