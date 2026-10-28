"""voice.tts: in-process Piper, spoken sentence by sentence.

The ``piper-tts`` Python API is used directly (no subprocess, no idle-timeout
guessing about where a sentence ends): the voice is loaded once, its sample rate
comes from the model's config, and each synthesized sentence is written to the
speaker in ~80 ms blocks, keeping at most ``MAX_AHEAD_S`` queued ahead of what
is audible. A cancel flag is checked between blocks, so ``stop`` silences the
robot within ~0.25 s. ``say()`` and :func:`play_wav` return when the sound has
**finished playing** (``speaker.drain``), not when it was written — the
``speaking`` flag and the post-chime flush depend on that.

The playback stream is opened at the *voice's* sample rate (mono), so no numpy
resampling is needed on the audio path — PulseAudio/aplay handles the device
rate. This replaces the old ``piper --output-raw`` subprocess whose 0.6 s idle
read added 0.6 s of dead air per sentence and could truncate under load.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import replace
from typing import Any, Iterable, Protocol

import numpy as np

_SENTENCE = re.compile(r"[^.!?]+[.!?]*")
BLOCK_S = 0.08  # playback block size, for prompt cancellation
MAX_AHEAD_S = 0.25  # audio queued beyond what is audible; bounds the cancel delay


def _drain(speaker: Any, cancelled: Any = None) -> None:
    """Wait for the speaker to go silent (fakes without a clock return at once)."""
    drain = getattr(speaker, "drain", None)
    if drain is not None:
        drain(cancelled)


def split_sentences(text: str) -> list[str]:
    """Split a reply on . ! ? so playback can start on the first sentence."""
    sentences = [part.strip() for part in _SENTENCE.findall(text or "") if part.strip()]
    if sentences:
        return sentences
    text = (text or "").strip()
    return [text] if text else []


def play_wav(speaker: Any, path: str) -> None:
    """Play a WAV file through the speaker HAL (wake/notify sounds)."""
    import soundfile as sf

    data, rate = sf.read(str(path), dtype="int16")
    if getattr(data, "ndim", 1) > 1:
        data = data[:, 0]
    speaker.play(np.asarray(data, dtype=np.int16), int(rate))
    _drain(speaker)


class Tts(Protocol):
    """Speak text out loud; ``cancel`` silences it promptly."""

    def say(self, text: str) -> None: ...

    def cancel(self) -> None: ...

    def close(self) -> None: ...


class PiperTts:
    """In-process Piper voice, sentence by sentence, cancellable."""

    def __init__(
        self,
        model_path: str,
        audio_config: Any,
        speaker: Any = None,
        voice: Any = None,
    ):
        self.voice = voice if voice is not None else self._load(model_path)
        self.rate = int(self.voice.config.sample_rate)  # from the model, not hardcoded
        if speaker is None:
            from rover.hal.audio import AlsaSpeaker

            speaker = AlsaSpeaker(replace(audio_config, speaker_rate=self.rate, speaker_channels=1))
        self.speaker = speaker
        self._block = max(1, int(self.rate * BLOCK_S))
        self._cancel = threading.Event()

    @staticmethod
    def _load(model_path: str) -> Any:
        from piper import PiperVoice

        return PiperVoice.load(str(model_path))

    def cancel(self) -> None:
        self._cancel.set()

    def _synthesize(self, sentence: str) -> Iterable[bytes]:
        return (chunk.audio_int16_bytes for chunk in self.voice.synthesize(sentence))

    def _wait_for_room(self) -> None:
        buffered = getattr(self.speaker, "buffered_s", None)
        if buffered is None:
            return
        while buffered() > MAX_AHEAD_S and not self._cancel.is_set():
            time.sleep(0.01)

    def say(self, text: str) -> None:
        """Speak ``text``; returns when it has finished playing or was cancelled."""
        self._cancel.clear()
        for sentence in split_sentences(text):
            if self._cancel.is_set():
                return
            for raw in self._synthesize(sentence):
                samples = np.frombuffer(raw, dtype=np.int16)
                for start in range(0, len(samples), self._block):
                    self._wait_for_room()
                    if self._cancel.is_set():
                        return
                    self.speaker.play(samples[start : start + self._block], self.rate)
        _drain(self.speaker, self._cancel.is_set)

    def close(self) -> None:
        speaker_close = getattr(self.speaker, "close", None)
        if speaker_close is not None:
            speaker_close()


class FakeTts:
    """Records what would have been spoken (laptop tests / sim)."""

    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.cancelled = False

    def say(self, text: str) -> None:
        self.spoken.append(text)

    def cancel(self) -> None:
        self.cancelled = True

    def close(self) -> None: ...
