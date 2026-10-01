"""voice.tts: persistent Piper process, spoken sentence by sentence.

One ``piper --output-raw`` process is kept alive (load once, many sentences).
The reply is split into sentences and each is written to Piper's stdin; raw
16-bit PCM is read from stdout until it goes idle (Piper synthesizes faster than
real time, so a short silent gap means the sentence is done) and streamed to the
speaker, which lets speech start before the whole reply is synthesized.
"""

from __future__ import annotations

import re
import select
import subprocess
from typing import Any, Protocol

import numpy as np

_SENTENCE = re.compile(r"[^.!?]+[.!?]*")
PIPER_RATE = 22050  # en_US-lessac-medium


def split_sentences(text: str) -> list[str]:
    """Split a reply on . ! ? so playback can start on the first sentence."""
    sentences = [part.strip() for part in _SENTENCE.findall(text or "") if part.strip()]
    if sentences:
        return sentences
    text = (text or "").strip()
    return [text] if text else []


def read_pcm_until_idle(stream: Any, timeout_s: float = 0.6, chunk: int = 4096) -> bytes:
    """Read raw bytes until the stream stalls for ``timeout_s`` or closes."""
    data = bytearray()
    while True:
        ready, _, _ = select.select([stream], [], [], timeout_s)
        if not ready:
            break
        block = stream.read(chunk)
        if not block:
            break
        data.extend(block)
    return bytes(data)


def play_wav(speaker: Any, path: str) -> None:
    """Play a WAV file through the speaker HAL (wake/notify sounds)."""
    import soundfile as sf

    data, rate = sf.read(str(path), dtype="int16")
    if getattr(data, "ndim", 1) > 1:
        data = data[:, 0]
    speaker.play(np.asarray(data, dtype=np.int16), int(rate))


class Tts(Protocol):
    """Speak text out loud (blocking until the speaker has been fed)."""

    def say(self, text: str) -> None: ...

    def close(self) -> None: ...


class PiperTts:
    """A long-lived ``piper --output-raw`` subprocess feeding the speaker HAL."""

    def __init__(
        self,
        model_path: str,
        speaker: Any,
        rate: int = PIPER_RATE,
        binary: str = "piper",
        idle_s: float = 0.6,
    ):
        self.speaker = speaker
        self.rate = int(rate)
        self.idle_s = float(idle_s)
        self._proc = subprocess.Popen(
            [binary, "-m", str(model_path), "--output-raw"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def _synthesize(self, sentence: str) -> np.ndarray:
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write((sentence + "\n").encode())
        self._proc.stdin.flush()
        raw = read_pcm_until_idle(self._proc.stdout, self.idle_s)
        if len(raw) % 2:
            raw = raw[:-1]
        return np.frombuffer(raw, dtype=np.int16).copy()

    def say(self, text: str) -> None:
        for sentence in split_sentences(text):
            samples = self._synthesize(sentence)
            if samples.size:
                self.speaker.play(samples, self.rate)

    def close(self) -> None:
        if self._proc.stdin is not None:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
        try:
            self._proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self._proc.kill()


class FakeTts:
    """Records what would have been spoken (laptop tests / sim)."""

    def __init__(self) -> None:
        self.spoken: list[str] = []

    def say(self, text: str) -> None:
        self.spoken.append(text)

    def close(self) -> None: ...
