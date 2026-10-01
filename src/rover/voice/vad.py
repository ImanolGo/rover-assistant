"""voice.vad: Silero VAD turn segmentation.

After the wake word, a turn runs until Silero reports ``end`` (configured to
800 ms of trailing silence) or the 10 s hard cap fires. Silero needs 512-sample
windows at 16 kHz, so 80 ms frames are buffered and split without losing
samples. The VAD is a plain ``numpy float32 -> dict | None`` callable, which
keeps this module torch-free and testable with a fake.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

VAD_WINDOW = 512  # Silero's 16 kHz window


def load_silero(
    end_silence_ms: int = 800, threshold: float = 0.5, sample_rate: int = 16000
) -> Callable[[np.ndarray], dict | None]:
    """Return a ``float32[512] -> {start|end} | None`` callable backed by Silero."""
    import torch
    from silero_vad import VADIterator, load_silero_vad

    iterator = VADIterator(
        load_silero_vad(onnx=True),
        sampling_rate=sample_rate,
        threshold=threshold,
        min_silence_duration_ms=end_silence_ms,
    )
    return lambda chunk: iterator(torch.from_numpy(chunk), return_seconds=False)


class SpeechSegmenter:
    """Accumulates one utterance and reports its start/end."""

    def __init__(
        self,
        vad: Callable[[np.ndarray], dict | None],
        max_utterance_s: float = 10.0,
        sample_rate: int = 16000,
    ):
        self._vad = vad
        self.max_utterance_s = float(max_utterance_s)
        self.sample_rate = int(sample_rate)
        self._pending = np.zeros(0, dtype=np.float32)
        self._frames: list[np.ndarray] = []
        self._started = False
        self._start_time = 0.0

    def reset(self) -> None:
        self._pending = np.zeros(0, dtype=np.float32)
        self._frames = []
        self._started = False
        self._start_time = 0.0

    def process(self, frame: np.ndarray) -> str | None:
        """Feed one int16 frame; return ``"start"``, ``"end"`` or ``None``."""
        frame = np.asarray(frame, dtype=np.int16)
        self._frames.append(frame)
        self._pending = np.concatenate([self._pending, frame.astype(np.float32) / 32768.0])

        started_now = False
        while len(self._pending) >= VAD_WINDOW:
            window = self._pending[:VAD_WINDOW]
            self._pending = self._pending[VAD_WINDOW:]
            event = self._vad(window)
            if not event:
                continue
            if "start" in event and not self._started:
                self._started = True
                self._start_time = time.monotonic()
                started_now = True
            if "end" in event and self._started:
                return "end"

        if started_now:
            return "start"
        if self._started and (time.monotonic() - self._start_time) > self.max_utterance_s:
            return "end"
        return None

    def audio(self) -> np.ndarray:
        """The accumulated turn as int16 (empty if nothing was captured)."""
        if not self._frames:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(self._frames)


class FakeVad:
    """Scripted Silero stand-in: emits start/end at given 512-window counts."""

    def __init__(self, start_after: int = 0, end_after: int | None = None):
        self.start_after = start_after
        self.end_after = end_after
        self.calls = 0

    def __call__(self, chunk: np.ndarray) -> dict | None:
        self.calls += 1
        if self.calls == self.start_after + 1:
            return {"start": self.calls}
        if self.end_after is not None and self.calls == self.end_after:
            return {"end": self.calls}
        return None
