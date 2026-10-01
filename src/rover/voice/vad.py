"""voice.vad: Silero VAD turn segmentation.

After the wake word, a turn runs until Silero reports ``end`` (configured to
800 ms of trailing silence) or the 10 s hard cap fires. Silero needs 512-sample
windows at 16 kHz, so 80 ms frames are buffered and split without losing
samples. The VAD is a plain ``numpy float32 -> dict | None`` callable, which
keeps this module torch-free and testable with a fake.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Callable

import numpy as np

VAD_WINDOW = 512  # Silero's 16 kHz window


class SileroVad:
    """Silero ``VADIterator`` wrapped as a resettable ``float32[512] -> dict`` callable."""

    def __init__(self, end_silence_ms: int = 800, threshold: float = 0.5, sample_rate: int = 16000):
        import torch
        from silero_vad import VADIterator, load_silero_vad

        self._torch = torch
        self._iterator = VADIterator(
            load_silero_vad(onnx=True),
            sampling_rate=sample_rate,
            threshold=threshold,
            min_silence_duration_ms=end_silence_ms,
        )

    def __call__(self, chunk: np.ndarray) -> dict | None:
        return self._iterator(self._torch.from_numpy(chunk), return_seconds=False)

    def reset(self) -> None:
        reset = getattr(self._iterator, "reset_states", None)
        if reset is not None:
            reset()


def load_silero(
    end_silence_ms: int = 800, threshold: float = 0.5, sample_rate: int = 16000
) -> SileroVad:
    """Build the Silero backend (tests inject a :class:`FakeVad` instead)."""
    return SileroVad(end_silence_ms=end_silence_ms, threshold=threshold, sample_rate=sample_rate)


class SpeechSegmenter:
    """Accumulates one utterance and reports its start/end."""

    def __init__(
        self,
        vad: Callable[[np.ndarray], dict | None],
        max_utterance_s: float = 10.0,
        onset_timeout_s: float = 4.0,
        sample_rate: int = 16000,
        preroll_frames: int = 4,
    ):
        self._vad = vad
        self.max_utterance_s = float(max_utterance_s)
        self.onset_timeout_s = float(onset_timeout_s)
        self.sample_rate = int(sample_rate)
        self._preroll: deque[np.ndarray] = deque(maxlen=max(1, int(preroll_frames)))
        self._pending = np.zeros(0, dtype=np.float32)
        self._turn: list[np.ndarray] = []
        self._started = False
        self._start_time = 0.0
        self._reset_time = 0.0
        self.reset()

    def reset(self) -> None:
        reset = getattr(self._vad, "reset", None)
        if reset is not None:
            reset()
        self._preroll.clear()
        self._pending = np.zeros(0, dtype=np.float32)
        self._turn = []
        self._started = False
        self._start_time = 0.0
        self._reset_time = time.monotonic()

    def process(self, frame: np.ndarray) -> str | None:
        """Feed one int16 frame; return ``"start"``, ``"end"``, ``"timeout"`` or None.

        The turn keeps a short pre-roll so the first phoneme is not clipped, and
        ``"timeout"`` means no speech arrived within ``onset_timeout_s``.
        """
        frame = np.asarray(frame, dtype=np.int16)
        if self._started:
            self._turn.append(frame)
        else:
            self._preroll.append(frame)
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
                self._turn = list(self._preroll)
            if "end" in event and self._started:
                return "end"

        if started_now:
            return "start"
        if self._started and (time.monotonic() - self._start_time) > self.max_utterance_s:
            return "end"
        if not self._started and (time.monotonic() - self._reset_time) > self.onset_timeout_s:
            return "timeout"
        return None

    def audio(self) -> np.ndarray:
        """The accumulated turn as int16 (empty if nothing was captured)."""
        if not self._turn:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(self._turn)


class FakeVad:
    """Scripted Silero stand-in: emits start/end at given 512-window counts."""

    def __init__(self, start_after: int = 0, end_after: int | None = None):
        self.start_after = start_after
        self.end_after = end_after
        self.calls = 0

    def reset(self) -> None:
        self.calls = 0

    def __call__(self, chunk: np.ndarray) -> dict | None:
        self.calls += 1
        if self.calls == self.start_after + 1:
            return {"start": self.calls}
        if self.end_after is not None and self.calls == self.end_after:
            return {"end": self.calls}
        return None
