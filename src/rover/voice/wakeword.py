"""voice.wakeword: the "Hey Rover" openWakeWord gate with a debounce cooldown.

Loads the legacy custom model (``models/wake_word/hey_roe_ver.onnx``) with the
ONNX framework. openWakeWord's feature models ship inside its package, so
nothing is fetched at runtime. One utterance must not fire twice, hence the
cooldown. The cooldown logic lives in :class:`WakeWord` so a fake score source
can exercise it without loading ONNX.
"""

from __future__ import annotations

import time
from typing import Any

# openWakeWord consumes 1280-sample int16 chunks (80 ms at 16 kHz).
WAKEWORD_CHUNK = 1280


class WakeWord:
    """Threshold + cooldown gate over a per-chunk score."""

    def __init__(self, threshold: float = 0.5, cooldown_s: float = 2.0):
        self.threshold = float(threshold)
        self.cooldown_s = float(cooldown_s)
        self.last_score = 0.0
        self._last_fire = float("-inf")

    def score(self, frame: Any) -> float:
        raise NotImplementedError

    def reset(self) -> None:
        """Clear residual detection state after a turn (cooldown is kept)."""
        self.last_score = 0.0

    def process(self, frame: Any) -> bool:
        """True exactly when the score crosses the threshold, once per cooldown."""
        score = self.score(frame)
        self.last_score = score
        now = time.monotonic()
        if score >= self.threshold and (now - self._last_fire) >= self.cooldown_s:
            self._last_fire = now
            return True
        return False


class OpenWakeWord(WakeWord):
    """Real detector backed by openWakeWord's ONNX model."""

    def __init__(self, model_path: str, threshold: float = 0.5, cooldown_s: float = 2.0):
        super().__init__(threshold, cooldown_s)
        from openwakeword.model import Model

        self._model = Model(wakeword_models=[str(model_path)], inference_framework="onnx")
        self.keyword = next(iter(self._model.models))

    def score(self, frame: Any) -> float:
        predictions = self._model.predict(frame)
        if self.keyword in predictions:
            return float(predictions[self.keyword])
        return float(max(predictions.values(), default=0.0))

    def reset(self) -> None:
        super().reset()
        # Drop openWakeWord's streaming context, or the wake word lingers in its
        # internal buffer and re-triggers the moment listening resumes.
        reset = getattr(self._model, "reset", None)
        if reset is not None:
            reset()


class FakeWakeWord(WakeWord):
    """Scripted scores for laptop tests/sim; cycles through ``scores``."""

    def __init__(self, scores: list[float] | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self._scores = list(scores or [])
        self._index = 0

    def score(self, frame: Any) -> float:
        if not self._scores:
            return 0.0
        value = self._scores[self._index % len(self._scores)]
        self._index += 1
        return float(value)

    def reset(self) -> None:
        super().reset()
        self._index = 0


def make_wakeword(model_path: str, threshold: float = 0.5, cooldown_s: float = 2.0) -> WakeWord:
    """Build the real openWakeWord detector (tests inject :class:`FakeWakeWord`)."""
    return OpenWakeWord(model_path, threshold=threshold, cooldown_s=cooldown_s)
