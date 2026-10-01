"""voice.pipeline: the wake -> VAD -> STT -> intent turn loop.

A small synchronous state machine fed one 80 ms frame at a time, so the listener
thread stays simple and tests can drive it with fakes. On wake it plays the
notify sound and opens a VAD turn; when the turn ends it transcribes,
classifies, speaks the acknowledgement (if the intent has one) and returns the
:class:`~rover.voice.intents.Intent`. Acting on the intent is the brain's job.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from rover.voice.intents import Intent, classify
from rover.voice.tts import play_wav

LISTENING = "LISTENING"
RECORDING = "RECORDING"


class VoiceLoop:
    """Frame-driven voice turn state machine."""

    def __init__(
        self,
        wakeword: Any,
        segmenter: Any,
        stt: Any,
        tts: Any = None,
        speaker: Any = None,
        wake_sound: str | None = None,
        on_transcript: Callable[[str], None] | None = None,
    ):
        self.wakeword = wakeword
        self.segmenter = segmenter
        self.stt = stt
        self.tts = tts
        self.speaker = speaker
        self.wake_sound = wake_sound
        self.on_transcript = on_transcript
        self.state = LISTENING

    def _begin_turn(self) -> None:
        self.segmenter.reset()
        if self.speaker is not None and self.wake_sound:
            play_wav(self.speaker, self.wake_sound)
        self.state = RECORDING

    def _finish_turn(self, audio: np.ndarray) -> Intent:
        self.state = LISTENING
        text = self.stt.transcribe(audio)
        if self.on_transcript is not None:
            self.on_transcript(text)
        intent = classify(text)
        if self.tts is not None and intent.response:
            self.tts.say(intent.response)
        return intent

    def process_frame(self, frame: np.ndarray) -> Intent | None:
        """Feed one 80 ms int16 frame; return an Intent when a turn completes."""
        if self.state == LISTENING:
            if self.wakeword.process(frame):
                self._begin_turn()
            return None
        if self.segmenter.process(frame) == "end":
            return self._finish_turn(self.segmenter.audio())
        return None
