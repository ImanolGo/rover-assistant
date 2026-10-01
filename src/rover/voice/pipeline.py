"""voice.pipeline: the wake -> VAD -> STT -> intent turn loop.

A small synchronous state machine fed one 80 ms frame at a time. On wake it
plays the notify sound and opens a VAD turn; when a turn ends it transcribes,
classifies, and speaks the acknowledgement (if any). The brain acts on the
returned :class:`~rover.voice.intents.Intent`.

Two robustness rules learned on hardware:

- If a turn ends but the transcript is only the wake phrase ("Hey Rover"), the
  turn stays open (up to ``max_turn_s``) for the command, so
  "Hey Rover" [pause] "go to the cup" works.
- ``flush`` is called whenever listening resumes, discarding audio the mic
  buffered while STT/TTS ran, so the next wake word is heard live.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from rover.voice.intents import Intent, classify, strip_wake_phrase
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
        flush: Callable[[], None] | None = None,
        max_turn_s: float = 8.0,
    ):
        self.wakeword = wakeword
        self.segmenter = segmenter
        self.stt = stt
        self.tts = tts
        self.speaker = speaker
        self.wake_sound = wake_sound
        self.on_transcript = on_transcript
        self.flush = flush
        self.max_turn_s = float(max_turn_s)
        self.state = LISTENING
        self._turn_deadline = 0.0

    def _flush(self) -> None:
        if self.flush is not None:
            self.flush()

    def _begin_turn(self) -> None:
        self.segmenter.reset()
        if self.speaker is not None and self.wake_sound:
            play_wav(self.speaker, self.wake_sound)
        self._turn_deadline = time.monotonic() + self.max_turn_s
        self.state = RECORDING

    def _listen_again(self) -> None:
        """Wake-only/empty turn: keep waiting for the command, but reset audio."""
        self.segmenter.reset()
        self._flush()

    def _resume_listening(self) -> None:
        """Return to idle: clear VAD + wake state and drop buffered audio.

        Resetting the wake model matters: its streaming context still holds the
        wake word, so without it the detector re-fires the instant listening
        resumes (no fresh chime, looks "stuck").
        """
        self.state = LISTENING
        self.segmenter.reset()
        reset = getattr(self.wakeword, "reset", None)
        if reset is not None:
            reset()
        self._flush()

    def _transcribe(self, audio: np.ndarray) -> str:
        text = self.stt.transcribe(audio)
        if self.on_transcript is not None:
            self.on_transcript(text)
        return text

    def _finish(self, intent: Intent) -> Intent:
        if self.tts is not None and intent.response:
            self.tts.say(intent.response)
        self._resume_listening()
        return intent

    def process_frame(self, frame: np.ndarray) -> Intent | None:
        """Feed one 80 ms int16 frame; return an Intent when a command completes."""
        if self.state == LISTENING:
            if self.wakeword.process(frame):
                self._begin_turn()
            return None

        event = self.segmenter.process(frame)
        if event == "end":
            clean = strip_wake_phrase(self._transcribe(self.segmenter.audio()))
            if not clean and time.monotonic() < self._turn_deadline:
                self._listen_again()  # the wake word only; wait for the command
                return None
            if not clean:
                self._resume_listening()
                return None
            return self._finish(classify(clean))
        if event == "timeout":
            if time.monotonic() < self._turn_deadline:
                self._listen_again()
                return None
            self._resume_listening()
            return None
        if time.monotonic() > self._turn_deadline:
            self._resume_listening()
        return None
