"""voice.pipeline: the wake -> VAD -> STT -> intent turn loop.

Two layers:

- :class:`VoiceLoop` is the listener-side state machine. It is fed one 80 ms
  frame at a time, runs the wake word and VAD, and finishes a turn either inline
  (tests, hardware: no ``worker``) or by handing the turn audio to a
  :class:`ThreadedWorker` so the listener never blocks on STT or TTS.
- :class:`AudioQueue` + :class:`CapturePump` put a bounded buffer between the
  mic and the listener: the capture thread always drains arecord (so ALSA never
  overruns), and when the listener is behind the oldest frames are dropped and
  counted rather than letting stale audio swallow the next wake word.

Continuation note: the inline path keeps a turn open when a transcript is only
the wake phrase ("Hey Rover" ... pause ... command). The threaded path does not
(it would need STT in the listener); say "Hey Rover, <command>" in one breath,
or wake again. Barge-in / stop-without-wake is deferred (C3, C6).
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Callable

import numpy as np

from rover.voice.intents import Intent, classify, is_stop, strip_wake_phrase
from rover.voice.tts import play_wav

LISTENING = "LISTENING"
RECORDING = "RECORDING"


class AudioQueue:
    """Bounded frame queue: drops the *oldest* frame when full and counts drops."""

    def __init__(self, maxsize: int = 25):
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self.dropped = 0

    def put(self, frame: np.ndarray) -> None:
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            try:
                self._queue.get_nowait()  # drop the oldest
                self.dropped += 1
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                pass

    def get(self, timeout: float = 0.5) -> np.ndarray | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> int:
        """Discard everything buffered (after a turn, before listening again)."""
        count = 0
        while True:
            try:
                self._queue.get_nowait()
                count += 1
            except queue.Empty:
                return count

    def qsize(self) -> int:
        return self._queue.qsize()


class CapturePump:
    """Thread that reads a capture source into an :class:`AudioQueue`."""

    def __init__(self, source: Any, audio_queue: AudioQueue, name: str = "capture"):
        self.source = source
        self.queue = audio_queue
        self.name = name
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name=self.name, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            frame = self.source.read_frame()
            if frame is None:
                time.sleep(0.005)
                continue
            self.queue.put(frame)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


class ThreadedWorker:
    """Serial STT -> intent -> TTS worker. One pending turn max; never blocks caller."""

    def __init__(
        self,
        stt: Any,
        tts: Any = None,
        on_intent: Callable[[Intent], None] | None = None,
        on_transcript: Callable[[str], None] | None = None,
        on_empty: Callable[[float], None] | None = None,
        on_speaking: Callable[[bool], None] | None = None,
    ):
        self.stt = stt
        self.tts = tts
        self.on_intent = on_intent
        self.on_transcript = on_transcript
        self.on_empty = on_empty  # called with the deadline to keep the turn open
        self.on_speaking = on_speaking  # suppress barge-in stop while the robot talks
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.dropped = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="stt-worker", daemon=True)
        self._thread.start()

    def submit(self, audio: np.ndarray, deadline: float | None = None) -> None:
        try:
            self._queue.put_nowait((audio, deadline))
        except queue.Full:
            try:
                self._queue.get_nowait()  # replace the pending turn
                self.dropped += 1
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait((audio, deadline))
            except queue.Full:
                pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                audio, deadline = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            text = self.stt.transcribe(audio)
            if self.on_transcript is not None:
                self.on_transcript(text)
            clean = strip_wake_phrase(text)
            if not clean:
                # Wake-only: keep the turn open for the command (threaded
                # continuation), so "Hey Rover" ... pause ... command works.
                if (
                    self.on_empty is not None
                    and deadline is not None
                    and time.monotonic() < deadline
                ):
                    self.on_empty(deadline)
                continue
            intent = classify(clean)
            if self.tts is not None and intent.response:
                if self.on_speaking is not None:
                    self.on_speaking(True)
                try:
                    self.tts.say(intent.response)
                finally:
                    if self.on_speaking is not None:
                        self.on_speaking(False)
            if self.on_intent is not None:
                self.on_intent(intent)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


class BargeInStop:
    """While the robot is moving: VAD-gated speech streamed to STT, stop on partials.

    No wake word. Requires a streaming STT (Moonshine); when only a whole-turn
    backend (Gemma) is available the caller leaves ``barge`` unset and stop falls
    back to wake + full turn. ``speaking`` suppresses matching while TTS plays so
    the robot cannot trigger itself.
    """

    def __init__(
        self,
        segmenter: Any,
        stream_factory: Any,
        on_stop: Callable[[], None],
        speaking: Callable[[], bool] = lambda: False,
    ):
        self._segmenter = segmenter
        self._stream_factory = stream_factory
        self._on_stop = on_stop
        self._speaking = speaking
        self._stream: Any = None

    def reset(self) -> None:
        self._segmenter.reset()
        if self._stream is not None:
            self._stream.cancel()
        self._stream = None

    def _matched(self) -> bool:
        if self._speaking():
            return False
        return is_stop(self._stream.partial() if self._stream else "")

    def process(self, frame: np.ndarray) -> bool:
        """Return True when a spoken stop was caught (``on_stop`` already called)."""
        event = self._segmenter.process(frame)
        if event == "start" and self._stream is None:
            self._stream = self._stream_factory()
        if self._stream is not None:
            self._stream.push(frame)
            if self._matched():
                self._on_stop()
                self.reset()
                return True
        if event == "end":
            if self._stream is not None and not self._speaking() and is_stop(self._stream.final()):
                self._on_stop()
                self.reset()
                return True
            self.reset()
        return False


class VoiceLoop:
    """Frame-driven listener state machine."""

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
        on_false_wake: Callable[[], None] | None = None,
        worker: ThreadedWorker | None = None,
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
        self.on_false_wake = on_false_wake
        self.worker = worker
        self.false_wakes = 0
        self.state = LISTENING
        self._turn_deadline = 0.0
        self._armed_until: float | None = None
        self.moving = False
        self.speaking = False
        self.barge: BargeInStop | None = None

    def set_moving(self, moving: bool) -> None:
        """Motion state changed: while moving, barge-in stop works without a wake word."""
        self.moving = bool(moving)
        if not self.moving and self.barge is not None:
            self.barge.reset()

    def set_speaking(self, speaking: bool) -> None:
        """TTS started/stopped; suppress stop matching while the robot talks."""
        self.speaking = bool(speaking)

    def arm_continuation(self, deadline: float) -> None:
        """Worker callback: keep the turn open for the command after a lone wake."""
        self._armed_until = deadline

    def _enter_recording(self, deadline: float | None = None) -> None:
        self._turn_deadline = (
            deadline if deadline is not None else time.monotonic() + self.max_turn_s
        )
        self.state = RECORDING

    def _false_wake(self) -> None:
        """A wake with nothing usable after it: log, count, and go back to idle."""
        self.false_wakes += 1
        if self.on_false_wake is not None:
            self.on_false_wake()
        self._resume_listening()

    def _flush(self) -> None:
        if self.flush is not None:
            self.flush()

    def _begin_turn(self) -> None:
        self.segmenter.reset()
        if self.speaker is not None and self.wake_sound:
            play_wav(self.speaker, self.wake_sound)
            # The chime plays through the listener; drop the audio buffered
            # during it so the turn starts clean (no chime in the pre-roll).
            self._flush()
            self.segmenter.reset()
        self._enter_recording()

    def _listen_again(self) -> None:
        """Wake-only/empty turn: keep waiting for the command, but reset audio."""
        self.segmenter.reset()
        self._flush()

    def _resume_listening(self) -> None:
        """Return to idle: clear VAD + wake state and drop buffered audio."""
        self.state = LISTENING
        self._armed_until = None
        self.segmenter.reset()
        reset = getattr(self.wakeword, "reset", None)
        if reset is not None:
            reset()
        self._flush()

    def _process_armed(self, frame: np.ndarray) -> None:
        """While a lone wake word still has the turn open, capture the command."""
        event = self.segmenter.process(frame)
        deadline = self._armed_until
        if event == "start":
            self._armed_until = None
            self._enter_recording(deadline)
        elif event == "end":
            self._armed_until = None
            audio = self.segmenter.audio()
            if self.worker is not None:
                self.worker.submit(audio, deadline)
            self._resume_listening()
        elif event == "timeout":
            self.segmenter.reset()  # keep waiting until the deadline

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
        """Feed one 80 ms int16 frame; return an Intent (inline) or None (worker)."""
        if self.state == LISTENING:
            if self.moving and self.barge is not None:
                self.barge.process(frame)
                return None
            if self._armed_until is not None:
                if time.monotonic() >= self._armed_until:
                    self._armed_until = None
                else:
                    self._process_armed(frame)
                    return None
            if self.wakeword.process(frame):
                self._begin_turn()
            return None

        event = self.segmenter.process(frame)
        if event == "end":
            audio = self.segmenter.audio()
            if self.worker is not None:
                self.worker.submit(audio, self._turn_deadline)
                self._resume_listening()
                return None
            clean = strip_wake_phrase(self._transcribe(audio))
            if not clean:
                if time.monotonic() < self._turn_deadline:
                    self._listen_again()  # the wake word only; wait for the command
                else:
                    self._false_wake()
                return None
            return self._finish(classify(clean))
        if event == "timeout":
            self._false_wake()
            return None
        if time.monotonic() > self._turn_deadline:
            self._false_wake()
        return None
