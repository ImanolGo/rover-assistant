"""rover.main: asyncio entry point for the brain (``rover`` console script).

Phase 3 runs the perception loop and the debug API; voice (Phase 4) and the
mission planner (Phase 5) plug into the same process later. This process owns
the camera thread, the single CUDA context (YOLO) and the rover writer.

    uv run rover              # Jetson / real hardware
    ROVER_SIM=1 uv run rover  # laptop: synthetic frames + DryRunRover + API
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import threading
import time
from typing import Any

from rover.api.server import VideoSnapshot, create_app
from rover.config import DEFAULT_CONFIG_PATH, RobotConfig, load_config
from rover.hal.camera import FrameSource, open_camera
from rover.hal.rover import Rover, open_rover
from rover.perception.detector import Detector, make_detector
from rover.perception.geometry import CameraGeometry
from rover.voice.intents import Intent, classify


def select_source(config: RobotConfig) -> None:
    """In sim, replace the CSI backend (impossible on a laptop) with synthetic."""
    if config.sim and config.camera.backend == "csi":
        config.camera.backend = "synthetic"


class App:
    """Owns the camera, detector, rover and the latest rendered state.

    Implements :class:`rover.api.server.ApiContext` structurally.
    """

    def __init__(
        self,
        config: RobotConfig,
        source: FrameSource | None = None,
        detector: Detector | None = None,
    ):
        self.config = config
        select_source(config)
        self.geometry = CameraGeometry.from_yaml(
            config.perception.calibration_file,
            output_size=config.camera.output,
            alpha=config.perception.undistort_alpha,
        )
        self.camera = open_camera(config.camera, source=source)
        self.detector = detector or make_detector(config.perception, self.geometry, sim=config.sim)
        self.rover: Rover = open_rover(
            config.rover, dry_run=config.sim or config.rover_backend == "dryrun"
        )
        self.state = "IDLE"
        self.left = 0.0
        self.right = 0.0
        self.fps = 0.0
        self.last_command: str | None = None
        self.last_transcript = ""
        self.last_intent: Intent | None = None
        self.voice_error: str | None = None
        self._voice_loop: Any = None
        self._voice_pump: Any = None
        self._voice_worker: Any = None
        self._lock = threading.Lock()
        self._snapshot = VideoSnapshot(state=self.state)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._voice_thread: threading.Thread | None = None
        self._frames = 0
        self._last_frame_time = 0.0
        self._last_t_capture: float | None = None

    # --- perception loop -----------------------------------------------------

    def perceive_once(self, timeout: float = 2.0) -> bool:
        """Grab the newest frame, detect, and publish a fresh snapshot.

        Returns False when no *new* frame has arrived, so the loop never
        re-runs the detector on the same image (which would peg the GPU).
        """
        try:
            frame, t_capture = self.camera.latest(timeout=timeout)
        except TimeoutError:
            return False
        if self._last_t_capture is not None and t_capture <= self._last_t_capture:
            return False
        self._last_t_capture = t_capture
        detections = self.detector.detect(frame)
        now = time.time()
        if self._last_frame_time:
            self.fps = 1.0 / max(1e-3, now - self._last_frame_time)
        self._last_frame_time = now
        self._frames += 1
        with self._lock:
            self._snapshot = VideoSnapshot(
                frame=frame,
                detections=detections,
                state=self.state,
                left=self.left,
                right=self.right,
                fps=self.fps,
            )
        return True

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not self.perceive_once(timeout=1.0):
                time.sleep(0.005)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="perception", daemon=True)
        self._thread.start()

    # --- voice listener ------------------------------------------------------

    def start_voice(self) -> None:
        self._voice_thread = threading.Thread(target=self._voice_loop, name="voice", daemon=True)
        self._voice_thread.start()

    def _voice_loop(self) -> None:
        try:
            from rover.hal.audio import AlsaCapture, AlsaSpeaker
            from rover.voice.pipeline import AudioQueue, CapturePump, ThreadedWorker, VoiceLoop
            from rover.voice.stt import GemmaStt
            from rover.voice.tts import PiperTts
            from rover.voice.vad import SpeechSegmenter, load_silero
            from rover.voice.wakeword import make_wakeword

            voice = self.config.voice
            capture = AlsaCapture(self.config.audio)
            speaker = AlsaSpeaker(self.config.audio)
            frames = AudioQueue(maxsize=25)
            pump = CapturePump(capture, frames)
            wakeword = make_wakeword(voice.wakeword_model, threshold=voice.wakeword_threshold)
            segmenter = SpeechSegmenter(
                load_silero(end_silence_ms=voice.end_silence_ms, threshold=voice.vad_threshold),
                max_utterance_s=voice.max_utterance_s,
                onset_timeout_s=voice.no_speech_timeout_s,
            )
            stt = GemmaStt(self.config.planner.url)
            tts = PiperTts(voice.tts_voice, speaker)
            worker = ThreadedWorker(
                stt, tts=tts, on_intent=self._on_intent, on_transcript=self._on_transcript
            )
            loop = VoiceLoop(
                wakeword,
                segmenter,
                stt,
                tts=tts,
                speaker=speaker,
                wake_sound=voice.wake_sound,
                flush=lambda: (capture.flush(), frames.drain()),
                max_turn_s=voice.max_turn_s,
                on_false_wake=lambda: print("voice: false wake (no speech)"),
                worker=worker,
            )
            self._voice_loop = loop
            self._voice_pump = pump
            self._voice_worker = worker
            worker.start()
            pump.start()
            print("voice: listening for 'Hey Rover'")
            while not self._stop.is_set():
                frame = frames.get(timeout=0.5)
                if frame is None:
                    continue
                loop.process_frame(frame)
            pump.stop()
            worker.close()
            capture.close()
            tts.close()
            stt.close()
        except Exception as exc:  # noqa: BLE001 - voice must never kill the brain
            self.voice_error = f"{type(exc).__name__}: {exc}"
            print(f"voice disabled: {self.voice_error}")

    def _on_transcript(self, text: str) -> None:
        self.last_transcript = text
        print(f"voice: transcript={text!r}")

    def _on_intent(self, intent: Intent) -> None:
        self.last_intent = intent
        print(f"voice: intent={intent.name} target={intent.target} attrs={intent.attributes}")
        if intent.name == "stop":
            self.stop()
        else:
            self.state = intent.name.upper()

    # --- ApiContext ----------------------------------------------------------

    def status(self) -> dict[str, Any]:
        with self._lock:
            snapshot = self._snapshot
        bearing = snapshot.detections[0].bearing_deg if snapshot.detections else None
        return {
            "state": self.state,
            "fps": round(self.fps, 1),
            "objects": len(snapshot.detections),
            "left": round(self.left, 3),
            "right": round(self.right, 3),
            "bearing_deg": bearing,
            "last_command": self.last_command,
            "transcript": self.last_transcript,
            "intent": self.last_intent.name if self.last_intent else None,
            "target": self.last_intent.target if self.last_intent else None,
            "voice_error": self.voice_error,
            "false_wakes": getattr(self._voice_loop, "false_wakes", 0),
            "frames_dropped": (
                getattr(getattr(self._voice_pump, "queue", None), "dropped", 0)
                + getattr(self._voice_worker, "dropped", 0)
            ),
            "camera_backend": self.config.camera.backend,
            "sim": self.config.sim,
            "frames": self._frames,
        }

    def snapshot(self) -> VideoSnapshot:
        with self._lock:
            return self._snapshot

    def submit_command(self, text: str) -> bool:
        """Debug command injection: "stop" is handled before anything else."""
        text = (text or "").strip()
        if not text:
            return False
        self.last_command = text
        if classify(text).name == "stop":
            self.stop()
            return True
        self.state = "COMMAND"
        return True

    def stop(self) -> None:
        self.rover.stop()
        self.state = "STOPPED"
        self.left = self.right = 0.0
        with self._lock:
            self._snapshot = VideoSnapshot(
                frame=self._snapshot.frame,
                detections=self._snapshot.detections,
                state=self.state,
                left=0.0,
                right=0.0,
                fps=self.fps,
            )

    # --- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._voice_thread is not None:
            self._voice_thread.join(timeout=2.0)
        self.camera.close()
        self.detector.close()
        self.rover.close()

    async def run(self) -> None:
        import uvicorn

        self.start()
        if self.config.voice.enabled and not self.config.sim:
            self.start_voice()
        server = uvicorn.Server(
            uvicorn.Config(
                create_app(self),
                host=self.config.api.host,
                port=self.config.api.port,
                log_level="warning",
            )
        )
        loop = asyncio.get_running_loop()

        def shutdown() -> None:
            self._stop.set()
            server.should_exit = True

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, shutdown)
            except NotImplementedError:  # pragma: no cover - non-POSIX
                pass
        print(
            f"rover up: sim={self.config.sim} camera={self.config.camera.backend} "
            f"api=http://{self.config.api.host}:{self.config.api.port}"
        )
        try:
            await server.serve()
        finally:
            self.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--sim", action="store_true", help="force simulation")
    parser.add_argument("--frames", default=None, help="replay a folder/file instead of a camera")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.sim:
        config.sim = True
    if args.frames:
        config.camera.backend = "file"
        config.camera.file_path = args.frames
    app = App(config)
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:  # pragma: no cover - interactive
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
