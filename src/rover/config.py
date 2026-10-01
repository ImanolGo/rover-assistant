"""rover.config: load ``config/robot.yaml`` into plain dataclasses.

One YAML file holds every tunable (ARCHITECTURE.md §7). The HAL dataclasses keep
their own defaults so laptop tests can build components without a config file;
this module only maps the file onto them. No framework, no environment loader:
``ROVER_SIM`` is the single env override, so a laptop can force simulation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from rover.hal.audio import AudioConfig
from rover.hal.camera import CameraConfig
from rover.hal.rover import RoverConfig

DEFAULT_CONFIG_PATH = "config/robot.yaml"


@dataclass
class PerceptionConfig:
    """Detector + geometry tunables (mirrors ``robot.yaml`` ``perception:``)."""

    engine: str = "models/yolo_trt/yolo11n_fp16.engine"
    weights_fallback: str = "models/yolo11n.pt"
    imgsz: int = 640
    conf: float = 0.35
    tracker: str = "bytetrack.yaml"
    torch_threads: int = 1
    target_classes_h_stop: dict[str, float] = field(default_factory=dict)
    calibration_file: str = "config/camera_calibration.yaml"
    undistort_alpha: float = 1.0


@dataclass
class ControlConfig:
    """Selector/skill gains (PLAN Phase 5; loaded here so the file is central)."""

    tick_hz: float = 10.0
    kp_turn: float = 0.9
    bearing_deadband_deg: float = 3.0
    approach_max_bearing_deg: float = 15.0
    v_max: float = 0.15
    search_step_deg: float = 35.0
    search_pause_s: float = 0.5
    search_max_steps: int = 12
    lost_reacquire_s: float = 1.0
    follow_h_frac: tuple[float, float] = (0.50, 0.65)
    follow_lost_give_up_s: float = 5.0
    verify_frames: int = 3
    verify_min_conf: float = 0.5
    max_replans: int = 2


@dataclass
class VoiceConfig:
    """Wake word / VAD / STT / TTS tunables (PLAN Phase 4)."""

    enabled: bool = True
    wakeword_model: str = "models/wake_word/hey_roe_ver.onnx"
    wakeword_threshold: float = 0.5
    speex_noise_suppression: bool = True
    vad_threshold: float = 0.5
    end_silence_ms: int = 800
    max_utterance_s: float = 10.0
    stt_backend: str = "gemma"
    whisper_model: str = "models/whisper/ggml-base.en.bin"
    tts_voice: str = "models/piper/en_US-lessac-medium.onnx"
    wake_sound: str = "assets/audio/notify_asc.wav"


@dataclass
class PlannerConfig:
    """Gemma HTTP client tunables (PLAN Phase 5)."""

    url: str = "http://127.0.0.1:8080"
    fallback_url: str | None = None
    mode: str = "jinja_tools"
    max_tokens: int = 96
    temperature: float = 0.2
    image_max_tokens: int = 70
    timeout_s: float = 8.0


@dataclass
class LlamaServerConfig:
    """How ``deploy/systemd/rover-llama.service`` should start llama-server."""

    binary: str = "~/llama.cpp/build/bin/llama-server"
    model: str = "models/gemma/gemma-4-E2B-it-Q4_K_M.gguf"
    mmproj: str = "models/gemma/mmproj-gemma4-e2b-q8_0.gguf"
    args: str = "-c 2048 -ngl 99 --jinja --cache-ram 0 --image-max-tokens 70"


@dataclass
class ApiConfig:
    """FastAPI bind address."""

    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class TelemetryConfig:
    """Where decision-log JSONL files are appended (PLAN Phase 5)."""

    decisions_dir: str = "logs"


@dataclass
class RobotConfig:
    """Everything loaded from ``robot.yaml``, plus the raw mapping."""

    sim: bool
    camera: CameraConfig
    rover: RoverConfig
    audio: AudioConfig
    perception: PerceptionConfig
    control: ControlConfig
    voice: VoiceConfig
    planner: PlannerConfig
    llama_server: LlamaServerConfig
    api: ApiConfig
    telemetry: TelemetryConfig
    rover_backend: str = "serial"
    low_battery_v: float = 10.5
    rts_dtr_false: bool = True
    path: str = DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = field(default_factory=dict)


def _env_sim_override(default: bool) -> bool:
    value = os.environ.get("ROVER_SIM")
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "")


def _sub(config: dict[str, Any], key: str) -> dict[str, Any]:
    return dict(config.get(key) or {})


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> RobotConfig:
    """Parse ``robot.yaml`` into :class:`RobotConfig` (``ROVER_SIM`` overrides sim)."""
    with open(path, "r", encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle) or {}

    camera_raw = _sub(data, "camera")
    rover_raw = _sub(data, "rover")
    audio_raw = _sub(data, "audio")
    perception_raw = _sub(data, "perception")
    control_raw = _sub(data, "control")
    voice_raw = _sub(data, "voice")
    planner_raw = _sub(data, "planner")
    llama_raw = _sub(data, "llama_server")
    api_raw = _sub(data, "api")
    telemetry_raw = _sub(data, "telemetry")

    camera = CameraConfig(
        backend=camera_raw.get("backend", "csi"),
        sensor_id=int(camera_raw.get("sensor_id", 0)),
        capture=tuple(camera_raw.get("capture", (1640, 1232))),
        output=tuple(camera_raw.get("output", (820, 616))),
        fps=int(camera_raw.get("fps", 30)),
        flip_method=int(camera_raw.get("flip_method", 0)),
        usb_index=int(camera_raw.get("usb_index", 0)),
        file_path=camera_raw.get("file_path", "bench/data/frames"),
    )
    rover = RoverConfig(
        port=rover_raw.get("port", "/dev/ttyTHS1"),
        baudrate=int(rover_raw.get("baudrate", 115200)),
        write_hz=float(rover_raw.get("write_hz", 20.0)),
        watchdog_s=float(rover_raw.get("watchdog_s", 0.5)),
        speed_cap=float(rover_raw.get("speed_cap", 0.15)),
        min_wheel_pwm=float(rover_raw.get("min_wheel_pwm", 0.30)),
        feedback=bool(rover_raw.get("feedback", True)),
        gyro_bias_calibrate_s=float(rover_raw.get("gyro_bias_calibrate_s", 2.0)),
    )
    audio = AudioConfig(
        mic_name=audio_raw.get("mic_name", "USB PnP Sound Device"),
        speaker_name=audio_raw.get("speaker_name", "UACDemoV1.0"),
        mic_rate=int(audio_raw.get("mic_rate", 16000)),
        speaker_rate=int(audio_raw.get("speaker_rate", 48000)),
        speaker_channels=int(audio_raw.get("speaker_channels", 2)),
        frame_ms=int(audio_raw.get("frame_ms", 80)),
    )
    perception = PerceptionConfig(
        engine=perception_raw.get("engine", "models/yolo_trt/yolo11n_fp16.engine"),
        weights_fallback=perception_raw.get("weights_fallback", "models/yolo11n.pt"),
        imgsz=int(perception_raw.get("imgsz", 640)),
        conf=float(perception_raw.get("conf", 0.35)),
        tracker=perception_raw.get("tracker", "bytetrack.yaml"),
        torch_threads=int(perception_raw.get("torch_threads", 1)),
        target_classes_h_stop=dict(perception_raw.get("target_classes_h_stop") or {}),
        calibration_file=camera_raw.get("calibration_file", "config/camera_calibration.yaml"),
        undistort_alpha=float(camera_raw.get("undistort_alpha", 1.0)),
    )
    control = ControlConfig(
        tick_hz=float(control_raw.get("tick_hz", 10.0)),
        kp_turn=float(control_raw.get("kp_turn", 0.9)),
        bearing_deadband_deg=float(control_raw.get("bearing_deadband_deg", 3.0)),
        approach_max_bearing_deg=float(control_raw.get("approach_max_bearing_deg", 15.0)),
        v_max=float(control_raw.get("v_max", 0.15)),
        search_step_deg=float(control_raw.get("search_step_deg", 35.0)),
        search_pause_s=float(control_raw.get("search_pause_s", 0.5)),
        search_max_steps=int(control_raw.get("search_max_steps", 12)),
        lost_reacquire_s=float(control_raw.get("lost_reacquire_s", 1.0)),
        follow_h_frac=tuple(control_raw.get("follow_h_frac", (0.50, 0.65))),
        follow_lost_give_up_s=float(control_raw.get("follow_lost_give_up_s", 5.0)),
        verify_frames=int(control_raw.get("verify_frames", 3)),
        verify_min_conf=float(control_raw.get("verify_min_conf", 0.5)),
        max_replans=int(control_raw.get("max_replans", 2)),
    )
    voice = VoiceConfig(
        enabled=bool(voice_raw.get("enabled", True)),
        wakeword_model=voice_raw.get("wakeword_model", VoiceConfig.wakeword_model),
        wakeword_threshold=float(voice_raw.get("wakeword_threshold", 0.5)),
        speex_noise_suppression=bool(voice_raw.get("speex_noise_suppression", True)),
        vad_threshold=float(voice_raw.get("vad_threshold", 0.5)),
        end_silence_ms=int(voice_raw.get("end_silence_ms", 800)),
        max_utterance_s=float(voice_raw.get("max_utterance_s", 10.0)),
        stt_backend=voice_raw.get("stt_backend", "gemma"),
        whisper_model=voice_raw.get("whisper_model", VoiceConfig.whisper_model),
        tts_voice=voice_raw.get("tts_voice", VoiceConfig.tts_voice),
        wake_sound=voice_raw.get("wake_sound", VoiceConfig.wake_sound),
    )
    planner = PlannerConfig(
        url=planner_raw.get("url", "http://127.0.0.1:8080"),
        fallback_url=planner_raw.get("fallback_url"),
        mode=planner_raw.get("mode", "jinja_tools"),
        max_tokens=int(planner_raw.get("max_tokens", 96)),
        temperature=float(planner_raw.get("temperature", 0.2)),
        image_max_tokens=int(planner_raw.get("image_max_tokens", 70)),
        timeout_s=float(planner_raw.get("timeout_s", 8.0)),
    )
    llama_server = LlamaServerConfig(
        binary=llama_raw.get("binary", LlamaServerConfig.binary),
        model=llama_raw.get("model", LlamaServerConfig.model),
        mmproj=llama_raw.get("mmproj", LlamaServerConfig.mmproj),
        args=llama_raw.get("args", LlamaServerConfig.args),
    )
    api = ApiConfig(
        host=api_raw.get("host", "0.0.0.0"),
        port=int(api_raw.get("port", 8000)),
    )
    telemetry = TelemetryConfig(decisions_dir=telemetry_raw.get("decisions_dir", "logs"))

    return RobotConfig(
        sim=_env_sim_override(bool(data.get("sim", False))),
        camera=camera,
        rover=rover,
        audio=audio,
        perception=perception,
        control=control,
        voice=voice,
        planner=planner,
        llama_server=llama_server,
        api=api,
        telemetry=telemetry,
        rover_backend=rover_raw.get("backend", "serial"),
        low_battery_v=float(rover_raw.get("low_battery_v", 10.5)),
        rts_dtr_false=bool(rover_raw.get("rts_dtr_false", True)),
        path=str(path),
        raw=data,
    )
