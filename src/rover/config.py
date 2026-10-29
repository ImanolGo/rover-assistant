"""rover.config: load ``config/robot.yaml`` into plain dataclasses.

One YAML file holds every tunable (ARCHITECTURE.md §7). The HAL dataclasses keep
their own defaults so laptop tests can build components without a config file;
this module only maps the file onto them. No framework, no environment loader:
``ROVER_SIM`` is the single env override, so a laptop can force simulation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
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
    v_max: float = 0.35
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
    wakeword_vad_threshold: float = 0.5
    speex_noise_suppression: bool = False
    vad_threshold: float = 0.5
    end_silence_ms: int = 800
    no_speech_timeout_s: float = 3.5
    max_utterance_s: float = 10.0
    max_turn_s: float = 8.0
    stt_backend: str = "moonshine"
    moonshine_model: str = "small"
    moonshine_vad_threshold: float = 0.01
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


def _section(cls: type, raw: dict[str, Any], **extra: Any) -> Any:
    """Build ``cls`` from a YAML section: keys the dataclass knows, coerced to the
    type of its default (so ``1`` in YAML still becomes ``1.0``); everything else
    keeps the dataclass default, which is the single source of defaults."""
    kwargs: dict[str, Any] = {}
    for item in fields(cls):
        if item.name not in raw:
            continue
        value, default = raw[item.name], getattr(cls, item.name, None)
        if isinstance(default, (bool, int, float, str, tuple)) and value is not None:
            value = type(default)(value)
        elif item.default_factory is dict:  # type: ignore[comparison-overlap]
            value = dict(value or {})
        kwargs[item.name] = value
    return cls(**{**kwargs, **extra})


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> RobotConfig:
    """Parse ``robot.yaml`` into :class:`RobotConfig` (``ROVER_SIM`` overrides sim)."""
    with open(path, "r", encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle) or {}

    camera_raw = _sub(data, "camera")
    rover_raw = _sub(data, "rover")
    # calibration lives under camera: in the YAML but belongs to perception.
    calibration = {
        k: camera_raw[k] for k in ("calibration_file", "undistort_alpha") if k in camera_raw
    }
    return RobotConfig(
        sim=_env_sim_override(bool(data.get("sim", False))),
        camera=_section(CameraConfig, camera_raw),
        rover=_section(RoverConfig, rover_raw),
        audio=_section(AudioConfig, _sub(data, "audio")),
        perception=_section(PerceptionConfig, {**_sub(data, "perception"), **calibration}),
        control=_section(ControlConfig, _sub(data, "control")),
        voice=_section(VoiceConfig, _sub(data, "voice")),
        planner=_section(PlannerConfig, _sub(data, "planner")),
        llama_server=_section(LlamaServerConfig, _sub(data, "llama_server")),
        api=_section(ApiConfig, _sub(data, "api")),
        telemetry=_section(TelemetryConfig, _sub(data, "telemetry")),
        rover_backend=rover_raw.get("backend", "serial"),
        low_battery_v=float(rover_raw.get("low_battery_v", 10.5)),
        rts_dtr_false=bool(rover_raw.get("rts_dtr_false", True)),
        path=str(path),
        raw=data,
    )
