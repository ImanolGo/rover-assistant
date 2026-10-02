"""Laptop-only tests for rover.config: YAML mapping and the ROVER_SIM override."""

from __future__ import annotations

from pathlib import Path

import pytest

from rover.config import load_config

CONFIG = Path(__file__).resolve().parents[1] / "config" / "robot.yaml"


def test_load_real_config_maps_every_section():
    config = load_config(CONFIG)
    assert config.sim is False
    assert config.camera.backend == "csi"
    assert config.camera.capture == (1640, 1232)
    assert config.camera.output == (820, 616)
    assert config.rover.port == "/dev/ttyTHS1"
    assert config.rover.speed_cap == pytest.approx(0.15)
    assert config.rover.min_wheel_pwm == pytest.approx(0.30)
    assert config.perception.conf == pytest.approx(0.35)
    assert config.perception.tracker == "bytetrack.yaml"
    assert config.perception.torch_threads == 1
    assert config.perception.target_classes_h_stop["person"] == pytest.approx(0.60)
    assert config.control.tick_hz == pytest.approx(10.0)
    assert config.control.follow_h_frac == (0.50, 0.65)
    assert config.voice.enabled is True
    assert config.voice.no_speech_timeout_s == pytest.approx(3.5)
    assert config.voice.stt_backend == "gemma"
    assert config.planner.url == "http://127.0.0.1:8080"
    assert config.api.port == 8000
    assert config.rover_backend == "serial"
    assert config.low_battery_v == pytest.approx(10.5)
    assert config.raw["perception"]["imgsz"] == 640


def test_rover_sim_environment_overrides_file(monkeypatch):
    monkeypatch.setenv("ROVER_SIM", "1")
    assert load_config(CONFIG).sim is True
    monkeypatch.setenv("ROVER_SIM", "0")
    assert load_config(CONFIG).sim is False
    monkeypatch.delenv("ROVER_SIM", raising=False)
    assert load_config(CONFIG).sim is False


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        load_config("does/not/exist.yaml")
