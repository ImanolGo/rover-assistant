"""Laptop-only tests for rover.main: sim wiring, snapshot, stop command."""

from __future__ import annotations

from pathlib import Path

from rover.config import load_config
from rover.main import App, select_source
from rover.perception.detector import FakeDetector

CONFIG = Path(__file__).resolve().parents[1] / "config" / "robot.yaml"


def _sim_config():
    config = load_config(CONFIG)
    config.sim = True
    config.camera.backend = "synthetic"
    config.camera.output = (160, 120)
    config.camera.fps = 200
    return config


def test_select_source_replaces_csi_only_in_sim():
    config = load_config(CONFIG)
    config.sim = True
    config.camera.backend = "csi"
    select_source(config)
    assert config.camera.backend == "synthetic"

    config.camera.backend = "usb"
    select_source(config)
    assert config.camera.backend == "usb"


def test_app_sim_perceive_once_publishes_a_snapshot():
    app = App(_sim_config(), detector=FakeDetector())
    try:
        assert app.perceive_once(timeout=2.0)
        snapshot = app.snapshot()
        assert snapshot.frame is not None
        assert snapshot.frame.shape == (120, 160, 3)
        status = app.status()
        assert status["sim"] is True
        assert status["camera_backend"] == "synthetic"
        assert status["frames"] >= 1
    finally:
        app.close()


def test_stop_command_zeroes_wheels_and_state():
    app = App(_sim_config(), detector=FakeDetector())
    try:
        app.perceive_once(timeout=2.0)
        app.left, app.right = 0.2, -0.2
        assert app.submit_command("please stop now") is True
        assert app.state == "STOPPED"
        assert app.rover.last_command == (0.0, 0.0)
        assert app.snapshot().state == "STOPPED"
    finally:
        app.close()


def test_empty_command_is_rejected():
    app = App(_sim_config(), detector=FakeDetector())
    try:
        assert app.submit_command("   ") is False
        assert app.submit_command("go to the cup") is True
        assert app.state == "COMMAND"
    finally:
        app.close()
