"""Laptop-only tests for rover.main: sim wiring, snapshot, stop command."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rover.config import load_config
from rover.main import App, select_source
from rover.perception.detector import FakeDetector
from rover.voice.intents import Intent

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


class _FrozenSource:
    """Frame source whose capture timestamp never advances."""

    def __init__(self, frame):
        self.frame = frame
        self.closed = False

    def read(self):
        return self.frame, 123.0

    def close(self):
        self.closed = True


def test_same_frame_is_not_reprocessed():
    source = _FrozenSource(np.zeros((120, 160, 3), dtype=np.uint8))
    app = App(_sim_config(), source=source, detector=FakeDetector())
    try:
        assert app.perceive_once() is True
        assert app.perceive_once() is False  # same timestamp -> skipped
        assert app.status()["frames"] == 1
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


def test_status_exposes_voice_fields():
    app = App(_sim_config(), detector=FakeDetector())
    try:
        status = app.status()
        assert status["transcript"] == ""
        assert status["intent"] is None
        assert "voice_error" in status
    finally:
        app.close()


def test_voice_stop_intent_zeroes_wheels():
    app = App(_sim_config(), detector=FakeDetector())
    try:
        app.rover.drive(0.1, 0.1)
        app._on_intent(Intent(name="stop"))
        assert app.state == "STOPPED"
        assert app.rover.last_command == (0.0, 0.0)
    finally:
        app.close()


def test_voice_go_to_intent_sets_state_and_status():
    app = App(_sim_config(), detector=FakeDetector())
    try:
        app._on_intent(Intent(name="go_to", target="cup", attributes=("red",)))
        status = app.status()
        assert app.state == "GO_TO"
        assert status["intent"] == "go_to"
        assert status["target"] == "cup"
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
