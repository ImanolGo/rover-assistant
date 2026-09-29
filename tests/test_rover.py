"""Laptop-only tests for hal.rover: kinematics, deadband, writer, watchdog, dry-run.

The kinematics cases are ported verbatim from the legacy
``tests/test_uart_motor_kinematics.py`` (commit 6a46836).
"""

from __future__ import annotations

import json
import time

import pytest

from rover.hal.rover import (
    DryRunRover,
    RoverConfig,
    SerialRover,
    compensate_deadband,
    twist_to_wheel_speeds,
)

WHEELBASE = 0.16
MAX_LINEAR = 0.3
MAX_SPEED = 0.5


def _map(linear: float, angular: float, min_pwm: float = 0.0) -> tuple[float, float]:
    return twist_to_wheel_speeds(linear, angular, WHEELBASE, MAX_LINEAR, MAX_SPEED, min_pwm)


# --- legacy kinematics cases -------------------------------------------------


def test_straight_forward_full_scale():
    left, right = _map(MAX_LINEAR, 0.0)
    assert left == pytest.approx(MAX_SPEED)
    assert right == pytest.approx(MAX_SPEED)


def test_pure_turn_is_differential():
    left, right = _map(0.0, 1.0)
    assert left < 0.0 < right
    assert left == pytest.approx(-0.1333, abs=1e-3)
    assert right == pytest.approx(0.1333, abs=1e-3)


def test_input_is_clipped():
    left, right = _map(10.0, 0.0)
    assert left == pytest.approx(MAX_SPEED)
    assert right == pytest.approx(MAX_SPEED)


def test_stop_stays_zero_even_with_deadband():
    assert _map(0.0, 0.0, min_pwm=0.3) == (0.0, 0.0)


def test_deadband_raises_small_commands():
    left, right = _map(0.0, 0.5, min_pwm=0.3)
    assert left == pytest.approx(-0.3)
    assert right == pytest.approx(0.3)


def test_deadband_disabled():
    left, right = _map(0.0, 0.5, min_pwm=0.0)
    assert left == pytest.approx(-0.0667, abs=1e-3)
    assert right == pytest.approx(0.0667, abs=1e-3)


def test_large_command_not_altered_by_deadband():
    left, right = _map(MAX_LINEAR, 0.0, min_pwm=0.3)
    assert left == pytest.approx(MAX_SPEED)
    assert right == pytest.approx(MAX_SPEED)


# --- deadband helper ---------------------------------------------------------


def test_compensate_deadband_sign_and_zero():
    assert compensate_deadband(0.0, 0.3) == 0.0
    assert compensate_deadband(0.05, 0.3) == pytest.approx(0.3)
    assert compensate_deadband(-0.05, 0.3) == pytest.approx(-0.3)
    assert compensate_deadband(0.4, 0.3) == pytest.approx(0.4)


# --- dry-run backend ---------------------------------------------------------


def test_dryrun_records_limited_commands():
    rover = DryRunRover(RoverConfig(speed_cap=0.5, min_wheel_pwm=0.3))
    rover.drive(0.4, -0.4)
    assert rover.last_command == (pytest.approx(0.4), pytest.approx(-0.4))
    rover.stop()
    assert rover.last_command == (0.0, 0.0)


def test_dryrun_applies_deadband_floor():
    rover = DryRunRover(RoverConfig(speed_cap=0.5, min_wheel_pwm=0.3))
    rover.drive(0.02, -0.02)
    assert rover.last_command == (pytest.approx(0.3), pytest.approx(-0.3))


def test_dryrun_battery_and_closed():
    rover = DryRunRover()
    assert rover.battery() == pytest.approx(12.2)
    rover.close()
    with pytest.raises(RuntimeError):
        rover.drive(0.1, 0.1)


# --- serial backend (fake transport) ----------------------------------------


class FakeSerial:
    """Minimal pyserial stand-in that records written JSON commands."""

    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.closed = False

    def setRTS(self, value: bool) -> None:  # noqa: N802 - pyserial API
        ...

    def setDTR(self, value: bool) -> None:  # noqa: N802 - pyserial API
        ...

    def reset_input_buffer(self) -> None: ...

    def reset_output_buffer(self) -> None: ...

    def write(self, data: bytes) -> int:
        self.writes.append(data)
        return len(data)

    def flush(self) -> None: ...

    def readline(self) -> bytes:
        time.sleep(0.01)
        return b""

    def close(self) -> None:
        self.closed = True

    def commands(self) -> list[dict]:
        return [json.loads(w.decode()) for w in self.writes]


def test_serial_writer_sends_and_watchdog_stops():
    config = RoverConfig(
        write_hz=100, watchdog_s=0.15, feedback=False, speed_cap=0.5, min_wheel_pwm=0.3
    )
    fake = FakeSerial()
    rover = SerialRover(config, serial_port=fake)
    try:
        rover.drive(0.4, 0.4)
        time.sleep(0.1)
        assert {"T": 1, "L": 0.4, "R": 0.4} in fake.commands()

        time.sleep(0.35)  # let the watchdog expire
        assert fake.commands()[-1] == {"T": 1, "L": 0.0, "R": 0.0}
    finally:
        rover.close()
    assert fake.closed


def test_parse_feedback_extracts_json_from_binary_noise():
    payload = SerialRover.parse_feedback('\x00\x13garbage{"T":1001,"v":11.8,"r":1}tail')
    assert payload == {"T": 1001, "v": 11.8, "r": 1}
    assert SerialRover.parse_feedback("no json here") is None
    assert SerialRover.parse_feedback('{"T": broken') is None


def test_handle_feedback_updates_battery():
    fake = FakeSerial()
    rover = SerialRover(RoverConfig(feedback=False), serial_port=fake)
    try:
        assert rover.battery() is None
        rover.handle_feedback({"T": 1001, "v": 11.7})
        assert rover.battery() == pytest.approx(11.7)
    finally:
        rover.close()
