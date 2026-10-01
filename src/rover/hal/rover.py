"""hal.rover: Wave Rover motor HAL (serial + dry-run backends).

Ports the legacy ROS2 ``uart_motor_controller.py`` logic without ROS:

- serial opened with RTS/DTR forced low (Wave Rover docs);
- a 20 Hz writer sending ``{"T":1,"L":x,"R":y}`` JSON lines;
- a watchdog that zeroes the wheels when no fresh command arrives (0.5 s);
- the DC-motor deadband mapping (nonzero commands are raised to
  ``min_wheel_pwm`` so the drivetrain actually moves);
- an optional T=131 feedback reader for battery voltage and IMU.

Motors are dangerous: ``drive()`` is the only motion entry point, it clamps to
``speed_cap`` and the watchdog always has the final say. No ROS, no encoders.
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol

DEFAULT_WHEELBASE_M = 0.16
DEFAULT_MAX_LINEAR_VELOCITY = 0.3
DEFAULT_MAX_SPEED = 0.5


@dataclass
class RoverConfig:
    """Tunables for the motor HAL (defaults mirror config/robot.yaml)."""

    port: str = "/dev/ttyTHS1"
    baudrate: int = 115200
    timeout_s: float = 1.0
    write_timeout_s: float = 1.0
    write_hz: float = 20.0
    watchdog_s: float = 0.5
    speed_cap: float = 0.15
    min_wheel_pwm: float = 0.30
    feedback: bool = True
    gyro_bias_calibrate_s: float = 2.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def compensate_deadband(speed: float, min_wheel_pwm: float) -> float:
    """Raise a small nonzero wheel command to the static-friction floor.

    Exact zero stays exactly zero (stop must remain stopped); the sign is kept.
    """
    if speed == 0.0 or min_wheel_pwm <= 0.0:
        return speed
    return math.copysign(max(abs(speed), min_wheel_pwm), speed)


def twist_to_wheel_speeds(
    linear_vel: float,
    angular_vel: float,
    wheelbase: float = DEFAULT_WHEELBASE_M,
    max_linear_velocity: float = DEFAULT_MAX_LINEAR_VELOCITY,
    max_speed: float = DEFAULT_MAX_SPEED,
    min_wheel_pwm: float = 0.0,
) -> tuple[float, float]:
    """Map (linear, angular) velocities to normalized left/right wheel speeds.

    Differential-drive kinematics; an optional deadband floor in
    ``(0, min_wheel_pwm)`` is raised to ``min_wheel_pwm`` (measured on the
    robot: in-place rotation only starts around 0.3, i.e. ~60% PWM).
    """
    v_left = linear_vel - (angular_vel * wheelbase) / 2.0
    v_right = linear_vel + (angular_vel * wheelbase) / 2.0

    def normalize(velocity: float) -> float:
        if max_linear_velocity <= 0.0:
            return 0.0
        speed = _clamp(velocity / max_linear_velocity * max_speed, -max_speed, max_speed)
        if 0.0 < abs(speed) < min_wheel_pwm:
            speed = math.copysign(min_wheel_pwm, speed)
        return speed

    return normalize(v_left), normalize(v_right)


class Rover(Protocol):
    """Minimal motor interface consumed by the brain/skills."""

    def drive(self, left: float, right: float) -> None:
        """Request wheel speeds (firmware scale, −0.5…0.5). Non-blocking."""
        ...

    def stop(self) -> None:
        """Zero both wheels immediately."""
        ...

    def battery(self) -> float | None:
        """Last feedback battery voltage in volts, or None if unknown."""
        ...

    def close(self) -> None:
        """Stop and release resources."""
        ...


def _limit_command(value: float, config: RoverConfig) -> float:
    """Clamp a request to the safety cap, then compensate the deadband."""
    capped = _clamp(value, -config.speed_cap, config.speed_cap)
    return compensate_deadband(capped, config.min_wheel_pwm)


class DryRunRover:
    """Fake rover: records commands without any hardware (ROVER_SIM / laptop)."""

    def __init__(self, config: RoverConfig | None = None):
        self.config = config or RoverConfig()
        self.commands: list[tuple[float, float, float]] = []
        self._battery_v: float | None = 12.2
        self._closed = False

    def drive(self, left: float, right: float) -> None:
        if self._closed:
            raise RuntimeError("rover is closed")
        self.commands.append(
            (time.time(), _limit_command(left, self.config), _limit_command(right, self.config))
        )

    def stop(self) -> None:
        self.drive(0.0, 0.0)

    def battery(self) -> float | None:
        return self._battery_v

    def close(self) -> None:
        self._closed = True

    @property
    def last_command(self) -> tuple[float, float]:
        return self.commands[-1][1:] if self.commands else (0.0, 0.0)


class SerialRover:
    """Real rover over ``/dev/ttyTHS1``. One writer thread, optional reader."""

    def __init__(self, config: RoverConfig | None = None, serial_port: Any | None = None):
        self.config = config or RoverConfig()
        if serial_port is not None:
            self._serial = serial_port
        else:
            import serial

            self._serial = serial.Serial(
                port=self.config.port,
                baudrate=self.config.baudrate,
                timeout=self.config.timeout_s,
                write_timeout=self.config.write_timeout_s,
            )
            # Wave Rover docs: hardware flow control must be off.
            self._serial.setRTS(False)
            self._serial.setDTR(False)
        if hasattr(self._serial, "reset_input_buffer"):
            self._serial.reset_input_buffer()
            self._serial.reset_output_buffer()

        self._lock = threading.Lock()
        self._target = (0.0, 0.0)
        self._target_time = 0.0
        self._battery_v: float | None = None
        self._roll_deg = 0.0
        self._pitch_deg = 0.0
        self._yaw_deg: float | None = None
        self._gyro_dps = (0.0, 0.0, 0.0)
        self._gyro_bias_rad_s = (0.0, 0.0, 0.0)
        self._closing = threading.Event()
        self._threads: list[threading.Thread] = []

        if self.config.feedback and self.config.gyro_bias_calibrate_s > 0:
            # Robot must be still: average the raw gyro before the reader runs.
            self._calibrate_gyro(self.config.gyro_bias_calibrate_s)
        if self.config.feedback:
            # Reader first, then enable continuous feedback (legacy ordering).
            self._spawn(self._read_loop)
            time.sleep(0.1)
            self._send({"T": 131, "cmd": 1})
        self._spawn(self._write_loop)

    def _spawn(self, target) -> None:
        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        self._threads.append(thread)

    def _write(self, command: dict[str, Any]) -> bool:
        try:
            with self._lock:
                self._serial.write((json.dumps(command) + "\n").encode())
                self._serial.flush()
            return True
        except Exception:  # noqa: BLE001 - serial errors must never kill the writer
            return False

    def _send(self, command: dict[str, Any]) -> bool:
        return self._write(command)

    def drive(self, left: float, right: float) -> None:
        with self._lock:
            self._target = (_limit_command(left, self.config), _limit_command(right, self.config))
            self._target_time = time.time()

    def stop(self) -> None:
        with self._lock:
            self._target = (0.0, 0.0)
            self._target_time = 0.0
        self._send({"T": 1, "L": 0.0, "R": 0.0})

    def battery(self) -> float | None:
        return self._battery_v

    def heading_deg(self) -> float | None:
        """Last yaw from feedback, in degrees (None until the first frame)."""
        return self._yaw_deg

    def gyro_rad_s(self) -> tuple[float, ...]:
        """Last raw angular velocity ``(gx, gy, gz)`` in rad/s."""
        return tuple(math.radians(value) for value in self._gyro_dps)

    def gyro_bias_rad_s(self) -> tuple[float, ...]:
        """Startup gyro bias ``(gx, gy, gz)`` in rad/s, averaged while still."""
        return self._gyro_bias_rad_s

    def _calibrate_gyro(self, duration_s: float) -> int:
        """Average the raw gyro for ``duration_s`` while the robot is still.

        Polls ``T=126`` (replied as ``T=1002`` with ``gx/gy/gz`` in deg/s) and
        stores the mean as the bias, converted to rad/s. Returns the number of
        samples used; zero leaves the bias at zero (e.g. no IMU answering).
        """
        samples: list[tuple[float, float, float]] = []
        previous_timeout = getattr(self._serial, "timeout", None)
        try:
            self._serial.timeout = 0.05  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - best effort on exotic transports
            previous_timeout = None
        try:
            deadline = time.monotonic() + duration_s
            while time.monotonic() < deadline:
                self._send({"T": 126})
                read_until = time.monotonic() + 0.2
                while time.monotonic() < read_until:
                    line = self._serial.readline()
                    if not line:
                        break
                    response = self.parse_feedback(line.decode("utf-8", errors="ignore"))
                    if (
                        response
                        and response.get("T") == 1002
                        and all(k in response for k in ("gx", "gy", "gz"))
                    ):
                        samples.append(tuple(float(response[k]) for k in ("gx", "gy", "gz")))
                        self._update_attitude(response)
                        break
                time.sleep(0.02)
        finally:
            if previous_timeout is not None:
                try:
                    self._serial.timeout = previous_timeout  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001
                    pass
        if samples:
            count = len(samples)
            self._gyro_bias_rad_s = tuple(
                math.radians(sum(sample[i] for sample in samples) / count) for i in range(3)
            )
        return len(samples)

    def _write_loop(self) -> None:
        period = 1.0 / self.config.write_hz
        while not self._closing.is_set():
            with self._lock:
                left, right = self._target
                fresh = (time.time() - self._target_time) < self.config.watchdog_s
            if not fresh:
                left = right = 0.0
            self._send({"T": 1, "L": left, "R": right})
            time.sleep(period)

    @staticmethod
    def parse_feedback(text: str) -> dict[str, Any] | None:
        """Extract the outermost JSON object from a mixed binary/JSON line."""
        start = text.find("{")
        end = text.rfind("}") + 1
        if start < 0 or end <= start:
            return None
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            return None

    def _update_attitude(self, response: dict[str, Any]) -> None:
        """Store roll/pitch/yaw (degrees) when present (T=1001 and T=1002)."""
        if "r" in response:
            self._roll_deg = float(response["r"])
        if "p" in response:
            self._pitch_deg = float(response["p"])
        if "y" in response:
            self._yaw_deg = float(response["y"])

    def handle_feedback(self, response: dict[str, Any]) -> None:
        """Apply a T=1001/1002/130 feedback object (battery, attitude, gyro)."""
        kind = response.get("T")
        if kind in (1001, 130):
            if "v" in response:
                self._battery_v = float(response["v"])
            self._update_attitude(response)
        elif kind == 1002:
            self._update_attitude(response)
            if all(k in response for k in ("gx", "gy", "gz")):
                self._gyro_dps = tuple(float(response[k]) for k in ("gx", "gy", "gz"))

    def _read_loop(self) -> None:
        while not self._closing.is_set():
            try:
                line = self._serial.readline()
            except Exception:  # noqa: BLE001
                time.sleep(0.1)
                continue
            if not line:
                continue
            response = self.parse_feedback(line.decode("utf-8", errors="ignore"))
            if response is not None:
                self.handle_feedback(response)

    def close(self) -> None:
        self._closing.set()
        try:
            self._send({"T": 1, "L": 0.0, "R": 0.0})
            if self.config.feedback:
                self._send({"T": 131, "cmd": 0})
        except Exception:  # noqa: BLE001
            pass
        for thread in self._threads:
            thread.join(timeout=1.0)
        try:
            self._serial.close()
        except Exception:  # noqa: BLE001
            pass


def open_rover(config: RoverConfig | None = None, *, dry_run: bool = False) -> Rover:
    """Build the configured rover backend (``dry_run`` for laptop/sim)."""
    if dry_run:
        return DryRunRover(config)
    return SerialRover(config)
