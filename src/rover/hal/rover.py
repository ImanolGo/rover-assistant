"""hal.rover: Wave Rover motor HAL (serial + dry-run backends).

Ports the legacy ROS2 ``uart_motor_controller.py`` logic without ROS:

- serial opened with RTS/DTR forced low (Wave Rover docs);
- a 20 Hz writer sending ``{"T":1,"L":x,"R":y}`` JSON lines;
- a watchdog that zeroes the wheels when no fresh command arrives (0.5 s);
- the DC-motor deadband mapping (a command rounds to 0 or up to
  ``min_wheel_pwm``, whichever is nearer, so the drivetrain never stalls);
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
_GYRO = ("gx", "gy", "gz")
SPIN_DEG_S_PER_UNIT = 300.0  # sim: measured ~150 deg/s spinning at 0.5 (2026-10-29)
MAX_GYRO_GAP_S = 0.25  # do not integrate across a feedback gap longer than this


@dataclass
class RoverConfig:
    """Tunables for the motor HAL (defaults mirror config/robot.yaml)."""

    port: str = "/dev/ttyTHS1"
    baudrate: int = 115200
    timeout_s: float = 1.0
    write_timeout_s: float = 1.0
    write_hz: float = 20.0
    watchdog_s: float = 0.5
    speed_cap: float = 0.35
    spin_speed_cap: float = 0.5  # in-place spins only (opposite, equal wheels); see _limit_pair
    min_wheel_pwm: float = 0.30
    feedback: bool = True
    gyro_bias_calibrate_s: float = 2.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def compensate_deadband(speed: float, min_wheel_pwm: float) -> float:
    """Map a wheel command out of the stall band ``(0, min_wheel_pwm)``.

    Below half the floor the command becomes 0 (a tiny correction must not jump
    to full floor speed); from there up to the floor it is raised to the floor.
    Exact zero stays exactly zero; the sign is kept.
    """
    if speed == 0.0 or min_wheel_pwm <= 0.0 or abs(speed) >= min_wheel_pwm:
        return speed
    if abs(speed) < min_wheel_pwm / 2:
        return 0.0
    return math.copysign(min_wheel_pwm, speed)


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
    ``(0, min_wheel_pwm)`` is raised to ``min_wheel_pwm`` (wheels start around 0.3
    wheels-up; on a hard floor an in-place spin needs ~0.5, see ``spin_speed_cap``).
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

    def heading_deg(self) -> float:
        """Integrated gyro-z heading in degrees since startup, + = left (CCW).

        Unwrapped (keeps counting past ±180). The firmware yaw is not used: it is
        anchored to the robot's own magnetic field and does not track rotation.
        """
        ...

    def close(self) -> None:
        """Stop and release resources."""
        ...


def _limit_pair(left: float, right: float, config: RoverConfig) -> tuple[float, float]:
    """Clamp a request to the safety cap, then compensate the deadband.

    An in-place spin (opposite signs, equal magnitude within 0.05) may use
    ``spin_speed_cap``: on a hard floor skid-steer scrub stalls anything below ~0.5
    (STATUS 2026-10-29). Everything else, including arcs, stays at ``speed_cap``.
    """
    spin = left * right < 0 and abs(abs(left) - abs(right)) <= 0.05
    cap = config.spin_speed_cap if spin else config.speed_cap
    return tuple(  # type: ignore[return-value]
        compensate_deadband(_clamp(v, -cap, cap), config.min_wheel_pwm) for v in (left, right)
    )


class DryRunRover:
    """Fake rover: records commands without any hardware (ROVER_SIM / laptop)."""

    def __init__(self, config: RoverConfig | None = None):
        self.config = config or RoverConfig()
        self.commands: list[tuple[float, float, float]] = []
        self._battery_v: float | None = 12.2
        self._closed = False
        self._heading = 0.0
        self._rate_since = (0.0, time.monotonic())  # (deg/s, since) of the last command

    def drive(self, left: float, right: float) -> None:
        if self._closed:
            raise RuntimeError("rover is closed")
        limited = _limit_pair(left, right, self.config)
        self.commands.append((time.time(), *limited))
        self._heading = self.heading_deg()  # bank the turn so far, then change rate
        rate = SPIN_DEG_S_PER_UNIT * (limited[1] - limited[0]) / 2.0
        self._rate_since = (rate, time.monotonic())

    def stop(self) -> None:
        self.drive(0.0, 0.0)

    def battery(self) -> float | None:
        return self._battery_v

    def heading_deg(self) -> float:
        """Simulated: turn rate from the commanded wheel difference (no watchdog)."""
        rate, since = self._rate_since
        return self._heading + rate * (time.monotonic() - since)

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
        self._heading = 0.0
        self._last_gyro_t: float | None = None
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

    def _send(self, command: dict[str, Any]) -> bool:
        try:
            with self._lock:
                self._serial.write((json.dumps(command) + "\n").encode())
                self._serial.flush()
            return True
        except Exception:  # noqa: BLE001 - serial errors must never kill the writer
            return False

    def drive(self, left: float, right: float) -> None:
        with self._lock:
            self._target = _limit_pair(left, right, self.config)
            self._target_time = time.monotonic()

    def stop(self) -> None:
        with self._lock:
            self._target = (0.0, 0.0)
            self._target_time = 0.0
        self._send({"T": 1, "L": 0.0, "R": 0.0})

    def battery(self) -> float | None:
        return self._battery_v

    def heading_deg(self) -> float:
        return self._heading

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
        samples: list[tuple[float, ...]] = []
        previous_timeout = getattr(self._serial, "timeout", None)
        self._serial.timeout = 0.05  # short reads while polling
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
                    if response and response.get("T") == 1002 and set(_GYRO) <= response.keys():
                        samples.append(tuple(float(response[k]) for k in _GYRO))
                        break
                time.sleep(0.02)
        finally:
            self._serial.timeout = previous_timeout
        if samples:
            self._gyro_bias_rad_s = tuple(math.radians(sum(c) / len(c)) for c in zip(*samples))
        return len(samples)

    def _write_loop(self) -> None:
        period = 1.0 / self.config.write_hz
        while not self._closing.is_set():
            with self._lock:
                left, right = self._target
                fresh = (time.monotonic() - self._target_time) < self.config.watchdog_s
            if not fresh:
                left = right = 0.0
            self._send({"T": 1, "L": left, "R": right})
            if self.config.feedback:
                self._send({"T": 126})  # IMU poll: its T=1002 reply drives heading_deg()
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

    def handle_feedback(self, response: dict[str, Any]) -> None:
        """Apply a T=1001/130 (battery) or T=1002 (IMU: integrate gyro z) object."""
        kind = response.get("T")
        if kind in (1001, 130) and "v" in response:
            self._battery_v = float(response["v"])
        elif kind == 1002 and set(_GYRO) <= response.keys():
            self._gyro_dps = tuple(float(response[k]) for k in _GYRO)
            now = time.monotonic()
            if self._last_gyro_t is not None and now - self._last_gyro_t <= MAX_GYRO_GAP_S:
                rate = self._gyro_dps[2] - math.degrees(self._gyro_bias_rad_s[2])
                self._heading += rate * (now - self._last_gyro_t)
            self._last_gyro_t = now

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
