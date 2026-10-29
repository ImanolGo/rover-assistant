#!/usr/bin/env python3
"""Manual hardware test: turn in place by integrated gyro, ON THE FLOOR (the robot moves).

Run on the Jetson with ``rover`` stopped and ~0.5 m clear around the robot:

    .venv/bin/python hardware_tests/test_yaw_turn.py --yes --degrees 90

Spins left until the bias-corrected gyro z, integrated, reaches ``--degrees`` (minus
``--lead`` for the coast after the stop), settles, then turns right by the same
amount. The firmware yaw is printed alongside: it follows the magnetometer, which the
robot's own motors/battery dominate, so it barely moves when the robot turns
(2026-10-29: a ~full-circle spin read +5°). On a hard floor nothing below ~0.5 spins
the robot in place (skid-steer scrub), hence the default speed.
"""

from __future__ import annotations

import argparse
import math
import time

from rover.config import load_config
from rover.hal.rover import RoverConfig, SerialRover


def wrap(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


class GyroHeading:
    """Integrate gyro z (deg) from ``T=126`` polls, bias removed (zero-order hold)."""

    def __init__(self, rover: SerialRover):
        self.rover = rover
        self.bias = math.degrees(rover.gyro_bias_rad_s()[2])
        self.angle = 0.0
        self._last = time.monotonic()

    def tick(self) -> None:
        self.rover._send({"T": 126})  # the reply (T=1002) lands on the reader thread
        time.sleep(0.05)
        now = time.monotonic()
        self.angle += (math.degrees(self.rover.gyro_rad_s()[2]) - self.bias) * (now - self._last)
        self._last = now


def turn(rover, heading, sign, degrees, lead, speed, timeout, settle_s=1.5):
    start, yaw0 = heading.angle, rover.heading_deg()
    began = time.monotonic()
    while time.monotonic() - began < timeout and abs(heading.angle - start) < degrees - lead:
        rover.drive(-sign * speed, sign * speed)  # sign +1 = left (as test_rover.py)
        heading.tick()
    rover.stop()
    at_stop = heading.angle - start
    end = time.monotonic() + settle_s
    while time.monotonic() < end:
        heading.tick()
    return at_stop, heading.angle - start, wrap(rover.heading_deg() - yaw0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degrees", type=float, default=90.0)
    parser.add_argument("--lead", type=float, default=10.0, help="stop this early (coast)")
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=4.0)
    parser.add_argument("--yes", action="store_true", help="confirm the floor around it is clear")
    args = parser.parse_args()
    if not args.yes:
        raise SystemExit("Refusing to move: pass --yes only with clear space around the robot.")

    config = load_config()
    rover = SerialRover(RoverConfig(port=config.rover.port, speed_cap=args.speed))
    try:
        heading = GyroHeading(rover)  # bias was calibrated (robot still) in SerialRover()
        for _ in range(20):
            heading.tick()
        for name, sign in (("left", +1), ("right", -1)):
            at_stop, settled, firmware = turn(
                rover, heading, sign, args.degrees, args.lead, args.speed, args.timeout
            )
            print(
                f"{name:5s} gyro: stopped {at_stop:+6.1f}, settled {settled:+6.1f} "
                f"(target {sign * args.degrees:+.0f})   firmware yaw {firmware:+6.1f}",
                flush=True,
            )
    finally:
        rover.stop()
        rover.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
