#!/usr/bin/env python3
"""Manual hardware test: turn in place by the HAL heading, ON THE FLOOR (the robot moves).

Run on the Jetson with ``rover`` stopped and ~0.5 m clear around the robot:

    .venv/bin/python hardware_tests/test_yaw_turn.py --yes --degrees 90

Spins left until ``SerialRover.heading_deg()`` (integrated, bias-corrected gyro z)
has changed by ``--degrees`` (minus ``--lead`` for the coast after the stop), settles,
then turns right by the same amount. (The firmware yaw is not used: it follows the
robot's own magnetic field; a ~full-circle spin read +5°, 2026-10-29.) On a hard
floor nothing below ~0.5 spins the robot in place (skid-steer scrub), hence the
default speed.
"""

from __future__ import annotations

import argparse
import time

from rover.config import load_config
from rover.hal.rover import RoverConfig, SerialRover


def turn(rover, sign, degrees, lead, speed, timeout, settle_s=1.5):
    start = rover.heading_deg()
    began = time.monotonic()
    while time.monotonic() - began < timeout and abs(rover.heading_deg() - start) < degrees - lead:
        rover.drive(-sign * speed, sign * speed)  # sign +1 = left (as test_rover.py)
        time.sleep(0.05)
    rover.stop()
    at_stop = rover.heading_deg() - start
    time.sleep(settle_s)
    return at_stop, rover.heading_deg() - start


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
        time.sleep(1.0)  # gyro polls running
        for name, sign in (("left", +1), ("right", -1)):
            at_stop, settled = turn(rover, sign, args.degrees, args.lead, args.speed, args.timeout)
            print(
                f"{name:5s} heading: stopped {at_stop:+6.1f}, settled {settled:+6.1f} "
                f"(target {sign * args.degrees:+.0f})",
                flush=True,
            )
    finally:
        rover.stop()
        rover.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
