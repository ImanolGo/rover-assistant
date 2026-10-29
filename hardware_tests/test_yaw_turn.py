#!/usr/bin/env python3
"""Manual hardware test: turn in place by IMU yaw, ON THE FLOOR (the robot moves).

Run on the Jetson with ``rover`` stopped and ~0.5 m clear around the robot:

    .venv/bin/python hardware_tests/test_yaw_turn.py --yes --degrees 90

Spins left until the firmware yaw has changed by ``--degrees``, settles, then spins
right by the same amount. Prints yaw before/after each turn and the settle drift
(motor current disturbing the magnetometer would show as yaw moving while still);
the operator compares the physical turn with the reported one. Each turn gives up
after ``--timeout`` s; the 0.5 s watchdog applies as always.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rover.config import load_config
from rover.hal.rover import RoverConfig, SerialRover


def wrap(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def settle(rover: SerialRover, seconds: float, log: list) -> float:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        log.append((time.monotonic(), rover.heading_deg(), 0.0))
        time.sleep(0.05)
    return rover.heading_deg()


def turn(rover: SerialRover, sign: int, degrees: float, speed: float, timeout: float, log):
    start = rover.heading_deg()
    began = time.monotonic()
    turned = 0.0
    while time.monotonic() - began < timeout:
        rover.drive(-sign * speed, sign * speed)  # sign +1 = left (as test_rover.py)
        time.sleep(0.05)
        turned = wrap(rover.heading_deg() - start)
        log.append((time.monotonic(), rover.heading_deg(), sign * speed))
        if abs(turned) >= degrees:
            break
    rover.stop()
    return turned, time.monotonic() - began


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degrees", type=float, default=90.0)
    parser.add_argument("--speed", type=float, default=0.30)
    parser.add_argument("--timeout", type=float, default=6.0)
    parser.add_argument("--yes", action="store_true", help="confirm the floor around it is clear")
    parser.add_argument("--log", default="/tmp/yaw_turn.json")
    args = parser.parse_args()
    if not args.yes:
        raise SystemExit("Refusing to move: pass --yes only with clear space around the robot.")

    config = load_config()
    rover = SerialRover(RoverConfig(port=config.rover.port, speed_cap=args.speed))
    log: list = []
    try:
        time.sleep(1.0)
        if rover.heading_deg() is None:
            raise SystemExit("no yaw from the base (feedback off?)")
        before = settle(rover, 2.0, log)
        left, t_left = turn(rover, +1, args.degrees, args.speed, args.timeout, log)
        after_left = settle(rover, 2.0, log)
        right, t_right = turn(rover, -1, args.degrees, args.speed, args.timeout, log)
        after_right = settle(rover, 2.0, log)
    finally:
        rover.stop()
        rover.close()

    print(f"yaw before            {before:8.2f}")
    print(f"left turn  reported {left:+7.2f} deg in {t_left:.2f} s")
    print(f"  settled {after_left:8.2f}  (total turn {wrap(after_left - before):+.2f})")
    print(f"right turn reported {right:+7.2f} deg in {t_right:.2f} s")
    print(f"  settled {after_right:8.2f}  (net vs start {wrap(after_right - before):+.2f})")
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log).write_text(json.dumps(log))
    print(f"log: {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
