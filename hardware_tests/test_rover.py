#!/usr/bin/env python3
"""Manual hardware test: rover motion + watchdog, wheels-up.

Run on the Jetson with the robot ON A BOX (wheels off the ground):

    .venv/bin/python hardware_tests/test_rover.py --yes

It spins left, spins right, drives forward briefly, then checks that the
watchdog zeroes the wheels when commands go stale.
"""

from __future__ import annotations

import argparse
import time

from rover.hal.rover import RoverConfig, SerialRover


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyTHS1")
    parser.add_argument("--speed", type=float, default=0.15)
    parser.add_argument("--seconds", type=float, default=0.5)
    parser.add_argument("--yes", action="store_true", help="confirm wheels are off the ground")
    args = parser.parse_args()

    if not args.yes:
        raise SystemExit("Refusing to drive: pass --yes only with the wheels raised.")

    rover = SerialRover(RoverConfig(port=args.port, speed_cap=args.speed))
    try:
        time.sleep(1.0)
        print(f"battery: {rover.battery()} V")

        print("spin left…")
        rover.drive(-args.speed, args.speed)
        time.sleep(args.seconds)

        print("spin right…")
        rover.drive(args.speed, -args.speed)
        time.sleep(args.seconds)

        print("forward…")
        rover.drive(args.speed, args.speed)
        time.sleep(args.seconds)

        print("releasing commands; waiting for the 0.5 s watchdog…")
        time.sleep(1.0)
        print("watchdog should have stopped the wheels")
    finally:
        rover.stop()
        rover.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
