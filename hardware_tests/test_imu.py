#!/usr/bin/env python3
"""Manual hardware test: IMU sanity + yaw drift, robot still (no motion).

Run on the Jetson with ``rover`` stopped (it owns the serial port):

    .venv/bin/python hardware_tests/test_imu.py --seconds 60

Polls ``T=126`` (reply ``T=1002``) and prints per-field mean/sd, the accelerometer
magnitude (≈ 1000 mg when the IMU is healthy, whatever the tilt), yaw drift (first
vs last 10% of samples) and how many raw gz samples exceed ±3 °/s.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import serial

FIELDS = ("r", "p", "y", "ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz", "temp")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyTHS1")
    parser.add_argument("--seconds", type=float, default=60.0)
    args = parser.parse_args()

    port = serial.Serial()
    port.port, port.baudrate, port.timeout = args.port, 115200, 0.1
    port.rts = port.dtr = False  # Wave Rover: flow control lines must stay low
    port.open()
    rows: list[dict] = []
    end = time.time() + args.seconds
    try:
        while time.time() < end:
            port.write(b'{"T":126}\n')
            until = time.time() + 0.2
            while time.time() < until:
                line = port.readline().decode(errors="replace").strip()
                if not line.startswith("{"):
                    continue
                try:
                    reply = json.loads(line)
                except ValueError:
                    continue
                if reply.get("T") == 1002:
                    rows.append(reply)
                    break
            time.sleep(0.05)
    finally:
        port.close()

    print(f"samples {len(rows)}")
    if len(rows) < 10:
        print("FAIL: IMU not answering")
        return 1
    for key in FIELDS:
        values = [float(row[key]) for row in rows if key in row]
        if values:
            print(
                f"{key:4s} mean={statistics.mean(values):9.3f} sd={statistics.pstdev(values):7.3f}"
            )
    accel = statistics.mean(
        (float(r["ax"]) ** 2 + float(r["ay"]) ** 2 + float(r["az"]) ** 2) ** 0.5 for r in rows
    )
    yaw = [float(row["y"]) for row in rows]
    tenth = max(1, len(yaw) // 10)
    drift = statistics.mean(yaw[-tenth:]) - statistics.mean(yaw[:tenth])
    spikes = sum(abs(float(row["gz"])) > 3.0 for row in rows)
    print(f"|a| mean {accel:.0f} mg (healthy ≈ 1000)")
    print(f"yaw drift {drift:+.2f} deg over {args.seconds:.0f} s")
    print(f"gz spikes >3 deg/s: {spikes}/{len(rows)}")
    ok = 900 < accel < 1100
    print("OK" if ok else "FAIL: accelerometer magnitude off (bad IMU state; power-cycle the base)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
