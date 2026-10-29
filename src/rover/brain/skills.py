"""brain.skills: deterministic motion controllers (PLAN 5.2, ARCHITECTURE §4).

Per-tick controllers are pure ``(...) -> (left, right)`` functions on the firmware
scale; the rover HAL applies the caps and the deadband. Turning in place is
never proportional: on a hard floor a skid-steer spin stalls below ~0.5, so
in-place turns are committed :func:`spin` calls stopped by the gyro heading
(STATUS 2026-10-29). Signs: bearing + = target right of centre; heading + = left.
"""

from __future__ import annotations

import time
from typing import Any, Callable

SPIN_LEAD_DEG = 10.0  # stop this early: the robot coasts ~3-13° at 0.5
BACK_OFF_S = 0.8  # (tune) ~20 cm in reverse at the deadband floor; no encoders


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def steer(bearing_deg: float, fwd: float, half_fov_deg: float, control: Any) -> tuple:
    """Arc toward ``bearing_deg`` while moving ``fwd`` (P on bearing, ±dead-band)."""
    if abs(bearing_deg) <= control.bearing_deadband_deg:
        turn = 0.0
    else:
        turn = control.kp_turn * control.v_max * _clamp(bearing_deg / half_fov_deg, -1.0, 1.0)
    return fwd + turn, fwd - turn


def approach(bearing_deg: float, h_frac: float, h_stop: float, half_fov_deg: float, control):
    """``(left, right)`` to close on a target, or ``None``: re-aim with a spin first."""
    if abs(bearing_deg) >= control.approach_max_bearing_deg:
        return None
    # Not proportional: the HAL zeroes < 0.15 and floors the rest at 0.30, so a
    # slowing ramp stopped the robot at ~57% of h_stop, short of VERIFY.
    fwd = control.v_max if h_frac < h_stop else 0.0
    return steer(bearing_deg, fwd, half_fov_deg, control)


def follow(bearing_deg: float, h_frac: float, half_fov_deg: float, control):
    """Keep ``h_frac`` inside ``control.follow_h_frac``, steering only while moving.

    In range there is no steering (it would be a weak proportional spin: opposite
    wheels that stall on the floor, "back and forward" wheels-up); ``None`` means
    re-aim with a committed spin because the person is far off-centre.
    """
    low, high = control.follow_h_frac
    fwd = control.v_max if h_frac < low else (-control.v_max if h_frac > high else 0.0)
    if fwd:
        return steer(bearing_deg, fwd, half_fov_deg, control)
    return None if abs(bearing_deg) >= control.approach_max_bearing_deg else (0.0, 0.0)


def spin(
    rover: Any,
    degrees: float,
    speed: float,
    cancelled: Callable[[], bool] = lambda: False,
    timeout_s: float = 4.0,
) -> float:
    """Committed in-place turn by ``degrees`` (+ = left); returns the heading change.

    Blocks; stops early on ``cancelled()`` (a spoken stop) or the timeout.
    """
    start = rover.heading_deg()
    sign = 1.0 if degrees > 0 else -1.0
    goal = max(0.0, abs(degrees) - SPIN_LEAD_DEG)
    began = time.monotonic()
    while abs(rover.heading_deg() - start) < goal and time.monotonic() - began < timeout_s:
        if cancelled():
            break
        rover.drive(-sign * speed, sign * speed)
        time.sleep(0.05)
    rover.stop()
    return rover.heading_deg() - start


def back_off(rover: Any, speed: float, cancelled: Callable[[], bool] = lambda: False) -> None:
    """Reverse ~20 cm (timed; the rover has no wheel encoders)."""
    end = time.monotonic() + BACK_OFF_S
    while time.monotonic() < end and not cancelled():
        rover.drive(-speed, -speed)
        time.sleep(0.05)
    rover.stop()
