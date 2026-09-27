#!/usr/bin/env python3
"""Phase 1.0: profile legacy camera pipeline variants. CPU% + FPS via gst-launch fakesink.

Variants (from ARCHITECTURE.md §10, cause #1):
  a) legacy full-res: nvvidconv -> BGRx 1640x1232 -> videoconvert -> BGR -> fakesink
  b) hardware downscale first: nvvidconv -> BGRx 820x616 -> videoconvert -> BGR -> fakesink
  c) BGRx 820x616 straight to fakesink (no videoconvert; alpha dropped in numpy later)
"""

from __future__ import annotations

import subprocess
import time

VARIANTS = {
    "a_legacy_fullres_1640_bgr": (
        "nvarguscamerasrc sensor-id=0 sensor-mode=-1 do-timestamp=true ! "
        "video/x-raw(memory:NVMM),width=1640,height=1232,framerate=30/1,format=NV12 ! "
        "nvvidconv flip-method=0 ! video/x-raw,width=1640,height=1232,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! fakesink"
    ),
    "b_downscale_820_bgr": (
        "nvarguscamerasrc sensor-id=0 sensor-mode=-1 do-timestamp=true ! "
        "video/x-raw(memory:NVMM),width=1640,height=1232,framerate=30/1,format=NV12 ! "
        "nvvidconv flip-method=0 ! video/x-raw,width=820,height=616,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! fakesink"
    ),
    "c_downscale_820_bgrx_noconvert": (
        "nvarguscamerasrc sensor-id=0 sensor-mode=-1 do-timestamp=true ! "
        "video/x-raw(memory:NVMM),width=1640,height=1232,framerate=30/1,format=NV12 ! "
        "nvvidconv flip-method=0 ! video/x-raw,width=820,height=616,format=BGRx ! fakesink"
    ),
}


def sample_cpu(pid: int, duration_s: float) -> float:
    """CPU% of a process (all threads) over duration, via /proc/<pid>/stat utime+stime."""
    with open(f"/proc/{pid}/stat") as f:
        parts = f.read().rsplit(") ", 1)[1].split()
    u0, s0 = int(parts[11]), int(parts[12])
    t0 = time.monotonic()
    time.sleep(duration_s)
    with open(f"/proc/{pid}/stat") as f:
        parts = f.read().rsplit(") ", 1)[1].split()
    u1, s1 = int(parts[11]), int(parts[12])
    dt = time.monotonic() - t0
    hz = 100.0
    return 100.0 * ((u1 - u0) + (s1 - s0)) / hz / dt


def run_variant(name: str, desc: str, warmup_s: int = 8, measure_s: int = 20) -> dict:
    print(f"\n=== {name} ===\n{desc}")
    proc = subprocess.Popen(
        ["gst-launch-1.0", "-e"] + desc.split(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(warmup_s)  # argus warm-up, first frames
    try:
        cpu = sample_cpu(proc.pid, measure_s)
    finally:
        proc.terminate()
        proc.wait()
    print(f"gst-launch process CPU: {cpu:.0f}% of one core (includes fakesink)")
    return {"cpu_percent_one_core": round(cpu, 1), "duration_s": measure_s}


if __name__ == "__main__":
    import sys

    sys.path.insert(0, "bench")
    from common import write_result

    results = {}
    for name, desc in VARIANTS.items():
        results[name] = run_variant(name, desc)

    # fakesink frees buffers immediately; CPU here is nvvidconv(BGRx in HW) + videoconvert(CPU) + copies.
    # Expected from ARCHITECTURE §10: (a) is 3-4x the CPU of (b); (c) ~0.
    path = write_result("p10_legacy_camera_pipeline", results)
    print(f"\nwrote {path}")
