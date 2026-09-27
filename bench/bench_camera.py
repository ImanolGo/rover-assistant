#!/usr/bin/env python3
"""bench 1.1: CSI capture FPS + CPU% at 820x616 via gi/Gst appsink (OpenCV has no GStreamer).

Measures the target design pipeline: nvarguscamerasrc 1640x1232 -> nvvidconv HW downscale
to 820x616 BGRx -> appsink; alpha dropped in numpy (no videoconvert).
Reports capture FPS, process CPU%, and undistort-per-query cost for comparison.
"""

from __future__ import annotations

import sys
import threading
import time

import numpy as np

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

Gst.init(None)

PIPELINE = (
    "nvarguscamerasrc sensor-id=0 sensor-mode=-1 do-timestamp=true ! "
    "video/x-raw(memory:NVMM),width=1640,height=1232,framerate=30/1,format=NV12 ! "
    "nvvidconv flip-method=0 ! video/x-raw,width=820,height=616,format=BGRx ! "
    "appsink name=appsink emit-signals=true max-buffers=1 drop=true sync=false"
)


class CameraBench:
    def __init__(self):
        self.pipeline = Gst.parse_launch(PIPELINE)
        self.appsink = self.pipeline.get_by_name("appsink")
        self.appsink.connect("new-sample", self._on_sample)
        self.count = 0
        self.lock = threading.Lock()
        self.last_frame = None

    def _on_sample(self, _sink):
        sample = self.appsink.emit("pull-sample")
        buf = sample.get_buffer()
        ok, info = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.ERROR
        w = info.width if hasattr(info, "width") and info.width else 820
        h = info.height if hasattr(info, "height") and info.height else 616
        data = np.frombuffer(info.data, dtype=np.uint8)
        # BGRx: stride may pad rows; infer from expected size
        exp = w * h * 4
        if data.size >= exp and data.size != exp:
            stride = data.size // h
            data = data.reshape(h, stride)[:, : w * 4]
        frame_bgrx = data.reshape(h, w, 4)
        frame_bgr = frame_bgrx[:, :, :3].copy()  # drop alpha (numpy, CPU-cheap)
        with self.lock:
            self.last_frame = frame_bgr
            self.count += 1
        buf.unmap(info)
        return Gst.FlowReturn.OK

    def run(self, warmup_s: float, measure_s: float) -> dict:
        self.pipeline.set_state(Gst.State.PLAYING)
        time.sleep(warmup_s)
        with self.lock:
            c0 = self.count
        cpu0 = _proc_cpu()
        t0 = time.monotonic()
        time.sleep(measure_s)
        dt = time.monotonic() - t0
        with self.lock:
            c1 = self.count
        cpu = _proc_cpu(cpu0, dt)
        fps = (c1 - c0) / dt
        frame = self.last_frame
        self.pipeline.set_state(Gst.State.NULL)
        return {
            "fps": round(fps, 1),
            "cpu_percent_one_core": round(cpu, 1),
            "frames": c1 - c0,
            "frame_shape": list(frame.shape) if frame is not None else None,
        }


def _proc_cpu(prev=None, dt=None):
    """CPU% of this process (all threads)."""
    with open("/proc/self/stat") as f:
        parts = f.read().rsplit(") ", 1)[1].split()
    jiffies = int(parts[11]) + int(parts[12])
    if prev is None:
        return jiffies
    hz = 100.0
    return 100.0 * (jiffies - prev) / hz / dt


if __name__ == "__main__":
    sys.path.insert(0, "bench")
    from common import write_result

    b = CameraBench()
    r = b.run(warmup_s=8, measure_s=20)
    print(r)
    print(write_result("p11_camera_csi_820x616", r))
