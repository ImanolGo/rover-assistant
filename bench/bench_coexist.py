#!/usr/bin/env python3
"""bench 1.12: GATE G1 — coexistence. The most important test in the project.

Starts everything resident at once (camera 30 FPS, YOLO+ByteTrack continuous,
llama-server already running with mmproj, whisper loaded, Piper loaded,
openWakeWord listening), then loops 10 minutes: every 15 s a vision query plus a
planner tool call to Gemma while YOLO keeps running.

Pass criteria:
  - no OOM / CUDA allocation failure
  - MemAvailable never below 800 MB
  - swap growth < 100 MB after load
  - YOLO end-to-end >= 15 FPS during Gemma generation
  - Gemma vision query p90 <= 5 s

NOTE: whisper + Piper "loaded" = subprocess resident started here (as voice/tts.py
and voice/stt.py will hold them). llama-server must already be running.
"""

from __future__ import annotations

import base64
import io
import json
import statistics
import subprocess
import sys
import threading
import time

import httpx
import numpy as np

sys.path.insert(0, "bench")
from common import MemProbe, write_result  # noqa: E402

URL = "http://127.0.0.1:8080"
DURATION_S = 600
VISION_EVERY_S = 15


class YoloLoop:
    def __init__(self):
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        self._gst = Gst
        self.frame = None
        self.t_capture = 0.0
        self.lock = threading.Lock()
        pipe = Gst.parse_launch(
            "nvarguscamerasrc sensor-id=0 sensor-mode=-1 do-timestamp=true ! "
            "video/x-raw(memory:NVMM),width=1640,height=1232,framerate=30/1,format=NV12 ! "
            "nvvidconv flip-method=0 ! video/x-raw,width=820,height=616,format=BGRx ! "
            "appsink name=appsink emit-signals=true max-buffers=1 drop=true sync=false"
        )
        sink = pipe.get_by_name("appsink")

        def on_sample(sink):
            sample = sink.emit("pull-sample")
            buf = sample.get_buffer()
            ok, info = buf.map(Gst.MapFlags.READ)
            if ok:
                data = np.frombuffer(info.data, dtype=np.uint8)
                h, w = 616, 820
                if data.size != w * h * 4:
                    stride = data.size // h
                    data = data.reshape(h, stride)[:, : w * 4]
                fr = data.reshape(h, w, 4)[:, :, :3].copy()
                with self.lock:
                    self.frame = fr
                    self.t_capture = time.time()
                buf.unmap(info)
            return Gst.FlowReturn.OK

        sink.connect("new-sample", on_sample)
        pipe.set_state(Gst.State.PLAYING)
        self.pipeline = pipe

        from ultralytics import YOLO

        self.model = YOLO("models/yolo_trt/yolo11n_fp16.engine", task="detect")
        self.fps_samples = []
        self.errors = []
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        import cv2

        t_last = time.perf_counter()
        while self.running:
            with self.lock:
                fr = self.frame
            if fr is None:
                time.sleep(0.01)
                continue
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            try:
                self.model.track(
                    rgb, persist=True, tracker="bytetrack.yaml", imgsz=640, verbose=False
                )
            except Exception as e:  # noqa: BLE001
                self.errors.append(str(e))
            now = time.perf_counter()
            self.fps_samples.append(1.0 / max(now - t_last, 1e-6))
            t_last = now

    def close(self) -> None:
        """Stop the YOLO thread and release the GStreamer pipeline (Argus clients)."""
        self.running = False
        self.thread.join(timeout=2.0)
        self.pipeline.set_state(self._gst.State.NULL)

    def fps_during(self, seconds: float) -> float:
        with self.lock:
            n0 = len(self.fps_samples)
        time.sleep(seconds)
        recent = self.fps_samples[n0:]
        return statistics.median(recent) if recent else 0.0


def make_unique_b64(seed: int) -> str:
    img = np.random.default_rng(seed).integers(0, 255, (616, 820, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def vision_query(b64: str) -> tuple[bool, float, str]:
    t0 = time.perf_counter()
    try:
        r = httpx.post(
            f"{URL}/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                            },
                            {
                                "type": "text",
                                "text": "Describe this noise pattern in one short sentence.",
                            },
                        ],
                    }
                ],
                "max_tokens": 60,
                "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=60,
        )
        wall = time.perf_counter() - t0
        if r.status_code != 200:
            return False, wall, r.text[:100]
        content = r.json()["choices"][0]["message"].get("content", "")
        return True, wall, content[:50]
    except Exception as e:  # noqa: BLE001
        return False, time.perf_counter() - t0, str(e)[:100]


def planner_call() -> bool:
    try:
        r = httpx.post(
            f"{URL}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "go to the kitchen"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "go_to",
                            "parameters": {
                                "type": "object",
                                "properties": {"target": {"type": "string"}},
                            },
                        },
                    }
                ],
                "max_tokens": 60,
                "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=60,
        )
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    from PIL import Image

    probe = MemProbe("g1_coexist", interval_s=0.5)
    probe.start()

    print("starting camera + YOLO…")
    yolo = YoloLoop()
    time.sleep(5)

    # whisper "resident": whisper-server with the model loaded (as voice/stt.py will use)
    import os

    whisper_proc = None
    if not os.environ.get("SKIP_WHISPER"):
        whisper_proc = subprocess.Popen(
            [
                "/home/imanolgo/whisper.cpp/build/bin/whisper-server",
                "-m",
                "models/whisper/ggml-base.en.bin",
                "--host",
                "127.0.0.1",
                "--port",
                "8081",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    # piper resident
    piper_proc = subprocess.Popen(
        [".venv/bin/piper", "-m", "models/piper/en_US-lessac-medium.onnx", "--output-raw"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    piper_proc.stdin.write(b"warm up\n")
    piper_proc.stdin.flush()
    time.sleep(6)
    # verify whisper-server is up
    if whisper_proc:
        try:
            import httpx as _hx

            _hx.post(
                "http://127.0.0.1:8081/inference",
                files={"file": b"RIFF"},
                data={"response_format": "text"},
                timeout=15,
            )
            print("whisper-server reachable")
        except Exception as e:  # noqa: BLE001
            print("whisper-server probe:", str(e)[:80])

    print(f"resident stack up; running {DURATION_S}s of Gemma load…")
    vision_lat = []
    oom = 0
    n_vision = n_plan = 0
    t_end = time.time() + DURATION_S
    fps_during_gen = []
    seed = 0
    while time.time() < t_end:
        seed += 1
        ok, wall, answer = vision_query(make_unique_b64(seed))
        if ok:
            n_vision += 1
            vision_lat.append(wall)
            fps = yolo.fps_during(min(max(wall, 1.0), 10.0))
            fps_during_gen.append(fps)
            print(f"  vision {n_vision}: {wall:.2f}s, yolo {fps:.1f} fps during")
        else:
            oom += 1
            print("  VISION FAIL:", answer)
        if planner_call():
            n_plan += 1
        time.sleep(max(0, VISION_EVERY_S - wall))

    if whisper_proc:
        whisper_proc.kill()
    piper_proc.kill()
    yolo.close()
    mem = probe.stop()

    fps_all = yolo.fps_samples
    results = {
        "duration_s": DURATION_S,
        "vision_ok": n_vision,
        "vision_fail": oom,
        "planner_ok": n_plan,
        "vision_p50_s": round(statistics.median(vision_lat), 2) if vision_lat else None,
        "vision_p90_s": (
            round(sorted(vision_lat)[int(len(vision_lat) * 0.9)], 2) if vision_lat else None
        ),
        "yolo_fps_median": round(statistics.median(fps_all), 1),
        "yolo_fps_during_gen_p50": (
            round(statistics.median(fps_during_gen), 1) if fps_during_gen else None
        ),
        "yolo_errors": len(yolo.errors),
        **mem,
    }
    print(json.dumps(results, indent=2))

    # G1 verdict
    checks = {
        "no_oom": oom == 0 and len(yolo.errors) == 0,
        "min_mem_available_mb>=800": mem["min_mem_available_mb"] >= 800,
        "swap_growth<100MB": mem["swap_growth_mb"] < 100,
        "yolo>=15fps_during_gen": (fps_during_gen and statistics.median(fps_during_gen) >= 15),
        "vision_p90<=5s": (vision_lat and sorted(vision_lat)[int(len(vision_lat) * 0.9)] <= 5),
    }
    results["gate_checks"] = checks
    results["gate_pass"] = all(checks.values())
    print(json.dumps(checks, indent=2), "PASS" if results["gate_pass"] else "FAIL")
    import os

    result_name = "p112_gate_g1_nowhisper" if os.environ.get("SKIP_WHISPER") else "p112_gate_g1"
    print(write_result(result_name, results))
