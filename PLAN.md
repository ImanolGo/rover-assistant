# PLAN.md — step-by-step work order

Legend: **[J]** = must run on the Jetson · **[L]** = laptop is fine ·
**GATE** = stop and record numbers in `STATUS.md` before continuing.

Phase order is deliberate. **Measure first (Phase 1), port second.** If Phase 1
shows the design does not fit in 8 GB, we change the design before writing
code around it.

---

## Phase 0 — Bootstrap (½–1 day)

### 0.1 Repo skeleton [L]
Create the tree exactly as in `ARCHITECTURE.md §7`, with empty modules plus
docstrings. Add:
- `pyproject.toml`: uv-managed, Python 3.10.
- Runtime deps: `numpy opencv-python-headless pyserial fastapi uvicorn httpx
  pyyaml sounddevice soundfile openwakeword silero-vad piper-tts ultralytics`.
- Dev deps: `pytest ruff black`.
- Optional extra `laya`: `laya`.

Add `.gitignore` (models/, *.engine, *.gguf, bench/results/raw/) and a CI
workflow that runs `ruff` + `pytest` on ubuntu-latest.

### 0.2 Copy verbatim assets from legacy [L]
Follow `PORTING.md` section A (calibration, wake-word model, test WAVs, torch
setup script, Wave Rover command reference). Commit them in one commit titled
`import legacy assets`.

### 0.3 Jetson environment [J]
1. Run `scripts/setup_pytorch_jetson.sh`, ported from legacy. It installs the
   JetPack 6 torch 2.8 / torchvision / onnxruntime-gpu / tensorrt wheels.
2. `uv venv --system-site-packages` (needed for system TensorRT and GStreamer
   bindings), then `uv sync`.
3. Verify with `python -c "import torch, tensorrt, cv2; print(torch.cuda.is_available(), cv2.getBuildInformation().count('GStreamer'))"`.
   Torch must see CUDA and OpenCV must have GStreamer. If OpenCV lacks
   GStreamer, fall back to the `gi`/`Gst` appsink approach used in legacy
   `camera_driver.py`.

### 0.4 llama.cpp [J]
Write `scripts/build_llamacpp.sh`. It clones ggml-org/llama.cpp at a
**pinned commit** and builds with:
```
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4 --target llama-server llama-cli llama-mtmd-cli
```
Record the commit hash in `models/MODELS.md`.
- Optional check: whether `ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin`
  can load our mmproj via a mounted volume. If yes, note it as a deployment
  option. The native build stays the default.

### 0.5 Models [J]
Write `scripts/download_models.sh` (idempotent; records SHA256 into
`models/MODELS.md`):

| Model | Source |
|---|---|
| Gemma 4 E2B-it GGUF `Q4_K_M` | ggml-org or unsloth `gemma-4-E2B-it-GGUF` |
| Gemma 4 E2B mmproj (f16) | ggml-org `gemma-4-E2B-it-GGUF` (vision + audio projector) |
| whisper.cpp `ggml-base.en.bin` | ggerganov/whisper.cpp |
| Piper `en_US-lessac-medium.onnx` + `.json` | same voice as legacy |
| YOLO11n `yolo11n.pt` | then `scripts/export_yolo.sh` builds `yolo11n_fp16.engine` **on the Jetson** (engines are not portable across TensorRT versions; legacy engines are not reused) |
| Laya checkpoints `convaiinnovations/laya` (root = English) + `multilingual` subfolder | via `laya` package cache |
| Legacy baseline, benchmark only: Moondream GGUF + mmproj | same files legacy used |

### 0.6 Measurement tooling [L+J]
Create `bench/common.py` with:
- `MemProbe`: samples `/proc/meminfo` MemAvailable, per-PID RSS (psutil) and
  `tegrastats` (RAM, SWAP, GR3D_FREQ, CPU%, temps) every 500 ms into a CSV.
  Use `tegrastats --interval 500 --logfile`.
- `timer()` context manager recording wall time.
- `write_result(name, dict)`: writes `bench/results/<name>.json` with commit
  hash, date, `nvpmodel -q` output, `jetson_clocks --show` summary.

Also `bench/report.py`, which renders all JSON results into the tables of
`STATUS.md §Measurements`.

**Exit:** skeleton committed; `uv run pytest` passes; llama-server starts;
`bench/common.py` produces a CSV.

---

## Phase 1 — Measure every component, then the combination (1–2 days) [J]

### 1.0 Profile the legacy slowdown first [J] (½ day)
Before measuring new components, confirm *why* the legacy stack was slow. See
`ARCHITECTURE.md §10` for the list of suspected causes.

1. Launch the legacy perception pipeline only (camera + undistort + YOLO,
   without depth) from `../local-ai-robot-assistant`.
2. Record per-process CPU% with `pidstat -u -p ALL 1 60` or `top -H`.
3. Profile the camera, undistort and detector nodes for 30 s with
   `py-spy record -o bench/results/raw/legacy_<node>.svg --pid <pid>`.
4. Record camera FPS and detector FPS.
5. Repeat with the depth + point-cloud nodes added.

Write a short table in `STATUS.md §Legacy slowdown`: each cause from
`ARCHITECTURE.md §10`, marked confirmed / not confirmed, with the CPU% or
ms it costs. If a cause is not confirmed, or a new one appears, update
`ARCHITECTURE.md §10`. If the legacy stack can't be launched any more, skip
the launch and profile `camera_driver.py`'s GStreamer string with
`gst-launch-1.0 ... ! fakesink` plus `top`. Record that you did this.

Follow `docs/BENCHMARKS.md` for exact protocol. For each script:
- run in MAXN SUPER;
- warm up 3 runs, then measure N;
- report p50/p90 latency, peak RSS, and the MemAvailable delta.

| # | Script | Measures |
|---|---|---|
| 1.1 | `bench/bench_camera.py` | CSI capture FPS + CPU% for: (a) the legacy pipeline string at 1640×1232, (b) the nvvidconv downscale to 820×616 then `videoconvert`, (c) 820×616 BGRx with the alpha channel dropped in numpy. Also the undistort cost per frame (CPU remap) at 820×616 |
| 1.2 | `bench/bench_yolo.py` | YOLO11n TensorRT FP16, imgsz 640: model-only and end-to-end FPS (pre+post), with and without ByteTrack; RSS. Optional: YOLO26n, YOLOE-11s/26s with a 100-word vocab |
| 1.3 | `bench/bench_stt.py` | whisper.cpp base.en (CUDA): latency and RSS on `assets/audio/*.wav` + 20 recorded commands (see 1.10) |
| 1.4 | `bench/bench_tts.py` | Piper persistent process: time-to-first-audio and real-time factor for 5 sentences; RSS |
| 1.5 | `bench/bench_wake.py` | openWakeWord `hey_roe_ver.onnx`: CPU% while idle-listening; detection on `HeyRover.wav` and false triggers on 10 min of recorded room/motor noise |
| 1.6 | `bench/bench_gemma_text.py` | llama-server E2B: load time, RSS, prompt-eval tok/s, generation tok/s, TTFT for a 300-token planner prompt; with `-c 2048` and `-c 4096` |
| 1.7 | `bench/bench_gemma_vision.py` | image + question → answer. **Unique frame per run** (legacy lesson: identical frames hit the KV cache and fake the speed). Compare `--image-max-tokens 70` vs 140 vs 280, and `--no-mmproj-offload` (projector on CPU) vs GPU. Compare against the legacy Moondream baseline (1.92 s e2e) on the same 20 frames |
| 1.8 | `bench/bench_gemma_audio.py` | audio clip → transcript and → intent JSON. Latency + word error rate vs whisper.cpp on the same clips |
| 1.9 | `bench/bench_gemma_tools.py` | 30 spoken-style commands (text) → tool call. Accuracy of tool name and arguments; count of thinking-mode leaks or non-JSON outputs. Test both `/v1/chat/completions` with `--jinja` tools (reasoning disabled) and raw `/completion` with a few-shot prompt + stop strings |
| 1.10 | `bench/record_commands.py` | helper: records 20 commands through the real mic **with motors running** (wheels up) to `bench/data/commands/`, plus `labels.jsonl` |
| 1.11 | `bench/bench_laya.py` | see Phase 1b |
| 1.12 | `bench/bench_coexist.py` | **GATE G1**, below |

### GATE G1 — coexistence (the most important test)

Start everything that must be resident at runtime, **all at once**:
- camera at 820×616 @ 30 FPS;
- YOLO11n TensorRT + ByteTrack running continuously;
- llama-server E2B with mmproj, all layers on GPU (`-ngl 99`);
- whisper.cpp loaded (unless 1.8 shows Gemma audio is good enough);
- Piper loaded;
- openWakeWord listening.

Then loop for **10 minutes**: every 15 s, send a vision query plus a planner
tool-call to Gemma while YOLO keeps running.

**Pass if all of these hold:**
- no OOM / CUDA allocation failure;
- MemAvailable never below 800 MB;
- swap growth < 100 MB after load;
- YOLO end-to-end ≥ 15 FPS **during** Gemma generation;
- Gemma vision query p90 ≤ 5 s.

**Fallbacks, in order** (apply the first that passes, and record it):
1. `--no-mmproj-offload` (projector on CPU).
2. `-c 2048` and `--image-max-tokens 70`.
3. Drop whisper.cpp, use Gemma audio (only if 1.8 is acceptable).
4. Gemma E2B `Q3_K_M`.
5. Text-only Gemma (no mmproj) + YOLO labels for "what do you see" (the
   Jarvis-home approach) + Moondream on demand for verification only.
6. Offload Gemma to a LAN GPU (see `ARCHITECTURE.md §9`).

### Phase 1b — Laya check (informational; does not block)

`bench/bench_laya.py` [J]:
1. Install `laya` into a **separate** venv extra and set `USE_TF=0`.
2. Measure load time + RSS + GPU memory for the English root checkpoint and
   `multilingual`, in fp32 and fp16, on **CPU** and **CUDA**.
3. Latency p50/p90 for 1, 5 and 10 questions per call. Also run the ONNX path
   (`laya[onnx]`) on CPU.
4. Accuracy: run the 40 skill-selector cases in `bench/data/selector_cases.jsonl`
   (schema + seed examples in `docs/BENCHMARKS.md`; write the remaining cases
   from the skill definitions in `ARCHITECTURE.md §4`). Compare:
   - Laya zero-shot;
   - the rule-based selector (`src/rover/brain/selector.py`, written in Phase 5,
     so re-run this bench then);
   - Gemma E2B answering the same question.
5. Repeat step 2's CUDA latency **while G1's stack is running**, to see
   whether it still fits.

Record everything in `STATUS.md §Laya`. Expected outcome: Laya is fast but
near-chance zero-shot. It becomes a Phase 8 distillation experiment, not a
runtime dependency.

**Exit Phase 1:** all tables in `STATUS.md §Measurements` filled; G1 PASS
(possibly via a fallback); Decision Log entries for STT choice (whisper vs
Gemma audio), image token budget, and quantization.

---

## Phase 2 — HAL port (1–2 days)

Port per `PORTING.md` section B. Each HAL gets a fake and a pytest.

### 2.1 `hal/camera.py` [L+J]
Backends: `csi` (GStreamer string from legacy `camera_driver.py`, **plus** an
`nvvidconv` downscale to 820×616, `drop=true max-buffers=1`), `usb`
(cv2.VideoCapture index), and `file` (video/JPEG folder, loops).

A background thread keeps **only the latest frame** and exposes
`latest() -> (frame, t_capture)`.

### 2.2 `perception/geometry.py` [L]
- Load `config/camera_calibration.yaml` (calibrated at 1640×1232) and scale K
  by 0.5 for 820×616.
- `bearing_deg(u, v)`: uses `cv2.undistortPoints` on the **bbox centre only**.
  No full-frame remap in the control loop.
- `undistort(frame)`: cached remap maps, used only for frames sent to Gemma.
- Tests use known pixel → bearing pairs computed from K.

### 2.3 `hal/rover.py` [L+J]
Port from legacy `uart_motor_controller.py`:
- serial open flags (RTS/DTR False);
- `{"T":1}` writer at 20 Hz;
- 0.5 s watchdog;
- `min_wheel_pwm` deadband mapping (legacy `twist_to_wheel_speeds`, keep its
  unit tests);
- optional T=131 feedback reader (battery voltage, IMU) with gyro-bias
  calibration at start;
- `drive(l, r)`, `stop()`, `battery()`;
- `DryRunRover` logs commands.

`hardware_tests/test_rover.py`, wheels up: spin left, spin right, forward
0.5 s, then check watchdog stop by killing the writer.

### 2.4 `hal/audio.py` [L+J]
- Select mic and speaker by device-name substring from `config/robot.yaml`.
- Capture 16 kHz mono in 80 ms frames.
- Playback resamples to 48 kHz stereo for `UACDemoV1.0`.
- `WavSource` fake.

**Exit:** hardware tests pass on the robot; laptop pytest green;
`STATUS.md` updated.

---

## Phase 3 — Perception loop [L then J] (1–2 days)

1. `perception/detector.py`:
   - Ultralytics YOLO (TensorRT engine on Jetson, `.pt` on laptop) with
     `track(persist=True, tracker="bytetrack.yaml")`.
   - Outputs `Detection(cls, conf, box, track_id, bearing_deg, h_frac)`,
     where `h_frac` is box height / frame height.
2. `perception/color.py`: HSV-histogram attribute check for basic colours.
3. `api/server.py`: FastAPI with `/health`, `/status`, `/video` (MJPEG with
   boxes, track IDs, state name, commanded L/R bars), `POST /cmd` (text command
   injection) and `POST /stop`. Port the legacy resource snapshot and
   thermal-zone reader.
4. Laptop sim: `ROVER_SIM=1 uv run rover` with a webcam and DryRunRover.

**Exit:** debug stream shows stable track IDs; bearing sign correct (object on
the left gives a negative bearing). The performance targets from
`ARCHITECTURE.md §10` are all met and recorded in `STATUS.md`:
- camera CPU < 25% of one core;
- perception ≥ 15 FPS **while Gemma is generating**;
- brain total CPU < 200%;
- detector-to-motor p90 < 120 ms.

If any target is missed, profile with py-spy before changing anything.

---

## Phase 4 — Voice front end [L then J] (1–2 days)

1. `voice/wakeword.py` (openWakeWord + legacy model/threshold), `voice/vad.py`
   (Silero; 800 ms of silence ends the turn, 10 s hard cap), `voice/stt.py`
   (backend from the Phase 1 decision: `whispercpp` via its server/CLI, or
   `gemma` audio).
2. `voice/tts.py`: **one persistent Piper process** streaming PCM, and
   sentence-by-sentence speaking. Plays the legacy `notify_asc.wav` on wake.
3. `brain/intents.py`: port legacy `SIMPLE_COMMANDS` regex. Add
   `go to|find <target>`, `follow me`, `what do you see|describe`,
   `is there <x>`. **`stop` is checked first and bypasses everything.**

**Exit:** wake → transcript → intent printed, p50 end-of-speech → intent
latency recorded; 20 recorded commands ≥ 90% correct intents.

---

## Phase 5 — Brain: planner, skills, state machine [L then J] (3–4 days)

1. `brain/planner.py`: Gemma client over HTTP. Tools:
   - `go_to(target, attributes[])`
   - `follow_person()`
   - `describe(question)`
   - `stop()`
   - `turn(direction, degrees)`
   - `say(text)`

   Use the calling mode that won bench 1.9 (JSON-schema / grammar-constrained
   if needed; port legacy `parse_json_intent` as a fallback parser).
   Regex intents skip the planner; Gemma is called only when regex fails or
   the target needs reasoning.
2. `brain/skills.py`, deterministic controllers:
   - `SEARCH`: committed 35° turn steps, pause 0.5 s, max 12 steps.
   - `APPROACH`: bearing P-controller + box-height distance.
   - `REACQUIRE`: 1 s re-association, then turn toward the last-seen side.
   - `FOLLOW`: bearing + keep `h_frac` near a target band.
   - `BACK_OFF`: reverse 20 cm.
3. `brain/selector.py`: **rule-based** skill selector (≤ 60 lines) over a
   `WorldState` dataclass. Its interface is `select(state) -> Skill`, so Laya
   can be swapped in later.
4. `brain/verify.py`: port legacy `build_verification_prompt` /
   `parse_verification_answer`. Arrival requires **all three**:
   - detector confidence ≥ 0.5 on 2 of 3 frames;
   - colour check passes;
   - Gemma says yes.
5. `brain/mission.py`: the loop `PLAN → (SEARCH|APPROACH|...) → VERIFY →
   DONE | REPLAN`, max 2 replans, spoken progress ("Looking for the red cup…").
6. `telemetry/decisions.py`: append every selector tick
   (`WorldState` as text + chosen skill + outcome) to
   `logs/decisions-YYYYMMDD.jsonl`. This is Laya's future training data.
7. Re-run `bench/bench_laya.py` accuracy with the real rule selector.

**Exit (laptop):** in sim mode with a webcam, "go to the cup" drives the
virtual L/R bars correctly and verification is called.
**Exit (Jetson, wheels up then floor at cap 0.15):** all three behaviours work
end-to-end once.

---

## Phase 6 — On-robot validation (2–3 days) [J]

1. Success sheet `docs/FIELD_TESTS.md`:
   - go_to: 10 objects × 3 start poses (target visible / behind / far);
   - follow: 5 runs of 60 s;
   - describe: 20 questions.

   Record success rate, time to arrive, and failure reason.
2. `bench/soak.py` (port legacy `soak_workload.py` idea): 60 min, all three
   behaviours cycled by `/cmd`, with MemProbe. Pass criteria:
   - 0 crashes;
   - MemAvailable ≥ 800 MB;
   - max junction temperature < 80 °C.
3. Tune gains/thresholds in `config/robot.yaml` only, and document them.

**Exit:** go_to ≥ 70%, follow ≥ 4/5, describe judged correct ≥ 15/20; soak PASS.

---

## Phase 7 — Deploy & clean up (1 day) [J]

1. `deploy/systemd/`: `rover-llama.service` (llama-server) and
   `rover-brain.service` (after llama, `Restart=on-failure`), plus
   `install.sh`. Headless boot (`systemctl set-default multi-user.target`).
   Stop the docker/containerd daemons unless used.
2. `scripts/doctor.sh`: checks devices, models + SHA, power mode, llama health,
   free RAM. One command.
3. Finalise `README.md` quick start (≤ 10 commands from a fresh flash).
4. Archive: add a `LEGACY.md` pointing to the old repo's last commit.

**Exit:** a cold reboot results in the robot answering "Hey Rover, what do you
see?" with no manual steps.

---

## Phase 8 (optional) — Laya distillation experiment

Only after Phase 6 passes.
1. Collect ≥ 2,000 logged decisions from real runs (Phase 5 telemetry).
   Correct the labels where the rule/Gemma choice was wrong.
2. Fine-tune `laya` using the upstream typed-decisions notebook on a free GPU
   (Kaggle/Colab), then fit temperatures.
3. `brain/selector.py` gets a `LayaSelector` behind the same interface, with a
   confidence gate: below the threshold, fall back to rules.
4. A/B on the field-test sheet. Adopt only if success rate ≥ rules **and** G1
   still passes with Laya resident. Record the result either way.
