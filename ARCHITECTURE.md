# ARCHITECTURE.md — rover-assistant v1.0 (target design)

Status: **target**. Numbers marked *(est.)* are replaced by measured values from
Phase 1. If a GATE forces a change, edit this file and log it in `STATUS.md`.

## 1. Goals and non-goals

**Goals:**
- Voice-commanded go-to-object with verification.
- Visual Q&A.
- Person following.
- Fully on-device, runs on a laptop in simulation, boots unattended.

**Non-goals (v1):**
- SLAM or maps.
- Depth estimation.
- Obstacle avoidance beyond low speed and a stop word.
- Multi-room memory.
- ROS2.

## 2. Processes

```
┌──────────────────── Jetson Orin Nano 8 GB (headless, MAXN SUPER) ───────────────────┐
│                                                                                      │
│  rover-brain  (Python 3.10, asyncio, ONE CUDA context: TensorRT YOLO)                 │
│  ┌──────────┐ latest ┌───────────────────┐ WorldState ┌──────────┐ skill ┌─────────┐ │
│  │ Camera   │ frame  │ Perception         │──────────▶│ Selector  │──────▶│ Skills  │ │
│  │ HAL      │──────▶│ YOLO11n TRT +      │  (10 Hz)   │ rules     │       │ P-ctrl, │ │
│  │ csi/usb/ │        │ ByteTrack + bearing│            │ (Laya v2) │       │ search..│ │
│  │ file     │        │ + HSV colour       │            └────▲─────┘       └───┬─────┘ │
│  └──────────┘        └───────────────────┘                 │ mission          │ L/R  │
│  ┌──────────┐ 16k   ┌────────┐ ┌─────┐ ┌────────┐ ┌───────┐ │                 ▼      │
│  │ Audio HAL│─────▶│wakeword│▶│ VAD │▶│  STT   │▶│intents│─┤  ┌────────────────────┐ │
│  └──────────┘       └────────┘ └─────┘ └────────┘ │(regex)│ │  │ Rover HAL 20 Hz    │ │
│        ▲                                          └───┬───┘ │  │ watchdog, deadband │ │
│        │  Piper (persistent)  ◀── say() ───────────────┤     │  │ serial | dryrun    │ │
│        │                                              ▼     │  └─────────┬──────────┘ │
│        │                          ┌────────────────────────┐│            │            │
│        │                          │ Mission / Planner       ├┘   /dev/ttyTHS1 JSON    │
│        │                          │ plan → act → verify     │            ▼            │
│        │                          └───────────┬────────────┘     Wave Rover ESP32     │
│  FastAPI :8000  /health /status /video /cmd /stop            │ HTTP localhost:8080   │
│                                                              ▼                       │
│  rover-llama  llama-server — Gemma 4 E2B Q4_K_M + mmproj (vision+audio), --jinja     │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

**Why two processes:** llama-server is a mature, restartable, benchmarkable
server. Keeping it out of the brain means a Gemma crash never kills the motor
watchdog, and the brain owns exactly one CUDA context.

## 3. Loop rates

| Loop | Rate | Work |
|---|---|---|
| Camera grabber (thread) | 30 Hz | keep latest frame only |
| Perception | as fast as possible (target ≥ 15 Hz on Jetson) | YOLO + track + bearing + colour |
| Selector + skills tick | 10 Hz | choose skill, compute L/R |
| Rover writer (thread) | 20 Hz | send `{"T":1}`; watchdog → zero if stale > 0.5 s |
| Audio | 80 ms frames | wake word, VAD |
| Planner / verify (Gemma) | on demand, 1–5 s | tool call, VQA, verification |

GPU-bound calls run in an executor thread; the 10 Hz tick never awaits Gemma.

## 4. Behaviours

### Missions (from planner tool calls or regex intents)

**`go_to(target, attributes)`**

1. `SEARCH`
   - Stop and look.
   - If there is no matching detection, turn 35° (committed), pause 0.5 s, and repeat up to 12 steps.
   - If the target is not in the detector vocabulary, or a full turn finds nothing, ask Gemma once per pause.
2. `APPROACH`
   - `turn = Kp · bearing/half_fov`, with a dead-band of ±3°.
   - `fwd = v_max · clamp(1 − h_frac/h_stop[class], 0, 1)` while |bearing| < 15°; otherwise turn in place.
   - Low-pass L/R, then the rover HAL applies the deadband mapping.
3. `REACQUIRE`
   - Triggered when the track is lost for more than 1 s.
   - Re-associate the same class near the last box, else turn toward the last-seen side, then go back to SEARCH.
4. `VERIFY`
   - Stop and take 3 frames.
   - Detector conf ≥ 0.5 on 2 of 3 frames, **and** the colour check passes, **and** Gemma yes/no = yes.
   - Fail → `BACK_OFF` + SEARCH (max 2 replans), then report failure by voice.

**`follow_person()`**
- Lock the largest, most central `person` track.
- Bearing P-control, with speed from the error to an `h_frac` target band (calibrated to ~1–1.5 m).
- Lost for more than 1 s → REACQUIRE; lost for more than 5 s → say "I lost you" and idle.

**`describe(question)`**
- Pause motion, take the freshest frame, undistort it, and send it with the question to Gemma.
- Add YOLO labels to the prompt as hints.
- Speak the answer sentence by sentence. Say "Let me look" immediately.

**`stop()`**
- Regex first. Zeroes the rover HAL immediately and cancels the mission.

### Selector interface (swappable)

```python
class Selector(Protocol):
    def select(self, s: WorldState) -> Skill: ...
# WorldState: mission, target_visible, bearing_deg, h_frac, track_age_s,
#             lost_for_s, search_steps, elapsed_s, last_skill, verify_result
# Skill: SEARCH | APPROACH | REACQUIRE | VERIFY | BACK_OFF | ASK_PLANNER | DONE | ABORT
```

- v1: `RuleSelector`.
- v2 (optional, Phase 8): `LayaSelector`, fine-tuned on logged decisions, with a confidence gate that falls back to rules.

## 5. Camera and geometry

- Capture on CSI with `nvarguscamerasrc` at 1640×1232, then `nvvidconv` downscales to **820×616** in hardware.
- The calibration (`config/camera_calibration.yaml`) was made at 1640×1232 with the pinhole + 5-coefficient model. K is scaled by 0.5 at load.
- Control uses `cv2.undistortPoints` on the bbox centre only, giving a bearing in degrees. No per-frame remap.
- Gemma frames are undistorted with cached remap maps, one frame per query.
- Known limit: a 5-coefficient model on a 160° lens is poor near the edges. Only trust distance (box height) once the target is centred. A `cv2.fisheye` recalibration is a later improvement.

## 6. Models and memory budget *(est.; replaced by G1 measurements)*

| Component | Where | Est. memory |
|---|---|---|
| OS headless, no desktop, no docker | — | ~1.0 GB |
| Gemma 4 E2B Q4_K_M (`-c 2048`) | llama-server, GPU | ~3.0 GB |
| mmproj (vision + audio) | llama-server, GPU or CPU | ~1.0 GB |
| YOLO11n TensorRT FP16 + buffers + ByteTrack | brain, GPU | ~0.3 GB |
| whisper.cpp base.en *(dropped if Gemma audio wins)* | brain subprocess, GPU | ~0.2 GB |
| Piper + openWakeWord + Silero + app + camera buffers | brain, CPU | ~0.4 GB |
| **Total** | | **~5.9 GB** (legacy: 5.6 GB *without* any VLM) |

## 7. Repository layout

```
rover-assistant/
├── README.md  AGENTS.md  PLAN.md  ARCHITECTURE.md  STATUS.md  PORTING.md  LEGACY.md
├── pyproject.toml  uv.lock  .gitignore  .github/workflows/ci.yml
├── config/
│   ├── robot.yaml                  # ALL tunables (devices, gains, thresholds, model paths)
│   ├── camera_calibration.yaml     # verbatim from legacy
│   └── prompts/ planner.md verify.md describe.md
├── src/rover/
│   ├── main.py                     # asyncio entry: `uv run rover`
│   ├── config.py                   # load robot.yaml into dataclasses (no framework)
│   ├── hal/        camera.py audio.py rover.py
│   ├── perception/ detector.py geometry.py color.py
│   ├── voice/      wakeword.py vad.py stt.py tts.py
│   ├── brain/      intents.py planner.py skills.py selector.py verify.py mission.py state.py
│   ├── api/        server.py dashboard.html
│   └── telemetry/  decisions.py memprobe.py
├── bench/          common.py bench_*.py record_commands.py soak.py report.py data/ results/
├── hardware_tests/ test_rover.py test_camera.py test_audio.py calibrate_camera.py
├── tests/          (laptop-only pytest)
├── scripts/        setup_pytorch_jetson.sh build_llamacpp.sh download_models.sh export_yolo.sh doctor.sh
├── deploy/systemd/ rover-llama.service rover-brain.service install.sh
├── assets/audio/   HeyRover.wav HeyJarvis.wav TheRainInSpain.wav notify_asc.wav notify_desc.wav
├── models/         (gitignored)  MODELS.md (names, SHA256, llama.cpp commit)
└── docs/           BENCHMARKS.md FIELD_TESTS.md wave_rover_json_commands.md legacy_baseline.md
```

## 8. Safety

- Speed cap `0.15` on the 0.5 scale by default.
- Watchdog of 0.5 s in the brain, plus the firmware heartbeat (verify its timeout in Phase 2).
- The "stop" regex runs before anything else. `/stop` on the API does the same.
- Low battery (from T=131 feedback, below the threshold) → refuse motion and announce it.

## 9. Alternatives kept open

- **LAN offload:** point `planner.url` at a desktop llama-server/vLLM with a larger VLM. The robot falls back to local E2B when that server is unreachable.
- **Moondream** for verification only, if the G1 fallback reaches step 5.
- **YOLOE** open-vocabulary engine, if COCO classes block real tasks. The engine is exported with a fixed household vocabulary.

## 10. Why the legacy stack was slow, and how this design fixes it

**Symptom.** YOLO11n reached **98 FPS on its own** (MAXN SUPER). Inside the full
ROS system, detection and depth dropped to **~8–10 FPS**, and the camera-to-YOLO
path was limited by CPU, not GPU. Legacy `STATUS.md` Known Issue 7 says the
full-system cap was CPU contention among all nodes, not undistortion alone.

The agent must **confirm each cause** in Phase 1 step 1.0 before relying on
this table. Causes marked *likely* are inferred from the code, not yet measured.

| # | Cause | Evidence in legacy repo | Fix in this design | Verified by |
|---|---|---|---|---|
| 1 | **Full-resolution CPU colour conversion.** Every frame went `nvvidconv → BGRx → videoconvert → BGR` at 1640×1232. `videoconvert` runs on the CPU, handling ~2 MP per frame at 30 FPS. | `camera_driver.py` `_setup_deepstream_pipeline` | `nvvidconv` scales to **820×616 in hardware first**, so `videoconvert` handles 4× fewer pixels. Measure the alternative of feeding BGRx directly and dropping the alpha channel with a numpy slice. | bench 1.1: camera process CPU% before and after |
| 2 | **Full-frame undistortion on CPU, every frame.** OpenCV has no CUDA, so `cv2.remap` runs on the CPU at ~15–24 FPS. | Known Issue 7; `image_undistort_node.py` | **No per-frame remap.** The control loop undistorts only the bbox centre (`cv2.undistortPoints`, microseconds). Full undistort happens once per Gemma query. | bench 1.1: remap ms/frame at 820×616; Phase 3 perception FPS |
| 3 | **Too many processes competing for 6 CPU cores.** 12 ROS nodes, each with its own Python interpreter, executor threads and callbacks. | Known Issues 6–7; the 60 min soak ran 12 nodes | **One brain process.** A camera thread keeps only the latest frame (`drop=true max-buffers=1`), and perception always takes the freshest frame instead of a queue. | bench_coexist: total CPU%, YOLO FPS |
| 4 | **Large images copied between processes** *(likely)*. Raw 1640×1232 frames were published, undistorted, then re-published and received by YOLO, depth and point-cloud nodes. Each hop means serialisation and a memory copy of ~6 MB. | topics `/camera/raw` → `/camera/undistorted` → detector/depth | Frames never leave the process; they are passed as numpy references. | py-spy profile of legacy vs new (step 1.0) |
| 5 | **Extra per-frame work nobody needed for the MVP.** Depth Anything V2 on GPU, plus point-cloud generation on CPU, for every frame. | `depth_estimation_node.py`, `pointcloud_generator.py` | **Deleted.** Distance comes from bbox height relative to a per-class stop threshold. | Memory + FPS in G1 |
| 6 | **GPU memory full, so the VLM was pushed to the CPU**, giving ~20 s per query and heavy CPU load during queries. | Known Issue 6 (CUDA OOM, CPU coexistence mode) | Depth removed, one CUDA context in the brain, Gemma in llama-server on the GPU. | GATE G1 |

**Performance targets for the new design** (enforced in PLAN Phases 1 and 3):
- Camera process CPU < 25% of one core at 820×616@30.
- Perception ≥ 15 FPS end to end **while Gemma is generating**.
- Brain total CPU < 200% (of the 6 cores = 600%).
- Detector-to-motor latency p90 < 120 ms.
