# AGENTS.md — rules for the coding agent

You are porting a working-but-convoluted ROS2 robot project into a small,
testable Python project.

## Ground truth
- `PLAN.md` is the work order. Do phases **in order**.
- Do not start phase N+1 until phase N's exit criteria are met and recorded in
  `STATUS.md`.
- `ARCHITECTURE.md` is the target design. If reality forces a change (e.g. a
  memory gate fails), update `ARCHITECTURE.md` **and** add an entry to the
  Decision Log in `STATUS.md`. Do not silently diverge.
- `PORTING.md` says what to copy, adapt, or drop from the legacy repo at
  `../local-ai-robot-assistant`. **Read legacy code before rewriting it.** It
  contains hard-won hardware fixes (UART flags, motor deadband, audio devices,
  camera pipeline).

## Hard rules
1. **No ROS2, no rclpy, no colcon, no custom msg types.** Plain Python 3.10, asyncio.
2. **One brain process.** Only one process in the brain may create a CUDA
   context (the YOLO/TensorRT one). Gemma runs in `llama-server`, a separate
   process, reached over HTTP on localhost.
3. **Every hardware thing sits behind a HAL interface with a fake.** Camera
   (`csi` | `usb` | `file`), audio (`alsa` | `wav`), rover (`serial` | `dryrun`).
   Everything must run on a laptop with `ROVER_SIM=1`.
4. **Motors are dangerous.**
   - Default rover speed cap is `0.15` (on the firmware's 0.5 scale).
   - The rover writer must stop the wheels if no command arrives for 0.5 s.
   - "stop" is handled by regex **before** any model runs.
   - First run of any motion code is wheels-up (robot on a box).
5. **Measure, don't guess.** Every performance claim in `STATUS.md` needs a
   number produced by a script in `bench/`, with date, power mode and commit hash.
6. **Keep it small.** Target < 3,000 lines of Python in `src/` (legacy: ~21k).
   Prefer deleting to abstracting. No plugin systems, no config-loader frameworks.
7. **No cloud.** Nothing at runtime may call the internet. Model downloads
   happen only in `scripts/`.
8. Pin versions: model files by exact filename + SHA256 in `models/MODELS.md`,
   and the llama.cpp build by commit hash.
9. Tests: `pytest` must pass on a laptop (no Jetson, no GPU) at every commit.
   Hardware-only tests go in `hardware_tests/` and are run manually.
10. Style: `ruff` + `black` (line length 100), type hints on public functions,
    one short docstring per module.

## When you are blocked
- If a GATE fails, stop. Write the numbers and your proposed fallback in
  `STATUS.md` (use the fallbacks listed in `PLAN.md`), then continue with the
  least-bad fallback.
- Never "fix" an OOM by adding swap and moving on. Swap is only a crash cushion
  during model load.

## Jetson facts you must not forget (from the legacy repo)
- **Device:** Jetson Orin Nano 8 GB Super, JetPack 6.2 (L4T 36.4.7), CUDA 12.6,
  TensorRT 10.3. Benchmark in MAXN SUPER: `sudo nvpmodel -m 2 && sudo jetson_clocks`.
- **Serial:** Wave Rover on `/dev/ttyTHS1` at 115200, with RTS/DTR set False.
  Motion command is `{"T":1,"L":x,"R":y}`, x/y in −0.5…0.5. No wheel encoders.
- **Motor deadband:** the wheels do nothing below |0.25|, start at ~0.3 and
  turn strongly at 0.4. Commands below the floor must be mapped up to
  `min_wheel_pwm` (0.3) or set to zero.
- **IMU:** the base IMU gyro has a ~0.36 rad/s constant bias. Calibrate at
  startup before trusting yaw.
- **Audio:** mic is `USB PnP Sound Device` (16 kHz mono via `plughw`). Speaker
  is `UACDemoV1.0` (native 48 kHz **stereo** S16). ALSA card numbers can change
  across boots, so select devices by name.
- **OpenCV:** built **without CUDA**. Full-frame undistortion and full-res `videoconvert` cost real CPU. See ARCHITECTURE.md §10 (why legacy was slow) before touching the camera or perception code.
  Avoid doing it per frame.
- **Legacy failure mode:** with camera + YOLO + Depth TensorRT engines resident,
  a GPU VLM could not allocate (CUDA OOM). The whole redesign exists to fix
  this. The coexistence GATE in Phase 1 is the most important test in the project.
