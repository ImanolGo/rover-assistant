# PORTING.md — what to take from the legacy repo

Legacy repo location: `../local-ai-robot-assistant` (≈277 files, ≈21k lines of Python).
Target: < 3k lines in `src/`.

**Rule:** anything touching ROS (`rclpy`, `Node`, publishers, msgs, launch,
`declare_parameter`) is stripped. Keep the logic, drop the plumbing.

## A. Copy verbatim (Phase 0.2)

| Legacy path | New path | Notes |
|---|---|---|
| `config/camera_calibration.yaml` | `config/camera_calibration.yaml` | **The real calibration** (1640×1232, fx≈786, 30 images, 30 Oct 2025). Already included in this kit. ⚠️ Do **not** use `src/perception_nodes/config/camera_calibration.yaml`; that one is a 1920×1080 placeholder with zero distortion. |
| `models/wake_word/hey_roe_ver.onnx` | `models/wake_word/hey_roe_ver.onnx` | Custom "Hey Rover" openWakeWord model (206 KB). Commit it; it is small. |
| `assets/audio/*.wav` | `assets/audio/` | `HeyRover.wav` / `HeyJarvis.wav` are the wake-word test clips. `TheRainInSpain.wav` is the STT test clip. `notify_asc/desc.wav` are UI sounds. |
| `scripts/setup/setup_pytorch_jetson.sh` | `scripts/setup_pytorch_jetson.sh` | Pinned JetPack 6 / cu126 wheels (torch 2.8, torchvision 0.23, onnxruntime-gpu 1.23, tensorrt 10.3). Only adjust paths. |
| `docs/guides/wave_rover_json_commands.md` | `docs/wave_rover_json_commands.md` | Command reference (T=1, 11, 126, 130, 131…). |
| `docs/model_performance.md` (JetPack 6.2 / MAXN sections) | `docs/legacy_baseline.md` | Baseline numbers to beat (see `STATUS.md §Legacy baseline`). |
| `docs/guides/jetson_orin_setup.md` | `docs/jetson_setup.md` | Reference; trim ROS parts. |

## B. Adapt (strip ROS, keep logic)

| Legacy file | New module | Take | Leave |
|---|---|---|---|
| `src/perception_nodes/perception_nodes/camera_driver.py` (`_setup_deepstream_pipeline`) | `hal/camera.py` | The working `nvarguscamerasrc … nvvidconv … appsink drop=true` string. Add the downscale to 820×616 and `max-buffers=1`. | ROS publishing, CameraInfo msgs |
| `…/image_undistort_node.py` | `perception/geometry.py` | `getOptimalNewCameraMatrix` (alpha), `initUndistortRectifyMap` map caching | Per-frame remap in the loop, cv2.cuda path (OpenCV has no CUDA here) |
| `…/object_detector.py` + `tools/conversion/convert_yolo.py` | `perception/detector.py`, `scripts/export_yolo.sh` | Engine path conventions, FP16 export flags (the export becomes ~5 lines of `YOLO().export(format="engine", half=True, imgsz=640)`) | 929-line converter, ROS events |
| `src/actuation_nodes/actuation_nodes/uart_motor_controller.py` | `hal/rover.py` | `_connect_serial` flags, `_send_command`, feedback parsing (T=1001/130/126), `twist_to_wheel_speeds` + `min_wheel_pwm` deadband, watchdog logic | Twist msgs, odometry publishing, services |
| `tests/test_uart_motor_kinematics.py` | `tests/test_rover_kinematics.py` | Deadband/kinematics unit tests | — |
| `src/audio_interface_nodes/…/audio_capture_node.py` | `hal/audio.py`, `voice/vad.py` | `_find_audio_device` (select by name), `_normalize_audio`, Silero `VADIterator` usage | Whisper-in-node, ROS events |
| `…/wake_word_detector_node.py` + `config/audio_config_optimized.yaml` | `voice/wakeword.py` | Model path, threshold, speex noise suppression flag, 50 ms prediction step | Node wrapper |
| `…/audio_playback_node.py`, `piper_tts_node.py`, `docs/piper_streaming_implementation.md` | `voice/tts.py` | Streaming approach, 48 kHz stereo output format | Lazy load per request, hardcoded `/home/imanolgo/...` paths |
| `src/behavioral_nodes/…/command_router_node.py` (`SIMPLE_COMMANDS`) | `brain/intents.py` | The regex table (stop first) | ROS routing |
| `src/behavioral_nodes/…/visual_verification_node.py` | `brain/verify.py` | `VERIFICATION_SYSTEM_PROMPT`, `build_verification_prompt`, `parse_verification_answer`, `rotation_duration` | Timers, executors |
| `src/cognitive_core_nodes/…/cognitive_client_node.py` | `brain/planner.py` | `parse_json_intent` (markdown-fence stripping + validation), GBNF grammar notes (one rule per line!) | Ollama fallback, CPU-coexistence mode |
| `…/llama_cpp_bridge.py` | reference only | Settings that mattered: `flash_attn=true`, `n_ctx`, all layers on GPU | In-process llama-cpp-python (we use llama-server) |
| `src/web_interface_nodes/…/web_server.py` | `api/server.py` | `read_thermal_zones`, `resource_snapshot`, `/health` `/status` `/api/resources`, dashboard HTML | ROS subscriptions, SystemState via topics |
| `hardware_tests/test_waveroever_uart.py`, `test_camera_capture.py`, `test_audio_devices.py`, `calibrate_camera.py` | `hardware_tests/` | Keep behaviour and CLI; remove ROS imports; shorten | — |
| `scripts/testing/llm/test_llamacpp_moondream.py` | `bench/bench_gemma_vision.py` | **Unique-frame-per-run** methodology, timing breakdown | Moondream-specific handler (except as baseline) |
| `scripts/testing/integration/soak_workload.py` + `run_soak.sh` | `bench/soak.py` | Soak structure, metrics | ROS launch |
| `hardware_tests/test_thermal_power.py` | reference only | Thermal zone paths, power readings | 1,241 lines; fold what's needed into MemProbe |

## C. Drop (do not port)

- ROS2: `src/robot_interfaces/` (msgs/srvs), all `launch/`, `package.xml`,
  `setup.cfg`, `.colcon/`, `ros2_venv.sh`, `launch_node.sh`, `activate_env.sh`,
  `docs/ros2_venv_usage.md`, `docs/guides/workspace_build_guide.md`.
- Depth: `depth_anything_v2_trt.py`, `depth_estimation_node*.py`,
  `pointcloud_generator.py`, `tools/calibrate_depth.py`,
  `tools/conversion/convert_depth.py`, `scripts/setup/setup_depth.sh`, depth
  benchmarks/tests, `docs/depth_conversion_optimization.md`.
- Ollama / Moondream runtime: `multimodal_llm_node.py`,
  `scripts/setup/setup_ollama.sh`, `docs/guides/ollama_setup.md` (Moondream is
  kept only as a Phase 1 baseline).
- Whisper TensorRT conversion (`tools/conversion/convert_whisper*.py`), because
  whisper.cpp or Gemma audio replaces it.
- DeepStream dewarper config (`config/dewarp_config.txt`,
  `scripts/utils/generate_dewarp_config.py`): not used, since we don't
  undistort full frames.
- `.copilot/`, `.copilot-workspace.yml`, `.vscode/copilot-*`, old
  `Plan.md`/`STATUS.md`/`dir_structure.md`/`docs/implementation_plan.md`
  (superseded; archived by `LEGACY.md`).
- `config/camera_config.yaml`, `uart_config.yaml`, `audio_config*.yaml`,
  `perception_config.yaml`: **values** are merged into `config/robot.yaml`
  (already done in this kit); the files themselves are dropped.

## D. Lessons to preserve (write them as code comments where they apply)

1. Ollama's KV cache fakes vision latency if the same frame is reused. Always
   benchmark with unique frames.
2. The motor deadband is ~0.25–0.3. The Wave Rover has no encoders and the
   IMU gyro bias is ~0.36 rad/s.
3. A GPU VLM plus camera + YOLO + Depth could not coexist on 8 GB. This is why
   depth is gone and G1 exists.
4. OpenCV has no CUDA, so CPU remap costs frames. Undistort points, not images.
5. Detection ran at 98 FPS alone but ~8–10 FPS in the full system. Full diagnosis and fixes: ARCHITECTURE.md §10.
   One process with a latest-frame buffer avoids this.
6. GBNF grammars in the llama.cpp build used: one rule per line.
