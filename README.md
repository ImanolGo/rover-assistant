# Rover Assistant

<p align="center">
  <img src="assets/images/jetson-rover-ai-assistant.jpg" alt="The rover: a Waveshare Wave Rover chassis carrying a Jetson Orin Nano, speaker and fan" width="420">
</p>

<p align="center"><em>One small robot, one big idea: everything it thinks about stays in the room.</em></p>

---

## What is this?

This is a voice- and vision-driven robot assistant that runs **entirely on a
NVIDIA Jetson Orin Nano 8 GB** — no cloud, no accounts, no data leaving your
home network. You talk to it, it talks back, and it drives around your living
room doing what you asked.

Talk to it like a person:

> **"Hey Rover — go to the red cup."**

and it will look around, spot the cup with its camera, drive over, and check
that it actually arrived before proudly announcing it. Ask **"what do you
see?"** and it will describe the scene in plain English. Say **"follow me"**
and it will tag along behind you.

It runs on roughly 6 GB of RAM and about €150 of hobby hardware:

| Part | What it does |
|---|---|
| Waveshare Wave Rover | The chassis, motors and wheel encoders-free firmware |
| Jetson Orin Nano 8 GB (Super) | The brain — see the [hardware notes](docs/jetson_setup.md) |
| IMX219 CSI camera | Where the vision happens |
| USB mic + speaker | Where the talking happens |

## Why a rewrite?

This project is the distilled rewrite of an earlier, working-but-sprawling
ROS2 stack (~21,000 lines). The legacy system was clever, but twelve ROS
nodes fought each other for the CPU, and when a vision-language model finally
moved in, the GPU ran out of memory. The full autopsy lives in
[ARCHITECTURE.md §10](ARCHITECTURE.md).

The fix was subtraction:

- **No ROS2.** One asyncio process — "the brain" — does perception, planning
  and control. About a tenth of the code.
- **One `llama-server` process** hosts Gemma 4 E2B (text, vision and audio in
  one small multimodal model) on the GPU. Nothing else touches CUDA.
- **No depth model.** Distance comes from how tall an object is in the frame.
- **No per-frame undistortion.** We fix single points, not whole images.

Everything is measured, not guessed: every performance claim in
[STATUS.md](STATUS.md) traces back to a script in `bench/` with a date, a
power mode and a commit hash attached.

## How it works

```
 you speak ──▶ wake word ──▶ VAD ──▶ STT ──▶ intents/planner ──▶ skills ──▶ wheels
                (openWakeWord) (Silero) (whisper.cpp)  (Gemma 4 E2B)   (deterministic)
                     ▲                                                       │
                     └──────────── "stop" regex bypasses everything ◀────────┘
```

- **Safety first.** Wheels stop if the brain hiccups for half a second, speed
  is capped low by default, and the word "stop" is handled by a regex *before
  any model runs*.
- **Every hardware device sits behind a small HAL with a fake** — so the whole
  stack runs on a laptop with a webcam and no robot at all (`ROVER_SIM=1`).
- **The cameras, audio and motors are boring on purpose.** The interesting
  decisions live in `src/rover/brain/`, and they are all logged to disk so a
  future model can learn from them.

## Repository layout

```
rover-assistant/
├── src/rover/        the brain: hal/ perception/ voice/ brain/ api/
├── config/           robot.yaml — every tunable in one file
├── bench/            measurement scripts (the numbers in STATUS.md come from here)
├── hardware_tests/   tests that need the real robot, run manually, wheels up
├── tests/            laptop-only pytest — must pass at every commit
├── scripts/          model downloads, llama.cpp build, YOLO export
├── deploy/           systemd units + install script
└── assets/           wake-word clips, UI sounds, test images
```

## Quick start (Jetson)

```bash
git clone https://github.com/ImanolGo/rover-assistant && cd rover-assistant
bash scripts/setup_pytorch_jetson.sh   # CUDA torch/tensorrt wheels
uv venv --system-site-packages && uv sync
bash scripts/build_llamacpp.sh         # builds llama-server (pinned commit)
bash scripts/download_models.sh        # Gemma, whisper, piper, YOLO — ~6 GB
bash scripts/export_yolo.sh            # TensorRT engine, built on-device
uv run rover                           # wheels up first!
```

A laptop works too: `ROVER_SIM=1 uv run rover` substitutes the camera, mic
and motors with fakes.

## Documentation, in reading order

| File | Purpose |
|---|---|
| [STATUS.md](STATUS.md) | Live progress with measured numbers. Start here to see where we are. |
| [PLAN.md](PLAN.md) | The work order: phases, gates, exit criteria. |
| [ARCHITECTURE.md](ARCHITECTURE.md) | The design, including why the old one was slow. |
| [PORTING.md](PORTING.md) | The file-by-file map from the legacy repo. |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | How every component is measured. |
| [AGENTS.md](AGENTS.md) | Ground rules for the coding agent doing the port. |

## Status

The port is in progress — see [STATUS.md](STATUS.md) for the honest, current
state. Phase 0 (environment, models, tooling) is done; Phase 1 (measuring
every component before trusting it) is underway.

## License

Apache-2.0 — see [LICENSE](LICENSE). The legacy repo this work derives from
is [local-ai-robot-assistant](https://github.com/ImanolGo/local-ai-robot-assistant).
