# rover-assistant

A privacy-first voice + vision assistant for the Waveshare Wave Rover on an
NVIDIA Jetson Orin Nano 8 GB. It is a simplified rewrite of
[`local-ai-robot-assistant`](https://github.com/ImanolGo/local-ai-robot-assistant)
(the "legacy repo").

**What it does:**
- "Hey Rover, go to the red cup": finds the cup, drives to it, and checks it arrived.
- "Hey Rover, what do you see?": answers spoken questions about the camera view.
- "Hey Rover, follow me": follows a person.

**How it differs from the legacy repo:**
- No ROS2. One Python asyncio process ("the brain") talks to one local
  `llama-server` running Gemma 4 E2B.
- No depth model.
- No per-frame full-image undistortion.

## Documents (read in this order)

| File | Purpose |
|---|---|
| `AGENTS.md` | Rules for the coding agent doing the port. Read first. |
| `PLAN.md` | Step-by-step phases with exit criteria. **The work order.** |
| `ARCHITECTURE.md` | Target design: processes, loops, skills, memory budget. |
| `PORTING.md` | File-by-file map of what to take from the legacy repo. |
| `docs/BENCHMARKS.md` | How every component is measured, including Laya. |
| `STATUS.md` | Live progress + measured numbers. Updated after every step. |

## Starting the port

1. Create an empty repo named `rover-assistant` and copy this kit into it.
2. Clone the legacy repo next to it: `../local-ai-robot-assistant`.
3. Hand the coding agent this instruction: *"Read AGENTS.md, then execute
   PLAN.md phase by phase. Update STATUS.md after each step. Stop at every
   GATE and report the numbers."*
