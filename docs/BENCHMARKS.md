# BENCHMARKS.md — how we measure

## Common rules
- **Power:** MAXN SUPER (`sudo nvpmodel -m 2 && sudo jetson_clocks`). Record
  `nvpmodel -q` in every result.
- **Headless:** no desktop session, docker/containerd stopped unless the test
  needs them, browser closed. Record the baseline MemAvailable before each run.
- **Runs:** 3 warm-up runs, then N ≥ 20 measured runs (N ≥ 10 for anything
  over 2 s). Report p50, p90, max.
- **Memory:** `MemProbe` (bench/common.py) samples every 500 ms:
  - `/proc/meminfo` MemAvailable
  - per-process RSS (psutil)
  - `tegrastats` RAM/SWAP/GR3D_FREQ/CPU/temperatures.

  Report the peak RSS per process, the minimum MemAvailable, and the swap delta.
- **Output:** `bench/results/<name>.json`, plus the raw CSV in
  `bench/results/raw/`. `bench/report.py` fills `STATUS.md`.
- **Unique inputs:** vision benches use a different frame every run (20
  recorded frames from the robot camera in `bench/data/frames/`, plus slight
  perturbation). Never reuse the identical image; that hits the KV cache and
  fakes the speed.

## Per-component protocols

| Bench | Inputs | Metrics |
|---|---|---|
| camera | 30 s capture per mode | fps, dropped frames, CPU% of process, remap ms/frame at 820×616 |
| yolo | 300 frames from `frames/` looped | model-only ms, e2e ms (pre+infer+post), with ByteTrack, fps, RSS, GPU util |
| stt (whisper.cpp) | `assets/audio/*.wav` + 20 `commands/` | latency from end of audio, RTF, WER vs `labels.jsonl`, RSS |
| tts (Piper) | 5 fixed sentences | time to first audio byte, RTF, RSS |
| wake | `HeyRover.wav` ×20 mixed with noise; 10 min of room+motor noise | hit rate, false triggers/hour, CPU% |
| gemma_text | 300-token planner prompt, 60-token output cap | load time, TTFT, prompt tok/s, gen tok/s, server RSS |
| gemma_vision | 20 frames × 3 questions; image tokens 70/140/280; mmproj GPU vs CPU | e2e latency, answer judged correct (manual y/n sheet) |
| gemma_audio | same clips as stt | latency, WER, intent accuracy |
| gemma_tools | 30 commands in `bench/data/tool_cases.jsonl` | tool-name accuracy, arg accuracy, invalid/"thinking" leaks, latency |
| moondream (baseline) | same as gemma_vision | same metrics (for the comparison only) |
| laya | see below | see below |
| coexist (G1) | full stack for 10 min, query every 15 s | pass/fail criteria in PLAN.md G1 |

## Laya protocol

```bash
USE_TF=0 uv run --extra laya python bench/bench_laya.py --device cpu|cuda --dtype fp32|fp16 \
    --checkpoint english|multilingual --questions 1|5|10 [--onnx]
```

1. **Cold load:** time for `laya.load(...)` or `Router(preload=...)`, plus
   RSS/GPU memory after load.
2. **Latency:** 50 calls each with 1, 5 and 10 questions. Report p50/p90.
3. **Accuracy:** `bench/data/selector_cases.jsonl` (40 cases). Compare against
   RuleSelector and Gemma E2B on the same cases.
4. **Contention:** repeat step 2 (cuda, 1 question) while `bench_coexist.py`
   is running. Record the latency and whether G1 still passes.

### `selector_cases.jsonl` schema (write 40; 6 seeds below)

The state is text, as Laya expects. The question is one `choice` over skills.

```json
{"id":"s01","state":"mission=go_to cup(red). target_visible=no. search_steps=2/12. lost_for_s=0. last_skill=SEARCH. elapsed_s=8","label":"SEARCH"}
{"id":"s02","state":"mission=go_to cup(red). target_visible=yes. bearing_deg=-18. h_frac=0.12. track_age_s=1.4. last_skill=SEARCH","label":"APPROACH"}
{"id":"s03","state":"mission=go_to cup(red). target_visible=yes. bearing_deg=2. h_frac=0.41. h_stop=0.40. last_skill=APPROACH","label":"VERIFY"}
{"id":"s04","state":"mission=go_to cup(red). target_visible=no. lost_for_s=1.6. last_bearing_deg=25. last_skill=APPROACH","label":"REACQUIRE"}
{"id":"s05","state":"mission=go_to cup(red). verify_result=no. replans=1/2. last_skill=VERIFY","label":"BACK_OFF"}
{"id":"s06","state":"mission=go_to plant. target_in_vocab=no. target_visible=unknown. search_steps=12/12","label":"ASK_PLANNER"}
```

Laya question used:

```json
{"skill": {"type": "choice", "instructions": "Which robot skill should run next?",
  "criteria": {
    "SEARCH": "target not visible yet, keep turning to look",
    "APPROACH": "target visible and not yet close, drive toward it",
    "REACQUIRE": "target was just lost, look where it was last seen",
    "VERIFY": "target close and centred, stop and confirm",
    "BACK_OFF": "verification failed, reverse a little and retry",
    "ASK_PLANNER": "target unknown to the detector or search exhausted, ask Gemma",
    "DONE": "mission verified complete",
    "ABORT": "too many retries or unsafe, stop"}}}
```
