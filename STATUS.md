# STATUS.md

**Current phase:** 5 — Brain: planner, skills, state machine (Phases 3–4 recorded)
**Last updated:** 2026-10-01 (Phase 4: voice front end verified live on hardware; HEAD 82235d3)
**Device:** Jetson Orin Nano 8 GB Super · JetPack 6.2 (L4T 36.4.7) · MAXN SUPER

## Progress

| Phase | State | Exit criteria met | Notes |
|---|---|---|---|
| 0 Bootstrap | ✅ | ☑ skeleton ☑ assets ☑ env ☑ llama.cpp ☑ models ☑ tooling | `uv sync` needs `tool.uv.sources` pinning torch to jetson-cu126 (PyPI aarch64 torch = cu130, too new for JP6 driver). First CUDA load needs `drop_caches` (do **not** raise `min_free_kbytes` — see G1). Image assets (robot photo, test frames) in assets/. |
| 1 Measure | ✅ | ☑ legacy profile ☑ benches 1.1–1.9 ☑ G1 (clean re-run: perf ✅, memory deviation accepted) ☑ Laya ☑ STT decision ☐ 1.10 mic recordings | STT = **Gemma audio**; whisper.cpp built (CUDA) but dropped from the runtime. 1.10 recordings still need the mic on the robot (non-blocking). See GATE G1. |
| 2 HAL | ✅ | ☑ camera ☑ geometry ☑ rover ☑ audio ☑ hw: camera / audio (3/3) / rover | 45 laptop tests green; all three `hardware_tests/` ran on the robot (2026-10-01). Camera 30.0 fps, audio through the UACDemo sink, rover wheels-up motions + watchdog pass, gyro bias calibrated per boot. See §Phase 2 HAL. |
| 3 Perception | 🔄 | ☑ tracker ☑ bearing sign ☑ API/video ☑ §10 camera/perception/brain-CPU · ☐ detector→motor p90 (Phase 5) | YOLO11n TRT + ByteTrack stable ids, correct bearing sign, 27–34 fps in-process; FastAPI + dashboard; `ROVER_SIM=1` synthetic sim; 78 laptop tests. See §Phase 3 Perception. |
| 4 Voice | ✅ | ☑ wake ☑ VAD ☑ STT ☑ TTS ☑ intents ≥90% | Live on hardware: "Hey Rover …" → transcript → intent (`go_to`/`describe`). 20-command bench **18/20 = 90%** intents, p50 0.56 s end-of-speech → intent (Piper-synthesized audio; 1.10 mic recordings pending). Wired into `rover.main`. See §Phase 4 Voice. |
| 5 Brain | ⏳ | ☐ planner ☐ skills ☐ selector ☐ verify ☐ sim ☐ robot | |
| 6 Field | ⏳ | ☐ go_to ≥70% ☐ follow ≥4/5 ☐ describe ≥15/20 ☐ soak | |
| 7 Deploy | ⏳ | ☐ systemd ☐ doctor ☐ cold boot | |
| 8 Laya (opt.) | — | | |

## Measurements (filled by `bench/report.py`)

### Legacy slowdown (Phase 1.0)
Legacy stack could not be launched (ROS2 workspace not built); per plan, the
launch was skipped and the causes were profiled directly with
`gst-launch-1.0 ... ! fakesink` + `/proc/<pid>/stat` CPU sampling
(`bench/profile_legacy_camera.py`, `bench/profile_legacy_undistort.py`,
MAXN SUPER, 2026-09-27, commit 359c2d2).

| # | Cause (ARCHITECTURE §10) | Confirmed? | Cost measured | Notes |
|---|---|---|---|---|
| 1 | Full-res CPU `videoconvert` | **Confirmed** | 50.2% of one core at 1640×1232 BGR→BGRx vs 26.7% with HW downscale to 820×616 first; 22.7% if BGRx feeds appsink directly (no `videoconvert` at all) | gst-launch fakesink, 20 s window. Downscale-first halves the cost; skipping videoconvert halves it again |
| 2 | Per-frame CPU undistort | **Confirmed** | 13.5 ms/frame at 1640×1232 (74 FPS ceiling) vs **2.1 ms at 820×616** (471 FPS ceiling) | legacy remapped *every* frame full-res; new design remaps only frames sent to Gemma, points otherwise |
| 3 | 12-process CPU contention | Partially (design-level) | camera alone: 23–56% core; each added ROS process pays interpreter+executor overhead | legacy STATUS known-issue 7 stands; G1 will measure the new stack's total |
| 4 | Inter-node image copies | Not directly measured | 1640×1232 BGR ≈ 6 MB/frame per hop at 30 FPS ≈ 180 MB/s serialisation | eliminated by design (single process); no further work |
| 5 | Depth + point cloud per frame | Confirmed by legacy numbers | 55.1 FPS depth alone but full system dropped to 8–10 FPS | depth deleted in this design; G1 verifies the gain |
| 6 | VLM forced to CPU | Confirmed by legacy numbers | ~20 s/query on CPU vs 1.92 s on GPU (Moondream) | root cause: CUDA OOM; G1 is the fix test |

**Implication already adopted:** camera HAL uses `nvvidconv` downscale to
820×616 and BGRx→numpy (drop alpha) with **no `videoconvert`** (23% core, 30 FPS measured).

### Performance targets (Phase 3)
| Target | Goal | Measured | Pass |
|---|---|---|---|
| Camera process CPU | < 25% of 1 core | **22.7%** (BGRx path, bench 1.1) | ✅ |
| Perception FPS during Gemma generation | ≥ 15 | **54.5–68.6** (GATE G1 run A/B) | ✅ |
| Brain total CPU | < 200% | **123.9%** (perception+API, after OpenMP cap) | ✅ |
| Detector→motor latency p90 | < 120 ms | pending — needs the Phase 5 control loop | ⏳ |

### Phase 2 HAL — hardware verification (Jetson, 2026-10-01, commit 41798d9)

| Check | Result | Notes |
|---|---|---|
| Camera CSI capture | ✅ 30.0 fps @ 820×616 | gi/Gst appsink, `drop=true max-buffers=1` |
| Audio capture + playback | ✅ 3/3 | pulseaudio (`-D pulse`, select by name); UACDemoV1.0 sink RUNNING |
| Rover serial + feedback | ✅ | `/dev/ttyTHS1` @115200, RTS/DTR False; battery **12.15 V**, attitude streaming (`T=1001`) |
| Rover motion (wheels up) | ✅ | spin left, spin right, forward 0.5 s each at cap 0.15 |
| Rover watchdog | ✅ | wheels zero 0.5 s after the last command goes stale |
| Gyro bias calibration | ✅ | `T=126` → `T=1002`; 44 samples over 2 s averaged to **(0.0004, −0.0001, 0.0003) rad/s** |

The gyro bias is **measured each boot, not assumed constant**: this boot's bias is
near zero, not the legacy ~0.36 rad/s, so a hard-coded constant would have injected
drift. Command reference: `docs/wave_rover_json_commands.md`.

### Phase 3 Perception — functional verification (Jetson, 2026-10-01, commit 4a4d901)

Ran the real stack on the robot (`sim=false`, CSI camera, `yolo11n_fp16.engine`):

| Check | Result |
|---|---|
| Camera + engine load | ✅ CSI 1640×1232 → nvvidconv 820×616; TensorRT FP16 engine loaded; 0 OOM |
| Perception rate in-process | ✅ **27–34 fps** (target ≥ 15) |
| Stable track id | ✅ the same object kept `#1` across frames (ByteTrack) |
| Bearing sign | ✅ object on the left → bearing **−43…−45°** |
| Overlay / API | ✅ `/frame` + `/video` MJPEG: box, `#id`, class, confidence, bearing, state, L/R bars |
| Brain CPU | before cap **517%** (5 spin-waiting OMP threads) → after **123.9%** |
| Brain RSS | 1350 MB |
| API | `/health` `/status` `/api/resources` `/frame` `/video` `/cmd` `/stop` `/` (dashboard) ✅ |

Modules: `config.py` (YAML → dataclasses, `ROVER_SIM` override),
`perception/detector.py` (`Detection`, `YoloDetector`, `FakeDetector`),
`perception/color.py` (HSV), `api/server.py` (+ `dashboard.html`), `main.py`
(asyncio app). Laptop sim: `ROVER_SIM=1 .venv/bin/rover` (synthetic frames +
DryRunRover + FakeDetector when torch is absent).

**Deferred:** the `detector→motor p90 < 120 ms` target cannot be measured until
the control loop/skills exist (Phase 5); recorded here so it is not silently
skipped.

### Phase 4 Voice — verification (Jetson, 2026-10-01, commit 82235d3)

Modules: `voice/wakeword.py` (openWakeWord "Hey Rover" + cooldown),
`voice/vad.py` (Silero turn: 800 ms silence ends it, 10 s cap),
`voice/stt.py` (Gemma audio via llama-server), `voice/tts.py` (persistent Piper,
sentence by sentence), `voice/intents.py` (legacy `SIMPLE_COMMANDS` + `go_to` /
`follow` / `describe` / `is_there`, **stop first**), `voice/pipeline.py` (the
turn loop). Wired into `rover.main`; the `stop` intent zeroes the rover at once.

**20-command bench** (`bench/bench_voice.py`; Piper-synthesized audio, because
the 1.10 mic recordings are still pending):

| Metric | Result | Target |
|---|---|---|
| Intent accuracy | **18/20 = 90%** | ≥ 90% ✅ |
| End-of-speech → intent p50 | **0.56 s** | recorded ✅ |
| End-of-speech → intent p90 | 0.67 s | — |

The two misses were STT slips on tiny utterances ("halt" → "Holt", "go home" →
"Gohon"), not regex errors.

**Live mic** (`hardware_tests/test_voice.py`, real wake model + Silero + Gemma
STT + Piper): "Go to the blue bottle" → `go_to` (blue, +0.71 s); "What do you
see?" → `describe` (+0.66 s); "Hey Rover" alone → `unknown`. Notify tone and
spoken acknowledgement both worked. Raw: `bench/results/p4_voice_commands.json`.

### Phase 4 voice v2 — Moonshine STT benchmark + fixes (Jetson, 2026-10-02, commit 75eeb18)

**A1 fixed:** `SpeechSegmenter` now ends a turn after `voice.no_speech_timeout_s`
(3.5 s) with no speech start — discards the turn, returns to LISTENING, logs and
counts `false_wakes` (exposed on `/status`); the turn buffer is hard-capped at
`max_utterance_s` worth of audio in every state.

**B2 benchmark** (`bench/bench_stt_moonshine.py`, synthesized 20 commands; the
real-mic set from `bench/record_commands.py` is still pending a human run):

| Candidate | Intent acc | WER | p50 end→text | p90 | CPU % | Model RSS Δ |
|---|---|---|---|---|---|---|
| Gemma audio (baseline) | 0.90 | 0.29 | 0.35 s | 0.44 s | 1.3% | ~50 MB² |
| Moonshine tiny-streaming | 0.90 | 0.40 | **0.14 s** | **0.18 s** | 123% | **154 MB** |
| Moonshine small-streaming | **0.95** | 0.48 | 0.26 s | 0.33 s | 180% | 373 MB |
| Moonshine medium-streaming | 0.80 | 0.41 | 0.40 s | 0.51 s | 214% | 592 MB |

² Gemma's weights live in llama-server, not the brain process; its RSS is shared.

**Contention** (`bench/contention_load.py`: camera + YOLO 26.6 fps + a continuous
Gemma vision loop; `--limit 6` synthesized commands):

| Candidate | idle p90 | contention p50 | contention p90 |
|---|---|---|---|
| Gemma audio | 0.44 s | 1.24 s | **1.34 s** |
| Moonshine tiny | 0.18 s | 0.40 s | **0.49 s** ✓ |
| Moonshine small | 0.33 s | 0.80 s | **0.88 s** |

Under load Gemma STT **exceeds** the 600 ms bar (1.34 s) because it shares
llama-server's single slot with vision; Moonshine tiny passes it (0.49 s) and
small does not (0.88 s). A clean full-stack `MemAvailable` was not obtained —
llama's vision path failed on the repeat attempt (863 vision errors), a Known
issue 1 flare — but Moonshine adds 154–373 MB *on top of* a stack already at the
~574 MB G1 floor, so the "≥ G1 floor" criterion is expected to fail.

**GATE B3 — NOT adopted; keep Gemma as the runtime backend.** Moonshine tiny
matches Gemma accuracy and is 2.7× faster under load with no GPU, but the gate
requires *all* criteria: small fails the contention p90, and tiny's extra
154 MB is expected to break the full-stack memory floor (unverified, but the
arithmetic on the G1 floor is clear). The real-mic set is also still pending.
Moonshine stays a candidate if memory is freed elsewhere (the brain's
torch/ultralytics ~1.3 GB, ARCHITECTURE §6, is the bigger lever). Recorded:
`bench/results/p4b_stt_contention_load.json`, `p4b_stt_moonshine.json`.

Recorded: `bench/results/p4b_stt_moonshine.json`. Models downloaded by
`scripts/download_models.sh` (tiny/small/medium streaming-en, `.ort`), hashed in
`models/MODELS.md`; package pinned as the `moonshine` extra (`moonshine-voice==0.1.5`).

### Components in isolation
| Component | Config | p50 | p90 | Throughput | Peak RSS | Notes |
|---|---|---|---|---|---|---|
| Camera | 820×616@30 | | | **30.0 fps** | — | **56%** core with numpy alpha-drop per frame; 23% without (BGRx kept). gi/Gst appsink, `drop=true max-buffers=1` |
| Undistort (remap) | 820×616 | 2.1 ms | — | 471 fps equiv. | — | full-res 1640×1232: 13.5 ms — done once per Gemma query only |
| YOLO11n TRT FP16 | 640, e2e + ByteTrack | 27.7 ms | — | **36.2 fps** | 1110 MB | model-only: 37.3 fps / 26.8 ms. Engine exported on-device (`scripts/export_yolo.sh`) |
| whisper.cpp base.en | CUDA | 0.92 s | 0.93 s | — | 708 MB | 2.2 s on CPU (`--no-gpu`) → GPU wins. All 3 test clips transcribe correctly. whisper.cpp @ master, CUDA sm_87 |
| Undistort (remap) | 820×616 | 2.1 ms | — | 471 fps equiv. | — | full-res 1640×1232: 13.5 ms/frame — legacy did this every frame |
| Piper lessac-medium | piper CLI, cold load | — | 2.2 s | RTF 1.16 | 147 MB | cold-load included; a persistent process amortizes load (see bench_tts.py) |
| openWakeWord hey_roe_ver | onnx, CPU | — | — | — | — | HeyRover.wav hit 0.915 (5 frames > 0.5); HeyJarvis.wav 0.001 (no false fire); idle-listening **17% of one core** |
| Gemma E2B text | Q4_K_M, c=2048, thinking off | TTFT **0.09 s** | — | — | 4.7 GB (incl. 1.8 GB file-mapped) | prompt 63 tok/s, gen 27 tok/s; planner answer 1.1 s wall |
| Gemma E2B vision | 70 img tok, mmproj GPU, unique frames | 1.28 s | **1.37 s** | — | — | 5/5/5/5/5 correct on 15 queries (bus scene, people count, colour). p50 wall 1.3 s |
| Gemma E2B vision | mmproj CPU (`--no-mmproj-offload`) | 9.4 s | — | — | +0.2 GB | **rejected**: 3× over the 5 s budget |
| Gemma E2B audio | wav 16 kHz in, mmproj audio | 0.29 s | — | — | — | transcripts correct on all 3 clips (mean WER 0.20 incl. formatting diffs); 10× faster than whisper CLI |
| Gemma E2B tools | 30 spoken-style cases | — | — | — | — | **jinja tools: 83% tool acc, 93% arg acc** · **/completion few-shot: 90% tool acc, 93% arg acc** · 0 thinking leaks either way · p50 ~0.8 s |
| Moondream (baseline) | llama.cpp | — | — | — | — | legacy files kept in models/moondream/ for on-demand comparison |

### GATE G1 — coexistence (clean re-run, MAXN SUPER, 2026-09-29, commits 7b4d5fa/8e20a30)

Stack resident: camera 820×616@30 (nvvidconv downscale, drop=true), YOLO11n TRT
FP16 + ByteTrack continuous, llama-server E2B Q4_K_M + mmproj q8_0 (`-ngl 99 -c
2048 --cache-ram 0 --image-max-tokens 70 --jinja`), Piper resident; every 15 s
one unique-frame vision query + one planner tool call. opencode and every other
dev process had been moved off the Jetson; jtop/ollama stopped; `vm.min_free_kbytes`
left at the kernel default (45056). Two clean-boot runs, differing only in
whisper.cpp residency:

| Criterion | Target | Run A (whisper) | Run B (no whisper) | Pass |
|---|---|---|---|---|
| OOM / CUDA alloc failures | 0 | 0 | 0 | ✅ |
| Min MemAvailable | ≥ 800 MB | **571 MB** | **574 MB** | ❌ deviation |
| Swap growth after load | < 100 MB | 86.8 MB | **−0.2 MB** | ✅ |
| YOLO e2e FPS during generation | ≥ 15 | 68.6 | 54.5 | ✅ |
| Gemma vision p90 under load | ≤ 5 s | 1.07 s | 1.32 s | ✅ |
| Vision queries ok / total | — | 36/36 | 35/35 | ✅ |

**Verdict: performance and functional criteria PASS; the 800 MB MemAvailable
criterion is a documented, operator-accepted deviation.** What the re-run
established:

1. **`vm.min_free_kbytes=1 G` was harmful and is removed.** The first "clean"
   run set it before llama-server; it deflated MemAvailable by ~2 GB (61 MB with
   vs 2078 MB without, llama-server alone) and forced ~1 GB of swap. The real
   NvMap fix is `drop_caches` (+ compaction) alone.
2. **llama-server leaked ~45 MB/min under unique vision queries.** Its default
   prompt cache is 8 GB (`--cache-ram 8192`); unique images were retained. With
   `--cache-ram 0` the server warms ~+382 MB then plateaus (RSS 4890→4892 MB
   over 20 queries). Now set in `config/robot.yaml`.
3. **The bench leaked nvargus clients** (pipeline left PLAYING), which broke the
   next camera session ("Failed to create CaptureSession"). Fixed in
   `bench_coexist.py` (pipeline → NULL on exit, commit 7b4d5fa).
4. **The 800 MB bar is below the Q4_K_M design's floor.** ARCHITECTURE §6
   budgets ~5.9 GB stack + ~1.0 GB OS on a 7.4 GB board ≈ 0.5 GB free; measured
   steady state is 574–623 MB. Fallback #3 (drop whisper) moved avail 235→574 MB
   and swap to 0; KV-cache q8 gave no gain.
5. **Fallback #4 (Q3_K_M) was tried and rejected for quality.** It frees ~520 MB
   but degrades vision: on `bus.jpg` Q4 answers "blue" (5/5) while Q3 answers
   "white and green" (foliage), so "go to the blue bus" would fail. Vision
   quality outweighs the last ~200 MB (operator decision, 2026-09-29).

Raw: `bench/results/raw/g1_run6_cleanboot_nowhisper_cacheram0.json` (Run B, the
accepted config) and `.../g1_run5c_minfree_default_whisper_cacheram0.json` (Run A).
Earlier runs on the same investigation: `g1_run1_clean_minfree1g.json` (sysctl
artifact), `g1_run2c`/`g1_run3` (pre-cache-ram), `g1_run4` (cache leak).

### Laya (Phase 1b, 2026-09-28, commit 31aff08)
| Checkpoint | Device/dtype | Load s | Mem | 1q p50 | 5q p50 | 10q p50 | Acc (40 cases) | Under G1 load |
|---|---|---|---|---|---|---|---|---|
| english | cpu fp32 | 21.6 | 2140 MB | **10.5 s** | 61.7 s | 123.5 s | 28% | — |
| english | cuda fp16 | 12.0 | 3.9 GB RSS / 1.6 GB GPU | **0.077 s** | 0.184 s | 0.344 s | 28% | not re-tested: see verdict |
| multilingual | cpu fp32 | — | — | — | — | — | — | — (CPU cost already disqualifying) |
| multilingual | cuda fp16 | 16.0 | 3.2 GB RSS / 1.3 GB GPU | 0.069 s | 0.104 s | 0.157 s | 18% | — |
| english | onnx cpu | — | — | — | — | — | — | needs manual `export_onnx.py`; skipped (informational) |
| RuleSelector (draft) | — | — | — | <1 ms | — | — | **75%** | — |
| Gemma E2B zero-shot | llama-server | resident | — | 0.46 s | — | — | 55% | — |

**Laya verdict:** fast on GPU (~70 ms) but **near-chance zero-shot on the skill-selection task
(18–28% over 8 classes; chance = 12.5%)**, and unusable on CPU (10.5 s for 1 question —
421M-param ModernBERT). Matches the plan's expected outcome: Laya becomes a **Phase 8
distillation experiment**, not a runtime dependency. The real RuleSelector (Phase 5) is
the runtime selector; a 5-minute offline rule draft already scores 75% on the same cases.
Selector comparison result: `bench/results/p1b_selector_comparison.json`.
Bench bug fixed along the way: multilingual must be loaded via `Agent(subfolder=...)`;
routing through `Router` internals caused a ~10.8 s/call hub round-trip.

### Legacy baseline (from the legacy repo, docs/model_performance.md + STATUS.md, Sep 2026)
| Item | Value |
|---|---|
| YOLO11n TensorRT FP16, MAXN SUPER | 98.2 FPS model loop (in the full ROS system: ~8–10 FPS due to CPU contention) |
| Depth Anything V2 Small TRT | 55.1 FPS |
| Moondream via llama.cpp (unique frames) | 1.92 s vision e2e, 55 tok/s, 2.67 GB RSS |
| Moondream on CPU (coexistence mode) | ~20 s per vision query |
| faster-whisper base.en | ~718 MB RAM |
| Full ROS system soak (no VLM) | 5.57 GB avg / 5.96 GB peak |
| GPU VLM + camera/YOLO/Depth | **did not fit** (CUDA OOM) |

## Decision log
| Date | Decision | Why | Evidence |
|---|---|---|---|
| (kit) | Drop ROS2, depth, full-frame undistort | Legacy could not fit a GPU VLM; CPU contention | Legacy STATUS known issues 6–7 |
| (kit) | Gemma 4 E2B via llama-server replaces Moondream | Tools + vision + audio in one model | to be confirmed by G1 |
| 2026-09-27 | Camera pipeline: no `videoconvert`; BGRx → numpy alpha-drop | `videoconvert` costs ~13% core at 820×616; BGRx straight to appsink is 22.7% vs 50.2% legacy full-res | bench/profile_legacy_camera.py, bench/bench_camera.py |
| 2026-09-27 | llama.cpp pin: v0.5.0 (`d2e54583`) | current release at port start; Gemma 4 supported | models/MODELS.md |
| 2026-09-27 | Gemma E2B is a *thinking* model: `/v1/chat/completions` fills `reasoning_content`, not `content` | observed live: answer empty, thinking in `reasoning_content`; planner/verify parsers must read both (and/or disable reasoning) | llama-server smoke test, chat1 response |
| 2026-09-27 | Jetson first CUDA load needs `drop_caches` + `min_free_kbytes=1G` — **superseded 2026-09-29: `min_free_kbytes` is harmful; use `drop_caches` alone** | NvMap contiguous-alloc ENOMEM at lfb≈50×4MB even with 5 GB available; after drop_caches lfb=147×4MB, load OK in 20 s | /tmp/llama_server logs, Phase 0; see 09-29 rows |
| 2026-09-27 | STT: **defer to clean-boot G1 re-run** (resolved 2026-09-29 → Gemma audio). whisper base.en 0.92 s (708 MB) vs Gemma audio 0.29 s (0 MB extra, WER 0.20 on 3 clips) | both work; memory headroom decides | bench 1.3 vs 1.8 |
| 2026-09-27 | Tool-calling mode: **raw `/completion` few-shot (90%) over jinja tools (83%)** | 30 spoken cases; jinja failed on "find the red cup"→describe, "come along with me"→empty; few-shot's misses are recoverable (say/describe for go_to phrasings). Port `parse_json_intent` as fallback parser either way | bench 1.9 |
| 2026-09-27 | Vision: mmproj stays on **GPU** | CPU mmproj = 9.4 s/query, 3× over the 5 s budget | bench 1.7 + fallback test |
| 2026-09-27 | G1 memory criteria fail in dev env; **re-run on clean headless boot before fallback escalation** | our stack grew +193 MB over 10 min vs 963 MB swap growth; opencode agent (this session) is resident and 1.5–2.9 GB swap pre-used | p112_gate_g1*.json decomposition |
| 2026-09-28 | mmproj switched **F16 → Q8_0** (531 MB file) | F16's largest CUDA alloc (589 MB contiguous) exceeds the boot-time lfb on some boots (137×4MB); Q8_0 loads every time, vision quality unchanged ("What colour is the bus?" → correct, 1.22 s) | llama_server log, models/MODELS.md |
| 2026-09-28 | Laya is **not a runtime dependency**; Phase 8 distillation experiment only | zero-shot 18–28% near chance (12.5%); CPU 10.5 s/1q unusable at 10 Hz; GPU 70 ms fine but accuracy is the blocker; RuleSelector draft 75%, Gemma 55% | p1b_selector_comparison.json, bench_laya.py |
| 2026-09-29 | **STT = Gemma audio**; whisper.cpp dropped from the runtime | whisper costs ~0.4–0.7 GB and a third CUDA process; Gemma audio is 0.29 s vs 0.92 s and memory is the binding constraint | bench 1.3 vs 1.8; G1 runs A/B |
| 2026-09-29 | llama-server runs with `--cache-ram 0` | default 8 GB prompt cache leaked ~45 MB per unique vision query (RSS 4.9→5.06 GB / 10 min); with 0 it warms +382 MB then plateaus | RSS probe, 20 unique queries; G1 run4 vs run6 |
| 2026-09-29 | Do **not** raise `vm.min_free_kbytes`; use `drop_caches` + compaction only | 1 GB min_free deflated MemAvailable ~2 GB (61 MB vs 2078 MB, llama only) and forced ~1 GB swap; `drop_caches` alone loads mmproj and the camera | sysctl A/B; G1 run1 vs run2c |
| 2026-09-29 | G1 memory near-miss **accepted** (MemAvailable ~574 MB < 800 MB); perf + no-OOM + swap pass | ARCHITECTURE §6 budget itself implies ~0.5 GB free on 8 GB; fallback #4 (Q3_K_M) would pass but breaks vision (blue bus → "white and green") | G1 run6; Q3 bus probe |
| 2026-09-29 | Bench must tear down its GStreamer pipeline on exit | leaked nvargus clients broke the following camera session | commit 7b4d5fa |
| 2026-09-29 | Moondream2 + whisper vs Gemma measured; **Gemma kept** | Moondream+whisper is ~1.5 GB lighter (moondream llama-server RSS 2416 MB vs Gemma ~4.6 GB) but vision accuracy was **5/15 vs 15/15** on the bench-1.7 questions (people count → "0", blue bus → "White"); Moondream has no audio or tool-calling, and whisper alone adds ~708 MB | inline bench on `bench_gemma_vision.py` questions, 2026-09-29 |
| 2026-09-29 | Labeled vision set `bench_vision_accuracy.py` (15 small + real images): **Gemma 14/15 (93%)**; Moondream via llama.cpp 4/15 but ~half were HTTP 500s, not model errors | Gemma: colour 8/8, presence 3/3, count 2/3 (only miss: 4 squares vs 3). Moondream on a **fresh** server answers correctly ("Blue" bus, colours) then intermittently returns `failed to process mtmd chunk` | p17b_vision_accuracy_gemma.json, p17b_vision_accuracy_moondream_llamacpp.json |
| 2026-09-29 | "Use llama.cpp directly for Moondream": possible but **not production-ready** with these files | pinned llama.cpp v0.5.0 warns "missing pre-tokenizer", "ffn up/down are swapped", "generation quality will be degraded"; the GGUF has no chat template (needs a hand-written one) and the server throws intermittent mtmd 500s. A GGUF converted for this build would be required; accuracy is then still below Gemma | moondream_server.log; bench_vision_accuracy.py |
| 2026-09-27 | Image token budget: **70 (min=max)** | smallest budget that kept 5/5 vision answers; 140/280 cost prompt time with no accuracy gain | bench 1.7 |
| 2026-10-01 | IMU bias is **calibrated per boot** (poll `T=126`, average `T=1002` gyro), not assumed constant | measured bias this boot is ~0, not the legacy ~0.36 rad/s; a fixed constant would inject drift | on-device 44-sample average → (0.0004, −0.0001, 0.0003) rad/s |
| 2026-10-01 | Two-machine dev loop: git remote `jetson` (`receive.denyCurrentBranch=updateInstead`) + `scripts/dev_sync.sh`; **GitHub stays source of truth** | fast local→robot push without a GitHub round-trip; `updateInstead` refuses to clobber a dirty Jetson tree | this session; STATUS.md §Known issues 8 |
| 2026-10-01 | Cap OMP + torch thread pools to 1 in `rover/__init__` | numpy/OpenCV/torch worker pools **spin-wait**; 5 unnamed threads burned ~85% each doing nothing (brain **517%** CPU) | on-device thread dump + A/B: 517% → 123.9%, fps 27–34 |
| 2026-10-01 | Perception processes only **new** frames (capture-timestamp guard) | the loop re-ran the detector on the same frame at ~1 kHz, which would peg the GPU | sim smoke: 997k frames in seconds → camera-rate |
| 2026-10-01 | STT confirmed live: **Gemma audio** via llama-server | 20 synthesized commands → 18/20 intents, p50 0.56 s; live "go to the blue bottle" / "what do you see" correct | `bench/bench_voice.py`, `hardware_tests/test_voice.py` |
| 2026-10-02 | B3 STT gate **not adopted → keep Gemma**; Moonshine tiny is the candidate | synthesized idle: tiny 0.90 acc/0.18 s p90/154 MB, small 0.95/0.33 s/373 MB, medium 0.80 vs Gemma 0.90/0.44 s. Under load (camera+YOLO+vision): Gemma p90 **1.34 s** (shares llama's slot) vs tiny **0.49 s**, small 0.88 s; small fails contention and tiny's +154 MB is expected to break the 574 MB G1 floor. Real-mic set pending | `bench_stt_moonshine.py`, `bench/contention_load.py`, `p4b_stt_*` |

## Known issues
1. CUDA/NvMap allocation failures — mmproj load (`NvMapMemAllocInternalTagged error 12`) and the camera (`Failed to create CaptureSession` / `(Argus) InsufficientMemory`) — after any heavy NvMap use. Fix (needs sudo): `sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory`, then retry. Do **not** raise `vm.min_free_kbytes` (it deflates MemAvailable ~2 GB and forces swap — see G1). llama-server sometimes needs 2–4 retries; must go into `scripts/doctor.sh` (Phase 7).
2. OpenCV (pip) has no GStreamer on this Jetson — camera HAL uses `gi`/Gst appsink (as legacy did). Confirmed working.
3. `uv sync` must keep `tool.uv.sources` for torch/torchvision or PyPI silently installs a cu130 build the JP6 driver rejects (`torch.cuda.is_available()` = False with no obvious error).
4. Gemma E2B emits thinking in `reasoning_content`; with `"chat_template_kwargs": {"enable_thinking": false}` content is filled correctly. Bench harness uses this; the planner must too.
5. Ops: long-running benches must run in tmux (a bare background process dies with the shell's process group). The gate script also held nvargus clients when it exited without tearing the pipeline down — fixed in `bench_coexist.py` (7b4d5fa).
6. G1 clean re-run (2026-09-29): performance/functional criteria pass, swap stable, no OOM; `MemAvailable` ~574 MB is below the 800 MB bar and is an accepted deviation (Q4_K_M cannot leave >800 MB on 8 GB; Q3_K_M would but degrades vision). The 60-min soak (Phase 6) re-tests this under MemProbe.
7. Desktop pipewire/pulseaudio owns the USB audio devices on this image: PortAudio cannot open `hw:0,0` ("Device unavailable") and direct `plughw` is intermittently busy. The audio HAL selects the PulseAudio sink/source by name and plays/captures through `-D pulse` (`PULSE_SINK`/`PULSE_SOURCE`), falling back to `plughw` when no sound server runs. Verified: UACDemoV1.0 sink RUNNING during playback; USB mic source. Headless production must either keep a sound server or rely on the fallback._
8. `bench/results/*.json` are **tracked**, so running any bench on the Jetson dirties its working tree and blocks `scripts/dev_sync.sh` (`updateInstead` will not clobber it). Commit the new results, or discard them with `scripts/dev_sync.sh --discard-results` before pushing code.
9. The laptop `.venv` runs `pytest`/`ruff` directly, but `uv run`/`uv sync` on x86_64 fail: `tool.uv.sources` pins `torch` to the Jetson cu126 index (aarch64/cp310 wheels only). **Do not** relax that pin — PyPI now ships aarch64 torch wheels (cu130) that the JP6 driver rejects (Known issue 3). The source-marker fallback was tried and collapses to PyPI on this uv version, so the pin stays. Use `.venv/bin/pytest` locally; keep `uv run` for the Jetson. (Laptop sim in Phase 3 must therefore use `.venv/bin/rover`, not `uv run rover`.)
10. Voice turns start capturing at wake detection, so saying "Hey Rover" and then pausing yields an extra `unknown` turn (the wake word itself). Harmless — no action is taken — but a pre-roll trim is a later refinement.
11. Tiny one-word utterances ("halt", "go home") are occasionally mis-transcribed by Gemma audio ("Holt", "Gohon") and fall through to `unknown`; the regex table itself matches them. Longer phrasing is robust (90% on the 20-command bench).
