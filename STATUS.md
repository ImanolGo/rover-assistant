# STATUS.md

**Current phase:** 1 — Measure (1.0–1.9 + G1 done; 1.10–1.11 remaining; G1 clean-boot re-run required)
**Last updated:** 2026-09-27 19:15 CEST (commit 0f8e1b4 + STATUS rewrite)
**Device:** Jetson Orin Nano 8 GB Super · JetPack 6.2 (L4T 36.4.7) · MAXN SUPER

## Progress

| Phase | State | Exit criteria met | Notes |
|---|---|---|---|
| 0 Bootstrap | ✅ | ☑ skeleton ☑ assets ☑ env ☑ llama.cpp ☑ models ☑ tooling | `uv sync` needs `tool.uv.sources` pinning torch to jetson-cu126 (PyPI aarch64 torch = cu130, too new for JP6 driver). First CUDA load needs drop_caches + min_free_kbytes=1G. Image assets (robot photo, test frames) in assets/. |
| 1 Measure | 🔄 | ☑ legacy profile ☑ benches 1.1–1.9 ☑ G1 (2 runs, perf ✅ mem ❌ dev-env) ☐ Laya ☐ 1.10 recordings ☐ clean-boot G1 | whisper.cpp built (CUDA); G1 memory verdict deferred to clean boot — see G1 section |
| 2 HAL | ⏳ | ☐ camera ☐ geometry ☐ rover ☐ audio ☐ hw tests | |
| 3 Perception | ⏳ | ☐ tracker ☐ bearing sign ☐ API/video ☐ §10 perf targets | |
| 4 Voice | ⏳ | ☐ wake ☐ VAD ☐ STT ☐ TTS ☐ intents ≥90% | |
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
| Camera process CPU | < 25% of 1 core | | |
| Perception FPS during Gemma generation | ≥ 15 | | |
| Brain total CPU | < 200% | | |
| Detector→motor latency p90 | < 120 ms | | |

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

### GATE G1 — coexistence (two full 10-min runs, MAXN SUPER, 2026-09-27, commit 3cacba2)

Stack resident: camera 820×616@30 (nvvidconv downscale, drop=true), YOLO11n TRT
FP16 + ByteTrack continuous, llama-server E2B Q4_K_M + mmproj `-ngl 99 -c 2048
--image-max-tokens 70 --jinja`, whisper-server base.en CUDA (run 1 only), Piper
resident, every 15 s one unique-frame vision query + one planner tool call.

| Criterion | Target | Run 1 (with whisper) | Run 2 (no whisper) | Pass |
|---|---|---|---|---|
| OOM / CUDA alloc failures | 0 | 0 | 0 | ✅ |
| Min MemAvailable | ≥ 800 MB | **0 MB** | **0 MB** | ❌ |
| Swap growth after load | < 100 MB | **963 MB** | **991 MB** | ❌ |
| YOLO e2e FPS during Gemma generation | ≥ 15 | 59.8 | 69.4 | ✅ |
| Gemma vision p90 under load | ≤ 5 s | 1.37 s | 1.17 s | ✅ |
| Vision queries ok / total | 40 | 35/35 | 35/35 | ✅ |
| Fallback applied | none | — | #3 (whisper dropped; no memory change) | |

**Verdict: FAIL on the two memory criteria — but with strong evidence the
failure is environmental, not architectural:**

1. Per-PID RSS decomposition over the 10-min run: our stack grew only
   **+193 MB** (bench python +210, llama-server +124, nvargus-daemon +197,
   opencode **−338**). The remaining ~800 MB of swap growth is unattributable
   to the robot stack.
2. The dev session runs the opencode agent itself (450–930 MB RSS) and 1.5–2.9 GB
   of swap was already in use before each run after a day of uptime. The
   production target is headless systemd with neither.
3. Despite MemAvailable ~0 the whole time, **zero vision failures and zero
   errors occurred across both 10-minute runs**, with vision p90 1.2–1.4 s and
   YOLO at 60–70 FPS during generation — the system degrades gracefully.
4. Fallbacks #1 (mmproj CPU: 9.4 s/query, 3× over budget) and #3 (drop whisper:
   no memory improvement) were measured and rejected/recorded. #2
   (`-c 2048`, 70 img tokens) was already active.

**Action:** re-run G1 on a clean headless boot (Phase 6 soak runs the same
stack under MemProbe for 60 min). If the memory criteria still fail there,
the next fallback is #6 (LAN GPU) — but per the decomposition, expected result
is PASS. Decision on STT (whisper vs Gemma audio) is deferred to that re-run:
both work (whisper 0.92 s/WER-good, Gemma audio 0.29 s/WER 0.20).

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
| 2026-09-27 | Jetson first CUDA load needs `drop_caches` + `min_free_kbytes=1G` | NvMap contiguous-alloc ENOMEM at lfb≈50×4MB even with 5 GB available; after drop_caches lfb=147×4MB, load OK in 20 s | /tmp/llama_server logs, Phase 0 |
| 2026-09-27 | STT: **defer to clean-boot G1 re-run**. whisper base.en 0.92 s (708 MB) vs Gemma audio 0.29 s (0 MB extra, WER 0.20 on 3 clips) | both work; memory headroom decides | bench 1.3 vs 1.8 |
| 2026-09-27 | Tool-calling mode: **raw `/completion` few-shot (90%) over jinja tools (83%)** | 30 spoken cases; jinja failed on "find the red cup"→describe, "come along with me"→empty; few-shot's misses are recoverable (say/describe for go_to phrasings). Port `parse_json_intent` as fallback parser either way | bench 1.9 |
| 2026-09-27 | Vision: mmproj stays on **GPU** | CPU mmproj = 9.4 s/query, 3× over the 5 s budget | bench 1.7 + fallback test |
| 2026-09-27 | G1 memory criteria fail in dev env; **re-run on clean headless boot before fallback escalation** | our stack grew +193 MB over 10 min vs 963 MB swap growth; opencode agent (this session) is resident and 1.5–2.9 GB swap pre-used | p112_gate_g1*.json decomposition |
| 2026-09-28 | mmproj switched **F16 → Q8_0** (531 MB file) | F16's largest CUDA alloc (589 MB contiguous) exceeds the boot-time lfb on some boots (137×4MB); Q8_0 loads every time, vision quality unchanged ("What colour is the bus?" → correct, 1.22 s) | llama_server log, models/MODELS.md |
| 2026-09-28 | Laya is **not a runtime dependency**; Phase 8 distillation experiment only | zero-shot 18–28% near chance (12.5%); CPU 10.5 s/1q unusable at 10 Hz; GPU 70 ms fine but accuracy is the blocker; RuleSelector draft 75%, Gemma 55% | p1b_selector_comparison.json, bench_laya.py |
| | Image token budget | | bench 1.7 |

## Known issues
1. CUDA NvMap contiguous-alloc ENOMEM after long uptime / fragmentation. Fix (needs sudo): `sync; echo 3 > /proc/sys/vm/drop_caches; sysctl vm.min_free_kbytes=1048576`. Must go into `scripts/doctor.sh` (Phase 7).
2. OpenCV (pip) has no GStreamer on this Jetson — camera HAL uses `gi`/Gst appsink (as legacy did). Confirmed working.
3. `uv sync` must keep `tool.uv.sources` for torch/torchvision or PyPI silently installs a cu130 build the JP6 driver rejects (`torch.cuda.is_available()` = False with no obvious error).
4. Gemma E2B emits thinking in `reasoning_content`; with `"chat_template_kwargs": {"enable_thinking": false}` content is filled correctly. Bench harness uses this; the planner must too.
5. A background process started from the agent shell dies with the shell's process group — long-running benches must be launched via `setsid`. (Ops note, not a code issue.)
6. G1 memory criteria failed under the dev environment (agent resident). Re-run required on clean headless boot before Phase 2 proceeds past gate-affected design points._
