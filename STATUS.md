# STATUS.md

**Current phase:** 1 — Measure (1.0–1.2 done; 1.3–1.12 in progress)
**Last updated:** 2026-09-27 (commit 359c2d2 + README/assets update)
**Device:** Jetson Orin Nano 8 GB Super · JetPack 6.2 (L4T 36.4.7) · MAXN SUPER

## Progress

| Phase | State | Exit criteria met | Notes |
|---|---|---|---|
| 0 Bootstrap | ✅ | ☑ skeleton ☑ assets ☑ env ☑ llama.cpp ☑ models ☑ tooling | venv also holds image assets (robot photo, test frames). Env note: `uv sync` needs `tool.uv.sources` pinning torch to the jetson-cu126 index (PyPI aarch64 torch is cu130, newer than the JP6 driver). Jetson memory fragmentation required `sudo drop_caches` + `min_free_kbytes=1G` before the first CUDA model load. |
| 1 Measure | 🔄 | ☑ legacy profile ☐ all benches ☐ **G1** ☐ Laya | 1.0–1.2 done; whisper.cpp built (CUDA) for 1.3 |
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
| Undistort (remap) | 820×616 | | | | | per-frame cost |
| YOLO11n TRT FP16 | 640, e2e + ByteTrack | | | fps | | |
| whisper.cpp base.en | CUDA | | | RTF | | WER |
| Piper lessac-medium | persistent | TTFA | | RTF | | |
| openWakeWord hey_roe_ver | | | | | | hit %, FA/h |
| Gemma E2B text | Q4_K_M, c=2048 | TTFT | | tok/s | | |
| Gemma E2B vision | 70 img tok, mmproj GPU | | | | | correct % |
| Gemma E2B vision | 70 img tok, mmproj CPU | | | | | |
| Gemma E2B audio | | | | | | WER, intent % |
| Gemma E2B tools | 30 cases | | | | | tool %, arg %, leaks |
| Moondream (baseline) | llama.cpp | | | | | correct % |

### GATE G1 — coexistence
| Criterion | Target | Measured | Pass |
|---|---|---|---|
| OOM / CUDA alloc failures | 0 | | |
| Min MemAvailable | ≥ 800 MB | | |
| Swap growth after load | < 100 MB | | |
| YOLO e2e FPS during Gemma generation | ≥ 15 | | |
| Gemma vision p90 under load | ≤ 5 s | | |
| Fallback applied | none | | |

### Laya
| Checkpoint | Device/dtype | Load s | Mem | 1q p50 | 5q p50 | 10q p50 | Acc (40 cases) | Under G1 load |
|---|---|---|---|---|---|---|---|---|
| english | cpu fp32 | | | | | | | — |
| english | cuda fp16 | | | | | | | |
| multilingual | cpu fp32 | | | | | | | — |
| multilingual | cuda fp16 | | | | | | | |
| english | onnx cpu | | | | | | | — |
| RuleSelector | — | — | — | | — | — | | — |
| Gemma E2B | — | — | — | | — | — | | — |

**Laya verdict:** _(fill after Phase 1b / Phase 5 re-run)_

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
| | STT: whisper.cpp vs Gemma audio | | bench 1.3 vs 1.8 |
| | Image token budget | | bench 1.7 |
| | Tool-calling mode (jinja vs /completion) | | bench 1.9 |

## Known issues
1. `sudo` required twice in Phase 0: drop_caches (memory fragmentation) and min_free_kbytes. If a cold boot still OOMs on model load, add these to `scripts/doctor.sh`.
2. OpenCV (pip) has no GStreamer on this Jetson — camera HAL must use `gi`/Gst appsink (as legacy did). Confirmed working.
3. `uv sync` must keep `tool.uv.sources` for torch/torchvision or PyPI silently installs a cu130 build the JP6 driver rejects (torch.cuda.is_available() = False with no obvious error)._
