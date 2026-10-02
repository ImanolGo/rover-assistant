# Removing torch/ultralytics from the brain — investigation (no code yet)

Status: **investigation only.** Opened 2026-10-02 because the brain's
torch/ultralytics footprint (~1.3 GB RSS, ARCHITECTURE §6) is the biggest lever
for freeing memory and letting a CPU STT (Moonshine) fit under the memory floor.

## What torch is actually used for

| Use | Where | What it needs |
|---|---|---|
| Silero VAD | `voice/vad.py` (`load_silero`) | a 2 MB JIT/ONNX model, 512-sample windows |
| YOLO11n TensorRT + ByteTrack | `perception/detector.py` (`YoloDetector`) | TRT engine execution + letterbox/NMS + tracker |
| (superseded) Gemma STT | `voice/stt.py` | already HTTP to llama-server — no torch |

Everything else (numpy, cv2, openWakeWord) is already torch-free. openWakeWord
and Piper already run on onnxruntime, which is resident regardless.

## The three replacements

1. **Silero VAD via onnxruntime.** `silero-vad` ships `silero_vad.onnx`; the
   `VADIterator` logic (window buffering, thresholds, min-silence hangover) is
   ~60 lines and already half-reimplemented in `SpeechSegmenter`. Drop the
   `torch` import in `vad.py` and call an `onnxruntime.InferenceSession`.
   **Estimated: removes the torch import for VAD; ~40 lines changed.**

2. **YOLO TensorRT without ultralytics.** Load the already-exported
   `yolo11n_fp16.engine` with the `tensorrt` Python API (or `cuda` bindings):
   allocate execution context + bindings, letterbox the 820×616 frame to 640,
   run inference, decode the YOLOv8/11 head, run NMS (cv2.dnn.NMSBoxes or a
   numpy implementation), scale boxes back. ~200–250 lines including the
   pre/post-processing. **Removes ultralytics + torch + the torch CUDA runtime.**

3. **Tracker.** Replace `model.track(persist=True, tracker="bytetrack.yaml")`
   with a small ByteTrack/IOU tracker (~120–180 lines) keyed on class + IOU,
   keeping the same `Detection` output (stable `track_id`). The current
   verification of stable ids (Phase 3) gives a regression baseline.

## Estimated savings and cost

| Item | Now | After |
|---|---|---|
| Brain RSS (torch/ultralytics/cuda) | ~1.3 GB | ~0.25 GB (TRT runtime + ORT) |
| **Memory freed** | | **~1.0–1.1 GB** |
| src/ lines added | | ~350–450 (VAD 40, TRT 250, tracker 150) |
| Risks | | TRT I/O binding, NMS/scale correctness, tracker stability |

A ~1 GB saving would move full-stack `MemAvailable` from ~574 MB to ~1.5 GB,
which is enough for Moonshine (154–373 MB) **and** restores comfortable headroom
for the G1 floor. It also removes the last torch CUDA context from the brain
(only llama-server's CUDA remains), simplifying the "one CUDA context" story in
ARCHITECTURE §2 (the brain's context becomes TensorRT-only).

## Recommendation

Do it **after** Phase 5/6 (it is a perception refactor with real regression
risk), triggered if/when the memory floor blocks a needed feature (Moonshine, a
second model, or the 30-min soak's headroom). Track it as a Phase 7 hardening
item; the `doctor.sh`/soak work will show whether the floor is actually binding.

This investigation adds no code; line estimates are to be refined when scoped.
