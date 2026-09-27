#!/usr/bin/env python3
"""bench 1.8: Gemma audio clip -> transcript. Latency + WER vs whisper.cpp on the same clips.

Gemma 4 E2B's mmproj includes an audio projector: 16 kHz wav -> transcript/intent
without a separate STT model.
"""

from __future__ import annotations

import base64
import json
import statistics
import sys
import time

import httpx

URL = "http://127.0.0.1:8080"
CLIPS = {
    "assets/audio/TheRainInSpain.wav": "the rain in spain stays mainly in the plane",
    "assets/audio/HeyRover.wav": "hey rover",
    "assets/audio/HeyJarvis.wav": "hey jarvis",
}


def wav_to_b64_16k(path: str) -> str:
    """Resample to 16 kHz mono wav, return b64 (Gemma audio needs 16 kHz)."""
    import io

    import scipy.signal
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]
    if sr != 16000:
        data = scipy.signal.resample(data, int(len(data) * 16000 / sr)).astype("float32")
    buf = io.BytesIO()
    sf.write(buf, data, 16000, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def audio_query(b64: str, prompt: str) -> tuple[dict, float]:
    t0 = time.perf_counter()
    r = httpx.post(
        f"{URL}/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": f"data:audio/wav;base64,{b64}",
                                "format": "wav",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": 100,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=120,
    )
    return r.json(), time.perf_counter() - t0


def wer(ref: str, hyp: str) -> float:
    ref_w, hyp_w = ref.split(), hyp.lower().split()
    import difflib

    sm = difflib.SequenceMatcher(None, ref_w, hyp_w)
    return 1.0 - sm.ratio()


if __name__ == "__main__":
    sys.path.insert(0, "bench")
    from common import write_result

    assert httpx.get(f"{URL}/health", timeout=10).json()["status"] == "ok"

    rows = []
    for path, ref in CLIPS.items():
        b64 = wav_to_b64_16k(path)
        walls, texts = [], []
        for _ in range(3):
            d, wall = audio_query(b64, "Transcribe this audio exactly. Output only the transcript.")
            walls.append(wall)
            texts.append(d.get("choices", [{}])[0].get("message", {}).get("content", "").strip())
        text = texts[-1]
        rows.append(
            {
                "clip": path.split("/")[-1],
                "transcript": text[:60],
                "wer": round(wer(ref, text), 2),
                "wall_p50_s": round(statistics.median(walls), 2),
            }
        )
        print(rows[-1])

    results = {
        "runs": rows,
        "mean_wer": round(sum(r["wer"] for r in rows) / len(rows), 3),
        "wall_p50_s": round(statistics.median(r["wall_p50_s"] for r in rows), 2),
    }
    print(json.dumps(results, indent=2))
    print(write_result("p18_gemma_audio", results))
