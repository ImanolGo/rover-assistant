#!/usr/bin/env python3
"""bench 1.7: Gemma vision — image + question -> answer. UNIQUE FRAME PER RUN (legacy lesson).

Uses the three test images, plus random-crop variants, so no two runs share a frame
(legacy Moondream lesson: identical frames hit the prompt cache and fake the speed).
Compares --image-max-tokens 70 (server current) and measures e2e latency + correctness.
"""

from __future__ import annotations

import base64
import io
import json
import statistics
import sys
import time

import httpx
import numpy as np
from PIL import Image

URL = "http://127.0.0.1:8080"
IMAGES = [
    ("assets/images/bus.jpg", "Describe this street scene in one sentence."),
    ("assets/images/demo01.jpg", "How many people are in this image, and what are they wearing?"),
    ("assets/images/bus.jpg", "What colour is the bus?"),
]

# expected keyword checks (loose "correctness")
CHECKS = {
    0: ["pedestrian", "people", "person", "walk"],
    1: ["two", "2", "men", "people"],
    2: ["blue"],
}


def make_unique(path: str, seed: int) -> str:
    """Returns a b64 JPEG of the image with a tiny random crop — defeats prompt caching."""
    img = Image.open(path).convert("RGB")
    rng = np.random.default_rng(seed)
    dx, dy = rng.integers(0, 8, 2)
    w, h = img.size
    img = img.crop((dx, dy, w, h))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def vision_query(image_b64: str, question: str, timeout: float = 60.0) -> tuple[dict, float]:
    t0 = time.perf_counter()
    r = httpx.post(
        f"{URL}/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ],
            "max_tokens": 120,
            "temperature": 0.2,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=timeout,
    )
    wall = time.perf_counter() - t0
    return r.json(), wall


if __name__ == "__main__":
    sys.path.insert(0, "bench")
    from common import write_result

    assert httpx.get(f"{URL}/health", timeout=10).json()["status"] == "ok"

    rows = []
    seed = 0
    for rep in range(5):
        for i, (path, q) in enumerate(IMAGES):
            seed += 1
            b64 = make_unique(path, seed)
            d, wall = vision_query(b64, q)
            t = d.get("timings", {})
            content = d.get("choices", [{}])[0].get("message", {}).get("content", "")
            ok = any(k in content.lower() for k in CHECKS[i])
            rows.append(
                {
                    "img": path.split("/")[-1],
                    "q": q[:30],
                    "wall_s": round(wall, 2),
                    "prompt_ms": round(t.get("prompt_ms", 0)),
                    "correct": ok,
                    "answer": content[:70].replace("\n", " "),
                }
            )
            print(rows[-1])

    walls = [r["wall_s"] for r in rows]
    results = {
        "runs": rows,
        "wall_p50_s": round(statistics.median(walls), 2),
        "wall_p90_s": round(sorted(walls)[int(len(walls) * 0.9)], 2),
        "correct_pct": round(100 * sum(r["correct"] for r in rows) / len(rows)),
    }
    print(json.dumps(results, indent=2))
    print(write_result("p17_gemma_vision_70tok", results))
