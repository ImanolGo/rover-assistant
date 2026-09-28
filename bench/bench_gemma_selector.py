#!/usr/bin/env python3
"""bench 1b step 3: Gemma E2B on the same 40 selector cases (comparison with Laya)."""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, "bench")
from common import write_result  # noqa: E402

URL = "http://127.0.0.1:8080"

PROMPT = """You are the skill selector of a home robot.
Choose exactly one skill for the state below.
Skills:
- SEARCH: target not visible yet, keep turning to look
- APPROACH: target visible and not yet close, drive toward it
- REACQUIRE: target was just lost, look where it was last seen
- VERIFY: target close and centred, stop and confirm
- BACK_OFF: verification failed, reverse a little and retry
- ASK_PLANNER: target unknown to the detector or search exhausted, ask Gemma
- DONE: mission verified complete
- ABORT: too many retries or unsafe, stop
Answer with ONLY the skill name.

State: {state}
Skill:"""


def run() -> dict:
    assert httpx.get(f"{URL}/health", timeout=10).json()["status"] == "ok"
    cases = [
        json.loads(line)
        for line in Path("bench/data/selector_cases.jsonl").read_text().splitlines()
        if line.strip()
    ]
    correct = 0
    misses = []
    walls = []
    for c in cases:
        t0 = time.perf_counter()
        r = httpx.post(
            f"{URL}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": PROMPT.format(state=c["state"])}],
                "max_tokens": 8,
                "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=60,
        )
        walls.append(time.perf_counter() - t0)
        got = (
            r.json()["choices"][0]["message"].get("content", "").strip().split()[0]
            if r.status_code == 200
            else "HTTP_ERR"
        )
        got = got.strip('."')
        if got == c["label"]:
            correct += 1
        else:
            misses.append({"id": c["id"], "want": c["label"], "got": got})
        time.sleep(0.2)

    return {
        "accuracy_pct": round(100 * correct / len(cases)),
        "misses": misses,
        "n_misses": len(misses),
        "wall_p50_s": round(statistics.median(walls), 2),
        "n": len(cases),
    }


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: v for k, v in r.items() if k != "misses"}, indent=2))
    for m in r["misses"][:8]:
        print(" MISS:", m)
    print(write_result("p1b_gemma_selector_40", r))
