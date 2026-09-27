#!/usr/bin/env python3
"""bench 1.6: llama-server E2B text — load time, RSS, prompt-eval tok/s, generation tok/s, TTFT.

Assumes llama-server is ALREADY RUNNING with the config args (rover-llama).
Usage: python3 bench/bench_gemma_text.py [ctx_size]
"""

from __future__ import annotations

import json
import statistics
import sys
import time

import httpx

URL = "http://127.0.0.1:8080"
PLANNER_PROMPT = (
    "You are the planner of a home robot. The user says: "
    '"um, could you bring me the, uh, the red thing I drink coffee from?" '
    "Available tools: go_to(target, attributes[]), follow_person(), describe(question), "
    "stop(), turn(direction, degrees), say(text). "
    "Respond with exactly one JSON tool call, no prose. Think about synonyms and "
    "disfluencies, then output the tool call. Keep your reasoning under 100 words."
)


def one_call(ctx_note: str, n: int = 5) -> dict:
    lat_first = []
    gen_speeds = []
    prompt_speeds = []
    ttfts = []
    for _ in range(n):
        t0 = time.perf_counter()
        r = httpx.post(
            f"{URL}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": PLANNER_PROMPT}],
                "max_tokens": 150,
                "temperature": 0.2,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=120,
        )
        wall = time.perf_counter() - t0
        d = r.json()
        t = d["timings"]
        ttfts.append(t.get("prompt_ms", 0) / 1000)
        prompt_speeds.append(t["prompt_per_second"])
        gen_speeds.append(t["predicted_per_second"])
        lat_first.append(wall)
        content = d["choices"][0]["message"].get("content", "")
    return {
        "ctx": ctx_note,
        "wall_p50_s": round(statistics.median(lat_first), 2),
        "ttft_p50_s": round(statistics.median(ttfts), 2),
        "prompt_eval_tok_s": round(statistics.median(prompt_speeds), 1),
        "gen_tok_s": round(statistics.median(gen_speeds), 1),
        "answer_sample": content[:80].replace("\n", " "),
        "n": n,
    }


if __name__ == "__main__":
    sys.path.insert(0, "bench")
    from common import write_result

    h = httpx.get(f"{URL}/health", timeout=10)
    assert h.json()["status"] == "ok", h.text
    slots = httpx.get(f"{URL}/props", timeout=10).json().get("total_slots")

    results = {"slots": slots, "run": one_call("server-current")}
    print(json.dumps(results, indent=2))
    print(write_result("p16_gemma_text", results))
