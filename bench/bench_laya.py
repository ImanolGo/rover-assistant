#!/usr/bin/env python3
"""bench 1b (Phase 1b): Laya skill-selector check — informational, does not block.

Protocol (docs/BENCHMARKS.md):
  1. cold load time + RSS/GPU mem for english / multilingual on cpu/cuda (fp32 cpu, fp16 cuda)
  2. latency p50/p90 for 1, 5, 10 questions per call (50 calls each)
  3. accuracy on bench/data/selector_cases.jsonl (40 cases) — Laya zero-shot
Run with .venv-laya/bin/python (separate venv, USE_TF=0).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import psutil

sys.path.insert(0, "bench")
from common import write_result  # noqa: E402

SKILL_QUESTION = {
    "skill": {
        "type": "choice",
        "instructions": "Which robot skill should run next?",
        "criteria": {
            "SEARCH": "target not visible yet, keep turning to look",
            "APPROACH": "target visible and not yet close, drive toward it",
            "REACQUIRE": "target was just lost, look where it was last seen",
            "VERIFY": "target close and centred, stop and confirm",
            "BACK_OFF": "verification failed, reverse a little and retry",
            "ASK_PLANNER": "target unknown to the detector or search exhausted, ask Gemma",
            "DONE": "mission verified complete",
            "ABORT": "too many retries or unsafe, stop",
        },
    }
}

# filler questions so 5/10-question calls resemble the real selector surface
FILLERS = {
    "obstacle": {"type": "noul", "instructions": "Is an obstacle blocking the path?"},
    "battery_ok": {"type": "noul", "instructions": "Is the battery level healthy?"},
    "person_mood": {
        "type": "choice",
        "instructions": "What is the person doing?",
        "criteria": {
            "waiting": "standing still, waiting",
            "moving": "walking or moving",
            "gone": "not present",
        },
    },
    "confidence": {
        "type": "score",
        "instructions": "How confident is the current track?",
        "criteria": ["low", "medium", "high"],
    },
    "verbose": {"type": "noul", "instructions": "Should the robot speak a status update now?"},
    "repeat": {"type": "noul", "instructions": "Is this a repeated identical state?"},
    "urgency": {
        "type": "score",
        "instructions": "How urgent is progress?",
        "criteria": ["low", "medium", "high"],
    },
    "safety": {"type": "noul", "instructions": "Is anything unsafe right now?"},
    "looking": {"type": "noul", "instructions": "Is the target roughly centred in view?"},
}


def questions_for(n: int) -> dict:
    q = {"skill": SKILL_QUESTION["skill"]}
    keys = list(FILLERS)
    for i in range(n - 1):
        q[f"f{i}"] = FILLERS[keys[i % len(keys)]]
    return q


def mem_snapshot() -> dict:
    rss = psutil.Process().memory_info().rss / (1024 * 1024)
    out = {"rss_mb": round(rss)}
    try:
        import torch

        if torch.cuda.is_available():
            out["gpu_mb"] = round(torch.cuda.memory_allocated() / (1024 * 1024))
    except ImportError:
        pass
    return out


def bench_config(checkpoint: str, device: str, nq_list=(1, 5, 10), calls: int = 50) -> dict:
    t0 = time.perf_counter()
    from laya.agent import Agent

    sub = None if checkpoint == "english" else checkpoint
    agent = Agent("convaiinnovations/laya", device=device, subfolder=sub)
    agent.predict("warm up state. target visible yes bearing 5", {"skill": SKILL_QUESTION["skill"]})
    load_s = time.perf_counter() - t0
    mem = mem_snapshot()

    lat = {}
    for nq in nq_list:
        qs = questions_for(nq)
        times = []
        for i in range(calls):
            state = (
                f"mission=go_to cup(red). target_visible={'yes' if i % 2 else 'no'}."
                f" bearing_deg={i % 40 - 20}. h_frac={0.1 + (i % 40) / 100:.2f}."
                f" search_steps={i % 12}/12"
            )
            t1 = time.perf_counter()
            agent.predict(state, qs)
            times.append(time.perf_counter() - t1)
        lat[nq] = {
            "p50_s": round(statistics.median(times), 4),
            "p90_s": round(sorted(times)[int(len(times) * 0.9)], 4),
        }

    # accuracy on the 40 selector cases (skill question only, zero-shot)
    cases = [
        json.loads(line)
        for line in Path("bench/data/selector_cases.jsonl").read_text().splitlines()
        if line.strip()
    ]
    correct = 0
    misses = []
    for c in cases:
        r = agent.predict(c["state"], {"skill": SKILL_QUESTION["skill"]})
        got = r["answers"]["skill"]["choice"]
        if got == c["label"]:
            correct += 1
        else:
            misses.append({"id": c["id"], "want": c["label"], "got": got})

    return {
        "checkpoint": checkpoint,
        "device": device,
        "load_s": round(load_s, 2),
        **mem,
        "latency": lat,
        "accuracy_pct": round(100 * correct / len(cases)),
        "misses": misses[:8],
        "n_misses": len(misses),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="english", choices=["english", "multilingual"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--calls", type=int, default=50)
    args = ap.parse_args()

    r = bench_config(args.checkpoint, args.device, calls=args.calls)
    print(json.dumps(r, indent=2))
    name = f"p1b_laya_{args.checkpoint}_{args.device}"
    print(write_result(name, r))
