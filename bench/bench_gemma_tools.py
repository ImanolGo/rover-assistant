#!/usr/bin/env python3
"""bench 1.9: 30 spoken-style commands -> tool call. Accuracy of tool name + args.

Compares two modes:
  A) /v1/chat/completions with --jinja tools (reasoning disabled)
  B) raw /completion with a few-shot prompt + stop strings
Counts thinking-mode leaks and non-JSON outputs.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
import time

import httpx

URL = "http://127.0.0.1:8080"

# 30 spoken-style commands with expected (tool, args-substring) targets
CASES = [
    ("go to the kitchen", "go_to", "kitchen"),
    ("could you go to the sofa please", "go_to", "sofa"),
    ("head over to the table", "go_to", "table"),
    ("find the red cup", "go_to", "cup"),
    ("i want the blue bottle, bring it here", "go_to", "bottle"),
    ("navigate to the door", "go_to", "door"),
    ("come to the bedroom", "go_to", "bedroom"),
    ("follow me", "follow_person", ""),
    ("stay behind me and keep up", "follow_person", ""),
    ("come along with me", "follow_person", ""),
    ("what do you see right now", "describe", ""),
    ("describe the room", "describe", ""),
    ("what's in front of you", "describe", ""),
    ("is there a chair anywhere", "describe", "chair"),
    ("do you see any person around", "describe", "person"),
    ("stop", "stop", ""),
    ("stop stop stop", "stop", ""),
    ("freeze", "stop", ""),
    ("halt right now", "stop", ""),
    ("turn left ninety degrees", "turn", "left"),
    ("rotate right 45 degrees", "turn", "right"),
    ("look to your left", "turn", "left"),
    ("turn around", "turn", "180"),
    ("say hello to everyone", "say", "hello"),
    ("tell me the time out loud", "say", ""),
    ("announce that lunch is ready", "say", "lunch"),
    ("um, the, uh, coffee thing", "go_to", "cup"),
    ("hey can you grab my charger", "go_to", "charger"),
    ("where is my phone", "go_to", "phone"),
    ("look for the potted plant", "go_to", "plant"),
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "go_to",
            "description": "Drive to a named object or place in the home.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "object or place name"},
                    "attributes": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["target"],
            },
        },
    },
    {"type": "function", "function": {"name": "follow_person", "description": "Follow the person.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "describe", "description": "Answer a question about what the camera sees.", "parameters": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}}},
    {"type": "function", "function": {"name": "stop", "description": "Stop all motion immediately.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "turn", "description": "Turn in place.", "parameters": {"type": "object", "properties": {"direction": {"type": "string", "enum": ["left", "right"]}, "degrees": {"type": "number"}}, "required": ["direction"]}}},
    {"type": "function", "function": {"name": "say", "description": "Speak a sentence aloud.", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
]

FEWSHOT = """You convert a spoken robot command into exactly one JSON tool call. Output ONLY the JSON object, nothing else.
Tools: go_to(target[, attributes]), follow_person(), describe(question), stop(), turn(direction[, degrees]), say(text).
Examples:
User: go to the kitchen
{{"tool": "go_to", "target": "kitchen"}}
User: follow me
{{"tool": "follow_person"}}
User: what do you see
{{"tool": "describe", "question": "what do you see"}}
User: stop
{{"tool": "stop"}}
User: turn left ninety degrees
{{"tool": "turn", "direction": "left", "degrees": 90}}
User: say hello
{{"tool": "say", "text": "hello"}}
User: {cmd}
"""


def bench_jinja() -> list[dict]:
    rows = []
    for cmd, tool, arg in CASES:
        t0 = time.perf_counter()
        r = httpx.post(
            f"{URL}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": cmd}],
                "tools": TOOLS,
                "max_tokens": 200,
                "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=60,
        )
        wall = time.perf_counter() - t0
        d = r.json()
        msg = d["choices"][0]["message"]
        leak = bool(msg.get("reasoning_content"))
        tool_calls = msg.get("tool_calls") or []
        name = tool_calls[0]["function"]["name"] if tool_calls else ""
        args = tool_calls[0]["function"].get("arguments", "") if tool_calls else ""
        rows.append(
            {
                "cmd": cmd,
                "want": tool,
                "got": name,
                "args": str(args)[:60],
                "arg_ok": arg.lower() in str(args).lower() if arg else True,
                "leak": leak,
                "wall_s": round(wall, 2),
            }
        )
    return rows


def bench_completion() -> list[dict]:
    rows = []
    for cmd, tool, arg in CASES:
        prompt = FEWSHOT.format(cmd=cmd)
        t0 = time.perf_counter()
        r = httpx.post(
            f"{URL}/completion",
            json={
                "prompt": prompt,
                "n_predict": 60,
                "temperature": 0.0,
                "stop": ["\nUser:", "\n\n"],
                "cache_prompt": False,
            },
            timeout=60,
        )
        wall = time.perf_counter() - t0
        text = r.json()["content"].strip()
        m = re.search(r"\{.*\}", text, re.S)
        parsed = json.loads(m.group(0)) if m else {}
        name = parsed.get("tool", "")
        args = json.dumps({k: v for k, v in parsed.items() if k != "tool"})
        rows.append(
            {
                "cmd": cmd,
                "want": tool,
                "got": name,
                "args": args[:60],
                "arg_ok": arg.lower() in args.lower() if arg else True,
                "leak": "thinking" in text.lower() or "reasoning" in text.lower(),
                "wall_s": round(wall, 2),
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    tool_ok = sum(r["got"] == r["want"] for r in rows)
    arg_ok = sum(r["arg_ok"] for r in rows)
    return {
        "tool_acc_pct": round(100 * tool_ok / len(rows)),
        "arg_acc_pct": round(100 * arg_ok / len(rows)),
        "leaks": sum(r["leak"] for r in rows),
        "wall_p50_s": round(statistics.median(r["wall_s"] for r in rows), 2),
    }


if __name__ == "__main__":
    sys.path.insert(0, "bench")
    from common import write_result

    assert httpx.get(f"{URL}/health", timeout=10).json()["status"] == "ok"

    print("mode A: jinja tools")
    a = bench_jinja()
    sa = summarize(a)
    print(sa)
    bad = [r for r in a if r["got"] != r["want"] or not r["arg_ok"]]
    for r in bad[:5]:
        print("  MISS:", r)

    print("mode B: /completion few-shot")
    b = bench_completion()
    sb = summarize(b)
    print(sb)
    for r in [r for r in b if r["got"] != r["want"] or not r["arg_ok"]][:5]:
        print("  MISS:", r)

    results = {"jinja_tools": {**sa, "runs": a}, "completion_fewshot": {**sb, "runs": b}}
    print(write_result("p19_gemma_tools", results))
