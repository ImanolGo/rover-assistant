"""brain.planner: the Gemma client — tool calls for free-form commands, and vision.

Regex intents (``voice.intents``) skip this entirely; the planner runs only when
regex finds nothing. Tool calling uses the mode that won bench 1.9: raw
``/completion`` with a few-shot prompt (90% vs 83% for jinja tools), parsed with
the legacy-style tolerant JSON extraction. Vision questions go through
``/v1/chat/completions`` with the image inline and thinking disabled.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from typing import Any

import cv2
import httpx
import numpy as np

TOOLS = ("go_to", "follow_person", "describe", "stop", "turn", "say")

FEWSHOT = """You convert a spoken robot command into exactly one JSON tool call.
Output ONLY the JSON object, nothing else.
Tools: go_to(target[, attributes]), follow_person(), describe(question),
stop(), turn(direction[, degrees]), say(text).
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


@dataclass
class ToolCall:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)


def parse_tool_call(text: str) -> ToolCall | None:
    """First JSON object in ``text`` naming a known tool (code fences tolerated)."""
    cleaned = re.sub(r"```(?:json)?", "", text or "")
    match = re.search(r"\{[^{}]*\}", cleaned)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    tool = str(data.pop("tool", "")) if isinstance(data, dict) else ""
    return ToolCall(tool, data) if tool in TOOLS else None


def encode_image(frame: np.ndarray, quality: int = 85) -> str:
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.b64encode(buffer.tobytes()).decode("ascii") if ok else ""


class GemmaClient:
    def __init__(self, config: Any, client: httpx.Client | None = None):
        self.config = config
        self.url = str(config.url).rstrip("/")
        self._client = client or httpx.Client(timeout=float(config.timeout_s))

    def plan(self, command: str) -> ToolCall | None:
        """Free-form command -> one tool call (None if Gemma's answer is unusable)."""
        response = self._client.post(
            f"{self.url}/completion",
            json={
                "prompt": FEWSHOT.format(cmd=command.strip()),
                "n_predict": int(self.config.max_tokens),
                "temperature": float(self.config.temperature),
                "stop": ["\nUser:", "\n\n"],
                "cache_prompt": False,
            },
        )
        response.raise_for_status()
        return parse_tool_call(response.json().get("content", ""))

    def ask_image(self, frame: np.ndarray, question: str, max_tokens: int = 80) -> str:
        """One vision question about ``frame``; returns the answer text."""
        image = {"url": f"data:image/jpeg;base64,{encode_image(frame)}"}
        content = [{"type": "image_url", "image_url": image}, {"type": "text", "text": question}]
        response = self._client.post(
            f"{self.url}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": content}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        response.raise_for_status()
        return (response.json()["choices"][0]["message"].get("content") or "").strip()

    def close(self) -> None:
        self._client.close()
