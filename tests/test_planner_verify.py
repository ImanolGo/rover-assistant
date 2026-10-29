"""Laptop-only tests for brain.planner (mock HTTP) and brain.verify."""

from __future__ import annotations

import json

import httpx
import numpy as np

from rover.brain.planner import GemmaClient, ToolCall, parse_tool_call
from rover.brain.verify import (
    detector_agrees,
    parse_verification_answer,
    verify_arrival,
)
from rover.config import PlannerConfig


def _client(handler) -> GemmaClient:
    return GemmaClient(PlannerConfig(), client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_parse_tool_call_tolerates_noise_and_rejects_unknown_tools():
    assert parse_tool_call('```json\n{"tool": "go_to", "target": "cup"}\n```') == ToolCall(
        "go_to", {"target": "cup"}
    )
    assert parse_tool_call('sure! {"tool": "follow_person"} ok') == ToolCall("follow_person")
    assert parse_tool_call('{"tool": "fly", "to": "moon"}') is None
    assert parse_tool_call("no json") is None
    assert parse_tool_call('{"tool": broken') is None


def test_plan_uses_the_fewshot_completion_endpoint():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"content": '{"tool": "go_to", "target": "charger"}'})

    call = _client(handler).plan("hey can you grab my charger")
    assert call == ToolCall("go_to", {"target": "charger"})
    assert seen["path"] == "/completion"
    assert seen["body"]["prompt"].endswith("User: hey can you grab my charger\n")
    assert seen["body"]["stop"] == ["\nUser:", "\n\n"]


def test_ask_image_sends_the_frame_and_disables_thinking():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": " Yes. "}}]})

    answer = _client(handler).ask_image(np.zeros((8, 8, 3), np.uint8), "Is it a cup?")
    content = seen["body"]["messages"][0]["content"]
    assert answer == "Yes."
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[1] == {"type": "text", "text": "Is it a cup?"}
    assert seen["body"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_parse_verification_answer():
    assert parse_verification_answer("Yes") is True
    assert parse_verification_answer('"No, it is a bottle."') is False
    assert parse_verification_answer("I can see the red cup") is True
    assert parse_verification_answer("The target is not visible") is False
    assert parse_verification_answer("hmm") is None
    assert parse_verification_answer(None) is None


def test_arrival_needs_detector_colour_and_gemma():
    asked: list[str] = []

    def gemma(answer):
        return lambda prompt: asked.append(prompt) or answer

    assert detector_agrees([0.6, None, 0.55], 0.5) and not detector_agrees([0.6, 0.4, None], 0.5)
    assert verify_arrival([0.4, 0.3, 0.6], True, gemma("Yes"), "red cup") == (False, "detector")
    assert verify_arrival([0.6, 0.6, 0.6], False, gemma("Yes"), "red cup") == (False, "colour")
    assert asked == []  # Gemma is only asked once the cheap checks pass
    assert verify_arrival([0.6, 0.6, 0.6], True, gemma("No"), "red cup") == (False, "gemma")
    assert verify_arrival([0.6, 0.6, 0.6], True, gemma("Yes."), "red cup") == (True, "ok")
    assert "'red cup'" in asked[-1]
