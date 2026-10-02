"""Laptop-only tests for brain.intents: regex table, stop-first, attributes."""

from __future__ import annotations

import pytest

from rover.voice.intents import Intent, classify, is_stop, strip_wake_phrase


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("stop", "stop"),
        ("STOP NOW", "stop"),
        ("please halt", "stop"),
        ("emergency stop", "stop"),
        ("go forward", "forward"),
        ("move forward please", "forward"),
        ("reverse", "backward"),
        ("back up", "backward"),
        ("turn left", "turn_left"),
        ("turn right", "turn_right"),
        ("turn around", "turn_around"),
        ("come here", "forward"),
        ("go home", "return_home"),
        ("what do you see", "describe"),
        ("describe the scene", "describe"),
        ("look around", "describe"),
        ("follow me", "follow"),
        ("come along with me", "follow"),
    ],
)
def test_simple_and_task_intents(text, name):
    assert classify(text).name == name


def test_go_to_extracts_target_and_colour_attribute():
    intent = classify("go to the red cup")
    assert intent.name == "go_to"
    assert intent.target == "cup"
    assert intent.attributes == ("red",)

    found = classify("find my green bottle")
    assert found.name == "go_to"
    assert found.target == "bottle"
    assert found.attributes == ("green",)


def test_go_to_plain_target_has_no_attributes():
    intent = classify("go to the chair")
    assert intent.name == "go_to"
    assert intent.target == "chair"
    assert intent.attributes == ()


def test_is_there_intent():
    intent = classify("is there a blue bus")
    assert intent.name == "is_there"
    assert intent.target == "bus"
    assert intent.attributes == ("blue",)


def test_stop_wins_over_other_words():
    intent = classify("stop and go to the red cup")
    assert intent.name == "stop"


def test_unknown_and_empty():
    assert classify("tell me a story about the moon").name == "unknown"
    assert classify("   ").name == "unknown"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hey Rover, go to the red cup", "go to the red cup"),
        ("Hey Rubber.", ""),
        ("Rofer stop", "stop"),
        ("Go forward", "Go forward"),
    ],
)
def test_strip_wake_phrase(text, expected):
    assert strip_wake_phrase(text) == expected


def test_classify_handles_command_with_wake_phrase_in_one_breath():
    assert classify("Hey Rover, go to the red cup").name == "go_to"
    assert classify("Hey Rover, stop").name == "stop"
    assert classify("Hey Rover").name == "unknown"


def test_go_back_to_is_a_destination_not_a_reverse():
    go_to = classify("go back to the kitchen")
    assert go_to.name == "go_to"
    assert go_to.target == "kitchen"
    assert classify("go back").name == "backward"


def test_follow_can_carry_a_target():
    targeted = classify("follow the red ball")
    assert targeted.name == "follow"
    assert targeted.target == "red ball"
    assert targeted.attributes == ("red",)
    bare = classify("follow me")
    assert bare.name == "follow"
    assert bare.target is None


def test_dont_stop_is_a_safe_stop():
    # Negation is not parsed; stopping is the safe failure.
    assert classify("don't stop").name == "stop"


def test_is_stop_shares_one_pattern():
    assert is_stop("cancel now")
    assert is_stop("Hey Rover, abort")
    assert not is_stop("go forward")


def test_intent_is_a_dataclass_with_raw_text():
    intent = classify("turn left")
    assert isinstance(intent, Intent)
    assert intent.raw == "turn left"
    assert intent.response == "Turning left."
