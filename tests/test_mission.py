"""Laptop-only tests for brain.mission: whole missions against a scripted world."""

from __future__ import annotations

import time

import numpy as np

from rover.api.server import VideoSnapshot
from rover.brain.mission import Mission
from rover.brain.selector import RuleSelector
from rover.config import ControlConfig
from rover.hal.rover import DryRunRover, RoverConfig
from rover.perception.detector import Detection
from rover.telemetry.decisions import DecisionLog

CONTROL = ControlConfig(tick_hz=100, search_pause_s=0.0, search_max_steps=2)


def _det(cls="cup", bearing=5.0, h_frac=0.2, conf=0.8, box=(10, 10, 30, 40)) -> Detection:
    return Detection(cls, conf, box, 1, bearing, h_frac)


class Body:
    """The app as the mission sees it, over a dry-run rover and a scripted camera."""

    def __init__(self, world):
        self.rover = DryRunRover(RoverConfig(spin_speed_cap=0.5))
        self.world, self.said, self.states, self.drives = world, [], [], []

    def snapshot(self):
        return VideoSnapshot(frame=np.zeros((60, 80, 3), np.uint8), detections=self.world(self))

    def drive(self, left, right):
        self.drives.append((left, right))
        self.rover.drive(left, right)

    def halt(self):
        self.drive(0.0, 0.0)

    def heading_deg(self):
        return self.rover.heading_deg()

    def say(self, text):
        self.said.append(text)

    def set_state(self, state):
        self.states.append(state)


class Gemma:
    def __init__(self, answer="Yes."):
        self.answer, self.asked = answer, []

    def ask_image(self, frame, prompt, max_tokens=80):
        self.asked.append(prompt)
        return self.answer


def _mission(body, gemma, tmp_path) -> Mission:
    log = DecisionLog(tmp_path)
    return Mission(body, gemma, RuleSelector(CONTROL), log, CONTROL, 0.5, 31.0, {"cup": 0.5})


def _run(mission, fn, *args, timeout=10.0):
    mission.start(fn, *args)
    mission._thread.join(timeout)
    assert not mission._thread.is_alive()


def test_go_to_drives_toward_the_cup_verifies_and_arrives(tmp_path):
    def world(body):  # the cup grows as the robot drives forward
        forward = sum(1 for left, right in body.drives if left > 0 or right > 0)
        return [_det(h_frac=min(0.6, 0.2 + 0.05 * forward))]

    body, gemma = Body(world), Gemma("Yes.")
    mission = _mission(body, gemma, tmp_path)
    _run(mission, mission.go_to, "cup")
    assert body.said[0] == "Looking for the cup." and body.said[-1] == "I'm at the cup."
    assert any(left > right > 0 for left, right in body.drives)  # arcs right toward bearing +5
    assert (
        "APPROACH" in body.states
        and "VERIFY" in body.states
        and body.states[-2:]
        == [
            "DONE",
            "IDLE",
        ]
    )
    assert len(gemma.asked) == 1 and "'cup'" in gemma.asked[0]
    assert body.rover.last_command == (0.0, 0.0)
    assert "skill" in mission.log.path().read_text()


def test_go_to_that_never_sees_it_searches_asks_once_and_gives_up(tmp_path):
    body, gemma = Body(lambda body: []), Gemma("No.")
    mission = _mission(body, gemma, tmp_path)
    _run(mission, mission.go_to, "cup", ("red",))
    assert body.states.count("SEARCH") == CONTROL.search_max_steps
    assert body.states[-3:] == ["ASK_PLANNER", "ABORT", "IDLE"]
    assert body.said[-1] == "I couldn't find the red cup."
    assert body.heading_deg() > 50  # it really turned (two committed spins)


def test_failed_verification_backs_off_and_replans(tmp_path):
    body, gemma = Body(lambda body: [_det(bearing=0.0, h_frac=0.6)]), Gemma("No, that's a bottle.")
    mission = _mission(body, gemma, tmp_path)
    _run(mission, mission.go_to, "cup")
    assert body.states.count("BACK_OFF") == CONTROL.max_replans
    assert body.states[-2:] == ["ABORT", "IDLE"]


def test_colour_attribute_filters_detections(tmp_path):
    body = Body(lambda body: [_det(box=(0, 0, 20, 20))])
    mission = _mission(body, Gemma(), tmp_path)
    frame = np.zeros((60, 80, 3), np.uint8)
    frame[0:20, 0:20] = (0, 0, 200)  # red patch (BGR) under the box
    red = VideoSnapshot(frame=frame, detections=[_det(box=(0, 0, 20, 20))])
    assert mission._match(red, "cup", ("red",)) is not None
    assert mission._match(red, "cup", ("blue",)) is None
    assert mission._match(red, "bottle", ()) is None


def test_cancel_stops_the_mission_and_no_drive_follows(tmp_path):
    body = Body(lambda body: [])
    mission = _mission(body, Gemma(), tmp_path)
    mission.start(mission.go_to, "cup")
    time.sleep(0.1)
    mission.cancel()
    assert not mission._thread.is_alive()
    count = len(body.drives)
    time.sleep(0.1)
    assert len(body.drives) == count and body.rover.last_command == (0.0, 0.0)


def test_describe_asks_gemma_with_detector_hints(tmp_path):
    body, gemma = Body(lambda body: [_det(cls="person"), _det(cls="cup")]), Gemma("A cup.")
    mission = _mission(body, gemma, tmp_path)
    _run(mission, mission.describe, "What do you see?")
    assert body.said == ["Let me look.", "A cup."]
    assert "cup, person" in gemma.asked[0]


def test_gemma_down_is_a_soft_failure(tmp_path):
    class DeadGemma:
        def ask_image(self, frame, prompt, max_tokens=80):
            raise ConnectionError("llama-server down")

    body = Body(lambda body: [])
    mission = _mission(body, DeadGemma(), tmp_path)
    _run(mission, mission.go_to, "cup")
    assert body.states[-2:] == ["ABORT", "IDLE"] and body.said[-1] == "I couldn't find the cup."
    _run(mission, mission.describe, "What do you see?")
    assert body.said[-1] == "Sorry, I can't tell right now."
