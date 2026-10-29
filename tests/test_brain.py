"""Laptop-only tests for brain.state / selector / skills and telemetry.decisions."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from rover.brain import skills
from rover.brain.selector import RuleSelector
from rover.brain.state import Skill, WorldState
from rover.config import ControlConfig
from rover.hal.rover import DryRunRover, RoverConfig
from rover.telemetry.decisions import DecisionLog

CONTROL = ControlConfig()
SELECT = RuleSelector(CONTROL).select


def _go_to(**kwargs) -> WorldState:
    return WorldState(mission="go_to", target="cup", **kwargs)


def test_go_to_searches_approaches_then_verifies():
    assert SELECT(_go_to()) is Skill.SEARCH
    assert SELECT(_go_to(target_visible=True, bearing_deg=5.0, h_frac=0.2)) is Skill.APPROACH
    assert SELECT(_go_to(target_visible=True, h_frac=0.6, h_stop=0.5)) is Skill.VERIFY


def test_go_to_verify_outcomes_and_replan_limit():
    assert SELECT(_go_to(verify_result=True)) is Skill.DONE
    assert SELECT(_go_to(verify_result=False, replans=0)) is Skill.BACK_OFF
    assert SELECT(_go_to(verify_result=False, replans=CONTROL.max_replans)) is Skill.ABORT


def test_lost_target_reacquires_once_then_searches():
    seen = dict(last_bearing_deg=20.0, last_skill=Skill.APPROACH)
    assert SELECT(_go_to(lost_for_s=0.5, **seen)) is Skill.REACQUIRE  # brief dropout
    assert SELECT(_go_to(lost_for_s=2.0, **seen)) is Skill.REACQUIRE  # turn toward it
    after = dict(last_bearing_deg=20.0, last_skill=Skill.REACQUIRE)
    assert SELECT(_go_to(lost_for_s=2.0, **after)) is Skill.SEARCH


def test_exhausted_search_asks_the_planner_once_then_aborts():
    steps = CONTROL.search_max_steps
    assert SELECT(_go_to(search_steps=steps)) is Skill.ASK_PLANNER
    assert SELECT(_go_to(search_steps=steps, asked_planner=True)) is Skill.ABORT


def test_follow_and_no_mission():
    assert SELECT(WorldState(mission="follow", target_visible=True)) is Skill.FOLLOW
    assert SELECT(WorldState(mission="follow", lost_for_s=2.0)) is Skill.REACQUIRE
    assert SELECT(WorldState(mission="follow", lost_for_s=9.0)) is Skill.ABORT
    assert SELECT(WorldState()) is Skill.DONE


def test_state_as_text_is_flat_and_drops_empty_fields():
    text = _go_to(target_visible=True, bearing_deg=3.14159, last_skill=Skill.SEARCH).as_text()
    assert "mission=go_to" in text and "bearing_deg=3.14" in text
    assert "last_skill=SEARCH" in text and "verify_result" not in text


def test_approach_arcs_toward_the_target_and_stops_at_h_stop():
    left, right = skills.approach(10.0, 0.1, 0.5, 31.0, CONTROL)
    assert left > right > 0  # target to the right: arc right while moving
    assert skills.approach(0.0, 0.3, 0.5, 31.0, CONTROL) == (CONTROL.v_max, CONTROL.v_max)
    left, right = skills.approach(0.0, 0.5, 0.5, 31.0, CONTROL)
    assert left == right == 0.0  # arrived: no forward (no slowing ramp: the floor is 0.30)
    assert skills.approach(40.0, 0.1, 0.5, 31.0, CONTROL) is None  # re-aim with a spin


def test_follow_keeps_the_band():
    low, high = CONTROL.follow_h_frac
    assert skills.follow(0.0, low - 0.1, 31.0, CONTROL)[0] > 0  # too far: forward
    assert skills.follow(0.0, high + 0.1, 31.0, CONTROL)[0] < 0  # too close: back
    assert skills.follow(0.0, (low + high) / 2, 31.0, CONTROL) == (0.0, 0.0)


def test_spin_turns_by_the_heading_and_stops(monkeypatch):
    monkeypatch.setattr(skills.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    clock = [0.0]
    monkeypatch.setattr("rover.hal.rover.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(skills.time, "monotonic", lambda: clock[0])
    rover = DryRunRover(RoverConfig(spin_speed_cap=0.5))
    turned = skills.spin(rover, 90.0, 0.5)
    assert 80.0 <= turned <= 90.0  # stops SPIN_LEAD_DEG early; the coast is physical
    assert rover.last_command == (0.0, 0.0)
    assert skills.spin(rover, -45.0, 0.5) < 0


def test_spin_is_cancelled_by_stop(monkeypatch):
    rover = DryRunRover(RoverConfig(spin_speed_cap=0.5))
    assert skills.spin(rover, 90.0, 0.5, cancelled=lambda: True) == pytest.approx(0.0, abs=1)
    assert rover.last_command == (0.0, 0.0)


def test_decision_log_appends_jsonl(tmp_path):
    log = DecisionLog(tmp_path)
    log.append(_go_to(search_steps=2), Skill.SEARCH, "turned 35")
    row = json.loads(log.path().read_text().splitlines()[0])
    assert row["skill"] == "SEARCH" and row["outcome"] == "turned 35"
    assert "search_steps=2" in row["state"] and row["fields"]["target"] == "cup"


def test_selector_reads_gains_from_any_object():
    selector = RuleSelector(SimpleNamespace(search_max_steps=1))
    assert selector.select(_go_to(search_steps=1)) is Skill.ASK_PLANNER
