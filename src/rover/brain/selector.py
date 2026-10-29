"""brain.selector: rule-based skill selection (PLAN 5.3, ≤ 60 lines).

``select(state) -> Skill`` is the whole interface, so a learned selector (Laya,
Phase 8) can replace :class:`RuleSelector` without touching the mission loop.
"""

from __future__ import annotations

from typing import Protocol

from rover.brain.state import Skill, WorldState


class Selector(Protocol):
    def select(self, state: WorldState) -> Skill: ...


class RuleSelector:
    def __init__(self, control: object):
        self.lost_reacquire_s = float(getattr(control, "lost_reacquire_s", 1.0))
        self.follow_give_up_s = float(getattr(control, "follow_lost_give_up_s", 5.0))
        self.search_max_steps = int(getattr(control, "search_max_steps", 12))
        self.max_replans = int(getattr(control, "max_replans", 2))
        self.max_mission_s = float(getattr(control, "max_mission_s", 120.0))

    def select(self, s: WorldState) -> Skill:
        if s.mission == "follow":
            if s.target_visible:
                return Skill.FOLLOW
            return Skill.ABORT if s.lost_for_s > self.follow_give_up_s else Skill.REACQUIRE
        if s.mission != "go_to":
            return Skill.DONE
        if s.elapsed_s > self.max_mission_s:
            return Skill.ABORT
        if s.verify_result is True:
            return Skill.DONE
        if s.verify_result is False:
            return Skill.ABORT if s.replans >= self.max_replans else Skill.BACK_OFF
        if s.target_visible:
            return Skill.VERIFY if (s.h_frac or 0.0) >= s.h_stop else Skill.APPROACH
        recently_seen = s.last_bearing_deg is not None and s.last_skill in (
            Skill.APPROACH,
            Skill.REACQUIRE,
        )
        if recently_seen and s.lost_for_s <= self.lost_reacquire_s:
            return Skill.REACQUIRE  # brief dropout: hold still, re-associate
        if recently_seen and s.last_skill is Skill.APPROACH:
            return Skill.REACQUIRE  # lost for real: one turn toward where it went
        if s.search_steps >= self.search_max_steps:
            return Skill.ABORT if s.asked_planner else Skill.ASK_PLANNER
        return Skill.SEARCH
