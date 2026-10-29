"""brain.state: the selector's view of the world, and the skills it can pick.

``WorldState`` is a flat snapshot built once per mission tick from the latest
detections and the mission bookkeeping (ARCHITECTURE.md §4, selector interface).
``as_text`` is the line logged per decision: Laya's future training input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


class Skill(str, Enum):
    SEARCH = "SEARCH"
    APPROACH = "APPROACH"
    REACQUIRE = "REACQUIRE"
    FOLLOW = "FOLLOW"
    VERIFY = "VERIFY"
    BACK_OFF = "BACK_OFF"
    ASK_PLANNER = "ASK_PLANNER"
    DONE = "DONE"
    ABORT = "ABORT"


@dataclass
class WorldState:
    mission: str = ""  # "go_to" | "follow" | ""
    target: str | None = None
    attributes: tuple[str, ...] = ()
    target_visible: bool = False
    bearing_deg: float | None = None  # + = right of the image centre
    h_frac: float | None = None
    h_stop: float = 0.5  # arrival box-height fraction for the target class
    lost_for_s: float = 0.0  # since the target was last seen (0 while visible)
    last_bearing_deg: float | None = None  # where it was last seen
    search_steps: int = 0
    asked_planner: bool = False
    elapsed_s: float = 0.0
    last_skill: Skill | None = None
    verify_result: bool | None = None
    replans: int = 0
    extra: dict = field(default_factory=dict)

    def as_text(self) -> str:
        """One stable ``key=value`` line (None/empty fields dropped)."""
        items = asdict(self)
        items.pop("extra")
        parts = []
        for key, value in items.items():
            if value in (None, (), "", {}):
                continue
            if isinstance(value, float):
                value = round(value, 2)
            if isinstance(value, Skill):
                value = value.value
            parts.append(f"{key}={value}")
        return " ".join(parts)
