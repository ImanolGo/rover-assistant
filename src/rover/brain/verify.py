"""brain.verify: has the robot arrived? (PLAN 5.4, ARCHITECTURE §4 VERIFY)

Arrival needs **all three**: the detector sees the target with conf ≥
``verify_min_conf`` on 2 of 3 frames, the colour check passes (when the command
named a colour), and Gemma answers yes. The prompt and the yes/no parsing are
ported from the legacy ``visual_verification_node``.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Sequence

_AFFIRMATIVE = re.compile(
    r"\b(yes|yeah|yep|yup|affirmative|correct|indeed|achieved|succeeded|"
    r"i can see|i see|visible|there is a|it is there|found it)\b",
    re.IGNORECASE,
)
_NEGATIVE = re.compile(
    r"\b(no|nope|negative|not|cannot|can't|don't|do not|absent|"
    r"not visible|not there|not found|no longer)\b",
    re.IGNORECASE,
)


def build_verification_prompt(goal: str) -> str:
    return (
        f"Is the goal '{goal}' achieved in this image? "
        "Is the target clearly visible? Answer with a single word: Yes or No."
    )


def parse_verification_answer(text: str | None) -> bool | None:
    """True / False for a clear yes / no, None when unsure. A leading yes/no wins.

    Negatives are checked first (legacy checked affirmatives first, so "not
    visible" read as yes): a false arrival is worse than a needless replan.
    """
    if not text:
        return None
    leading = re.match(r"^[\s\"']*(yes|no)\b", text.strip(), re.IGNORECASE)
    if leading:
        return leading.group(1).lower() == "yes"
    if _NEGATIVE.search(text):
        return False
    if _AFFIRMATIVE.search(text):
        return True
    return None


def detector_agrees(confidences: Sequence[float | None], min_conf: float) -> bool:
    """Target seen with ``min_conf`` on at least 2 of the (3) frames."""
    return sum(c is not None and c >= min_conf for c in confidences) >= 2


def verify_arrival(
    confidences: Sequence[float | None],
    colour_ok: bool,
    ask_gemma: Callable[[str], str],
    goal: str,
    min_conf: float = 0.5,
) -> tuple[bool, str]:
    """``(arrived, reason)``; Gemma is asked only if the cheap checks pass."""
    if not detector_agrees(confidences, min_conf):
        return False, "detector"
    if not colour_ok:
        return False, "colour"
    answer: Any = parse_verification_answer(ask_gemma(build_verification_prompt(goal)))
    return (answer is True), ("ok" if answer else "gemma")
