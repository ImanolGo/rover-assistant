"""brain.intents: regex intents, checked before any model runs.

Ports the legacy ``SIMPLE_COMMANDS`` table (stop first, always) and adds the
task intents the planner would otherwise have to handle: ``go_to``,
``follow``, ``describe`` and ``is_there``. Anything unmatched is ``unknown``
and goes to Gemma. The table is ordered; the first match wins.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

COLORS = (
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "brown",
    "black",
    "white",
    "gray",
    "grey",
)

# The single source of truth for the safety stop, shared by the API and voice.
STOP_RE = re.compile(r"\b(stop|halt|freeze|emergency|abort|cancel)\b", re.I)


def is_stop(text: str) -> bool:
    """True if ``text`` (minus any wake phrase) contains a stop word."""
    return bool(STOP_RE.search(strip_wake_phrase(text)))


# (name, response) after the trigger phrase. Ported from legacy command_router.
_SIMPLE: list[tuple[re.Pattern[str], str, str | None]] = [
    (STOP_RE, "stop", "Stopping."),
    (
        re.compile(r"\b(go\s+forward|move\s+forward|advance)\b", re.I),
        "forward",
        "Moving forward.",
    ),
    # "go back to X" is a destination (go_to), not a reverse; block the "to" case.
    (
        re.compile(r"\b(go\s+back|move\s+back|reverse|back\s+up)\b(?!\s+to\b)", re.I),
        "backward",
        "Moving backward.",
    ),
    (re.compile(r"\b(turn\s+left|go\s+left)\b", re.I), "turn_left", "Turning left."),
    (re.compile(r"\b(turn\s+right|go\s+right)\b", re.I), "turn_right", "Turning right."),
    (re.compile(r"\b(turn\s+around|do\s+a?\s*180)\b", re.I), "turn_around", "Turning around."),
    (re.compile(r"\b(come\s+here|come\s+to\s+me)\b", re.I), "forward", "Coming to you."),
    (
        re.compile(r"\b(go\s+home|return\s+home|return\s+to\s+base)\b", re.I),
        "return_home",
        "Returning home.",
    ),
]

_FOLLOW = re.compile(r"\bfollow\b(?:\s+(?P<rest>.+))?|\bcome\s+along\s+with\s+me\b", re.I)
_DESCRIBE = re.compile(
    r"\b(what\s+do\s+you\s+see|what\s+can\s+you\s+see|describe|look\s+around"
    r"|what\s+is\s+in\s+front\s+of\s+you)\b",
    re.I,
)
_IS_THERE = re.compile(r"\bis\s+there\s+(?P<rest>.+)", re.I)
_GO_TO = re.compile(
    r"\b(?:go\s+back\s+to|go\s+to|go\s+find|find|look\s+for|fetch|bring\s+me|get\s+me)\s+"
    r"(?P<rest>.+)",
    re.I,
)

_ARTICLE = re.compile(r"^(?:the|a|an|my|your|his|her|their)\s+", re.I)
_WORD = re.compile(r"[a-z0-9]+")

# A leading wake phrase as Gemma may transcribe it: "Hey Rover", "Hey Rubber",
# "Rover", ... Stripping it lets "Hey Rover, go to the cup" classify as go_to,
# and lets "Hey Rover." alone mean "no command" rather than "unknown".
_WAKE_PHRASE = re.compile(
    r"^\s*(?:(?:hey|hay|hi|a)\b[\s,]*)?r[ou][a-z]{1,5}\b[\s,.!?]*",
    re.I,
)


def strip_wake_phrase(text: str) -> str:
    """Remove a leading wake phrase; empty result means the user only woke it."""
    return _WAKE_PHRASE.sub("", (text or "").strip(), count=1).strip()


@dataclass
class Intent:
    """A classified command. ``name`` is a fixed verb; ``target`` may be None."""

    name: str
    target: str | None = None
    attributes: tuple[str, ...] = ()
    response: str | None = None
    raw: str = ""
    slots: dict[str, str] = field(default_factory=dict)


def _clean_phrase(rest: str) -> str:
    """Drop a leading article and punctuation: 'the red ball' -> 'red ball'."""
    return _ARTICLE.sub("", rest.strip()).strip().rstrip(".!?").lower()


def _split_target(rest: str) -> tuple[str, tuple[str, ...]]:
    """Turn 'the red cup' into ('cup', ('red',)): drop articles, pull out colours."""
    cleaned = _clean_phrase(rest)
    words = _WORD.findall(cleaned)
    if not words:
        return "", ()
    attributes = tuple(word for word in words if word in COLORS)
    # The head noun is the last non-colour word, else the last word.
    nouns = [word for word in words if word not in COLORS]
    target = nouns[-1] if nouns else words[-1]
    return target, attributes


_VOCAB_VERBS = (
    "go",
    "to",
    "find",
    "look",
    "for",
    "follow",
    "me",
    "stop",
    "halt",
    "freeze",
    "cancel",
    "abort",
    "turn",
    "left",
    "right",
    "around",
    "forward",
    "backward",
    "reverse",
    "advance",
    "describe",
    "see",
    "what",
    "is",
    "there",
    "home",
    "return",
    "base",
    "come",
    "here",
    "bring",
    "fetch",
)


def vocabulary(targets: Iterable[str] = ()) -> list[str]:
    """The word list for Moonshine ``set_keyterms``: verbs + colours + object names."""
    words = set(_VOCAB_VERBS) | set(COLORS) | {str(t).lower() for t in targets}
    return sorted(words)


def classify(text: str) -> Intent:
    """Classify a transcript. ``stop`` is matched first and bypasses everything."""
    raw = strip_wake_phrase(text)
    if not raw:
        return Intent(name="unknown", raw=(text or "").strip())
    lowered = raw.lower()

    for pattern, name, response in _SIMPLE:
        if pattern.search(lowered):
            return Intent(name=name, response=response, raw=raw)

    if match := _IS_THERE.search(lowered):
        target, attributes = _split_target(match.group("rest"))
        return Intent(name="is_there", target=target, attributes=attributes, raw=raw)

    if _DESCRIBE.search(lowered):
        return Intent(name="describe", raw=raw)

    if match := _FOLLOW.search(lowered):
        rest = (match.group("rest") or "").strip()
        if rest and rest not in ("me", "us"):
            return Intent(
                name="follow",
                target=_clean_phrase(rest),
                attributes=_split_target(rest)[1],
                raw=raw,
            )
        return Intent(name="follow", raw=raw)

    if match := _GO_TO.search(lowered):
        target, attributes = _split_target(match.group("rest"))
        return Intent(name="go_to", target=target or None, attributes=attributes, raw=raw)

    return Intent(name="unknown", raw=raw)
