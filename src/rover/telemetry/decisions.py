"""telemetry.decisions: one JSONL line per selector tick (PLAN 5.6).

``logs/decisions-YYYYMMDD.jsonl`` with the world state (as text and as fields),
the chosen skill and its outcome. This is Laya's future training data, so the
format stays flat and append-only.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any


class DecisionLog:
    def __init__(self, directory: str | Path = "logs"):
        self.directory = Path(directory)

    def path(self) -> Path:
        return self.directory / f"decisions-{time.strftime('%Y%m%d')}.jsonl"

    def append(self, state: Any, skill: Any, outcome: str = "") -> None:
        row = {
            "t": round(time.time(), 3),
            "state": state.as_text(),
            "fields": asdict(state),
            "skill": getattr(skill, "value", skill),
            "outcome": outcome,
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.path().open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=str) + "\n")
