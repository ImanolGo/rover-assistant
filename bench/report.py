"""Renders all bench/results/*.json into the STATUS.md measurement tables."""

from __future__ import annotations

import json
from pathlib import Path


def render() -> str:
    lines = ["| Result | Key numbers |", "|---|---|"]
    for path in sorted(Path("bench/results").glob("*.json")):
        data = json.loads(path.read_text())
        skip = {"name", "date", "commit", "power_mode", "clocks"}
        kv = ", ".join(f"{k}={v}" for k, v in data.items() if k not in skip)
        lines.append(f"| {data['name']} ({data['date']}, {data['commit']}) | {kv} |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(render())
