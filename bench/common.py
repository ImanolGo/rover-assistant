"""Common benchmark utilities: MemProbe (memory/tegrastats sampling), timer, write_result."""

from __future__ import annotations

import json
import os
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path("bench/results")
RAW_DIR = RESULTS_DIR / "raw"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def power_mode() -> str:
    try:
        out = subprocess.check_output(["nvpmodel", "-q"], text=True, stderr=subprocess.DEVNULL)
        return " ".join(out.split())
    except Exception:
        return "n/a"


def clocks_summary() -> str:
    try:
        out = subprocess.check_output(
            ["jetson_clocks", "--show"], text=True, stderr=subprocess.DEVNULL
        )
        lines = out.strip().splitlines()
        return "; ".join(lines[:3])
    except Exception:
        return "n/a"


def mem_available_mb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    return -1.0


class MemProbe:
    """Samples MemAvailable, per-PID RSS and (on Jetson) tegrastats into a CSV."""

    def __init__(self, name: str, interval_s: float = 0.5):
        self.name = name
        self.interval_s = interval_s
        self.csv_path = RAW_DIR / f"{name}_mem.csv"
        self.tsys_path = RAW_DIR / f"{name}_tegrastats.log"
        self._proc: subprocess.Popen | None = None
        self._stop = False
        self._thread = None
        self._min_available_mb = float("inf")
        self._swap_start_kb: int = 0

    def start(self) -> None:
        import threading

        RAW_DIR.mkdir(parents=True, exist_ok=True)
        self._swap_start_kb = self._swap_kb()
        self._stop = False
        tegrastats = "/usr/bin/tegrastats"
        if os.path.exists(tegrastats):
            self._proc = subprocess.Popen(
                [
                    tegrastats,
                    "--interval",
                    str(int(self.interval_s * 1000)),
                    "--logfile",
                    str(self.tsys_path.absolute()),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @staticmethod
    def _swap_kb() -> int:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("SwapTotal:"):
                    total = int(line.split()[1])
                if line.startswith("SwapFree:"):
                    free = int(line.split()[1])
        return total - free

    def _loop(self) -> None:
        import csv

        import psutil

        with open(self.csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "mem_available_mb", "rss_mb_by_pid"])
            while not self._stop:
                procs = []
                for p in psutil.process_iter(["pid", "name", "memory_info"]):
                    try:
                        rss = p.info["memory_info"].rss / (1024 * 1024)
                        if rss > 10:
                            procs.append(f"{p.info['pid']}:{p.info['name']}={rss:.0f}")
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                avail = mem_available_mb()
                self._min_available_mb = min(self._min_available_mb, avail)
                w.writerow([time.time(), f"{avail:.1f}", " ".join(procs)])
                f.flush()
                time.sleep(self.interval_s)

    def stop(self) -> dict:
        self._stop = True
        if self._proc:
            self._proc.terminate()
        swap_growth_mb = (self._swap_kb() - self._swap_start_kb) / 1024
        return {
            "min_mem_available_mb": round(
                self._min_available_mb if self._min_available_mb != float("inf") else -1,
                1,
            ),
            "swap_growth_mb": round(swap_growth_mb, 1),
            "csv": str(self.csv_path),
            "tegrastats_log": str(self.tsys_path) if self._proc else None,
        }


@contextmanager
def timer():
    """Records wall time of the block into a dict under 'elapsed_s'."""
    out: dict = {}
    t0 = time.perf_counter()
    try:
        yield out
    finally:
        out["elapsed_s"] = time.perf_counter() - t0


def write_result(name: str, result: dict) -> Path:
    """Writes bench/results/<name>.json with commit hash, date, power mode."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "date": datetime.now().isoformat(timespec="seconds"),
        "commit": git_commit(),
        "power_mode": power_mode(),
        "clocks": clocks_summary(),
        **result,
    }
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path
