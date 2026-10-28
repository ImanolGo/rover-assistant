#!/usr/bin/env python3
"""B2: STT benchmark — Moonshine streaming (tiny/small/medium) vs Gemma audio.

Feeds 16 kHz mono PCM to Moonshine yourself (``Transcriber.add_audio``) in 80 ms
chunks at real-time pace — the HAL owns the mic, Moonshine never touches it.
Metrics per candidate, per dataset: intent accuracy (via
``rover.voice.intents.classify``), WER, end-of-speech -> final transcript
p50/p90, time to first partial containing a stop word, process CPU %, RSS.

Datasets: the synthesized 20 commands (``bench_voice``) and the real-mic set
(``bench/data/commands/labels.jsonl``) when recorded.

    .venv/bin/python bench/bench_stt_moonshine.py --dataset synth
    .venv/bin/python bench/bench_stt_moonshine.py --dataset mic --contention
    .venv/bin/python bench/bench_stt_moonshine.py --dataset mic --keyterms  # runtime bias

``--keyterms`` applies the same ``set_keyterms`` vocabulary as ``rover.main`` and
writes ``p4b_stt_moonshine_keyterms.json``. ``stop_recall`` is the share of
stop-labelled takes classified as ``stop`` (the safety number).

Each Moonshine candidate runs in its own spawned process with the runtime's
``voice.moonshine_vad_threshold`` (``--moonshine-vad`` overrides): libmoonshine's VAD
state outlives a Transcriber, so results used to depend on run order (2026-10-28).
"""

from __future__ import annotations

import argparse
import difflib
import glob
import json
import statistics
import sys
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import psutil
import soundfile as sf

sys.path.insert(0, "bench")
from bench_voice import COMMANDS as SYNTH_COMMANDS  # noqa: E402
from bench_voice import synthesize  # noqa: E402

from rover.config import load_config  # noqa: E402
from rover.voice.intents import classify, strip_wake_phrase, vocabulary  # noqa: E402
from rover.voice.stt import GemmaStt  # noqa: E402

URL = "http://127.0.0.1:8080"
FRAME = 1280
STOP_WORDS = ("stop", "halt", "freeze", "cancel", "abort")

MOONSHINE = {
    "moonshine-tiny": ("tiny-streaming-en", "TINY_STREAMING"),
    "moonshine-small": ("small-streaming-en", "SMALL_STREAMING"),
    "moonshine-medium": ("medium-streaming-en", "MEDIUM_STREAMING"),
}


def _model_dir(name: str) -> str:
    hits = sorted(glob.glob(f"models/moonshine/**/model/{name}/quantized_*", recursive=True))
    if not hits:
        raise FileNotFoundError(f"moonshine model {name} not found under models/moonshine/")
    return hits[-1]


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = reference.lower().split(), hypothesis.lower().split()
    return 1.0 - difflib.SequenceMatcher(None, ref, hyp).ratio()


def _resample16k(path: str) -> np.ndarray:
    data, rate = sf.read(path, dtype="float32")
    if getattr(data, "ndim", 1) > 1:
        data = data[:, 0]
    if rate != 16000:
        count = int(len(data) * 16000 / rate)
        data = np.interp(np.linspace(0, len(data) - 1, count), np.arange(len(data)), data)
    return data.astype(np.float32)


def synth_dataset() -> list[tuple[str, np.ndarray, str, str | None]]:
    out = []
    scratch = Path("bench/results/raw/voice")
    scratch.mkdir(parents=True, exist_ok=True)
    tmp = scratch / "stt_tmp.wav"
    for text, intent, target in SYNTH_COMMANDS:
        clip = synthesize(text, tmp).astype(np.float32) / 32768.0  # float32 [-1, 1]
        out.append((text, clip, intent, target))
    return out


def mic_dataset() -> list[tuple[str, np.ndarray, str, str | None]]:
    labels = Path("bench/data/commands/labels.jsonl")
    if not labels.exists():
        return []
    out = []
    for row in labels.read_text().splitlines():
        if not row.strip():
            continue
        rec = json.loads(row)
        wav = Path("bench/data/commands") / rec["file"]
        if wav.exists():
            out.append((rec["text"], _resample16k(str(wav)), rec["intent"], rec.get("target")))
    return out


def run_moonshine_isolated(model_name: str, dataset, pace: float, contention, keyterms, vad):
    """``run_moonshine`` in a fresh process: libmoonshine's VAD keeps state for the life of
    the process, so a candidate run after another one (or after many clips) heard less."""
    import multiprocessing

    with multiprocessing.get_context("spawn").Pool(1) as pool:
        return pool.apply(
            run_moonshine, (model_name, dataset, pace, contention is not None, keyterms, vad)
        )


def run_moonshine(
    model_name: str,
    dataset,
    pace: float,
    contention=None,
    keyterms: list[str] | None = None,
    vad_threshold: float | None = None,
) -> dict:
    from moonshine_voice import ModelArch
    from moonshine_voice.transcriber import LineCompleted, LineUpdated, Transcriber

    arch = getattr(ModelArch, MOONSHINE[model_name][1])
    model_dir = _model_dir(MOONSHINE[model_name][0])

    proc = psutil.Process()
    baseline_rss = proc.memory_info().rss
    proc.cpu_percent(None)
    options = {"vad_threshold": str(vad_threshold)} if vad_threshold is not None else None
    transcriber = Transcriber(model_dir, arch, options=options)  # load once per candidate
    if keyterms:
        transcriber.set_keyterms(keyterms)
    loaded_rss = proc.memory_info().rss

    rows = []
    for text, audio, want_intent, want_target in dataset:
        final: list[str] = []
        partials: list[str] = []

        def listener(event):
            if isinstance(event, LineCompleted):
                final.append(event.line.text)
            elif isinstance(event, LineUpdated):
                partials.append(event.line.text)

        stream = transcriber.create_stream()
        stream.add_listener(listener)
        stream.start()
        stop_partial = None
        t_feed_start = time.perf_counter()
        for i in range(0, len(audio), FRAME):
            chunk = audio[i : i + FRAME]
            stream.add_audio(chunk.tolist(), 16000)
            partial = strip_wake_phrase(partials[-1]) if partials else ""
            if stop_partial is None and any(word in partial.lower().split() for word in STOP_WORDS):
                stop_partial = time.perf_counter() - t_feed_start
            time.sleep(pace)
        t_end = time.perf_counter()
        stream.stop()
        latency = time.perf_counter() - t_end
        close = getattr(stream, "close", None)
        if close is not None:
            close()
        hypothesis = " ".join(final).strip() or (partials[-1] if partials else "")
        intent = classify(hypothesis)
        rows.append(
            {
                "said": text,
                "want": want_intent,
                "transcript": hypothesis[:60],
                "intent": intent.name,
                "target": intent.target,
                "ok": intent.name == want_intent
                and (want_target is None or intent.target == want_target),
                "wer": round(wer(text, hypothesis), 2),
                "latency_s": round(latency, 3),
                "stop_partial_s": round(stop_partial, 3) if stop_partial is not None else None,
            }
        )
        if contention:  # a flag in the child; the parent's thread does the hammering
            time.sleep(0.05)
    return _summary(model_name, rows, proc, baseline_rss, loaded_rss)


def run_gemma(dataset, contention=None) -> dict:
    stt = GemmaStt(URL)
    proc = psutil.Process()
    proc.cpu_percent(None)
    rows = []
    for text, audio, want_intent, want_target in dataset:
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        started = time.perf_counter()
        try:
            hypothesis = stt.transcribe(pcm, 16000)
        except Exception as exc:  # noqa: BLE001
            hypothesis = f"<error {type(exc).__name__}>"
        latency = time.perf_counter() - started
        intent = classify(hypothesis)
        rows.append(
            {
                "said": text,
                "want": want_intent,
                "transcript": hypothesis[:60],
                "intent": intent.name,
                "target": intent.target,
                "ok": intent.name == want_intent
                and (want_target is None or intent.target == want_target),
                "wer": round(wer(text, hypothesis), 2),
                "latency_s": round(latency, 3),
                "stop_partial_s": None,
            }
        )
        if contention is not None:
            contention.wait()
    stt.close()
    return _summary("gemma", rows, proc)


def _summary(
    name: str,
    rows: list[dict],
    proc: psutil.Process,
    baseline_rss: int | None = None,
    loaded_rss: int | None = None,
) -> dict:
    latencies = sorted(row["latency_s"] for row in rows)
    p90 = latencies[min(len(latencies) - 1, int(0.9 * len(latencies)))]
    correct = sum(row["ok"] for row in rows)
    stops = [row for row in rows if row["want"] == "stop"]
    return {
        "candidate": name,
        "commands": rows,
        "accuracy": round(correct / len(rows), 3) if rows else 0.0,
        "correct": correct,
        "total": len(rows),
        "stop_recall": (
            round(sum(row["intent"] == "stop" for row in stops) / len(stops), 3) if stops else None
        ),
        "stop_total": len(stops),
        "wer_mean": round(statistics.mean(row["wer"] for row in rows), 3) if rows else None,
        "latency_p50_s": round(statistics.median(latencies), 3) if rows else None,
        "latency_p90_s": round(p90, 3) if rows else None,
        "cpu_percent": round(proc.cpu_percent(None), 1),
        "rss_mb": round(((loaded_rss or proc.memory_info().rss) - (baseline_rss or 0)) / 1e6, 1),
    }


class Contention:
    """Hammer llama-server with small text requests while STT runs."""

    def __init__(self, stop_after: int = 4):
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        client = httpx.Client(timeout=30)
        while not self._stop.is_set():
            try:
                client.post(
                    f"{URL}/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": "Say one short word."}],
                        "max_tokens": 8,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
            except Exception:  # noqa: BLE001
                time.sleep(0.2)

    def wait(self) -> None:
        time.sleep(0.05)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=30)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["synth", "mic", "both"], default="synth")
    parser.add_argument("--candidates", nargs="*", default=["gemma", *MOONSHINE])
    parser.add_argument("--pace", type=float, default=0.08, help="seconds between 80 ms chunks")
    parser.add_argument("--contention", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="only the first N commands (0 = all)")
    parser.add_argument("--keyterms", action="store_true", help="bias Moonshine like rover.main")
    parser.add_argument("--name", default=None, help="result name (default from --keyterms)")
    parser.add_argument(
        "--moonshine-vad",
        type=float,
        default=None,
        help="libmoonshine vad_threshold (default: config voice.moonshine_vad_threshold)",
    )
    args = parser.parse_args()
    vad = args.moonshine_vad
    if vad is None:
        vad = load_config().voice.moonshine_vad_threshold
    keyterms = None
    if args.keyterms:
        keyterms = vocabulary(load_config().perception.target_classes_h_stop.keys())

    datasets: dict[str, list] = {}
    if args.dataset in ("synth", "both"):
        datasets["synth"] = synth_dataset()
    if args.dataset in ("mic", "both"):
        mic = mic_dataset()
        if not mic:
            print("WARN: no real-mic recordings yet (run bench/record_commands.py); skipping")
        else:
            datasets["mic"] = mic
    if args.limit:
        datasets = {name: rows[: args.limit] for name, rows in datasets.items()}

    contention = Contention() if args.contention else None
    if args.contention:
        try:
            httpx.get(f"{URL}/health", timeout=10)
        except Exception:  # noqa: BLE001
            print("WARN: llama-server not reachable; contention will be idle")

    results = {
        "contention": bool(args.contention),
        "keyterms": keyterms,
        "moonshine_vad_threshold": vad,
        "moonshine_isolated": True,
        "datasets": {},
    }
    try:
        for name, rows in datasets.items():
            print(f"=== dataset: {name} ({len(rows)} commands) ===", flush=True)
            results["datasets"][name] = {}
            for candidate in args.candidates:
                if candidate.startswith("moonshine"):
                    summary = run_moonshine_isolated(
                        candidate, rows, args.pace, contention, keyterms, vad
                    )
                else:
                    summary = run_gemma(rows, contention)
                results["datasets"][name][candidate] = summary
                print(
                    f"  {candidate:16s} acc={summary['accuracy']:.2f} "
                    f"wer={summary['wer_mean']} p50={summary['latency_p50_s']} "
                    f"p90={summary['latency_p90_s']} stop={summary['stop_recall']} "
                    f"cpu={summary['cpu_percent']}% "
                    f"rss={summary['rss_mb']}MB",
                    flush=True,
                )
    finally:
        if contention is not None:
            contention.close()

    from common import write_result

    name = args.name or ("p4b_stt_moonshine_keyterms" if keyterms else "p4b_stt_moonshine")
    print(write_result(name, results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
