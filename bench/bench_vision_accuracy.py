#!/usr/bin/env python3
"""bench 1.7b: small labeled vision set with unambiguous ground truth.

Builds a deterministic set of small (224x224) synthetic shape images plus the
existing real assets, each with a question and accepted answers, then scores a
running llama-server. Run the same script against Gemma and against Moondream to
compare accuracy fairly:

    .venv/bin/python bench/bench_vision_accuracy.py --url http://127.0.0.1:8080 \
        --protocol gemma --name p17b_vision_accuracy_gemma
    .venv/bin/python bench/bench_vision_accuracy.py --url http://127.0.0.1:8090 \
        --protocol moondream --name p17b_vision_accuracy_moondream

Ground truth is defined here in code; nothing is downloaded.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import statistics
import sys
import time

import httpx
import numpy as np
from PIL import Image, ImageDraw

SIZE = 224
WHITE = (255, 255, 255)
COLOURS = {
    "red": (220, 40, 40),
    "green": (40, 180, 70),
    "blue": (40, 80, 220),
    "yellow": (240, 210, 40),
}


def _blank() -> Image.Image:
    return Image.new("RGB", (SIZE, SIZE), WHITE)


def _circle(img: Image.Image, cx: int, cy: int, r: int, colour: str) -> None:
    draw = ImageDraw.Draw(img)
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=COLOURS[colour])


def _square(img: Image.Image, cx: int, cy: int, half: int, colour: str) -> None:
    draw = ImageDraw.Draw(img)
    draw.rectangle((cx - half, cy - half, cx + half, cy + half), fill=COLOURS[colour])


def _triangle(img: Image.Image, cx: int, cy: int, size: int, colour: str) -> None:
    draw = ImageDraw.Draw(img)
    draw.polygon(
        [(cx, cy - size), (cx - size, cy + size), (cx + size, cy + size)],
        fill=COLOURS[colour],
    )


def synthetic_set() -> list[dict]:
    """Deterministic small images with ground truth."""
    items: list[dict] = []

    def add(name, builder, question, accept, category):
        img = _blank()
        builder(img)
        items.append(
            {
                "name": name,
                "image": img,
                "question": question,
                "accept": accept,
                "category": category,
            }
        )

    add(
        "colour_red_circle",
        lambda im: _circle(im, 112, 112, 60, "red"),
        "What colour is the circle?",
        ["red"],
        "colour",
    )
    add(
        "colour_blue_square",
        lambda im: _square(im, 112, 112, 55, "blue"),
        "What colour is the square?",
        ["blue"],
        "colour",
    )
    add(
        "colour_green_triangle",
        lambda im: _triangle(im, 112, 120, 60, "green"),
        "What colour is the triangle?",
        ["green"],
        "colour",
    )
    add(
        "colour_yellow_circle",
        lambda im: _circle(im, 112, 112, 60, "yellow"),
        "What colour is the circle?",
        ["yellow"],
        "colour",
    )

    def two_circles(im):
        _circle(im, 70, 112, 35, "red")
        _circle(im, 160, 112, 35, "red")

    add(
        "count_two_circles",
        two_circles,
        "How many circles are in the image?",
        ["2", "two"],
        "count",
    )

    def three_squares(im):
        _square(im, 50, 70, 25, "blue")
        _square(im, 140, 70, 25, "blue")
        _square(im, 95, 165, 25, "blue")

    add(
        "count_three_squares",
        three_squares,
        "How many squares are in the image?",
        ["3", "three"],
        "count",
    )
    add(
        "count_one_shape",
        lambda im: _circle(im, 112, 112, 60, "green"),
        "How many shapes are in the image?",
        ["1", "one"],
        "count",
    )

    def red_circle_blue_square(im):
        _circle(im, 70, 112, 40, "red")
        _square(im, 160, 112, 40, "blue")

    add(
        "attr_square_colour",
        red_circle_blue_square,
        "What colour is the square?",
        ["blue"],
        "colour",
    )
    add(
        "attr_circle_colour",
        red_circle_blue_square,
        "What colour is the circle?",
        ["red"],
        "colour",
    )

    add(
        "presence_no_blue",
        lambda im: _circle(im, 112, 112, 60, "green"),
        "Is there a blue square in the image? Answer yes or no.",
        ["no"],
        "presence",
    )
    add(
        "presence_yes_blue",
        lambda im: _square(im, 112, 112, 55, "blue"),
        "Is there a blue square in the image? Answer yes or no.",
        ["yes"],
        "presence",
    )

    def red_left_blue_right(im):
        _circle(im, 60, 112, 35, "red")
        _circle(im, 164, 112, 35, "blue")

    add(
        "position_red_left",
        red_left_blue_right,
        "Is the red circle on the left side? Answer yes or no.",
        ["yes"],
        "position",
    )
    return items


def real_set() -> list[dict]:
    items: list[dict] = []
    bus = Image.open("assets/images/bus.jpg").convert("RGB")
    bus_small = bus.copy()
    bus_small.thumbnail((320, 320))
    items.append(
        {
            "name": "bus_colour",
            "image": bus_small,
            "question": "What colour is the bus? Answer with one colour word.",
            "accept": ["blue"],
            "category": "colour",
        }
    )
    items.append(
        {
            "name": "bus_presence",
            "image": bus_small,
            "question": "Is there a bus in this image? Answer yes or no.",
            "accept": ["yes"],
            "category": "presence",
        }
    )
    plain = Image.open("assets/test_image.png").convert("RGB")
    plain.thumbnail((320, 320))
    items.append(
        {
            "name": "plain_colour",
            "image": plain,
            "question": "What colour is this image? Answer with one colour word.",
            "accept": ["blue", "grey", "gray"],
            "category": "colour",
        }
    )
    return items


def encode(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def unique_encode(image: Image.Image, seed: int) -> str:
    """Encode with a tiny random crop so no two requests share an image.

    llama-server's mtmd media cache keys on image content; duplicate images in a
    run collide and return `failed to process mtmd chunk`.
    """
    rng = np.random.default_rng(seed)
    dx, dy = (int(v) for v in rng.integers(0, 7, 2))
    width, height = image.size
    return encode(image.crop((dx, dy, width, height)))


def ask_gemma(url: str, image_b64: str, question: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    r = httpx.post(
        f"{url}/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ],
            "max_tokens": 60,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=90,
    )
    try:
        content = r.json()["choices"][0]["message"].get("content", "")
    except Exception:  # noqa: BLE001
        content = f"HTTP {r.status_code} {r.text[:100]}"
    return content.strip(), time.perf_counter() - t0


def _media_marker(url: str) -> str:
    """llama-server generates a per-build media marker; fetch it from /props."""
    try:
        return httpx.get(f"{url}/props", timeout=5).json().get("media_marker", "<__media__>")
    except Exception:  # noqa: BLE001
        return "<__media__>"


def ask_moondream(url: str, image_b64: str, question: str) -> tuple[str, float]:
    marker = _media_marker(url)
    t0 = time.perf_counter()
    r = httpx.post(
        f"{url}/completion",
        json={
            "prompt": f"USER: {marker}\n{question}\nASSISTANT:",
            "image_data": [{"data": image_b64}],
            "n_predict": 60,
            "temperature": 0.0,
        },
        timeout=90,
    )
    try:
        return r.json().get("content", "").strip(), time.perf_counter() - t0
    except Exception:  # noqa: BLE001
        return f"HTTP {r.status_code} {r.text[:100]}", time.perf_counter() - t0


def score(answer: str, accept: list[str]) -> bool:
    lowered = answer.lower()
    return any(word in lowered for word in accept)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--protocol", choices=["gemma", "moondream"], default="gemma")
    parser.add_argument("--name", default="p17b_vision_accuracy")
    args = parser.parse_args()

    ask = ask_gemma if args.protocol == "gemma" else ask_moondream
    items = synthetic_set() + real_set()

    rows = []
    for index, item in enumerate(items):
        answer, wall = ask(args.url, unique_encode(item["image"], index), item["question"])
        ok = score(answer, item["accept"])
        rows.append(
            {
                **{k: item[k] for k in ("name", "category", "question")},
                "answer": answer[:80],
                "correct": ok,
                "wall_s": round(wall, 2),
            }
        )
        print(f"{'OK ' if ok else 'XX '}{item['name']:<24} {answer[:50]!r} ({wall:.2f}s)")

    by_cat: dict[str, list[bool]] = {}
    for row in rows:
        by_cat.setdefault(row["category"], []).append(row["correct"])
    summary = {
        "protocol": args.protocol,
        "n": len(rows),
        "correct": sum(r["correct"] for r in rows),
        "accuracy_pct": round(100 * sum(r["correct"] for r in rows) / len(rows)),
        "by_category": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_cat.items())},
        "wall_p50_s": round(statistics.median(r["wall_s"] for r in rows), 2),
        "rows": rows,
    }
    print(json.dumps({k: summary[k] for k in summary if k != "rows"}, indent=2))

    sys.path.insert(0, "bench")
    from common import write_result

    print(write_result(args.name, summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
