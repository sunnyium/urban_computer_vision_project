"""
A fake dataset shaped exactly like BDD100K, so the pipeline runs without it.

Scene:
    draw_scene() | a crude street with coloured boxes for objects, and the
        boxes that describe them

Attributes:
    sample_condition() | a condition drawn from a fixed quota
    attributes_for() | condition -> the (weather, timeofday) pair that produces
        it

Writing:
    build() | write n images and one BDD100K-shaped label file
    main() | the CLI, n as the one positional argument

Colour is fixed per class so the task is learnable rather than noise. A smoke run
whose loss cannot fall cannot tell a wiring bug from a task with no signal.

Conditions come from a quota, not the real BDD100K proportions. Sampling the real
distribution gives a 100-image fixture ~1 foggy image or none, and an empty
condition exercises the skip path instead of the scoring path.
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

import config as cfg
import labels as lb

# Same aspect as the real 1280x720 frames, small enough to train on a CPU
HEIGHT = 180
WIDTH = 320

# Flatter than reality: val has 15 train instances in 10,000 images, so faithful
# sampling leaves class 6 unseen and lets a class-index bug survive
CLASS_WEIGHTS = {
    "car": 0.30,
    "traffic sign": 0.14,
    "traffic light": 0.12,
    "pedestrian": 0.12,
    "truck": 0.08,
    "bus": 0.07,
    "bicycle": 0.06,
    "rider": 0.05,
    "motorcycle": 0.03,
    "train": 0.03,
}

# One distinct hue per class, so class identity is recoverable from pixels
_COLOURS = {
    "pedestrian": (220, 90, 90),
    "rider": (200, 120, 40),
    "car": (70, 130, 220),
    "truck": (40, 80, 160),
    "bus": (250, 190, 60),
    "train": (150, 90, 200),
    "motorcycle": (240, 140, 200),
    "bicycle": (90, 200, 140),
    "traffic light": (60, 220, 90),
    "traffic sign": (230, 60, 60),
}


# 1. Scene
def draw_scene(rng: np.random.Generator) -> tuple[np.ndarray, list[dict]]:
    """One frame and the boxes in it, as (image, [{category, box2d}, ...])."""
    img = np.zeros((HEIGHT, WIDTH, 3), np.float32)
    horizon = int(HEIGHT * rng.uniform(0.35, 0.5))

    sky = rng.uniform(150, 210)
    road = rng.uniform(60, 100)
    img[:horizon] = (sky * 0.85, sky * 0.92, sky)
    img[horizon:] = (road, road, road * 0.98)
    img += rng.normal(0, 4, img.shape)

    names = list(CLASS_WEIGHTS)
    probs = np.array([CLASS_WEIGHTS[n] for n in names])
    out = []
    for _ in range(int(rng.integers(1, 7))):
        cat = str(rng.choice(names, p=probs / probs.sum()))

        w = int(rng.integers(14, 52))
        h = int(w * rng.uniform(0.5, 1.6))
        x1 = int(rng.integers(0, max(WIDTH - w, 1)))
        y1 = int(rng.integers(max(horizon - h // 2, 0), max(HEIGHT - h, 1)))
        x2, y2 = min(x1 + w, WIDTH - 1), min(y1 + h, HEIGHT - 1)

        # img is RGB and stays RGB until the single conversion on write
        cv2.rectangle(img, (x1, y1), (x2, y2), _COLOURS[cat], -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (20, 20, 20), 1)
        out.append({
            "category": cat,
            "box2d": {"x1": float(x1), "y1": float(y1),
                      "x2": float(x2), "y2": float(y2)},
        })

    return np.clip(img, 0, 255).astype(np.uint8), out


# 2. Attributes
# Clear dominates because train, val and the clear test set all come out of it.
# "other" is present so the discard path runs too
_QUOTA = {
    "clear": 0.45,
    "night": 0.20,
    "rain": 0.10,
    "snow": 0.10,
    "fog": 0.05,
    "other": 0.10,
}

_ATTRIBUTES = {
    "clear": ("clear", "daytime"),
    "rain": ("rainy", "daytime"),
    "snow": ("snowy", "daytime"),
    "fog": ("foggy", "daytime"),
    "night": ("rainy", "night"),
    "other": ("overcast", "dawn/dusk"),
}


def sample_condition(i: int, n: int) -> str:
    """The condition for image `i` of `n`, filled to quota rather than sampled.
    A 100-image fixture gets exactly 5 foggy frames every time, so a test that
    counts them is not flaky."""
    edges, acc = [], 0.0
    for name, share in _QUOTA.items():
        acc += share
        edges.append((name, acc * n))
    return next(name for name, edge in edges if i < edge)


def attributes_for(condition: str) -> dict:
    """A BDD100K attributes block that derive_condition maps back to `condition`.
    Night gets rainy weather so the fixture covers the precedence rule, not just
    the case where weather and time of day agree."""
    weather, timeofday = _ATTRIBUTES[condition]
    return {"weather": weather, "scene": "city street", "timeofday": timeofday}


# 3. Writing
def build(n: int, split: str = "val") -> dict:
    """Write `n` images and one label file, and return the condition counts."""
    img_dir = cfg.SYNTHETIC / "images" / split
    lbl_path = cfg.SYNTHETIC / "labels" / f"bdd100k_labels_images_{split}.json"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.SEED)
    recs, counts = [], {}
    for i in range(n):
        cond = sample_condition(i, n)
        counts[cond] = counts.get(cond, 0) + 1

        img, boxes = draw_scene(rng)
        name = f"{i:06d}-synthetic.jpg"
        cv2.imwrite(str(img_dir / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        recs.append({
            "name": name,
            "timestamp": 10000,
            "attributes": attributes_for(cond),
            "labels": [
                {"id": j, "category": b["category"], "box2d": b["box2d"],
                 "attributes": {"occluded": False, "truncated": False}}
                for j, b in enumerate(boxes)
            ],
        })

    lbl_path.write_text(json.dumps(recs), encoding="utf-8")
    return counts


def main() -> None:
    """Build the fixture, then read the labels back and check they round-trip.
    Counting through derive_condition rather than the quota catches an
    attributes block that no longer maps where it claims."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("n", nargs="?", type=int, default=100)
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    counts = build(args.n, args.split)
    root = cfg.SYNTHETIC / "images" / args.split
    print(f"wrote {args.n} images to {root.relative_to(cfg.ROOT)}")

    print("\nconditions, as derive_condition reads them back")
    recs = json.loads((cfg.SYNTHETIC / "labels"
                       / f"bdd100k_labels_images_{args.split}.json").read_text())
    got = {}
    for r in recs:
        c = lb.derive_condition(r["attributes"]["weather"],
                                r["attributes"]["timeofday"])
        got[c] = got.get(c, 0) + 1
    for c in (*lb.CONDITIONS, lb.OTHER):
        flag = "" if got.get(c, 0) == counts.get(c, 0) else "  MISMATCH"
        print(f"  {c:<8}{got.get(c, 0):>5}  intended {counts.get(c, 0):>5}{flag}")

    boxes = [l["category"] for r in recs for l in r["labels"]]
    print(f"\n{len(boxes)} boxes over {len(recs)} images")
    for name in lb.CLASSES:
        print(f"  {name:<15}{boxes.count(name):>5}")


if __name__ == "__main__":
    main()
