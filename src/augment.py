"""
The four augmentation strategies, one callable per arm.

Pipeline:
    pipeline() | hflip always, colour jitter when the arm asks for it
    sample_weather() | one of the six corruptions at a training severity

Building:
    build_augmenter() | arm name -> the callable dataset.py applies per image

config.ARMS holds the two switches deciding which half runs and nothing
else in the pipeline branches on the arm. The four runs are controlled.

Only hflip moves a box, so it goes through albumentations, which transforms the
boxes with the image. The weather half moves no pixel to a new coordinate.

Colour jitter runs first, so the corruption is the last thing to reach the
pixels.
"""

from __future__ import annotations

import os

# albumentations pings PyPI for a version check at import
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import numpy as np

import config as cfg
import weather as wx

# motion_blur and noise have no BDD100K attribute to build a real test column from
TRAIN_CORRUPTIONS = tuple(wx.CORRUPTIONS)

# pascal_voc is xyxy in absolute pixels
# hflip is the only op that moves a box and it moves none out of frame, so there
# is nothing for min_area or min_visibility to catch. 
# Boxes must already lie inside the frame: clipping is
# left off so an upstream bug surfaces
BBOX_PARAMS = A.BboxParams(format="pascal_voc", label_fields=["labels"])


# 1. Pipeline
def pipeline(photometric: bool, seed: int = cfg.SEED) -> A.Compose:
    """The albumentations half for one arm, hflip always and jitter optionally.
    Seeded per Compose rather than globally: left alone, albumentations 1.4
    draws from the process-wide numpy and random state, which is the one source
    of randomness in this project that nothing else reaches into."""
    ops = [A.HorizontalFlip(p=cfg.HFLIP_P)]
    if photometric:
        ops += [
            A.RandomBrightnessContrast(p=cfg.AUG_PHOTOMETRIC_P),
            A.HueSaturationValue(p=cfg.AUG_PHOTOMETRIC_P),
        ]
    return A.Compose(ops, bbox_params=BBOX_PARAMS, seed=seed)


def sample_weather(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One corruption at a training severity, drawn fresh for this image.
    Severity comes from AUG_SEVERITY, which sits below EVAL_SEVERITY, so no arm
    trains on the exact corruption it is later scored against. The seed handed
    to corrupt comes off `rng` too, so one Generator drives the whole draw."""
    name = str(rng.choice(TRAIN_CORRUPTIONS))
    sev = int(rng.choice(cfg.AUG_SEVERITY))
    return wx.corrupt(img, name, sev, int(rng.integers(1 << 31)))


# 2. Building
def build_augmenter(arm: str, seed: int = cfg.SEED):
    """The callable dataset.py applies, as augment(img, boxes, labels, rng).
    Boxes come back as a float32 (N, 4) array and labels as a list, in matching
    order. The arm is read once here, so the dataset never learns which arm it
    is serving and cannot branch on it."""
    if arm not in cfg.ARMS:
        raise KeyError(f"unknown arm {arm!r}, expected one of {tuple(cfg.ARMS)}")

    flags = cfg.ARMS[arm]
    boxed = pipeline(flags["photometric"], seed)
    weather_on = flags["weather"]

    def augment(img: np.ndarray, boxes, labels,
        rng: np.random.Generator,) -> tuple[np.ndarray, np.ndarray, list]:
        out = boxed(image=img, bboxes=np.asarray(boxes, np.float32),
                    labels=list(labels))
        img, boxes, labels = out["image"], out["bboxes"], out["labels"]

        if weather_on and rng.random() < cfg.AUG_WEATHER_P:
            img = sample_weather(img, rng)

        return img, np.asarray(boxes, np.float32).reshape(-1, 4), labels

    return augment


if __name__ == "__main__":
    import cv2

    import make_synthetic as ms

    img, raw = ms.draw_scene(np.random.default_rng(cfg.SEED))
    boxes = np.array([[b["box2d"][k] for k in ("x1", "y1", "x2", "y2")]
                      for b in raw], np.float32)
    labels = [b["category"] for b in raw]
    print(f"{len(boxes)} boxes on a {img.shape[1]}x{img.shape[0]} frame\n")

    print("arms")
    for name, flags in cfg.ARMS.items():
        on = [k for k, v in flags.items() if v] or ["none"]
        print(f"  {name:<14}hflip + {', '.join(on)}")

    n = 400
    print(f"\n{n} calls per arm")
    print(f"  {'':<14}{'changed':<10}{'flipped':<10}{'mean |dpx|':<12}boxes")
    for name in cfg.ARMS:
        aug = build_augmenter(name)
        rng = np.random.default_rng(0)
        changed = flipped = 0
        delta, kept = [], set()
        for _ in range(n):
            out, bx, lb = aug(img, boxes, labels, rng)
            changed += not np.array_equal(out, img)
            flipped += not np.allclose(bx, boxes)
            delta.append(float(np.abs(out.astype(np.int16)
                                      - img.astype(np.int16)).mean()))
            kept.add((len(bx), len(lb)))
        mark = "" if kept == {(len(boxes), len(boxes))} else "  BOX COUNT MOVED"
        print(f"  {name:<14}{changed / n:<10.2f}{flipped / n:<10.2f}"
              f"{np.mean(delta):<12.1f}{sorted(kept)}{mark}")
    print(f"\nflipped should sit at HFLIP_P={cfg.HFLIP_P} for every arm, and "
          "baseline changes pixels only when it flips")

    strip = [aug(img, boxes, labels, np.random.default_rng(3))[0]
             for aug in (build_augmenter(a) for a in cfg.ARMS)]
    grid = np.hstack([img] + strip)
    out_path = cfg.RESULTS / "augment_examples.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"wrote {out_path.relative_to(cfg.ROOT)}, source then the four arms")
