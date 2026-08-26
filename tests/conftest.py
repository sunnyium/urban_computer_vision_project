"""
Fixtures shared across the suite.

Data:
    image | a textured 96x128 frame, standing in for a street scene
    records | a miniature index, shaped exactly like prepare_data.py writes

Paths:
    paths | config's index and splits paths redirected into tmp_path
    on_disk | records whose images actually exist, with cfg.ROOT moved to them

The suite runs without BDD100K on disk. The dataset sits behind a registration
wall, and a suite that cannot run on a fresh clone is a suite nobody runs.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

import config as cfg
import labels as lb
import make_synthetic as ms

HEIGHT = 96
WIDTH = 128

# Clear is much the largest because train, val and the clear test bucket all
# come out of it. Fog is tiny on purpose, so the thin-bucket path runs
_COUNTS = {"clear": 40, "rain": 10, "snow": 10, "fog": 3, "night": 12}


@pytest.fixture
def records():
    """A miniature index, one record per image, as prepare_data.py writes them.
    Attributes come from make_synthetic.attributes_for rather than being typed
    out, so the pair really does derive back to the condition claimed here."""
    out = []
    for cond, n in _COUNTS.items():
        for i in range(n):
            name = f"{cond}_{i:03d}.jpg"
            attrs = ms.attributes_for(cond)
            out.append({
                "name": name,
                "file": f"data/synthetic/images/val/{name}",
                "condition": cond,
                "weather": attrs["weather"],
                "timeofday": attrs["timeofday"],
                # two boxes matching the blocks in `image`, and the class
                # rotates so every id including 1 and 10 appears somewhere
                "boxes": [[10.0, 20.0, 60.0, 80.0], [95.0, 30.0, 120.0, 78.0]],
                "labels": [lb.CLASSES[len(out) % len(lb.CLASSES)], "car"],
            })
    return out


@pytest.fixture
def paths(tmp_path, monkeypatch):
    """config's index and splits paths, pointed into tmp_path, as (index, splits).
    Both are module-level constants that build() reads and writes, so without
    this a test run overwrites whatever the last real run produced."""
    index = tmp_path / "index.json"
    splits = tmp_path / "splits.json"
    monkeypatch.setattr(cfg, "INDEX_PATH", index)
    monkeypatch.setattr(cfg, "SPLITS_PATH", splits)
    return index, splits


@pytest.fixture
def image():
    """A textured frame with a road, a sky and two blocks standing in for objects.
    Textured rather than flat because several corruptions are contrast
    operations, and a uniform field would let them pass as no-ops."""
    rng = np.random.default_rng(0)
    img = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    img[:HEIGHT // 2] = (110, 140, 190)
    img[HEIGHT // 2:] = (70, 70, 75)
    cv2.rectangle(img, (10, 20), (60, 80), (30, 60, 160), -1)
    cv2.rectangle(img, (95, 30), (120, 78), (200, 190, 180), -1)
    noise = rng.integers(-12, 13, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def on_disk(tmp_path, monkeypatch, records, image):
    """`records` with their images written, and cfg.ROOT pointed at tmp_path.
    dataset.py joins cfg.ROOT with each record's relative path, so the root has
    to move for the fixture to be readable at all."""
    monkeypatch.setattr(cfg, "ROOT", tmp_path)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for r in records:
        p = tmp_path / r["file"]
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), bgr)
    return records
