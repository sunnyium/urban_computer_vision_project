"""
Fixtures shared across the suite.

Data:
    records | a miniature index, shaped exactly like prepare_data.py writes

Paths:
    paths | config's index and splits paths redirected into tmp_path

The suite runs without BDD100K on disk. The dataset sits behind a registration
wall, and a suite that cannot run on a fresh clone is a suite nobody runs.
"""

from __future__ import annotations

import pytest

import config as cfg
import make_synthetic as ms

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
                "boxes": [[10.0, 20.0, 60.0, 80.0]],
                "labels": ["car"],
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
