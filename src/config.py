"""
Every path, switch and hyperparameter the pipeline reads, in one place.

Run scale:
    SMOKE | shrink every loop so the whole pipeline runs on CPU in minutes
    device() | cuda when there is one, cpu otherwise

Paths:
    IMAGE_ROOT, LABEL_ROOT | where images and labels are read from, redirected
        to data/synthetic under SMOKE
    INDEX_PATH, SPLITS_PATH | the files prepare_data.py and splits.py write
    ensure_dirs() | create the directories a run writes into

Arms:
    ARMS | the four training strategies, as a 2x2 over two augmentation
        switches

Arms differ only in which augmentations are on. Architecture, schedule, seed and
images are fixed.
"""

from __future__ import annotations

import os
from pathlib import Path

# Environment switch so every module and subprocess agrees. One fast epoch on a
# few hundred images, to catch wiring errors.
SMOKE = os.environ.get("SMOKE", "0") == "1"


# 1. Run scale
SEED = 42

EPOCHS = 1 if SMOKE else 10
BATCH_SIZE = 2 if SMOKE else 4

# Faster R-CNN rescales internally, so these bound the shorter and longer edge
MIN_SIZE = 256 if SMOKE else 600
MAX_SIZE = 448 if SMOKE else 1000

LR = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
LR_STEP = 3
LR_GAMMA = 0.1

# Windows does not fork worker processes; it spawns them
NUM_WORKERS = 0

# Fractions of the clear-weather pool. The remainder is the clear test set.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

# Cap per condition to prevent small vs. large test set comparisons
TEST_CAP = 50 if SMOKE else 500

# Only 5 real foggy daytime images in val, too few for a retention ratio, so fog
# is scored on corrupted clear-test images only and the real count is a footnote
REAL_CONDITIONS = ("rain", "snow", "night")
SYNTHETIC_CONDITIONS = ("rain", "snow", "fog", "night")

# Medium severity to separate the arms but not have every arm score ~0
EVAL_SEVERITY = 3

SCORE_THRESH = 0.5


def device():
    """The device to train and evaluate on, cuda when one is visible.
    torch is imported inside the call because config reaches modules that never
    touch a tensor."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 2. Paths
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
SPLITS = DATA / "splits"
RESULTS = DATA / "results"
SYNTHETIC = DATA / "synthetic"
CHECKPOINTS = ROOT / "checkpoints"
MLRUNS = ROOT / "mlruns"

# Both archives unpack to a nested bdd100k/ level, so neither root is where the
# download page implies
BDD_IMAGES = RAW / "bdd100k" / "bdd100k" / "images" / "100k"
BDD_LABELS = RAW / "bdd100k_labels_release" / "bdd100k" / "labels"

IMAGE_ROOT = SYNTHETIC / "images" if SMOKE else BDD_IMAGES
LABEL_ROOT = SYNTHETIC / "labels" if SMOKE else BDD_LABELS

# val only: its 10,000 images arrived complete, 100k/train is short of 70,000
SOURCE_SPLITS = ("val",)

# Smoke artifacts have a unique name
_SUFFIX = "_smoke" if SMOKE else ""
INDEX_PATH = PROCESSED / f"index{_SUFFIX}.json"
SPLITS_PATH = SPLITS / f"splits{_SUFFIX}.json"


def label_json(split: str) -> Path:
    """The label file for one BDD100K split.
    make_synthetic.py writes fake labels under the same name, so prepare_data.py
    reads real and generated through one path."""
    return LABEL_ROOT / f"bdd100k_labels_images_{split}.json"


def checkpoint_dir(arm: str) -> Path:
    """Where one arm's weights go, one directory per arm."""
    return CHECKPOINTS / f"{arm}{_SUFFIX}"


def ensure_dirs() -> None:
    """Create every directory a run writes into."""
    for d in (PROCESSED, SPLITS, RESULTS, SYNTHETIC, CHECKPOINTS):
        d.mkdir(parents=True, exist_ok=True)


# 3. Arms
# Two switches, four configurations. Pairs differing by one switch isolate that
# switch's effect. Horizontal flip is on everywhere.
ARMS = {
    "baseline": {"photometric": False, "weather": False},
    "photometric": {"photometric": True, "weather": False},
    "weather": {"photometric": False, "weather": True},
    "combined": {"photometric": True, "weather": True},
}

HFLIP_P = 0.5

# How often a training image gets non-clear weather
AUG_WEATHER_P = 0.5

# Below EVAL_SEVERITY so models never train on the exact test corruptions
AUG_SEVERITY = (1, 2)


if __name__ == "__main__":
    ensure_dirs()

    print(f"SMOKE={'on' if SMOKE else 'off'}  device={device()}  seed={SEED}\n")

    print(f"{'epochs':<14}{EPOCHS}")
    print(f"{'batch size':<14}{BATCH_SIZE}")
    print(f"{'image size':<14}{MIN_SIZE}-{MAX_SIZE}")
    print(f"{'test cap':<14}{TEST_CAP} per condition")

    print("\npaths")
    for name, p in (("images", IMAGE_ROOT), ("labels", LABEL_ROOT),
                    ("index", INDEX_PATH), ("splits", SPLITS_PATH)):
        mark = "" if p.exists() else "  (missing)"
        print(f"  {name:<10}{p.relative_to(ROOT)}{mark}")

    print("\narms")
    for name, flags in ARMS.items():
        on = [k for k, v in flags.items() if v] or ["none"]
        print(f"  {name:<14}hflip + {', '.join(on)}")
