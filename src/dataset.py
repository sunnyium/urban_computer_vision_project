"""
A torch Dataset over the prepared index, and the loaders that wrap it.

Reading:
    load_splits() | splits.json -> records per bucket, joined against the index

Dataset:
    UrbanDataset | one record -> (float32 CHW image in [0, 1], target dict)

Loaders:
    collate() | keep the per-image lists a detector wants, with no stacking
    make_loader() | a DataLoader over records, for training or for scoring

Nothing is resized here. The detector's own GeneralizedRCNNTransform rescales
image and boxes together, so a resize in this file would be a second chance to
desynchronise them for no gain.

augment and corrupt are mutually exclusive. augment perturbs a training set at a
severity drawn per call; corrupt builds an evaluation set at one fixed severity,
seeded by the record's index so all four arms are scored on pixel-identical
images. Applying both would corrupt a test image twice, at two severities.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import config as cfg
import labels as lb
import splits as sp
import weather as wx


# 1. Reading
def load_splits() -> dict:
    """Every bucket from splits.json, as records rather than names.
    The join happens here so the index stays the one description of an image and
    splits.json stays a list of names that cannot disagree with it."""
    if not cfg.SPLITS_PATH.exists():
        raise FileNotFoundError(
            f"{cfg.SPLITS_PATH} not found, run splits.py first")

    partition = json.loads(cfg.SPLITS_PATH.read_text(encoding="utf-8"))
    index = {r["name"]: r
             for r in json.loads(cfg.INDEX_PATH.read_text(encoding="utf-8"))}
    return {label: [index[n] for n in names]
            for label, names in sp.buckets(partition).items()}


# 2. Dataset
class UrbanDataset(Dataset):
    """Records in, what a torchvision detector expects out.
    Pass augment for a training set or corrupt for an evaluation set, never
    both. `rng` is per dataset rather than per worker, which is correct only
    while NUM_WORKERS is 0; workers would each inherit a copy and augment every
    epoch identically."""

    def __init__(self, records: list[dict], augment=None, corrupt: str | None = None,
        severity: int = cfg.EVAL_SEVERITY, seed: int = cfg.SEED,):
        if augment is not None and corrupt is not None:
            raise ValueError("pass augment or corrupt, not both")

        self.records = records
        self.augment = augment
        self.corrupt = corrupt
        self.severity = severity
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        r = self.records[i]
        path = cfg.ROOT / r["file"]
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise FileNotFoundError(f"could not read {path}")

        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        boxes = np.asarray(r["boxes"], np.float32).reshape(-1, 4)
        ids = [lb.class_id(n) for n in r["labels"]]

        if self.augment is not None:
            img, boxes, ids = self.augment(img, boxes, ids, self.rng)
        elif self.corrupt is not None:
            # seeded by index, not by the run, so every arm scores the same pixels
            img = wx.corrupt(img, self.corrupt, self.severity, i)

        target = {
            "boxes": torch.from_numpy(np.asarray(boxes, np.float32)),
            "labels": torch.tensor(list(ids), dtype=torch.int64),
        }
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0, target


# 3. Loaders
def collate(batch):
    """A batch as (images, targets), two tuples rather than two stacked tensors.
    Detectors take a list because the images keep their own sizes until the
    model's transform pads them together."""
    return tuple(zip(*batch))


def make_loader(records: list[dict], augment=None, corrupt: str | None = None,
    batch_size: int | None = None, shuffle: bool = False,) -> DataLoader:
    """A DataLoader over `records`, training flavoured when augment is passed."""
    ds = UrbanDataset(records, augment=augment, corrupt=corrupt)
    return DataLoader(ds, batch_size=batch_size or cfg.BATCH_SIZE,
                      shuffle=shuffle, num_workers=cfg.NUM_WORKERS,
                      collate_fn=collate)


if __name__ == "__main__":
    import augment as ag

    buckets = load_splits()
    print(f"{sum(len(v) for v in buckets.values())} images over "
          f"{len(buckets)} buckets\n")

    print("buckets")
    for label, recs in buckets.items():
        print(f"  {label:<14}{len(recs):>7}")

    recs = buckets["test/clear"][:8]
    images, targets = next(iter(make_loader(recs, batch_size=4)))
    img, t = images[0], targets[0]
    print(f"\none batch of {len(images)}")
    print(f"  {'image':<14}{tuple(img.shape)}  {img.dtype}  "
          f"[{img.min():.2f}, {img.max():.2f}]")
    print(f"  {'boxes':<14}{tuple(t['boxes'].shape)}  {t['boxes'].dtype}")
    print(f"  {'labels':<14}{tuple(t['labels'].shape)}  {t['labels'].dtype}  "
          f"ids {t['labels'].min().item()}-{t['labels'].max().item()}")

    checks = {
        "image is float32 CHW": img.dtype == torch.float32 and img.shape[0] == 3,
        "image is in [0, 1]": 0.0 <= float(img.min()) and float(img.max()) <= 1.0,
        "boxes are float32 xyxy": t["boxes"].dtype == torch.float32
                                  and t["boxes"].shape[1] == 4,
        "labels are 1-based ids": int(t["labels"].min()) >= 1
                                  and int(t["labels"].max()) < lb.NUM_CLASSES,
        "one label per box": len(t["labels"]) == len(t["boxes"]),
    }
    print()
    for name, ok in checks.items():
        print(f"  {name:<26}{'ok' if ok else 'FAILED'}")

    print(f"\ncorrupted at severity {cfg.EVAL_SEVERITY}, twice over")
    for name in wx.WEATHER_NAMES:
        a = UrbanDataset(recs, corrupt=name)[0][0]
        b = UrbanDataset(recs, corrupt=name)[0][0]
        clean = UrbanDataset(recs)[0][0]
        mark = "" if torch.equal(a, b) else "  NOT REPRODUCIBLE"
        print(f"  {name:<14}mean |dpx| {float((a - clean).abs().mean()) * 255:>6.1f}"
              f"{mark}")
    print("\ntwo datasets over the same records must agree pixel for pixel, or "
          "the arms are not scored on the same test set")
