"""
Tests for the loader, mostly about the contract a torchvision detector expects.

Contract:
    test_image_is_float32_chw | 3xHxW, the layout the model takes
    test_image_is_in_unit_range | scaled by 255 exactly once
    test_boxes_are_float32_xyxy | (N, 4), the format the index stores
    test_labels_are_one_based_ids | 0 is background and never appears
    test_labels_round_trip_to_the_names | an off-by-one a range check misses
    test_one_label_per_box | the pairing nothing downstream re-checks
    test_length_matches_records | no record is skipped

Corruption:
    test_corruption_is_reproducible | two datasets agree pixel for pixel
    test_corruption_is_seeded_by_index | the same frame corrupts differently
    test_corruption_changes_the_image | and it is not a no-op
    test_a_clean_dataset_leaves_pixels_alone | no corruption unless asked

Guards:
    test_augment_and_corrupt_together_is_rejected | the two modes are exclusive
    test_a_missing_image_names_the_file | cv2 returns None rather than raising
    test_missing_splits_names_the_fix | the error says what to run

Loaders:
    test_collate_keeps_per_image_lists | images are not stacked
    test_loader_covers_every_record | batching loses nothing

test_corruption_is_reproducible is the one that matters. Every arm is scored on
the same corrupted test set, and if those pixels differ between runs then part
of the gap between two arms is just corruption sampling noise.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import config as cfg
import dataset as ds
import labels as lb
import weather as wx


# 1. Contract
def test_image_is_float32_chw(on_disk):
    img, _ = ds.UrbanDataset(on_disk)[0]
    assert img.dtype == torch.float32
    assert img.shape[0] == 3 and img.ndim == 3


def test_image_is_in_unit_range(on_disk):
    img, _ = ds.UrbanDataset(on_disk)[0]
    assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0


def test_boxes_are_float32_xyxy(on_disk):
    _, t = ds.UrbanDataset(on_disk)[0]
    assert t["boxes"].dtype == torch.float32
    assert t["boxes"].shape[1] == 4
    assert bool((t["boxes"][:, 2] > t["boxes"][:, 0]).all())
    assert bool((t["boxes"][:, 3] > t["boxes"][:, 1]).all())


def test_labels_are_one_based_ids(on_disk):
    _, t = ds.UrbanDataset(on_disk)[0]
    assert t["labels"].dtype == torch.int64
    assert int(t["labels"].min()) >= 1
    assert int(t["labels"].max()) < lb.NUM_CLASSES


def test_labels_round_trip_to_the_names(on_disk):
    """Ids come from a position in CLASSES, so a range check alone passes an
    off-by-one. Going back through class_name pins the mapping itself."""
    data = ds.UrbanDataset(on_disk)
    for i in (0, 7, 30, len(on_disk) - 1):
        _, t = data[i]
        assert [lb.class_name(int(v)) for v in t["labels"]] == on_disk[i]["labels"]


def test_one_label_per_box(on_disk):
    data = ds.UrbanDataset(on_disk)
    for i in range(len(data)):
        _, t = data[i]
        assert len(t["labels"]) == len(t["boxes"])


def test_length_matches_records(on_disk):
    assert len(ds.UrbanDataset(on_disk)) == len(on_disk)


# 2. Corruption
@pytest.mark.parametrize("name", wx.WEATHER_NAMES)
def test_corruption_is_reproducible(on_disk, name):
    a = ds.UrbanDataset(on_disk, corrupt=name)[3][0]
    b = ds.UrbanDataset(on_disk, corrupt=name)[3][0]
    assert torch.equal(a, b)


@pytest.mark.parametrize("name", wx.WEATHER_NAMES)
def test_corruption_is_seeded_by_index(on_disk, name):
    """Every record holds the same frame, so any difference is the index seed."""
    data = ds.UrbanDataset(on_disk, corrupt=name)
    assert not torch.equal(data[0][0], data[1][0])


@pytest.mark.parametrize("name", wx.WEATHER_NAMES)
def test_corruption_changes_the_image(on_disk, name):
    clean = ds.UrbanDataset(on_disk)[0][0]
    assert not torch.equal(ds.UrbanDataset(on_disk, corrupt=name)[0][0], clean)


def test_a_clean_dataset_leaves_pixels_alone(on_disk):
    a = ds.UrbanDataset(on_disk)[0][0]
    b = ds.UrbanDataset(on_disk)[0][0]
    assert torch.equal(a, b)


# 3. Guards
def test_augment_and_corrupt_together_is_rejected(on_disk):
    with pytest.raises(ValueError, match="not both"):
        ds.UrbanDataset(on_disk, augment=lambda *a: a, corrupt="fog")


def test_a_missing_image_names_the_file(on_disk, tmp_path):
    gone = dict(on_disk[0], file="data/nowhere/missing.jpg")
    with pytest.raises(FileNotFoundError, match="missing.jpg"):
        ds.UrbanDataset([gone])[0]


def test_missing_splits_names_the_fix(paths):
    with pytest.raises(FileNotFoundError, match="splits.py"):
        ds.load_splits()


def test_load_splits_joins_against_the_index(on_disk, paths, tmp_path):
    """Names in, records out, with every bucket carrying whole records."""
    import splits as sp
    index, out = paths
    index.write_text(json.dumps(on_disk), encoding="utf-8")
    partition = sp.build_splits(on_disk)
    out.write_text(json.dumps(partition), encoding="utf-8")

    buckets = ds.load_splits()
    assert set(buckets) == set(sp.buckets(partition))
    assert sum(len(v) for v in buckets.values()) == sum(
        len(v) for v in sp.buckets(partition).values())
    assert all("boxes" in r and "condition" in r
               for recs in buckets.values() for r in recs)


# 4. Loaders
def test_collate_keeps_per_image_lists(on_disk):
    loader = ds.make_loader(on_disk[:6], batch_size=3)
    images, targets = next(iter(loader))
    assert isinstance(images, tuple) and isinstance(targets, tuple)
    assert len(images) == len(targets) == 3
    assert all(isinstance(i, torch.Tensor) and i.ndim == 3 for i in images)


def test_loader_covers_every_record(on_disk):
    seen = sum(len(images)
               for images, _ in ds.make_loader(on_disk[:10], batch_size=3))
    assert seen == 10
