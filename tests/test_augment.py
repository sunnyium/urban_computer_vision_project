"""
Tests for the four arms, mostly about what augmentation is allowed to touch.

Contract:
    test_shape_and_dtype_survive | RGB uint8 HxWx3 in, the same out
    test_input_is_not_mutated | the caller's image and boxes are left alone
    test_box_count_is_preserved | nothing here may drop a box
    test_labels_stay_aligned | one label per box, still matching

Boxes:
    test_flip_mirrors_the_boxes | x1 becomes W - x2, the property all of it rests on
    test_weather_never_moves_a_box | why the weather half sits outside albumentations

Arms:
    test_baseline_is_only_ever_a_flip | two possible outputs and no third
    test_only_the_weather_arms_corrupt | the switch in config.ARMS does something
    test_severity_stays_below_eval | no arm trains on the corruption it is scored on
    test_unknown_arm_is_rejected | the CLI hands this an arbitrary string

Determinism:
    test_same_seed_same_sequence | one augmenter, replayed
    test_different_seed_differs | and the seed is actually used

The box tests are the point. A flip that moved pixels without moving boxes would
not crash, would not show in the loss, and would quietly train every arm against
mislabelled data.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

import augment as ag
import config as cfg
import weather as wx

BOXES = np.array([[10., 20., 60., 80.], [95., 30., 120., 78.]], np.float32)
LABELS = [3, 1]


def fingerprint(img: np.ndarray) -> str:
    """A hash of the pixels, because mean() cannot see a horizontal flip."""
    return hashlib.blake2s(np.ascontiguousarray(img).tobytes(),
                           digest_size=8).hexdigest()


def augmenter(arm, seed=cfg.SEED):
    """One augmenter, built once and then called repeatedly.
    Rebuilding it per call replays the first draw of a fresh Compose every time,
    which reads as an arm that never fires rather than as a broken test."""
    return ag.build_augmenter(arm, seed=seed)


def once(arm, image, seed=cfg.SEED):
    """A single augmentation of the fixture image, as (img, boxes, labels)."""
    return augmenter(arm, seed)(image, BOXES, LABELS, np.random.default_rng(0))


# 1. Contract
@pytest.mark.parametrize("arm", list(cfg.ARMS))
def test_shape_and_dtype_survive(arm, image):
    out, _, _ = once(arm, image)
    assert out.shape == image.shape
    assert out.dtype == np.uint8


@pytest.mark.parametrize("arm", list(cfg.ARMS))
def test_input_is_not_mutated(arm, image):
    before, boxes = image.copy(), BOXES.copy()
    once(arm, image)
    assert np.array_equal(image, before)
    assert np.array_equal(BOXES, boxes)


@pytest.mark.parametrize("arm", list(cfg.ARMS))
def test_box_count_is_preserved(arm, image):
    aug, rng = augmenter(arm), np.random.default_rng(0)
    for _ in range(50):
        _, boxes, _ = aug(image, BOXES, LABELS, rng)
        assert boxes.shape == BOXES.shape


@pytest.mark.parametrize("arm", list(cfg.ARMS))
def test_labels_stay_aligned(arm, image):
    aug, rng = augmenter(arm), np.random.default_rng(0)
    for _ in range(50):
        _, boxes, labels = aug(image, BOXES, LABELS, rng)
        assert len(labels) == len(boxes)
        assert sorted(labels) == sorted(LABELS)


# 2. Boxes
def test_flip_mirrors_the_boxes(image, monkeypatch):
    """Boxes must follow the pixels exactly, or every arm trains on bad labels."""
    monkeypatch.setattr(cfg, "HFLIP_P", 1.0)
    out, boxes, _ = once("baseline", image)
    w = image.shape[1]
    want = [[w - b[2], b[1], w - b[0], b[3]] for b in BOXES]

    assert np.array_equal(out, np.fliplr(image))
    assert np.allclose(sorted(boxes.tolist()), sorted(want))


def test_weather_never_moves_a_box(image, monkeypatch):
    monkeypatch.setattr(cfg, "HFLIP_P", 0.0)
    aug, rng = augmenter("weather"), np.random.default_rng(0)
    for _ in range(100):
        _, boxes, _ = aug(image, BOXES, LABELS, rng)
        assert np.array_equal(boxes, BOXES)


# 3. Arms
def test_baseline_is_only_ever_a_flip(image):
    aug, rng = augmenter("baseline"), np.random.default_rng(0)
    seen = {fingerprint(aug(image, BOXES, LABELS, rng)[0]) for _ in range(200)}
    assert seen == {fingerprint(image), fingerprint(np.fliplr(image))}


@pytest.mark.parametrize("arm", list(cfg.ARMS))
def test_only_the_weather_arms_corrupt(arm, image, monkeypatch):
    """With the flip off, only an arm whose weather switch is on may change pixels."""
    monkeypatch.setattr(cfg, "HFLIP_P", 0.0)
    aug, rng = augmenter(arm), np.random.default_rng(0)
    changed = any(not np.array_equal(aug(image, BOXES, LABELS, rng)[0], image)
                  for _ in range(60))
    assert changed == (cfg.ARMS[arm]["photometric"] or cfg.ARMS[arm]["weather"])


def test_severity_stays_below_eval():
    assert max(cfg.AUG_SEVERITY) < cfg.EVAL_SEVERITY
    assert set(cfg.AUG_SEVERITY) <= set(wx.SEVERITIES)


def test_unknown_arm_is_rejected():
    with pytest.raises(KeyError, match="standard"):
        ag.build_augmenter("standard")


# 4. Determinism
@pytest.mark.parametrize("arm", list(cfg.ARMS))
def test_same_seed_same_sequence(arm, image):
    def seq():
        aug, rng = augmenter(arm, 11), np.random.default_rng(4)
        return [fingerprint(aug(image, BOXES, LABELS, rng)[0]) for _ in range(8)]

    assert seq() == seq()


@pytest.mark.parametrize("arm", [a for a in cfg.ARMS if a != "baseline"])
def test_different_seed_differs(arm, image):
    def seq(seed):
        aug, rng = augmenter(arm, seed), np.random.default_rng(4)
        return [fingerprint(aug(image, BOXES, LABELS, rng)[0]) for _ in range(8)]

    assert seq(11) != seq(12)
