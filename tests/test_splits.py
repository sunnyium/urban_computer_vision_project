"""
Tests for the split protocol, mostly about the two properties it asserts.

Protocol:
    test_train_and_val_are_clear_only | the pool those two are drawn from
    test_each_test_bucket_holds_one_condition | no mixing inside a bucket
    test_train_and_val_take_their_fractions | TRAIN_FRAC and VAL_FRAC
    test_the_clear_pool_is_fully_spent | no clear image quietly disappears
    test_buckets_respect_the_cap | TEST_CAP binds on every condition

Leakage:
    test_a_shared_image_is_caught | the disjointness guard fires
    test_an_adverse_image_is_caught | the clear-only guard fires, per bucket
    test_a_clean_partition_passes_both | and neither fires on good input

Determinism:
    test_same_seed_same_partition | reruns compare
    test_different_seed_differs | and the seed is actually used
    test_index_order_does_not_matter | the sort inside _shuffled

Writing:
    test_build_round_trips | what build writes, json reads back
    test_nothing_is_written_when_a_check_fails | the old file survives
    test_a_missing_index_names_the_fix | the error says what to run

A leaked train image makes every score rise without raising an error.
A guard that stopped firing would appears like a good run.
"""

from __future__ import annotations

import json

import pytest

import config as cfg
import splits as sp


def conditions(records) -> dict:
    """name -> condition, the mapping assert_clear_only wants."""
    return {r["name"]: r["condition"] for r in records}


# 1. Protocol
@pytest.mark.parametrize("label", ["train", "val"])
def test_train_and_val_are_clear_only(records, label):
    cond = conditions(records)
    assert {cond[n] for n in sp.build_splits(records)[label]} == {"clear"}


def test_each_test_bucket_holds_one_condition(records):
    cond = conditions(records)
    for c, names in sp.build_splits(records)["test"].items():
        assert {cond[n] for n in names} == {c}, f"{c} bucket is mixed"


def test_train_and_val_take_their_fractions(records):
    n = sum(r["condition"] == "clear" for r in records)
    splits = sp.build_splits(records)
    assert len(splits["train"]) == int(n * cfg.TRAIN_FRAC)
    assert len(splits["val"]) == int(n * cfg.VAL_FRAC)


def test_the_clear_pool_is_fully_spent(records, monkeypatch):
    """Nothing falls between the two fractions and the remainder."""
    monkeypatch.setattr(cfg, "TEST_CAP", 10_000)
    clear = {r["name"] for r in records if r["condition"] == "clear"}
    s = sp.build_splits(records)
    assert set(s["train"]) | set(s["val"]) | set(s["test"]["clear"]) == clear


@pytest.mark.parametrize("cap", [1, 3, 5])
def test_buckets_respect_the_cap(records, monkeypatch, cap):
    monkeypatch.setattr(cfg, "TEST_CAP", cap)
    for names in sp.build_splits(records)["test"].values():
        assert len(names) <= cap


# 2. Leakage
def test_a_shared_image_is_caught(records):
    s = sp.build_splits(records)
    s["test"]["rain"].append(s["train"][0])
    with pytest.raises(AssertionError, match="train"):
        sp.assert_disjoint(s)


@pytest.mark.parametrize("label", ["train", "val", "clear"])
def test_an_adverse_image_is_caught(records, label):
    cond = conditions(records)
    s = sp.build_splits(records)
    night = next(n for n, c in cond.items() if c == "night")
    bucket = s["test"]["clear"] if label == "clear" else s[label]
    bucket.append(night)
    with pytest.raises(AssertionError, match="non-clear"):
        sp.assert_clear_only(s, cond)


def test_a_clean_partition_passes_both(records):
    s = sp.build_splits(records)
    sp.assert_disjoint(s)
    sp.assert_clear_only(s, conditions(records))


# 3. Determinism
def test_same_seed_same_partition(records):
    assert sp.build_splits(records, seed=7) == sp.build_splits(records, seed=7)


def test_different_seed_differs(records):
    assert sp.build_splits(records, seed=1) != sp.build_splits(records, seed=2)


def test_index_order_does_not_matter(records):
    """_shuffled sorts first, so the partition rests on the seed alone and not
    on the order prepare_data happened to write."""
    assert sp.build_splits(records[::-1]) == sp.build_splits(records)


# 4. Writing
def test_build_round_trips(records, paths):
    index, out = paths
    index.write_text(json.dumps(records), encoding="utf-8")

    splits, got = sp.build()
    assert got == records
    assert json.loads(out.read_text(encoding="utf-8")) == splits


def test_nothing_is_written_when_a_check_fails(records, paths, monkeypatch):
    """A run that trips a guard leaves the last good splits file in place."""
    index, out = paths
    index.write_text(json.dumps(records), encoding="utf-8")
    out.write_text('{"keep": "me"}', encoding="utf-8")

    leaky = sp.build_splits(records)
    leaky["test"]["rain"].append(leaky["train"][0])
    monkeypatch.setattr(sp, "build_splits", lambda *a, **k: leaky)

    with pytest.raises(AssertionError):
        sp.build()
    assert json.loads(out.read_text(encoding="utf-8")) == {"keep": "me"}


def test_a_missing_index_names_the_fix(paths):
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        sp.build()
