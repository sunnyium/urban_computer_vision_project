"""
The split protocol: train and val from clear weather alone, test by condition.

Building:
    build_splits() | index records -> {train, val, test} as image names
    buckets() | the splits flattened to one list per label, test kept apart

Checking:
    assert_disjoint() | no image appears in two buckets
    assert_clear_only() | train, val and the clear test bucket are clear only

Writing:
    build() | read the index, split, check both properties, write SPLITS_PATH

Training on clear weather alone is what makes this a measurement of distribution
shift rather than of memorisation. If adverse images reached train, a good rain
score would only show the model had seen rain before.

Both properties raise AssertionError rather than using a bare assert, because
python -O strips the latter, and these are the two failures that would
invalidate every number the project reports while nothing errored.

Names are stored, not records. The index stays the one description of what an
image is, so the two files cannot drift into disagreeing.
"""

from __future__ import annotations

import json

import numpy as np

import config as cfg
import labels as lb

# Fog cannot fill a bucket: BDD100K val holds 5 foggy daytime images
# Produces a warning instead of flagging all buckets
MIN_BUCKET = 2 if cfg.SMOKE else 50


def _shuffled(names: list[str], rng: np.random.Generator) -> list[str]:
    """A shuffled copy, sorted first so the result depends on the seed alone.
    Without the sort the partition also depends on the order prepare_data
    happened to write, which no test would catch and no reader would expect."""
    names = sorted(names)
    return [names[i] for i in rng.permutation(len(names))]


# 1. Building
def build_splits(recs: list[dict], seed: int = cfg.SEED) -> dict:
    """The three splits as image names, train and val drawn from clear only.
    The clear test bucket is what is left of the clear pool after train and val
    take their fractions, so it is the one bucket TEST_CAP rarely binds on."""
    rng = np.random.default_rng(seed)

    by_cond = {}
    for r in recs:
        by_cond.setdefault(r["condition"], []).append(r["name"])

    clear = _shuffled(by_cond.get("clear", []), rng)
    n_train = int(len(clear) * cfg.TRAIN_FRAC)
    n_val = int(len(clear) * cfg.VAL_FRAC)

    test = {"clear": clear[n_train + n_val:][:cfg.TEST_CAP]}
    for c in lb.ADVERSE:
        test[c] = _shuffled(by_cond.get(c, []), rng)[:cfg.TEST_CAP]

    return {
        "train": clear[:n_train],
        "val": clear[n_train:n_train + n_val],
        "test": test,
    }


def buckets(splits: dict) -> dict:
    """One flat label -> names mapping, with each test condition kept apart.
    Both checks and the report want the test conditions separate; nothing wants
    them pooled, so no caller has to remember to keep them apart."""
    out = {"train": splits["train"], "val": splits["val"]}
    for c, names in splits["test"].items():
        out[f"test/{c}"] = names
    return out


# 2. Checking
def assert_disjoint(splits: dict) -> None:
    """No image appears in two buckets.
    A train image reappearing in test raises every number in the study, and the
    failure is silent: the loss falls, the mAP rises, nothing errors."""
    where = {}
    for label, names in buckets(splits).items():
        for name in names:
            if name in where:
                raise AssertionError(
                    f"{name} is in both {where[name]} and {label}")
            where[name] = label


def assert_clear_only(splits: dict, cond: dict) -> None:
    """Train, val and the clear test bucket hold clear-weather images only.
    `cond` is name -> condition from the index, read back rather than carried
    through the split, so a bug in build_splits cannot also supply its alibi."""
    b = buckets(splits)
    for label in ("train", "val", "test/clear"):
        bad = [n for n in b[label] if cond[n] != "clear"]
        if bad:
            raise AssertionError(
                f"{label} holds {len(bad)} non-clear, first "
                f"{bad[0]} ({cond[bad[0]]})")


# 3. Writing
def build() -> tuple[dict, list[dict]]:
    """Split the index, check both properties, then write SPLITS_PATH.
    Nothing is written until the checks pass, so a failed run leaves the last
    good splits file in place rather than a half-valid one."""
    if not cfg.INDEX_PATH.exists():
        raise FileNotFoundError(
            f"{cfg.INDEX_PATH.relative_to(cfg.ROOT)} not found, "
            "run prepare_data.py first")

    recs = json.loads(cfg.INDEX_PATH.read_text(encoding="utf-8"))
    cond = {r["name"]: r["condition"] for r in recs}

    splits = build_splits(recs)
    assert_disjoint(splits)
    assert_clear_only(splits, cond)

    cfg.ensure_dirs()
    cfg.SPLITS_PATH.write_text(json.dumps(splits), encoding="utf-8")
    return splits, recs


if __name__ == "__main__":
    splits, recs = build()
    cond = {r["name"]: r["condition"] for r in recs}
    b = buckets(splits)

    print(f"{len(recs)} indexed images -> "
          f"{cfg.SPLITS_PATH.relative_to(cfg.ROOT)}\n")

    print("buckets")
    for label, names in b.items():
        flag = "" if len(names) >= MIN_BUCKET else "  THIN"
        print(f"  {label:<14}{len(names):>7}{flag}")

    # read the conditions back from the index rather than from the partition
    print("\nconditions found, which must be clear alone for the first three")
    for label, names in b.items():
        found = sorted({cond[n] for n in names}) or ["none"]
        print(f"  {label:<14}{', '.join(found)}")

    used = [n for names in b.values() for n in names]
    dupes = len(used) - len(set(used))
    mark = "" if not dupes else f"  {dupes} DUPLICATE"
    print(f"\n{len(used)} assignments, {len(set(used))} distinct{mark}")

    clear = sum(r["condition"] == "clear" for r in recs)
    print(f"{clear} clear images split {len(splits['train'])} train / "
          f"{len(splits['val'])} val / {len(b['test/clear'])} test")
    print(f"{len(recs) - len(used)} indexed images unused, held back by "
          f"TEST_CAP={cfg.TEST_CAP}")
