"""
BDD100K's label JSON flattened to one record per image, tagged by condition.

Reading:
    boxes_for() | one image's label list -> (boxes, class names), size filtered
    read_split() | one split's label file -> records, and what did not survive

Writing:
    build() | every split in SOURCE_SPLITS -> INDEX_PATH, and the same counts

Condition is determined in this file, so the project has one definition to 
inspect. Images that can not be determined are classified as "other."

Class names are stored as a position in labels.CLASSES and will become invalid 
if the tuple is reordered.
"""

from __future__ import annotations

import json

import config as cfg
import labels as lb


# 1. Reading
def boxes_for(labels: list[dict]) -> tuple[list[list[float]], list[str]]:
    """The boxes worth training on, as (xyxy boxes, matching class names).
    Drops the lane and drivable-area entries, which carry a polygon and no box,
    then the categories outside the ten, then anything under MIN_BOX_SIZE."""
    boxes, names = [], []
    for l in labels:
        box = l.get("box2d")
        if box is None:
            continue
        name = lb.normalise_category(l.get("category", ""))
        if name is None:
            continue

        x1, y1, x2, y2 = (float(box[k]) for k in ("x1", "y1", "x2", "y2"))
        if x2 - x1 < cfg.MIN_BOX_SIZE or y2 - y1 < cfg.MIN_BOX_SIZE:
            continue

        boxes.append([x1, y1, x2, y2])
        names.append(name)

    return boxes, names


def read_split(split: str) -> tuple[list[dict], dict]:
    """Records for one BDD100K split, and a count of what did not survive.
    Reads through config.label_json, so a SMOKE run parses make_synthetic.py's
    fixture by the same path a real run parses the download."""
    recs = json.loads(cfg.label_json(split).read_text(encoding="utf-8"))
    root = cfg.IMAGE_ROOT / split

    out, dropped = [], {"other": 0, "empty": 0, "boxes": 0}
    for r in recs:
        attrs = r.get("attributes", {})
        cond = lb.derive_condition(attrs.get("weather", ""),
                                   attrs.get("timeofday", ""))
        if cond == lb.OTHER:
            dropped["other"] += 1
            continue

        labels = r.get("labels") or []
        boxes, names = boxes_for(labels)
        dropped["boxes"] += sum(l.get("box2d") is not None
                                for l in labels) - len(boxes)
        if not boxes:
            dropped["empty"] += 1
            continue

        out.append({
            "name": r["name"],
            # relative to ROOT, so the index does not bake in a checkout location
            "file": (root / r["name"]).relative_to(cfg.ROOT).as_posix(),
            "condition": cond,
            "weather": attrs.get("weather", ""),
            "timeofday": attrs.get("timeofday", ""),
            "boxes": boxes,
            "labels": names,
        })

    return out, dropped


# 2. Writing
def build() -> tuple[list[dict], dict]:
    """Write one index over every split in SOURCE_SPLITS, and return what went in.
    One file rather than one per split, because splits.py pools and repartitions
    on condition anyway and BDD100K's own boundary never survives that."""
    cfg.ensure_dirs()

    recs, dropped = [], {}
    for split in cfg.SOURCE_SPLITS:
        part, d = read_split(split)
        recs += part
        for k, v in d.items():
            dropped[k] = dropped.get(k, 0) + v

    cfg.INDEX_PATH.write_text(json.dumps(recs), encoding="utf-8")
    return recs, dropped


if __name__ == "__main__":
    recs, dropped = build()
    print(f"{len(recs)} images from {', '.join(cfg.SOURCE_SPLITS)} "
          f"-> {cfg.INDEX_PATH.relative_to(cfg.ROOT)}\n")

    print("dropped")
    for name, key in (("other condition", "other"), ("no scored box", "empty"),
                      ("boxes filtered", "boxes")):
        print(f"  {name:<18}{dropped[key]:>7}")

    print("\nconditions")
    for c in lb.CONDITIONS:
        n = sum(r["condition"] == c for r in recs)
        flag = "" if n else "  EMPTY"
        print(f"  {c:<15}{n:>7}{flag}")

    print("\nclasses")
    counts = {}
    for r in recs:
        for name in r["labels"]:
            counts[name] = counts.get(name, 0) + 1
    for name in lb.CLASSES:
        flag = "" if counts.get(name) else "  UNSEEN"
        print(f"  {name:<15}{counts.get(name, 0):>7}{flag}")

    total = sum(counts.values())
    print(f"\n{total} boxes over {len(recs)} images, "
          f"{total / max(len(recs), 1):.1f} per image")

    # A wrong IMAGE_ROOT is invisible here and fails much later, in the loader
    sample = recs[::max(len(recs) // 20, 1)]
    missing = [r["file"] for r in sample if not (cfg.ROOT / r["file"]).exists()]
    mark = f"  {len(missing)} MISSING, first {missing[0]}" if missing else "  ok"
    print(f"spot check {len(sample)} image files{mark}")
