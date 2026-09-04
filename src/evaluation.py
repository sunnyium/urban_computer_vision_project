"""
Stratified scoring. Nothing here ever reports one overall number.

Scoring:
    flatten() | torchmetrics output as plain floats, per class spelled out
    evaluate() | one loader -> a flat metric dict
    evaluate_all() | every scored condition, real and synthetic, kept apart

Retention:
    retention() | one condition's mAP over the clear control
    mean_retention() | the headline, averaged over the real conditions
    sensitivity_by_class() | classes ranked by how much AP the shift costs them

Reporting:
    results_table() | one row per condition, ready to print or log
    plot_results() | retention by condition beside sensitivity by class

A single overall mAP would average away the whole effect being studied, so every
number is broken down by condition, and within condition by class.

The headline is retention, not mAP. Absolute mAP under rain mixes how good a
model is with how well it holds up, and only the second is the question. A model
scoring 0.30 on rain from a 0.60 clear baseline is less robust than one scoring
0.28 from 0.40, and ranking on the raw number gets that backwards.

Real and synthetic conditions sit side by side and are never pooled. Real is the
honest test but its buckets are small; synthetic holds the scene fixed and
varies only the weather, but measures robustness to this repo's own simulation,
which two of the four arms have trained against.
"""

from __future__ import annotations

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import config as cfg
import dataset as ds
import labels as lb

# The columns worth printing, in the order a reader wants them
KEY_METRICS = ("map_50_95", "map_50", "map_75", "map_small", "mar_100")


def _clean(v) -> float | None:
    """A metric value, or None where torchmetrics had nothing to score.
    It returns -1 for a metric with no eligible objects, such as map_small on a
    set holding no small boxes. That is a sentinel, not a zero, and averaging or
    dividing with it would quietly drag every summary down."""
    return None if float(v) < 0.0 else float(v)


# 1. Scoring
def flatten(computed: dict) -> dict:
    """torchmetrics output as plain floats, with the per-class arrays spelled out.
    map becomes map_50_95, because "map" sitting next to "map_50" reads as a
    typo rather than as the headline it is."""
    out = {}
    classes = computed.get("classes", torch.tensor([])).flatten().tolist()
    for k, v in computed.items():
        if k == "classes":
            continue
        if k.endswith("_per_class"):
            stem = "ap" if k.startswith("map") else "ar"
            for c, x in zip(classes, v.flatten().tolist()):
                out[f"{stem}/{lb.class_name(int(c))}"] = _clean(x)
        else:
            out["map_50_95" if k == "map" else k] = _clean(v)
    return out


@torch.inference_mode()
def evaluate(model, loader, device=None) -> dict:
    """Run `model` over `loader` and return its metrics, per class included."""
    device = device or cfg.device()
    model.eval().to(device)

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox",
                                  class_metrics=True,
                                  backend="faster_coco_eval")
    for images, targets in loader:
        preds = model([i.to(device) for i in images])
        metric.update([{k: v.cpu() for k, v in p.items()} for p in preds],
                      [dict(t) for t in targets])
    return flatten(metric.compute())


def evaluate_all(model, buckets: dict, device=None) -> dict:
    """Every scored condition, keyed clear, real/<c> and synth/<c>.
    The synthetic sets are the clear test images with one corruption applied, so
    the scenes and the objects are held fixed and only the weather moves."""
    clear = buckets["test/clear"]
    out = {"clear": evaluate(model, ds.make_loader(clear), device)}

    for c in cfg.REAL_CONDITIONS:
        recs = buckets.get(f"test/{c}") or []
        if recs:
            out[f"real/{c}"] = evaluate(model, ds.make_loader(recs), device)

    for c in cfg.SYNTHETIC_CONDITIONS:
        out[f"synth/{c}"] = evaluate(model, ds.make_loader(clear, corrupt=c),
                                     device)
    return out


# 2. Retention
def retention(results: dict, key: str, metric: str = "map_50_95") -> float | None:
    """`key`'s score over the clear control, or None when either side is absent.
    A clear baseline of zero leaves retention undefined rather than infinite: a
    model that detects nothing has no performance left to retain."""
    got = results.get(key, {}).get(metric)
    base = results.get("clear", {}).get(metric)
    if got is None or not base:
        return None
    return got / base


def mean_retention(results: dict, kind: str = "real") -> float | None:
    """The headline number, averaged over the conditions of one kind.
    Defaults to real, because the synthetic columns only measure robustness to
    this repo's own simulation."""
    conditions = (cfg.REAL_CONDITIONS if kind == "real"
                  else cfg.SYNTHETIC_CONDITIONS)
    got = [retention(results, f"{kind}/{c}") for c in conditions]
    got = [g for g in got if g is not None]
    return sum(got) / len(got) if got else None


def sensitivity_by_class(results: dict, kind: str = "real") -> list[tuple]:
    """Classes ranked by the share of their clear AP the shift costs them.
    Ranked on the drop rather than on what is left, because a class the model
    never found in clear weather has nothing to lose and is not a finding."""
    keys = [k for k in results if k.startswith(f"{kind}/")]

    out = []
    for name in lb.CLASSES:
        base = results.get("clear", {}).get(f"ap/{name}")
        if not base:
            continue
        vals = [results[k][f"ap/{name}"] for k in keys
                if results[k].get(f"ap/{name}") is not None]
        if vals:
            out.append((name, 1.0 - (sum(vals) / len(vals)) / base))
    return sorted(out, key=lambda t: -t[1])


# 3. Reporting
def results_table(results: dict) -> list[dict]:
    """One row per scored condition, in the order evaluate_all produced them."""
    return [{"condition": key,
             **{m: results[key].get(m) for m in KEY_METRICS},
             "retention": retention(results, key)}
            for key in results]


def plot_results(results: dict, path=None, title: str = "") -> str:
    """Retention by condition beside sensitivity by class, written as one png.
    matplotlib is imported here because scoring a run does not need it and the
    serving image never installs it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = path or cfg.RESULTS / "retention.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    keys = [k for k in results if k != "clear"]
    vals = [retention(results, k) or 0.0 for k in keys]
    colours = ["#4C72B0" if k.startswith("real/") else "#B0B7C3" for k in keys]
    ax1.bar(range(len(keys)), vals, color=colours)
    ax1.axhline(1.0, color="#333333", linewidth=0.8, linestyle="--")
    ax1.set_xticks(range(len(keys)))
    ax1.set_xticklabels([k.replace("/", "\n") for k in keys], fontsize=8)
    ax1.set_ylabel("retention vs clear")
    ax1.set_title("retention by condition, real in blue")

    drops = sensitivity_by_class(results)
    ax2.barh([n for n, _ in drops][::-1], [d for _, d in drops][::-1],
             color="#C44E52")
    ax2.set_xlabel("share of clear AP lost")
    ax2.set_title("sensitivity by class")

    fig.suptitle(title or "robustness under environmental shift")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


if __name__ == "__main__":
    # Perfect predictions must score 1.0. Scoring the ground truth against
    # itself pins the metric wiring without needing a trained checkpoint
    recs = ds.load_splits()["test/clear"]
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox",
                                  class_metrics=True, backend="faster_coco_eval")
    for _, targets in ds.make_loader(recs):
        metric.update([{"boxes": t["boxes"], "labels": t["labels"],
                        "scores": torch.ones(len(t["labels"]))} for t in targets],
                      [dict(t) for t in targets])
    oracle = flatten(metric.compute())
    scored = sum(1 for k in oracle if k.startswith("ap/"))
    print(f"an oracle over {len(recs)} clear test images")
    print(f"  {'map_50_95':<14}{oracle['map_50_95']:.3f}")
    print(f"  {'classes':<14}{scored} scored")

    # hand-made results, so the arithmetic has an answer to be checked against
    fake = {
        "clear": {"map_50_95": 0.50, "ap/car": 0.60, "ap/pedestrian": 0.40},
        "real/rain": {"map_50_95": 0.25, "ap/car": 0.45, "ap/pedestrian": 0.10},
        "real/snow": {"map_50_95": 0.35, "ap/car": 0.51, "ap/pedestrian": 0.30},
        "real/night": {"map_50_95": 0.20, "ap/car": 0.36, "ap/pedestrian": 0.10},
        "synth/fog": {"map_50_95": 0.10, "ap/car": 0.30, "ap/pedestrian": 0.05},
    }
    print("\nworked example, clear map_50_95 = 0.50")
    for row in results_table(fake):
        r = row["retention"]
        cell = "     -" if r is None else f"{r:>6.2f}"
        print(f"  {row['condition']:<14}{row['map_50_95']:>6.2f}   {cell}")

    print("\nsensitivity, mean real AP against clear AP")
    for name, drop in sensitivity_by_class(fake):
        print(f"  {name:<14}{drop:>7.1%}")

    zero = {"clear": {"map_50_95": 0.0}, "x": {"map_50_95": 0.1}}
    checks = {
        "oracle scores a perfect 1.0": abs(oracle["map_50_95"] - 1.0) < 1e-6,
        "retention divides by clear": abs(retention(fake, "real/rain") - 0.5) < 1e-9,
        "clear retains 1.0": abs(retention(fake, "clear") - 1.0) < 1e-9,
        "mean covers real only": abs(mean_retention(fake)
                                     - (0.5 + 0.7 + 0.4) / 3) < 1e-9,
        "pedestrian is most sensitive": sensitivity_by_class(fake)[0][0]
                                        == "pedestrian",
        "a missing condition is None": retention(fake, "real/fog") is None,
        "a zero baseline is None": retention(zero, "x") is None,
    }
    print()
    for name, ok in checks.items():
        print(f"  {name:<30}{'ok' if ok else 'FAILED'}")

    print(f"\nwrote {plot_results(fake, title='worked example, not real results')}")
