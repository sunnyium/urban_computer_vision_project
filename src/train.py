"""
Fine-tune one arm and keep the checkpoint that scored best on clear val.

Setup:
    set_seed() | every generator a run touches
    build_optimizer() | SGD and the step schedule, shared by all four arms

Training:
    train_one_epoch() | one pass, with warmup and gradient clipping
    train() | the loop, the selection rule and the checkpoint it writes

Runnable on its own, one arm at a time:

    python src/train.py combined

Nothing here branches on the arm. The augmenter is built from config.ARMS and
handed to the dataset, every other hyperparameter is a shared constant, and that
is the whole basis on which the four runs are comparable.

The best checkpoint is chosen on CLEAR validation mAP alone. Selecting on
adverse conditions would leak the test distribution into model selection: the
model would have been tuned for the thing it is about to be tested on, and the
retention numbers would be measuring that rather than robustness.
"""

from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import torch

import augment as ag
import config as cfg
import dataset as ds
import evaluation as ev
import models as md


# 1. Setup
def set_seed(seed: int = cfg.SEED) -> None:
    """Seed python, numpy and torch together, so one arm reruns identically.
    albumentations is seeded separately, per Compose, inside augment.py."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model) -> tuple:
    """SGD over the trainable parameters, and the step schedule beside it.
    Only unfrozen parameters go in. The frozen stem produces no gradient, and
    handing it to SGD would still let weight decay shrink it every step."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=cfg.LR, momentum=cfg.MOMENTUM,
                          weight_decay=cfg.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.LR_STEP,
                                            gamma=cfg.LR_GAMMA)
    return opt, sched


# 2. Training
def train_one_epoch(model, loader, opt, device, epoch: int) -> dict:
    """One pass over the training set, returning the mean of each loss.
    Warmup runs in epoch 0 only. A non-finite loss raises rather than carrying
    on, because the weights are already ruined by then and every later epoch
    would just be burning hours to produce a checkpoint nobody can use."""
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]

    warmup = None
    if epoch == 0 and cfg.WARMUP_ITERS > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.001, total_iters=cfg.WARMUP_ITERS)

    totals, steps = {}, 0
    for images, targets in loader:
        images = [i.to(device) for i in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        losses = model(images, targets)
        loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"loss went non-finite at epoch {epoch}, step {steps}")

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.GRAD_CLIP)
        opt.step()
        if warmup is not None and steps < cfg.WARMUP_ITERS:
            warmup.step()

        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + float(v.detach())
        totals["loss_total"] = totals.get("loss_total", 0.0) + float(loss.detach())
        steps += 1

    return {k: v / max(steps, 1) for k, v in totals.items()}


def train(arm: str, device=None) -> dict:
    """Fine-tune `arm`, returning its history and where the checkpoint landed.
    Epoch 0 always writes, so a run that is cut short still leaves something
    loadable rather than an empty directory."""
    if arm not in cfg.ARMS:
        raise KeyError(f"unknown arm {arm!r}, expected one of {tuple(cfg.ARMS)}")

    set_seed()
    device = device or cfg.device()

    buckets = ds.load_splits()
    train_recs = buckets["train"]
    if cfg.TRAIN_SUBSET:
        train_recs = train_recs[:cfg.TRAIN_SUBSET]

    loader = ds.make_loader(train_recs, augment=ag.build_augmenter(arm),
                            shuffle=True)
    val_loader = ds.make_loader(buckets["val"])

    model = md.build_detector().to(device)
    opt, sched = build_optimizer(model)

    out = cfg.checkpoint_dir(arm)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "best.pth"

    print(f"{arm}: {len(train_recs)} train / {len(buckets['val'])} val "
          f"on {device}, {cfg.EPOCHS} epochs")

    best, history = -1.0, []
    for epoch in range(cfg.EPOCHS):
        t0 = time.time()
        losses = train_one_epoch(model, loader, opt, device, epoch)
        sched.step()

        val_map = ev.evaluate(model, val_loader, device).get("map_50_95") or 0.0
        saved = val_map > best
        if saved:
            best = val_map
            torch.save({md.CHECKPOINT_KEY: model.state_dict(), "arm": arm,
                        "epoch": epoch, "val_map_50_95": val_map}, path)

        history.append({"epoch": epoch, "seconds": time.time() - t0,
                        "val_map_50_95": val_map, "saved": saved, **losses})
        print(f"  epoch {epoch:<3}loss {losses['loss_total']:.4f}  "
              f"val mAP {val_map:.4f}  {time.time() - t0:.0f}s"
              f"{'  saved' if saved else ''}")

    return {"arm": arm, "best_val_map_50_95": best,
            "checkpoint": str(path), "history": history}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("arm", choices=tuple(cfg.ARMS))
    args = ap.parse_args()

    result = train(args.arm)

    print(f"\n{'epoch':<8}{'loss':<10}{'val mAP':<11}{'seconds':<10}kept")
    for row in result["history"]:
        mark = "yes" if row["saved"] else ""
        print(f"  {row['epoch']:<6}{row['loss_total']:<10.4f}"
              f"{row['val_map_50_95']:<11.4f}{row['seconds']:<10.0f}{mark}")

    print(f"\nbest clear-val mAP {result['best_val_map_50_95']:.4f}")
    print(f"checkpoint {result['checkpoint']}")

    blob = torch.load(result["checkpoint"], map_location="cpu", weights_only=True)
    checks = {
        "a checkpoint was written": bool(result["history"]),
        "it records its arm": blob.get("arm") == args.arm,
        "it loads back": md.load_checkpoint(result["checkpoint"]) is not None,
        "selection used clear val": all("val_map_50_95" in r
                                        for r in result["history"]),
        "every loss stayed finite": all(np.isfinite(r["loss_total"])
                                        for r in result["history"]),
    }
    print()
    for name, ok in checks.items():
        print(f"  {name:<28}{'ok' if ok else 'FAILED'}")

    # beside the checkpoint it describes, so checkpoint_dir carries the suffix
    log = cfg.checkpoint_dir(args.arm) / "history.json"
    log.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote {log.relative_to(cfg.ROOT)}")
