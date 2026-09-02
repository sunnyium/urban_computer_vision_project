"""
The detector: COCO-pretrained Faster R-CNN with a box predictor of our own.

Building:
    build_detector() | a v2 ResNet50-FPN with a head sized to the ten classes
    count_params() | trainable and total, for the run log

Loading:
    load_checkpoint() | weights back into an empty shell, for scoring and serving

Everything transfers from COCO except the head. The backbone and FPN keep their
pretrained weights and only the box predictor is replaced, because COCO already
holds cars, people, buses and traffic lights. That is what makes four arms
affordable on one machine.

The architecture, the image bounds and the frozen depth are the same for every
arm. Only augmentation varies, which is the whole basis of the comparison.
"""

from __future__ import annotations

import torch
from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_V2_Weights,
                                          fasterrcnn_resnet50_fpn_v2)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import config as cfg
import labels as lb

# What train.py writes and load_checkpoint reads back
CHECKPOINT_KEY = "model"


# 1. Building
def build_detector(num_classes: int = lb.NUM_CLASSES,
    pretrained: bool = True,) -> torch.nn.Module:
    """A Faster R-CNN whose head predicts `num_classes`, background included.
    Scored with EVAL_SCORE_THRESH rather than SCORE_THRESH: mAP needs the
    low-confidence tail, and a display threshold here would silently cap recall
    and cost every arm the same invisible slice of AP."""
    if cfg.ARCHITECTURE != "fasterrcnn_resnet50_fpn_v2":
        raise ValueError(f"unsupported architecture {cfg.ARCHITECTURE!r}")

    kwargs = dict(min_size=cfg.MIN_SIZE, max_size=cfg.MAX_SIZE,
                  box_score_thresh=cfg.EVAL_SCORE_THRESH)
    if pretrained:
        kwargs["weights"] = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        # only meaningful alongside pretrained weights, and torchvision warns
        # and ignores it otherwise, so it stays off the shell a checkpoint fills
        kwargs["trainable_backbone_layers"] = cfg.TRAINABLE_BACKBONE_LAYERS

    model = fasterrcnn_resnet50_fpn_v2(**kwargs)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    """(trainable, total) parameter counts, for the line the run log carries."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


# 2. Loading
def load_checkpoint(path, device=None) -> torch.nn.Module:
    """A detector holding `path`'s weights, in eval mode.
    Built unpretrained because every weight is about to be overwritten anyway,
    which spares a serving container the 170MB COCO download it would never
    use."""
    model = build_detector(pretrained=False)
    blob = torch.load(path, map_location=device or "cpu", weights_only=True)
    model.load_state_dict(blob[CHECKPOINT_KEY])
    model.eval()
    return model


if __name__ == "__main__":
    torch.manual_seed(cfg.SEED)
    model = build_detector()
    trainable, total = count_params(model)

    print(f"{cfg.ARCHITECTURE}  on {cfg.device()}\n")
    print(f"{'classes':<16}{lb.NUM_CLASSES} ({len(lb.CLASSES)} + background)")
    print(f"{'image bounds':<16}{cfg.MIN_SIZE}-{cfg.MAX_SIZE}")
    print(f"{'score thresh':<16}{cfg.EVAL_SCORE_THRESH} for scoring, "
          f"{cfg.SCORE_THRESH} for showing")
    print(f"{'parameters':<16}{total:,}")
    print(f"{'trainable':<16}{trainable:,}  ({trainable / total:.0%})")
    print(f"{'frozen':<16}{total - trainable:,}")

    print(f"\nbackbone, with TRAINABLE_BACKBONE_LAYERS={cfg.TRAINABLE_BACKBONE_LAYERS}")
    for name in ("layer1", "layer2", "layer3", "layer4"):
        body = model.backbone.body
        on = any(p.requires_grad for p in getattr(body, name).parameters())
        print(f"  {name:<10}{'training' if on else 'frozen'}")

    img = torch.rand(3, 64, 96)
    target = {"boxes": torch.tensor([[8.0, 10.0, 40.0, 50.0]]),
              "labels": torch.tensor([3])}

    model.train()
    losses = model([img], [target])
    print(f"\ntrain mode returns {len(losses)} losses")
    for name, v in losses.items():
        flag = "" if torch.isfinite(v.detach()) else "  NOT FINITE"
        print(f"  {name:<22}{float(v.detach()):>8.4f}{flag}")

    model.eval()
    with torch.inference_mode():
        out = model([img])[0]
    print(f"\neval mode returns {sorted(out)}")
    print(f"  {'boxes':<10}{tuple(out['boxes'].shape)}  {out['boxes'].dtype}")
    print(f"  {'labels':<10}{tuple(out['labels'].shape)}  {out['labels'].dtype}")
    print(f"  {'scores':<10}{tuple(out['scores'].shape)}  {out['scores'].dtype}")

    head = model.roi_heads.box_predictor.cls_score
    checks = {
        "head is sized to the taxonomy": head.out_features == lb.NUM_CLASSES,
        "head was reinitialised": head.out_features != 91,
        "every loss is finite": all(torch.isfinite(v) for v in losses.values()),
        "eval keeps the weak tail": bool(
            (out["scores"] < cfg.SCORE_THRESH).any()) or not len(out["scores"]),
        "labels stay inside the taxonomy": bool(
            (out["labels"] < lb.NUM_CLASSES).all()),
    }
    print()
    for name, ok in checks.items():
        print(f"  {name:<32}{'ok' if ok else 'FAILED'}")
