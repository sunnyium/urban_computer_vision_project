"""
The ten detection classes and the condition taxonomy they are scored under.

Classes:
    normalise_category() | raw BDD100K label string -> canonical name, or None
        for a category outside the ten
    class_id() | canonical name -> 1-based id
    class_name() | id -> canonical name, 0 being background

Conditions:
    derive_condition() | (weather, time of day) -> one of the five buckets, or
        "other" for an image the taxonomy drops

Night > Weather meaning a rainy night image is considered night. Anything the 
rule cannot place cleanly returns "other".
"""

from __future__ import annotations

import re

CLASSES = (
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
)

BACKGROUND = "__background__"

# torchvision counts background as a class, so real ids start at 1.
NUM_CLASSES = len(CLASSES) + 1

# BDD100K label aliases
_ALIASES = {
    "person": "pedestrian",
    "motor": "motorcycle",
    "bike": "bicycle",
}

_INDEX = {name: i for i, name in enumerate(CLASSES, start=1)}


def _norm(s: str) -> str:
    """Lowercased, single-spaced form of an attribute or category string."""
    return re.sub(r"[\s_\-]+", " ", str(s).strip().lower())


# 1. Classes
def normalise_category(raw: str) -> str | None:
    """Canonical class name for a raw label, or None if it is not one of the ten.
    BDD100K carries categories this study does not score, so we filter to None"""
    s = _norm(raw)
    s = _ALIASES.get(s, s)
    return s if s in _INDEX else None


def class_id(name: str) -> int:
    """1-based model id for a canonical class name."""
    return _INDEX[name]


def class_name(idx: int) -> str:
    """Canonical name for a model id, 0 being background."""
    return BACKGROUND if idx == 0 else CLASSES[idx - 1]


# Every value the real attributes can hold
WEATHER_VOCAB = (
    "clear",
    "partly cloudy",
    "overcast",
    "rainy",
    "snowy",
    "foggy",
    "undefined",
)

TIMEOFDAY_VOCAB = ("daytime", "night", "dawn/dusk", "undefined")

CONDITIONS = ("clear", "rain", "snow", "fog", "night")
ADVERSE = CONDITIONS[1:]
OTHER = "other"

_WEATHER_TO_CONDITION = {
    "clear": "clear",
    "rainy": "rain",
    "snowy": "snow",
    "foggy": "fog",
}


# 2. Conditions
def derive_condition(weather: str, timeofday: str) -> str:
    """Condition bucket for one image, or "other" if the taxonomy drops it.
    Time of day is read first so Night > Weather"""
    tod = _norm(timeofday)
    if tod == "night":
        return "night"
    if tod != "daytime":
        return OTHER
    return _WEATHER_TO_CONDITION.get(_norm(weather), OTHER)


if __name__ == "__main__":
    print(f"{NUM_CLASSES} model classes ({len(CLASSES)} + background)\n")
    for i in range(NUM_CLASSES):
        print(f"  {i:>2}  {class_name(i)}")

    print("\nalias normalisation")
    for raw in ("person", "Traffic_Light", "  BIKE ", "motor", "drivable area"):
        print(f"  {raw!r:>18} -> {normalise_category(raw)}")

    print("\ncondition table (rows weather, columns time of day)")
    print(f"  {'':<14}" + "".join(f"{t:<11}" for t in TIMEOFDAY_VOCAB))
    for w in WEATHER_VOCAB:
        cells = "".join(f"{derive_condition(w, t):<11}" for t in TIMEOFDAY_VOCAB)
        print(f"  {w:<14}{cells}")

    kept = sum(
        derive_condition(w, t) != OTHER
        for w in WEATHER_VOCAB
        for t in TIMEOFDAY_VOCAB
    )
    print(f"\n{kept} of {len(WEATHER_VOCAB) * len(TIMEOFDAY_VOCAB)} attribute "
          "pairs land in a scored bucket")
