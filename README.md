# Robust Urban Object Detection Under Environmental Distribution Shift

Detecting ten object classes (vehicles, pedestrians, traffic lights and signs)
in urban street scenes, and measuring how much of that performance survives when
the weather changes. Four detectors train on clear-weather images alone under
four different augmentation strategies, then get evaluated on rain, snow, fog
and night.

The question is not which model scores highest. It is which training strategy
degrades least when the world stops looking like the training set.

Nothing here is finished. The environment is set up and the modules are being
written one at a time, so this README claims no results yet.

## Why this project is structured the way it is

A detector that scores well on a benchmark and then fails in rain is the normal
outcome, not an unusual one. Benchmark splits are drawn from the same
distribution as the training data, so they cannot reveal it.

Making that failure visible and measurable takes a specific experimental setup:
train on one condition, test on others, hold everything else fixed. Building
that setup correctly is the substance of the work, and it is mostly data
engineering: a clear-only training pool, splits that are checked for leakage
rather than assumed clean, and a metric that separates "good" from "robust". The
detector itself is a fine-tuned torchvision model and is the least interesting
part.

## Planned pipeline

```
BDD100K  (100K driving images, weather + time-of-day labeled)
   │  prepare_data.py    JSON -> flat index, tag each image's condition
   ▼
data/processed/index_*.json
   │  splits.py          train/val = clear ONLY; test stratified by condition
   ▼
data/splits/splits.json
   │  run_experiment.py  train one arm, evaluate on every condition
   ▼
checkpoints/<arm>/best.pth  +  mlruns/  +  data/results/*.png
```

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements.txt
```

torch and torchvision install first, from their own index, so you get the CPU or
CUDA build that matches your machine. requirements.txt leaves them unpinned to a
build for the same reason.

```bash
python -c "import torch, torchvision, cv2, albumentations, torchmetrics, mlflow"
```

### Windows on ARM64

On an ARM64 machine the obvious `python -m venv .venv` picks up an ARM64
interpreter, and two dependencies have no win_arm64 wheel:
opencv-python-headless and faster-coco-eval. pip falls back to building them
from source and fails for lack of a C compiler.

The error names the wrong package. It surfaces as `metadata-generation-failed`
on numpy, because numpy is being pulled in as one of opencv's *build*
dependencies, which sends you off debugging a package that was never the
problem.

Build the venv from an x64 interpreter instead. It runs under emulation and gets
prebuilt wheels for everything:

```bash
"$LOCALAPPDATA/Microsoft/WindowsApps/PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0/python.exe" -m venv .venv
python -c "import platform; print(platform.machine())"   # must print AMD64
```

Check that second line rather than trusting the launcher: `py -0p` labels this
interpreter `3.11-arm64` and the label is wrong.

## Build status

Written module by module in dependency order, so every step ends at something
runnable rather than at a file that only compiles.

Done:

    scaffold, .gitignore, requirements.txt, venv | imports resolve, pytest runs

Remaining:

    labels.py | the 10 classes and the (weather, timeofday) -> condition rule
    config.py | paths, hyperparameters, the four arm definitions
    weather.py | six corruptions at severity 1-5, pixel-only
    make_synthetic.py | fake dataset, so the pipeline runs without the download
    prepare_data.py | BDD100K JSON -> flat, condition-tagged index
    splits.py | the split protocol and its leakage assertions
    augment.py | the four augmentation strategies
    dataset.py | torch Dataset + loaders
    models.py | COCO-pretrained Faster R-CNN with a fresh box predictor
    evaluation.py | stratified mAP, retention, per-class sensitivity, figures
    train.py | fine-tune one arm, select on clear-weather val
    run_experiment.py | one arm end to end, logged to MLflow
    api.py | FastAPI service

prepare_data.py is the one module with no gate until BDD100K is on disk. It gets
written blind and validated later, so expect to come back to it once the layout
of the actual download is known.

## Next

labels.py. Small, but it carries the one real judgment call in the data
pipeline: night takes precedence over weather, so a rainy night image is
bucketed night, not rain. Low light is the dominant corruption in that image,
and letting it into the rain bucket would mean a drop in the rain column could
be caused by either rain or darkness. That ambiguity is the thing the project
exists to remove.
