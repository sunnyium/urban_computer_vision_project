# Robust Urban Object Detection Under Environmental Distribution Shift

Detecting ten object classes (vehicles, pedestrians, traffic lights and signs)
in urban street scenes, and measuring how much of that performance survives when
the weather changes. Four detectors are trained on clear-weather images under
four different augmentation strategies, then get evaluated on rain, snow, fog
and night.

The goal of this project is determining which training strategy
degrades least when the images stops looking like the training set.

[CURRENTLY IN PROGRESS]

## Why this project is structured the way it is

A detector that scores well on a benchmark and then fails in rain is normal. Benchmark splits are drawn from the same
distribution as the training data, so they cannot reveal this.

Setup:

Train on one condition, test on others, hold everything else fixed. 
There is a data engineering focus with a clear-only training pool and splits that are checked for leakage.
The detector is a fine-tuned torchvision model.

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

torch and torchvision install first so you get the CPU or
CUDA that matches your machine. 

```bash
python -c "import torch, torchvision, cv2, albumentations, torchmetrics, mlflow"
```

### Windows on ARM64 (mine)

On an ARM64 machine the obvious `python -m venv .venv` picks up an ARM64
interpreter, and two dependencies have no win_arm64 wheel.
pip falls back to building them
from source and fails for lack of a C compiler.

Build the venv from an x64 interpreter instead.

```bash
"$LOCALAPPDATA/Microsoft/WindowsApps/PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0/python.exe" -m venv .venv
python -c "import platform; print(platform.machine())"
```

## Build status

Written module by module in dependency order

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
