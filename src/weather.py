"""
Six synthetic corruptions, each at severity 1-5, applied to pixels alone.

Weather:
    rain() | bright motion-blurred streaks over a slightly hazed frame
    snow() | shorter, fatter, less directional flakes plus a white veil
    fog() | a non-uniform white veil, thickest where the haze field is
    night() | gamma darkening, a cool cast and the sensor noise that comes
        with it

Sensor:
    motion_blur() | a directional blur, what a long exposure does to a moving
        camera
    noise() | additive gaussian noise

Applying:
    corrupt() | name, severity and seed -> a corrupted copy
    CORRUPTIONS | name -> function, so callers loop rather than branch

No pixel moves coordinates, so boxes stay valid and these sit outside the
albumentations pipeline. No bbox bookkeeping to get wrong.

RGB uint8 HxWx3 in and out, from a Generator the caller seeds. Seeding per image
index rather than per run scores all four arms on pixel-identical test sets
without storing a copy.
"""

from __future__ import annotations

import cv2
import numpy as np

SEVERITIES = (1, 2, 3, 4, 5)


def _motion_kernel(length: int, angle: float) -> np.ndarray:
    """A normalised line kernel of `length` pixels, rotated `angle` degrees."""
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    m = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
    k = cv2.warpAffine(k, m, (length, length))
    return k / max(k.sum(), 1e-6)


def _streaks(h: int, w: int, density: float, length: int, angle: float,
    rng: np.random.Generator,) -> np.ndarray:
    """A 0-1 field of blurred streaks, for rain and snow to tint with.
    Blurred sparse mask rather than thousands of cv2.line calls, same look and
    faster."""
    mask = (rng.random((h, w)) < density).astype(np.float32)
    field = cv2.filter2D(mask, -1, _motion_kernel(length, angle))
    return np.clip(field / max(field.max(), 1e-6), 0.0, 1.0)


def _haze_field(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """A smooth 0-1 field, so a veil thickens and thins across the frame.
    Real fog has structure; a flat veil trains the detector on the wrong thing."""
    small = rng.random((max(h // 32, 2), max(w // 32, 2))).astype(np.float32)
    f = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    f = cv2.GaussianBlur(f, (0, 0), sigmaX=max(max(h, w) / 48.0, 1.0))
    return (f - f.min()) / (np.ptp(f) + 1e-6)


# 1. Weather
_RAIN = {
    "density": (0.002, 0.004, 0.007, 0.011, 0.016),
    "length": (9, 13, 17, 21, 25),
    "bright": (0.35, 0.45, 0.55, 0.65, 0.75),
}


def rain(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Rain streaks, plus the contrast loss of a wet atmosphere."""
    i = severity - 1
    h, w = img.shape[:2]
    angle = float(rng.uniform(-25.0, 25.0)) + 90.0
    s = _streaks(h, w, _RAIN["density"][i], _RAIN["length"][i], angle, rng)

    out = img.astype(np.float32)
    out = out * (1.0 - 0.05 * severity) + 12.0 * severity # atmospheric lift
    out += 255.0 * _RAIN["bright"][i] * s[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


_SNOW = {
    "density": (0.0015, 0.003, 0.005, 0.008, 0.012),
    "length": (5, 7, 9, 11, 13),
    "bright": (0.60, 0.70, 0.80, 0.90, 1.00),
    "veil": (0.04, 0.07, 0.11, 0.15, 0.20),
}


def snow(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Falling snow: fatter and rounder than rain, over a white veil.
    Flakes are near vertical but not aligned; one uniform direction is the
    clearest synthetic tell."""
    i = severity - 1
    h, w = img.shape[:2]
    angle = float(rng.uniform(-60.0, 60.0)) + 90.0
    s = _streaks(h, w, _SNOW["density"][i], _SNOW["length"][i], angle, rng)
    s = cv2.dilate(s, np.ones((3, 3), np.uint8))

    v = _SNOW["veil"][i]
    out = img.astype(np.float32) * (1.0 - v) + 255.0 * v
    out += 255.0 * _SNOW["bright"][i] * s[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


_FOG = (0.15, 0.28, 0.42, 0.56, 0.70)


def fog(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """A white veil whose thickness varies smoothly across the frame."""
    h, w = img.shape[:2]
    t = _FOG[severity - 1] * (0.55 + 0.45 * _haze_field(h, w, rng))[..., None]
    out = img.astype(np.float32) * (1.0 - t) + 255.0 * t
    return np.clip(out, 0, 255).astype(np.uint8)


_NIGHT = {
    "gamma": (1.6, 2.0, 2.5, 3.0, 3.6),
    "gain": (0.90, 0.80, 0.70, 0.60, 0.50),
    "sigma": (2.0, 4.0, 6.0, 9.0, 12.0),
}

# Street lighting is cooler than daylight, so red loses more than blue
_NIGHT_CAST = np.array([0.88, 0.95, 1.06], np.float32)


def night(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Low light: gamma darkening, a cool cast, and read noise.
    Darkening alone is invertible and a detector learns through it. The noise
    floor rising as signal falls is what kills small distant objects."""
    i = severity - 1
    x = img.astype(np.float32) / 255.0
    x = np.power(x, _NIGHT["gamma"][i]) * _NIGHT["gain"][i] * _NIGHT_CAST
    x = x * 255.0 + rng.normal(0.0, _NIGHT["sigma"][i], img.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


# 2. Sensor
_BLUR = (5, 9, 13, 17, 21)


def motion_blur(img: np.ndarray, severity: int,
    rng: np.random.Generator,) -> np.ndarray:
    """A directional blur at a random angle."""
    k = _motion_kernel(_BLUR[severity - 1], float(rng.uniform(0.0, 180.0)))
    return cv2.filter2D(img, -1, k)


_NOISE = (4.0, 8.0, 13.0, 19.0, 26.0)


def noise(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Additive gaussian noise."""
    x = img.astype(np.float32) + rng.normal(0.0, _NOISE[severity - 1], img.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


# 3. Applying
CORRUPTIONS = {
    "rain": rain,
    "snow": snow,
    "fog": fog,
    "night": night,
    "motion_blur": motion_blur,
    "noise": noise,
}

# The four scored as test columns. motion_blur and noise are trained against but
# never scored: BDD100K has no attribute for either, so no real subset to
# compare against
WEATHER_NAMES = ("rain", "snow", "fog", "night")


def corrupt(img: np.ndarray, name: str, severity: int, seed: int) -> np.ndarray:
    """A corrupted copy of `img`, identical for a given (name, severity, seed).
    Seed is an argument, not module state, so the eval loop derives it from the
    image index and gets the same pixels without writing a dataset to disk."""
    return CORRUPTIONS[name](img, severity, np.random.default_rng(seed))


if __name__ == "__main__":
    from pathlib import Path

    import config as cfg

    rng = np.random.default_rng(cfg.SEED)
    src = sorted(cfg.BDD_IMAGES.glob("val/*.jpg"))[:1]
    if src:
        img = cv2.cvtColor(cv2.imread(str(src[0])), cv2.COLOR_BGR2RGB)
        print(f"source {src[0].name}  {img.shape}")
    else:
        img = rng.integers(0, 255, (360, 640, 3), dtype=np.uint8)
        print(f"no images on disk, using noise {img.shape}")

    print(f"\n{'':<12}" + "".join(f"sev {s:<7}" for s in SEVERITIES))
    for name in CORRUPTIONS:
        cells = ""
        for s in SEVERITIES:
            out = corrupt(img, name, s, 0)
            cells += f"{float(np.abs(out.astype(np.int16) - img).mean()):<11.1f}"
        print(f"  {name:<10}{cells}")
    print("\nmean absolute pixel change, so every row should climb left to right")

    strip = [corrupt(img, n, cfg.EVAL_SEVERITY, 0) for n in CORRUPTIONS]
    grid = np.vstack([np.hstack([img] + strip[:3]), np.hstack([img] + strip[3:])])
    out = Path(cfg.RESULTS) / "weather_examples.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"wrote {out.relative_to(cfg.ROOT)}")
