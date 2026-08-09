"""
Tests for the six corruptions, mostly about the guarantees callers rely on.

Contract:
    test_shape_and_dtype_survive | RGB uint8 HxWx3 in, the same out
    test_input_is_not_mutated | the caller's array is left alone
    test_output_is_in_range | nothing wraps around or goes NaN
    test_something_actually_changed | a corruption that is a no-op is a bug

Determinism:
    test_same_seed_same_pixels | the property the corrupted test sets rest on
    test_different_seed_differs | and the seed is actually being used
    test_global_numpy_state_is_ignored | no hidden dependence on np.random

Severity:
    test_severity_increases_change | monotone in mean absolute change
    test_night_darkens, test_fog_brightens | in the direction their name claims
    test_night_is_cooler_than_it_is_warm | the cast leans blue, not red

Sizes:
    test_odd_and_tiny_shapes | kernels and the haze field survive a 7x5 image

Registry:
    test_weather_names_are_the_scored_four | the scored names exist as functions

The determinism tests matter most. If a corruption depends on global numpy state,
two arms scored on "the same" rain set get different pixels and the retention gap
between them means nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

import weather as wx


def make_img(h: int = 48, w: int = 64, seed: int = 0) -> np.ndarray:
    """A random RGB frame, not flat so blurs have something to do."""
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


ALL = list(wx.CORRUPTIONS)


# 1. Contract
@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("sev", wx.SEVERITIES)
def test_shape_and_dtype_survive(name, sev):
    img = make_img()
    out = wx.corrupt(img, name, sev, 0)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


@pytest.mark.parametrize("name", ALL)
def test_input_is_not_mutated(name):
    img = make_img()
    before = img.copy()
    wx.corrupt(img, name, 5, 0)
    assert np.array_equal(img, before)


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("sev", wx.SEVERITIES)
def test_output_is_in_range(name, sev):
    out = wx.corrupt(make_img(), name, sev, 3)
    assert out.min() >= 0 and out.max() <= 255
    assert np.isfinite(out.astype(np.float32)).all()


@pytest.mark.parametrize("name", ALL)
def test_something_actually_changed(name):
    img = make_img()
    assert not np.array_equal(wx.corrupt(img, name, 3, 0), img)


# 2. Determinism
@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("sev", wx.SEVERITIES)
def test_same_seed_same_pixels(name, sev):
    img = make_img()
    a = wx.corrupt(img, name, sev, 12345)
    b = wx.corrupt(img, name, sev, 12345)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("name", ALL)
def test_different_seed_differs(name):
    img = make_img()
    a = wx.corrupt(img, name, 4, 1)
    b = wx.corrupt(img, name, 4, 2)
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("name", ALL)
def test_global_numpy_state_is_ignored(name):
    """Seeding np.random must not change the result, or determinism is a lie."""
    img = make_img()
    np.random.seed(0)
    a = wx.corrupt(img, name, 3, 7)
    np.random.seed(999)
    b = wx.corrupt(img, name, 3, 7)
    assert np.array_equal(a, b)


# 3. Severity
@pytest.mark.parametrize("name", ALL)
def test_severity_increases_change(name):
    img = make_img(96, 128)
    d = [float(np.abs(wx.corrupt(img, name, s, 5).astype(np.int16)
                      - img.astype(np.int16)).mean()) for s in wx.SEVERITIES]
    assert d == sorted(d), f"{name} not monotone: {d}"


@pytest.mark.parametrize("sev", wx.SEVERITIES)
def test_night_darkens(sev):
    img = make_img(96, 128)
    assert wx.corrupt(img, "night", sev, 0).mean() < img.mean()


@pytest.mark.parametrize("sev", wx.SEVERITIES)
def test_fog_brightens(sev):
    img = make_img(96, 128)
    assert wx.corrupt(img, "fog", sev, 0).mean() > img.mean()


def test_night_is_cooler_than_it_is_warm():
    """Blue survives the darkening better than red, per _NIGHT_CAST."""
    img = np.full((64, 64, 3), 200, np.uint8)
    out = wx.corrupt(img, "night", 3, 0).astype(np.float32)
    assert out[..., 2].mean() > out[..., 0].mean()


# 4. Sizes
@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("shape", [(7, 5), (1, 1), (33, 17), (64, 3)])
def test_odd_and_tiny_shapes(name, shape):
    img = make_img(*shape)
    out = wx.corrupt(img, name, 5, 0)
    assert out.shape == img.shape and out.dtype == np.uint8


# 5. Registry
def test_weather_names_are_the_scored_four():
    assert set(wx.WEATHER_NAMES) <= set(wx.CORRUPTIONS)
    assert set(wx.WEATHER_NAMES) == {"rain", "snow", "fog", "night"}
