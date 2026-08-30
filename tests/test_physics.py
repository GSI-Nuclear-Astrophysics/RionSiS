"""Unit tests for rionsis.physics: shared decay-law and noise-model relations."""

from __future__ import annotations

import numpy as np
import pytest

from rionsis.physics.decay_law import halflife_to_rate, rate_to_halflife
from rionsis.physics.noise_model import sigma_of_occupancy


def test_halflife_rate_roundtrip():
    half_life = 0.037  # seconds
    rate = halflife_to_rate(half_life)
    assert rate_to_halflife(rate) == pytest.approx(half_life, rel=1e-12)


def test_halflife_to_rate_known_value():
    # T_1/2 = 1 s -> rate = ln(2) 1/s
    assert halflife_to_rate(1.0) == pytest.approx(np.log(2.0))


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_halflife_to_rate_rejects_nonpositive(bad):
    with pytest.raises(ValueError):
        halflife_to_rate(bad)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_rate_to_halflife_rejects_nonpositive(bad):
    with pytest.raises(ValueError):
        rate_to_halflife(bad)


def test_halflife_to_rate_is_array_safe():
    half_lives = np.array([1.0, 2.0, 0.5])
    rates = halflife_to_rate(half_lives)
    np.testing.assert_allclose(rates, np.log(2.0) / half_lives)


def test_sigma_of_occupancy_at_zero_is_floor():
    assert sigma_of_occupancy(0.0, sigma0=0.2, k=0.1) == pytest.approx(0.2)


def test_sigma_of_occupancy_increases_with_occupancy():
    n = np.array([0.0, 1.0, 2.0, 5.0])
    sigma = sigma_of_occupancy(n, sigma0=0.2, k=0.1)
    assert np.all(np.diff(sigma) > 0)


def test_sigma_of_occupancy_matches_closed_form():
    sigma0, k, n = 0.3, 0.15, 4.0
    expected = np.sqrt(sigma0**2 + (k * n) ** 2)
    assert sigma_of_occupancy(n, sigma0, k) == pytest.approx(expected)


def test_sigma_of_occupancy_clips_negative_occupancy_to_zero():
    assert sigma_of_occupancy(-5.0, sigma0=0.2, k=0.1) == pytest.approx(sigma_of_occupancy(0.0, 0.2, 0.1))
