"""Tests for band integration and ground-state noise normalization."""

from __future__ import annotations

import numpy as np
import pytest

from rionsis.levels.bands import integrate_bands, normalize_ground_noise


def test_integrate_bands_subtracts_background_mean():
    n_files, n_freq, n_time = 3, 10, 5
    power = np.zeros((n_files, n_freq, n_time))
    frequency = np.arange(n_freq, dtype=float) - 5  # -5..4
    # signal band: bins with frequency in (0, 3) -> indices where freq in {1,2}
    # background band: frequency in (-5, -2) -> indices where freq in {-4,-3}
    power[:, (frequency > 0) & (frequency < 3), :] = 10.0
    power[:, (frequency > -5) & (frequency < -2), :] = 2.0
    result = integrate_bands(power, frequency, band_signal=(0, 3), background_bands=[(-5, -2)])
    np.testing.assert_allclose(result.y_files, 8.0)


def test_integrate_bands_rejects_overlapping_bands():
    frequency = np.linspace(-10, 10, 21)
    power = np.zeros((1, len(frequency), 2))
    with pytest.raises(ValueError):
        integrate_bands(power, frequency, band_signal=(-5, 5), background_bands=[(-2, 2)])


def test_integrate_bands_rejects_empty_band():
    frequency = np.linspace(-10, 10, 21)
    power = np.zeros((1, len(frequency), 2))
    with pytest.raises(ValueError):
        integrate_bands(power, frequency, band_signal=(100, 200), background_bands=[(-2, 2)])


def test_normalize_ground_noise_rescales_to_ref_sigma():
    rng = np.random.default_rng(0)
    n_files, n_time = 40, 30
    raw_sigma = 3.0
    y_files = rng.normal(0.0, raw_sigma, size=(n_files, n_time))
    result = normalize_ground_noise(y_files, ref_sigma=0.66, noise_bins=15)
    rescaled_end = result.y_files[result.ground_noise_idx, -15:]
    # The stable subset's median terminal SD should now sit at ref_sigma
    # (it is exactly ref_sigma for the files whose SD equals the pre-scale
    # median, by construction of the rescaling factor).
    assert np.median(rescaled_end.std(axis=1)) == pytest.approx(0.66, rel=0.15)
    assert result.sigma == pytest.approx(0.66)


def test_normalize_ground_noise_requires_at_least_three_files():
    y_files = np.zeros((2, 10))
    with pytest.raises(RuntimeError):
        normalize_ground_noise(y_files)
