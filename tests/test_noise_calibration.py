"""Tests for calibrate_noise_vs_occupancy (sigma(N) from real plateaus)."""

from __future__ import annotations

import numpy as np
import pytest

from rionsis.levels.noise_calibration import calibrate_noise_vs_occupancy


def _synthetic_plateaus(rng, sigma0, k, states=(0, 1, 2, 3, 4), n_per_state=150, length=30, a1=1.0):
    lengths_per_file, states_per_file, file_ids = [], [], []
    rows = []
    file_id = 0
    for n in states:
        sigma = np.sqrt(sigma0**2 + (k * n) ** 2)
        for _ in range(n_per_state):
            segment = n * a1 + sigma * rng.standard_normal(length)
            rows.append(segment)
            lengths_per_file.append(np.array([length]))
            states_per_file.append(np.array([n]))
            file_ids.append(file_id)
            file_id += 1
    y_files_e = np.stack(rows)
    return y_files_e, lengths_per_file, states_per_file, np.asarray(file_ids)


def test_calibrate_noise_vs_occupancy_recovers_known_sigma0_and_k():
    rng = np.random.default_rng(0)
    true_sigma0, true_k = 0.2, 0.08
    y_files_e, lengths, states, file_ids = _synthetic_plateaus(rng, true_sigma0, true_k)
    calibration = calibrate_noise_vs_occupancy(y_files_e, lengths, states, file_ids, a1=1.0, min_length=5, max_n=4)
    assert calibration.sigma0 == pytest.approx(true_sigma0, abs=0.03)
    assert calibration.k == pytest.approx(true_k, abs=0.03)
    np.testing.assert_allclose(calibration.sigma_of(0.0), calibration.sigma0)


def test_calibrate_noise_vs_occupancy_raises_with_only_ground_state():
    rng = np.random.default_rng(1)
    y_files_e, lengths, states, file_ids = _synthetic_plateaus(rng, 0.2, 0.08, states=(0,), n_per_state=50)
    with pytest.raises(RuntimeError):
        calibrate_noise_vs_occupancy(y_files_e, lengths, states, file_ids, a1=1.0)


def test_calibrate_noise_vs_occupancy_two_support_point_fallback():
    rng = np.random.default_rng(2)
    y_files_e, lengths, states, file_ids = _synthetic_plateaus(rng, 0.2, 0.1, states=(0, 3), n_per_state=100)
    calibration = calibrate_noise_vs_occupancy(y_files_e, lengths, states, file_ids, a1=1.0, min_length=5, max_n=4)
    assert len(calibration.states) == 2
    assert np.all(np.isnan(calibration.fit_covariance))
