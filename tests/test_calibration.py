"""Tests for the independent A1 (single-ion power) mixture calibration."""

from __future__ import annotations

import numpy as np
import pytest

from rionsis.levels.calibration import calibrate_a1, fit_power_mixture, moment_a1


def _synthetic_mixture(rng, n_time=25, n_files=6000, dt=0.01, a1=1.0, b=0.05, s0=0.15, k=0.05, mu0=3.0, lam=15.0):
    """Draw data i.i.d. per time bin exactly as the mixture model assumes."""
    t = np.arange(n_time, dtype=float) * dt
    mu_t = mu0 * np.exp(-lam * t)
    n = rng.poisson(mu_t[:, None], size=(n_time, n_files))
    sigma = np.sqrt(s0**2 + (k * n) ** 2)
    x = b + a1 * n + sigma * rng.standard_normal((n_time, n_files))
    return x, t, n


def test_fit_power_mixture_recovers_known_parameters():
    rng = np.random.default_rng(0)
    true = dict(a1=1.0, b=0.05, s0=0.15, k=0.05, mu0=3.0, lam=15.0)
    x, t, _ = _synthetic_mixture(rng, **true)
    fit = fit_power_mixture(x, t, n_max=15, mode="exp", n_iter=300)
    assert fit.a1 == pytest.approx(true["a1"], abs=0.05)
    assert fit.b == pytest.approx(true["b"], abs=0.05)
    assert fit.lam == pytest.approx(true["lam"], rel=0.15)


def test_fit_power_mixture_rejects_malformed_input():
    with pytest.raises(ValueError):
        fit_power_mixture(np.array([[1.0, 2.0]]), t_s=np.array([0.0, 0.01, 0.02]))


def test_moment_a1_recovers_slope_from_linear_variance_model():
    rng = np.random.default_rng(1)
    n_files, n_time = 4000, 20
    a1_true = 0.8
    m = np.linspace(0.1, 3.0, n_time)
    variance = 0.05 + a1_true * m  # c2 = 0 -> a1_moment = c1 / (1 + 0) = c1 = a1_true
    y = rng.normal(m[None, :], np.sqrt(variance)[None, :], size=(n_files, n_time))
    a1_est, _c0, _c1, c2, _mean, _var = moment_a1(y)
    assert c2 == pytest.approx(0.0, abs=0.05)
    assert a1_est == pytest.approx(a1_true, abs=0.1)


def test_calibrate_a1_end_to_end_recovers_known_a1():
    rng = np.random.default_rng(42)
    true_a1 = 1.0
    x, t, n = _synthetic_mixture(rng, a1=true_a1, n_files=4000)
    # calibrate_a1 expects y_files_e with shape (n_files, n_time_bins).
    y_files_e = x.T
    dt = float(t[1] - t[0])
    calibration = calibrate_a1(y_files_e, dt, n_max=15, mc_reps=5, em_iter=200, seed=123, external_a1=true_a1)
    assert calibration.a1 == pytest.approx(true_a1, abs=0.1)
    assert calibration.n_files == y_files_e.shape[0]
    assert calibration.external_a1_within_ci68 is not None


def test_calibrate_a1_is_deterministic_given_seed():
    rng = np.random.default_rng(5)
    x, t, _ = _synthetic_mixture(rng, n_files=1000)
    y_files_e = x.T
    dt = float(t[1] - t[0])
    cal_a = calibrate_a1(y_files_e, dt, n_max=15, mc_reps=4, em_iter=100, seed=7)
    cal_b = calibrate_a1(y_files_e, dt, n_max=15, mc_reps=4, em_iter=100, seed=7)
    assert cal_a.a1 == cal_b.a1
    assert cal_a.sd == cal_b.sd
