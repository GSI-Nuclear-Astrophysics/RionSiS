"""Tests for the doubly truncated geometric half-life MLE and its grid/bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from rionsis.halflife.events import EVENT_DTYPE
from rionsis.halflife.truncated_geometric import (
    HalflifeScanConfig,
    bootstrap_halflife_grid,
    fit_truncated_geometric_halflife,
    scan_halflife_grid,
    summarize_bootstrap,
)
from rionsis.validation import check_geometric_mle_recovery


def test_geometric_mle_recovery_promoted_toy_test():
    # This is the original notebook's embedded "toy test", promoted to a
    # real regression test: the estimator must recover a known half-life.
    fit = check_geometric_mle_recovery()
    assert np.isfinite(fit.halflife_ms)
    assert abs(fit.halflife_ms - 37.0) < 1.0


def test_fit_truncated_geometric_returns_nan_below_min_events():
    fit = fit_truncated_geometric_halflife(k=np.array([5, 6, 7]), lo=1, hi=np.array([20, 20, 20]), dt=0.01)
    assert np.isnan(fit.halflife_ms)
    assert fit.n_events == 3


def test_fit_truncated_geometric_rejects_inconsistent_bounds():
    with pytest.raises(ValueError):
        fit_truncated_geometric_halflife(k=np.array([5]), lo=10, hi=np.array([20]), dt=0.01)


def test_fit_truncated_geometric_empty_input():
    fit = fit_truncated_geometric_halflife(k=np.array([], dtype=int), lo=1, hi=np.array([], dtype=int), dt=0.01)
    assert fit.n_events == 0
    assert np.isnan(fit.halflife_ms)


def _make_synthetic_events(rng, n_files, halflife_ms, dt):
    """Build a terminal-event table by directly drawing geometric dwell times."""
    rate = np.log(2.0) / (halflife_ms / 1000.0)
    k = rng.geometric(1.0 - np.exp(-rate * dt), size=n_files)
    h = rng.integers(30, 80, size=n_files)
    k = np.minimum(k, h - 1)  # keep each event inside its file's window
    g = h - k
    file_ids = np.arange(n_files)
    return np.asarray(list(zip(file_ids, k, g, h)), dtype=EVENT_DTYPE), file_ids


def test_scan_halflife_grid_recovers_known_halflife():
    rng = np.random.default_rng(2024)
    dt = 0.01
    events, _ = _make_synthetic_events(rng, n_files=20_000, halflife_ms=40.0, dt=dt)
    config = HalflifeScanConfig(l_values=np.array([2, 5]), g_values=np.array([2, 5]), dt=dt)
    grid = scan_halflife_grid(events, config)
    assert grid.halflife_ms.shape == (2, 2)
    finite = grid.halflife_ms[np.isfinite(grid.halflife_ms)]
    assert len(finite) > 0
    assert np.all(np.abs(finite - 40.0) < 3.0)


def test_bootstrap_halflife_grid_and_summary_shapes():
    rng = np.random.default_rng(7)
    dt = 0.01
    events, file_ids = _make_synthetic_events(rng, n_files=500, halflife_ms=40.0, dt=dt)
    config = HalflifeScanConfig(l_values=np.array([2, 5]), g_values=np.array([2]), dt=dt)
    boot, block_used = bootstrap_halflife_grid(
        events, file_ids, config, draws=30, block_files=10, seed=123
    )
    assert boot.shape == (30, 1, 2)
    assert block_used == 10
    summary = summarize_bootstrap(boot)
    assert summary.sd.shape == (1, 2)
    assert summary.ci95.shape == (2, 1, 2)


def test_bootstrap_halflife_grid_is_deterministic_given_seed():
    rng = np.random.default_rng(3)
    dt = 0.01
    events, file_ids = _make_synthetic_events(rng, n_files=200, halflife_ms=30.0, dt=dt)
    config = HalflifeScanConfig(l_values=np.array([2]), g_values=np.array([2]), dt=dt)
    boot1, _ = bootstrap_halflife_grid(events, file_ids, config, draws=10, block_files=5, seed=99)
    boot2, _ = bootstrap_halflife_grid(events, file_ids, config, draws=10, block_files=5, seed=99)
    np.testing.assert_array_equal(boot1, boot2)


def test_bootstrap_halflife_grid_rejects_unknown_event_file():
    events = np.asarray([(999, 5, 5, 10)], dtype=EVENT_DTYPE)
    config = HalflifeScanConfig(l_values=np.array([1]), g_values=np.array([1]), dt=0.01)
    with pytest.raises(ValueError):
        bootstrap_halflife_grid(events, all_file_ids=[0, 1, 2], config=config, draws=5, block_files=1, seed=1)
