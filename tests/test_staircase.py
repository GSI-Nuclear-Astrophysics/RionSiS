"""Tests for the exact monotonic integer staircase fit (rionsis.levels.staircase)."""

from __future__ import annotations

import numpy as np
import pytest

from rionsis.levels.staircase import (
    StaircaseFitConfig,
    default_change_penalty,
    exact_integer_staircase,
    fit_integer_dataset,
)

A1 = 1.0
NMAX_TABLE = 10


def _flat_variance_table(value: float = 0.02, nmax: int = NMAX_TABLE) -> np.ndarray:
    return np.full(nmax + 1, value, dtype=float)


def test_exact_integer_staircase_recovers_noiseless_levels():
    lengths = [20, 15, 10]
    states = [0, 1, 2]
    signal = np.concatenate([np.full(n, s * A1) for n, s in zip(lengths, states)])
    breakpoints, means, out_lengths, out_states, score = exact_integer_staircase(
        signal, A1, _flat_variance_table(), change_penalty=1.0
    )
    assert list(out_states) == states
    np.testing.assert_array_equal(out_lengths, lengths)
    assert breakpoints[-1] == len(signal)
    assert np.isfinite(score)
    np.testing.assert_allclose(means, states, atol=1e-12)


def test_exact_integer_staircase_is_robust_to_small_noise():
    rng = np.random.default_rng(0)
    lengths = [20, 15, 10]
    states = [0, 1, 2]
    signal = np.concatenate([np.full(n, s * A1) for n, s in zip(lengths, states)])
    signal = signal + rng.normal(0.0, 0.05, size=signal.shape)
    _, _, out_lengths, out_states, _ = exact_integer_staircase(signal, A1, _flat_variance_table(), change_penalty=2.0)
    assert list(out_states) == states
    np.testing.assert_array_equal(out_lengths, lengths)


def test_exact_integer_staircase_monotonic_by_construction():
    rng = np.random.default_rng(1)
    lengths = [12, 8, 8, 6]
    states = [0, 1, 3, 5]  # occupancy can skip values; must stay strictly increasing
    signal = np.concatenate([np.full(n, s * A1) for n, s in zip(lengths, states)])
    signal = signal + rng.normal(0.0, 0.03, size=signal.shape)
    _, _, _, out_states, _ = exact_integer_staircase(signal, A1, _flat_variance_table(nmax=8), change_penalty=2.0)
    assert np.all(np.diff(out_states) > 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(signal=np.array([1.0]), a1=1.0, sigma2_by_n=_flat_variance_table()),  # too short
        dict(signal=np.array([1.0, np.nan]), a1=1.0, sigma2_by_n=_flat_variance_table()),  # non-finite
        dict(signal=np.array([1.0, 2.0]), a1=-1.0, sigma2_by_n=_flat_variance_table()),  # bad a1
        dict(signal=np.array([1.0, 2.0]), a1=1.0, sigma2_by_n=np.array([0.1])),  # too-short variance table
        dict(signal=np.array([1.0, 2.0]), a1=1.0, sigma2_by_n=np.array([0.1, -0.1])),  # negative variance
    ],
)
def test_exact_integer_staircase_validates_inputs(kwargs):
    with pytest.raises(ValueError):
        exact_integer_staircase(**kwargs)


def test_default_change_penalty_matches_2log_n():
    assert default_change_penalty(100, factor=1.0) == pytest.approx(2.0 * np.log(100))
    assert default_change_penalty(100, factor=0.3) == pytest.approx(0.3 * 2.0 * np.log(100))


def test_fit_integer_dataset_recovers_states_across_files():
    # Two files, forward-time decay from N=2 down to N=0 at different times.
    n_time = 40
    row_a = np.concatenate([np.full(10, 2.0), np.full(10, 1.0), np.full(20, 0.0)])
    row_b = np.concatenate([np.full(5, 2.0), np.full(15, 1.0), np.full(20, 0.0)])
    data = np.stack([row_a, row_b])
    config = StaircaseFitConfig(
        a1=A1, sigma2_by_n=_flat_variance_table(), tau2_by_n=np.zeros(NMAX_TABLE + 1),
        change_penalty=default_change_penalty(n_time),
    )
    result = fit_integer_dataset(data, config, file_ids=[7, 9])
    np.testing.assert_array_equal(result.file_ids, [7, 9])
    # Row is reversed internally: forward-time [2,2..,1,1..,0,0..] becomes
    # time-reversed states starting at 0 and increasing to 2.
    assert list(result.states[0]) == [0, 1, 2]
    assert list(result.states[1]) == [0, 1, 2]
    np.testing.assert_array_equal(result.segment_lengths[0], [20, 10, 10])
    np.testing.assert_array_equal(result.segment_lengths[1], [20, 15, 5])
    for levels, states in zip(result.fit_levels, result.states):
        np.testing.assert_allclose(levels, np.asarray(states, dtype=float) * A1)


def test_fit_integer_dataset_rejects_mismatched_file_ids():
    data = np.zeros((2, 10))
    config = StaircaseFitConfig(
        a1=A1, sigma2_by_n=_flat_variance_table(), tau2_by_n=np.zeros(NMAX_TABLE + 1),
        change_penalty=default_change_penalty(10),
    )
    with pytest.raises(ValueError):
        fit_integer_dataset(data, config, file_ids=[1, 2, 3])
