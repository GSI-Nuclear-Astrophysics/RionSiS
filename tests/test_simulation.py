"""Tests for the pure end-to-end decay Monte Carlo simulation."""

from __future__ import annotations

import numpy as np
import pytest

from rionsis.simulation.decay_simulation import simulate_decay_traces
from rionsis.simulation.noise_bootstrap import EmpiricalNoiseLibrary, build_empirical_noise_library
from rionsis.simulation.window_response import hann_squared_window_response

A1 = 1.0
DT_S = 0.01
N_TIME = 60
NPERSEG = 500
FS = 50_000.0


def _zero_noise_library() -> EmpiricalNoiseLibrary:
    return EmpiricalNoiseLibrary(blocks=np.zeros((4, 15)))


def _flat_sigma(occupancy):
    return np.full_like(np.asarray(occupancy, dtype=float), 0.05)


def _default_kwargs(**overrides):
    kwargs = dict(
        halflife_true_ms=37.0,
        a1=A1,
        dt_s=DT_S,
        n_time=N_TIME,
        n_files=200,
        nperseg=NPERSEG,
        fs=FS,
        window_response=hann_squared_window_response(NPERSEG),
        noise_library=_zero_noise_library(),
        sigma_of_occupancy=_flat_sigma,
        initial_mean_ions=1.5,
        seed=12345,
    )
    kwargs.update(overrides)
    return kwargs


def test_simulate_decay_traces_is_deterministic_given_seed():
    result_a = simulate_decay_traces(**_default_kwargs(seed=42))
    result_b = simulate_decay_traces(**_default_kwargs(seed=42))
    np.testing.assert_array_equal(result_a.y, result_b.y)
    np.testing.assert_array_equal(result_a.n0, result_b.n0)


def test_simulate_decay_traces_different_seeds_differ():
    result_a = simulate_decay_traces(**_default_kwargs(seed=1))
    result_b = simulate_decay_traces(**_default_kwargs(seed=2))
    assert not np.array_equal(result_a.y, result_b.y)


def test_simulate_decay_traces_reproduces_seed_sequence_spawn_semantics():
    # A numpy.random.Generator passed directly must give the same result
    # as passing the equivalent integer seed (see decay_simulation's
    # docstring: rng.spawn(3) after default_rng(seed) is bit-identical to
    # SeedSequence(seed).spawn(3)).
    result_seed = simulate_decay_traces(**_default_kwargs(seed=777))
    result_rng = simulate_decay_traces(**_default_kwargs(seed=np.random.default_rng(777)))
    np.testing.assert_array_equal(result_seed.y, result_rng.y)


def test_noiseless_plateaus_are_exactly_occupancy_times_a1_with_zero_ions():
    # initial_mean_ions cannot be zero (validated), but with a huge true
    # half-life and essentially instantaneous window, N stays ~constant;
    # check the exact-plateau invariant the generator itself asserts on
    # every call still holds and is inspectable from outside.
    result = simulate_decay_traces(**_default_kwargs(halflife_true_ms=1e6, n_files=20))
    integer_mask = np.isclose(result.n_eff, np.rint(result.n_eff), atol=1e-6)
    np.testing.assert_allclose(
        result.noiseless[integer_mask], A1 * np.rint(result.n_eff[integer_mask]), atol=1e-5
    )


def test_very_long_halflife_keeps_occupancy_near_initial_value():
    result = simulate_decay_traces(**_default_kwargs(halflife_true_ms=1e9, n_files=50, initial_mean_ions=3.0))
    # With an astronomically long half-life, essentially no ion decays
    # within the observed window: occupancy at the end should match n0.
    np.testing.assert_allclose(result.n_eff[:, -1], result.n0, atol=1e-6)


def test_very_short_halflife_decays_to_zero_by_end_of_window():
    result = simulate_decay_traces(**_default_kwargs(halflife_true_ms=1e-6, n_files=50, initial_mean_ions=3.0))
    # With a vanishingly short half-life, every ion has decayed well
    # before the first analyzed bin.
    np.testing.assert_allclose(result.n_eff[:, -1], 0.0, atol=1e-6)


def test_zero_initial_ions_rejected():
    with pytest.raises(ValueError):
        simulate_decay_traces(**_default_kwargs(initial_mean_ions=0.0))


def test_n_files_must_be_positive():
    with pytest.raises(ValueError):
        simulate_decay_traces(**_default_kwargs(n_files=0))


def test_build_empirical_noise_library_rejects_degenerate_input():
    y_files = np.ones((5, 20))  # zero variance after mean-subtraction
    with pytest.raises(RuntimeError):
        build_empirical_noise_library(y_files, ground_noise_idx=np.arange(5))


def test_build_empirical_noise_library_draw_has_unit_scale_order():
    rng_source = np.random.default_rng(0)
    y_files = rng_source.normal(0.0, 1.0, size=(30, 15))
    library = build_empirical_noise_library(y_files, ground_noise_idx=np.arange(30))
    draw = library.draw(1000, np.random.default_rng(1))
    assert draw.shape == (1000,)
    assert 0.5 < np.std(draw) < 2.0
