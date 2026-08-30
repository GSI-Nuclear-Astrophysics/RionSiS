"""End-to-end regression test: simulate -> staircase fit -> half-life MLE.

Generates synthetic traces with a known half-life, reconstructs them with
the same staircase fit used for real data, extracts terminal decay
events, and checks the half-life MLE recovers the injected value.
"""

from __future__ import annotations

import numpy as np

from rionsis.halflife.events import extract_terminal_events
from rionsis.halflife.truncated_geometric import HalflifeScanConfig, scan_halflife_grid
from rionsis.levels.staircase import StaircaseFitConfig, default_change_penalty, fit_integer_dataset
from rionsis.simulation.decay_simulation import simulate_decay_traces
from rionsis.simulation.noise_bootstrap import EmpiricalNoiseLibrary
from rionsis.simulation.window_response import hann_squared_window_response


def test_simulation_staircase_fit_recovers_injected_halflife():
    a1 = 1.0
    dt_s = 0.005
    n_time = 80
    nperseg = 500
    fs = 100_000.0
    halflife_true_ms = 40.0
    sigma0, k = 0.10, 0.03

    rng = np.random.default_rng(0)
    noise_library = EmpiricalNoiseLibrary(blocks=rng.standard_normal((20, 15)))

    def sigma_of_occupancy(occupancy):
        occupancy = np.maximum(np.asarray(occupancy, dtype=float), 0.0)
        return np.sqrt(sigma0**2 + (k * occupancy) ** 2)

    sim = simulate_decay_traces(
        halflife_true_ms=halflife_true_ms,
        a1=a1,
        dt_s=dt_s,
        n_time=n_time,
        n_files=6000,
        nperseg=nperseg,
        fs=fs,
        window_response=hann_squared_window_response(nperseg),
        noise_library=noise_library,
        sigma_of_occupancy=sigma_of_occupancy,
        initial_mean_ions=1.0,
        seed=2026,
    )

    nmax_cap = 20
    n_grid = np.arange(nmax_cap + 1, dtype=float)
    config = StaircaseFitConfig(
        a1=a1,
        sigma2_by_n=sigma_of_occupancy(n_grid) ** 2,
        tau2_by_n=np.zeros(nmax_cap + 1),
        change_penalty=default_change_penalty(n_time),
        nmax_cap=nmax_cap,
    )
    fit = fit_integer_dataset(sim.y, config, file_ids=np.arange(sim.n_files))

    events = extract_terminal_events(fit.segment_lengths, fit.states, fit.file_ids, n_time)
    assert len(events) > 200  # enough statistics for the check below to be meaningful

    scan_config = HalflifeScanConfig(l_values=np.array([2, 5]), g_values=np.array([2, 5]), dt=dt_s)
    grid = scan_halflife_grid(events, scan_config)
    finite = grid.halflife_ms[np.isfinite(grid.halflife_ms)]
    assert len(finite) > 0
    # The staircase fit + geometric MLE round trip should recover the
    # injected half-life to within ~25% at these modest statistics.
    assert np.all(np.abs(finite - halflife_true_ms) / halflife_true_ms < 0.25)
