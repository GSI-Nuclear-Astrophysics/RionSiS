"""Pure end-to-end decay simulation with a known, injected half-life.

For each simulated file: a Poisson-distributed initial ion count is
drawn, each ion is assigned an independent exponential lifetime, the STFT
window response turns each decay into a smooth sub-bin transition of the
effective occupancy, the noiseless plateau height is exactly ``occupancy
* a1``, and calibrated, empirically resampled bin noise is added.

RNG use is always explicit (seed or ``numpy.random.Generator``); no
hidden module-level RNG state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt

from rionsis.physics.decay_law import halflife_to_rate
from rionsis.simulation.noise_bootstrap import EmpiricalNoiseLibrary
from rionsis.simulation.window_response import WindowResponse, window_decay_response


@dataclass(frozen=True)
class SimulationResult:
    """Synthetic power-vs-time traces with a known, injected half-life.

    Attributes
    ----------
    seed : int or numpy.random.Generator
        RNG seed (or generator) the simulation was run with.
    halflife_true_ms : float
        Injected true half-life, in milliseconds.
    n_files : int
        Number of simulated files.
    initial_mean_ions : float
        Mean of the Poisson distribution used to draw each file's initial
        ion count.
    n0 : ndarray of int, shape (n_files,)
        Realized initial ion count per file.
    lifetimes_s : list of ndarray
        Per-file individual ion decay times, in seconds since the start
        of the analyzed window.
    n_eff : ndarray, shape (n_files, n_time)
        True (noiseless) effective occupancy per file and time bin,
        including the sub-bin window-response transition.
    noiseless : ndarray, shape (n_files, n_time)
        Noiseless power, exactly ``a1 * n_eff``.
    noise_sigma : ndarray, shape (n_files, n_time)
        Per-bin noise standard deviation actually used, ``a1 * sigma(n_eff)``.
    y : ndarray, shape (n_files, n_time), float32
        Simulated observed power: ``noiseless + noise``.
    """

    seed: int | np.random.Generator
    halflife_true_ms: float
    n_files: int
    initial_mean_ions: float
    n0: npt.NDArray[np.int_]
    lifetimes_s: list[npt.NDArray[np.float64]]
    n_eff: npt.NDArray[np.floating]
    noiseless: npt.NDArray[np.floating]
    noise_sigma: npt.NDArray[np.floating]
    y: npt.NDArray[np.floating]


def simulate_decay_traces(
    *,
    halflife_true_ms: float,
    a1: float,
    dt_s: float,
    n_time: int,
    n_files: int,
    nperseg: int,
    fs: float,
    window_response: WindowResponse,
    noise_library: EmpiricalNoiseLibrary,
    sigma_of_occupancy: Callable[[npt.ArrayLike], npt.ArrayLike],
    initial_mean_ions: float,
    seed: int | np.random.Generator,
) -> SimulationResult:
    """Simulate ``n_files`` independent single-ion decay traces.

    Parameters
    ----------
    halflife_true_ms : float
        True half-life to inject, in milliseconds.
    a1 : float
        Single-ion power (same value used for the staircase fit whose
        output this simulation is meant to validate).
    dt_s : float
        Time-bin width in seconds.
    n_time : int
        Number of time bins per file.
    n_files : int
        Number of files to simulate. Must be at least 1.
    nperseg : int
        STFT window length in samples (must match ``window_response`` and
        the real-data spectrogram it is compared against).
    fs : float
        Sampling rate in Hz, used with ``nperseg`` to get the window
        duration.
    window_response : WindowResponse
        From ``rionsis.simulation.window_response.hann_squared_window_response``.
    noise_library : EmpiricalNoiseLibrary
        From ``rionsis.simulation.noise_bootstrap.build_empirical_noise_library``.
    sigma_of_occupancy : callable
        Maps occupancy (in units of ``a1``) to bin-noise standard
        deviation (also in units of ``a1``), e.g.
        ``rionsis.levels.noise_calibration.NoiseCalibration.sigma_of``.
    initial_mean_ions : float
        Mean of the Poisson distribution for the initial ion count.
    seed : int or numpy.random.Generator
        Seeds three independent sub-streams (initial count, lifetimes,
        noise) via ``numpy.random.default_rng(seed).spawn(3)``.

    Returns
    -------
    SimulationResult

    Raises
    ------
    ValueError
        If ``n_files`` or ``initial_mean_ions`` is not positive.
    RuntimeError
        If the noiseless plateaus at exactly integer occupancy do not
        land on ``occupancy * a1`` to numerical precision (an internal
        consistency check on the generator itself).
    """
    if n_files < 1:
        raise ValueError("n_files must be at least 1.")
    if not np.isfinite(initial_mean_ions) or initial_mean_ions <= 0:
        raise ValueError("initial_mean_ions must be positive and finite.")

    rate_true_s = halflife_to_rate(halflife_true_ms / 1000.0)
    time_s = np.arange(n_time, dtype=float) * dt_s
    window_duration_s = float(nperseg) / float(fs)

    rng = np.random.default_rng(seed)
    rng_n0, rng_tau, rng_noise = rng.spawn(3)

    n0 = rng_n0.poisson(initial_mean_ions, size=n_files).astype(int)
    y = np.empty((n_files, n_time), dtype=np.float32)
    noiseless = np.empty_like(y)
    n_eff = np.empty_like(y)
    noise_sigma = np.empty_like(y)
    lifetimes_s: list[npt.NDArray[np.float64]] = []
    for file_index, count in enumerate(n0):
        lifetimes = rng_tau.exponential(1.0 / rate_true_s, int(count))
        response = window_decay_response(lifetimes, time_s, window_duration_s, window_response)
        occupancy = np.sum(response, axis=0)
        signal = a1 * occupancy
        sigma = a1 * np.asarray(sigma_of_occupancy(occupancy), dtype=float)
        noise = sigma * noise_library.draw(n_time, rng_noise)
        n_eff[file_index] = occupancy
        noiseless[file_index] = signal
        noise_sigma[file_index] = sigma
        y[file_index] = signal + noise
        lifetimes_s.append(lifetimes)

    integer_mask = np.isclose(n_eff, np.rint(n_eff), atol=1e-7, rtol=0.0)
    if not np.allclose(
        noiseless[integer_mask], a1 * np.rint(n_eff[integer_mask]), atol=2e-6 * max(1.0, a1), rtol=2e-6
    ):
        raise RuntimeError("Internal error: simulation plateaus do not lie exactly at occupancy * a1.")

    return SimulationResult(
        seed=seed, halflife_true_ms=float(halflife_true_ms), n_files=int(n_files),
        initial_mean_ions=float(initial_mean_ions), n0=n0, lifetimes_s=lifetimes_s,
        n_eff=n_eff, noiseless=noiseless, noise_sigma=noise_sigma, y=y,
    )
