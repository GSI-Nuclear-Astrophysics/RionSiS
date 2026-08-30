"""Calibrate the occupancy-dependent bin noise ``sigma(N)`` from real plateaus.

The mean plateau height of the exact integer staircase fit is exactly
``N*A1`` by construction (see ``rionsis.levels.staircase``). This module
separately measures how much the bin-to-bin power *fluctuates* within
sufficiently long real plateaus, as a function of the occupancy ``N``,
using the functional form in ``rionsis.physics.noise_model``. The result
feeds both the Gaussian likelihood inside the staircase fit and the
synthetic noise generator in ``rionsis.simulation``.

Because the true lifetime here is comparatively short, long (>=10 bin)
plateaus are rare; a documented compromise of >=5-bin plateaus is used
instead, with the sample-standard-deviation's known small-sample low bias
removed via the c4 correction factor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit
from scipy.special import gammaln

from rionsis.physics.noise_model import sigma_of_occupancy


def _unbiased_std_correction(n: float) -> float:
    """c4(n) = E[sample SD] / true SD for an n-sample normal draw."""
    n = float(n)
    if n < 2:
        return np.nan
    return float(np.sqrt(2.0 / (n - 1.0)) * np.exp(gammaln(n / 2.0) - gammaln((n - 1.0) / 2.0)))


@dataclass(frozen=True)
class NoiseCalibration:
    """Fitted ``sigma(N)/A1 = sqrt(sigma0**2 + (k*N)**2)``.

    Attributes
    ----------
    sigma0 : float
        Noise floor at zero occupancy, in units of ``A1``.
    k : float
        Linear growth coefficient with occupancy, in units of ``A1``.
    states : ndarray of int
        Occupancy values with enough long plateaus to be used as support
        points for the fit.
    sigma_by_state : ndarray
        Median per-plateau noise (in units of ``A1``) at each of
        ``states``.
    counts_by_state : ndarray of int
        Number of qualifying plateaus contributing to each support point.
    level_all, sigma_all : ndarray
        Per-plateau (mean level, noise) scatter, for diagnostic plotting.
    min_length : int
        Minimum plateau length (in bins) used.
    min_segments : int
        Minimum number of plateaus required per occupancy state.
    fit_covariance : ndarray, shape (2, 2)
        Covariance of ``(sigma0, k)`` from ``curve_fit``; all-NaN when the
        two-support-point fallback was used.
    """

    sigma0: float
    k: float
    states: npt.NDArray[np.int_]
    sigma_by_state: npt.NDArray[np.float64]
    counts_by_state: npt.NDArray[np.int_]
    level_all: npt.NDArray[np.float64]
    sigma_all: npt.NDArray[np.float64]
    min_length: int
    min_segments: int
    fit_covariance: npt.NDArray[np.float64]

    def sigma_of(self, occupancy: npt.ArrayLike) -> npt.ArrayLike:
        """Fitted bin SD in ``A1`` units at the given (possibly fractional) occupancy."""
        return sigma_of_occupancy(occupancy, self.sigma0, self.k)


def calibrate_noise_vs_occupancy(
    y_files_e: npt.ArrayLike,
    lengths_per_file: list[npt.ArrayLike],
    states_per_file: list[npt.ArrayLike],
    file_ids: npt.ArrayLike,
    a1: float,
    *,
    min_length: int = 5,
    max_n: int = 6,
    min_segments: int | None = None,
) -> NoiseCalibration:
    """Calibrate ``sigma(N)`` from the staircase fit's real plateaus.

    Parameters
    ----------
    y_files_e : array_like, shape (n_files_total, n_time_bins)
        Background-subtracted, noise-normalized power traces.
    lengths_per_file, states_per_file : list of array_like
        Per-file segment lengths and occupancy states, as returned by
        ``rionsis.levels.staircase.fit_integer_dataset`` (time-reversed
        segment order).
    file_ids : array_like of int
        Row index into ``y_files_e`` for each entry of
        ``lengths_per_file`` / ``states_per_file``.
    a1 : float
        Calibrated single-ion power, used to express traces in occupancy
        units.
    min_length : int, default 5
        Minimum plateau length in bins to be used.
    max_n : int, default 6
        Largest occupancy state used as a fit support point.
    min_segments : int, optional
        Minimum number of qualifying plateaus required per support point.
        Defaults to ``min(10, max(4, ceil(0.10 * len(file_ids))))``.

    Returns
    -------
    NoiseCalibration

    Raises
    ------
    RuntimeError
        If fewer than two occupancy states have enough qualifying
        plateaus (the ground state and at least one N>0 level are
        required).
    """
    y_files_e = np.asarray(y_files_e, dtype=float)
    file_ids = np.asarray(file_ids, dtype=int)
    if min_segments is None:
        min_segments = min(10, max(4, int(np.ceil(0.10 * len(file_ids)))))

    noise_by_state: dict[int, list[float]] = {}
    level_all: list[float] = []
    sigma_all: list[float] = []
    for file_id, length, state in zip(file_ids, lengths_per_file, states_per_file):
        y_rev_a1 = np.asarray(y_files_e[int(file_id)][::-1], dtype=float) / a1
        length = np.asarray(length, dtype=int)
        state = np.asarray(state, dtype=int)
        edges = np.concatenate([[0], np.cumsum(length)]).astype(int)
        for start, end, n in zip(edges[:-1], edges[1:], state):
            m = int(end - start)
            if m < min_length:
                continue
            segment = y_rev_a1[start:end]
            c4 = _unbiased_std_correction(m)
            sigma_corr = float(np.std(segment, ddof=1) / c4)
            if not np.isfinite(sigma_corr) or sigma_corr <= 0:
                continue
            noise_by_state.setdefault(int(n), []).append(sigma_corr)
            level_all.append(float(np.mean(segment)))
            sigma_all.append(sigma_corr)

    states = np.asarray(
        sorted(n for n, values in noise_by_state.items() if n <= max_n and len(values) >= min_segments), dtype=int
    )
    if len(states) < 2 or not np.any(states > 0):
        raise RuntimeError(
            "Too few populated ion counts for sigma(N). Check min_length/min_segments; "
            "the ground state and at least one N>0 level are required."
        )
    sigma_by_state = np.asarray([np.median(noise_by_state[int(n)]) for n in states], dtype=float)
    counts_by_state = np.asarray([len(noise_by_state[int(n)]) for n in states], dtype=int)

    i0 = np.flatnonzero(states == 0)
    sigma0_guess = float(sigma_by_state[i0[0]]) if len(i0) else float(np.min(sigma_by_state))
    positive = states > 0
    k_candidates = np.sqrt(np.maximum(sigma_by_state[positive] ** 2 - sigma0_guess**2, 0.0)) / states[positive]
    k_guess = max(0.05, float(np.median(k_candidates)))

    if len(states) >= 3:
        # sigma=1/sqrt(count) weights the robust median support points by
        # their effective sample size; absolute_sigma=False because these
        # are not measurement-error bars.
        popt, covariance = curve_fit(
            sigma_of_occupancy,
            states.astype(float),
            sigma_by_state,
            p0=(sigma0_guess, k_guess),
            sigma=1.0 / np.sqrt(counts_by_state.astype(float)),
            absolute_sigma=False,
            bounds=([1e-6, 0.0], [np.inf, np.inf]),
            maxfev=20000,
        )
        sigma0_fit, k_fit = float(popt[0]), float(popt[1])
    else:
        # Fallback for exactly two support points: the ground state sets
        # sigma0 and the single positive level sets k.
        sigma0_fit, k_fit = sigma0_guess, k_guess
        covariance = np.full((2, 2), np.nan)

    return NoiseCalibration(
        sigma0=sigma0_fit, k=k_fit, states=states, sigma_by_state=sigma_by_state,
        counts_by_state=counts_by_state, level_all=np.asarray(level_all, dtype=float),
        sigma_all=np.asarray(sigma_all, dtype=float), min_length=int(min_length),
        min_segments=int(min_segments), fit_covariance=covariance,
    )
