"""Doubly truncated discrete (geometric) half-life MLE.

The dwell time ``k`` of a single ion in a state, in whole time bins,
follows a geometric distribution with per-bin survival probability
``exp(-rate*dt)``. Both ends of the observation window are
hard-truncated: ``k < lo = L+1`` is never observed (a "strict cut"
removing ambiguous short-lived detections), and ``k > hi = H-G`` is not
observed either (the window ends, or is shortened by requiring ``G``
bins of confirmed terminal ground state). Uncertainty is quantified with
a moving-block bootstrap over files, since dwell times within a file are
not independent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize_scalar

#: Cut grids used by the original analysis; pass your own to
#: ``HalflifeScanConfig`` for a different data set.
DEFAULT_L_VALUES: npt.NDArray[np.int_] = np.asarray([5, 8, 10, 12, 15, 17, 20, 25], dtype=int)
DEFAULT_G_VALUES: npt.NDArray[np.int_] = np.asarray([5, 8, 10, 12, 15, 20, 25], dtype=int)


@dataclass(frozen=True)
class GeometricFitResult:
    """Doubly truncated geometric MLE result for one ``(G, L)`` cut.

    Attributes
    ----------
    halflife_ms : float
        Fitted half-life in milliseconds, or NaN if there was not enough
        data (fewer than ``min_events`` events, or fewer than
        ``min_informative`` events with a non-degenerate truncation
        window).
    n_events : int
        Number of events available for this cut.
    rate_hz : float
        Fitted decay rate in inverse seconds (NaN alongside ``halflife_ms``).
    at_boundary : bool
        True if the MLE lies within 1e-5 of ``rate_bounds`` (fit is not
        well constrained by the data at this cut).
    n_informative : int
        Number of events with a non-degenerate truncation window
        (``hi > lo``).
    """

    halflife_ms: float
    n_events: int
    rate_hz: float
    at_boundary: bool
    n_informative: int


def fit_truncated_geometric_halflife(
    k: npt.ArrayLike,
    lo: npt.ArrayLike,
    hi: npt.ArrayLike,
    dt: float,
    *,
    rate_bounds: tuple[float, float] = (0.1, 100.0),
    min_events: int = 8,
    min_informative: int = 5,
) -> GeometricFitResult:
    """MLE of the doubly truncated geometric distribution in integer bins.

    Parameters
    ----------
    k : array_like of int
        Observed dwell time of each event, in whole time bins.
    lo : array_like of int (or scalar, broadcast)
        Per-event lower truncation bound (inclusive).
    hi : array_like of int
        Per-event upper truncation bound (inclusive).
    dt : float
        Time-bin width in seconds.
    rate_bounds : (float, float), default (0.1, 100.0)
        Bracket for the decay rate (1/s) search.
    min_events : int, default 8
        Minimum number of events required to attempt a fit.
    min_informative : int, default 5
        Minimum number of events with ``hi > lo`` required.

    Returns
    -------
    GeometricFitResult

    Raises
    ------
    ValueError
        If ``k`` and ``hi`` have different shapes, or the truncation
        bounds are inconsistent (``lo < 1``, ``hi < lo``, or ``k`` outside
        ``[lo, hi]``).
    """
    k = np.asarray(k, dtype=int)
    lo = np.broadcast_to(np.asarray(lo, dtype=int), k.shape)
    hi = np.asarray(hi, dtype=int)
    if k.shape != hi.shape:
        raise ValueError("k and hi must have the same shape.")
    if len(k) == 0:
        return GeometricFitResult(np.nan, 0, np.nan, False, 0)
    if np.any(lo < 1) or np.any(hi < lo) or np.any(k < lo) or np.any(k > hi):
        raise ValueError("Invalid discrete truncation interval.")

    width = hi - lo + 1
    informative = int(np.count_nonzero(width > 1))
    n_events = int(len(k))
    if n_events < min_events or informative < min_informative:
        return GeometricFitResult(np.nan, n_events, np.nan, False, informative)
    distance = k - lo

    def nll(rate: float) -> float:
        eta = float(rate) * dt
        log_event = np.log(-np.expm1(-eta))
        log_norm = np.log(-np.expm1(-eta * width))
        return float(-(n_events * log_event - eta * np.sum(distance) - np.sum(log_norm)))

    result = minimize_scalar(
        nll, bounds=tuple(map(float, rate_bounds)), method="bounded", options={"xatol": 1e-10, "maxiter": 500}
    )
    if not result.success or not np.isfinite(result.fun):
        return GeometricFitResult(np.nan, n_events, np.nan, False, informative)
    rate = float(result.x)
    span = rate_bounds[1] - rate_bounds[0]
    at_boundary = bool(rate - rate_bounds[0] < 1e-5 * span or rate_bounds[1] - rate < 1e-5 * span)
    halflife_ms = float(np.log(2) / rate * 1000.0)
    return GeometricFitResult(halflife_ms, n_events, rate, at_boundary, informative)


@dataclass(frozen=True)
class HalflifeScanConfig:
    """Grid of ``(G, L)`` cut choices and MLE settings for a half-life scan."""

    l_values: npt.NDArray[np.int_] = DEFAULT_L_VALUES
    g_values: npt.NDArray[np.int_] = DEFAULT_G_VALUES
    dt: float = 0.0
    rate_bounds: tuple[float, float] = (0.1, 100.0)
    min_events: int = 8
    min_informative: int = 5


@dataclass(frozen=True)
class HalflifeGrid:
    """Half-life MLE evaluated over every ``(G, L)`` cut in a ``HalflifeScanConfig``."""

    halflife_ms: npt.NDArray[np.float64]
    counts: npt.NDArray[np.int_]
    rate_hz: npt.NDArray[np.float64]
    at_boundary: npt.NDArray[np.bool_]
    n_informative: npt.NDArray[np.int_]
    config: HalflifeScanConfig


def scan_halflife_grid(events: npt.NDArray, config: HalflifeScanConfig) -> HalflifeGrid:
    """Fit the half-life for every ``(G, L)`` combination in ``config``.

    Parameters
    ----------
    events : structured ndarray
        Decay events with fields ``k``, ``g``, ``H`` (see
        ``rionsis.halflife.events.extract_terminal_events``).
    config : HalflifeScanConfig

    Returns
    -------
    HalflifeGrid
    """
    shape = (len(config.g_values), len(config.l_values))
    halflife_ms = np.full(shape, np.nan)
    counts = np.zeros(shape, dtype=int)
    rate_hz = np.full(shape, np.nan)
    at_boundary = np.zeros(shape, dtype=bool)
    n_informative = np.zeros(shape, dtype=int)
    for i_g, big_g in enumerate(config.g_values):
        hi_all = events["H"] - int(big_g)
        for i_l, big_l in enumerate(config.l_values):
            lo = int(big_l) + 1
            use = (events["g"] >= big_g) & (events["k"] >= lo)
            fit = fit_truncated_geometric_halflife(
                events["k"][use], lo, hi_all[use], config.dt,
                rate_bounds=config.rate_bounds, min_events=config.min_events, min_informative=config.min_informative,
            )
            halflife_ms[i_g, i_l] = fit.halflife_ms
            counts[i_g, i_l] = fit.n_events
            rate_hz[i_g, i_l] = fit.rate_hz
            at_boundary[i_g, i_l] = fit.at_boundary
            n_informative[i_g, i_l] = fit.n_informative
    return HalflifeGrid(
        halflife_ms=halflife_ms, counts=counts, rate_hz=rate_hz,
        at_boundary=at_boundary, n_informative=n_informative, config=config,
    )


def bootstrap_halflife_grid(
    events: npt.NDArray,
    all_file_ids: npt.ArrayLike,
    config: HalflifeScanConfig,
    *,
    draws: int,
    block_files: int,
    seed: int | np.random.Generator,
) -> tuple[npt.NDArray[np.float64], int]:
    """Circular moving-block bootstrap of the half-life grid, resampling over files.

    Parameters
    ----------
    events : structured ndarray
        As returned by ``extract_terminal_events``.
    all_file_ids : array_like of int
        Every file id that was eligible to contribute an event (including
        files with none), defining the resampling population.
    config : HalflifeScanConfig
    draws : int
        Number of bootstrap replicates.
    block_files : int
        Nominal circular block length in files (clipped to
        ``[1, len(all_file_ids) // 2]``).
    seed : int or numpy.random.Generator

    Returns
    -------
    boot, block_files_used
        ``boot`` has shape ``(draws, len(config.g_values), len(config.l_values))``.

    Raises
    ------
    ValueError
        If ``all_file_ids`` is empty or contains duplicates, or an event
        references a file id absent from ``all_file_ids``.
    """
    all_file_ids = np.asarray(all_file_ids, dtype=int)
    if len(all_file_ids) == 0 or len(np.unique(all_file_ids)) != len(all_file_ids):
        raise ValueError("all_file_ids must be nonempty and unique.")
    block_files_used = int(min(max(1, block_files), max(1, len(all_file_ids) // 2)))
    event_at_position = np.full(len(all_file_ids), -1, dtype=int)
    lookup = {int(file_id): pos for pos, file_id in enumerate(all_file_ids)}
    for event_index, file_id in enumerate(events["file"]):
        if int(file_id) not in lookup:
            raise ValueError("An event file is missing from all_file_ids.")
        event_at_position[lookup[int(file_id)]] = event_index

    rng = np.random.default_rng(seed)
    boot = np.full((int(draws), len(config.g_values), len(config.l_values)), np.nan)
    n_files = len(all_file_ids)
    for b in range(int(draws)):
        positions: list[int] = []
        while len(positions) < n_files:
            start = int(rng.integers(0, n_files))
            positions.extend(((start + np.arange(block_files_used)) % n_files).tolist())
        event_indices = event_at_position[np.asarray(positions[:n_files], dtype=int)]
        event_indices = event_indices[event_indices >= 0]
        boot[b] = scan_halflife_grid(events[event_indices], config).halflife_ms
    return boot, block_files_used


@dataclass(frozen=True)
class BootstrapSummary:
    """Per-cell bootstrap standard deviation, 95% CI, and valid replicate count."""

    sd: npt.NDArray[np.float64]
    ci95: npt.NDArray[np.float64]
    valid_counts: npt.NDArray[np.int_]


def summarize_bootstrap(boot: npt.NDArray[np.float64]) -> BootstrapSummary:
    """Summarize a bootstrap array from ``bootstrap_halflife_grid`` per grid cell."""
    sd = np.full(boot.shape[1:], np.nan)
    ci95 = np.full((2,) + boot.shape[1:], np.nan)
    valid = np.zeros(boot.shape[1:], dtype=int)
    for i_g in range(boot.shape[1]):
        for i_l in range(boot.shape[2]):
            values = boot[:, i_g, i_l]
            values = values[np.isfinite(values)]
            valid[i_g, i_l] = len(values)
            if len(values) >= 2:
                sd[i_g, i_l] = float(np.std(values, ddof=1))
                ci95[:, i_g, i_l] = np.percentile(values, [2.5, 97.5])
    return BootstrapSummary(sd=sd, ci95=ci95, valid_counts=valid)
