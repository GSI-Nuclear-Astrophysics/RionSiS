"""Exact monotonic integer staircase fit.

The core idea: reverse the signal in time. A radioactive decay then
becomes a strictly monotonic *increase* of the stored ion count,
``0 -> 1 -> 2 -> ...``. Because the ion count is part of the dynamic
programming (DP) state, the fit finds the globally optimal monotonic
integer path for a fixed, consistent Gaussian negative log-likelihood --
there is no post-hoc reassignment of levels.

``fit_integer_dataset`` applies ``exact_integer_staircase`` to every file
in a data set. Real and simulated data must go through the *same*
``StaircaseFitConfig`` for the Monte Carlo validation in
``rionsis.simulation`` to mean anything: this is enforced by construction
here rather than left to two call sites staying in sync by convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


def exact_integer_staircase(
    signal: npt.ArrayLike,
    a1: float,
    sigma2_by_n: npt.ArrayLike,
    tau2_by_n: npt.ArrayLike | None = None,
    *,
    nmax: int | None = None,
    min_size: int = 1,
    change_penalty: float | None = None,
) -> tuple[list[int], npt.NDArray[np.float64], npt.NDArray[np.int_], npt.NDArray[np.int_], float]:
    """Exact monotonic integer DP fit with a consistent Gaussian NLL.

    Parameters
    ----------
    signal : array_like, 1D
        Time-reversed power signal.
    a1 : float
        Mean power of one ion.
    sigma2_by_n : array_like, 1D
        Bin-to-bin variance for each integer occupancy state ``N``.
    tau2_by_n : array_like, 1D, optional
        Variance of a level offset that is constant across an entire
        segment. This marginalizes the real spread of individual ion
        amplitudes without incorrectly treating it as additional bin
        noise. Defaults to zero (no such spread).
    nmax : int, optional
        Largest allowed integer occupancy state. Defaults to a data-driven
        estimate from the 99.5th percentile of ``signal``.
    min_size : int, default 1
        Minimum segment length in bins.
    change_penalty : float, optional
        Penalty per additional breakpoint, in units of ``-2*log(L)``.
        Defaults to ``2*log(max(len(signal), 2))`` (a BIC-like penalty).

    Returns
    -------
    breakpoints, means, lengths, states, score
        ``breakpoints`` includes the right edge of the final segment.
        ``means`` are the observed segment means (a diagnostic only: the
        latent model staircase is exactly ``states * a1``).

    Raises
    ------
    ValueError
        If any input array is malformed, non-finite, or has an invalid
        shape.
    RuntimeError
        If no admissible monotonic path exists, or an internal
        invariant is violated (should not happen for valid inputs).
    """
    y = np.asarray(signal, dtype=float)
    if y.ndim != 1 or len(y) < 2 or np.any(~np.isfinite(y)):
        raise ValueError("signal must be a finite 1D array with at least two bins.")
    if not np.isfinite(a1) or a1 <= 0:
        raise ValueError("a1 must be positive and finite.")

    s2_all = np.asarray(sigma2_by_n, dtype=float)
    if s2_all.ndim != 1 or len(s2_all) < 2 or np.any(~np.isfinite(s2_all)) or np.any(s2_all <= 0):
        raise ValueError("sigma2_by_n must be a positive, finite 1D table.")
    if tau2_by_n is None:
        t2_all = np.zeros_like(s2_all)
    else:
        t2_all = np.asarray(tau2_by_n, dtype=float)
    if t2_all.shape != s2_all.shape or np.any(~np.isfinite(t2_all)) or np.any(t2_all < 0):
        raise ValueError("tau2_by_n must have the same shape and be nonnegative.")

    n = len(y)
    if nmax is None:
        nmax = max(20, int(np.ceil(max(0.0, np.percentile(y, 99.5)) / a1)) + 4)
    nmax = int(min(max(1, nmax), len(s2_all) - 1))
    nlev = nmax + 1
    mu = np.arange(nlev, dtype=float) * float(a1)
    s2 = s2_all[:nlev]
    t2 = t2_all[:nlev]
    min_size = int(max(1, min_size))
    if change_penalty is None:
        change_penalty = 2.0 * np.log(max(n, 2))
    change_penalty = float(change_penalty)
    if not np.isfinite(change_penalty) or change_penalty < 0:
        raise ValueError("change_penalty must be finite and nonnegative.")

    p1 = np.concatenate(([0.0], np.cumsum(y)))
    p2 = np.concatenate(([0.0], np.cumsum(y * y)))
    inf = float("inf")
    score = np.full((n + 1, nlev), inf)
    prev_pos = np.full((n + 1, nlev), -1, dtype=np.int16)
    prev_lev = np.full((n + 1, nlev), -1, dtype=np.int16)
    strict_value = np.full((n + 1, nlev), inf)
    strict_arg = np.full((n + 1, nlev), -1, dtype=np.int16)

    def segment_cost(starts: npt.NDArray[np.int_], end: int) -> npt.NDArray[np.float64]:
        starts = np.asarray(starts, dtype=int)
        length = (end - starts).astype(float)
        total = p1[end] - p1[starts]
        square = p2[end] - p2[starts]
        mean = total / length
        rss = np.maximum(square - total * total / length, 0.0)
        den = s2[None, :] + length[:, None] * t2[None, :]
        # The omitted term n*log(2*pi) is identical for all paths.
        return (
            rss[:, None] / s2[None, :]
            + (length[:, None] - 1.0) * np.log(s2[None, :])
            + np.log(den)
            + length[:, None] * (mean[:, None] - mu[None, :]) ** 2 / den
        )

    for end in range(min_size, n + 1):
        base = segment_cost(np.array([0]), end)[0]
        score[end] = base
        prev_pos[end] = 0

        starts = np.arange(min_size, end - min_size + 1, dtype=int)
        if len(starts):
            candidate = strict_value[starts] + change_penalty + segment_cost(starts, end)
            jj = np.argmin(candidate, axis=0)
            best = candidate[jj, np.arange(nlev)]
            take = best < score[end]
            if np.any(take):
                chosen_start = starts[jj]
                score[end, take] = best[take]
                prev_pos[end, take] = chosen_start[take]
                prev_lev[end, take] = strict_arg[chosen_start[take], np.flatnonzero(take)]

        best_value = inf
        best_level = -1
        for level in range(nlev):
            strict_value[end, level] = best_value
            strict_arg[end, level] = best_level
            if score[end, level] < best_value:
                best_value = score[end, level]
                best_level = level

    last = int(np.argmin(score[n]))
    if not np.isfinite(score[n, last]):
        raise RuntimeError("No admissible integer staircase path found.")

    segments = []
    end, level = n, last
    while True:
        start = int(prev_pos[end, level])
        if start < 0:
            raise RuntimeError("Invalid backtracking in the integer staircase fit.")
        segments.append((start, end, level))
        if start == 0:
            break
        old_level = int(prev_lev[end, level])
        if old_level < 0 or old_level >= level:
            raise RuntimeError("Nonmonotonic backtracking in the integer staircase fit.")
        end, level = start, old_level
    segments.reverse()

    breakpoints = [int(seg_end) for _, seg_end, _ in segments]
    lengths = np.asarray([seg_end - seg_start for seg_start, seg_end, _ in segments], dtype=int)
    states = np.asarray([lvl for _, _, lvl in segments], dtype=int)
    means = np.asarray(
        [(p1[seg_end] - p1[seg_start]) / (seg_end - seg_start) for seg_start, seg_end, _ in segments], dtype=float
    )
    if breakpoints[-1] != n or lengths.sum() != n or np.any(np.diff(states) <= 0):
        raise RuntimeError("Internal invariant of the integer staircase fit violated.")
    return breakpoints, means, lengths, states, float(score[n, last])


@dataclass(frozen=True)
class StaircaseFitConfig:
    """Configuration shared by every call to ``fit_integer_dataset``.

    Using one config object for both the real-data fit and the
    simulation-reconstruction fit is what guarantees "real and simulated
    data are reconstructed with exactly the same staircase fit" -- an
    invariant the original analysis relied on but did not enforce in
    code.

    Attributes
    ----------
    a1 : float
        Calibrated single-ion power (e.g. ``A1Calibration.a1``).
    sigma2_by_n : ndarray
        Bin-to-bin variance table, indexed by occupancy state.
    tau2_by_n : ndarray
        Constant-offset variance table, indexed by occupancy state.
    nmax_cap : int, default 80
        Hard upper bound on the per-file occupancy search range.
    min_size : int, default 1
        Minimum segment length in bins.
    change_penalty : float
        Penalty per additional breakpoint (see ``exact_integer_staircase``).
    """

    a1: float
    sigma2_by_n: npt.NDArray[np.float64]
    tau2_by_n: npt.NDArray[np.float64]
    change_penalty: float
    nmax_cap: int = 80
    min_size: int = 1


def default_change_penalty(n_time: int, factor: float = 0.30) -> float:
    """``factor * 2*log(n_time)``, the original notebook's breakpoint penalty."""
    return float(factor) * 2.0 * float(np.log(n_time))


@dataclass(frozen=True)
class StaircaseFitResult:
    """Per-file exact integer staircase fits for a data set.

    Attributes
    ----------
    file_ids : ndarray of int
        Row index into the input data for each fitted file, in order.
    segment_means : list of ndarray
        Observed segment means per file (diagnostic; not the latent fit).
    segment_lengths : list of ndarray of int
        Segment lengths in bins per file (time-reversed order).
    states : list of ndarray of int
        Strictly increasing occupancy states per file (time-reversed
        order).
    fit_levels : list of ndarray
        Latent model staircase per file, exactly ``states * a1``.
    scores : ndarray
        Final DP score (``-2*log(L)`` plus breakpoint penalties) per file.
    """

    file_ids: npt.NDArray[np.int_]
    segment_means: list[npt.NDArray[np.float64]]
    segment_lengths: list[npt.NDArray[np.int_]]
    states: list[npt.NDArray[np.int_]]
    fit_levels: list[npt.NDArray[np.float64]]
    scores: npt.NDArray[np.float64]


def fit_integer_dataset(
    data: npt.ArrayLike, config: StaircaseFitConfig, file_ids: npt.ArrayLike | None = None
) -> StaircaseFitResult:
    """Apply the exact integer staircase fit to every row (file) of ``data``.

    Parameters
    ----------
    data : array_like, shape (n_files, n_time_bins)
        Power traces, forward time order (each row is reversed internally
        before fitting).
    config : StaircaseFitConfig
        Shared fit configuration (see ``StaircaseFitConfig``).
    file_ids : array_like of int, optional
        Identifier to report for each row. Defaults to ``0..n_files-1``.

    Returns
    -------
    StaircaseFitResult

    Raises
    ------
    ValueError
        If ``data`` is not a finite 2D array, or ``file_ids`` has the
        wrong length.
    RuntimeError
        If the resulting latent model staircase is not strictly
        monotonic for some file (should not happen; indicates an
        internal invariant violation).
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[1] < 2 or np.any(~np.isfinite(data)):
        raise ValueError("data must be a finite 2D array (files x time bins).")
    if file_ids is None:
        file_ids = np.arange(len(data), dtype=int)
    file_ids = np.asarray(file_ids, dtype=int)
    if len(file_ids) != len(data):
        raise ValueError("file_ids and data have different lengths.")

    means_out, lengths_out, states_out, scores_out = [], [], [], []
    for row in data:
        reversed_row = np.asarray(row[::-1], dtype=float)
        nmax = max(20, int(np.ceil(max(0.0, np.percentile(reversed_row, 99.5)) / config.a1)) + 4)
        nmax = min(config.nmax_cap, nmax)
        _, mean, length, state, score = exact_integer_staircase(
            reversed_row, config.a1, config.sigma2_by_n, config.tau2_by_n,
            nmax=nmax, min_size=config.min_size, change_penalty=config.change_penalty,
        )
        means_out.append(np.asarray(mean, dtype=float))
        lengths_out.append(np.asarray(length, dtype=int))
        states_out.append(np.asarray(state, dtype=int))
        scores_out.append(float(score))

    fit_levels = [config.a1 * np.asarray(state, dtype=float) for state in states_out]
    if any(np.any(np.diff(level) <= 0) for level in fit_levels):
        raise RuntimeError("Internal error: the model staircase is not strictly monotonic.")

    return StaircaseFitResult(
        file_ids=file_ids, segment_means=means_out, segment_lengths=lengths_out,
        states=states_out, fit_levels=fit_levels, scores=np.asarray(scores_out, dtype=float),
    )
