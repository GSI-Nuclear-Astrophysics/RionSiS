"""Diagnostic plots for the single-ion level-detection pipeline."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure
from scipy.special import gammaln

from rionsis.levels.calibration import A1Calibration
from rionsis.levels.noise_calibration import NoiseCalibration


def plot_frequency_bands(
    frequency: npt.ArrayLike,
    power: npt.ArrayLike,
    band_signal: tuple[float, float],
    background_bands: list[tuple[float, float]],
    *,
    n_files: int,
    xlim: tuple[float, float] = (-3000, -1500),
) -> Figure:
    """Summed frequency profile with the signal and background bands overlaid."""
    frequency = np.asarray(frequency)
    profile = np.asarray(power).sum(axis=(0, 2))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.semilogy(frequency, profile, color="steelblue")
    ax.axvline(0, color="k", ls=":", lw=0.9, label="Ground state (0 Hz)")
    ax.axvspan(band_signal[0], band_signal[1], color="tomato", alpha=0.25, label=f"excited {band_signal}")
    for j, (lo, hi) in enumerate(background_bands):
        ax.axvspan(lo, hi, color="green", alpha=0.18, label="Background (bg)" if j == 0 else None)
    ax.set_xlim(*xlim)
    ax.set(xlabel="Frequency relative to ground state [Hz]", ylabel="Sum power |Z|^2 (all files and times)")
    ax.set_title(f"Summed frequency profile ({n_files} files)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def plot_a1_calibration(calibration: A1Calibration, y_files_e: npt.ArrayLike, dt_e: float) -> Figure:
    """Three-panel A1 calibration diagnostic: moments, profile likelihood, mixture fit."""
    y = np.asarray(y_files_e, dtype=float)
    from rionsis.levels.calibration import moment_a1

    a1_moment, c0, c1, c2, m, v = moment_a1(y)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9), constrained_layout=True)

    ax = axes[0]
    grid_m = np.linspace(0, m.max() * 1.05, 200)
    ax.plot(m, v, "o", ms=4.5, color="steelblue", alpha=0.7, label="Time bins")
    ax.plot(grid_m, c0 + c1 * grid_m + c2 * grid_m**2, color="black", lw=2, label=f"Fit -> A1={a1_moment:.2f}")
    ax.set(xlabel="Ensemble mean m(t) [power]", ylabel="Ensemble variance v(t) [power^2]",
           title="Method of moments: slope = A1")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    d2 = 2.0 * (calibration.profile_nll - calibration.profile_nll.min())
    ax.plot(calibration.profile_grid, d2, "o-", color="#31688e", lw=1.8, ms=5)
    ax.axvspan(calibration.ci68[0], calibration.ci68[1], color="#35b779", alpha=0.22, label="68% (bootstrap)")
    ax.axvline(calibration.a1, color="#35b779", lw=2.2, label=f"A1={calibration.a1:.3f}")
    if calibration.external_a1 is not None:
        ax.axvline(calibration.external_a1, color="tomato", ls="--", lw=1.8,
                    label=f"external A1={calibration.external_a1}")
    ax.set(xlabel="A1 [Power]", ylabel="2*Delta NLL", ylim=(-1, min(60, d2.max() * 1.05)),
           title="Profile likelihood of the mixture")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[2]
    fit = calibration.fit_exp
    n_max = len(fit.mu) - 1
    win = np.flatnonzero((fit.mu > 0.25) & (fit.mu < 3.5))
    if len(win) < 3:
        win = np.arange(min(15, y.shape[1]))
    pool = y[:, win].ravel()
    hi = float(fit.b + calibration.a1 * (np.ceil(fit.mu[win].max()) + 4))
    pool_view = pool[(pool >= -3.0 * fit.s0) & (pool <= hi)]
    edges = np.histogram_bin_edges(pool_view, bins=55)
    ax.hist(pool_view, bins=edges, color="#31688e", alpha=0.75, density=True, label=f"Data, bins {win[0]}-{win[-1]}")
    x_grid = np.linspace(edges[0], edges[-1], 600)
    n_grid = np.arange(n_max + 1, dtype=float)
    sigma_n = np.sqrt(fit.s0**2 + (fit.k * n_grid) ** 2)
    log_w = np.log(np.maximum(fit.mu[win], 1e-12))[:, None] * n_grid - fit.mu[win][:, None] - gammaln(n_grid + 1.0)[None, :]
    w_n = np.exp(log_w)
    w_n /= w_n.sum(axis=1, keepdims=True)
    w_mean = w_n.mean(axis=0)
    components = (
        np.exp(-0.5 * ((x_grid[:, None] - (fit.b + calibration.a1 * n_grid)) / sigma_n) ** 2) / (sigma_n * np.sqrt(2 * np.pi))
    ) * w_mean
    ax.plot(x_grid, components.sum(axis=1), color="black", lw=2.0, label="Mixture fit")
    for n in range(min(6, n_max + 1)):
        if w_mean[n] > 0.01:
            ax.plot(x_grid, components[:, n], color="tomato", lw=1.0, alpha=0.8)
            ax.axvline(fit.b + calibration.a1 * n, color="0.4", ls=":", lw=0.8)
    ax.set(xlim=(edges[0], edges[-1]), xlabel="Power", ylabel="Density",
           title=f"Mixture in window mu={fit.mu[win].min():.2f}..{fit.mu[win].max():.2f}")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    return fig


def plot_staircase_fit(
    y_row: npt.ArrayLike,
    fit_levels: npt.ArrayLike,
    segment_means: npt.ArrayLike,
    segment_lengths: npt.ArrayLike,
    states: npt.ArrayLike,
    a1: float,
    *,
    title: str = "Exact integer staircase fit",
) -> Figure:
    """Time-reversed power trace with the fitted monotonic staircase overlaid.

    Parameters mirror one file's worth of
    ``rionsis.levels.staircase.StaircaseFitResult`` (already time-reversed,
    as produced by the fit): ``y_row`` is the raw forward-time trace (this
    function reverses it for display, matching the fit's own convention).
    """
    y_row = np.asarray(y_row, dtype=float)
    fit_levels = np.asarray(fit_levels, dtype=float)
    segment_means = np.asarray(segment_means, dtype=float)
    segment_lengths = np.asarray(segment_lengths, dtype=int)
    states = np.asarray(states, dtype=int)

    y = y_row[::-1] / a1
    edges = np.concatenate([[0], np.cumsum(segment_lengths)]).astype(int)
    fig, ax = plt.subplots(figsize=(12, 4.6), constrained_layout=True)
    ax.plot(np.arange(len(y)), y, color="steelblue", alpha=0.55, lw=0.9, label="Data (time reversed)")
    for n in range(int(max(states.max(), np.ceil(np.nanmax(y)))) + 1):
        ax.axhline(n, color="0.5", ls=":", lw=0.45, alpha=0.35)
    for j, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        fit_level = fit_levels[j] / a1
        raw_mean = segment_means[j] / a1
        ax.hlines(fit_level, start, end, color="tomato", lw=2.6, zorder=3,
                  label="monotonic model staircase N*A1" if j == 0 else None)
        if j:
            ax.vlines(start, fit_levels[j - 1] / a1, fit_level, color="tomato", lw=1.3, zorder=3)
        ax.text((start + end) / 2, fit_level + 0.18, f"N={states[j]} | mean={raw_mean:.2f}",
                ha="center", va="bottom", fontsize=8, color=("green" if states[j] == 0 else "purple"))
    ax.set(xlabel="Time bin (reversed: left = late/ground state)", ylabel="Power / A1", title=title)
    ax.grid(alpha=0.18)
    ax.legend(fontsize=8)
    return fig


def plot_level_height_histogram(
    level_heights: npt.ArrayLike, a1: float, *, bin_width_a1: float = 0.05, title: str = "Observed segment means"
) -> Figure:
    """Histogram of observed staircase segment means, in power units."""
    level_heights = np.asarray(level_heights, dtype=float)
    level_heights = level_heights[np.isfinite(level_heights)]
    width = bin_width_a1 * a1
    low = min(-0.5 * a1, np.floor(np.quantile(level_heights, 0.001) / width) * width)
    high = max(10.0 * a1, np.ceil(np.quantile(level_heights, 0.995) / width) * width)
    edges = np.arange(low, high + 1.001 * width, width)

    fig, ax = plt.subplots(figsize=(11.5, 5.0), constrained_layout=True)
    ax.hist(level_heights, bins=edges, color="#31688e", alpha=0.84, edgecolor="none")
    for n in range(max(0, int(np.floor(high / a1))) + 1):
        ax.axvline(n * a1, color="black", lw=0.55, alpha=0.22)
    ax.set(xlabel="Observed segment mean / power", ylabel="Counts (one count per segment)",
           title=title, xlim=(edges[0], edges[-1]))
    ax.grid(axis="y", alpha=0.22)
    return fig


def plot_noise_vs_occupancy(calibration: NoiseCalibration) -> Figure:
    """Per-plateau noise scatter with the fitted sigma(N) curve."""
    x_max = max(float(calibration.states.max() + 1), min(12.0, float(np.ceil(np.quantile(calibration.level_all, 0.98)))))
    x_grid = np.linspace(0.0, x_max, 300)

    fig, ax = plt.subplots(figsize=(10.5, 6.0), constrained_layout=True)
    ax.scatter(calibration.level_all, calibration.sigma_all, s=12, color="steelblue", alpha=0.28,
               label=f"individual plateaus (length >= {calibration.min_length} bins)")
    ax.plot(calibration.states, calibration.sigma_by_state, "o", color="darkred", ms=7,
            label="Median for each ion count used", zorder=4)
    for n, sigma, count in zip(calibration.states, calibration.sigma_by_state, calibration.counts_by_state):
        ax.annotate(f"n={int(count)}", (n, sigma), xytext=(5, 5), textcoords="offset points", fontsize=8, color="darkred")
    ax.plot(x_grid, calibration.sigma_of(x_grid), color="black", lw=2.0,
            label=rf"Fit: $\sigma(N)=\sqrt{{\sigma_0^2+(kN)^2}}$ ({calibration.sigma0:.2f}, {calibration.k:.2f})")
    ax.set(xlabel="Observed plateau mean / A1", ylabel="SD within the plateau / A1",
           title="Bin noise versus level height", xlim=(-0.5, x_max))
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.25)
    ax.legend()
    return fig
