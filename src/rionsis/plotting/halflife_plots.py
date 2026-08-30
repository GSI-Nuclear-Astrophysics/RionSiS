"""Diagnostic plots for the half-life (G, L) cut-grid scan."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from rionsis.halflife.truncated_geometric import HalflifeGrid


def plot_halflife_heatmap(
    ax: Axes,
    grid: HalflifeGrid,
    error: npt.ArrayLike,
    title: str,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Draw one half-life heatmap (with per-cell value/error/count labels) on ``ax``."""
    half_life = grid.halflife_ms
    error = np.asarray(error)
    image = ax.imshow(
        half_life, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax,
        extent=[-0.5, len(grid.config.l_values) - 0.5, -0.5, len(grid.config.g_values) - 0.5],
    )
    ax.set_xticks(range(len(grid.config.l_values)), grid.config.l_values)
    ax.set_yticks(range(len(grid.config.g_values)), grid.config.g_values)
    ax.set_xlabel(r"strict cut $k>L$  ($L$ [bins])")
    ax.set_ylabel(r"minimum terminal ground state $g\geq G$  ($G$ [bins])")
    ax.set_title(title)
    finite = half_life[np.isfinite(half_life)]
    middle = float(np.nanmedian(finite)) if len(finite) else np.nan
    for i_g in range(half_life.shape[0]):
        for i_l in range(half_life.shape[1]):
            value = half_life[i_g, i_l]
            if np.isfinite(value):
                err = error[i_g, i_l]
                err_text = f"{err:.0f}" if np.isfinite(err) else "?"
                ax.text(i_l, i_g, f"{value:.0f}\n+/-{err_text} ({grid.counts[i_g, i_l]})",
                        ha="center", va="center", fontsize=6.3, color="white" if value <= middle else "black")
    return image


def plot_halflife_comparison(
    real_grid: HalflifeGrid, real_error: npt.ArrayLike,
    sim_grid: HalflifeGrid, sim_error: npt.ArrayLike, halflife_input_ms: float,
) -> Figure:
    """Side-by-side real-data vs. simulation-reconstruction half-life heatmaps.

    Shares a common color scale so the two panels are directly
    comparable; the decisive check is whether a common, stable cut range
    exists in both.
    """
    heat_values = np.concatenate(
        [real_grid.halflife_ms[np.isfinite(real_grid.halflife_ms)], sim_grid.halflife_ms[np.isfinite(sim_grid.halflife_ms)]]
    )
    vmin, vmax = float(np.min(heat_values)), float(np.max(heat_values))
    fig, axes = plt.subplots(1, 2, figsize=(17.2, 6.7), constrained_layout=True)
    plot_halflife_heatmap(axes[0], real_grid, real_error, "Real data: discrete T1/2 MLE", vmin=vmin, vmax=vmax)
    im_sim = plot_halflife_heatmap(
        axes[1], sim_grid, sim_error, f"Simulation: reconstructed T1/2 MLE (input {halflife_input_ms:.1f} ms)",
        vmin=vmin, vmax=vmax,
    )
    fig.colorbar(im_sim, ax=axes, label=r"$T_{1/2}$ [ms]", shrink=0.88)
    return fig
