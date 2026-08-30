"""Reusable plotting helpers.

Pure presentation: takes already-computed data and returns Matplotlib
``Figure``/``Axes`` objects. Requires the optional ``plotting`` extra
(``pip install rionsis[plotting]``).
"""

from rionsis.plotting.halflife_plots import plot_halflife_comparison, plot_halflife_heatmap
from rionsis.plotting.levels_plots import (
    plot_a1_calibration,
    plot_frequency_bands,
    plot_level_height_histogram,
    plot_noise_vs_occupancy,
    plot_staircase_fit,
)

__all__ = [
    "plot_halflife_comparison",
    "plot_halflife_heatmap",
    "plot_a1_calibration",
    "plot_frequency_bands",
    "plot_level_height_histogram",
    "plot_noise_vs_occupancy",
    "plot_staircase_fit",
]
