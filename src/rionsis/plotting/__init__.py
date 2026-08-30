"""Reusable plotting helpers.

Pure presentation: every function here takes already-computed data
(arrays, or the dataclasses returned by ``rionsis.levels`` /
``rionsis.halflife`` / ``rionsis.simulation``) and returns the created
Matplotlib ``Figure``/``Axes``. No physics or statistics is computed in
this package. Requires the optional ``plotting`` extra
(``pip install rionsis[plotting]``); nothing else in ``rionsis`` imports
Matplotlib.
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
