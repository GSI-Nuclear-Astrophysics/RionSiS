"""Promoted self-consistency checks.

These were originally an inline "toy test" cell in the analysis notebook,
run every time before trusting the real-data half-life result. Promoting
them to a reusable, importable function lets them run as an ordinary
regression test (see ``tests/test_truncated_geometric.py``) instead of
silently depending on someone re-running a specific notebook cell.
"""

from __future__ import annotations

import numpy as np

from rionsis.halflife.truncated_geometric import GeometricFitResult, fit_truncated_geometric_halflife
from rionsis.physics.decay_law import halflife_to_rate


def check_geometric_mle_recovery(
    *,
    dt: float = 0.01,
    halflife_true_ms: float = 37.0,
    n_samples: int = 120_000,
    lo: int = 11,
    big_g: int = 15,
    window_low: int = 35,
    window_high: int = 101,
    tolerance_ms: float = 1.0,
    seed: int | np.random.Generator = 271828,
) -> GeometricFitResult:
    """Check that ``fit_truncated_geometric_halflife`` recovers a known half-life.

    Draws ``n_samples`` truncated geometric dwell times at a known true
    half-life and truncation window, fits them back, and raises if the
    recovered half-life deviates from the truth by more than
    ``tolerance_ms``.

    Parameters
    ----------
    dt : float, default 0.01
        Time-bin width in seconds.
    halflife_true_ms : float, default 37.0
        Injected true half-life, in milliseconds.
    n_samples : int, default 120000
        Number of synthetic geometric draws (large, so the tolerance
        check is about correctness of the estimator, not sampling noise).
    lo : int, default 11
        Strict lower truncation cut (``k > lo - 1``, i.e. ``k >= lo``).
    big_g : int, default 15
        Minimum terminal ground-state bins subtracted from each sample's
        random total window ``H`` to get its upper truncation bound.
    window_low, window_high : int, default 35, 101
        ``H`` is drawn uniformly from ``[window_low, window_high)`` bins
        per sample (a stand-in for real files' varying window lengths).
    tolerance_ms : float, default 1.0
        Maximum allowed deviation between recovered and true half-life.
    seed : int or numpy.random.Generator, default 271828

    Returns
    -------
    GeometricFitResult

    Raises
    ------
    AssertionError
        If the recovered half-life is not finite or deviates from
        ``halflife_true_ms`` by more than ``tolerance_ms``.
    """
    rng = np.random.default_rng(seed)
    rate = halflife_to_rate(halflife_true_ms / 1000.0)
    k = rng.geometric(1.0 - np.exp(-rate * dt), size=n_samples)
    big_h = rng.integers(window_low, window_high, size=len(k))
    hi = big_h - big_g
    use = (k >= lo) & (k <= hi)
    fit = fit_truncated_geometric_halflife(k[use], lo, hi[use], dt)
    if not np.isfinite(fit.halflife_ms) or abs(fit.halflife_ms - halflife_true_ms) > tolerance_ms:
        raise AssertionError(
            f"Geometric MLE recovery check failed: fit={fit.halflife_ms} ms vs truth={halflife_true_ms} ms."
        )
    return fit
