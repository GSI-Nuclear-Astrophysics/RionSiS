"""Occupancy-dependent bin-noise model ``sigma(N) = sqrt(sigma0^2 + (k*N)^2)``.

``N`` is the (possibly fractional, e.g. during a decay transition) ion
occupancy in units of the single-ion power ``A1``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def sigma_of_occupancy(
    occupancy: npt.ArrayLike, sigma0: float, k: float
) -> npt.ArrayLike:
    """Evaluate ``sigma(N) = sqrt(sigma0**2 + (k*N)**2)``.

    Parameters
    ----------
    occupancy : array_like
        Ion occupancy ``N`` (in units of the single-ion power ``A1``).
        Negative values are clipped to zero.
    sigma0 : float
        Noise floor at zero occupancy.
    k : float
        Linear growth coefficient of the noise with occupancy.

    Returns
    -------
    array_like
        Bin-to-bin standard deviation at the given occupancy, in the same
        units as ``sigma0``.
    """
    occupancy = np.maximum(np.asarray(occupancy, dtype=float), 0.0)
    return np.sqrt(sigma0 * sigma0 + (k * occupancy) ** 2)
