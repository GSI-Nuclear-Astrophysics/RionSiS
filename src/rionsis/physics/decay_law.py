"""Exponential decay law: half-life <-> rate conversion.

Both directions of this relation are used independently by the
``simulation`` domain (turning an assumed half-life into a sampling rate
for exponential lifetimes) and by the ``halflife`` domain (turning a
fitted decay rate back into a reported half-life). Keeping a single
implementation avoids the two domains silently drifting apart.

These functions are unit-agnostic: pass a half-life and rate in whatever
consistent time unit you like (seconds, milliseconds, ...); the caller is
responsible for unit bookkeeping at the boundary. They are NumPy-safe and
accept scalars or arrays.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

_LN2 = float(np.log(2.0))


def halflife_to_rate(half_life: npt.ArrayLike) -> npt.ArrayLike:
    """Convert a half-life ``T_1/2`` to a decay rate ``lambda = ln(2) / T_1/2``.

    Parameters
    ----------
    half_life : array_like
        Half-life, must be positive and finite. Any consistent time unit.

    Returns
    -------
    array_like
        Decay rate in the inverse of the input's time unit.

    Raises
    ------
    ValueError
        If any half-life is not finite and strictly positive.
    """
    half_life = np.asarray(half_life, dtype=float)
    if np.any(~np.isfinite(half_life)) or np.any(half_life <= 0):
        raise ValueError("half_life must be finite and strictly positive.")
    return _LN2 / half_life


def rate_to_halflife(rate: npt.ArrayLike) -> npt.ArrayLike:
    """Convert a decay rate ``lambda`` to a half-life ``T_1/2 = ln(2) / lambda``.

    Parameters
    ----------
    rate : array_like
        Decay rate, must be positive and finite.

    Returns
    -------
    array_like
        Half-life in the input's time unit.

    Raises
    ------
    ValueError
        If any rate is not finite and strictly positive.
    """
    rate = np.asarray(rate, dtype=float)
    if np.any(~np.isfinite(rate)) or np.any(rate <= 0):
        raise ValueError("rate must be finite and strictly positive.")
    return _LN2 / rate
