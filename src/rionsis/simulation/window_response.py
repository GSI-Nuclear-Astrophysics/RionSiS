"""STFT window response to a decay occurring mid-window.

An ion that decays partway through an STFT analysis window is captured by
that time bin only for the fraction of the (Hann^2) window *energy* that
falls before the decay time. This turns a discrete decay into a smooth,
sub-bin transition of the effective occupancy ``N_eff``, which is exactly
what is seen in real data at the bin where a decay happens.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.signal import get_window


@dataclass(frozen=True)
class WindowResponse:
    """Cumulative energy fraction of an analysis window vs. normalized position.

    Attributes
    ----------
    x : ndarray
        Position within the window, normalized to ``[-0.5, 0.5]``
        (window start to window end).
    cdf : ndarray
        Cumulative fraction of the window's energy (``sum(window**2)``)
        up to position ``x``.
    """

    x: npt.NDArray[np.float64]
    cdf: npt.NDArray[np.float64]


def hann_squared_window_response(nperseg: int) -> WindowResponse:
    """Cumulative energy-fraction curve of a Hann window of length ``nperseg``.

    Parameters
    ----------
    nperseg : int
        STFT window length in samples (must match the analysis window
        used to build the power spectrogram, see
        ``rionsis.levels.spectral.compute_power_spectrogram``).

    Returns
    -------
    WindowResponse
    """
    window = np.asarray(get_window("hann", int(nperseg), fftbins=True), dtype=float)
    energy = window * window
    cdf = np.concatenate([[0.0], np.cumsum(energy) / np.sum(energy)])
    x = np.linspace(-0.5, 0.5, len(cdf))
    return WindowResponse(x=x, cdf=cdf)


def window_decay_response(
    lifetimes: npt.ArrayLike, time_s: npt.ArrayLike, window_duration_s: float, response: WindowResponse
) -> npt.NDArray[np.float64]:
    """Fraction of each analysis window's energy that precedes each decay time.

    Parameters
    ----------
    lifetimes : array_like, 1D
        Decay times in seconds (one per simulated ion).
    time_s : array_like, 1D, shape (n_time,)
        Center time of each analysis window, in seconds.
    window_duration_s : float
        Analysis window duration in seconds (``nperseg / fs``).
    response : WindowResponse
        From ``hann_squared_window_response``.

    Returns
    -------
    ndarray, shape (len(lifetimes), n_time)
        Fractional occupancy contribution of each ion to each time bin:
        1.0 well before the decay, 0.0 well after, smoothly transitioning
        across the analysis window that straddles the decay.
    """
    lifetimes = np.asarray(lifetimes, dtype=float).ravel()
    time_s = np.asarray(time_s, dtype=float)
    if len(lifetimes) == 0:
        return np.zeros((0, len(time_s)), dtype=float)
    u = (lifetimes[:, None] - time_s[None, :]) / window_duration_s
    return np.interp(u.ravel(), response.x, response.cdf, left=0.0, right=1.0).reshape(u.shape)
