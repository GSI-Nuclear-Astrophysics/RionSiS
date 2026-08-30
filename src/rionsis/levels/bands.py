"""Signal/background band integration and ground-state noise normalization.

Two independent steps, kept as two functions:

1. ``integrate_bands`` reduces the (frequency x time) power spectrogram to
   a single background-subtracted power-vs-time trace per file, by
   averaging over a signal frequency band and one or more background
   bands.
2. ``normalize_ground_noise`` rescales those traces to a fixed reference
   noise level, using a robust, size-adaptive selection of files whose
   terminal (late-time) segment is a stable, uncontaminated ground state.
   This is what lets a single hard-coded noise scale (``sigma_e``) be used
   throughout the rest of the pipeline regardless of data-set size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class BandIntegration:
    """Background-subtracted power-vs-time trace per file.

    Attributes
    ----------
    y_files : ndarray, shape (n_files, n_time)
        Signal-band mean power minus background-band mean power.
    mask_signal : ndarray of bool, shape (n_freq,)
        Frequency bins used for the signal band.
    mask_background : ndarray of bool, shape (n_freq,)
        Frequency bins used for the (possibly multi-band) background.
    """

    y_files: npt.NDArray[np.float64]
    mask_signal: npt.NDArray[np.bool_]
    mask_background: npt.NDArray[np.bool_]


def integrate_bands(
    power: npt.ArrayLike,
    frequency: npt.ArrayLike,
    band_signal: tuple[float, float],
    background_bands: list[tuple[float, float]],
) -> BandIntegration:
    """Integrate a signal band and subtract a (multi-band) background.

    Parameters
    ----------
    power : array_like, shape (n_files, n_freq, n_time)
        Power spectrogram, e.g. ``SpectrogramResult.power``.
    frequency : array_like, shape (n_freq,)
        Frequency axis in Hz (0 Hz = ground state).
    band_signal : (float, float)
        ``(low, high)`` frequency bounds of the excited-state band.
    background_bands : list of (float, float)
        One or more flat-background frequency bands.

    Returns
    -------
    BandIntegration

    Raises
    ------
    ValueError
        If either band selects no frequency bin, or the signal and
        background bands overlap.
    """
    power = np.asarray(power)
    frequency = np.asarray(frequency)
    lo, hi = band_signal
    mask_signal = (frequency > lo) & (frequency < hi)
    mask_background = np.zeros(len(frequency), dtype=bool)
    for lo_bg, hi_bg in background_bands:
        mask_background |= (frequency > lo_bg) & (frequency < hi_bg)

    if not np.any(mask_signal):
        raise ValueError(f"band_signal={band_signal} contains no frequency bin.")
    if not np.any(mask_background):
        raise ValueError(f"background_bands={background_bands} contain no frequency bin.")
    if np.any(mask_signal & mask_background):
        raise ValueError("The signal and background bands overlap.")

    signal_mean = power[:, mask_signal, :].mean(axis=1)
    background_mean = power[:, mask_background, :].mean(axis=1)
    y_files = signal_mean - background_mean
    return BandIntegration(y_files=y_files, mask_signal=mask_signal, mask_background=mask_background)


@dataclass(frozen=True)
class GroundNoiseNormalization:
    """Result of rescaling traces to a fixed reference ground-state noise level.

    Attributes
    ----------
    y_files : ndarray, shape (n_files, n_time)
        ``y_files`` input, rescaled so the ground-state bin noise equals
        ``sigma``.
    sigma : float
        Reference noise level all traces are normalized to (``ref_sigma``).
    ground_noise_idx : ndarray of int
        Indices of files used as the stable ground-state noise reference.
    ground_sigma_raw : float
        Median raw (pre-rescaling) terminal-segment noise standard
        deviation across ``ground_noise_idx``.
    ground_acf1 : float
        Lag-1 autocorrelation of the ground-state noise residuals; a
        diagnostic for whether the noise is closer to white or colored.
    """

    y_files: npt.NDArray[np.float64]
    sigma: float
    ground_noise_idx: npt.NDArray[np.int_]
    ground_sigma_raw: float
    ground_acf1: float


def normalize_ground_noise(
    y_files: npt.ArrayLike,
    *,
    ref_sigma: float = 0.66,
    noise_bins: int = 15,
    min_ground_files: int | None = None,
    stability_factor: float = 1.5,
) -> GroundNoiseNormalization:
    """Rescale traces to a fixed reference ground-state bin-noise level.

    Without stable contamination, the late, terminal segment of every file
    belongs to the ground state. This function selects a robust subset of
    files whose terminal segment is both statistically valid (finite,
    positive spread) and stable (its first half and second half agree to
    within ``stability_factor`` times the typical spread), then rescales
    all traces so the median terminal noise standard deviation across that
    subset equals ``ref_sigma``.

    Parameters
    ----------
    y_files : array_like, shape (n_files, n_time)
        Background-subtracted power-vs-time traces, e.g.
        ``BandIntegration.y_files``.
    ref_sigma : float, default 0.66
        Target bin-noise standard deviation after rescaling.
    noise_bins : int, default 15
        Number of terminal time bins used to estimate ground-state noise
        (clipped to the number of time bins available).
    min_ground_files : int, optional
        Minimum number of files required in the stable ground-state
        subset. If ``None``, uses
        ``min(n_files, max(3, min(20, ceil(0.20 * n_files))))``, which
        scales with data-set size but saturates for large data sets.
    stability_factor : float, default 1.5
        A file's terminal segment is called "stable" if the absolute
        change between its first and second half is at most
        ``stability_factor`` times the median terminal standard
        deviation across valid files.

    Returns
    -------
    GroundNoiseNormalization

    Raises
    ------
    RuntimeError
        If fewer than three files are given, if too few files have a
        valid terminal segment, or if the resulting reference noise level
        is not finite and positive.
    """
    y_files = np.asarray(y_files, dtype=float)
    n_files = y_files.shape[0]
    noise_bins = min(noise_bins, y_files.shape[1])
    if min_ground_files is None:
        min_ground_files = min(n_files, max(3, min(20, int(np.ceil(0.20 * n_files)))))
    if n_files < 3:
        raise RuntimeError("At least three files are required for the noise calibration.")

    end = np.asarray(y_files[:, -noise_bins:], dtype=float)
    end_mean = end.mean(axis=1)
    end_std = end.std(axis=1)
    valid = np.isfinite(end_mean) & np.isfinite(end_std) & (end_std > 0)
    if np.count_nonzero(valid) < min_ground_files:
        raise RuntimeError(
            f"Only {np.count_nonzero(valid)} valid files; at least {min_ground_files} are required."
        )

    half = max(2, noise_bins // 3)
    end_change = np.abs(end[:, :half].mean(axis=1) - end[:, -half:].mean(axis=1))
    change_scale = float(np.median(end_std[valid]))
    stable = valid & (end_change <= stability_factor * change_scale)
    if np.count_nonzero(stable) < min_ground_files:
        # For small data sets, use the most stable valid terminal segments
        # instead of failing because of an absolute count intended for
        # large data sets.
        order = np.flatnonzero(valid)[np.argsort(end_change[valid])]
        stable = np.zeros(n_files, dtype=bool)
        stable[order[:min_ground_files]] = True
    ground_noise_idx = np.flatnonzero(stable)

    ground_sigma_raw = float(np.median(end_std[ground_noise_idx]))
    if not np.isfinite(ground_sigma_raw) or ground_sigma_raw <= 0:
        raise RuntimeError("Invalid ground-state noise width.")

    ground_residual = end[ground_noise_idx] - end[ground_noise_idx].mean(axis=1, keepdims=True)
    acf_den = float(np.sum(ground_residual[:, :-1] ** 2))
    ground_acf1 = (
        float(np.sum(ground_residual[:, :-1] * ground_residual[:, 1:]) / acf_den) if acf_den > 0 else np.nan
    )

    y_files_normalized = y_files * (ref_sigma / ground_sigma_raw)
    return GroundNoiseNormalization(
        y_files=y_files_normalized,
        sigma=float(ref_sigma),
        ground_noise_idx=ground_noise_idx,
        ground_sigma_raw=ground_sigma_raw,
        ground_acf1=ground_acf1,
    )
