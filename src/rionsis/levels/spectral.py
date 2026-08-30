"""STFT power spectrogram and ground-state frequency centering.

Turns raw complex IQ traces into a per-file power spectrogram
(frequency x time), and removes file-to-file drift of the ground-state
frequency by recentering each spectrogram on its (median-filtered)
strongest frequency bin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.ndimage import median_filter
from scipy.signal import stft


@dataclass(frozen=True)
class SpectrogramResult:
    """Centered power spectrogram for a set of files.

    Attributes
    ----------
    power : ndarray, shape (n_files, n_freq, n_time), float32
        Recentered power spectrogram ``|STFT|**2`` for every file.
    frequency : ndarray, shape (n_freq,)
        Frequency axis in Hz, relative to the (per-file-corrected)
        ground-state frequency (0 Hz = ground state).
    times : ndarray, shape (n_time,)
        STFT time axis in seconds, shared by all files.
    base_freq : ndarray, shape (n_freq,)
        Uncentered (fftshifted) frequency axis in Hz, as returned by the
        STFT of the first file.
    raw_gbin : ndarray of int, shape (n_files,)
        Per-file argmax frequency bin before median-filter smoothing.
    gbin : ndarray of int, shape (n_files,)
        Per-file ground-state bin actually used for centering (median
        filtered across files for robustness against single-file noise).
    centering_delta_bins : ndarray of int, shape (n_files,)
        ``raw_gbin - gbin``; large values flag files where the smoothed
        centering estimate disagrees with the raw estimate.
    centering_suspect_idx : ndarray of int
        Indices into ``power`` where ``abs(centering_delta_bins) > 1``.
    dt : float
        STFT time step in seconds.
    df : float
        Frequency bin width in Hz.
    """

    power: npt.NDArray[np.float32]
    frequency: npt.NDArray[np.float64]
    times: npt.NDArray[np.float64]
    base_freq: npt.NDArray[np.float64]
    raw_gbin: npt.NDArray[np.int_]
    gbin: npt.NDArray[np.int_]
    centering_delta_bins: npt.NDArray[np.int_]
    centering_suspect_idx: npt.NDArray[np.int_]
    dt: float
    df: float


def compute_power_spectrogram(
    iq_all: npt.ArrayLike, fs: float, *, dt_bin: float = 0.01, nperseg: int = 500
) -> SpectrogramResult:
    """Compute a centered STFT power spectrogram for every file.

    Parameters
    ----------
    iq_all : array_like, shape (n_files, n_samples), complex
        Raw IQ traces, e.g. ``rionsis.io.IQDataset.iq``.
    fs : float
        Sampling rate in Hz.
    dt_bin : float, default 0.01
        Desired STFT time step in seconds; converted to an integer hop
        size via ``round(dt_bin * fs)``.
    nperseg : int, default 500
        STFT window length in samples.

    Returns
    -------
    SpectrogramResult

    Raises
    ------
    ValueError
        If ``nperseg`` is smaller than the hop size implied by ``dt_bin``.
    """
    iq_all = np.asarray(iq_all)
    n_files = iq_all.shape[0]
    hop = int(round(dt_bin * fs))
    noverlap = nperseg - hop
    if noverlap < 0:
        raise ValueError(f"nperseg={nperseg} < hop={hop}; nperseg must be at least {hop} for the requested dt_bin.")

    f0, t0, _ = stft(
        iq_all[0], fs=fs, nperseg=nperseg, noverlap=noverlap, return_onesided=False, boundary=None, padded=False
    )
    base_freq = np.fft.fftshift(f0)
    times = np.asarray(t0)
    n_freq, n_time = len(base_freq), len(times)
    df = fs / nperseg

    power = np.empty((n_files, n_freq, n_time), dtype=np.float32)
    for i in range(n_files):
        _, _, z = stft(
            iq_all[i], fs=fs, nperseg=nperseg, noverlap=noverlap, return_onesided=False, boundary=None, padded=False
        )
        power[i] = (np.abs(np.fft.fftshift(z, axes=0)) ** 2).astype(np.float32)

    raw_gbin = np.asarray([np.argmax(s.sum(axis=1)) for s in power], dtype=int)
    # scipy.ndimage.median_filter degenerates to all-zero output when the
    # requested window ("size") exceeds the array length (reproduced with
    # scipy 1.15: median_filter([133]*4, size=11, mode="nearest") -> zeros).
    # This has no effect for any realistically sized data set (the original
    # analysis used ~100 files; behaviour for n_files >= 5 is numerically
    # identical with or without this clip), but without it, a data set with
    # four or fewer files would silently center every spectrogram on bin 0
    # instead of the ground-state bin. Clipping the window to the number of
    # files fixes that degenerate case without changing anything else.
    smoothing_window = min(11, n_files)
    gbin = np.asarray(median_filter(raw_gbin, size=smoothing_window, mode="nearest"), dtype=int)
    center = n_freq // 2
    centering_delta_bins = raw_gbin - gbin
    centering_suspect_idx = np.flatnonzero(np.abs(centering_delta_bins) > 1)
    for i in range(n_files):
        power[i] = np.roll(power[i], center - int(gbin[i]), axis=0)
    frequency = (np.arange(n_freq) - center) * df

    return SpectrogramResult(
        power=power,
        frequency=frequency,
        times=times,
        base_freq=base_freq,
        raw_gbin=raw_gbin,
        gbin=gbin,
        centering_delta_bins=centering_delta_bins,
        centering_suspect_idx=centering_suspect_idx,
        dt=float(times[1] - times[0]),
        df=float(df),
    )
