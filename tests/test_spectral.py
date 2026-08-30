"""Tests for the STFT power spectrogram and ground-state centering."""

from __future__ import annotations

import numpy as np
import pytest

from rionsis.levels.spectral import compute_power_spectrogram


def _pure_tone_iq(n_files, n_samples, fs, tone_hz):
    t = np.arange(n_samples) / fs
    tone = np.exp(2j * np.pi * tone_hz * t)
    return np.tile(tone, (n_files, 1)).astype(np.complex64)


def test_compute_power_spectrogram_shapes():
    fs = 50_000.0
    nperseg = 256
    n_samples = 4096
    iq = _pure_tone_iq(n_files=6, n_samples=n_samples, fs=fs, tone_hz=1000.0)
    result = compute_power_spectrogram(iq, fs, dt_bin=0.005, nperseg=nperseg)
    assert result.power.shape[0] == 6
    assert result.power.shape[1] == nperseg
    assert result.dt == pytest.approx(0.005, abs=1e-3)
    assert result.df == pytest.approx(fs / nperseg)


def test_compute_power_spectrogram_centers_the_dominant_tone():
    fs = 50_000.0
    nperseg = 256
    n_samples = 4096
    iq = _pure_tone_iq(n_files=4, n_samples=n_samples, fs=fs, tone_hz=1000.0)
    result = compute_power_spectrogram(iq, fs, dt_bin=0.005, nperseg=nperseg)
    center = result.power.shape[1] // 2
    # After centering, the dominant frequency bin (0 Hz relative) should
    # carry (close to) the maximum summed power for every file.
    peak_bins = np.argmax(result.power.sum(axis=2), axis=1)
    np.testing.assert_array_equal(peak_bins, center)
    assert len(result.centering_suspect_idx) == 0


def test_compute_power_spectrogram_centers_correctly_with_few_files():
    # Regression test for a scipy.ndimage.median_filter edge case: with
    # size=11 and fewer than 5 files, the unclipped call degenerates to an
    # all-zero smoothed bin index, centering every spectrogram on bin 0
    # instead of the ground-state bin. See the comment in
    # rionsis.levels.spectral.compute_power_spectrogram.
    fs = 50_000.0
    nperseg = 256
    n_samples = 4096
    for n_files in (1, 2, 3, 4):
        iq = _pure_tone_iq(n_files=n_files, n_samples=n_samples, fs=fs, tone_hz=1000.0)
        result = compute_power_spectrogram(iq, fs, dt_bin=0.005, nperseg=nperseg)
        center = result.power.shape[1] // 2
        peak_bins = np.argmax(result.power.sum(axis=2), axis=1)
        np.testing.assert_array_equal(peak_bins, center)


def test_compute_power_spectrogram_rejects_too_small_nperseg():
    fs = 50_000.0
    iq = _pure_tone_iq(n_files=2, n_samples=1000, fs=fs, tone_hz=100.0)
    with pytest.raises(ValueError):
        compute_power_spectrogram(iq, fs, dt_bin=1.0, nperseg=10)
