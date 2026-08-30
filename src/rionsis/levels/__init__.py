"""Single-ion level detection: from raw IQ traces to an integer occupancy staircase.

The pipeline, in order, is::

    IQ traces (rionsis.io)
      -> compute_power_spectrogram      (STFT + ground-state centering)
      -> integrate_bands / normalize_ground_noise
      -> calibrate_a1                    (independent single-ion power calibration)
      -> calibrate_noise_vs_occupancy    (sigma(N) from real plateaus)
      -> fit_integer_dataset             (exact monotonic integer staircase fit)

Each stage is a plain function operating on explicit arrays: no notebook
globals are read implicitly, so any stage can be used, tested, or
replaced independently.
"""

from rionsis.levels.bands import BandIntegration, GroundNoiseNormalization, integrate_bands, normalize_ground_noise
from rionsis.levels.calibration import A1Calibration, calibrate_a1
from rionsis.levels.noise_calibration import NoiseCalibration, calibrate_noise_vs_occupancy
from rionsis.levels.spectral import SpectrogramResult, compute_power_spectrogram
from rionsis.levels.staircase import (
    StaircaseFitConfig,
    StaircaseFitResult,
    default_change_penalty,
    exact_integer_staircase,
    fit_integer_dataset,
)

__all__ = [
    "BandIntegration",
    "GroundNoiseNormalization",
    "integrate_bands",
    "normalize_ground_noise",
    "A1Calibration",
    "calibrate_a1",
    "NoiseCalibration",
    "calibrate_noise_vs_occupancy",
    "SpectrogramResult",
    "compute_power_spectrogram",
    "StaircaseFitConfig",
    "StaircaseFitResult",
    "default_change_penalty",
    "exact_integer_staircase",
    "fit_integer_dataset",
]
