"""Pure end-to-end decay Monte Carlo simulation.

Generates synthetic power-vs-time traces with a known, injected half-life,
using the *same* single-ion power ``A1`` and occupancy-noise model
``sigma(N)`` that were calibrated from real data (``rionsis.levels``).
Feeding the output back through ``rionsis.levels.fit_integer_dataset``
and ``rionsis.halflife`` with the identical configuration is what makes
this an honest end-to-end plausibility check of the staircase fit and
half-life estimator, not merely a synthetic-data generator.

This package has no import-time dependency on ``rionsis.levels`` or
``rionsis.halflife``: it consumes plain arrays and callables (a noise
model callable, an empirical noise library built from plain arrays), so
it can be used and tested completely standalone.
"""

from rionsis.simulation.decay_simulation import SimulationResult, simulate_decay_traces
from rionsis.simulation.noise_bootstrap import EmpiricalNoiseLibrary, build_empirical_noise_library
from rionsis.simulation.window_response import WindowResponse, hann_squared_window_response, window_decay_response

__all__ = [
    "SimulationResult",
    "simulate_decay_traces",
    "EmpiricalNoiseLibrary",
    "build_empirical_noise_library",
    "WindowResponse",
    "hann_squared_window_response",
    "window_decay_response",
]
