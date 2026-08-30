"""Pure end-to-end decay Monte Carlo simulation.

Generates synthetic power-vs-time traces with a known, injected
half-life, using the single-ion power ``A1`` and occupancy-noise model
``sigma(N)`` calibrated from real data (``rionsis.levels``). Has no
import-time dependency on ``rionsis.levels``/``rionsis.halflife``: it
consumes plain arrays and callables, so it can be used standalone.
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
