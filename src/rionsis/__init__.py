"""RionSiS: single-ion level detection, half-life measurement, and Monte Carlo validation.

Subpackages: :mod:`rionsis.io` (raw IQ loading), :mod:`rionsis.levels`
(IQ traces -> exact integer occupancy staircase fit),
:mod:`rionsis.halflife` (staircase fit -> half-life measurement),
:mod:`rionsis.simulation` (Monte Carlo validation), :mod:`rionsis.physics`
(shared decay-law/noise-model relations), :mod:`rionsis.plotting` and
:mod:`rionsis.validation` (presentation/QA helpers).
"""

from rionsis._version import __version__

__all__ = ["__version__"]
