"""RionSiS: single-ion level detection, half-life measurement, and Monte Carlo validation.

Three independently usable subpackages:

* :mod:`rionsis.levels` -- from raw IQ traces to an exact integer
  occupancy staircase fit (STFT, band integration, single-ion power
  calibration, occupancy-noise calibration, the staircase fit itself).
* :mod:`rionsis.halflife` -- from a staircase fit to a half-life
  measurement (terminal-decay event extraction, doubly truncated
  geometric MLE, cut-grid scan, moving-block bootstrap).
* :mod:`rionsis.simulation` -- a pure end-to-end Monte Carlo generator
  with a known, injected half-life, used to validate the ``levels`` +
  ``halflife`` pipeline by feeding its output back through them.

:mod:`rionsis.physics` holds the handful of relations genuinely shared
between those three (the decay law, the occupancy-noise functional form).
:mod:`rionsis.io` loads and validates raw IQ recordings.
:mod:`rionsis.plotting` (optional ``matplotlib`` extra) and
:mod:`rionsis.validation` are presentation/QA helpers, not core physics.
"""

from rionsis._version import __version__

__all__ = ["__version__"]
