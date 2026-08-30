"""Shared physical relations and domain models used across RionSiS.

Only relations that are genuinely used by more than one domain
(``simulation``, ``levels``, ``halflife``) live here. Anything specific to
a single domain stays in that domain's package.
"""

from rionsis.physics.decay_law import halflife_to_rate, rate_to_halflife
from rionsis.physics.noise_model import sigma_of_occupancy

__all__ = ["halflife_to_rate", "rate_to_halflife", "sigma_of_occupancy"]
