"""Empirical (block-bootstrap) noise generator.

Real bin-to-bin noise in the recorded power traces is not simply
independent Gaussian noise: it carries whatever correlation structure the
STFT and detector chain impose. Rather than assume a noise *shape*, this
generator resamples real, normalized ground-state noise segments (in
random order and orientation) and stitches them end to end, which
preserves that structure by construction while still letting each
simulated file draw an independent noise realization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class EmpiricalNoiseLibrary:
    """Zero-mean, unit-variance ground-state noise segments for bootstrapping.

    Attributes
    ----------
    blocks : ndarray, shape (n_blocks, tail_bins)
        Normalized noise segments (mean-subtracted, jointly rescaled to
        unit standard deviation).
    """

    blocks: npt.NDArray[np.float64]

    def draw(self, length: int, rng: np.random.Generator) -> npt.NDArray[np.float64]:
        """Draw a length-``length`` unit-scale noise trace.

        Concatenates randomly chosen blocks (each independently,
        randomly time-reversed) until at least ``length`` samples are
        available, then truncates to exactly ``length``.

        Parameters
        ----------
        length : int
        rng : numpy.random.Generator
            Explicit generator; no hidden global RNG state.

        Returns
        -------
        ndarray, shape (length,)
        """
        pieces = []
        total = 0
        while total < int(length):
            block = self.blocks[int(rng.integers(0, len(self.blocks)))]
            if rng.random() < 0.5:
                block = block[::-1]
            pieces.append(block)
            total += len(block)
        return np.concatenate(pieces)[: int(length)]


def build_empirical_noise_library(
    y_files_e: npt.ArrayLike, ground_noise_idx: npt.ArrayLike, *, tail_bins: int = 15
) -> EmpiricalNoiseLibrary:
    """Build a normalized noise-block library from real ground-state terminal segments.

    Parameters
    ----------
    y_files_e : array_like, shape (n_files, n_time_bins)
        Background-subtracted, noise-normalized power traces.
    ground_noise_idx : array_like of int
        Indices of files whose terminal segment is a stable ground-state
        noise reference (e.g.
        ``GroundNoiseNormalization.ground_noise_idx``).
    tail_bins : int, default 15
        Number of terminal time bins used per file (clipped to the number
        of time bins available).

    Returns
    -------
    EmpiricalNoiseLibrary

    Raises
    ------
    RuntimeError
        If the resulting noise scale is not finite and positive (e.g. too
        few reference files, or degenerate data).
    """
    y_files_e = np.asarray(y_files_e, dtype=float)
    tail_bins = min(tail_bins, y_files_e.shape[1])
    blocks = np.asarray(y_files_e[np.asarray(ground_noise_idx, dtype=int), -tail_bins:], dtype=float)
    blocks = blocks - blocks.mean(axis=1, keepdims=True)
    scale = float(np.std(blocks, ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("The empirical ground-state noise library is invalid.")
    return EmpiricalNoiseLibrary(blocks=blocks / scale)
