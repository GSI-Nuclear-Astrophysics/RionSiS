"""Extract terminal 1->0 decay events from a staircase fit.

A file contributes a decay-time observation only if its very last
segment (latest time) is the ground state (``N=0``) and the segment just
before it is ``N=1``: exactly one ion was present, and it decayed once,
cleanly, before the recording ended.

``rionsis.levels.staircase`` fits the time-*reversed* signal, so the
first segment in ``states``/``lengths`` is the terminal (latest-time)
segment: ``state[0] == 0 and state[1] == 1`` means "the file ends in the
ground state, preceded by exactly one ion".
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

#: Structured dtype of the returned event table: file id, dwell time in
#: the N=1 state (``k``), dwell time in the terminal N=0 state (``g``),
#: and their sum, the total window available for truncation (``H``).
EVENT_DTYPE = [("file", "i8"), ("k", "i8"), ("g", "i8"), ("H", "i8")]


def extract_terminal_events(
    segment_lengths: list[npt.ArrayLike],
    states: list[npt.ArrayLike],
    file_ids: npt.ArrayLike,
    n_time: int,
) -> npt.NDArray:
    """Extract, at most, one terminal 1->0 decay event per file.

    Parameters
    ----------
    segment_lengths, states : list of array_like
        Per-file staircase segments in time-reversed order, as returned
        by ``rionsis.levels.staircase.fit_integer_dataset``
        (``.segment_lengths``, ``.states``).
    file_ids : array_like of int
        Identifier reported for each file in the output table.
    n_time : int
        Number of time bins per file; used to check that the segments
        fully cover the recorded window.

    Returns
    -------
    ndarray
        Structured array with dtype ``EVENT_DTYPE``, one row per file
        that ends in an unambiguous terminal 1->0 decay.

    Raises
    ------
    RuntimeError
        If a file's segments are malformed, do not cover the full time
        window, or are not strictly monotonic in occupancy.
    """
    rows = []
    for length, state, file_id in zip(segment_lengths, states, file_ids):
        length = np.asarray(length, dtype=int)
        state = np.asarray(state, dtype=int)
        if len(length) != len(state) or len(length) == 0 or np.any(length < 1):
            raise RuntimeError(f"Invalid staircase path in file {int(file_id)}.")
        if int(length.sum()) != int(n_time):
            raise RuntimeError(f"Paths in file {int(file_id)} do not cover the time window.")
        if len(state) > 1 and np.any(np.diff(state) <= 0):
            raise RuntimeError(f"Non-strictly-monotonic integer path in file {int(file_id)}.")
        if len(state) >= 2 and state[0] == 0 and state[1] == 1:
            g = int(length[0])
            k = int(length[1])
            rows.append((int(file_id), k, g, k + g))
    return np.asarray(rows, dtype=EVENT_DTYPE)
