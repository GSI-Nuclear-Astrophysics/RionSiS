"""Loading and validation of raw complex-IQ recordings.

Each recording is a Schottky-type complex IQ time trace for one file
("shot"). Expected on-disk format: a NumPy object array (``np.save(...,
allow_pickle=True)``) of ``dict`` entries with keys ``filename``, ``data``
(complex IQ samples), and ``fs`` (sampling rate in Hz). All files in a
data set must share the same sampling rate and number of samples.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

_TIMESTAMP_RE = re.compile(r"(\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{3})")


@dataclass(frozen=True)
class IQDataset:
    """A validated, time-ordered collection of equal-length IQ recordings.

    Attributes
    ----------
    iq : ndarray, shape (n_files, n_samples), complex64
        Complex IQ samples for every file.
    fs : float
        Common sampling rate in Hz.
    names : ndarray of str
        Base file names (``os.path.basename``), in load/sort order.
    source_files : list of str
        Full source paths, in the same order as ``iq``.
    timestamps : list of str
        Extracted timestamp strings (``YYYY.MM.DD.HH.MM.SS.mmm``), or the
        empty string where a file name carried no recognizable timestamp.
    """

    iq: npt.NDArray[np.complex64]
    fs: float
    names: npt.NDArray[np.str_]
    source_files: list[str]
    timestamps: list[str]

    @property
    def n_files(self) -> int:
        return int(self.iq.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.iq.shape[1])

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.fs


def _extract_timestamp(path: str) -> str:
    match = _TIMESTAMP_RE.search(str(path))
    return match.group(1) if match else ""


def load_iq_dataset(path: str | Path) -> IQDataset:
    """Load, validate, and time-sort a set of IQ recordings.

    Parameters
    ----------
    path : str or Path
        Path to a ``.npy`` file containing an object array of
        ``{'filename', 'data', 'fs'}`` dictionaries.

    Returns
    -------
    IQDataset

    Raises
    ------
    ValueError
        If the file is empty, entries are missing required fields, the
        sampling rates differ across files, or the number of IQ samples
        differs across files.
    """
    raw = np.load(path, allow_pickle=True)
    if len(raw) == 0:
        raise ValueError(f"{path} contains no files.")
    required = {"filename", "data", "fs"}
    if not all(isinstance(r, dict) and required <= set(r) for r in raw):
        raise ValueError("Expected the fields for each entry: 'filename', 'data', and 'fs'.")

    fs_values = np.asarray([float(r["fs"]) for r in raw])
    sample_counts = np.asarray([len(r["data"]) for r in raw], dtype=int)
    if not np.allclose(fs_values, fs_values[0]):
        raise ValueError("The sampling rates of the files are not identical.")
    if np.any(sample_counts != sample_counts[0]):
        raise ValueError("The number of IQ samples is not identical in all files.")

    fs = float(fs_values[0])
    iq_all = np.stack([np.asarray(r["data"], dtype=np.complex64) for r in raw])
    source_files = [str(r["filename"]) for r in raw]
    names = np.asarray([os.path.basename(p) for p in source_files])
    timestamps = [_extract_timestamp(p) for p in source_files]

    if all(timestamps) and timestamps != sorted(timestamps):
        order = np.argsort(timestamps)
        iq_all = iq_all[order]
        names = names[order]
        source_files = [source_files[i] for i in order]
        timestamps = [timestamps[i] for i in order]

    return IQDataset(iq=iq_all, fs=fs, names=names, source_files=source_files, timestamps=timestamps)
