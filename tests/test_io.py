"""Tests for rionsis.io.iq_loader: raw IQ dataset loading and validation."""

from __future__ import annotations

import numpy as np
import pytest

from rionsis.io.iq_loader import load_iq_dataset


def _write_dataset(tmp_path, entries):
    path = tmp_path / "dataset.npy"
    arr = np.empty(len(entries), dtype=object)
    for i, entry in enumerate(entries):
        arr[i] = entry
    np.save(path, arr, allow_pickle=True)
    return path


def test_load_iq_dataset_round_trip(tmp_path):
    fs = 50_000.0
    entries = [
        {"filename": "b_2024.01.01.00.00.02.000.npy", "data": np.zeros(10, dtype=complex), "fs": fs},
        {"filename": "a_2024.01.01.00.00.01.000.npy", "data": np.ones(10, dtype=complex), "fs": fs},
    ]
    path = _write_dataset(tmp_path, entries)
    dataset = load_iq_dataset(path)
    assert dataset.n_files == 2
    assert dataset.n_samples == 10
    assert dataset.fs == fs
    # Sorted by embedded timestamp, not by original list order.
    assert dataset.timestamps == sorted(dataset.timestamps)
    np.testing.assert_array_equal(dataset.iq[0], np.ones(10))


def test_load_iq_dataset_rejects_empty(tmp_path):
    path = _write_dataset(tmp_path, [])
    with pytest.raises(ValueError):
        load_iq_dataset(path)


def test_load_iq_dataset_rejects_missing_field(tmp_path):
    entries = [{"filename": "a", "data": np.zeros(4, dtype=complex)}]  # missing 'fs'
    path = _write_dataset(tmp_path, entries)
    with pytest.raises(ValueError):
        load_iq_dataset(path)


def test_load_iq_dataset_rejects_mismatched_sampling_rate(tmp_path):
    entries = [
        {"filename": "a", "data": np.zeros(4, dtype=complex), "fs": 1000.0},
        {"filename": "b", "data": np.zeros(4, dtype=complex), "fs": 2000.0},
    ]
    path = _write_dataset(tmp_path, entries)
    with pytest.raises(ValueError):
        load_iq_dataset(path)


def test_load_iq_dataset_rejects_mismatched_sample_count(tmp_path):
    entries = [
        {"filename": "a", "data": np.zeros(4, dtype=complex), "fs": 1000.0},
        {"filename": "b", "data": np.zeros(8, dtype=complex), "fs": 1000.0},
    ]
    path = _write_dataset(tmp_path, entries)
    with pytest.raises(ValueError):
        load_iq_dataset(path)
