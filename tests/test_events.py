"""Tests for rionsis.halflife.events: terminal 1->0 decay-event extraction."""

from __future__ import annotations

import numpy as np
import pytest

from rionsis.halflife.events import extract_terminal_events


def test_extracts_terminal_event_when_file_ends_in_ground_after_one_ion():
    # Time-reversed order: state 0 (ground) for 12 bins, then state 1 for 8 bins,
    # then state 3 for 5 bins (older history, irrelevant to the terminal event).
    lengths = [np.array([12, 8, 5])]
    states = [np.array([0, 1, 3])]
    events = extract_terminal_events(lengths, states, file_ids=[42], n_time=25)
    assert len(events) == 1
    assert events["file"][0] == 42
    assert events["g"][0] == 12
    assert events["k"][0] == 8
    assert events["H"][0] == 20


def test_no_event_when_file_never_reaches_ground():
    lengths = [np.array([10, 10])]
    states = [np.array([1, 2])]
    events = extract_terminal_events(lengths, states, file_ids=[1], n_time=20)
    assert len(events) == 0


def test_no_event_when_second_segment_is_not_state_one():
    # Ends in ground state, but the state right before it was N=2, not N=1:
    # ambiguous (two ions could have decayed near-simultaneously), so excluded.
    lengths = [np.array([10, 10])]
    states = [np.array([0, 2])]
    events = extract_terminal_events(lengths, states, file_ids=[1], n_time=20)
    assert len(events) == 0


def test_single_segment_file_never_contributes_an_event():
    lengths = [np.array([20])]
    states = [np.array([0])]
    events = extract_terminal_events(lengths, states, file_ids=[1], n_time=20)
    assert len(events) == 0


def test_multiple_files_mixed():
    lengths = [np.array([12, 8]), np.array([20]), np.array([5, 5, 10])]
    states = [np.array([0, 1]), np.array([2]), np.array([0, 1, 4])]
    events = extract_terminal_events(lengths, states, file_ids=[0, 1, 2], n_time=20)
    assert sorted(events["file"].tolist()) == [0, 2]


def test_rejects_paths_not_covering_full_window():
    lengths = [np.array([5, 5])]
    states = [np.array([0, 1])]
    with pytest.raises(RuntimeError):
        extract_terminal_events(lengths, states, file_ids=[1], n_time=100)


def test_rejects_non_monotonic_states():
    lengths = [np.array([5, 5, 5])]
    states = [np.array([0, 2, 1])]
    with pytest.raises(RuntimeError):
        extract_terminal_events(lengths, states, file_ids=[1], n_time=15)
