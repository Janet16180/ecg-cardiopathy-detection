"""Checks of CODE-15 trimming, resampling, scoring and duplicate resolution."""

import numpy as np
import pandas as pd

from ecg_experiment.code15_clean import duplicate_status, exclusion_reasons, score_tracing, window_500hz


def tracing(active: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    stored = np.zeros((4096, 12), dtype=np.float32)
    left = (4096 - active) // 2
    stored[left:left + active] = rng.normal(scale=0.3, size=(active, 12))
    return stored


def test_window_is_ten_seconds_at_500_hz():
    window = window_500hz(tracing(4096)[:4096])
    assert window.shape == (12, 5000)
    assert window.dtype == np.float32


def test_short_exams_are_trimmed_but_not_scored():
    row = score_tracing(tracing(2934))
    assert (row["edge_zero_left"], row["active_samples"]) == (581, 2934)
    assert "window_sha256" not in row


def test_ten_second_exams_are_scored_at_both_scales():
    row = score_tracing(tracing(4000))
    assert row["active_samples"] == 4000
    assert {"exclusion_stored", "exclusion_halved", "range_II"} <= set(row)


def test_reasons_are_the_union_of_both_scales():
    frame = pd.DataFrame({"exclusion_stored": ["extreme_amplitude", "", np.nan],
                          "exclusion_halved": ["", "near_flat_record", np.nan]})
    assert exclusion_reasons(frame).tolist() == ["extreme_amplitude", "near_flat_record", ""]


def test_copies_are_kept_once_per_patient_and_dropped_across_patients():
    frame = pd.DataFrame({"native_sha256": ["a", "a", "b", "b", "c"], "patient_id": [1, 1, 2, 3, 4]},
                         index=[20, 10, 30, 40, 50])
    assert duplicate_status(frame).to_dict() == {20: "dropped_copy", 10: "kept", 30: "dropped_conflict",
                                                 40: "dropped_conflict", 50: "unique"}
