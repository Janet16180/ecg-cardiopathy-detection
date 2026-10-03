"""Scientific invariants for frozen physiological wave delineation and truth matching."""

import numpy as np

from ecg_experiment.ludb_boundary_validation import record_summary, score_lead
from ecg_experiment.morphology_boundaries import (
    annotation_triplets,
    associate_waves,
    fixed_boundaries,
    interval_measurements,
    match_peaks,
    wave_support,
)


def test_frozen_offsets_exact() -> None:
    """Fixed P/QRS/T windows retain exactly the 042 sample offsets."""
    found = fixed_boundaries(np.array([500]))
    np.testing.assert_array_equal(found[0, 0], [[375, 470], [470, 540], [600, 725]])


def test_triplet_parser_audits_incomplete_events() -> None:
    """Unexpected annotation events are explicit and complete triplets survive."""
    found, irregular = annotation_triplets(np.array([10, 20, 30, 40]), ["(", "N", ")", "("])
    np.testing.assert_array_equal(found["QRS"], [[10, 20, 30]])
    assert len(irregular) == 1


def test_matching_maximizes_valid_cardinality_and_never_reuses() -> None:
    """Independent assignment keeps two valid nearby truth peaks distinct."""
    qrs = np.array([[0, 100, 120], [130, 160, 180]])
    matched = match_peaks(np.array([100, 190]), qrs)
    assert matched == {0: 0, 1: 1}
    assert len(set(matched.values())) == len(matched)


def test_excess_atrial_truth_does_not_reuse_one_slot() -> None:
    """Two annotated P waves competing for one beat leave one truth unmatched."""
    qrs = np.array([[450, 500, 550]])
    p = np.array([[280, 300, 320], [380, 400, 420]])
    found = associate_waves(p, qrs, "P")
    assert found == {1: 0}


def test_missing_truth_remains_in_record_average() -> None:
    """An unmatched annotated wave contributes zero IoU instead of disappearing."""
    row = {
        "wave": "P",
        "fixed": interval_measurements(np.array([-1, -1]), np.array([1, 2])),
        "adaptive": interval_measurements(np.array([1, 2]), np.array([1, 2])),
    }
    summary = record_summary([row])
    assert summary["primary"] == {"fixed": 0.0, "adaptive": 1.0}
    assert row["fixed"]["onset_mae_ms"] == 500


def test_evaluator_preserves_unmatched_qrs_wave() -> None:
    """No detected anchor never removes annotated QRS from either arm's denominator."""
    annotations = {"P": np.empty((0, 3), int), "QRS": np.array([[500, 520, 550]]), "T": np.empty((0, 3), int)}
    empty = np.empty((0, 12, 3, 2), int)
    measurements, _ = score_lead(np.array([], int), {"fixed": empty, "adaptive": empty}, annotations, 0)
    assert len(measurements) == 1
    assert measurements[0]["adaptive"]["iou"] == 0


def test_wave_support_polarity_and_amplitude_invariant() -> None:
    """Morphology supports do not shift when polarity or physical amplitude changes."""
    x = np.arange(500)
    signal = np.exp(-(((x - 200) / 25) ** 2))
    original = wave_support(signal, 100, 300, 0.001, 0.1, 500)
    transformed = wave_support(-3 * signal, 100, 300, 0.003, 0.1, 500)
    assert original == transformed
    assert original[0] < 200 < original[1]


def test_no_noise_only_support() -> None:
    """A constant low-amplitude search does not invent a physiological wave."""
    assert wave_support(np.ones(500), 100, 300, 0.001, 0.1, 500) == (-1, -1)


def test_record_weights_wave_classes_equally() -> None:
    """Many atrial waves cannot outweigh one ventricular wave in the primary record mean."""
    correct = interval_measurements(np.array([1, 2]), np.array([1, 2]))
    missing = interval_measurements(np.array([-1, -1]), np.array([1, 2]))
    rows = [{"wave": "P", "fixed": correct, "adaptive": correct} for _ in range(100)]
    rows.append({"wave": "QRS", "fixed": missing, "adaptive": missing})
    assert record_summary(rows)["primary"]["adaptive"] == 0.5


def test_matching_audit_is_json_serializable() -> None:
    """Counts retain native scalar types for durable experiment receipts."""
    import json

    annotations = {"P": np.empty((0, 3), int), "QRS": np.array([[500, 520, 550]]), "T": np.empty((0, 3), int)}
    peaks = np.array([520])
    fixed = fixed_boundaries(peaks)
    _, audit = score_lead(peaks, {"fixed": fixed, "adaptive": fixed}, annotations, 0)
    json.dumps(audit)


def test_recovered_peak_preserves_boundaries_without_reusing_events() -> None:
    """A peak just outside a valid span is recovered without altering the endpoints."""
    found, audit = annotation_triplets(
        np.array([965, 967, 996, 1089, 1132, 1159]), ["N", "(", ")", "(", "t", ")"]
    )
    np.testing.assert_array_equal(found["QRS"], [[967, 965, 996]])
    np.testing.assert_array_equal(found["T"], [[1089, 1132, 1159]])
    assert audit[0]["outside_samples"] == 2


def test_partial_interval_does_not_borrow_distant_following_peak() -> None:
    """Unknown boundaries cannot become truth by borrowing a neighboring typed peak."""
    found, audit = annotation_triplets(np.array([100, 120, 200, 220, 250]), ["(", ")", "N", "(", ")"])
    assert len(found["QRS"]) == 0
    assert len(audit) == 5
