"""Scientific invariants for sparse joint-channel QTDB evaluation."""

import numpy as np

from ecg_experiment.hybrid_boundaries059 import evaluate_joint, joint_hybrid, parse_annotations, summarize


def test_qrs_types_and_partial_t_preserve_unknown_onset():
    samples = np.array([10, 20, 30, 45, 70, 80, 90, 100])
    symbols = ["(", "A", ")", "t", ")", "(", "p", ")"]
    nums = np.array([1, 0, 1, 0, 2, 0, 0, 0])
    truth, audit = parse_annotations(samples, symbols, nums)
    assert truth["QRS"].tolist() == [[10, 20, 30]]
    assert truth["T"].tolist() == [[-1, 45, 70]]
    assert audit["complete"] == 2
    assert audit["offset_only"] == 1
    assert not audit["unknown"]


def test_malformed_events_are_unknown_not_borrowed():
    truth, audit = parse_annotations(np.array([10, 20, 30]), ["(", "p", ")"], np.array([1, 0, 1]))
    assert not len(truth["P"])
    assert len(audit["unknown"]) == 3


def test_unmatched_truth_and_partial_t_missing_are_penalized():
    annotations = {
        "QRS": np.array([[230, 250, 270]]),
        "P": np.empty((0, 3), int),
        "T": np.array([[-1, 330, 370]]),
    }
    predictions = {name: np.empty((0, 3, 2), int) for name in ("fixed", "hybrid")}
    rows, _ = evaluate_joint(np.array([], int), predictions, annotations)
    summary = summarize(rows)
    assert summary["classes"]["QRS"]["hybrid"]["iou"] == 0
    assert summary["t_offset"]["hybrid"] == 500
    assert "T" not in summary["classes"]


def test_annotation_slots_cannot_reuse_prediction():
    annotations = {
        "QRS": np.array([[230, 250, 270]]),
        "P": np.empty((0, 3), int),
        "T": np.array([[-1, 310, 340], [-1, 360, 390]]),
    }
    predictions = {name: np.array([[[185, 235], [230, 270], [300, 390]]]) for name in ("fixed", "hybrid")}
    rows, _ = evaluate_joint(np.array([250]), predictions, annotations)
    assert sum(row["anchor"] is not None for row in rows if row["wave"] == "T") == 1


def test_waveform_only_hybrid_keeps_p_exact_under_polarity_scaling():
    times = np.arange(3000) / 250
    waveform = sum(np.exp(-(((times - position) / 0.025) ** 2)) for position in np.arange(1.0, 11.0, 1.0))
    signal = np.array([waveform, -0.7 * waveform])
    peaks, predictions = joint_hybrid(signal)
    reverse_peaks, reverse = joint_hybrid(-3 * signal)
    assert len(peaks) > 5
    assert np.array_equal(predictions["fixed"][:, 0], predictions["hybrid"][:, 0])
    assert np.array_equal(peaks, reverse_peaks)
    assert np.array_equal(predictions["hybrid"], reverse["hybrid"])
