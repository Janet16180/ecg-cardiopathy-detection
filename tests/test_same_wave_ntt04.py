"""Reservation, physical geometry and fixed prototype contracts for NTT-04."""

import json
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from ecg_experiment.clustering01 import train_patient
from ecg_experiment.eda.signals import LEADS
from ecg_experiment.same_wave_ntt04 import (
    FINDINGS,
    SEEDS,
    allowed_identities,
    classify,
    exact_gain,
    finding,
    labeled_cohort,
    physical_median,
    raw_descriptors,
    render_median,
    select_patients,
    support_pool,
)


def test_complete_bbb_threshold_and_conflict_preserve_strict_normal_definition():
    classes = {"NORM": "NORM", "CRBBB": "CD", "CLBBB": "CD"}
    assert finding({"CRBBB": 50}, classes)[0] == "CRBBB"
    assert finding({"CLBBB": 100}, classes)[0] == "CLBBB"
    assert finding({"CRBBB": 49}, classes)[0] is None
    assert finding({"CRBBB": 100, "CLBBB": 0}, classes)[1] == "both_complete_bbb_codes"
    assert finding({"NORM": 100, "SR": 100}, classes)[0] == "NORM"
    assert finding({"NORM": 100, "SR": 100, "IRBBB": 100}, classes)[0] is None


def test_mixed_fold_and_reserved_patients_are_removed_from_identity_table():
    train = next(p for p in range(1, 100) if train_patient(p))
    reserved = next(p for p in range(1, 100) if not train_patient(p))
    frame = pd.DataFrame(
        {
            "ecg_id": [1, 2, 3],
            "patient_id": [train, train, reserved],
            "strat_fold": [1, 10, 1],
            "filename_hr": ["unused"] * 3,
        }
    )
    assert allowed_identities(frame).empty


def test_nonselected_diagnostic_payload_is_never_parsed(tmp_path):
    pd.DataFrame({"diagnostic": [1], "diagnostic_class": ["NORM"]}, index=["NORM"]).to_csv(
        tmp_path / "scp_statements.csv"
    )
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("ecg_id,scp_codes\n1,\"{'NORM': 100}\"\n999,poison_must_not_parse\n")
    identities = pd.DataFrame({"ecg_id": [1], "patient_id": [1], "filename_hr": ["allowed"]})
    rows, _ = labeled_cohort(identities, metadata)
    assert len(rows) == 1
    assert rows[0]["finding"] == "NORM"


def test_support_targets_are_disjoint_in_every_repeated_draw():
    rows = [{"patient_id": p, "ecg_id": p, "finding": FINDINGS[p % 3]} for p in range(1, 601)]
    targets, draws, _ = select_patients(rows)
    assert len(targets) == 60
    assert {label: sum(r["finding"] == label for r in targets) for label in FINDINGS} == dict.fromkeys(
        FINDINGS, 20
    )
    target_ids = {row["patient_id"] for row in targets}
    assert all(not support_pool(patient) for patient in target_ids)
    for draw in draws:
        assert len(draw["supports"]) == 3
        assert not target_ids & {r["patient_id"] for r in draw["supports"]}
        assert all(support_pool(r["patient_id"]) for r in draw["supports"])
    assert select_patients(list(reversed(rows))) == (targets, draws, select_patients(rows)[2])


def test_insufficient_finding_pool_stops_without_replacement():
    with pytest.raises(ValueError, match="Insufficient"):
        select_patients([{"patient_id": 1, "ecg_id": 1, "finding": "NORM"}])


def test_fixed_baseline_uses_exact_samples_10_through_49(monkeypatch):
    signal = np.zeros((5000, 12))
    beat = np.tile(np.arange(475)[:, None], (1, 12)).astype(float)
    monkeypatch.setattr("ecg_experiment.same_wave_ntt04.qrs_peaks", lambda *_: np.array([500, 1500, 2500]))
    monkeypatch.setattr("ecg_experiment.same_wave_ntt04.median_beat", lambda *_: (beat, 1.0))
    monkeypatch.setattr("ecg_experiment.same_wave_ntt04.qrs_bounds", lambda *_: (100, 200))
    corrected, quality = physical_median(signal)
    np.testing.assert_allclose(corrected, beat - 29.5)
    np.testing.assert_allclose(corrected[10:50].mean(axis=0), 0)
    assert quality["complete_beats"] == 3
    assert quality["qrs_duration_seconds"] == 0.2


def test_fewer_than_three_complete_beats_stops(monkeypatch):
    monkeypatch.setattr("ecg_experiment.same_wave_ntt04.qrs_peaks", lambda *_: np.array([50, 500, 1500]))
    with pytest.raises(ValueError, match="three complete"):
        physical_median(np.zeros((5000, 12)))


def test_raw_representations_have_the_frozen_qrs_and_full_context_dimensions():
    beat = np.zeros((475, 12))
    beat[:, LEADS.index("V1")] = np.linspace(-1, 1, 475)
    beat[:, LEADS.index("V6")] = np.linspace(2, -1, 475)
    qrs, full, scalars = raw_descriptors(beat, 0.12)
    assert qrs.shape == (302,)
    assert full.shape == (950,)
    assert scalars.shape == (5,)
    np.testing.assert_allclose(np.linalg.norm(qrs), 1)
    np.testing.assert_allclose(np.linalg.norm(full), 1)
    assert scalars[0] == np.log(0.12)


def test_rendered_rois_have_equal_programmed_width_without_answer_input():
    rgb, qrs, late, coordinates = render_median(np.zeros((475, 12)))
    assert rgb.shape == (518, 518, 3)
    assert qrs.sum() > 0
    assert late.sum() > 0
    qrs_columns = np.flatnonzero(qrs.any(axis=0))
    late_columns = np.flatnonzero(late.any(axis=0))
    assert abs(len(qrs_columns) - len(late_columns)) <= 1
    assert not set(qrs_columns) & set(late_columns)
    assert coordinates["qrs_roi"] == [-0.15, 0.15]
    assert coordinates["late_roi"] == [0.25, 0.55]


def test_prototype_classes_and_cyclic_control_follow_only_support_labels():
    features, targets, draws = {}, [], []
    for index, label in enumerate(FINDINGS):
        vector = np.eye(3)[index]
        references = []
        for j in range(3):
            ecg = index * 10 + j + 1
            references.append({"ecg_id": ecg})
            features[ecg] = {
                "dino": vector,
                "late": -vector,
                "raw": vector,
                "raw_full_context": vector,
                "scalars": np.full(5, index),
            }
        ecg = 100 + index
        features[ecg] = features[references[0]["ecg_id"]]
        targets.append({"patient_id": ecg, "ecg_id": ecg, "finding": label})
        draws.extend({"seed": seed, "finding": label, "supports": references} for seed in SEEDS)
    predictions = classify(features, targets, draws)
    assert len(predictions) == 9
    assert all(row["correct"]["dino"] == row["correct"]["raw"] == 1 for row in predictions)
    assert all(row["correct"]["shuffled_labels"] == 0 for row in predictions)
    assert all(
        row["chosen"][method] == int(np.argmax(scores))
        for row in predictions
        for method, scores in row["scores"].items()
    )


def test_exact_five_point_boundary_does_not_fail_due_to_float_subtraction():
    rows = []
    for label in FINDINGS:
        rows.extend(
            {"finding": label, "correct": {"dino": int(i < 10), "raw": int(i < 9)}} for i in range(20)
        )
    assert 0.5 - 0.45 < 0.05
    assert exact_gain(rows, "dino", "raw") == Fraction(1, 20)


def test_failure_receipt_keeps_record_identity_without_substitution(tmp_path, monkeypatch):
    from scripts.reports.same_wave_ntt04 import extract_physical

    monkeypatch.setattr("scripts.reports.same_wave_ntt04.sha256_file", lambda _: "fake_hash")
    monkeypatch.setattr(
        "scripts.reports.same_wave_ntt04.wfdb.rdsamp",
        lambda _: (np.zeros((1000, 12)), {"fs": 100, "units": ["mV"] * 12}),
    )
    with pytest.raises(ValueError, match="500 Hz"):
        extract_physical([{"ecg_id": 123, "filename_hr": "unused"}], tmp_path, float("inf"))
    saved = json.loads((tmp_path / "quality_failure_local.json").read_text())
    assert saved["ecg_id"] == 123
    assert saved["replacement"] is False
