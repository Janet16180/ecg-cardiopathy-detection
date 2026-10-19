"""Artificial checks for raw extraction and frozen external transfer invariants."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecg_experiment import ROOT, sph
from ecg_experiment.measurement_extractor042 import (
    FEATURE_NAMES,
    FS,
    INDEPENDENT,
    LEADS,
    PLAUSIBLE,
    POST,
    PRE,
    SOS_CLEAN,
    SOS_QRS,
    extract_array,
    median_beat,
)
from ecg_experiment.measurement_screening042 import (
    evaluation_rows,
    extract_record,
    feature_matrix,
    notebook_extractor_hash,
    patient_role,
    serialized_features,
)


def artificial_ecg() -> np.ndarray:
    t = np.arange(5000) / FS
    base = 0.03 * np.sin(2 * np.pi * t)
    for peak in np.arange(0.8, 9.3, 0.8):
        base += np.exp(-(((t - peak) / 0.015) ** 2))
        base -= 0.2 * np.exp(-(((t - peak + 0.025) / 0.01) ** 2))
        base += 0.2 * np.exp(-(((t - peak - 0.25) / 0.055) ** 2))
    return np.column_stack([base * (1 + lead / 12) for lead in range(12)])


def original_functions() -> dict:
    from scipy import stats
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks, sosfiltfilt, welch

    cells = json.loads((ROOT / "notebooks/Avance2.8.ipynb").read_text())["cells"]
    text = "\n".join("".join(cells[i]["source"]) for i in (15, 20))
    names = {"detect_r_peaks", "median_beat", "delineate", "spectral_features", "extract_features"}
    nodes = [
        node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {
        "np": np,
        "stats": stats,
        "uniform_filter1d": uniform_filter1d,
        "Path": Path,
        "find_peaks": find_peaks,
        "sosfiltfilt": sosfiltfilt,
        "welch": welch,
        "FS": FS,
        "LEADS": LEADS,
        "INDEPENDENT": INDEPENDENT,
        "PLAUSIBLE": PLAUSIBLE,
        "PRE": PRE,
        "POST": POST,
        "SOS_CLEAN": SOS_CLEAN,
        "SOS_QRS": SOS_QRS,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "original_notebook", "exec"), namespace)
    return namespace


def test_original_source_binding():
    assert notebook_extractor_hash(ROOT / "notebooks/Avance2.8.ipynb").startswith("ee39c8f6f7")


def test_synthetic_exact_original_arithmetic():
    raw = artificial_ecg()
    original = original_functions()
    original["read_ecg"] = lambda path: raw
    expected = original["extract_features"](Path("unused"))
    actual = extract_array(raw)
    assert list(actual) == list(expected)
    for key in actual:
        if isinstance(actual[key], str):
            assert actual[key] == expected[key]
        else:
            np.testing.assert_equal(actual[key], expected[key])
    assert [key for key in actual if key not in {"status", "implausible", "flat_leads"}] == FEATURE_NAMES
    assert len(FEATURE_NAMES) == 130


def test_flat_extraction_retained_and_missing(monkeypatch):
    window = np.zeros((12, 5000), dtype=np.float32)
    monkeypatch.setattr(sph, "read_window", lambda ecg_id: window)
    with pytest.warns(RuntimeWarning):
        record = extract_record("artificial")
    assert record["status"] == "derivacion_plana"
    assert record["flat_leads"] == 12
    assert record["quality_fail"]
    assert all(np.isnan(record[name]) for name in FEATURE_NAMES)
    assert len(record["signal_sha256"]) == 64


def test_no_beat_template_is_missing():
    template, similarity = median_beat(np.zeros((5000, 12)), np.array([250]))
    assert np.isnan(template).all()
    assert np.isnan(similarity)


@pytest.mark.parametrize(
    "raw", [np.zeros((12, 5000)), np.zeros((5000, 12), dtype=np.float32), np.full((5000, 12), np.nan)]
)
def test_invalid_array_raises(raw):
    with pytest.raises(ValueError, match="finite float64"):
        extract_array(raw)


def test_feature_order_and_nonfinite_mapping():
    columns = [f"x{i}" for i in range(130)]
    frame = pd.DataFrame([np.arange(130, dtype=float)], columns=columns[::-1])
    frame.loc[0, "x10"] = np.inf
    result = feature_matrix(frame, columns)
    assert result.columns.to_list() == columns
    assert result.loc[0, "x0"] == 129
    assert np.isnan(result.loc[0, "x10"])
    with pytest.raises(ValueError, match="130 distinct"):
        feature_matrix(frame, columns[:-1])


def test_six_digit_serialization():
    frame = pd.DataFrame({"measurement": [1.23456789, np.nan, 0.0000123456789]})
    expected = np.array([1.23457, np.nan, 0.0000123457])
    np.testing.assert_equal(serialized_features(frame).measurement.to_numpy(), expected)


def patient_manifest() -> pd.DataFrame:
    patients = [str(i) for i in range(1200)]
    return pd.DataFrame(
        {
            "patient_id": patients,
            "role": [patient_role(p) for p in patients],
            "primary": [i % 2 for i in range(1200)],
            "use_evaluation": True,
            "quality_fail": True,
        }
    )


def test_patient_roles_disjoint_undefined_and_duplicates_excluded():
    frame = patient_manifest()
    frame.loc[0, "primary"] = np.nan
    frame.loc[1, "use_evaluation"] = False
    calibration, evaluation = evaluation_rows(frame)
    assert len(calibration) + len(evaluation) == 1198
    assert not set(calibration.patient_id) & set(evaluation.patient_id)
    assert calibration.quality_fail.all()
    assert evaluation.quality_fail.all()
    assert patient_role("one patient") == patient_role("one patient")


def test_tampered_role_or_inadequate_roles_fail():
    frame = patient_manifest()
    frame.loc[0, "role"] = "tampered"
    with pytest.raises(ValueError, match="frozen hash"):
        evaluation_rows(frame)
    with pytest.raises(ValueError, match="100 normals"):
        evaluation_rows(patient_manifest().iloc[:20])


def test_existing_sph_labels_and_duplicate_policy_unchanged():
    assert sph.labels({"1"})["primary"] == 0
    assert np.isnan(sph.labels({"21"})["primary"])
    assert sph.labels({"21", "160"})["primary"] == 1
    index = pd.Index(["A", "B", "C", "D"])
    status = sph.duplicate_status(
        pd.Series(["a", "a", "b", "b"], index=index), pd.Series(["1", "1", "1", "160"], index=index)
    )
    assert status.to_list() == ["kept", "dropped_copy", "dropped_conflict", "dropped_conflict"]


def test_frozen_threshold_ties_are_conservative():
    from ecg_experiment.cpu_screening041 import RAW_FEATURES, calibrate_threshold

    assert list(RAW_FEATURES) == FEATURE_NAMES
    scores = np.array([0.2] * 95 + [0.8] * 5)
    threshold = calibrate_threshold(np.zeros(100), scores)
    assert threshold["threshold"] == 0.2
    assert int((scores > threshold["threshold"]).sum()) == 5
    tied = calibrate_threshold(np.zeros(100), np.ones(100))
    assert tied["normal_referral_rate"] == 0


def test_no_beat_record_keeps_available_measurements(monkeypatch):
    time = np.arange(5000) / 500
    window = np.vstack([np.sin(2 * np.pi * 0.05 * time) * (1 + lead / 12) for lead in range(12)])
    monkeypatch.setattr(sph, "read_window", lambda ecg_id: window.astype(np.float32))
    result = extract_record("artificial_no_beats")
    assert result["status"] == "sin_latidos"
    assert result["n_beats"] == 0
    assert np.isfinite(result["sp_centroid_hz"])
    assert np.isnan(result["qrs_dur_ms"])


def test_unbound_alternate_model_rejected_before_inference(tmp_path):
    from ecg_experiment.measurement_screening042 import evaluate

    with pytest.raises(ValueError, match="bound fitted_models.joblib"):
        evaluate(tmp_path, tmp_path / "protocol.md", tmp_path / "alternate_model.joblib")
