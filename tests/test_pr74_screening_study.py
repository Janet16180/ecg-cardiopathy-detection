"""Check actual inference abstention, identity isolation and the prespecified practical decision."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecg_experiment.cpu_screening041 import RAW_FEATURES
from ecg_experiment.pr74_screening_inference import VerifiedPredictor, load_bundle
from ecg_experiment.pr74_screening_study import assert_evaluation_independence, primary_comparison


def test_inference_flat_signal_abstains_before_a_classifier_or_encoder_is_used(tmp_path):
    bundle = {"schema_version": 1, "features": list(RAW_FEATURES), "models": {"model": {}}, "fitted": {}}
    result = VerifiedPredictor(tmp_path, bundle).predict(np.zeros((12, 5000)), "model")
    assert result["available"]
    assert result["abstained"]
    assert "score" not in result
    assert "constant_lead" in result["quality_reasons"]


def test_failed_physiological_measurement_abstains_before_model_access(tmp_path, monkeypatch):
    from ecg_experiment import pr74_screening_inference as inference

    monkeypatch.setattr(inference, "assess", lambda signal: ([], []))
    monkeypatch.setattr(inference, "extract_array", lambda signal: {"status": "sin_latidos"})
    bundle = {"schema_version": 1, "features": list(RAW_FEATURES), "models": {"model": {}}, "fitted": {}}
    result = VerifiedPredictor(tmp_path, bundle).predict(np.ones((12, 5000)), "model")
    assert result["abstained"]
    assert result["quality_reasons"] == ["sin_latidos"]


def test_bundle_loader_does_not_accept_uploaded_or_arbitrary_pickle_paths(tmp_path):
    with pytest.raises(ValueError, match="fixed corrective bundle"):
        load_bundle(tmp_path, Path("uploads/patient.joblib"))


@pytest.mark.parametrize(("record", "patient"), [("ptbxl:1", "ptbxl:99.0"), ("ptbxl:2", "ptbxl:11.0")])
def test_known_patient_or_record_overlap_blocks_evaluation(record, patient):
    arms = {"strict_young": pd.DataFrame({"record_id": [record], "patient_id": [patient]})}
    frame = pd.DataFrame({"ecg_id": ["1"], "patient_id": ["ptbxl:11.0"]})
    with pytest.raises(ValueError, match="identity overlap"):
        assert_evaluation_independence(arms, "ptb", frame)


def test_positive_interval_does_not_override_two_percentage_point_rule(monkeypatch):
    from ecg_experiment import pr74_screening_study as study

    monkeypatch.setattr(study, "sensitivity_difference", lambda *args, **kwargs: {
        "difference": 0.015, "ci_low": 0.005, "ci_high": 0.025,
    })
    records = [{"model_id": name, "evaluation": {"threshold": 0.5}}
               for name in ["strict_young_blend", "matched_general_blend"]]
    frame = pd.DataFrame({"patient_id": ["a", "b"], "target": [0, 1], "evaluation": True})
    arrays = {name: np.array([0.2, 0.9]) for name in ["strict_young_blend", "matched_general_blend"]}
    result = primary_comparison(records, arrays, frame)
    assert not result["development_advantage"]
    assert not result["pipeline_promoted"]
    assert result["gates"]["paired_95_lower_bound_positive"]
