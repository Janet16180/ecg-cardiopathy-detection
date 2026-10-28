"""Exercise fresh cohort and feature admission without reading historical scientific caches."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecg_experiment.files import sha256_file
from ecg_experiment.pr74_fresh_baseline import finish_stage
from ecg_experiment.pr74_raw_study import matching_extraction, subset, validate_manifest


def manifest():
    return pd.DataFrame(
        {
            "record_id": ["ptbxl:1"],
            "source": ["ptbxl"],
            "target": ["0"],
            "label_evidence_rank": ["1"],
            "patient_id": ["ptbxl:5.0"],
            "signal_sha256": ["canonical"],
            "split": ["train"],
            "quality_passed": ["True"],
            "window_start": ["0"],
            "samples": ["5000"],
        }
    )


def test_false_csv_quality_is_not_treated_as_truthy():
    rows = manifest()
    rows["quality_passed"] = "False"
    with pytest.raises(ValueError, match="admission"):
        validate_manifest(rows)


@pytest.mark.parametrize(("column", "value"), [("split", "test"), ("window_start", "5000"), ("target", "2")])
def test_fresh_manifest_blocks_reserved_roles_views_and_undefined_targets(column, value):
    rows = manifest()
    rows[column] = value
    with pytest.raises(ValueError, match="admission|binary"):
        validate_manifest(rows)


def test_selected_identity_cannot_change_target_or_known_patient():
    pool = validate_manifest(manifest())
    selected = manifest()
    selected["patient_id"] = "ptbxl:9.0"
    with pytest.raises(ValueError, match="admission changed: patient_id"):
        subset(selected, pool)
    selected = manifest()
    selected["target"] = "1"
    with pytest.raises(ValueError, match="admission changed: target"):
        subset(selected, pool)


def test_warm_features_must_match_every_training_view_and_role():
    expected = manifest()
    cached = manifest()
    cached["signal_sha256"] = "wrong window"
    with pytest.raises(ValueError, match="signal_sha256"):
        matching_extraction(cached, expected)
    cached = manifest()
    cached["split"] = "test"
    with pytest.raises(ValueError, match="split"):
        matching_extraction(cached, expected)


def test_completion_rechecks_upstream_recipe_sources_after_execution(tmp_path, monkeypatch):
    from ecg_experiment import pr74_fresh_baseline as fresh

    monkeypatch.setattr(fresh, "ROOT", tmp_path)
    code = tmp_path / "recipe.py"
    code.write_text("original")
    receipt = tmp_path / "outputs/upstream/receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({"recipe_sha256": {"recipe.py": sha256_file(code)}}))
    hashes = {"outputs/upstream/receipt.json": sha256_file(receipt)}
    directory = tmp_path / "outputs/study/extract"
    directory.mkdir(parents=True)
    code.write_text("mutated while running")
    with pytest.raises(ValueError, match="Artifact changed: recipe.py"):
        finish_stage(tmp_path, Path("outputs/study"), "extract", {"source_sha256": {}}, hashes, {})
    assert not (directory / "completion.json").exists()


def test_fresh_bundle_loader_rejects_all_arbitrary_pickle_paths(tmp_path):
    from ecg_experiment.pr74_raw_inference import load_bundle

    with pytest.raises(ValueError, match="fixed fresh corrective bundle"):
        load_bundle(tmp_path, Path("uploads/untrusted.joblib"))


def test_scientific_parity_requires_every_matrix_and_model_site():
    import copy

    from ecg_experiment.pr74_raw_study import compare_aggregates

    artifact = {
        "models": [
            {
                "id": "baseline",
                "evaluations": [
                    {
                        "pool": "sph",
                        "threshold": 0.5,
                        "calibration": {"flagged_normals": 1},
                        "metrics": {"tp": 3, "tn": 4, "fp": 1, "fn": 2, "auroc": 0.8},
                    }
                ],
            }
        ],
        "primary": {"development_advantage": False},
    }
    assert compare_aggregates(artifact, artifact)["all_confusion_matrices_exact"]
    wrong = copy.deepcopy(artifact)
    wrong["models"][0]["evaluations"][0]["metrics"]["tp"] = 4
    with pytest.raises(ValueError, match="confusion matrix"):
        compare_aggregates(wrong, artifact)


def test_fresh_feature_alignment_needs_no_historical_artifact_and_blocks_calibration_rows(tmp_path):
    from ecg_experiment.cpu_screening041 import RAW_FEATURES
    from ecg_experiment.pr74_raw_study import feature_matrix

    pool = pd.concat([manifest(), manifest()], ignore_index=True)
    pool.loc[1, ["record_id", "source", "signal_sha256"]] = ["ningbo:1", "ningbo", "challenge"]
    extracted = tmp_path / "outputs/new/extract"
    baseline = tmp_path / "outputs/fresh/features"
    extracted.mkdir(parents=True)
    baseline.mkdir(parents=True)
    pool.iloc[[1]].to_csv(extracted / "rows.csv", index=False)
    pd.DataFrame([[2.0] * 130], columns=RAW_FEATURES).to_csv(extracted / "measurements.csv", index=False)
    np.save(extracted / "xecg.npy", np.full((1, 1024), 2, dtype=np.float32))
    ptb = pd.DataFrame({"ecg_id": ["1"], "fit": ["True"], "signal_sha256": ["canonical"], "target": ["0"]})
    ptb.to_csv(baseline / "ptb_rows.csv", index=False)
    pd.DataFrame([[1.0] * 130], columns=RAW_FEATURES).to_csv(baseline / "ptb_measurements.csv", index=False)
    np.save(baseline / "ptb_xecg.npy", np.ones((1, 1024), dtype=np.float32))
    raw, vectors = feature_matrix(tmp_path, Path("outputs/new"), Path("outputs/fresh"), pool)
    np.testing.assert_array_equal(raw.iloc[:, 0], [1.0, 2.0])
    np.testing.assert_array_equal(vectors[:, 0], [1.0, 2.0])
    ptb["fit"] = "False"
    ptb.to_csv(baseline / "ptb_rows.csv", index=False)
    with pytest.raises(ValueError, match="training features changed roles"):
        feature_matrix(tmp_path, Path("outputs/new"), Path("outputs/fresh"), pool)
