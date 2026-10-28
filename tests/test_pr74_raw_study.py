"""Exercise fresh cohort and feature admission without reading historical scientific caches."""

import json
from pathlib import Path

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
