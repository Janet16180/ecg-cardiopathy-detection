"""Cold paths must bootstrap honest roles and references without historical scientific prerequisites."""

import json
from pathlib import Path

import pandas as pd
import pytest

from ecg_experiment import cold_data_stages as data
from ecg_experiment.cold_start_pipeline import plan


def test_cold_dependency_graph_uses_new_roles_and_profiles_before_each_new_full_stage(tmp_path):
    stages = plan(tmp_path, workers=4, seed=42)
    names = [stage["name"] for stage in stages]
    assert not any("screen01" in str(stage).lower() or "fitted_models" in str(stage) for stage in stages)
    assert names.index("fresh_raw_roles_full") < names.index("fresh_references_profile")
    for profile, full in [("fresh_raw_roles_profile", "fresh_raw_roles_full"),
                          ("fresh_references_profile", "fresh_references_full"),
                          ("fresh_quality_profile", "fresh_quality_full")]:
        assert names.index(profile) < names.index(full)
    assert names[-2:] == ["fresh_25k_cohorts", "fresh_strict_matched"]
    assert all("--reuse-verified-caches" not in stage["args"] for stage in stages)


def test_sph_reference_reader_uses_fresh_administration_without_diagnosis_columns(tmp_path, monkeypatch):
    directory = tmp_path / data.RAW_ROLES
    directory.mkdir(parents=True)
    pd.DataFrame({"ecg_id": ["A1"], "patient_id": ["S1"], "role": ["evaluation"],
                  "signal_sha256": ["a" * 64], "target": ["private_target"]}).to_csv(
                      directory / "sph_rows.csv", index=False)
    monkeypatch.setattr(data, "raw_roles", lambda root: ({}, {}))
    original = pd.read_csv
    consumed = []

    def reader(*args, **kwargs):
        consumed.extend(kwargs["usecols"])
        return original(*args, **kwargs)

    monkeypatch.setattr(data.pd, "read_csv", reader)
    rows, _ = data.fresh_sph_references(tmp_path)
    assert "target" not in consumed
    assert rows[0]["patient_id"] == "sph:S1"
    assert rows[0]["canonical_sha256"] == "a" * 64


def test_absent_legacy_scientific_inputs_force_cold_path_instead_of_fake_receipts(tmp_path):
    directory = Path("old_cache")
    (tmp_path / directory).mkdir()
    (tmp_path / directory / "receipt.json").write_text(json.dumps({
        "input_sha256": {"missing_old_scientific/completion.json": "a" * 64},
    }))
    assert data.verified_cache(tmp_path, directory, "training_quality") is None


def test_full_reference_profile_gate_runs_before_source_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "source_recipe", lambda: {})

    def no_profile(*args):
        raise ValueError("missing real profile")

    def forbidden(*args):
        raise AssertionError("Unprofiled full raw stage reached source data")

    monkeypatch.setattr(data, "profile_gate", no_profile)
    monkeypatch.setattr(data, "fresh_sph_references", forbidden)
    with pytest.raises(ValueError, match="missing real profile"):
        data.references(tmp_path)
