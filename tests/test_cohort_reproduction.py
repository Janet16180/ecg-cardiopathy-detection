"""Resumption preserves original seeds, source bytes and verified recovery boundaries."""

import hashlib
import io
import json
from pathlib import Path

import pytest

from ecg_experiment import cohort_reproduction as recovery


def test_dependency_plan_keeps_frozen_seeds_and_profiles_before_full_jobs(tmp_path):
    plan = recovery.stages(tmp_path, Path("outputs/new"), 25000, 17, 4)
    names = [stage["name"] for stage in plan]
    for profile, full in [("ptb_profile", "ptb_quality"), ("challenge_profile", "challenge_split"),
                          ("references_profile", "references_full"),
                          ("challenge_quality_profile", "challenge_quality_full")]:
        assert names.index(profile) < names.index(full)
    assert plan[0]["args"][plan[0]["args"].index("--seed") + 1] == "42"
    assert plan[-1]["args"][plan[-1]["args"].index("--seed") + 1] == "17"
    assert all("train" not in stage["module"] or "training_quality" in stage["module"] for stage in plan)


def test_downloaded_mapping_is_verified_before_publication(tmp_path, monkeypatch):
    contents = b"verified public mapping"
    expected = hashlib.sha256(contents).hexdigest()
    monkeypatch.setattr(recovery, "MAPPING_SHA256", {"mapping.csv": expected})
    requested = []

    def public_download(url, timeout):
        requested.append(url)
        return io.BytesIO(contents)

    monkeypatch.setattr(recovery.urllib.request, "urlopen", public_download)
    recovery.ensure_mappings(tmp_path)
    assert recovery.MAPPING_COMMIT in requested[0]
    assert (tmp_path / recovery.MAPPING_ROOT / "mapping.csv").read_bytes() == contents
    recovery.ensure_mappings(tmp_path)
    assert len(requested) == 1
    (tmp_path / recovery.MAPPING_ROOT / "mapping.csv").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Existing public mapping changed"):
        recovery.ensure_mappings(tmp_path)


def test_wrong_mapping_download_never_creates_final_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "MAPPING_SHA256", {"mapping.csv": "0" * 64})
    monkeypatch.setattr(recovery.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(b"wrong"))
    with pytest.raises(ValueError, match="Downloaded public mapping hash mismatch"):
        recovery.ensure_mappings(tmp_path)
    assert not (tmp_path / recovery.MAPPING_ROOT / "mapping.csv").exists()


def test_resume_rejects_changed_artifacts_and_different_seeds(tmp_path):
    table = tmp_path / "rows.csv"
    table.write_text("unchanged")
    receipt = {
        "complete": True, "scope": "new_supervised_age_label_cohorts", "source_sha256": {},
        "input_sha256": {}, "recipe_sha256": {}, "seed": 42, "size_ceiling": 25000,
        "artifacts": {"rows.csv": hashlib.sha256(table.read_bytes()).hexdigest()},
    }
    stage = {"kind": "receipt", "receipt": "receipt.json", "scope": receipt["scope"],
             "seed": 42, "size": 25000}
    (tmp_path / "receipt.json").write_text(json.dumps(receipt))
    recovery.verify_stage(tmp_path, stage)
    with pytest.raises(ValueError, match="different seed"):
        recovery.verify_stage(tmp_path, stage | {"seed": 43})
    table.write_text("changed")
    with pytest.raises(ValueError, match="artifact changed"):
        recovery.verify_stage(tmp_path, stage)
