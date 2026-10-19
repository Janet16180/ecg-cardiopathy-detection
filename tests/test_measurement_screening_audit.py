"""Artificial independent-audit checks; no production predictions are opened."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecg_experiment import measurement_screening_audit as audit


def evaluation_table():
    return pd.DataFrame(
        {
            "patient_id": ["b", "b", "a", "a", "c", "c", "d", "d"],
            "target": [0, 1] * 4,
            "MI": [0, 1] * 4,
            "STTC": [0, 1] * 4,
            "CD": [0] * 8,
            "HYP": [0] * 8,
        }
    )


def test_known_point_metrics_and_adjacent_strict_threshold():
    y = np.array([0, 0, 1, 1])
    probability = np.array([0.2, 0.5, 0.5, np.nextafter(0.5, np.inf)])
    observed = audit.metrics(y, probability, 0.5)
    assert observed["sensitivity"] == 0.5
    assert observed["specificity"] == 1
    assert observed["auroc"] == 0.875
    assert observed["brier"] == pytest.approx(np.mean((probability - y) ** 2))


def test_calibration_order_statistic_ties_and_small_sample():
    result = audit.threshold_record(np.arange(40, dtype=float) / 40)
    assert result["threshold"] == 37 / 40
    assert result["referred_normal_count"] == 2
    assert audit.threshold_record(np.ones(40))["normal_referral_rate"] == 0
    assert audit.threshold_record(np.arange(19))["referred_normal_count"] == 0
    with pytest.raises(ValueError, match="nonempty and finite"):
        audit.threshold_record(np.array([np.nan]))


def test_patient_samples_have_exact_sorted_patient_draw_order():
    table = evaluation_table()
    actual = next(audit.patient_samples(table.patient_id.to_numpy(), table.target.to_numpy(), 1))
    choices = np.random.default_rng(audit.SEED).integers(0, 4, 4)
    members = [np.array([2, 3]), np.array([0, 1]), np.array([4, 5]), np.array([6, 7])]
    np.testing.assert_array_equal(actual, np.concatenate([members[i] for i in choices]))


def test_constant_perfect_and_reversed_scores_have_exact_intervals_and_misses():
    table = evaluation_table()
    perfect = np.where(table.target == 1, 0.9, 0.1)
    analysis = audit.analyze_predictions(
        table, {"logistic": perfect, "hgb": 1 - perfect}, {"logistic": 0.5, "hgb": 0.5}, draws=30
    )
    assert analysis["auroc"] == {
        "difference": -1,
        "ci_low": -1,
        "ci_high": -1,
        "draws": 30,
        "skipped_draws": 0,
        "seed": audit.SEED,
    }
    assert analysis["sensitivity"]["difference"] == -1
    assert analysis["arms"]["logistic"]["metrics"]["brier"]["value"] == pytest.approx(0.01)
    assert analysis["arms"]["hgb"]["exploratory_superclass_misses"]["MI"] == {
        "positive_count": 4,
        "missed_count": 4,
    }
    assert analysis["arms"]["hgb"]["exploratory_superclass_misses"]["STTC"]["missed_count"] == 4
    assert audit.decision_gates(analysis) == [False, False, False, False]


def test_single_class_samples_are_counted_and_patient_repeats_remain_whole():
    table = evaluation_table().iloc[[0, 1]].copy()
    table["patient_id"] = ["one", "two"]
    probability = np.array([0.1, 0.9])
    analysis = audit.analyze_predictions(
        table, dict.fromkeys(audit.ARMS, probability), dict.fromkeys(audit.ARMS, 0.5), draws=30
    )
    skipped = analysis["auroc"]["skipped_draws"]
    assert 0 < skipped < 30
    assert analysis["sensitivity"]["skipped_draws"] == skipped
    assert all(analysis["arms"][name]["skipped_draws"] == skipped for name in audit.ARMS)


def test_recursive_comparison_detects_missing_extra_and_perturbed_fields():
    checks = []
    audit.compare(
        {"metric": {"value": 0.75, "ci_low": 0.5}}, {"metric": {"value": 0.76, "extra": 1}}, "audit", checks
    )
    failures = [check["name"] for check in checks if not check["passed"]]
    assert "audit.metric.keys" in failures
    assert "audit.metric.value" in failures
    checks = []
    audit.compare(0.75, 0.75 + 0.5e-12, "within_tolerance", checks)
    assert checks[0]["passed"]


@pytest.mark.parametrize(
    "path", ["../secret.json", "/tmp/raw.csv", "data/raw/record.dat", "record.h5", "raw.npy"]
)
def test_waveform_or_escaping_binding_refused(path):
    with pytest.raises(ValueError, match="non-waveform artifact"):
        audit.declared_file(path)


def test_stage_hash_verification_ignores_qualification_waveform_map(tmp_path, monkeypatch):
    source = tmp_path / "source.py"
    source.write_text("pass\n")
    stage = tmp_path / "stage"
    stage.mkdir()
    payload = stage / "payload.csv"
    payload.write_text("value\n1\n")
    receipt = {
        "source_sha256": {"source.py": audit.sha256_file(source)},
        "protocol": {"path": "source.py", "sha256": audit.sha256_file(source)},
        "artifacts": {"payload.csv": audit.sha256_file(payload)},
        "raw_inputs": {"raw_waveform.dat": "must_not_be_opened"},
    }
    (stage / "completion.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    checks = []
    audit.check_stage(stage, checks)
    assert all(check["passed"] for check in checks)
    payload.write_text("value\n2\n")
    checks = []
    audit.check_stage(stage, checks)
    assert any(not check["passed"] for check in checks)


def test_declared_public_sph_pdf_documents_are_hashed_without_waveform_access(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    small_files = {}
    for name in ("19738468/rule.pdf", "17914088/example-code.pdf"):
        relative = f"data/raw/sph/sph/{name}"
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        content = b"%PDF artificial public documentation\n"
        path.write_bytes(content)
        small_files[name] = hashlib.md5(content).hexdigest()
        assert audit.declared_file(relative) == path
    checks = []
    audit.check_stage_inputs(tmp_path / "profile", {"small_files_md5": small_files}, checks)
    assert len(checks) == 2
    assert all(check["passed"] for check in checks)
    for name in ("record.dat", "record.h5"):
        with pytest.raises(ValueError, match="non-waveform artifact"):
            audit.declared_file(f"data/raw/sph/sph/{name}")


def test_auditor_never_imports_production_experiments_or_unpickles_models():
    source = __import__("pathlib").Path(audit.__file__).read_text()
    assert "from ecg_experiment.cpu_screening041" not in source
    assert "from ecg_experiment.measurement_screening042" not in source
    assert "joblib.load" not in source
    assert "predict_proba" not in source


def test_external_csv_round_trip_preserves_adjacent_threshold_probability(tmp_path):
    directory = tmp_path / "evaluation"
    directory.mkdir()
    probability = np.nextafter(0.5, np.inf)
    evaluation = pd.DataFrame(
        {
            "ecg_id": ["eval"],
            "patient_id": ["patient-evaluation"],
            "target": [1],
            "hgb_probability": [probability],
        }
    ).set_index("ecg_id")
    evaluation.to_csv(directory / "predictions.csv", float_format="%.17g")
    for name in audit.ARMS:
        normal = pd.DataFrame(
            {"ecg_id": ["cal"], "patient_id": ["patient-normal"], "target": [0], "probability": [probability]}
        ).set_index("ecg_id")
        normal.to_csv(directory / f"{name}_normal_calibration.csv", float_format="%.17g")
    checks = []
    loaded, calibration = audit.read_042_tables(tmp_path, checks)
    assert loaded.hgb_probability.iloc[0] == probability
    assert calibration["hgb"].probability.iloc[0] == probability
    assert all(check["passed"] for check in checks)


def test_nonbinary_target_is_rejected_before_integer_conversion():
    table = evaluation_table()
    table["target"] = table.target.astype(float)
    table.loc[0, "target"] = 0.4
    with pytest.raises(ValueError, match="binary targets"):
        audit.analyze_predictions(
            table, dict.fromkeys(audit.ARMS, np.full(8, 0.5)), dict.fromkeys(audit.ARMS, 0.5), draws=10
        )


def test_failed_audit_preserves_receipt_and_cannot_overwrite(tmp_path, monkeypatch):
    runner = tmp_path / "scripts/reports/audit_measurement_screening.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("# artificial runner\n")
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "to_stored", lambda path: Path(path).name)
    monkeypatch.setattr(audit.subprocess, "check_output", lambda *_args, **_kwargs: "artificial-revision\n")

    def fail(*_args):
        raise ValueError("artificial missing input")

    monkeypatch.setattr(audit, "audit_041", fail)
    output = tmp_path / "audit"
    receipt = audit.run_audit(tmp_path / "041", tmp_path / "042", output)
    assert receipt["status"] == "failed"
    assert len(receipt["mismatches"]) == 1
    assert "artificial missing input" in receipt["mismatches"][0]["traceback"]
    saved = json.loads((output / "completion.json").read_text())
    assert saved["artifacts"]["audit.json"] == audit.sha256_file(output / "audit.json")
    with pytest.raises(FileExistsError):
        audit.run_audit(tmp_path / "041", tmp_path / "042", output)


def test_independent_duplicate_retention_and_frozen_role_table_tampering(tmp_path):
    directory = tmp_path / "preparation"
    directory.mkdir()
    table = pd.DataFrame(
        {
            "patient_id": ["p2", "p1", "p4", "p3"],
            "role": ["evaluation", "calibration", "evaluation", "calibration"],
            "signal_sha256": ["same-a", "same-a", "same-b", "same-b"],
            "aha_codes": ["1", "1", "160", "1"],
            "duplicate_status": ["dropped_copy", "kept", "dropped_conflict", "dropped_conflict"],
            "use_evaluation": [False, True, False, False],
        },
        index=pd.Index(["B", "A", "D", "C"], name="ecg_id"),
    )
    table[["patient_id", "role"]].to_csv(directory / "patient_roles.csv")
    checks = []
    audit.check_042_retention(tmp_path, table, checks)
    assert all(check["passed"] for check in checks)
    table.loc["B", "use_evaluation"] = True
    checks = []
    audit.check_042_retention(tmp_path, table, checks)
    assert not next(check["passed"] for check in checks if check["name"] == "042.retention_policy")
    roles = table[["patient_id", "role"]].copy()
    roles.loc["A", "role"] = "evaluation"
    roles.to_csv(directory / "patient_roles.csv")
    checks = []
    audit.check_042_retention(tmp_path, table, checks)
    assert not next(
        check["passed"] for check in checks if check["name"] == "042.patient_roles_bound_before_inference"
    )


def test_qualification_and_resource_gate_tampering_is_not_accepted():
    qualification = {
        "passed": True,
        "records": 32,
        "features": 130,
        "serialized_max_abs_difference": 0.0,
        "missingness_equal": True,
        "status_plausibility_equal": True,
        "flat_leads_equal": True,
        "notebook_extractor_sha256": "ee39c8f6f7remaining",
        "wall_seconds": 1,
    }
    profile = {
        "records": 64,
        "resource_gate_passed": True,
        "forecast_wall_seconds": 3600,
        "forecast_storage_bytes": 512 * 1024**2,
        "wall_seconds": 1,
    }
    stages = {
        "qualification": qualification,
        "profile": profile,
        "preparation": {"wall_seconds": 1},
        "evaluation": {"wall_seconds": 1},
    }
    checks = []
    audit.check_042_gates(stages, checks)
    assert all(check["passed"] for check in checks)
    qualification["passed"] = False
    profile["forecast_wall_seconds"] += 1
    checks = []
    audit.check_042_gates(stages, checks)
    assert not next(check["passed"] for check in checks if check["name"] == "042.qualification.passed")
    assert not next(check["passed"] for check in checks if check["name"] == "042.profile.wall_ceiling")
