"""Artificial arithmetic and artifact-safety checks without production output reads."""

import json
from pathlib import Path

import numpy as np
import pytest

from ecg_experiment import lead_grounding_audit043 as audit


def artificial_payload():
    offered = np.zeros((3, 12), dtype=bool)
    offered[:, :2] = True
    target = np.full((3, 12), -1, dtype=np.int8)
    target[:, :2] = [[1, 0], [0, 1], [1, 0]]
    payload = {
        "ecg_id": np.array([1, 2, 3]),
        "patient_id": np.array([11, 11, 22]),
        "strat_fold": np.full(3, 9),
        "target_dx": np.array(["condition"] * 3),
        "question": np.array(["Exact question"] * 3),
        "informative": np.ones(3, dtype=bool),
        "offered": offered,
        "target": target,
        "lead_order": np.array(audit.LEADS),
    }
    for name in audit.MODELS:
        probability = np.full((3, 12), np.nan)
        probability[:, :2] = [[0.5, 0.5]] * 3 if name == "context_hgb" else [[0.9, 0.1]] * 3
        payload[f"{name}__probability"] = probability
        payload.update({f"{name}__{metric}": np.zeros(3) for metric in audit.METRICS})
        calculated = audit.query_statistics(payload, name)
        payload.update({f"{name}__{metric}": values for metric, values in calculated.items()})
    return payload


def test_query_ties_strict_half_and_patient_macro_differ_from_query_pooling():
    payload = artificial_payload()
    audit.validate_payload(payload, frozen_counts=False)
    reconstructed = audit.reconstruct(payload, draws=30)
    summaries = reconstructed["summary"]
    assert summaries["models"]["context_hgb"]["average_precision"] == 0.5
    assert summaries["models"]["context_hgb"]["exact_set_accuracy"] == 0
    assert summaries["models"]["context_hgb"]["brier"] == 0.25
    assert summaries["models"]["wave_hgb"]["average_precision"] == 0.875
    assert payload["wave_hgb__average_precision"].mean() == pytest.approx(5 / 6)
    assert summaries["primary_wave_minus_context"]["difference"] == 0.375
    assert summaries["models"]["wave_hgb"]["brier"] == pytest.approx(0.21)


def test_adjacent_representable_probability_is_referred_but_exact_half_is_not():
    payload = artificial_payload()
    payload["wave_hgb__probability"][0, :2] = [np.nextafter(0.5, np.inf), 0.5]
    calculated = audit.query_statistics(payload, "wave_hgb")
    assert calculated["exact_set_accuracy"][0] == 1
    payload["wave_hgb__probability"][0, 0] = 0.5
    assert audit.query_statistics(payload, "wave_hgb")["exact_set_accuracy"][0] == 0


@pytest.mark.parametrize(
    "mutation", ["unknown_target", "unoffered_score", "known_unknown", "all_positive", "empty", "mask_type"]
)
def test_unknown_masks_and_malformed_targets_rejected(mutation):
    payload = artificial_payload()
    if mutation == "unknown_target":
        payload["target"][0, 3] = 0
    if mutation == "unoffered_score":
        payload["wave_hgb__probability"][0, 3] = 0.1
    if mutation == "known_unknown":
        payload["target"][0, 0] = -1
    if mutation == "all_positive":
        payload["target"][0, :2] = 1
    if mutation == "empty":
        payload["target"][0, :2] = 0
    if mutation == "mask_type":
        payload["offered"] = payload["offered"].astype(np.uint8)
    with pytest.raises(ValueError, match=".+"):
        audit.validate_payload(payload, frozen_counts=False)


@pytest.mark.parametrize(
    "mutation", ["duplicate", "fold8", "lead_order", "patient_mismatch", "frozen_counts"]
)
def test_query_identity_and_frozen_accounting_checks(mutation):
    payload = artificial_payload()
    if mutation == "duplicate":
        payload["ecg_id"][1] = 1
    if mutation == "fold8":
        payload["strat_fold"][0] = 8
    if mutation == "lead_order":
        payload["lead_order"] = payload["lead_order"][::-1]
    if mutation == "patient_mismatch":
        payload["ecg_id"][2] = 1
    with pytest.raises(ValueError, match=".+"):
        audit.validate_payload(payload, frozen_counts=mutation == "frozen_counts")


def test_independent_sorted_patient_bootstrap_exact_draw_order():
    patients = np.array([20, 20, 10])
    first, second = np.array([0.5, 1, 0.5]), np.array([0.25] * 3)
    calculated = audit.paired_difference(patients, first, second, draws=100)
    assert calculated["difference"] == 0.375
    rng = np.random.default_rng(audit.SEED)
    sampled = [np.mean(np.array([0.25, 0.5])[rng.integers(0, 2, 2)]) for _ in range(100)]
    assert calculated["ci_low"] == np.percentile(sampled, 2.5)
    assert calculated["ci_high"] == np.percentile(sampled, 97.5)
    assert calculated["skipped_draws"] == 0
    assert calculated["patients"] == 2


@pytest.mark.parametrize(
    ("gain", "lower", "informative", "lookup", "advance"),
    [
        (0.05, 1e-12, 1e-12, 0, True),
        (0.049999, 0.01, 0.01, 0.01, False),
        (0.05, 0, 0.01, 0.01, False),
        (0.05, 0.01, 0, 0.01, False),
        (0.05, 0.01, 0.01, -1e-12, False),
    ],
)
def test_exact_gate_conjunction(gain, lower, informative, lookup, advance):
    calculated = audit.decision(
        {"difference": gain, "ci_low": lower}, {"difference": informative}, {"difference": lookup}
    )
    assert (calculated["outcome"] == "advance_benchmark_grounding") == advance


def test_saved_query_and_patient_tampering_detected_with_recursive_tolerance():
    payload = artificial_payload()
    reconstructed = audit.reconstruct(payload, draws=30)
    result, means = reconstructed["summary"], reconstructed["patient_means"]
    checks = []
    audit.compare_saved(result, payload, means, reconstructed, checks)
    assert all(check["passed"] for check in checks)
    payload["context_hgb__average_precision"][0] += 0.01
    checks = []
    audit.compare_saved(result, payload, means, reconstructed, checks)
    assert any(not check["passed"] for check in checks if check["name"].startswith("query.context_hgb"))
    checks = []
    audit.compare([{"value": 0.5}], [{"value": 0.5 + 0.5e-12}], "nested", checks)
    assert all(check["passed"] for check in checks)


def frozen_qualification():
    roles = {}
    for name, counts in {
        "fit": (1117, 823, 776),
        "unused": (154, 110, 99),
        "development": (137, 100, 97),
    }.items():
        roles[name] = dict(zip(("queries", "ecgs", "patients"), counts, strict=True))
        roles[name].update(answer_cardinality={"1": counts[0]}, offered_cardinality={"2": counts[0]})
    return {
        "roles": roles,
        "training_contexts": 23,
        "constant_training_contexts": 0,
        "informative_development_queries": 69,
        "informative_development_patients": 56,
        "parser": dict.fromkeys(
            (
                "duplicate_queries",
                "conflicting_queries",
                "missing_joins",
                "empty_targets",
                "all_positive_targets",
            ),
            0,
        ),
    }


def test_frozen_qualification_accounting_checked_conditionally():
    qualification = frozen_qualification()
    checks = []
    audit.verify_qualification(qualification, checks)
    assert all(check["passed"] for check in checks)
    qualification["roles"]["unused"]["patients"] = 100
    checks = []
    audit.verify_qualification(qualification, checks)
    assert any(not check["passed"] for check in checks)


def test_npz_round_trip_preserves_masks_and_adjacent_probability_exactly(tmp_path):
    payload = artificial_payload()
    payload["wave_hgb__probability"][0, 0] = np.nextafter(0.5, np.inf)
    path = tmp_path / "predictions.npz"
    np.savez_compressed(path, **payload)
    with np.load(path, allow_pickle=False) as saved:
        for name, expected in payload.items():
            np.testing.assert_array_equal(saved[name], expected)


def test_declared_jsonl_and_model_hashes_do_not_parse_or_unpickle(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    inputs = {}
    for name in ("source.jsonl", "fitted_models.joblib", "missingness.json"):
        path = tmp_path / name
        path.write_bytes(b"deliberately not valid JSON or pickle\n")
        inputs[name] = audit.sha256_file(path)
    checks = []
    audit.verify_bindings(inputs, checks, "artificial")
    assert all(check["passed"] for check in checks)
    for name in ("record.dat", "record.h5", "raw.npy", "../source.jsonl"):
        with pytest.raises(ValueError, match="non-waveform"):
            audit.verify_bindings({name: "forbidden"}, [], "artificial")


def test_exact_feature_manifest_and_thread_cap_tampering():
    shared = ["context_code", "lead_code", *(f"offered_{lead}" for lead in audit.LEADS)]
    waveform = [
        *shared,
        *(f"raw__{name}" for name in audit.RAW_FEATURES),
        *(f"candidate__{name}" for name in audit.CANDIDATE_MEASUREMENTS),
    ]
    config = {
        "feature_columns": {"context_hgb": shared, "wave_hgb": waveform},
        "raw_feature_columns": list(audit.RAW_FEATURES),
        "candidate_measurements": list(audit.CANDIDATE_MEASUREMENTS),
        "nonfinite_policy": "NaN; no imputation or clipping",
        "threads": 4,
    }
    assert len(shared) == 14
    assert len(waveform) == 153
    assert len(audit.RAW_FEATURES) == 130
    checks = []
    audit.verify_feature_manifest(config, checks)
    assert all(check["passed"] for check in checks)
    config["threads"] = 5
    config["feature_columns"]["wave_hgb"][14] = "raw__patient_id"
    checks = []
    audit.verify_feature_manifest(config, checks)
    assert not next(check["passed"] for check in checks if check["name"] == "config.thread_cap")
    assert any(not check["passed"] for check in checks if check["name"].startswith("config.feature_columns"))


def test_failure_receipts_and_output_binding_are_exclusive(tmp_path, monkeypatch):
    for name in (
        "scripts/reports/audit_lead_grounding043.py",
        "ecg_experiment/files.py",
        "ecg_experiment/paths.py",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# artificial source\n")
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "to_stored", lambda path: Path(path).name)
    monkeypatch.setattr(audit.subprocess, "check_output", lambda *_args, **_kwargs: "artificial-revision\n")

    def fail(*_args):
        raise ValueError("artificial schema failure")

    monkeypatch.setattr(audit, "audit_artifacts", fail)
    output = tmp_path / "audit"
    result = audit.run_audit(tmp_path / "043", output)
    assert result["status"] == "failed"
    assert "artificial schema failure" in result["mismatches"][0]["traceback"]
    completion = json.loads((output / "completion.json").read_text())
    assert completion["artifacts"]["audit.json"] == audit.sha256_file(output / "audit.json")
    with pytest.raises(FileExistsError):
        audit.run_audit(tmp_path / "043", output)


def test_auditor_has_no_production_imports_model_decoding_or_inference():
    source = Path(audit.__file__).read_text()
    assert "import lead_grounding043" not in source
    assert "from ecg_experiment.lead_grounding043" not in source
    assert "joblib.load" not in source
    assert "predict_proba" not in source
