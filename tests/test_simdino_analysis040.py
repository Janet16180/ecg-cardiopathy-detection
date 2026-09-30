"""Scientific regression checks using synthetic probabilities and temporary receipts only."""

import json

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from ecg_experiment import encoder_context_analysis039 as paired
from ecg_experiment import encoder_context_interactions039 as interactions
from ecg_experiment import paths
from ecg_experiment import simdino_analysis040 as analysis
from ecg_experiment.files import sha256_file, sha256_json, write_json_atomic
from ecg_experiment.intervals import patient_groups, two_class_resamples


def _arrays():
    rng = np.random.default_rng(73)
    y = np.tile([0, 1], 12)
    values = {"targets": y, "patient_ids": np.repeat(np.arange(12), 2), "record_ids": np.arange(24)}
    names = {name for first, second, _ in analysis.comparisons().values() for name in (first, second)}
    for name in sorted(names):
        logits = rng.normal(size=(3, len(y))) + 0.5 * y
        values[name] = 1 / (1 + np.exp(-logits))
    return values


def _write_cell(root, group, seed, arrays):
    path = analysis.cell_path(root, group, seed)
    path.mkdir(parents=True)
    index = analysis.SEEDS.index(seed)
    probabilities = {name: arrays[name] for name in ("targets", "record_ids", "patient_ids")}
    scores = {}
    for budget in analysis.BUDGETS:
        scores[budget] = {}
        for context in analysis.CONTEXTS:
            p = arrays[f"{group}_{budget}_{context}"][index]
            probabilities[f"{budget}_{context}"] = p
            scores[budget][context] = {"auroc": roc_auc_score(arrays["targets"], p),
                                       "average_precision": average_precision_score(arrays["targets"], p)}
    np.savez(path / "development_predictions.npz", **probabilities)
    np.savez(path / "features.npz", fixture=np.zeros((8, 512)))
    np.savez(path / "head_parameters.npz", fixture=np.zeros(512))
    manifest = {"files_sha256": {}, "protocol_commit": "prospective-original",
                "architectures": {context: {"student_parameters": 1000,
                                            "active_student_parameters": 900 if group == "simdino" else 1000,
                                            "total_parameters": 1000 if group == "patch" else 1800}
                                  for context in analysis.CONTEXTS}}
    identity = sha256_json(manifest)
    write_json_atomic(path / "training.json", {"status": "complete", "identity_sha256": identity})
    write_json_atomic(path / "profile.json", {"gate_passed": True, "identity_sha256": identity})
    write_json_atomic(path / "manifest.json", manifest)
    attempts = [{"stage": stage, "status": "complete", "elapsed_seconds": 1}
                for stage in analysis.study.STAGES]
    write_json_atomic(path / "stage_walltime.json", {"attempts": attempts, "total_seconds": 5})
    result = {"status": "complete_development_only", "scores": scores, "calibration_or_test_scored": False,
              "identity_sha256": identity}
    for filename, key in (("development_predictions.npz", "predictions_sha256"),
                          ("features.npz", "features_sha256"),
                          ("head_parameters.npz", "head_parameters_sha256"),
                          ("training.json", "training_sha256")):
        result[key] = sha256_file(path / filename)
    write_json_atomic(path / "result.json", result)
    write_json_atomic(path / "audit.json", {"status": "passed_development_only",
                                           "result_sha256": sha256_file(path / "result.json")})
    return path


def _refresh_hashes(path):
    result = json.loads((path / "result.json").read_text())
    result["training_sha256"] = sha256_file(path / "training.json")
    write_json_atomic(path / "result.json", result)
    audit = json.loads((path / "audit.json").read_text())
    audit["result_sha256"] = sha256_file(path / "result.json")
    write_json_atomic(path / "audit.json", audit)


def _refresh_identity(path):
    identity = sha256_json(json.loads((path / "manifest.json").read_text()))
    for name in ("result", "profile", "training"):
        receipt = json.loads((path / f"{name}.json").read_text())
        receipt["identity_sha256"] = identity
        write_json_atomic(path / f"{name}.json", receipt)
    _refresh_hashes(path)


@pytest.fixture
def completed(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    values = _arrays()
    for group in analysis.GROUPS:
        for seed in analysis.SEEDS:
            _write_cell(tmp_path, group, seed, values)
    return tmp_path, values


def test_primary_family_is_exactly_six_and_component_decisions_require_both():
    pairs = analysis.comparisons()
    assert sum(primary for _, _, primary in pairs.values()) == 6
    assert pairs["limited_cpc_minus_patch_gru"] == ("cpc_limited_gru", "patch_limited_gru", True)
    assert pairs["limited_simdino_minus_cpc_gru"][2] is False
    inference = {"primary_family_size": 6,
                 "contrasts": {name: {"promising": True} for name in pairs}}
    assert analysis.combined_benefit(inference) == {"gru": True, "xlstm": True}
    inference["contrasts"]["limited_hybrid_minus_simdino_gru"]["promising"] = False
    assert analysis.combined_benefit(inference) == {"gru": False, "xlstm": True}
    inference["primary_family_size"] = 5
    assert analysis.combined_benefit(inference) == {"gru": False, "xlstm": False}


def test_bootstrap_matches_manual_shared_patient_draws_and_mean_seed_differences():
    arrays = _arrays()
    result = analysis.joint_intervals(arrays, draws=12, seed=9)
    pairs = analysis.comparisons()
    primary = [name for name, (_, _, flag) in pairs.items() if flag]
    names = {key for first, second, _ in pairs.values() for key in (first, second)}
    draws = {name: [] for name in pairs}
    rng = np.random.default_rng(9)
    for rows in two_class_resamples(patient_groups(arrays["patient_ids"]), arrays["targets"], 12, rng):
        scores = {name: np.asarray([roc_auc_score(arrays["targets"][rows], seed[rows])
                                   for seed in arrays[name]]) for name in names}
        for name, (first, second, _) in pairs.items():
            draws[name].append(np.mean(scores[first] - scores[second]))
    centered = []
    for name in primary:
        item = result["contrasts"][name]
        np.testing.assert_array_equal([item["ci_low"], item["ci_high"]],
                                      np.percentile(draws[name], [2.5, 97.5]))
        centered.append(np.asarray(draws[name]) - item["difference"])
        assert item["promising"] == (item["difference"] >= 0.005 and item["simultaneous_low"] > 0)
    assert result["simultaneous_radius"] == np.percentile(np.max(np.abs(centered), axis=0), 95)
    first, second, _ = pairs[primary[0]]
    expected = [roc_auc_score(arrays["targets"], a) - roc_auc_score(arrays["targets"], b)
                for a, b in zip(arrays[first], arrays[second], strict=True)]
    np.testing.assert_array_equal(result["contrasts"][primary[0]]["seed_differences"], expected)
    assert np.mean(expected) != (roc_auc_score(arrays["targets"], arrays[first].mean(axis=0))
                                 - roc_auc_score(arrays["targets"], arrays[second].mean(axis=0)))


def test_bootstrap_hooks_restore_after_success_and_failure(monkeypatch):
    original, original_interaction = paired.comparisons, interactions._interaction_keys
    arrays = _arrays()
    analysis.joint_intervals(arrays, draws=2)
    assert paired.comparisons is original
    result = analysis.interaction_intervals(arrays, draws=2)
    assert interactions._interaction_keys is original_interaction
    assert len(result["interactions"]) == 8

    def fail(*args, **kwargs):
        raise RuntimeError("bootstrap failed")

    monkeypatch.setattr(paired, "factorial_intervals", fail)
    with pytest.raises(RuntimeError, match="bootstrap failed"):
        analysis.joint_intervals(arrays, draws=2)
    assert paired.comparisons is original
    monkeypatch.setattr(interactions, "analyze_interactions", fail)
    with pytest.raises(RuntimeError, match="bootstrap failed"):
        analysis.interaction_intervals(arrays, draws=2)
    assert interactions._interaction_keys is original_interaction


def test_interaction_sign_matches_difference_of_differences():
    arrays = _arrays()
    result = analysis.interaction_intervals(arrays, draws=3)
    name = "limited_hybrid_minus_cpc_by_context"
    first, second, third, fourth = analysis._interaction_keys()[name]
    scores = {key: np.asarray([roc_auc_score(arrays["targets"], row) for row in arrays[key]])
              for key in (first, second, third, fourth)}
    expected = (scores[first] - scores[second]) - (scores[third] - scores[fourth])
    np.testing.assert_array_equal(result["interactions"][name]["seed_differences"], expected)
    assert result["primary_decision"] is False


def test_load_replays_all_twelve_cells_and_binds_every_consumed_artifact(completed):
    root, expected = completed
    arrays, hashes = analysis.load_cells(root)
    for name, values in expected.items():
        np.testing.assert_array_equal(arrays[name], values)
    assert len(hashes) == 12 * len(analysis.ARTIFACTS)


def test_prediction_alignment_mismatch_is_rejected(completed):
    root, arrays = completed
    altered = {name: values.copy() for name, values in arrays.items()}
    altered["patient_ids"][0] = 100
    path = analysis.cell_path(root, "hybrid", 39044)
    with np.load(path / "development_predictions.npz") as old:
        saved = {name: old[name].copy() for name in old.files}
    saved["patient_ids"] = altered["patient_ids"]
    np.savez(path / "development_predictions.npz", **saved)
    result = json.loads((path / "result.json").read_text())
    result["predictions_sha256"] = sha256_file(path / "development_predictions.npz")
    write_json_atomic(path / "result.json", result)
    write_json_atomic(path / "audit.json", {"status": "passed_development_only",
                                           "result_sha256": sha256_file(path / "result.json")})
    with pytest.raises(ValueError, match="identities or targets"):
        analysis.load_cells(root)


@pytest.mark.parametrize("filename", ["features.npz", "head_parameters.npz", "training.json"])
def test_changed_scientific_artifacts_rejected(completed, filename):
    root, _ = completed
    path = analysis.cell_path(root, "cpc", 39042) / filename
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed|Changed"):
        analysis.load_cells(root)


def test_latest_failed_attempt_prevents_complete_inference(completed):
    root, _ = completed
    path = analysis.cell_path(root, "simdino", 39043) / "stage_walltime.json"
    ledger = json.loads(path.read_text())
    ledger["attempts"].append({"stage": "train", "status": "failed", "elapsed_seconds": 0.1})
    write_json_atomic(path, ledger)
    with pytest.raises(ValueError, match="successful latest train"):
        analysis.aggregate(root)


def test_aggregate_is_immutable_and_source_drift_is_rejected(completed, monkeypatch):
    root, _ = completed
    joint, interaction = analysis.joint_intervals, analysis.interaction_intervals
    monkeypatch.setattr(analysis, "joint_intervals", lambda arrays: joint(arrays, draws=3))
    monkeypatch.setattr(analysis, "interaction_intervals", lambda arrays: interaction(arrays, draws=3))
    result = analysis.aggregate(root)
    assert result["completed_fits"] == 18
    assert analysis.aggregate(root) == result
    profile = analysis.cell_path(root, "cpc", 39042) / "profile.json"
    saved = json.loads(profile.read_text())
    saved["changed"] = True
    write_json_atomic(profile, saved)
    with pytest.raises(ValueError, match="aggregate inputs changed"):
        analysis.aggregate(root)


@pytest.mark.parametrize("directory", ["ecg_experiment", "third_party/upstream", "tests", "scripts"])
def test_changed_pinned_source_is_rejected(completed, directory):
    root, _ = completed
    source = root / directory / "scientific.py"
    source.parent.mkdir(parents=True)
    source.write_text("original")
    manifest = analysis.cell_path(root, "cpc", 39042) / "manifest.json"
    saved = json.loads(manifest.read_text())
    saved["files_sha256"][f"{directory}/scientific.py"] = sha256_file(source)
    write_json_atomic(manifest, saved)
    _refresh_identity(manifest.parent)
    source.write_text("changed")
    with pytest.raises(ValueError, match="pinned scientific source"):
        analysis.load_cells(root)


@pytest.mark.parametrize("receipt_name", ["manifest", "result", "training", "profile"])
def test_manifest_identity_cannot_drift_even_when_parent_artifact_hashes_match(completed, receipt_name):
    root, _ = completed
    path = analysis.cell_path(root, "cpc", 39042)
    receipt_path = path / f"{receipt_name}.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt_name == "manifest":
        receipt["protocol_commit"] = "different-executable-identity"
    else:
        receipt["identity_sha256"] = "0" * 64
    write_json_atomic(receipt_path, receipt)
    _refresh_hashes(path)
    with pytest.raises(ValueError, match="different manifest identity"):
        analysis.load_cells(root)


def test_incomplete_suite_has_no_family_decision_or_aggregate_file(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    _write_cell(tmp_path, "cpc", 39042, _arrays())
    result = analysis.aggregate(tmp_path)
    assert result["status"] == "incomplete_original_study"
    assert result["completed_fits"] == 2
    assert result["primary_decisions"] is None
    assert result["primary_family_complete"] is False
    assert not (tmp_path / analysis.OUTPUT / "aggregate.json").exists()


def test_failure_rule_uses_correct_seed_and_baseline():
    arrays = _arrays()
    for name in arrays:
        if arrays[name].ndim == 2:
            arrays[name] = np.tile(arrays["targets"], (3, 1)).astype(float)
    arrays["hybrid_limited_xlstm"][1] = 1 - arrays["targets"]
    failures = analysis.severe_failures(analysis._scores(arrays))
    assert failures == [{"objective": "hybrid", "context": "xlstm", "seed": 39043,
                         "auroc": 0.0, "baseline_auroc": 1.0, "difference": -1.0, "control": "cpc"}]


def test_partial_report_never_announces_complete_primary_decisions(tmp_path, monkeypatch):
    from scripts.reports.report_cpc_simdino040 import report

    monkeypatch.setattr(paths, "ROOT", tmp_path)
    _write_cell(tmp_path, "cpc", 39042, _arrays())
    path = report(tmp_path)
    text = path.read_text()
    assert "Study incomplete: 2 of 18" in text
    assert "No complete-family superiority or combined-benefit decision is available" in text
    assert "All 18 original fits completed" not in text


def test_complete_report_separates_architecture_objective_and_combination_decisions(completed, monkeypatch):
    from scripts.reports.report_cpc_simdino040 import report

    root, _ = completed
    joint, interaction = analysis.joint_intervals, analysis.interaction_intervals
    monkeypatch.setattr(analysis, "joint_intervals", lambda arrays: joint(arrays, draws=2))
    monkeypatch.setattr(analysis, "interaction_intervals", lambda arrays: interaction(arrays, draws=2))
    text = report(root).read_text()
    assert "All 18 original fits completed" in text
    assert "cpc-minus-patch contrasts test the frontend" in text
    assert "requires both component comparisons to pass" in text
    assert "diagnosis outstanding" in text
    assert "prospective-original" in text
    assert "Aggregate receipt SHA-256" in text
    assert "| 1000 | 900 |" in text
    assert "039 patch/gru reference has 1,000 active parameters" in text
