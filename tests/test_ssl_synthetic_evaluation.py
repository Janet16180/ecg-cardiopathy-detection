"""Artificial-only checks of frozen evaluation, truth separation and failure retention."""

import contextlib
import copy
import io
import json

import numpy as np
import pytest
import torch

from ecg_experiment import ssl_synthetic_evaluation as evaluation
from ecg_experiment.ssl_synthetic_metrics import control_threshold
from ecg_experiment.ssl_synthetic_model import state_digest
from ecg_experiment.ssl_synthetic_model_v2 import DenseWaveformModel


@pytest.fixture
def budget():
    return {"wall_deadline": float("inf"), "cpu_deadline": float("inf")}


@pytest.fixture
def bank():
    signals = np.zeros((8, 12, 2500), dtype=np.float32)
    signals[[0, 1, 4, 5], 6, 1200] = 0.02
    return {
        "signals": signals,
        "baselines": np.zeros_like(signals),
        "support": signals > 0.005,
        "prior": np.zeros_like(signals, dtype=bool),
        "subjects": np.repeat(np.arange(2), 4),
        "kinds": np.tile(["positive", "positive", "sham", "nuisance"], 2),
        "modes": np.tile(["focal", "persistent", "focal", "focal"], 2),
        "metadata": [],
    }


@pytest.fixture
def training():
    arms = {
        arm: {
            "steps": 600,
            "presentations": 9600,
            "initialization": {
                "head_sha256": "same-head",
                "encoder_sha256": "fresh" if arm == "R" else "ssl",
            },
            "final_state": {"encoder_sha256": "ssl" if arm == "F" else "trained"},
        }
        for arm in evaluation.ARMS
    }
    return {
        "training_completed": True,
        "performance_scored": False,
        "ssl": {"steps": 1000, "presentations": 16000, "encoder_sha256": "ssl"},
        "initialization": {"head_sha256": "same-head", "encoder_sha256": "fresh"},
        "arms": arms,
    }


def test_partial_training_and_initialization_mismatch_reject_before_scores(training):
    config = {"ssl_steps": 1000, "localization_steps": 600}
    evaluation.check_training_contract(training, config)
    partial = copy.deepcopy(training)
    partial["arms"]["S"]["steps"] = 599
    with pytest.raises(ValueError, match="Incomplete matched"):
        evaluation.check_training_contract(partial, config)
    mismatched = copy.deepcopy(training)
    mismatched["arms"]["F"]["initialization"]["head_sha256"] = "changed"
    with pytest.raises(ValueError, match="head initialization"):
        evaluation.check_training_contract(mismatched, config)
    mismatched["arms"]["F"]["initialization"]["head_sha256"] = "same-head"
    mismatched["arms"]["F"]["final_state"]["encoder_sha256"] = "changed"
    with pytest.raises(ValueError, match="Frozen encoder"):
        evaluation.check_training_contract(mismatched, config)


def test_conflicting_receipt_source_hashes_reject():
    target = {"source.py": "original"}
    with pytest.raises(ValueError, match="Receipts disagree"):
        evaluation._merge_hashes(target, {"source.py": "changed"})


def test_single_signal_inference_never_consumes_counterparts_or_parameters(tmp_path, bank, budget):
    observed = []

    class RecordingModel(torch.nn.Module):
        def forward(self, signal):
            observed.append(signal.detach().numpy().copy())
            return torch.zeros_like(signal)

    bank["baselines"][:] = -999
    maps = evaluation.infer_probability_maps(
        RecordingModel(), bank["signals"], tmp_path / "maps.npy", budget, batch_size=3, device="cpu"
    )
    np.testing.assert_array_equal(np.concatenate(observed), bank["signals"])
    np.testing.assert_array_equal(maps, np.full_like(bank["signals"], 0.5))
    assert maps.dtype == np.float32
    with pytest.raises(FileExistsError):
        evaluation.infer_probability_maps(
            RecordingModel(), bank["signals"], tmp_path / "maps.npy", budget, device="cpu"
        )


def test_nonfinite_model_logits_abort_and_retain_partial_map(tmp_path, bank, budget):
    class InvalidModel(torch.nn.Module):
        def forward(self, signal):
            return torch.full_like(signal, float("inf"))

    with pytest.raises(ValueError, match="nonfinite native"):
        evaluation.infer_probability_maps(
            InvalidModel(), bank["signals"], tmp_path / "partial.npy", budget, device="cpu"
        )
    assert (tmp_path / "partial.npy").exists()


def test_frozen_final_state_loading_checks_component_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "TRAINING", tmp_path)
    model = DenseWaveformModel(width=4)
    torch.save(model.state_dict(), tmp_path / "R.pt")
    final = {
        "encoder_sha256": state_digest(model.encoder.state_dict()),
        "head_sha256": state_digest(model.head.state_dict()),
    }
    training = {"arms": {"R": {"final_state": final}}}
    config = {"encoder_width": 4, "encoder_stride": 5, "physical_scale_mv": 2.0}
    loaded = evaluation.load_final_model("R", training, config)
    assert not loaded.training
    assert state_digest(loaded.state_dict()) == state_digest(model.state_dict())
    final["head_sha256"] = "changed"
    with pytest.raises(ValueError, match="component bytes"):
        evaluation.load_final_model("R", training, config)


def test_validation_threshold_aligns_controls_by_subject_and_uses_higher_quantile():
    n = 128
    sham, nuisance = np.linspace(0.1, 0.6, n), np.linspace(0.6, 0.1, n)
    bank = {"subjects": np.tile(np.arange(n), 2), "kinds": np.repeat(["sham", "nuisance"], n)}
    maps = np.concatenate((sham, nuisance))[:, None, None]
    permutation = np.random.default_rng(3).permutation(2 * n)
    scrambled = {key: value[permutation] for key, value in bank.items()}
    actual = evaluation.fit_calibration(scrambled, dict.fromkeys(evaluation.ARMS, maps[permutation]))
    assert set(actual["thresholds"].values()) == {control_threshold(sham, nuisance)}
    threshold = actual["thresholds"]["R"]
    assert not threshold > threshold
    assert actual["validation_subjects"] == 128
    assert actual["record_flag_comparison"] == "strictly greater"


def test_invalid_or_missing_counterfactual_cases_cannot_be_dropped(bank):
    evaluation.validate_bank(bank, 2)
    bad = dict(bank, support=bank["support"].copy())
    bad["support"][0] = False
    with pytest.raises(ValueError, match="Counterfactual support"):
        evaluation.validate_bank(bad, 2)
    bad = dict(bank, kinds=bank["kinds"].copy())
    bad["kinds"][3] = "sham"
    with pytest.raises(ValueError, match="Missing/duplicated"):
        evaluation.validate_bank(bad, 2)


def test_full_map_metrics_ignore_candidate_restrictions_and_retain_tied_area(bank, budget):
    probabilities = {arm: bank["support"].astype(np.float32) for arm in evaluation.ARMS}
    rows = evaluation._method_rows(bank, probabilities, dict.fromkeys(evaluation.ARMS, 1.0), "B", budget)
    model_rows = [row for row in rows if row["method"] == "R" and row["kind"] == "positive"]
    assert all(row["average_precision"] == 1.0 for row in model_rows)
    assert all(row["top_one_percent_precision"] == 1 / 300 for row in model_rows)
    assert all(row["top_one_percent_recall"] == 1.0 for row in model_rows)
    assert all(not row["flag"] for row in rows if row["method"] in evaluation.ARMS)
    uniform_rows = [row for row in rows if row["method"] == "uniform"]
    assert all(row["average_precision"] == row["prevalence"] for row in uniform_rows)
    assert all(row["top_one_percent_recall"] == 0.01 for row in uniform_rows)
    summary = evaluation.summarize_profile(rows, {"bootstrap_draws": 20, "bootstrap_seed": 3})
    assert summary["s_minus_r_average_precision"]["all"]["subjects"] == 2
    assert summary["s_minus_r_average_precision"]["all"]["cases"] == 4
    assert summary["s_minus_r_average_precision"]["all"]["difference"] == 0
    assert summary["record_flags"]["S"]["positive"]["rate"] == 0


def test_engineering_rule_uses_point_control_increase_and_positive_ap_lower_bound():
    primary = {"difference": 0.05, "ci_low": 0.001}
    controls = {name: {"difference": 0.05, "ci_high": 0.9} for name in ["sham", "nuisance"]}
    assert evaluation.advancement(primary, controls)["engineering_advancement_passed"]
    assert not evaluation.advancement(dict(primary, ci_low=0), controls)["engineering_advancement_passed"]
    controls["nuisance"]["difference"] = 0.050000001
    assert not evaluation.advancement(primary, controls)["engineering_advancement_passed"]


def test_calibration_file_is_frozen_before_any_b_render_or_inference(tmp_path, monkeypatch, bank, budget):
    monkeypatch.setattr(evaluation, "EVALUATION", tmp_path)
    monkeypatch.setattr(evaluation, "load_bank", lambda path: bank)
    events = []
    maps = {arm: np.full_like(bank["signals"], 0.5) for arm in evaluation.ARMS}

    def inference(actual_bank, profile, *args):
        assert actual_bank is bank
        if profile != "validation":
            assert (tmp_path / "calibration.json").exists()
        events.append(profile)
        for arm in evaluation.ARMS:
            np.save(tmp_path / f"{profile}_prob_{arm}.npy", maps[arm])
        return maps

    def build(role, count, profile, *args):
        assert (tmp_path / "calibration.json").exists()
        events.append("build" + profile)
        return bank

    empty_summary = {
        "s_minus_r_average_precision": {"all": {"difference": 0, "ci_low": 0}},
        "s_minus_r_control_flag_rates": {name: {"difference": 0} for name in ["sham", "nuisance"]},
    }
    monkeypatch.setattr(evaluation, "_infer_arms", inference)
    monkeypatch.setattr(evaluation, "build_bank", build)
    monkeypatch.setattr(evaluation, "save_bank", lambda *args: None)
    monkeypatch.setattr(evaluation, "_method_rows", lambda *args: [])
    monkeypatch.setattr(evaluation, "summarize_profile", lambda *args: empty_summary)
    progress = {"calibration_frozen": False, "completed_profiles": [], "scored_positive_maps": 0}
    evaluation._run(
        {},
        {"validation_subjects": 2, "evaluation_subjects": 2, "cache_bytes_ceiling": 4294967296},
        {},
        {},
        budget,
        progress,
    )
    assert events == ["validation", "buildB", "B", "buildA", "A"]
    assert progress["calibration_frozen"]


def test_preflight_source_failure_is_bound_and_retained_without_inference(tmp_path, monkeypatch):
    output = tmp_path / "evaluation"
    monkeypatch.setattr(evaluation, "EVALUATION", output)
    monkeypatch.setattr(evaluation, "gpu_lock", lambda *args, **kwargs: contextlib.nullcontext())

    def broken_source(config):
        raise ValueError("Frozen input/source hash changed")

    monkeypatch.setattr(evaluation, "_inputs", broken_source)
    with pytest.raises(ValueError, match="source hash changed"):
        evaluation.evaluate()
    failure = json.loads((output / "failure.json").read_text())
    assert not failure["evaluation_completed"]
    assert failure["source_sha256"]
    assert failure["progress"]["predicted_record_maps"] == 0
    assert not failure["progress"]["calibration_frozen"]
    assert not (output / "summary.json").exists()


def test_storage_projection_counts_exact_npy_headers_and_all_native_shapes():
    buffer = io.BytesIO()
    array = np.zeros((2, 12, 10), dtype=np.float32)
    np.save(buffer, array, allow_pickle=False)
    assert evaluation._npy_bytes(array.shape, "float32") == buffer.tell()
    result = evaluation.storage_projection(
        {"samples": 2500, "validation_subjects": 128, "evaluation_subjects": 256}
    )
    assert result["validation_probability_shape"] == (512, 12, 2500)
    assert result["evaluation_probability_shape"] == (1024, 12, 2500)
    assert result["probability_files"] == 9
    assert result["bank_files"] == 14
    assert result["probability_npy_bytes"] == 921601152
    assert result["projected_new_bytes"] == (
        result["probability_npy_bytes"] + result["bank_npy_bytes"] + 64 * 1024**2
    )


def test_real_training_payload_digest_is_bound_without_opening_the_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "BASE", tmp_path)
    monkeypatch.setattr(evaluation, "to_stored", str)
    checked = []
    monkeypatch.setattr(evaluation, "verify_hashes", lambda values: checked.append(values))
    cache = tmp_path / "real_cache_v1"
    cache.mkdir()
    (cache / "summary.json").write_text(json.dumps({"output_sha256": {"signals.npy": "frozen"}}))
    signal = str(cache / "signals.npy")
    evaluation.verify_inference_hashes({signal: "frozen", "checkpoint.pt": "weight-hash"})
    assert checked == [{"checkpoint.pt": "weight-hash"}]
    assert not (cache / "signals.npy").exists()
    with pytest.raises(ValueError, match="Real-cache provenance digest disagrees"):
        evaluation.verify_inference_hashes({signal: "changed"})
