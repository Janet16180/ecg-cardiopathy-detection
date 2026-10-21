"""Artificial saved-map checks; no checkpoint, real waveform or GPU is used."""

import copy
import json

import numpy as np
import pytest

from ecg_experiment import ssl_synthetic_counterfactual as audit
from ecg_experiment.paths import to_stored


@pytest.fixture
def matched_bank():
    signals = np.zeros((1024, 1, 6), dtype=np.float32)
    baselines = np.zeros_like(signals)
    truth = np.zeros_like(signals, dtype=bool)
    prior = np.zeros_like(truth)
    prior[:, :, 1:4] = True
    metadata = []
    for subject in range(256):
        seed = 10000 + subject
        noisy = bool(subject % 2)
        clean = np.full((1, 6), 0.1 + subject * 0.00001, dtype=np.float32)
        nuisance = clean + 0.01
        for offset, kind in enumerate(["positive", "positive", "sham", "nuisance"]):
            index = subject * 4 + offset
            flag = noisy if kind == "positive" else kind == "nuisance"
            values = nuisance if flag else clean
            signals[index] = values
            baselines[index] = values
            if kind == "positive":
                signals[index, 0, 1] += 0.02
                truth[index, 0, 1] = True
            metadata.append(
                {
                    "subject": subject,
                    "seed": seed,
                    "nuisance": flag,
                    "parameters": {"latent": {"seed": seed}, "nuisance": {"enabled": flag}, "profile": "B"},
                }
            )
    bank = {
        "signals": signals,
        "baselines": baselines,
        "support": truth,
        "prior": prior,
        "subjects": np.repeat(np.arange(256), 4),
        "kinds": np.tile(["positive", "positive", "sham", "nuisance"], 256),
        "modes": np.tile(["focal", "persistent", "focal", "focal"], 256),
    }
    return bank, metadata


def test_complete_counterparts_use_same_latent_and_actual_nuisance(matched_bank):
    bank, metadata = matched_bank
    pairs = audit.match_counterparts(bank, metadata)
    assert len(pairs) == 512
    assert pairs[0]["control_case"] == 2
    assert pairs[2]["control_case"] == 7
    assert {pair["T_cells"] for pair in pairs} == {1}
    assert {pair["U_cells"] for pair in pairs} == {2}
    assert {pair["O_cells"] for pair in pairs} == {3}


def test_even_one_changed_counterpart_byte_rejects_all_comparison(matched_bank):
    bank, metadata = matched_bank
    value = bank["baselines"][0, 0, 0]
    bank["baselines"][0, 0, 0] = np.nextafter(value, np.float32(1))
    with pytest.raises(ValueError, match="byte-identical"):
        audit.match_counterparts(bank, metadata)


def test_wrong_latent_or_missing_unselected_control_rejects(matched_bank):
    bank, metadata = matched_bank
    wrong = copy.deepcopy(metadata)
    wrong[2]["seed"] = 3
    with pytest.raises(ValueError, match="latent identity"):
        audit.match_counterparts(bank, wrong)
    bank["kinds"][3] = "sham"
    with pytest.raises(ValueError, match="subject variant"):
        audit.match_counterparts(bank, metadata)


def test_original_audit_must_bind_current_failed_pilot_before_diagnostic():
    key = to_stored(audit.ORIGINAL / "summary.json")
    valid = {"all_passed": True, "mismatch_count": 0, "input_sha256": {key: "original"}}
    audit.require_original_audit(valid, "original")
    with pytest.raises(ValueError, match="Original independent reproduction"):
        audit.require_original_audit(valid, "different")
    with pytest.raises(ValueError, match="Original independent reproduction"):
        audit.require_original_audit(dict(valid, mismatch_count=1), "original")


def test_high_baseline_site_ap_with_identical_maps_has_zero_input_response():
    truth = np.array([[False, True, False, False, False, False]])
    prior = np.array([[False, True, True, True, False, False]])
    unchanged = np.array([[0.2, 0.8, 0.2, 0.2, 0.2, 0.2]], dtype=np.float32)
    result = audit.response_metrics(truth, prior, unchanged, unchanged)
    assert result["edited_average_precision"] == result["baseline_site_average_precision"] == 1
    assert result["average_precision_change"] == 0
    assert result["mean_probability_change_T"] == 0
    assert result["mean_probability_change_U"] == 0
    assert result["mean_probability_change_O"] == 0
    assert result["selectivity_contrast"] == 0


def test_response_partition_and_same_mask_metric_differences():
    truth = np.array([[False, True, False, False, False, False]])
    prior = np.array([[False, True, True, True, False, False]])
    baseline = np.full((1, 6), 0.2)
    edited = np.array([[0.2, 0.8, 0.3, 0.3, 0.2, 0.2]])
    result = audit.response_metrics(truth, prior, edited, baseline)
    assert result["mean_probability_change_T"] == pytest.approx(0.6)
    assert result["mean_probability_change_U"] == pytest.approx(0.1)
    assert result["selectivity_contrast"] == pytest.approx(0.5)
    assert result["edited_average_precision"] == 1
    assert result["baseline_site_average_precision"] == 1 / 6
    assert result["average_precision_change"] == pytest.approx(5 / 6)
    with pytest.raises(ValueError, match="Empty counterfactual region"):
        audit.region_counts(truth, np.ones_like(truth))
    with pytest.raises(ValueError, match="declared partition"):
        audit.region_counts(truth, np.zeros_like(truth))
    with pytest.raises(ValueError, match="Nonfinite"):
        audit.response_metrics(truth, prior, edited * np.nan, baseline)


def test_nuisance_metric_keeps_original_threshold_and_strict_ties():
    sham, nuisance = np.array([[0.2, 0.5]]), np.array([[0.4, 0.6]])
    at_tie = audit.nuisance_metrics(sham, nuisance, 0.6)
    assert at_tie["mean_absolute_nuisance_change"] == pytest.approx(0.15)
    assert not at_tie["nuisance_flag"]
    assert at_tie["flag_change"] == 0
    flagged = audit.nuisance_metrics(sham, nuisance, 0.55)
    assert flagged["flag_change"] == 1


def test_subject_macro_and_paired_exploratory_bootstrap_keep_both_modes():
    rows = []
    for arm in audit.ARMS:
        for subject, mode, value in [
            (0, "focal", 0.1),
            (0, "persistent", 0.9),
            (1, "focal", 0.2),
            (1, "persistent", 0.2),
        ]:
            value += 0.1 if arm == "S" else 0
            rows.append(
                {
                    "method": arm,
                    "subject": subject,
                    "mode": mode,
                    "case": subject * 4 + (mode == "persistent"),
                    **dict.fromkeys(audit.RESPONSE_METRICS, value),
                }
            )
    result = audit.summarize_responses(rows, {"bootstrap_draws": 20, "bootstrap_seed": 4})
    actual = result["arms"]["R"]["all"]["average_precision_change"]
    assert actual["mean"] == pytest.approx(0.35)
    assert actual["subjects"] == 2
    assert actual["cases"] == 4
    assert result["arms"]["R"]["focal"]["average_precision_change"]["mean"] == pytest.approx(0.15)
    assert result["exploratory_s_minus_r"]["all"]["average_precision_change"]["difference"] == pytest.approx(
        0.1
    )


def test_failed_original_audit_is_retained_without_model_or_map_access(tmp_path, monkeypatch):
    destination = tmp_path / "audit"
    monkeypatch.setattr(audit, "DESTINATION", destination)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        monkeypatch.setenv(name, "1")

    def reject(path):
        raise ValueError("Original independent reproduction failed")

    monkeypatch.setattr(audit, "_admission", reject)
    with pytest.raises(ValueError, match="reproduction failed"):
        audit.audit_counterfactual(tmp_path / "original_audit.json")
    failure = json.loads((destination / "failure.json").read_text())
    assert not failure["audit_completed"]
    assert failure["progress"]["verified_positive_pairs"] == 0
    assert not failure["progress"]["completed_profiles"]
    assert failure["source_sha256"]
