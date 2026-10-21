"""Artificial-only transfer execution checks; no selected patient cache or GPU is opened."""

import contextlib
import copy
import hashlib
import json
import time

import numpy as np
import pytest
import torch

from ecg_experiment import ssl_synthetic_real_transfer as transfer


def plan():
    output = []
    for global_subject in range(192):
        role = "calibration" if global_subject < 64 else "evaluation"
        local = global_subject if global_subject < 64 else global_subject - 64
        for offset, variant in enumerate(["focal", "repeated", "sham", "nuisance"]):
            positive = offset < 2
            output.append(
                {
                    "case_index": global_subject * 4 + offset,
                    "cache_index": global_subject,
                    "subject_index": global_subject,
                    "subject": global_subject,
                    "role_subject_index": local,
                    "role": role,
                    "profile": "A" if global_subject < 64 else "B",
                    "patient_id": f"artificial_{global_subject}",
                    "seed": global_subject,
                    "variant": variant,
                    "kind": "positive" if positive else variant,
                    "mode": variant if positive else "none",
                    "target": positive,
                    "nuisance": bool(global_subject & 1) if positive else variant == "nuisance",
                }
            )
    return output


def test_exact_global_cases_and_nonzero_evaluation_subject_coordinates():
    cases = plan()
    roles = transfer.validate_case_plan(cases)
    assert roles["calibration"] == list(range(256))
    assert roles["evaluation"] == list(range(256, 768))
    assert cases[256]["role_subject_index"] == 0
    assert cases[256]["cache_index"] == 64
    bad = copy.deepcopy(cases)
    bad[256]["subject_index"] = 0
    with pytest.raises(ValueError, match="Global cache"):
        transfer.validate_case_plan(bad)
    bad = copy.deepcopy(cases)
    bad[256]["case_index"] = 0
    with pytest.raises(ValueError, match="global case grid"):
        transfer.validate_case_plan(bad)


def test_conditional_gate_uses_only_original_b_all_mode_for_both_arms():
    metrics = {name: {"ci_low": 0.001} for name in ["average_precision_change", "selectivity_contrast"]}
    receipt = {
        "audit_completed": True,
        "original_advancement_passed": False,
        "profiles": {"B": {"arms": {arm: {"all": copy.deepcopy(metrics)} for arm in transfer.ARMS}}},
    }
    assert all(transfer.conditional_admission(receipt).values())
    receipt["profiles"]["B"]["arms"]["R"]["all"]["selectivity_contrast"]["ci_low"] = 0
    with pytest.raises(ValueError, match="admission failed"):
        transfer.conditional_admission(receipt)


def test_storage_counts_all_five_headers_once_and_subtracts_existing_preparation(tmp_path, monkeypatch):
    study = tmp_path / "study"
    study.mkdir()
    (study / "existing").write_bytes(b"a" * 1000)
    monkeypatch.setattr(transfer, "STUDY", study)
    monkeypatch.setattr(transfer, "stream_bytes", lambda: 2_000_001_000)
    result = transfer.storage_projection()
    assert result["total_array_payload_bytes"] == 253440000
    assert result["total_array_npy_bytes"] == 253440640
    assert result["additional_bytes_required"] == 253440640 + 64 * 1024**2 - 1000
    assert result["projected_whole_stream_bytes"] == 2_000_000_000 + 253440640 + 64 * 1024**2


def test_actual_batch_forecast_uses_96_batches_double_safety_and_reserve():
    actual = transfer.profile_projection(1, 0.5, 10, 5, 4)
    assert actual["inference_batches"] == 96
    assert actual["projected_stage_gpu_wall_seconds"] == 322
    assert actual["projected_stage_cpu_seconds"] == 225


def test_b_values_cannot_affect_new_calibration_and_calibration_slice_digest():
    cases = plan()
    maps = {arm: np.arange(768, dtype=np.float32)[:, None, None] / 1000 for arm in transfer.ARMS}
    first = transfer.calibration_thresholds(cases, maps)
    digest = transfer._slice_hash(maps["R"], list(range(256)))
    for values in maps.values():
        values[256:] = 999
    assert transfer.calibration_thresholds(cases, maps) == first
    assert transfer._slice_hash(maps["R"], list(range(256))) == digest
    assert first["R"] == float(np.float32(0.243))


def test_rendered_input_hashes_and_masks_checked_without_real_waveform_voltage_gate(monkeypatch):
    def digest(value):
        return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()

    background = np.full((12, 2500), 20, dtype=np.float32)
    edited = background.copy()
    edited[0, 1000:1020] += 0.02
    support = np.abs(edited.astype(float) - background.astype(float)) > 0.005
    candidate = np.zeros_like(support)
    candidate[:, 990:1100] = True
    rendered = {
        "edited": edited,
        "baseline": background,
        "support": support,
        "candidate": candidate,
        "metadata": {},
    }
    case = {
        "cache_index": 0,
        "processed_background_sha256": digest(background),
        **{
            f"{name}_sha256": digest(rendered[name])
            for name in ["edited", "baseline", "support", "candidate"]
        },
    }
    context = {
        "cases": [case],
        "backgrounds": background[None],
        "support": support[None],
        "candidate": candidate[None],
        "regeneration_checks": {},
    }
    monkeypatch.setattr(transfer, "_background_api", lambda: (lambda source, case: rendered, digest))
    assert transfer.render_checked(context, 0)["edited"].max() > 10
    assert len(context["regeneration_checks"]) == 1
    case["edited_sha256"] = "changed"
    with pytest.raises(ValueError, match="immutable case hash"):
        transfer.render_checked(context, 0)


def test_observed_single_signal_only_reaches_model(monkeypatch):
    observed = np.full((12, 2500), 0.1, dtype=np.float32)
    monkeypatch.setattr(
        transfer, "render_checked", lambda context, index: {"edited": observed, "baseline": observed - 999}
    )
    seen = []

    class RecordingModel(torch.nn.Module):
        def forward(self, signal):
            seen.append(signal.numpy().copy())
            return torch.zeros_like(signal)

    output = transfer._predict_batch(RecordingModel(), {}, [256, 257], device="cpu")
    np.testing.assert_array_equal(seen[0], np.stack([observed, observed]))
    np.testing.assert_array_equal(output, np.full((2, 12, 2500), 0.5, dtype=np.float32))


def test_global_eval_map_row_256_becomes_within_role_patient_zero(monkeypatch):
    cases = plan()[256:260]
    truth = np.zeros((12, 2500), dtype=bool)
    truth[0, 1000:1020] = True
    prior = np.zeros_like(truth)
    prior[:, 990:1100] = True
    blank = np.zeros((12, 2500), dtype=np.float32)
    context = {
        "cases": cases,
        "support": dict.fromkeys([256, 257], truth),
        "candidate": dict.fromkeys([256, 257], prior),
    }
    maps = {
        arm: {256: truth.astype(np.float32), 257: truth.astype(np.float32), 258: blank, 259: blank}
        for arm in transfer.ARMS
    }
    monkeypatch.setattr(transfer, "_guard", lambda *args: None)
    monkeypatch.setattr(
        transfer, "render_checked", lambda context, index: {"edited": blank, "baseline": blank}
    )
    rows, responses = transfer._case_rows(context, maps, dict.fromkeys(transfer.ARMS, 0.5), {})
    assert {row["subject"] for row in rows} == {0}
    assert {row["global_subject"] for row in rows} == {64}
    assert {row["control_case"] for row in responses} == {258}
    selected = [row for row in rows if row["method"] == "R" and row["kind"] == "positive"]
    assert all(row["average_precision"] == 1 for row in selected)
    summary = transfer.summarize(rows, responses, {"bootstrap_draws": 20, "bootstrap_seed": 20261023})
    assert summary["s_minus_r_ap"]["all"]["subjects"] == 1
    assert summary["s_minus_r_ap"]["all"]["cases"] == 2
    assert summary["s_minus_r_ap"]["all"]["difference"] == 0
    assert summary["new_advancement_rule"] is None


def test_live_stage_deadline_stops_before_next_batch(monkeypatch):
    monkeypatch.setattr(transfer, "stream_bytes", lambda: 0)
    context = {
        "config": {"gpu_seconds_stage_ceiling": 900, "cpu_seconds_stage_ceiling": 1800},
        "prior_gpu_wall_seconds": 0,
        "prior_cpu_seconds": 0,
    }
    clock = {
        "wall": time.perf_counter() - 901,
        "cpu": time.process_time(),
        "prior_stage_wall": 0,
        "prior_stage_cpu": 0,
    }
    with pytest.raises(TimeoutError, match="resource ceiling"):
        transfer._guard(context, clock)


def test_failed_qualification_is_not_opened_as_model_input(tmp_path, monkeypatch):
    monkeypatch.setattr(transfer, "PREPARATION", tmp_path)
    (tmp_path / "summary.json").write_text(json.dumps({"qualification_gate_passed": False}))
    with pytest.raises(ValueError, match="no profile/inference"):
        transfer._admission()


def test_profile_preflight_failure_retains_new_receipt_without_cuda(tmp_path, monkeypatch):
    monkeypatch.setattr(transfer, "PROFILE", tmp_path / "profile")
    monkeypatch.setattr(transfer, "gpu_lock", lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(transfer, "_thread_environment", lambda: {})

    def reject():
        raise ValueError("Source/checkpoint binding failed")

    monkeypatch.setattr(transfer, "_admission", reject)
    with pytest.raises(RuntimeError, match="local failure receipt"):
        transfer.profile_transfer()
    failure = json.loads((transfer.PROFILE / "failure.json").read_text())
    assert failure["progress"]["timed_batches"] == 0
    assert failure["source_sha256"]
    with pytest.raises(FileExistsError):
        transfer.profile_transfer()
