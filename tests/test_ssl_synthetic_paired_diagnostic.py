"""Artificial saved-array checks; no scientific artifacts, models or waveform payloads are opened."""

import ast
import copy
from pathlib import Path

import numpy as np
import pytest

from ecg_experiment import ssl_synthetic_paired_diagnostic as diagnostic

CONFIG = {"bootstrap_draws": 20, "bootstrap_seed": 20261026}


def test_access_boundary_and_metadata_authority_before_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    opened = []
    monkeypatch.setattr(diagnostic, "sha256_file", lambda path: opened.append(str(path)) or "ok")
    allowed = ["outputs/current/probabilities_R.npy"]
    bindings = {allowed[0]: "ok", "outputs/old/backgrounds.npy": "closed", "outputs/old/model.pt": "weights"}
    exceptions = diagnostic.verify_bindings(
        bindings, {"outputs/old/backgrounds.npy": "closed", "outputs/old/model.pt": "weights"}, allowed
    )
    assert len(opened) == 1
    assert len(exceptions) == 2
    assert all(not item["payload_read"] for item in exceptions.values())
    with pytest.raises(ValueError, match="authority"):
        diagnostic.verify_bindings({"outputs/unknown.pt": "x"}, {}, allowed)
    assert len(opened) == 1
    assert not diagnostic.payload_allowed("data/raw/clinical.json", allowed)
    assert not diagnostic.payload_allowed("outputs/other.npy", allowed)
    with pytest.raises(ValueError, match="relative"):
        diagnostic.payload_allowed("../outside.json", allowed)


def test_conflicting_declarations_reject_and_live_guard_precedes_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="Conflicting"):
        diagnostic._merge({"model.pt": "a"}, {"model.pt": "b"})

    def guard():
        raise TimeoutError("deadline")

    monkeypatch.setattr(diagnostic, "sha256_file", lambda path: pytest.fail("Opened after deadline"))
    with pytest.raises(TimeoutError, match="deadline"):
        diagnostic.verify_bindings({"outputs/test.json": "x"}, {}, [], guard)


def test_numeric_paired_flags_keep_both_flip_directions() -> None:
    result = diagnostic.transition_table(
        np.array([False, True, False, True]), np.array([False, False, True, True]), CONFIG
    )
    assert [result[key] for key in ["neither", "first_only", "second_only", "both"]] == [1, 1, 1, 1]
    assert result["second_minus_first"]["difference"] == 0.0
    assert result["second_minus_first"]["subjects"] == 4


def test_empty_descriptive_category_is_explicit_null_and_linear_quantile() -> None:
    empty = diagnostic.descriptive(np.array([]))
    assert empty["count"] == 0
    assert empty["mean"] is None
    actual = diagnostic.descriptive(np.array([0.0, 10.0]))
    assert actual["q05"] == 0.5
    assert actual["q95"] == 9.5


def test_positive_regions_float64_and_ap_use_same_fixed_mask() -> None:
    baseline = np.array([[0.8, 0.8, 0.8, 0.8]], dtype=np.float32)
    edited = np.array([[0.9, 0.7, 0.2, 0.1]], dtype=np.float32)
    target = np.array([[True, False, False, False]])
    candidate = np.array([[True, True, False, False]])
    result = diagnostic.positive_statistics(edited, baseline, target, candidate)
    assert result["baseline_site_average_precision"] == 0.25
    assert result["edited_average_precision"] == 1.0
    assert result["average_precision_change"] == 0.75
    assert result["change_T_minus_U"] == pytest.approx(
        float(np.float64(edited[0, 0]) - np.float64(edited[0, 1]))
    )
    assert result["full_map_mean_absolute_change"] == float(
        np.abs(edited.astype(np.float64) - baseline.astype(np.float64)).mean()
    )
    with pytest.raises(ValueError, match="nonempty"):
        diagnostic.positive_statistics(edited, baseline, target, target)


def _cases() -> list[dict]:
    rows = []
    for subject in range(692):
        role = "train" if subject < 500 else "calibration" if subject < 564 else "evaluation"
        first = 0 if role == "train" else 500 if role == "calibration" else 564
        for offset, variant in enumerate(["focal", "repeated", "sham", "nuisance"]):
            rows.append(
                {
                    "case_index": subject * 4 + offset,
                    "variant": variant,
                    "patient_id": str(subject),
                    "role": role,
                    "role_subject_index": subject - first,
                    "cache_index": subject,
                    "subject_index": subject,
                    "nuisance": False,
                    "matched_control_case_index": subject * 4 + 2,
                    "baseline_sha256": "same",
                    "edited_sha256": "same",
                    "nuisance_parameters": {},
                }
            )
    return rows


def test_global_case_offsets_and_exact_counterpart_identity() -> None:
    cases = _cases()
    groups = diagnostic.case_groups(cases)
    assert groups["calibration"][0][0]["case_index"] == 2000
    assert groups["evaluation"][0][0]["case_index"] == 2256
    assert groups["evaluation"][-1][-1]["case_index"] == 2767
    damaged = copy.deepcopy(cases)
    damaged[2256]["matched_control_case_index"] = 2002
    with pytest.raises(ValueError, match="matched-control"):
        diagnostic.case_groups(damaged)
    damaged = copy.deepcopy(cases)
    damaged[2256]["role_subject_index"] = 564
    with pytest.raises(ValueError, match="coordinates"):
        diagnostic.case_groups(damaged)


def test_training_six_milestones_and_all_600_losses() -> None:
    values = np.arange(600, dtype=float) / 600
    training = {
        "arms": {
            arm: {"losses": values.tolist(), "first_loss": values[0], "last_loss": values[-1]}
            for arm in ["R", "S"]
        }
    }
    logs = [
        {"arm": arm, "step": step, "loss": values[step - 1]}
        for arm in ["R", "S"]
        for step in range(100, 601, 100)
    ]
    result = diagnostic.loss_blocks(training, logs)
    assert result["R"]["logged_milestones"] == 6
    assert len(result["R"]["blocks"]) == 6
    assert result["R"]["blocks"][0]["std_ddof0"] == float(values[:100].std(ddof=0))
    wrong = copy.deepcopy(logs)
    wrong[0]["loss"] += 0.01
    with pytest.raises(ValueError, match="milestone"):
        diagnostic.loss_blocks(training, wrong)
    training["arms"]["S"]["losses"][10] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        diagnostic.loss_blocks(training, logs)


@pytest.mark.parametrize(("wall_now", "cpu_now"), [(119.0, 0.0), (0.0, 119.0)])
def test_wall_and_cpu_deadlines_preserve_failure_reserve(
    monkeypatch: pytest.MonkeyPatch, wall_now: float, cpu_now: float
) -> None:
    monkeypatch.setattr(diagnostic.time, "perf_counter", lambda: wall_now)
    monkeypatch.setattr(diagnostic.time, "process_time", lambda: cpu_now)
    context = {"config": {"cpu_seconds_run_ceiling": 120, "failure_reporting_reserve_seconds": 2}}
    with pytest.raises(TimeoutError, match="run allowance"):
        diagnostic._guard(context, 0.0, 0.0)


def test_source_has_no_model_or_waveform_reader_imports() -> None:
    tree = ast.parse(Path(diagnostic.__file__).read_text())
    imported = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(
        name and ("model" in name or "training" in name or "evaluation" in name or "common" in name)
        for name in imported
    )


def test_control_threshold_ties_are_not_flags() -> None:
    masks = np.zeros((2004, 1, 4), bool)
    candidate = masks.copy()
    candidate[2002:, :, :2] = True
    maps = np.full((4, 1, 4), 0.5, np.float32)
    maps[3, 0, 0] = np.nextafter(np.float32(0.5), np.float32(1.0))
    group = [{}, {}, {"case_index": 2002}, {"case_index": 2003}]
    context = {
        "groups": {role: [group] for role in diagnostic.ROLES},
        "maps": dict.fromkeys(diagnostic.ARMS, maps),
        "support": masks,
        "candidate": candidate,
        "thresholds": {"R": 0.5, "S": 0.5},
    }
    rows = diagnostic.control_rows(context)
    assert all(not row["sham_flag"] for row in rows)
    assert all(row["nuisance_flag"] for row in rows)
    assert all(row["sham_margin"] == 0.0 for row in rows)


def test_failed_admission_retained_without_payload_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    monkeypatch.setattr(diagnostic, "DESTINATION", tmp_path / "diagnostic")
    monkeypatch.setattr(diagnostic, "_guard", lambda *args: None)

    def stop(guard=None):
        raise ValueError("Anchored receipt changed")

    monkeypatch.setattr(diagnostic, "admit_diagnostic", stop)
    for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(RuntimeError, match="diagnostic stopped"):
        diagnostic.diagnose_paired_background()
    failure = json.loads((tmp_path / "diagnostic/failure.json").read_text())
    assert failure["progress"]["positive_rows"] == 0
    assert failure["waveform_payload_reads"] == 0
    assert failure["checkpoint_payload_reads"] == 0
    with pytest.raises(FileExistsError, match="must be new"):
        diagnostic.diagnose_paired_background()
