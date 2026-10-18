"""Real-helper regression for the complete corrected four-arm case scorer, with artificial data only."""

import ast
from pathlib import Path

import numpy as np
import pytest

from ecg_experiment import ssl_synthetic_counterfactual_factorial_common as common
from ecg_experiment import ssl_synthetic_counterfactual_factorial_evaluation as pinned
from ecg_experiment import ssl_synthetic_counterfactual_factorial_evaluation_v2 as successor


def test_complete_case_scorer_real_helper_manual_ap_and_tuo(monkeypatch: pytest.MonkeyPatch) -> None:
    target = np.array([[True, False, False, False]])
    candidate = np.array([[True, True, False, False]])
    before = np.array([[0.2, 0.4, 0.3, 0.1]], np.float32)
    after = np.array([[0.7, 0.5, 0.2, 0.1]], np.float32)
    cases = [{} for _ in range(2768)]
    for subject in range(128):
        for variant in range(4):
            index = 2256 + subject * 4 + variant
            nuisance = bool(subject % 2) if variant < 2 else variant == 3
            cases[index] = {
                "case_index": index,
                "role_subject_index": subject,
                "subject_index": 564 + subject,
                "target": variant < 2,
                "kind": "positive" if variant < 2 else "sham" if variant == 2 else "nuisance",
                "mode": "focal" if variant == 0 else "repeated" if variant == 1 else "none",
                "nuisance": nuisance,
                "matched_control_case_index": 2256 + subject * 4 + (3 if nuisance else 2),
            }

    def render(context: dict, index: int) -> dict:
        case = cases[index]
        baseline = np.full((1, 4), case["subject_index"] + int(case["nuisance"]), np.float32)
        return {
            "baseline": baseline,
            "edited": baseline + target * np.float32(0.05) if case["target"] else baseline,
        }

    monkeypatch.setattr(common, "check_budget", lambda *args: None)
    monkeypatch.setattr(common, "render_checked", render)
    monkeypatch.setattr(
        common,
        "support_for_case",
        lambda context, index: target if cases[index]["target"] else np.zeros_like(target),
    )
    monkeypatch.setattr(common, "candidate_for_case", lambda context, index: candidate)
    maps = {arm: np.zeros((768, 1, 4), np.float32) for arm in successor.ARMS}
    for values in maps.values():
        for index in range(2256, 2768):
            values[index - 2000] = after if cases[index]["target"] else before
    context = {"cases": cases, "roles": {"evaluation": list(range(2256, 2768))}}
    rows, responses = successor.case_metrics(context, maps, dict.fromkeys(successor.ARMS, 0.6), {})
    assert len(rows) == 2816
    assert len(responses) == 1024
    assert {row["method"] for row in responses} == set(successor.ARMS)
    assert {row["subject"] for row in responses} == set(range(128))
    assert responses[0]["case"] == 2256
    assert responses[-1]["case"] == 2765
    delta = after.astype(np.float64) - before.astype(np.float64)
    for row in responses:
        assert row["edited_average_precision"] == 1.0
        assert row["baseline_site_average_precision"] == pytest.approx(1 / 3)
        assert row["average_precision_change"] == pytest.approx(2 / 3)
        assert row["mean_probability_change_T"] == delta[0, 0]
        assert row["mean_probability_change_U"] == delta[0, 1]
        assert row["mean_probability_change_O"] == float(delta[0, 2:].mean())
        assert row["selectivity_contrast"] == delta[0, 0] - delta[0, 1]
        assert row["baseline_mean_T"] == float(before[0, 0])
        assert row["edited_mean_U"] == float(after[0, 1])
    with pytest.raises(TypeError, match="invert"):
        pinned.case_metrics(context, maps, dict.fromkeys(successor.ARMS, 0.6), {})


def test_gpu_helpers_are_exact_frozen_objects_and_no_global_mutation() -> None:
    assert successor._infer is pinned._infer
    assert successor._after_training is pinned._after_training
    assert successor.summarize is pinned.summarize
    tree = ast.parse(Path(successor.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            assert not (isinstance(node.value, ast.Name) and node.value.id == "pinned")


def test_imported_helper_call_signatures_bind() -> None:
    import importlib
    import inspect

    tree = ast.parse(Path(successor.__file__).read_text())
    helpers = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = importlib.import_module(node.module)
            helpers.update({alias.asname or alias.name: getattr(module, alias.name) for alias in node.names})
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in helpers:
            arguments = [object() for _ in node.args]
            keywords = {key.arg: object() for key in node.keywords}
            inspect.signature(helpers[node.func.id]).bind(*arguments, **keywords)


@pytest.mark.parametrize("stop_during_scoring", [False, True])
def test_orchestrator_reaches_strict_receipt_signature_and_truthful_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stop_during_scoring: bool
) -> None:
    import json
    from contextlib import nullcontext

    import torch

    from ecg_experiment import ssl_synthetic_paired_background_training as receipt_library

    study = tmp_path / "study"
    study.mkdir()
    config = {
        "study_id": "artificial",
        "cache_bytes_ceiling": 4 * 1024**3,
        "complete_study_new_bytes": 504869760,
    }
    context = {
        "config": config,
        "sources": {},
        "inputs": {},
        "cases": [],
        "roles": {"calibration": list(range(2000, 2256)), "evaluation": list(range(2256, 2768))},
        "stage_preparation_cpu_seconds": 0.0,
        "regeneration_checks": {},
        "scope_exceptions": {},
    }
    profile = {
        "wall_seconds": 0.0,
        "cpu_seconds": 0.0,
        "source_sha256": {successor.V1_SOURCE: successor.sha256_file(successor.ROOT / successor.V1_SOURCE)},
    }
    training = {
        "wall_seconds": 0.0,
        "cpu_seconds": 0.0,
        "bce_reproduction": {"RB": {"passed": True}, "SB": {"passed": True}},
    }
    monkeypatch.setattr(common, "STUDY", study)
    monkeypatch.setattr(common, "admit_context", lambda *args: context)
    monkeypatch.setattr(common, "check_budget", lambda *args: None)
    monkeypatch.setattr(common, "storage_projection", lambda: {"projected_whole_stream_bytes": 0})
    monkeypatch.setattr(common, "verify_context", lambda *args: None)
    monkeypatch.setattr(successor, "_after_training", lambda *args: (profile, training))
    monkeypatch.setattr(successor, "configure_inference", lambda: None)
    monkeypatch.setattr(successor, "_thread_environment", lambda: {})
    monkeypatch.setattr(successor, "gpu_lock", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(successor, "open_memmap", lambda *args, **kwargs: np.zeros((768, 1, 4), np.float32))
    monkeypatch.setattr(successor, "stream_bytes", lambda: 0)
    monkeypatch.setattr(successor, "calibration_thresholds", lambda *args: dict.fromkeys(successor.ARMS, 0.9))
    calls = []

    def infer(ctx, arm, indices, maps, state, clock, progress):
        phase = "A" if indices[0] == 2000 else "B"
        if phase == "B":
            assert (study / "evaluation_v2/calibration.json").exists()
        calls.append((phase, arm))
        maps[np.asarray(indices) - 2000] = 0.5 if phase == "A" else 0.7
        progress["predicted_record_maps"] += len(indices)

    monkeypatch.setattr(successor, "_infer", infer)

    def metrics(*args):
        if stop_during_scoring:
            raise ValueError("artificial partial scoring stop")
        return [], []

    monkeypatch.setattr(successor, "case_metrics", metrics)
    monkeypatch.setattr(
        successor, "summarize", lambda *args: {"decision": {"engineering_replication_admitted": False}}
    )

    def receipt(path, ctx, clock, fields):
        assert path == study / "evaluation_v2"
        assert ctx is context
        assert fields["calibration_sha256"] == successor.sha256_file(path / "calibration.json")
        return {"wall_seconds": 0.0, "cpu_seconds": 0.0, **fields}

    monkeypatch.setattr(receipt_library, "_receipt", receipt)
    if stop_during_scoring:
        with pytest.raises(RuntimeError, match="evaluation stopped"):
            successor.evaluate_factorial_v2()
        failed = json.loads((study / "evaluation_v2/failure.json").read_text())
        assert failed["performance_scoring_started"]
        assert not failed["performance_scoring_completed"]
        assert "performance_scored" not in failed
    else:
        result = successor.evaluate_factorial_v2()
        assert result["evaluation_completed"]
        assert result["progress"]["predicted_record_maps"] == 3072
        assert result["progress"]["performance_scoring_started"]
        assert result["profiled_gpu_path_reused"]
        assert not result["v1_evaluation_ran"]
    assert calls == [(phase, arm) for phase in ["A", "B"] for arm in successor.ARMS]
    assert not (study / "evaluation").exists()
