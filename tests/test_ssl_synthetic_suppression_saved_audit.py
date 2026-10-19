"""Artificial independent loss, factorial sign and uncertainty checks."""

import copy
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from ecg_experiment.ssl_synthetic_suppression_saved_audit import (
    contrast_values,
    control_top_values,
    interval_values,
    objective_components,
    suppression_summary,
    verify_components,
)


def target_batch():
    target = np.zeros((8, 12, 5000), dtype=bool)
    for row, cells in ((2, 1), (3, 100), (6, 2), (7, 50)):
        target[row].ravel()[:cells] = True
    return target


def production_core():
    path = Path(
        os.environ.get(
            "ECG_SUPPRESSION_TEST_SOURCE",
            str(Path(__file__).parents[1] / "ecg_experiment/ssl_synthetic_paired_suppression.py"),
        )
    )
    if not path.exists():
        pytest.skip("Production suppression source not integrated in this worktree yet")
    spec = importlib.util.spec_from_file_location("suppression_source_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def artificial_fits():
    result = []
    for identifier in range(1, 4):
        fit = {"fit_id": identifier}
        for arm, ap, control in (("B", 0.10, 0.5), ("P", 0.12, 0.49), ("K", 0.11, 0.45), ("PK", 0.13, 0.44)):
            rows = []
            for index in range(64):
                rows.append(
                    {
                        "index": 2000 + index,
                        "seed": 2026105000 + index,
                        **{
                            mode: {"edited_ap": ap, "response_ap": 0.05, "selectivity": 0.04}
                            for mode in ("focal", "repeated")
                        },
                    }
                )
            fit[arm] = {
                "cases": rows,
                "top600_control": {
                    mode: {"record_means": [control] * 64, "mean": control}
                    for mode in ("sham", "nuisance", "pooled")
                },
            }
        result.append(fit)
    return result


def test_objective_zero_logits_known_weighted_bce_and_active_terms():
    logits = np.zeros((8, 12, 5000), dtype=np.float32)
    target = target_batch()
    bce = np.log(2) * (target.size + 19 * target.sum()) / target.size
    expected = {"B": (0, 0, 0), "P": (0.2, 0, 0), "K": (0, 0, 0.5), "PK": (0.2, 0, 0.5)}
    for arm, terms in expected.items():
        values = objective_components(logits, target, 20, arm)
        assert values[0] == pytest.approx(bce)
        assert np.allclose(values[1:4], terms, atol=1e-12, rtol=0)
        assert values[4] == pytest.approx(bce + sum(terms))
    logits[2][target[2]] = 4
    assert objective_components(logits, target, 20, "P")[1] == pytest.approx(0.15)


def test_actual_component_corruption_and_inactive_term_detection():
    row = objective_components(np.zeros((8, 12, 5000), dtype=np.float32), target_batch(), 20, "PK")
    components = np.repeat(row[None], 400, axis=0)
    losses = components[:, 4].copy()
    verify_components(components, losses, "PK")
    components[271, 1] += 1e-4
    with pytest.raises(ValueError, match="component total tolerance"):
        verify_components(components, losses, "PK")
    values = np.repeat(
        objective_components(np.zeros((8, 12, 5000), dtype=np.float32), target_batch(), 20, "B")[None],
        400,
        axis=0,
    )
    values[0, 3] = 1e-12
    with pytest.raises(ValueError, match="exactly absent"):
        verify_components(values, values[:, 4], "B")


def test_production_control_gradient_exact_tied_flat_indices():
    source = production_core()
    probability = torch.full((8, 12, 5000), 0.5, dtype=torch.float32, requires_grad=True)
    penalty = source.control_penalty(probability)
    penalty.backward()
    expected = np.zeros((8, 60000), dtype=np.float32)
    expected[[0, 1, 4, 5], :600] = np.float32(1 / 2400)
    assert penalty.item() == 0.5
    assert np.array_equal(probability.grad.numpy().reshape(8, -1), expected)


def test_production_objective_matches_independent_equations_and_attached_controls():
    source = production_core()
    values = np.random.default_rng(1701).normal(0, 0.3, size=(8, 12, 5000)).astype(np.float32)
    target = target_batch()
    for arm in ("B", "P", "K", "PK"):
        logits = torch.tensor(values, requires_grad=True)
        terms = source.objective(logits, torch.from_numpy(target), 20, arm)
        actual = np.array(
            [terms[name].item() for name in ("bce", "hinge", "consistency", "control", "total")]
        )
        assert np.max(np.abs(actual - objective_components(values, target, 20, arm))) <= 1e-6
        if arm == "P":
            terms["hinge"].backward()
            assert logits.grad[1].abs().sum() > 0
            assert logits.grad[5].abs().sum() > 0
            assert logits.grad[2].abs().sum() > 0


def test_top600_probability_value_uses_full_grid_and_equal_controls():
    values = np.zeros((64, 4, 12, 5000), dtype=np.float32)
    values[:, 0, 11, :600] = 0.8
    values[:, 1] = 0.2
    result = control_top_values(values)
    assert result["sham"]["mean"] == pytest.approx(0.8)
    assert result["nuisance"]["mean"] == pytest.approx(0.2)
    assert result["pooled"]["mean"] == pytest.approx(0.5)


def test_factorial_signs_additive_zero_interaction_and_complete_primary_gates():
    fits = artificial_fits()
    primary, secondary = contrast_values(fits)
    assert np.allclose(primary["B_minus_PK_top600_pooled"], 0.06)
    assert np.allclose(primary["PK_minus_B_direct_AP_pooled"], 0.03)
    assert np.allclose(secondary["K_minus_B__pooled_top600"], -0.05)
    assert np.allclose(secondary["PK_minus_P_minus_K_plus_B__pooled_edited_ap"], 0)
    assert np.allclose(secondary["half_P_minus_B_plus_PK_minus_K__pooled_edited_ap"], 0.02)
    summary = suppression_summary(fits)
    assert summary["scientific_passed"]
    assert len(summary["scientific_gates"]) == 15
    worse = copy.deepcopy(fits)
    for row in worse[0]["PK"]["cases"]:
        for mode in ("focal", "repeated"):
            row[mode]["edited_ap"] = 0.09
    failure = suppression_summary(worse)
    assert not failure["scientific_gates"]["fit_1_direct_AP_gain_positive"]
    assert not failure["scientific_passed"]


def test_crossed_column_dependence_and_unchanged_shared_contrast_draws():
    values = np.broadcast_to(np.arange(64, dtype=np.float64), (3, 64))
    result = interval_values({"a": values, "negative": -values}, True)
    rng = np.random.default_rng(2026105501)
    means = []
    for _ in range(2000):
        rng.integers(0, 3, 3)
        means.append(float(rng.integers(0, 64, 64).mean()))
    low, high = np.percentile(means, [2.5, 97.5])
    assert result["a"]["ci_low"] == low
    assert result["a"]["ci_high"] == high
    assert result["negative"]["ci_low"] == pytest.approx(-high, abs=1e-12)
    assert result["negative"]["ci_high"] == pytest.approx(-low, abs=1e-12)


def adam_state(model, step):
    return {
        "param_groups": [
            {
                "lr": 0.001,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0,
                "amsgrad": False,
                "foreach": False,
                "fused": False,
                "maximize": False,
                "capturable": False,
                "differentiable": False,
                "params": list(range(6)),
            }
        ],
        "state": {
            index: {
                "step": torch.tensor(float(step)),
                "exp_avg": torch.zeros_like(value),
                "exp_avg_sq": torch.zeros_like(value),
            }
            for index, value in enumerate(model.values())
        },
    }


def json_file(path, value):
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_saved_profile_capture_independent_component_corruption(tmp_path):
    from ecg_experiment.ssl_synthetic_replication_saved_audit import initialization, schedule_for_seed
    from scripts.validation.audit_ssl_synthetic_paired_suppression import profile_recovery

    study, parent = tmp_path / "new", tmp_path / "old"
    parent.mkdir()
    support = np.zeros((128, 4, 12, 5000), dtype=bool)
    support[:80, 2:, 0, :10] = True
    np.save(parent / "support.npy", support)
    schedule = schedule_for_seed(2026103901)
    state = initialization(2026103801)
    logits = np.zeros((8, 12, 5000), dtype=np.float32)
    for arm in ("B", "P", "K", "PK"):
        directory = study / "fit_1/profile" / arm
        directory.mkdir(parents=True)
        row = objective_components(logits, support[:80].reshape(320, 12, 5000)[schedule[0]], 20, arm)
        original, replay = np.repeat(row[None], 3, axis=0), np.repeat(row[None], 2, axis=0)
        for prefix, values in (("original", original), ("replay", replay)):
            (directory / prefix).mkdir()
            np.save(directory / prefix / "losses.npy", values[:, 4])
            np.save(directory / prefix / "loss_components.npy", values)
        np.save(directory / "step1_logits.npy", logits)
        np.save(directory / "step1_case_indices.npy", schedule[0])
        torch.save(state, directory / "step1_model.pt")
        torch.save(adam_state(state, 1), directory / "step1_optimizer.pt")
        for step in (2, 3):
            for prefix in ("original", "replay"):
                torch.save(state, directory / f"{prefix}_step{step}_model.pt")
                torch.save(adam_state(state, step), directory / f"{prefix}_step{step}_optimizer.pt")
        json_file(
            directory / "summary.json",
            {
                "losses": original[:, 4].tolist(),
                "replay_losses": replay[:, 4].tolist(),
                "components": original.tolist(),
                "replay_components": replay.tolist(),
                "trajectory_state_exact": [True, True],
                "exact_replay": True,
                "recurring_persistence_included": True,
            },
        )
    result = profile_recovery(study, parent, 20, lambda: None)
    assert set(result["first_objective"]) == {"B", "P", "K", "PK"}
    assert all(row["maximum_absolute_error"] == 0 for row in result["first_objective"].values())
    np.save(study / "fit_1/profile/PK/step1_logits.npy", logits + 1)
    with pytest.raises(ValueError, match="independent captured objective"):
        profile_recovery(study, parent, 20, lambda: None)
    np.save(study / "fit_1/profile/PK/step1_logits.npy", logits)
    bad = adam_state(state, 2)
    bad["state"][0]["exp_avg"].view(-1)[0] += 1
    torch.save(bad, study / "fit_1/profile/P/replay_step2_optimizer.pt")
    with pytest.raises(ValueError, match="complete recovered profile Adam"):
        profile_recovery(study, parent, 20, lambda: None)


def test_binding_roots_literal_admission_and_payload_mutation(tmp_path):
    from ecg_experiment.synthetic_audit_io import file_sha256
    from scripts.validation.audit_ssl_synthetic_paired_suppression import bindings

    source, study = tmp_path / "source", tmp_path / "study"
    parents = {label: tmp_path / label for label in ("generator", "learning", "replication")}
    catalogs = {}
    for label, directory in parents.items():
        directory.mkdir()
        (directory / "input.json").write_text(label)
        catalogs[label] = {"input.json": file_sha256(directory / "input.json")}
    for directory in (source, study / "source_snapshot"):
        directory.mkdir(parents=True)
        (directory / "source.py").write_text("reviewed")
    (study / "fresh.npy").write_bytes(b"synthetic")
    (study / "metric.json").write_text("result")
    main = {
        "source_sha256": {"source.py": file_sha256(source / "source.py")},
        "input_sha256": catalogs,
        "generated_input_sha256": {"fresh.npy": file_sha256(study / "fresh.npy")},
        "output_sha256": {"metric.json": file_sha256(study / "metric.json")},
    }
    config = {"fixed_parents": {label: {"sha256": catalogs[label]} for label in ("generator", "learning")}}
    admission = {"replication": {"sha256": catalogs["replication"]}}
    assert bindings(source, study, parents, main, config, admission, lambda: None) == 7
    (study / "fresh.npy").write_bytes(b"changed")
    with pytest.raises(ValueError, match="immutable new payload"):
        bindings(source, study, parents, main, config, admission, lambda: None)


def test_early_failure_cost_binding_no_cumulative_double_count(tmp_path, monkeypatch):
    import time

    from ecg_experiment.synthetic_audit_io import read_json
    from scripts.validation import audit_ssl_synthetic_paired_suppression as audit

    study, parent = tmp_path / "new", tmp_path / "replication"
    study.mkdir()
    prior = parent / "independent_audit_v1/prospective_ledger.json"
    prior_cpu, prior_gpu = 190.681380118, 178.56192114995793
    json_file(
        prior,
        {
            "current_machine_cumulative_cpu_seconds": prior_cpu,
            "current_machine_cumulative_gpu_wall_seconds": prior_gpu,
        },
    )
    original_hash = audit.file_sha256

    def identity(path, guard=lambda: None):
        return (
            "f17f0c112a3eeed8b5f46be389c2a06ff8dcb8c2dd6169b1833a398f20a0d264"
            if path == prior
            else original_hash(path, guard)
        )

    monkeypatch.setattr(audit, "file_sha256", identity)
    main = {"cpu_seconds": 2, "gpu_lock_wall_seconds": 3}
    json_file(study / "summary.json", main)
    ledger = {
        "receipt": "summary.json",
        "receipt_sha256": original_hash(study / "summary.json"),
        "prior_ledger_sha256": identity(prior),
        "cpu_seconds": 2.1,
        "wall_seconds": 4,
        "gpu_wall_seconds": 3,
        "prior_current_machine_cpu_seconds": prior_cpu,
        "prior_current_machine_gpu_wall_seconds": prior_gpu,
        "current_machine_cumulative_cpu_seconds": prior_cpu + 2.1,
        "current_machine_cumulative_gpu_wall_seconds": prior_gpu + 3,
    }
    json_file(study / "prospective_ledger.json", ledger)
    costs = audit.verified_accounting(study, {"replication": parent}, main, lambda: None)
    destination = study / "independent_audit_v1"
    json_file(destination / "failure.json", {"all_passed": False, "error": "artificial later corruption"})
    audit.write_ledger(destination, "failure.json", (time.process_time(), time.perf_counter()), costs)
    own = read_json(destination / "prospective_ledger.json")
    assert own["current_machine_cumulative_cpu_seconds"] == prior_cpu + 2.1 + own["cpu_seconds"]
    assert own["current_machine_cumulative_gpu_wall_seconds"] == prior_gpu + 3
    ledger["current_machine_cumulative_cpu_seconds"] += prior_cpu
    json_file(study / "prospective_ledger.json", ledger)
    with pytest.raises(ValueError, match="once-only cumulative addition"):
        audit.verified_accounting(study, {"replication": parent}, main, lambda: None)


def test_corrupt_baseline_loss_detected_after_component_totals_still_match(tmp_path):
    from ecg_experiment.ssl_synthetic_replication_saved_audit import initialization, schedule_for_seed
    from scripts.validation.audit_ssl_synthetic_paired_suppression import states_and_schedules

    study, parent = tmp_path / "new", tmp_path / "replication"
    for fit in (1, 2, 3):
        initial = initialization(2026103800 + fit)
        directory, old = study / f"fit_{fit}", parent / f"fit_{fit}"
        directory.mkdir(parents=True)
        old.mkdir(parents=True)
        torch.save(initial, directory / "initial_state.pt")
        torch.save(initial, old / "initial_state.pt")
        schedule = schedule_for_seed(2026103900 + fit)
        np.save(directory / "training_schedule.npy", schedule)
        np.save(old / "training_schedule.npy", schedule)
        np.save(directory / "background_schedule.npy", schedule[:, [0, 4]] // 4)
        json_file(
            directory / "fit_design.json",
            {"fit_id": fit, "initialization_seed": 2026103800 + fit, "schedule_seed": 2026103900 + fit},
        )
        for arm in ("B", "P", "K", "PK"):
            location = directory / arm
            location.mkdir()
            torch.save(initial, location / "initial_state.pt")
            final = copy.deepcopy(initial)
            final["0.bias"][0] += 1
            torch.save(final, location / "final_state.pt")
            adam = adam_state(final, 400)
            torch.save(adam, location / "optimizer.pt")
            losses = np.linspace(0.2, 0.1, 400, dtype=np.float64)
            components = np.zeros((400, 5), dtype=np.float64)
            components[:, 0] = components[:, 4] = losses
            np.save(location / "losses.npy", losses)
            np.save(location / "loss_components.npy", components)
            if arm == "B":
                (old / "supervised").mkdir()
                torch.save(final, old / "supervised/final_state.pt")
                torch.save(adam, old / "supervised/optimizer.pt")
                np.save(old / "supervised/losses.npy", losses)
        json_file(
            directory / "baseline_closure.json",
            dict.fromkeys(
                ("all_exact", "all_400_losses_exact", "final_model_exact", "complete_adam_exact"), True
            ),
        )
    assert set(states_and_schedules(study, parent, lambda: None)) == {"1", "2", "3"}
    losses = np.load(study / "fit_2/B/losses.npy")
    components = np.load(study / "fit_2/B/loss_components.npy")
    losses[271] += 0.01
    components[271, 0] = components[271, 4] = losses[271]
    np.save(study / "fit_2/B/losses.npy", losses)
    np.save(study / "fit_2/B/loss_components.npy", components)
    with pytest.raises(ValueError, match="baseline all400 losses exact"):
        states_and_schedules(study, parent, lambda: None)
