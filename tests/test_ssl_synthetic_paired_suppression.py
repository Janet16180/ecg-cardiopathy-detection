"""Artificial objective, crossed-inference and controller integrity checks."""

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from ecg_experiment import ssl_synthetic_paired_suppression as study
from ecg_experiment import ssl_synthetic_paired_suppression_run as runner
from ecg_experiment.intervals import patient_groups, patient_resample


def targets() -> np.ndarray:
    """Create unequal positive areas with exact paired zero controls."""
    support = np.zeros((8, 12, 5000), dtype=bool)
    for row, size in zip([2, 3, 6, 7], [10, 100, 30, 300], strict=True):
        support[row, 0, :size] = True
    return support


def fits() -> list[dict[str, Any]]:
    """Provide transparent four-arm metrics with favorable primary results."""
    result = []
    for identifier in [1, 2, 3]:
        fit = {"fit_id": identifier}
        for arm, ap, control in [("B", 0.2, 0.8), ("P", 0.3, 0.7), ("K", 0.4, 0.6), ("PK", 0.6, 0.4)]:
            rows = []
            for offset in range(64):
                row = {"index": 2000 + offset, "seed": 2026105000 + offset}
                for mode in ["focal", "repeated"]:
                    row[mode] = {"edited_ap": ap, "response_ap": 0.1, "selectivity": 0.1}
                rows.append(row)
            fit[arm] = {
                "cases": rows,
                "top600_control": {
                    mode: {"record_means": [control] * 64, "mean": control}
                    for mode in ["sham", "nuisance", "pooled"]
                },
            }
        result.append(fit)
    return result


def test_control_ties_select_lower_flat_indices_and_preserve_live_gradients() -> None:
    probability = torch.full((8, 12, 5000), 0.5, requires_grad=True)
    value = study.control_penalty(probability)
    assert value.item() == 0.5
    value.backward()
    expected = torch.zeros_like(probability).flatten(1)
    expected[[0, 1, 4, 5], :600] = 1 / (4 * 600)
    assert torch.equal(probability.grad, expected.reshape_as(probability))


def test_paired_terms_equal_case_weighting_and_both_observation_gradients() -> None:
    logits = torch.zeros((8, 12, 5000), requires_grad=True)
    truth = torch.from_numpy(targets())
    terms = study.objective(logits, truth, 20.0, "P")
    assert terms["hinge"].item() == pytest.approx(0.2)
    assert terms["consistency"].item() == 0
    assert terms["control"].item() == 0
    terms["hinge"].backward()
    assert logits.grad[2, 0, 0].item() == pytest.approx(-0.25 / (4 * 10))
    assert logits.grad[3, 0, 0].item() == pytest.approx(-0.25 / (4 * 100))
    assert logits.grad[1, 0, 0].item() > 0
    assert logits.grad[5, 0, 0].item() > 0
    assert logits.grad[0].abs().sum() == 0
    for arm in study.ARMS:
        terms = study.objective(logits.detach(), truth, 20.0, arm)
        assert terms["total"].item() == pytest.approx(
            sum(terms[name].item() for name in study.COMPONENTS[:-1]), abs=1e-6
        )
        if arm in {"B", "K"}:
            assert terms["hinge"] == terms["consistency"] == 0
        if arm in {"B", "P"}:
            assert terms["control"] == 0


def test_bce_update_exactly_matches_frozen_update_and_persists_preupdate_evidence(tmp_path: Path) -> None:
    torch.manual_seed(6)
    first = nn.Conv1d(12, 12, 1)
    second = copy.deepcopy(first)
    old_optimizer = runner.fit_runtime.optimizer_for(first)
    new_optimizer = runner.fit_runtime.optimizer_for(second)
    observed = np.random.default_rng(7).normal(size=(8, 12, 5000)).astype(np.float32)
    support = targets()
    indices = np.arange(8)
    banks = runner.component_banks(tmp_path / "logs", 2)
    with torch.inference_mode():
        expected_logits = second(torch.from_numpy(observed)).numpy().copy()
    for step in range(2):
        expected = runner.fit_runtime.update(first, old_optimizer, observed, support, indices, 20.0, "cpu")
        values = runner.update_and_persist(
            second,
            new_optimizer,
            observed,
            support,
            indices,
            20.0,
            "B",
            banks,
            step,
            "cpu",
            tmp_path if step == 0 else None,
        )
        assert values == [expected, 0.0, 0.0, 0.0, expected]
        assert runner.fit_runtime.trees_equal(first.state_dict(), second.state_dict())
        assert runner.fit_runtime.trees_equal(old_optimizer.state_dict(), new_optimizer.state_dict())
    assert np.array_equal(np.load(tmp_path / "step1_logits.npy"), expected_logits)
    assert np.array_equal(np.load(tmp_path / "step1_case_indices.npy"), indices)
    assert np.isfinite(np.load(tmp_path / "logs/loss_components.npy")).all()
    with pytest.raises(FileExistsError):
        runner.component_banks(tmp_path / "logs", 2)


def test_shared_crossed_draws_and_all_signed_factorial_contrasts() -> None:
    primary, secondary = study.contrast_matrices(fits())
    assert np.allclose(primary["PK_minus_B_direct_AP_pooled"], 0.4)
    assert np.allclose(secondary["PK_minus_P_minus_K_plus_B__pooled_edited_ap"], 0.1)
    assert np.allclose(secondary["half_P_minus_B_plus_PK_minus_K__pooled_edited_ap"], 0.15)
    assert np.allclose(secondary["half_K_minus_B_plus_PK_minus_P__pooled_edited_ap"], 0.25)
    values = np.tile(np.linspace(-0.3, 0.7, 64) ** 3, (3, 1))
    actual = study.intervals({"value": values}, True)
    rng = np.random.default_rng(2026105501)
    groups = patient_groups(np.arange(64))
    expected = []
    for _ in range(2000):
        rng.integers(0, 3, 3)
        columns = patient_resample(groups, rng)
        expected.append(float(values[:, columns].mean()))
    assert actual["value"]["ci_low"] == np.quantile(expected, 0.025, method="linear")
    assert actual["value"]["ci_high"] == np.quantile(expected, 0.975, method="linear")


@pytest.mark.parametrize("identifier", [0, 1, 2])
def test_each_fit_primary_guard_can_fail_independently(identifier: int) -> None:
    records = fits()
    for row in records[identifier]["PK"]["cases"]:
        for mode in ["focal", "repeated"]:
            row[mode]["edited_ap"] = 0.19
    summary = study.aggregate(records)
    assert len(summary["scientific_gates"]) == 15
    assert not summary["scientific_gates"][f"fit_{identifier + 1}_direct_AP_gain_positive"]
    assert summary["scientific_gates"]["PK_minus_B_direct_AP_pooled_mean"]
    assert not summary["scientific_passed"]
    records[0]["P"]["cases"].reverse()
    with pytest.raises(ValueError, match="Ordered fresh panel"):
        study.aggregate(records)


def test_profile_includes_persistence_and_full_two_step_state_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(9)
    initial = nn.Conv1d(12, 12, 1).state_dict()

    def create(state: dict[str, Any]) -> tuple[nn.Module, torch.optim.Optimizer]:
        model = nn.Conv1d(12, 12, 1)
        model.load_state_dict(state)
        return model, runner.fit_runtime.optimizer_for(model)

    monkeypatch.setattr(runner, "new_model", create)
    monkeypatch.setattr(runner.torch.cuda, "synchronize", lambda *args: None)
    original_update = runner.update_and_persist

    def cpu_update(*args: Any, **kwargs: Any) -> list[float]:
        return original_update(*args, device="cpu", **kwargs)

    monkeypatch.setattr(runner, "update_and_persist", cpu_update)
    original_infer = runner.fit_runtime.infer
    monkeypatch.setattr(
        runner.fit_runtime, "infer", lambda model, observed, device: original_infer(model, observed, "cpu")
    )
    observed = np.random.default_rng(1).normal(size=(8, 12, 5000)).astype(np.float32)
    profile = runner.profile_arm(
        "PK",
        {"initial": initial, "schedule": np.tile(np.arange(8), (400, 1)), "directory": tmp_path},
        observed,
        targets(),
        20.0,
        runner.initial_counts()["training_profile"],
        lambda: None,
    )
    assert profile["trajectory_state_exact"] == [True, True]
    assert profile["components"][1:] == profile["replay_components"]
    assert profile["recurring_persistence_included"]
    for step in [2, 3]:
        directory = tmp_path / "profile/PK"
        assert runner.fit_runtime.trees_equal(
            torch.load(directory / f"original_step{step}_optimizer.pt", weights_only=True),
            torch.load(directory / f"replay_step{step}_optimizer.pt", weights_only=True),
        )
    assert (tmp_path / "profile/PK/step1_logits.npy").exists()


def test_combined_forecast_uses_4800_updates_384_inferences_and_charges_actual_profile_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner.time, "process_time", lambda: 20.0)
    monkeypatch.setattr(runner.time, "perf_counter", lambda: 1100.0)
    monkeypatch.setattr(runner.torch.cuda, "max_memory_allocated", lambda index: 1024)
    (tmp_path / "captured_logits").write_bytes(bytes(1024))
    generation = {
        "original_timings": [
            {"cpu_seconds": 99.0, "wall_seconds": 99.0},
            {"cpu_seconds": 0.01, "wall_seconds": 0.1},
            {"cpu_seconds": 0.02, "wall_seconds": 0.2},
        ]
    }
    arm = {
        "timings": [
            {"cpu_seconds": 99.0, "wall_seconds": 99.0},
            {"cpu_seconds": 0.001, "wall_seconds": 0.01},
            {"cpu_seconds": 0.002, "wall_seconds": 0.02},
        ],
        "inference": {"cpu_seconds": 0.0004, "wall_seconds": 0.004},
    }
    result = runner.combined_profile(
        generation,
        {name: copy.deepcopy(arm) for name in study.ARMS},
        tmp_path,
        {"cpu": 0.0, "wall": 1000.0, "gpu_lock_start": 1050.0},
        {"resources": {"output_forecast_reserve_bytes": 32 * 1024**2}},
    )
    assert result["forecast_cpu_seconds"] == pytest.approx(132.0672)
    assert result["forecast_wall_seconds"] == pytest.approx(410.672)
    assert result["forecast_gpu_lock_wall_seconds"] == pytest.approx(360.672)
    assert result["forecast_saved_bytes"] > 737280000 + 94720000 + 1024


def test_all_three_fits_interleave_every_objective_before_baseline_closures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = []

    def models(initial: Any) -> tuple[nn.Module, torch.optim.Optimizer]:
        model = nn.Linear(1, 1)
        return model, runner.fit_runtime.optimizer_for(model)

    monkeypatch.setattr(runner, "new_model", models)

    def fake_update(
        model: Any,
        optimizer: Any,
        observed: Any,
        support: Any,
        indices: Any,
        weight: Any,
        arm: str,
        banks: Any,
        step: int,
        **kwargs: Any,
    ) -> list[float]:
        events.append((arm, step))
        return [0.0] * 5

    monkeypatch.setattr(runner, "update_and_persist", fake_update)
    monkeypatch.setattr(runner, "baseline_closure", lambda plan, parent, model: {"all_exact": True})
    monkeypatch.setattr(runner.fit_runtime, "save_state", lambda *args: None)
    plans = [
        {
            "fit_id": i,
            "initial": {},
            "schedule": np.zeros((400, 8), dtype=int),
            "directory": tmp_path / f"fit_{i}",
        }
        for i in [1, 2, 3]
    ]
    counts = runner.initial_counts()
    models, _ = runner.train_all(
        plans, np.zeros(1), np.zeros(1), 20.0, tmp_path, counts, lambda: None, lambda: None
    )
    assert len(events) == 4800
    assert events == [(arm, step) for _ in range(3) for step in range(400) for arm in study.ARMS]

    def inference(*args: Any) -> None:
        events.append(("infer", args[5]))

    monkeypatch.setattr(runner.previous, "evaluate_panel", inference)
    with pytest.raises(ValueError, match="All twelve"):
        runner.evaluate_all(
            plans, {"1": models["1"]}, {"observed": np.zeros(1)}, counts, lambda: None, lambda: None
        )
    assert len(events) == 4800
    runner.evaluate_all(plans, models, {"observed": np.zeros(1)}, counts, lambda: None, lambda: None)
    assert len(events) == 4812
    assert all(event[0] == "infer" for event in events[4800:])


def test_completed_update_is_counted_when_component_flush_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = nn.Conv1d(12, 12, 1)
    initial = copy.deepcopy(model.state_dict())
    optimizer = runner.fit_runtime.optimizer_for(model)
    banks = runner.component_banks(tmp_path, 1)
    counters = {"attempts": 1, "updates": 0}

    def failed_flush(self: np.memmap) -> None:
        raise OSError("artificial full disk after optimizer step")

    monkeypatch.setattr(np.memmap, "flush", failed_flush)
    with pytest.raises(OSError, match="full disk"):
        runner.update_and_persist(
            model,
            optimizer,
            np.ones((8, 12, 5000), dtype=np.float32),
            targets(),
            np.arange(8),
            20.0,
            "B",
            banks,
            0,
            "cpu",
            completed=counters,
        )
    assert counters == {"attempts": 1, "updates": 1}
    assert not runner.fit_runtime.trees_equal(initial, model.state_dict())
    assert all(value["step"].item() == 1 for value in optimizer.state.values())


def test_baseline_single_loss_corruption_stops_and_retains_negative_closure(tmp_path: Path) -> None:
    model = nn.Linear(1, 1)
    parent = tmp_path / "parent"
    directory = tmp_path / "new"
    for path in [parent / "fit_2/supervised", directory / "B"]:
        path.mkdir(parents=True)
        torch.save(model.state_dict(), path / "final_state.pt")
        torch.save({"state": {}, "param_groups": []}, path / "optimizer.pt")
        np.save(path / "losses.npy", np.arange(400, dtype=np.float64))
    losses = np.load(directory / "B/losses.npy")
    losses[271] += 0.1
    np.save(directory / "B/losses.npy", losses)
    with pytest.raises(ValueError, match="baseline400"):
        runner.baseline_closure({"fit_id": 2, "directory": directory}, parent, model)
    receipt = runner.read_json(directory / "baseline_closure.json")
    assert not receipt["all_exact"]
    assert receipt["final_model_exact"]
    assert receipt["complete_adam_exact"]


def test_controller_bad_parent_stops_before_nn_and_preserves_failure_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = runner.read_json(runner.from_stored(runner.CONFIG))
    (tmp_path / config["output_root"]).parent.mkdir(parents=True)
    monkeypatch.setattr(runner.gpu_runtime, "set_runtime", lambda: None)
    monkeypatch.setattr(runner, "snapshot", lambda *args: {})

    def bad_parent(*args: Any) -> None:
        raise ValueError("artificial parent identity mismatch")

    monkeypatch.setattr(runner, "admit", bad_parent)

    def forbidden(*args: Any) -> None:
        raise AssertionError("No neural operation before identity admission")

    monkeypatch.setattr(runner.previous, "load_public_generator", forbidden)
    with pytest.raises(ValueError, match="parent identity"):
        runner.run_study(tmp_path)
    destination = tmp_path / config["output_root"]
    failure = runner.read_json(destination / "failure.json")
    ledger = runner.read_json(destination / "prospective_ledger.json")
    assert all(value == 0 for value in failure["operation_totals"].values())
    assert ledger["cpu_seconds"] >= failure["cpu_seconds"]
    assert ledger["gpu_wall_seconds"] == 0
    assert ledger["prior_accounting_verified"] is False
    with pytest.raises(FileExistsError):
        runner.run_study(tmp_path)


def test_input_mutation_during_fit_stops_before_inference_and_retains_failure_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = runner.read_json(runner.from_stored(runner.CONFIG))
    destination = tmp_path / config["output_root"]
    destination.parent.mkdir(parents=True)
    monkeypatch.setattr(runner.gpu_runtime, "set_runtime", lambda: None)
    monkeypatch.setattr(runner, "snapshot", lambda *args: {})
    monkeypatch.setattr(
        runner, "admit", lambda *args: {"generator": tmp_path, "learning": tmp_path, "replication": tmp_path}
    )
    monkeypatch.setattr(runner, "predecessor_closure", lambda *args: {"all_exact": True})
    original_load = np.load
    support = np.broadcast_to(targets()[:4], (128, 4, 12, 5000))
    observed = np.broadcast_to(np.zeros((4, 12, 5000), dtype=np.float32), support.shape)

    def load(path: Path, **kwargs: Any) -> np.ndarray:
        if path == tmp_path / "support.npy":
            return support
        if path == tmp_path / "observed.npy":
            return observed
        return original_load(path, **kwargs)

    monkeypatch.setattr(runner.np, "load", load)
    monkeypatch.setattr(runner, "training_prior", lambda *args: np.zeros((12, 5000)))
    monkeypatch.setattr(runner.previous, "load_public_generator", lambda *args: (None, "unused"))
    monkeypatch.setattr(runner, "prepare_plans", lambda *args: [])
    monkeypatch.setattr(runner.previous, "plan_hashes", lambda *args: {})

    def create(initial: Any) -> tuple[nn.Module, torch.optim.Optimizer]:
        model = nn.Linear(1, 1)
        return model, runner.fit_runtime.optimizer_for(model)

    monkeypatch.setattr(runner, "new_model", create)
    target = destination / "fresh_noise.npy"

    def mutate(*args: Any, **kwargs: Any) -> list[float]:
        target.write_bytes(b"changed after admission during fitting")
        return [0.0] * 5

    monkeypatch.setattr(runner, "update_and_persist", mutate)

    def forbidden(*args: Any) -> None:
        raise AssertionError("No inference or baseline closure after immutable input mutation")

    monkeypatch.setattr(runner, "evaluate_all", forbidden)
    monkeypatch.setattr(runner, "baseline_closure", forbidden)

    def execute(*args: Any) -> None:
        generated, guard, verify = args[-3:]
        target.write_bytes(b"admitted noise")
        generated[target.name] = runner.file_sha256(target)
        plan = {
            "fit_id": 1,
            "directory": destination / "fit_1",
            "initial": {},
            "schedule": np.zeros((1, 8), dtype=int),
        }
        runner.train_all(
            [plan], np.zeros(1), np.zeros(1), 20.0, tmp_path, runner.initial_counts(), guard, verify
        )
        raise AssertionError("Post-fit mutation must reject")

    monkeypatch.setattr(runner, "execute_gpu", execute)
    original_finish = runner.finish_ledger
    monkeypatch.setattr(
        runner,
        "finish_ledger",
        lambda destination, root, amendment, receipt, clock, verified: original_finish(
            destination, root, amendment, receipt, clock, False
        ),
    )
    with pytest.raises(ValueError, match="immutable replication inputs changed"):
        runner.run_study(tmp_path)
    failure = runner.read_json(destination / "failure.json")
    ledger = runner.read_json(destination / "prospective_ledger.json")
    assert runner.file_sha256(target) != failure["generated_input_sha256"][target.name]
    assert ledger["cpu_seconds"] >= failure["cpu_seconds"]
    assert not (destination / "summary.json").exists()
