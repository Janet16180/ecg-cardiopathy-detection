"""Artificial replication tests without old study payloads, public weights or CUDA."""

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from ecg_experiment import ssl_synthetic_fit_replication as replication
from ecg_experiment import ssl_synthetic_fit_replication_run as runner
from ecg_experiment.intervals import patient_groups, patient_resample


def metric_fits() -> list[dict[str, Any]]:
    """Build three declared new fits with transparent favorable synthetic scalar measurements."""
    fits = []
    for identifier in [1, 2, 3]:
        fit = {"fit_id": identifier, "supervised": [], "permuted": []}
        for index in range(64):
            identity = {"index": 1000 + index, "seed": 2026104000 + index}
            true_mode = {"edited_ap": 0.8, "amplitude_ap": 0.1, "response_ap": 0.3, "selectivity": 0.2}
            null_mode = {"edited_ap": 0.2}
            fit["supervised"].append({**identity, "focal": true_mode.copy(), "repeated": true_mode.copy()})
            fit["permuted"].append({**identity, "focal": null_mode.copy(), "repeated": null_mode.copy()})
        fits.append(fit)
    return fits


def test_schedule_preserves_old_bytes_and_complete_training_exposure() -> None:
    assert np.array_equal(replication.training_schedule(2026103502), runner.learning.training_schedule())
    schedules = [replication.training_schedule(seed) for seed in [2026103901, 2026103902, 2026103903]]
    for schedule in schedules:
        assert schedule.shape == (400, 8)
        assert schedule.min() == 0
        assert schedule.max() == 319
        assert np.array_equal(np.bincount(schedule.ravel()), np.full(320, 10))
    assert not np.array_equal(schedules[0], schedules[1])
    assert not np.array_equal(schedules[1], schedules[2])


def test_generic_48_scoring_is_exactly_the_frozen_helper() -> None:
    case = np.zeros((4, 12, 5000), dtype=np.float32)
    target = np.zeros(case.shape, dtype=bool)
    target[2, 0, 100:200] = True
    target[3, 1, 300:600] = True
    probability = np.full(case.shape, 0.1, dtype=np.float32)
    probability[target] = 0.9
    maps = np.broadcast_to(probability, (48, *case.shape))
    observed = np.broadcast_to(case, maps.shape)
    support = np.broadcast_to(target, maps.shape)
    assert replication.verify_legacy_scoring(maps, observed, support, np.zeros((12, 5000), dtype=np.float64))
    fresh_maps = np.broadcast_to(probability, (64, *case.shape))
    rows = replication.score_panel(
        fresh_maps,
        np.broadcast_to(case, fresh_maps.shape),
        np.broadcast_to(target, fresh_maps.shape),
        np.arange(1000, 1064),
        np.arange(2026104000, 2026104064),
        np.zeros((12, 5000)),
    )
    assert len(rows) == 64
    assert rows[-1]["seed"] == 2026104063
    assert rows[0]["focal"]["edited_ap"] == 1
    assert rows[0]["repeated"]["selectivity"] > 0.79
    assert replication.control_scores(fresh_maps)["sham"]["record_maximum_gt_0p5_fraction"] == 0


def test_crossed_bootstrap_uses_one_shared_background_vector_after_fit_draws() -> None:
    vector = np.linspace(-0.4, 0.6, 64) ** 3
    values = np.tile(vector, (3, 1))
    result = replication.bootstrap_summaries({"value": values, "twice": values * 2}, crossed=True)
    groups = patient_groups(np.arange(64))
    rng = np.random.default_rng(2026104501)
    expected = []
    for _ in range(2000):
        rng.integers(0, 3, 3)
        background = patient_resample(groups, rng)
        expected.append(float(np.tile(vector[background], (3, 1)).mean()))
    low, high = np.percentile(expected, [2.5, 97.5])
    assert result["value"]["ci_low"] == low
    assert result["value"]["ci_high"] == high
    assert result["twice"]["ci_low"] == 2 * low
    assert result["value"]["fit_count"] == 3
    assert result["value"]["background_count"] == 64


def test_primary_excludes_fit0_and_requires_identical_ordered_fresh_cases() -> None:
    fits = metric_fits()
    summary = replication.aggregate_replication(fits)
    assert summary["scientific_passed"]
    assert len(summary["scientific_gates"]) == 18
    fits[0]["fit_id"] = 0
    with pytest.raises(ValueError, match="fit0 is descriptive"):
        replication.aggregate_replication(fits)
    fits[0]["fit_id"] = 1
    fits[2]["permuted"].reverse()
    with pytest.raises(ValueError, match="ordered fresh background identities"):
        replication.aggregate_replication(fits)


@pytest.mark.parametrize("fit_index", [0, 1, 2])
@pytest.mark.parametrize("comparator", ["absolute_amplitude", "permuted"])
def test_each_of_six_direction_guards_has_its_own_fit_and_comparator(fit_index: int, comparator: str) -> None:
    fits = metric_fits()
    if comparator == "absolute_amplitude":
        for row in fits[fit_index]["supervised"]:
            for mode in ["focal", "repeated"]:
                row[mode]["amplitude_ap"] = 0.95
    else:
        for row in fits[fit_index]["permuted"]:
            for mode in ["focal", "repeated"]:
                row[mode]["edited_ap"] = 0.95
    result = replication.aggregate_replication(fits)
    key = f"fit_{fit_index + 1}_supervised_minus_{comparator}_pooled_mean_positive"
    assert result["scientific_gates"][key] is False
    assert result["scientific_gates"][f"supervised_minus_{comparator}_pooled_mean"] is True
    other = "permuted" if comparator == "absolute_amplitude" else "absolute_amplitude"
    assert (
        result["scientific_gates"][f"fit_{fit_index + 1}_supervised_minus_{other}_pooled_mean_positive"]
        is True
    )
    assert not result["scientific_passed"]


def test_offending_profile_raw_is_saved_before_editor_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = np.zeros((8, 5000), dtype=np.float32)

    def failed_editor(*args: Any) -> None:
        raise ValueError("artificial robust scale failure")

    monkeypatch.setattr(runner.learning, "make_case", failed_editor)
    with pytest.raises(ValueError, match="robust scale"):
        runner.save_generation_case(tmp_path, "original_0", raw, 1000)
    assert np.array_equal(np.load(tmp_path / "original_0_raw.npy", allow_pickle=False), raw)
    assert not (tmp_path / "original_0_observed.npy").exists()


def test_bank_duplicate_stops_without_replacement_and_retains_offending_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = np.random.default_rng(7).normal(size=(8, 5000)).astype(np.float32)
    expected = runner.learning.make_case(raw, 1000)[0]
    monkeypatch.setattr(runner.generator_runtime, "gpu_forward", lambda model, noise: raw.copy())
    counts = runner.initial_counts()
    rows = []
    with pytest.raises(ValueError, match="no replacement"):
        runner.generate_fresh_bank(
            None,
            np.zeros((64, 1, 8, 5000), dtype=np.float32),
            {"fresh_panel": {"global_indices": list(range(1000, 1064))}},
            tmp_path,
            expected[None, 0],
            rows,
            counts,
            lambda: None,
        )
    assert counts["gpu_generator_bank_attempts"] == counts["gpu_generator_bank_forwards"] == 1
    assert len(rows) == 1
    assert np.array_equal(np.load(tmp_path / "fresh_raw.npy", mmap_mode="r", allow_pickle=False)[0], raw)
    assert np.array_equal(
        np.load(tmp_path / "fresh_observed.npy", mmap_mode="r", allow_pickle=False)[0], expected
    )


def test_combined_forecast_has_exact_remaining_counts_and_distinct_lock_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner.time, "process_time", lambda: 20.0)
    monkeypatch.setattr(runner.time, "perf_counter", lambda: 1100.0)
    monkeypatch.setattr(runner.torch.cuda, "max_memory_allocated", lambda index: 1024)
    generation = {
        "original_timings": [
            {"cpu_seconds": 50.0, "wall_seconds": 50.0},
            {"cpu_seconds": 0.03, "wall_seconds": 0.1},
            {"cpu_seconds": 0.04, "wall_seconds": 0.2},
        ]
    }
    arm = {
        "timings": [
            {"cpu_seconds": 50.0, "wall_seconds": 50.0},
            {"cpu_seconds": 0.001, "wall_seconds": 0.01},
            {"cpu_seconds": 0.002, "wall_seconds": 0.02},
        ],
        "inference": {"cpu_seconds": 0.0004, "wall_seconds": 0.004},
    }
    config = {"resources": {"output_forecast_reserve_bytes": 32 * 1024**2, "output_bytes": 1024**3}}
    result = runner.combined_profile(
        generation,
        {"supervised": arm, "permuted": copy.deepcopy(arm)},
        tmp_path,
        {"cpu": 0.0, "wall": 1000.0, "gpu_lock_start": 1050.0},
        config,
    )
    assert result["forecast_cpu_seconds"] == pytest.approx(128.24)
    assert result["forecast_wall_seconds"] == pytest.approx(346.8)
    assert result["forecast_gpu_lock_wall_seconds"] == pytest.approx(296.8)
    assert result["forecast_saved_bytes"] < 1024**3


def test_all_four_fits_finish_before_any_panel_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = []

    def train(initial, observed, targets, schedule, weight, destination, counts, guard):
        events.append(("train", destination.name))
        return {arm: nn.Linear(1, 1) for arm in runner.ARMS}

    def closure(parent, plan, models):
        events.append(("closure", plan["fit_id"]))
        return {"all_exact": True}

    def infer(model, observed, path, panel, arm, counts, guard):
        events.append(("infer", panel, path.parent.parent.name, observed.shape[0]))

    monkeypatch.setattr(runner.fit_runtime, "train_pair", train)
    monkeypatch.setattr(runner, "exact_old_fit_replay", closure)
    monkeypatch.setattr(runner, "evaluate_panel", infer)
    observed = np.broadcast_to(np.zeros((4, 12, 5000), dtype=np.float32), (128, 4, 12, 5000))
    plans = [
        {
            "fit_id": index,
            "initial": {},
            "schedule": np.empty((400, 8)),
            "directory": tmp_path / f"fit_{index}",
        }
        for index in range(4)
    ]
    runner.train_and_evaluate(
        plans,
        {"observed": observed},
        {"observed": observed[:64]},
        {"learning": tmp_path},
        {},
        20,
        runner.initial_counts(),
        lambda: None,
        lambda: None,
    )
    assert [event for event in events if event[0] == "train"] == [("train", f"fit_{i}") for i in range(4)]
    assert events.index(("closure", 0)) < events.index(("train", "fit_1"))
    assert next(i for i, event in enumerate(events) if event[0] == "infer") > events.index(("train", "fit_3"))
    assert sum(event[0] == "infer" for event in events) == 14
    assert not any(event[0] == "infer" and event[1] == "spent" and event[2] == "fit_0" for event in events)


def test_old_loss_corruption_is_a_retained_stop_before_new_fits(tmp_path: Path) -> None:
    parent, destination = tmp_path / "parent", tmp_path / "fit_0"
    models = {arm: nn.Linear(1, 1) for arm in runner.ARMS}
    for arm, model in models.items():
        (parent / arm).mkdir(parents=True)
        (destination / arm).mkdir(parents=True)
        previous = np.zeros(400, dtype=np.float64)
        current = previous.copy()
        if arm == "supervised":
            current[271] = 0.1
        np.save(parent / arm / "losses.npy", previous)
        np.save(destination / arm / "losses.npy", current)
        torch.save(model.state_dict(), parent / arm / "final_state.pt")
        optimizer = {"state": {}, "param_groups": []}
        torch.save(optimizer, parent / arm / "optimizer.pt")
        torch.save(optimizer, destination / arm / "optimizer.pt")
    with pytest.raises(ValueError, match="400-update"):
        runner.exact_old_fit_replay(parent, {"directory": destination}, models)
    saved = json.loads((destination / "old_training_closure.json").read_text())
    assert not saved["all_exact"]
    assert not saved["arms"]["supervised"]["all_400_losses_exact"]
    assert saved["arms"]["permuted"]["all_400_losses_exact"]


def test_actual_controller_bad_parent_stops_before_any_nn_operation_and_charges_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = runner.read_json(runner.from_stored(runner.CONFIG))
    first = config["parents"]["generator"]
    parent = tmp_path / first["root"]
    parent.mkdir(parents=True)
    (parent / "summary.json").write_text("{}")
    monkeypatch.setattr(runner.generator_runtime, "set_runtime", lambda: None)

    def forbidden(*args: Any) -> None:
        raise AssertionError("No new neural-network operation may precede identity admission")

    monkeypatch.setattr(runner, "load_public_generator", forbidden)
    with pytest.raises(ValueError, match="parent identity mismatch"):
        runner.run_study(tmp_path)
    destination = tmp_path / config["output_root"]
    failure = json.loads((destination / "failure.json").read_text())
    ledger = json.loads((destination / "prospective_ledger.json").read_text())
    assert all(value == 0 for value in failure["operation_totals"].values())
    assert ledger["cpu_seconds"] >= failure["cpu_seconds"]
    assert ledger["gpu_wall_seconds"] == 0
    assert ledger["prior_accounting_verified"] is False
    with pytest.raises(FileExistsError):
        runner.run_study(tmp_path)


def test_fit_input_mutation_stops_before_inference_and_retains_failure_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed = np.broadcast_to(np.zeros((4, 12, 5000), dtype=np.float32), (128, 4, 12, 5000))
    monkeypatch.setattr(runner.generator_runtime, "set_runtime", lambda: None)
    monkeypatch.setattr(runner, "snapshot", lambda *args: {})
    monkeypatch.setattr(runner, "admit_parents", lambda *args: {"generator": tmp_path, "learning": tmp_path})
    monkeypatch.setattr(runner, "predecessor_closures", lambda *args: {})
    monkeypatch.setattr(runner, "load_public_generator", lambda *args: (None, "unused"))
    monkeypatch.setattr(runner, "prepare_fit_plans", lambda *args: [])
    monkeypatch.setattr(runner, "plan_hashes", lambda *args: {})

    def mutate_during_fit(*args: Any) -> dict[str, nn.Module]:
        target.write_bytes(b"changed during a fit")
        return {arm: nn.Linear(1, 1) for arm in runner.ARMS}

    def forbidden_inference(*args: Any) -> None:
        raise AssertionError("Input corruption must stop before any panel inference")

    config = runner.read_json(runner.from_stored(runner.CONFIG))
    destination = tmp_path / config["output_root"]
    destination.parent.mkdir(parents=True)
    target = destination / "fresh_noise.npy"
    monkeypatch.setattr(runner.fit_runtime, "train_pair", mutate_during_fit)
    monkeypatch.setattr(runner, "evaluate_panel", forbidden_inference)

    def execute(*args: Any) -> None:
        generated, guard, verify = args[-3:]
        target.write_bytes(b"admitted numerical input")
        generated[target.name] = runner.file_sha256(target)
        plan = {"fit_id": 0, "initial": {}, "schedule": None, "directory": destination / "fit_0"}
        runner.train_and_evaluate(
            [plan],
            {"observed": observed},
            {"observed": observed[:64]},
            {"learning": tmp_path},
            {},
            20.0,
            runner.initial_counts(),
            guard,
            verify,
        )
        raise AssertionError("The post-fit binding guard must reject mutation")

    monkeypatch.setattr(runner, "execute_gpu", execute)
    with pytest.raises(ValueError, match="immutable replication inputs changed"):
        runner.run_study(tmp_path)
    failure = runner.read_json(destination / "failure.json")
    ledger = runner.read_json(destination / "prospective_ledger.json")
    assert target.name in failure["generated_input_sha256"]
    assert runner.file_sha256(target) != failure["generated_input_sha256"][target.name]
    assert ledger["cpu_seconds"] >= failure["cpu_seconds"]
    assert ledger["gpu_wall_seconds"] == 0
    assert not (destination / "summary.json").exists()
