"""Artificial checks of independent fit-replication arithmetic and saved evidence."""

import copy

import numpy as np
import pytest
import torch

from ecg_experiment.ssl_synthetic_replication_saved_audit import (
    contrast_matrices,
    hotspot_rows,
    initialization,
    panel_controls,
    panel_metrics,
    replication_summary,
    resampled_summaries,
    schedule_for_seed,
)


def artificial_rows(gain=0.1):
    fits = []
    for fit in range(3):
        arms = {arm: [] for arm in ("supervised", "permuted")}
        for background in range(64):
            for arm in arms:
                value = 0.2 + fit * 0.01 + background * 0.0001
                if arm == "supervised":
                    value += gain
                row = {}
                for mode in ("focal", "repeated"):
                    row[mode] = {
                        "edited_ap": value,
                        "amplitude_ap": 0.2,
                        "response_ap": 0.1,
                        "selectivity": 0.1,
                    }
                arms[arm].append(row)
        fits.append(arms)
    return fits


def test_initialization_no_forwards_and_preserves_rng(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("independent audit performed a forward")

    monkeypatch.setattr(torch.nn.Conv1d, "forward", forbidden)
    torch.manual_seed(77)
    before = torch.random.get_rng_state().clone()
    first, repeat, second = (initialization(seed) for seed in (2026103801, 2026103801, 2026103802))
    assert torch.equal(before, torch.random.get_rng_state())
    assert list(first) == ["0.weight", "0.bias", "2.weight", "2.bias", "4.weight", "4.bias"]
    assert all(torch.equal(first[key], repeat[key]) for key in first)
    assert not torch.equal(first["0.weight"], second["0.weight"])


def test_schedule_full_coverage_each_epoch_and_variant_pairing():
    first, second = schedule_for_seed(2026103901), schedule_for_seed(2026103902)
    assert first.shape == (400, 8)
    assert first.dtype == np.int64
    assert not np.array_equal(first, second)
    for epoch in first.reshape(10, 40, 8):
        assert np.array_equal(np.sort(epoch.ravel()), np.arange(320))
        assert np.array_equal(epoch.reshape(40, 2, 4) % 4, np.broadcast_to(np.arange(4), (40, 2, 4)))


def test_crossed_shared_background_draw_prevents_pseudoreplication():
    values = np.tile(np.arange(64, dtype=np.float64), (3, 1))
    result = resampled_summaries({"same_background": values, "paired": values + 5}, True)
    rng = np.random.default_rng(2026104501)
    expected, wrong = [], []
    for _ in range(2000):
        rng.integers(0, 3, 3)
        backgrounds = rng.integers(0, 64, 64)
        expected.append(float(backgrounds.mean()))
    wrong_rng = np.random.default_rng(2026104501)
    for _ in range(2000):
        wrong.append(float(wrong_rng.choice(values.ravel(), 192, replace=True).mean()))
    low, high = np.percentile(expected, [2.5, 97.5])
    assert result["same_background"]["ci_low"] == low
    assert result["same_background"]["ci_high"] == high
    assert low != np.percentile(wrong, 2.5)
    assert result["paired"]["ci_low"] == pytest.approx(low + 5)
    assert result["paired"]["ci_high"] == pytest.approx(high + 5)
    with pytest.raises(ValueError, match="interval geometry"):
        resampled_summaries({"flattened": values.reshape(1, 192)}, True)


def test_fit_variance_is_retained_and_zero_fit_direction_guard_fails():
    fits = artificial_rows()
    positive = replication_summary(fits)
    assert positive["scientific_passed"]
    assert len(positive["scientific_gates"]) == 18
    vectors = contrast_matrices(fits)
    conditional = resampled_summaries({key: value[:1] for key, value in vectors.items()}, False)
    assert positive["per_fit_contrasts"][0] == conditional
    edited = copy.deepcopy(fits)
    for row, other in zip(edited[0]["supervised"], edited[0]["permuted"], strict=True):
        for mode in ("focal", "repeated"):
            row[mode]["edited_ap"] = other[mode]["edited_ap"]
    negative = replication_summary(edited)
    assert not negative["scientific_gates"]["fit_1_supervised_minus_permuted_pooled_mean_positive"]
    assert not negative["scientific_passed"]


def test_tied_map_ap_nuisance_pairing_and_full_grid_denominator():
    support = np.zeros((1, 4, 12, 5000), dtype=bool)
    support[:, 2:, 0, :10] = True
    probability = np.full((1, 4, 12, 5000), 0.5, dtype=np.float32)
    observed = np.ones_like(probability)
    prior = np.zeros((12, 5000), dtype=np.float64)
    rows = panel_metrics(probability, observed, support, prior, [1000])
    assert rows[0]["seed"] == 2026104000
    assert rows[0]["focal"]["edited_ap"] == pytest.approx(10 / 60000)
    assert rows[0]["focal"]["response_ap"] == 0
    assert rows[0]["focal"]["selectivity"] == 0
    assert panel_controls(probability)["sham"]["record_maximum_gt_0p5_fraction"] == 0
    probability[0, 1, 0, 0] = np.nextafter(np.float32(0.5), np.float32(1))
    assert panel_controls(probability)["nuisance"]["record_maximum_gt_0p5_fraction"] == 1
    probability[0, 2, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite native"):
        panel_metrics(probability, observed, support, prior, [1000])


def test_hotspot_stable_tie_cells_and_changed_target_pairing():
    probability = np.zeros((1, 4, 12, 5000), dtype=np.float32)
    support = np.zeros_like(probability, dtype=bool)
    support[:, 2:, 0, :300] = True
    row = hotspot_rows(probability, support, [1000])[0]
    assert row["sham_nuisance_persistence"] == 1
    assert row["focal"]["support_precision"] == 0.5
    assert row["focal"]["support_recall"] == 1
    assert row["focal"]["precision_response"] == 0
    probability[0, 2, 1, :600] = 1
    changed = hotspot_rows(probability, support, [1000])[0]
    assert changed["focal"]["support_precision"] == 0
    assert changed["focal"]["precision_response"] == -0.5
    assert changed["focal"]["nuisance_hotspot_persistence"] == 0


def optimizer_fixture(state, step):
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
            for index, value in enumerate(state.values())
        },
    }


def json_file(path, value):
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_recovery_corrupt_step2_optimizer_is_detected_even_if_step3_matches(tmp_path):
    from scripts.validation.audit_ssl_synthetic_fit_replication import profile_recovery

    state = initialization(2026103501)
    for arm in ("supervised", "permuted"):
        directory = tmp_path / "fit_0/profile" / arm
        directory.mkdir(parents=True)
        torch.save(state, directory / "step1_model.pt")
        torch.save(optimizer_fixture(state, 1), directory / "step1_optimizer.pt")
        json_file(
            directory / "summary.json",
            {
                "losses": [0.3, 0.2, 0.1],
                "replay_losses": [0.2, 0.1],
                "trajectory_state_exact": [True, True],
                "exact_replay": True,
            },
        )
        for step in (2, 3):
            for kind in ("original", "replay"):
                torch.save(state, directory / f"{kind}_step{step}_model.pt")
                torch.save(optimizer_fixture(state, step), directory / f"{kind}_step{step}_optimizer.pt")
    assert set(profile_recovery(tmp_path, lambda: None)) == {"supervised", "permuted"}
    corrupted = optimizer_fixture(state, 2)
    corrupted["state"][0]["exp_avg"].view(-1)[0] = 1
    torch.save(corrupted, tmp_path / "fit_0/profile/supervised/replay_step2_optimizer.pt")
    with pytest.raises(ValueError, match="complete recovered profile Adam"):
        profile_recovery(tmp_path, lambda: None)


def test_old_fit_all400_loss_replay_detects_single_loss_corruption(tmp_path):
    from scripts.validation.audit_ssl_synthetic_fit_replication import states_and_schedules

    study, parent = tmp_path / "new", tmp_path / "old"
    for fit in range(4):
        seed = 2026103501 if fit == 0 else 2026103800 + fit
        schedule_seed = 2026103502 if fit == 0 else 2026103900 + fit
        state = initialization(seed)
        directory = study / f"fit_{fit}"
        directory.mkdir(parents=True)
        torch.save(state, directory / "initial_state.pt")
        schedule = schedule_for_seed(schedule_seed)
        np.save(directory / "training_schedule.npy", schedule)
        np.save(directory / "background_schedule.npy", schedule[:, [0, 4]] // 4)
        json_file(
            directory / "fit_design.json",
            {"fit_id": fit, "initialization_seed": seed, "schedule_seed": schedule_seed},
        )
        for arm in ("supervised", "permuted"):
            arm_dir = directory / arm
            arm_dir.mkdir()
            torch.save(state, arm_dir / "initial_state.pt")
            final = copy.deepcopy(state)
            final["0.bias"][0] += 1
            torch.save(final, arm_dir / "final_state.pt")
            optimizer = optimizer_fixture(final, 400)
            torch.save(optimizer, arm_dir / "optimizer.pt")
            losses = np.linspace(0.4, 0.2, 400, dtype=np.float64)
            np.save(arm_dir / "losses.npy", losses)
            if fit == 0:
                parent.mkdir(exist_ok=True)
                torch.save(state, parent / "initial_state.pt")
                (parent / arm).mkdir(parents=True)
                np.save(parent / arm / "losses.npy", losses)
                torch.save(final, parent / arm / "final_state.pt")
                torch.save(optimizer, parent / arm / "optimizer.pt")
        if fit == 0:
            json_file(
                directory / "old_training_closure.json",
                {
                    "all_exact": True,
                    "arms": {
                        arm: {
                            "all_400_losses_exact": True,
                            "final_model_exact": True,
                            "complete_adam_exact": True,
                        }
                        for arm in ("supervised", "permuted")
                    },
                },
            )
    assert set(states_and_schedules(study, parent, lambda: None)) == {"0", "1", "2", "3"}
    bad = np.load(study / "fit_0/supervised/losses.npy")
    bad[271] += 0.01
    np.save(study / "fit_0/supervised/losses.npy", bad)
    with pytest.raises(ValueError, match="old replay all400 losses exact"):
        states_and_schedules(study, parent, lambda: None)


def test_binding_roots_allowlist_and_midrun_input_change(tmp_path):
    from ecg_experiment.synthetic_audit_io import file_sha256
    from scripts.validation.audit_ssl_synthetic_fit_replication import bindings

    source, study = tmp_path / "source", tmp_path / "study"
    parents = {label: tmp_path / label for label in ("generator", "learning", "diagnosis")}
    catalogs = {}
    for label, directory in parents.items():
        directory.mkdir()
        (directory / "input.json").write_text(label)
        catalogs[label] = {"input.json": file_sha256(directory / "input.json")}
    for directory in (source, study / "source_snapshot"):
        directory.mkdir(parents=True)
        (directory / "source.py").write_text("frozen")
    (study / "generated.npy").write_bytes(b"new")
    (study / "result.json").write_text("result")
    main = {
        "source_sha256": {"source.py": file_sha256(source / "source.py")},
        "input_sha256": catalogs,
        "generated_input_sha256": {"generated.npy": file_sha256(study / "generated.npy")},
        "output_sha256": {"result.json": file_sha256(study / "result.json")},
    }
    config = {"parents": {label: {"sha256": catalog} for label, catalog in catalogs.items()}}
    assert bindings(source, study, parents, main, config, lambda: None) == 7
    (parents["learning"] / "input.json").write_text("changed after fitting")
    with pytest.raises(ValueError, match="parent binding learning"):
        bindings(source, study, parents, main, config, lambda: None)
    main["input_sha256"] = copy.deepcopy(catalogs)
    main["input_sha256"]["learning"]["unapproved_raw.npy"] = "forbidden"
    with pytest.raises(ValueError, match="literal parent allowlist"):
        bindings(source, study, parents, main, config, lambda: None)


def test_fresh_reconstruction_uses_new_editor_indices_and_truth_final_float32():
    from ecg_experiment.ssl_synthetic_randomized_saved_audit import reconstruct_case

    raw = np.random.default_rng(17).normal(size=(8, 5000)).astype(np.float32)
    observed, strict, support, row = reconstruct_case(raw, 1000)
    difference = observed[2:].astype(np.float64) - observed[1].astype(np.float64)
    assert row["seed"] == 2026104000
    assert np.array_equal(strict, difference != 0)
    assert np.array_equal(support[2:], np.abs(difference) > 0.01)
    assert not support[:2].any()
    assert row["edits"] != reconstruct_case(raw, 80)[3]["edits"]
    assert all(value["outside_hull_cells"] == 0 for value in row["checks"].values())
    assert all(value["retained_energy_fraction"] >= 0.99 for value in row["checks"].values())


def test_main_terminal_accounting_rejects_nested_prior_double_charge(monkeypatch, tmp_path):
    from scripts.validation import audit_ssl_synthetic_fit_replication as audit

    study, diagnosis = tmp_path / "stage", tmp_path / "diagnosis"
    study.mkdir()
    prior_path = diagnosis / "independent_audit_v1/prospective_ledger.json"
    prior_cpu, prior_gpu = 44.99976453399999, 38.433311437256634
    json_file(
        prior_path,
        {
            "current_machine_cumulative_cpu_seconds": prior_cpu,
            "current_machine_cumulative_gpu_wall_seconds": prior_gpu,
        },
    )
    old_hash = audit.file_sha256

    def bound_hash(path, guard=lambda: None):
        if path == prior_path:
            return "a2e2c31b6eda8a5aaed2498e69f6ee2bed97bd90d6b8b3d133e6c15a0f617f77"
        return old_hash(path, guard)

    monkeypatch.setattr(audit, "file_sha256", bound_hash)
    timing = {"cpu_seconds": 0.001, "wall_seconds": 0.001}
    generation = {"original_timings": [timing] * 3, "first_input_exact_replay": True}
    training = {arm: {"timings": [timing] * 3, "inference": timing} for arm in ("supervised", "permuted")}
    future = 128 * 0.001 + 6400 * 0.001 + 800 * 0.001 + 90
    profile = {
        "generation": generation,
        "training": training,
        "spent": {"cpu_seconds": 1, "wall_seconds": 1},
        "spent_gpu_lock_wall_seconds": 1,
        "peak_allocated_bytes": 1,
        "forecast_saved_bytes": 100000,
        "forecast_gpu_lock_wall_seconds": 1 + future,
    }
    for kind in ("cpu", "wall"):
        profile[f"recurrent_{kind}_seconds"] = {"generation": 0.001, "update": 0.001, "inference": 0.001}
        profile[f"forecast_{kind}_seconds"] = 1 + future
    json_file(study / "combined_profile.json", profile)
    json_file(study / "generation_profile.json", generation)
    counts = {
        "cpu_generator_attempts": 4,
        "cpu_generator_forwards": 4,
        "gpu_generator_profile_attempts": 4,
        "gpu_generator_profile_forwards": 4,
        "gpu_generator_bank_attempts": 64,
        "gpu_generator_bank_forwards": 64,
        "training_profile": {
            "profile_update_attempts": 10,
            "profile_updates": 10,
            "profile_inference_attempts": 2,
            "profile_inferences": 2,
        },
        "fits": {},
    }
    for fit in range(4):
        values = {}
        for arm in ("supervised", "permuted"):
            values[arm + "_update_attempts"] = values[arm + "_updates"] = 400
            values[arm + "_fresh_inference_attempts"] = values[arm + "_fresh_inferences"] = 32
            values[arm + "_spent_inference_attempts"] = values[arm + "_spent_inferences"] = (
                0 if fit == 0 else 24
            )
        counts["fits"][str(fit)] = values
    operations = {
        "cpu_generator_forwards": 4,
        "gpu_generator_forwards": 68,
        "profile_optimizer_updates": 10,
        "old_replay_optimizer_updates": 800,
        "new_fit_optimizer_updates": 2400,
        "profile_localizer_inference_batches": 2,
        "fresh_localizer_inference_batches": 256,
        "spent_localizer_inference_batches": 144,
        "model_selection_updates": 0,
    }
    main = {
        "profile": profile,
        "counts": counts,
        "operation_totals": operations,
        "completed_fresh_indices": list(range(1000, 1064)),
        "gpu_metadata": {
            "free_bytes_at_admission": 2 * 1024**3,
            "peak_allocated_bytes": 1,
            "torch_threads": 1,
            "interop_threads": 1,
            "deterministic": True,
            "matmul_tf32": False,
            "cudnn_tf32": False,
        },
        "runtime_environment": dict.fromkeys(
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"), "1"
        ),
        "cpu_seconds": 2,
        "wall_seconds": 3,
        "gpu_lock_wall_seconds": 1.5,
        "output_bytes_before_receipt": sum(path.stat().st_size for path in study.iterdir()),
        "output_sha256": {path.name: old_hash(path) for path in study.iterdir()},
    }
    main["runtime_environment"]["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    json_file(study / "summary.json", main)
    ledger = {
        "receipt": "summary.json",
        "receipt_sha256": old_hash(study / "summary.json"),
        "cpu_seconds": 2.1,
        "wall_seconds": 3.1,
        "gpu_wall_seconds": 1.5,
        "cpu_budget_passed": True,
        "wall_budget_passed": True,
        "gpu_budget_passed": True,
        "prior_accounting_verified": True,
        "prior_ledger_sha256": bound_hash(prior_path),
        "prior_current_machine_cpu_seconds": prior_cpu,
        "prior_current_machine_gpu_wall_seconds": prior_gpu,
        "current_machine_cumulative_cpu_seconds": prior_cpu + 2.1,
        "current_machine_cumulative_gpu_wall_seconds": prior_gpu + 1.5,
        "output_bytes_before_terminal_ledger": sum(path.stat().st_size for path in study.iterdir()),
    }
    json_file(study / "prospective_ledger.json", ledger)
    config = {"allowed_operations": operations}
    result = audit.main_resources(tmp_path, study, {"diagnosis": diagnosis}, main, config, lambda: None)
    assert result["main_cpu_seconds"] == 2.1
    ledger["current_machine_cumulative_cpu_seconds"] += prior_cpu
    json_file(study / "prospective_ledger.json", ledger)
    with pytest.raises(ValueError, match="once-only cumulative addition"):
        audit.main_resources(tmp_path, study, {"diagnosis": diagnosis}, main, config, lambda: None)


def test_failure_audit_ledger_charges_self_once_and_refuses_overwrite(tmp_path):
    import time

    from ecg_experiment.synthetic_audit_io import read_json
    from scripts.validation.audit_ssl_synthetic_fit_replication import write_ledger

    json_file(tmp_path / "failure.json", {"all_passed": False, "error": "artificial corruption"})
    costs = {
        "main_cpu_seconds": 2,
        "main_ledger_sha256": "bound",
        "current_machine_cpu_through_main": 47,
        "current_machine_gpu_wall_through_main": 40,
    }
    write_ledger(tmp_path, "failure.json", (time.process_time(), time.perf_counter()), costs)
    ledger = read_json(tmp_path / "prospective_ledger.json")
    assert ledger["stage_main_plus_audit_cpu_seconds"] == 2 + ledger["cpu_seconds"]
    assert ledger["current_machine_cumulative_cpu_seconds"] == 47 + ledger["cpu_seconds"]
    assert ledger["current_machine_cumulative_gpu_wall_seconds"] == 40
    assert ledger["gpu_wall_seconds"] == 0
    assert ledger["audit_cpu_budget_passed"]
    assert ledger["audit_wall_budget_passed"]
    with pytest.raises(FileExistsError):
        write_ledger(tmp_path, "failure.json", (time.process_time(), time.perf_counter()), costs)
