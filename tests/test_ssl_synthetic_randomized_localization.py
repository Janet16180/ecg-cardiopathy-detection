"""Artificial learning/support checks without public weights, saved study arrays or CUDA."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from ecg_experiment import ssl_synthetic_randomized_localization as learning
from ecg_experiment import ssl_synthetic_randomized_localization_run as runner


class Tiny(nn.Sequential):
    """Map a tiny artificial waveform to logits while routing test CUDA placement to CPU."""

    def __init__(self) -> None:
        super().__init__(nn.Conv1d(1, 1, 1))

    def to(self, *args, **kwargs):
        return super().to("cpu")


def tiny_path(monkeypatch: pytest.MonkeyPatch):
    """Use real optimizer/loss/state operations with tiny CPU-only artificial inputs."""
    monkeypatch.setattr(learning, "build_model", Tiny)
    actual_update, actual_infer = runner.update, runner.infer
    calls = []

    def cpu_update(model, optimizer, observed, targets, indices, weight, device):
        calls.append(id(model))
        return actual_update(model, optimizer, observed, targets, indices, weight, "cpu")

    monkeypatch.setattr(runner, "update", cpu_update)
    monkeypatch.setattr(runner, "infer", lambda model, values, device: actual_infer(model, values, "cpu"))
    torch.manual_seed(7)
    initial = runner.cpu_tree(Tiny().state_dict())
    observed = np.random.default_rng(8).normal(size=(320, 1, 4)).astype(np.float32)
    targets = (observed > 0).astype(np.float32)
    return initial, observed, targets, calls


def test_new_randomization_and_final_numerical_masks_are_exact() -> None:
    raw = np.random.default_rng(3).normal(size=(8, 5000)).astype(np.float32)
    observed, strict, support, row = learning.make_case(raw, 0)
    repeated = learning.make_case(raw.copy(), 0)
    assert all(np.array_equal(a, b) for a, b in zip((observed, strict, support), repeated[:3], strict=True))
    assert observed.shape == (4, 12, 5000)
    assert not support[:2].any()
    for position in range(2):
        difference = observed[position + 2].astype(np.float64) - observed[1].astype(np.float64)
        assert np.array_equal(strict[position], difference != 0)
        assert np.array_equal(support[position + 2], np.abs(difference) > 0.01)
    for check in row["checks"].values():
        assert check["outside_hull_cells"] == 0
        assert check["retained_energy_fraction"] >= 0.99
    assert learning.edit_parameters(0) != learning.edit_parameters(80)


def test_exact_rng_interfaces_and_all_limb_hulls() -> None:
    rng = np.random.default_rng(2026103400)
    expected = {
        "mode": "focal",
        "independent_lead": int(rng.integers(0, 8)),
        "sign": float(rng.choice([-1.0, 1.0])),
        "amplitude": float(rng.uniform(0.4, 0.8)),
        "width": int(rng.choice([61, 81, 101])),
        "centers": [int(rng.integers(500, 4501))],
        "spacing": None,
    }
    assert learning.edit_parameters(0)[0] == expected
    raw = np.random.default_rng(7).normal(size=(8, 5000)).astype(np.float32)
    affected = set()
    for index in range(12):
        _, _, _, row = learning.make_case(raw, index)
        for parameter, check in zip(row["edits"], row["checks"].values(), strict=True):
            if parameter["independent_lead"] == 0:
                assert check["affected_canonical_leads"] == [0, 2, 3, 4, 5]
            if parameter["independent_lead"] == 1:
                assert check["affected_canonical_leads"] == [1, 2, 3, 4, 5]
            affected.add(parameter["independent_lead"])
    assert len(affected) > 5


def test_schedule_presents_every_training_variant_ten_times_without_evaluation() -> None:
    schedule = learning.training_schedule()
    assert schedule.shape == (400, 8)
    assert schedule.min() == 0
    assert schedule.max() == 319
    assert np.array_equal(np.bincount(schedule.ravel()), np.full(320, 10))
    assert np.array_equal(schedule[:, :4] % 4, np.tile(np.arange(4), (400, 1)))


def test_permuted_mask_derangement_preserves_counts_and_excludes_evaluation() -> None:
    support = np.zeros((80, 4, 12, 5000), dtype=bool)
    for index in range(80):
        support[index, 2, index % 12, index + 500] = True
        support[index, 3, index % 12, index + 1500] = True
    direct = learning.training_targets(support, "supervised").reshape(support.shape)
    null = learning.training_targets(support, "permuted").reshape(support.shape)
    assert not null[:, :2].any()
    assert np.count_nonzero(null) == np.count_nonzero(direct)
    assert np.array_equal(null[79, 2:], support[0, 2:])
    assert all(not np.array_equal(null[i, 2:], direct[i, 2:]) for i in range(80))
    assert learning.positive_weight(support) == 20
    with pytest.raises(ValueError, match="80 training backgrounds"):
        learning.training_targets(np.zeros((128, 4, 12, 5000), dtype=bool), "supervised")


def test_network_geometry_and_observed_only_signature() -> None:
    model = learning.build_model()
    assert model(torch.zeros(2, 12, 5000)).shape == (2, 12, 5000)
    assert [
        (layer.kernel_size, layer.dilation, layer.padding) for layer in model if isinstance(layer, nn.Conv1d)
    ] == [((25,), (1,), (12,)), ((25,), (4,), (48,)), ((1,), (1,), (0,))]
    assert sum(parameter.numel() for parameter in model.parameters()) == 35660


def test_profile_recovers_complete_step2_and_step3_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initial, observed, targets, _ = tiny_path(monkeypatch)
    counts = runner.count_template()
    result = runner.profile_arm(
        "supervised",
        initial,
        observed,
        targets,
        learning.training_schedule(),
        2,
        tmp_path,
        counts,
        lambda: None,
    )
    assert result["exact_replay"]
    assert result["trajectory_state_exact"] == [True, True]
    assert result["losses"][1:] == result["replay_losses"]
    assert counts["profile_updates"] == counts["profile_update_attempts"] == 5
    assert counts["profile_inferences"] == 1
    path = tmp_path / "profile/supervised"
    for step in [2, 3]:
        for kind in ["model", "optimizer"]:
            original = torch.load(path / f"original_step{step}_{kind}.pt", weights_only=True)
            replay = torch.load(path / f"replay_step{step}_{kind}.pt", weights_only=True)
            assert runner.trees_equal(original, replay)


def test_real_optimizer_controller_interleaves_400_steps_and_keeps_fresh_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initial, observed, targets, calls = tiny_path(monkeypatch)
    counts = runner.count_template()
    runner.train_pair(
        initial,
        observed,
        {"supervised": targets, "permuted": targets[::-1].copy()},
        learning.training_schedule(),
        2,
        tmp_path,
        counts,
        lambda: None,
    )
    assert len(calls) == 800
    assert calls[0] != calls[1]
    assert calls[::2] == [calls[0]] * 400
    assert calls[1::2] == [calls[1]] * 400
    for arm in ["supervised", "permuted"]:
        assert counts[arm + "_updates"] == counts[arm + "_update_attempts"] == 400
        saved = torch.load(tmp_path / arm / "initial_state.pt", weights_only=True)
        assert runner.trees_equal(initial, saved)
        optimizer = torch.load(tmp_path / arm / "optimizer.pt", weights_only=True)
        assert all(float(state["step"]) == 400 for state in optimizer["state"].values())


def test_support_failure_retains_offending_memmap_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = np.random.default_rng(5).normal(size=(1, 8, 5000)).astype(np.float32)
    actual = learning.make_case

    def bad_case(*args):
        observed, strict, support, row = actual(*args)
        row["checks"]["focal"]["outside_hull_cells"] = 1
        return observed, strict, support, row

    monkeypatch.setattr(learning, "make_case", bad_case)
    rows = []
    with pytest.raises(ValueError, match="integrity/energy"):
        runner.prepare_bank(raw, tmp_path, rows, lambda: None)
    assert len(rows) == 1
    saved = np.load(tmp_path / "observed.npy", mmap_mode="r", allow_pickle=False)
    assert np.array_equal(saved[0], actual(raw[0], 0)[0])
    assert np.load(tmp_path / "support.npy", mmap_mode="r", allow_pickle=False)[0].any()


def test_complete_twelve_gate_rule_and_shared_seed_bootstrap() -> None:
    supervised, permuted = [], []
    for _ in range(48):
        direct_mode = {"edited_ap": 0.8, "amplitude_ap": 0.2, "response_ap": 0.3, "selectivity": 0.1}
        null_mode = {"edited_ap": 0.4}
        supervised.append({"focal": direct_mode.copy(), "repeated": direct_mode.copy()})
        permuted.append({"focal": null_mode.copy(), "repeated": null_mode.copy()})
    result = learning.summarize_metrics(supervised, permuted)
    assert len(result["scientific_gates"]) == 12
    assert result["scientific_passed"]
    assert result["contrasts"]["supervised_minus_absolute_amplitude_pooled"]["mean"] == pytest.approx(0.6)
    assert result["contrasts"]["supervised_minus_permuted_pooled"]["mean"] == pytest.approx(0.4)
    for row in supervised:
        row["repeated"]["response_ap"] = -0.1
    failed = learning.summarize_metrics(supervised, permuted)
    assert not failed["scientific_passed"]
    assert not failed["scientific_gates"]["supervised_repeated_response_ap_lower"]


def test_profile_forecasts_distinguish_whole_stage_and_gpu_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = {
        "timings": [
            {"cpu_seconds": 2.0, "wall_seconds": 10.0},
            {"cpu_seconds": 0.1, "wall_seconds": 0.2},
            {"cpu_seconds": 0.1, "wall_seconds": 0.2},
        ],
        "inference": {"cpu_seconds": 0.01, "wall_seconds": 0.02},
        "exact_replay": True,
    }
    monkeypatch.setattr(runner, "profile_arm", lambda *args: row)
    monkeypatch.setattr(runner.time, "process_time", lambda: 20.0)
    monkeypatch.setattr(runner.time, "perf_counter", lambda: 1100.0)
    monkeypatch.setattr(runner.torch.cuda, "max_memory_allocated", lambda index: 1024)
    clock = {"cpu": 0.0, "wall": 1000.0, "gpu_lock_start": 1050.0}
    result = runner.profile_models(
        {},
        np.empty(0),
        {"supervised": np.empty(0), "permuted": np.empty(0)},
        np.empty(0),
        20,
        tmp_path,
        clock,
        runner.count_template(),
        lambda: None,
    )
    assert result["forecast_wall_seconds"] == pytest.approx(481.92)
    assert result["forecast_gpu_lock_wall_seconds"] == pytest.approx(431.92)
    assert result["forecast_cpu_seconds"] == pytest.approx(240.96)


def test_actual_controller_missing_parent_stops_before_updates_and_retains_own_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = json.loads((runner.ROOT / runner.CONFIG).read_text())
    (tmp_path / config["output_root"]).parent.mkdir(parents=True)
    monkeypatch.setattr(runner.runtime, "set_runtime", lambda: None)
    with pytest.raises(FileNotFoundError):
        runner.run_study(tmp_path)
    destination = tmp_path / config["output_root"]
    failure = json.loads((destination / "failure.json").read_text())
    ledger = json.loads((destination / "prospective_ledger.json").read_text())
    assert all(value == 0 for value in failure["counts"].values())
    assert ledger["cpu_seconds"] >= failure["cpu_seconds"]
    assert ledger["prior_accounting_verified"] is False
    assert ledger["current_machine_cumulative_cpu_seconds"] is None
    with pytest.raises(FileExistsError):
        runner.run_study(tmp_path)
