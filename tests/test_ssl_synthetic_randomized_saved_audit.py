"""Artificial checks for independent randomized-support localization verification."""

import numpy as np
import pytest
import torch

from ecg_experiment.ssl_synthetic_randomized_saved_audit import (
    expected_schedule,
    initial_state,
    intervention_values,
    parameters_for_index,
    reconstruct_case,
)
from ecg_experiment.synthetic_audit_io import compare_values, file_sha256, relative_file


def native_fixture():
    return np.random.default_rng(11).normal(0, 0.3, (8, 5000)).astype(np.float32)


def test_each_lead_hull_and_exact_randomized_support():
    seen = set()
    for index in range(20):
        observed, strict, support, row = reconstruct_case(native_fixture(), index)
        assert not support[:2].any()
        assert row["edits"] == parameters_for_index(index)
        for edit in row["edits"]:
            seen.add(edit["independent_lead"])
        assert all(value["retained_energy_fraction"] >= 0.99 for value in row["checks"].values())
        if index == 0:
            observed[2, 0, 10] += 0.5
            difference = observed[2:].astype(float) - observed[1].astype(float)
            corrupt_support = support.copy()
            corrupt_support[2:] = np.abs(difference) > 0.01
            with pytest.raises(ValueError, match="confined to hull"):
                intervention_values(observed, difference != 0, corrupt_support, row["edits"])
    assert seen == set(range(8))


def test_all_training_cases_seen_ten_times_and_no_evaluation_cases():
    schedule = expected_schedule()
    assert schedule.shape == (400, 8)
    assert schedule.min() == 0
    assert schedule.max() == 319
    assert np.array_equal(np.bincount(schedule.ravel()), np.full(320, 10))
    for batch in schedule:
        assert np.array_equal(batch[:4] % 4, np.arange(4))
        assert np.array_equal(batch[4:] % 4, np.arange(4))
        assert batch[0] // 4 != batch[4] // 4


def test_initialization_reproducible_without_forward_and_rng_preserved():
    before = torch.get_rng_state().clone()
    left = initial_state()
    right = initial_state()
    assert set(left) == {"0.weight", "0.bias", "2.weight", "2.bias", "4.weight", "4.bias"}
    assert all(torch.equal(left[key], right[key]) for key in left)
    assert torch.equal(before, torch.get_rng_state())


def test_scalar_and_binding_utilities_detect_corruption(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text("fixture")
    assert len(file_sha256(path)) == 64
    assert relative_file(tmp_path, "metadata.json") == path
    with pytest.raises(ValueError, match="relative audit binding"):
        relative_file(tmp_path, "../private")
    assert compare_values({"a": [0.3, True, None]}, {"a": [0.3, True, None]}, "fixture") == 3
    with pytest.raises(ValueError, match="numeric"):
        compare_values({"a": 0.3}, {"a": 0.31}, "fixture")


def test_main_independent_fixture_integration():
    main = pytest.importorskip("ecg_experiment.ssl_synthetic_randomized_localization")
    for index in (0, 79, 80, 127):
        raw = native_fixture()
        observed, strict, support, row = reconstruct_case(raw, index)
        actual_observed, actual_strict, actual_support, actual_row = main.make_case(raw, index)
        assert actual_observed.tobytes() == observed.tobytes()
        assert actual_strict.tobytes() == strict.tobytes()
        assert actual_support.tobytes() == support.tobytes()
        compare_values(row, actual_row, f"case{index}")
    assert np.array_equal(main.training_schedule(), expected_schedule())


def test_twelve_scientific_gates_preserve_unfavorable_mode():
    from ecg_experiment.ssl_synthetic_randomized_saved_audit import summarize_rows

    def rows(edited):
        return [
            {
                mode: {"edited_ap": edited, "amplitude_ap": 0.1, "response_ap": 0.2, "selectivity": 0.1}
                for mode in ("focal", "repeated")
            }
            for _ in range(48)
        ]

    supervised, permuted = rows(0.3), rows(0.1)
    result = summarize_rows(supervised, permuted)
    assert len(result["scientific_gates"]) == 12
    assert result["scientific_passed"]
    for row in permuted:
        row["repeated"]["edited_ap"] = 0.5
    result = summarize_rows(supervised, permuted)
    assert not result["scientific_passed"]
    assert not result["scientific_gates"]["supervised_minus_permuted_repeated_guard"]


def test_exact_optimizer_steps_and_moment_corruption():
    from ecg_experiment.ssl_synthetic_randomized_saved_audit import exact_tree, verify_optimizer

    parameters = [torch.nn.Parameter(torch.ones(1)) for _ in range(6)]
    optimizer = torch.optim.Adam(parameters, lr=0.001, foreach=False, fused=False)
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    state = optimizer.state_dict()
    models = {str(index): parameter.detach() for index, parameter in enumerate(parameters)}
    verify_optimizer(state, models, 1)
    assert exact_tree(state, state)
    state["state"][0]["step"] += 1
    with pytest.raises(ValueError, match="optimizer step"):
        verify_optimizer(state, models, 1)


def test_control_flags_use_strict_cutoff():
    from ecg_experiment.ssl_synthetic_randomized_saved_audit import control_rows

    probability = np.full((48, 4, 12, 5000), 0.5, dtype=np.float32)
    result = control_rows(probability)
    assert result["sham"]["record_maximum_gt_0p5_fraction"] == 0
    probability[0, 0, 1, 1] = np.nextafter(np.float32(0.5), np.float32(1))
    result = control_rows(probability)
    assert result["sham"]["record_maximum_gt_0p5_fraction"] == 1 / 48


def test_artifact_binding_roots_are_separate(tmp_path):
    from scripts.validation.audit_ssl_synthetic_randomized_localization import verify_bindings

    source, study, parent = (tmp_path / name for name in ("source", "study", "parent"))
    for directory in (source, study, parent, study / "source_snapshot"):
        directory.mkdir(parents=True, exist_ok=True)
    (source / "code.py").write_text("authored")
    (study / "source_snapshot/code.py").write_text("authored")
    (parent / "raw.npy").write_text("parent")
    (study / "observed.npy").write_text("newbank")
    (study / "maps.npy").write_text("maps")
    main = {
        "source_sha256": {"code.py": file_sha256(source / "code.py")},
        "input_sha256": {"raw.npy": file_sha256(parent / "raw.npy")},
        "bank_sha256": {"observed.npy": file_sha256(study / "observed.npy")},
        "output_sha256": {"maps.npy": file_sha256(study / "maps.npy")},
    }
    assert verify_bindings(source, study, parent, main, lambda: None) == 5
    (study / "observed.npy").write_text("corrupt")
    with pytest.raises(ValueError, match="artifact binding"):
        verify_bindings(source, study, parent, main, lambda: None)


def test_saved_training_profile_recovery_checks_every_model_and_optimizer(tmp_path):
    import copy
    import json

    from scripts.validation.audit_ssl_synthetic_randomized_localization import verify_states

    common = initial_state()
    torch.save(common, tmp_path / "initial_state.pt")
    model = torch.nn.Sequential(
        torch.nn.Conv1d(12, 32, 25, padding=12),
        torch.nn.ReLU(),
        torch.nn.Conv1d(32, 32, 25, dilation=4, padding=48),
        torch.nn.ReLU(),
        torch.nn.Conv1d(32, 12, 1),
    )
    model.load_state_dict(common)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, foreach=False, fused=False)
    states = {}
    for step in range(1, 401):
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 0.01)
        optimizer.step()
        if step <= 3:
            states[step] = (copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()))
    for arm in ("supervised", "permuted"):
        directory = tmp_path / arm
        directory.mkdir()
        torch.save(common, directory / "initial_state.pt")
        torch.save(model.state_dict(), directory / "final_state.pt")
        torch.save(optimizer.state_dict(), directory / "optimizer.pt")
        np.save(directory / "losses.npy", np.ones(400, dtype=np.float64))
        profile = tmp_path / "profile" / arm
        profile.mkdir(parents=True)
        torch.save(states[1][0], profile / "step1_model.pt")
        torch.save(states[1][1], profile / "step1_optimizer.pt")
        for step in (2, 3):
            for prefix in ("original", "replay"):
                torch.save(states[step][0], profile / f"{prefix}_step{step}_model.pt")
                torch.save(states[step][1], profile / f"{prefix}_step{step}_optimizer.pt")
        (profile / "summary.json").write_text(
            json.dumps(
                {
                    "losses": [1.0, 0.8, 0.7],
                    "replay_losses": [0.8, 0.7],
                    "trajectory_state_exact": [True, True],
                    "exact_replay": True,
                }
            )
        )
    assert verify_states(tmp_path, lambda: None)["arms"]["supervised"]["adam_steps"] == 400
    corrupt = copy.deepcopy(states[2][0])
    corrupt["0.bias"][0] += 0.1
    torch.save(corrupt, tmp_path / "profile/supervised/replay_step2_model.pt")
    with pytest.raises(ValueError, match="profile model recovery bytes"):
        verify_states(tmp_path, lambda: None)
