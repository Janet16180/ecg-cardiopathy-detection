"""Artificial checks for independent frozen-localizer audit arithmetic and CPU architecture."""

import copy

import numpy as np
import pytest
from scipy.stats import spearmanr

from ecg_experiment import ssl_synthetic_ludb_localizer_audit as audit


def test_tied_rank_association_matches_scipy_and_keeps_undefined() -> None:
    first = np.array([0.0, 0.0, 2.0, 4.0, 4.0, 9.0, 3.0, 2.0, 1.0, 5.0])
    second = np.array([4.0, 3.0, 2.0, 1.0, 5.0, 6.0, 7.0, 8.0, 8.0, 8.0])
    row = audit.association(first, second)
    assert row["spearman"] == pytest.approx(spearmanr(first, second).statistic, abs=1e-15)
    assert row["jaccard"] == 0
    flat = audit.association(np.ones(10), np.ones(10))
    assert flat == {
        "spearman": None,
        "spearman_undefined_reason": "constant_map_and_energy",
        "jaccard": 0.0,
        "empty_jaccard_union": True,
    }
    assert audit.association(np.ones(10), second)["spearman_undefined_reason"] == "constant_map"
    assert audit.association(first, np.ones(10))["spearman_undefined_reason"] == "constant_energy"


def test_midrank_multiplicity_crop_and_positive_roll_are_explicit() -> None:
    values = np.zeros(5000)
    values[17:117] = np.tile(np.arange(10) / 10, 10)
    structural = {"lead": "i", "start": 17, "end_exclusive": 117, "N_samples": [26, 26, 17]}
    row = audit.lead_row(values, np.arange(5000), structural)
    assert row["N_midrank_values"] == [0.95, 0.95, 0.05]
    assert row["N_mean_midrank"] == pytest.approx(0.65)
    ranks = np.tile((np.arange(10) + 0.5) / 10, 10)
    for offset, control in zip(row["shift_samples"], row["shift_N_midrank_values"], strict=True):
        assert control == np.roll(ranks, offset)[[9, 9, 0]].tolist()
    assert row["probability_max"] == 0.9
    invalid = copy.deepcopy(structural)
    invalid["N_samples"] = [117]
    with pytest.raises(ValueError, match="landmark"):
        audit.lead_row(values, np.arange(5000), invalid)


def test_native_input_derivation_and_energy_come_from_standardized_inputs() -> None:
    wave = np.linspace(-3, 4, 5000)
    raw = np.stack([wave ** (1 + index % 2) + index for index in range(12)])
    inputs, transform, energy = audit.prepare(raw)
    assert inputs.dtype == np.float32
    assert energy.dtype == np.float64
    np.testing.assert_array_equal(inputs[2], inputs[1] - inputs[0])
    np.testing.assert_array_equal(inputs[3], -(inputs[0] + inputs[1]) / 2)
    assert len(transform["median"]) == 8
    delta = np.diff(inputs[0].astype(np.float64), prepend=float(inputs[0, 0]))
    expected = np.convolve(delta**2, np.ones(20) / 20, mode="same")
    np.testing.assert_array_equal(energy[0], expected / expected.max())


def fixture_rows() -> list[dict]:
    structural = {"lead": "i", "start": 0, "end_exclusive": 100, "N_samples": [7, 19]}
    base = audit.lead_row(np.linspace(0, 1, 5000), np.arange(5000), structural)
    rows = []
    for fit in audit.FITS:
        for arm_index, arm in enumerate(audit.ARMS):
            for identifier in range(1, 201):
                if not identifier % 5:
                    continue
                gain = identifier / 1000 + fit / 100 + arm_index / 10
                leads = [{**base, "lead": lead, "gain": gain} for lead in audit.LEADS]
                rows.append(
                    {
                        "record_id": identifier,
                        "fit_id": fit,
                        "arm": arm,
                        "leads": leads,
                        "N_mean_midrank": gain + 0.2,
                        "shift_N_mean_midrank": [0.2] * 6,
                        "mean_shift_N_midrank": 0.2,
                        "gain": gain,
                    }
                )
    return rows


def test_record_only_bootstrap_keeps_fixed_fits_and_paired_arm_contrast() -> None:
    rows = fixture_rows()
    config = {"bootstrap": {"seed": 2026106501, "draws": 2000}}
    statistics, draws = audit.aggregate(rows, config)
    assert draws.shape == (2000, 5)
    assert statistics["primary_descriptive"]["mean"] == pytest.approx(0.12)
    np.testing.assert_allclose(draws[:, 4], 0.3, atol=1e-15, rtol=0)
    rng = np.random.default_rng(2026106501)
    from ecg_experiment.intervals import patient_groups, patient_resample

    ids = np.array([identifier for identifier in range(1, 201) if identifier % 5])
    expected = (ids[patient_resample(patient_groups(np.arange(160)), rng)] / 1000 + 0.02).mean()
    assert draws[0, 0] == pytest.approx(expected, abs=1e-15)
    with pytest.raises(ValueError, match="all160"):
        audit.aggregate(rows[:-1], config)
    with pytest.raises(ValueError, match="all160"):
        audit.aggregate([*rows, rows[0]], config)


def test_functional_replay_uses_exact_six_tensor_native_architecture() -> None:
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(129)
    model = torch.nn.Sequential(
        torch.nn.Conv1d(12, 32, 25, padding=12),
        torch.nn.ReLU(),
        torch.nn.Conv1d(32, 32, 25, dilation=4, padding=48),
        torch.nn.ReLU(),
        torch.nn.Conv1d(32, 12, 1),
    ).eval()
    assert sum(parameter.numel() for parameter in model.parameters()) == 35660
    values = np.random.default_rng(32).normal(size=(2, 12, 5000)).astype(np.float32)
    result = audit.functional_replay(model.state_dict(), values)
    with torch.inference_mode():
        expected = model(torch.from_numpy(values)).sigmoid().numpy()
    np.testing.assert_array_equal(result, expected)
    invalid = dict(model.state_dict())
    invalid["2.weight"] = invalid["2.weight"][:, :, :-1]
    with pytest.raises(ValueError, match="architecture"):
        audit.functional_replay(invalid, values)


def test_gpu_main_accounting_preserves_actual_nonzero_gpu_cost_and_rejects_duplicate_charge(tmp_path) -> None:
    import hashlib
    import json

    main = {"status": "PASS", "technical_passed_before_independent_audit": True, "output_sha256": {}}
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(main))
    prior = {
        "sha256": "prior30",
        "current_machine_cumulative_cpu_seconds": 460.0,
        "current_machine_cumulative_gpu_wall_seconds": 447.0,
    }
    ledger = {
        "receipt_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
        "prior_ledger_sha256": "prior30",
        "prior_accounting_verified": True,
        "current_machine_stages": 31,
        "cpu_seconds": 15.0,
        "gpu_wall_seconds": 7.0,
        "wall_seconds": 20.0,
        "peak_rss_bytes": 1000,
        "technical_passed": True,
        "status": "PASS",
        "resource_gates": {"cpu": True, "gpu": True},
        "current_machine_cumulative_cpu_seconds": 475.0,
        "current_machine_cumulative_gpu_wall_seconds": 454.0,
        "output_bytes_before_terminal_ledger": summary.stat().st_size,
    }
    path = tmp_path / "prospective_ledger.json"
    path.write_text(json.dumps(ledger))
    config = {
        "resources": {
            "cpu_seconds": 900,
            "wall_seconds": 900,
            "gpu_lock_wall_seconds": 240,
            "peak_rss_bytes": 2 * 1024**3,
            "output_bytes": 768 * 1024**2,
        }
    }
    costs = audit.main_accounting(tmp_path, tmp_path, {"prior_ledger": prior}, config)
    assert costs["gpu_through_main"] == 454.0
    ledger["current_machine_cumulative_gpu_wall_seconds"] += 7.0
    path.write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match="once-only"):
        audit.main_accounting(tmp_path, tmp_path, {"prior_ledger": prior}, config)
