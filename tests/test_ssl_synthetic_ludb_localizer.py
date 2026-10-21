"""Artificial input, landmark, fixed-fit and failure-accounting checks without released data."""

import json
import time

import numpy as np
import pytest
import torch
from scipy.stats import spearmanr

from ecg_experiment import ssl_synthetic_ludb_localizer as core
from ecg_experiment import ssl_synthetic_ludb_localizer_run as runtime


def structure():
    """Provide explicit duplicate landmarks in a complete artificial twelve-lead record."""
    names = runtime.structural.LEADS
    return {
        "record_id": 1,
        "leads": [
            {"lead": lead, "start": 100, "end_exclusive": 4900, "N_samples": [400, 400, 1200, 3400]}
            for lead in names
        ],
    }


def test_input_uses_only_native_leads_and_rejects_inadmissible_scale():
    physical = np.random.default_rng(5).normal(size=(12, 5000))
    first, parameters = core.prepare_input(physical)
    physical[[2, 3, 4, 5]] += 900
    second, other = core.prepare_input(physical)
    np.testing.assert_array_equal(first, second)
    assert parameters == other
    assert first.dtype == np.float32
    np.testing.assert_array_equal(first[2], first[1] - first[0])
    physical[0] = 1
    with pytest.raises(ValueError, match="robust range"):
        core.prepare_input(physical)


def test_marker_ranks_preserve_ties_duplicates_and_literal_rolls():
    values = np.tile(np.arange(10), 500).astype(np.float32) / 10
    lead = structure()["leads"][0]
    row = core.score_lead(values, values.astype(np.float64), lead)
    reference = values[100:4900]
    positions = np.asarray(lead["N_samples"]) - 100
    expected = [
        (
            np.count_nonzero(reference < reference[position])
            + 0.5 * np.count_nonzero(reference == reference[position])
        )
        / len(reference)
        for position in positions
    ]
    assert row["N_midrank_values"] == expected
    for shift, actual in zip(row["shift_samples"], row["shift_N_midrank_values"], strict=True):
        rolled = np.roll(reference, shift)
        target = [
            (
                np.count_nonzero(reference < rolled[position])
                + 0.5 * np.count_nonzero(reference == rolled[position])
            )
            / len(reference)
            for position in positions
        ]
        assert actual == target
    assert row["N_samples"] == [400, 400, 1200, 3400]


def test_constant_correlations_stay_null_while_primary_remains_half_rank():
    row = core.score_lead(np.full(5000, 0.2, dtype=np.float32), np.zeros(5000), structure()["leads"][0])
    assert row["N_mean_midrank"] == 0.5
    assert row["gain"] == 0
    assert row["energy_association"]["spearman"] is None
    assert row["energy_association"]["spearman_undefined_reason"] == "constant_map_and_energy"
    assert row["energy_association"]["empty_jaccard_union"]
    assert row["energy_association"]["jaccard"] == 0
    assert row["defined_shift_spearman_count"] == 0
    assert row["mean_shift_energy_spearman"] is None


def test_average_tie_spearman_matches_independent_scipy_formula():
    first = np.array([0, 0, 1, 2, 3, 3, 3, 4.0])
    second = np.array([4, 0, 0, 3, 3, 2, 2, 1.0])
    actual = core.association(first, second)
    assert actual["spearman"] == pytest.approx(float(spearmanr(first, second).statistic), abs=1e-14)
    expected_first = first > np.quantile(first, 0.9, method="linear")
    expected_second = second > np.quantile(second, 0.9, method="linear")
    expected = np.count_nonzero(expected_first & expected_second) / np.count_nonzero(
        expected_first | expected_second
    )
    assert actual["jaccard"] == expected


def test_derived_output_maps_never_enter_native_metrics():
    maps = np.random.default_rng(8).uniform(size=(12, 5000)).astype(np.float32)
    energy = np.random.default_rng(9).uniform(size=(8, 5000))
    first = core.score_record(maps, energy, structure(), 1, "B")
    maps[[2, 3, 4, 5]] = 0
    second = core.score_record(maps, energy, structure(), 1, "B")
    assert first == second
    assert [row["lead"] for row in first["leads"]] == list(core.NATIVE_LEADS)


def test_all_fixed_fits_share_record_bootstrap_and_no_fit_resampling():
    rows = []
    for identifier in range(1, 201):
        if identifier % 5 == 0:
            continue
        for fit in core.FITS:
            for arm_index, arm in enumerate(core.ARMS):
                gain = identifier / 1000 + fit / 100 + arm_index / 100
                lead = {
                    "gain": gain,
                    "energy_association": {"spearman": None, "jaccard": 0.0, "empty_jaccard_union": False},
                    "mean_shift_energy_spearman": None,
                    "mean_shift_energy_jaccard": 0.0,
                    "constant_map": False,
                    "probability_mean": 0.2,
                    "probability_q95": 0.3,
                    "probability_max": 0.4,
                }
                rows.append(
                    {
                        "record_id": identifier,
                        "fit_id": fit,
                        "arm": arm,
                        "leads": [lead] * 8,
                        "gain": gain,
                        "N_mean_midrank": 0.5 + gain,
                        "shift_N_mean_midrank": [0.5] * 6,
                    }
                )
    result, draws = core.aggregate(rows, {"bootstrap": {"seed": 2026106501, "draws": 2000}})
    assert draws.shape == (2000, 5)
    np.testing.assert_allclose(draws[:, 4], 0.03, atol=1e-14)
    assert result["per_arm"]["B"]["energy_spearman"]["mean"] is None
    assert result["per_arm"]["B"]["energy_spearman"]["undefined_count"] == 3840
    assert "scientific_passed" not in result
    with pytest.raises(ValueError, match="exactly once"):
        core.aggregate(rows[:-1], {"bootstrap": {"seed": 1, "draws": 2000}})


def test_strict_checkpoint_digest_must_match_parent(tmp_path):
    directory = tmp_path / "fit_1/B"
    directory.mkdir(parents=True)
    state = runtime.architecture.build_model().state_dict()
    torch.save(state, directory / "final_state.pt")
    (tmp_path / "summary.json").write_text(json.dumps({"final_state_sha256": {"1": {"B": "wrong"}}}))
    with pytest.raises(ValueError, match="named-tensor digest"):
        runtime.load_model({"model_root": str(tmp_path)}, 1, "B")


def test_final_enumeration_failure_retains_failed_ledger_and_actual_cost(tmp_path, monkeypatch):
    def interrupted(_path):
        raise TimeoutError("artificial final enumeration interruption")

    monkeypatch.setattr(runtime.shared, "directory_bytes", interrupted)
    config = {
        "study_id": "artificial",
        "resources": {
            "cpu_seconds": 900,
            "wall_seconds": 900,
            "gpu_lock_wall_seconds": 240,
            "peak_rss_bytes": 2 * 1024**3,
            "output_bytes": 768 * 1024**2,
        },
    }
    prior = {
        "sha256": "prior",
        "current_machine_stages": 30,
        "current_machine_cumulative_cpu_seconds": 460.0,
        "current_machine_cumulative_gpu_wall_seconds": 447.0,
    }
    clock = {"cpu": time.process_time(), "wall": time.perf_counter(), "gpu_cost": 0.25}
    with pytest.raises(RuntimeError, match="technical failure"):
        runtime.finish(tmp_path, config, {"prior_ledger": prior}, {"status": "PASS"}, clock, True, {})
    ledger = json.loads((tmp_path / "prospective_ledger.json").read_text())
    assert ledger["status"] == "FAIL"
    assert ledger["technical_passed"] is False
    assert ledger["current_machine_stages"] == 31
    assert ledger["current_machine_cumulative_gpu_wall_seconds"] == 447.25
    assert ledger["current_machine_cumulative_cpu_seconds"] == 460.0 + ledger["cpu_seconds"]
    assert ledger["receipt"] == "finalization_failure.json"


def test_failed_replay_retains_maximum_and_state_identity_before_gate(tmp_path):
    predicted = np.zeros((4, 12, 5000), dtype=np.float32)
    row = runtime.retain_replay(tmp_path, "fit_1_B", predicted, np.ones_like(predicted), "state", 1e-6)
    assert row["all_passed"] is False
    retained = json.loads((tmp_path / "fit_1_B_synthetic_replay.json").read_text())
    assert retained["maximum_absolute_difference"] == 1
    assert retained["initial_tensor_state_sha256"] == "state"
    np.testing.assert_array_equal(np.load(tmp_path / "fit_1_B_synthetic_replay.npy"), predicted)


def test_profile_and_full_use_identical_flush_persistence_and_refuse_overwrite(tmp_path):
    profile = runtime.new_probabilities(tmp_path / "profile.npy", 8)
    full = runtime.new_probabilities(tmp_path / "full.npy", 160)
    batch = np.random.default_rng(1).uniform(size=(8, 12, 5000)).astype(np.float32)
    runtime.write_batch(profile, 0, batch)
    runtime.write_batch(full, 0, batch)
    np.testing.assert_array_equal(np.load(tmp_path / "profile.npy"), np.load(tmp_path / "full.npy")[:8])
    with pytest.raises(FileExistsError):
        runtime.new_probabilities(tmp_path / "profile.npy", 8)


def test_landmark_input_must_match_original_parent_catalog(monkeypatch):
    fixed = {
        "m": "8850ab4119e9d964aa6177ed33edc93559b810dd96038304da6d364685360111",
        "t": "2e48b4e06512f126bb0eddf12306b54c462da1e300fa2624c0effed1c1bc8c29",
        "c": "b34fa682898d0cf381518a43dcc12c73afe3577fc6ccf2b959faf19b02254e0b",
    }

    def read(path):
        root = path.parts[-3] if path.parent.name.startswith("independent") else path.parent.name
        if path.parent.name.startswith("independent"):
            key = "main_sha256" if root == "m" else "main_summary_sha256"
            return {key: fixed[root], "all_passed": True, "mismatch_count": 0}
        return {"output_sha256": {"record_rows.jsonl": "original_N_timestamp_bytes"}}

    monkeypatch.setattr(runtime.shared, "read_json", read)
    config = {"model_root": "m", "landmark_root": "t", "coherence_root": "c"}
    admitted = {root + "/summary.json": digest for root, digest in fixed.items()}
    admitted["t/record_rows.jsonl"] = "altered_current_N_timestamp_bytes"
    with pytest.raises(ValueError, match="original parent manifest"):
        runtime.verify_parent_manifests(config, {"input_sha256": admitted})
