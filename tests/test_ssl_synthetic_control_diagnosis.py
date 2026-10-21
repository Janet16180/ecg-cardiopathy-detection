"""Artificial saved-map diagnosis checks without opening any actual study payloads."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from ecg_experiment import ssl_synthetic_control_diagnosis as diagnosis


def artificial_case() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Provide tied maps, constant nonzero waveforms and known positive support."""
    observed = np.ones((4, 12, 5000), dtype=np.float32)
    support = np.zeros(observed.shape, dtype=bool)
    support[2:, 0, 100:200] = True
    probabilities = {arm: np.full(observed.shape, 0.5, dtype=np.float32) for arm in diagnosis.ARMS}
    prior, _ = diagnosis.ranked_mask(np.zeros((12, 5000), dtype=np.float64))
    return probabilities, observed, support, prior


def test_rank_area_and_stable_flat_index_ties_are_explicit() -> None:
    values = np.array([[0.1, 0.5, 0.5, 0.5], [0.7, 0.5, 0, 0.1]], dtype=np.float32)
    mask, cutoff = diagnosis.ranked_mask(values, 3)
    assert np.flatnonzero(mask).tolist() == [1, 2, 4]
    assert cutoff == {
        "score": 0.5,
        "strictly_above_count": 1,
        "equal_cutoff_count": 4,
        "selected_tied_count": 2,
    }
    all_tied, meta = diagnosis.ranked_mask(np.zeros((12, 5000)))
    assert all_tied.sum() == 600
    assert np.flatnonzero(all_tied).tolist() == list(range(600))
    assert meta["equal_cutoff_count"] == 60000
    with pytest.raises(ValueError, match="finite"):
        diagnosis.ranked_mask(np.full((12, 5000), np.nan))


def test_activity_prepends_the_first_sample_and_tied_decile_is_strict() -> None:
    constant = np.full((12, 5000), 3, dtype=np.float32)
    assert not diagnosis.activity_mask(constant).any()
    pulse = constant.copy()
    pulse[:, ::500] += 1
    active = diagnosis.activity_mask(pulse)
    assert 0 < active.mean() < 0.1
    with pytest.raises(ValueError, match="float32"):
        diagnosis.activity_mask(constant.astype(np.float64))


def test_case_shift_is_common_and_support_persistence_denominators_are_exact() -> None:
    probabilities, observed, support, prior = artificial_case()
    row = diagnosis.diagnose_case(probabilities, observed, support, prior, 80)
    expected_shift = int(np.random.default_rng(2026103680).integers(500, 4501))
    assert row["common_shift"] == expected_shift
    assert row["seed"] == 2026103080
    for arm in diagnosis.ARMS:
        values = row["arms"][arm]
        assert values["control_persistence"] == 1
        assert values["primary_activity_enrichment"] == 0
        for mode in ["focal", "repeated"]:
            assert values["positives"][mode]["target_precision"] == 1 / 6
            assert values["positives"][mode]["target_recall"] == 1
            assert values["positives"][mode]["nuisance_target_precision"] == 1 / 6
            assert values["positives"][mode]["precision_response"] == 0
            assert values["positives"][mode]["nuisance_hotspot_persistence"] == 1
        for variant in values["variants"].values():
            assert variant["lead_counts"] == [600] + [0] * 11
            assert variant["activity_occupancy_per_lead"] == [0] * 12
            assert variant["prior_precision"] == 1
    with pytest.raises(ValueError, match="evaluation"):
        diagnosis.diagnose_case(probabilities, observed, support, prior, 79)
    probabilities["supervised"] = probabilities["supervised"].astype(np.float64)
    with pytest.raises(ValueError, match="float32"):
        diagnosis.diagnose_case(probabilities, observed, support, prior, 80)


def test_shift_changes_alignment_without_changing_activity_occupancy() -> None:
    hotspot = np.zeros((12, 5000), dtype=bool)
    hotspot[0, 100:700] = True
    active = hotspot.copy()
    prior = hotspot.copy()
    native = diagnosis.variant_values(hotspot, active, prior, 1000)
    assert native["activity_precision"] == 1
    assert native["shifted_activity_precision"] == 0
    assert native["activity_enrichment"] == 1
    assert native["activity_occupancy"] == 0.01
    assert np.roll(active, 1000, axis=1).mean() == active.mean()


def test_three_primary_estimates_use_paired_seed_values_and_csv_closes(tmp_path: Path) -> None:
    probabilities, observed, support, prior = artificial_case()
    template = diagnosis.diagnose_case(probabilities, observed, support, prior, 80)
    rows = []
    for index in range(48):
        row = copy.deepcopy(template)
        row["index"], row["seed"] = index + 80, 2026103080 + index
        row["arms"]["supervised"]["primary_activity_enrichment"] = index / 100
        row["arms"]["permuted"]["primary_activity_enrichment"] = index / 100 - 0.1
        rows.append(row)
    summary = diagnosis.summarize(rows)
    assert set(summary["primary"]) == {"supervised", "permuted", "supervised_minus_permuted"}
    assert summary["primary"]["supervised_minus_permuted"]["mean"] == pytest.approx(0.1)
    assert summary["primary"]["supervised_minus_permuted"]["ci_low"] == pytest.approx(0.1)
    assert summary["secondary_means"]["supervised"]["variants"]["sham"]["lead_fractions"] == [1] + [0] * 11
    columns = diagnosis.write_cases(tmp_path, rows)
    assert "cutoff_equal_cutoff_count" in columns
    assert "activity_occupancy_per_lead_11" in columns
    assert len((tmp_path / "cases.csv").read_text().splitlines()) == 385
    assert len((tmp_path / "cases.jsonl").read_text().splitlines()) == 48


def test_actual_controller_tampered_identity_stops_before_new_scores_and_retains_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = diagnosis.read_json(diagnosis.from_stored(diagnosis.CONFIG))
    parent = tmp_path / config["predecessor_root"]
    parent.mkdir(parents=True)
    (parent / "observed.npy").write_bytes(b"not the frozen array")
    monkeypatch.setattr(diagnosis, "configure", lambda: None)

    def forbidden(*args: object) -> None:
        raise AssertionError("New diagnostic calculations must remain closed")

    monkeypatch.setattr(diagnosis, "diagnose_case", forbidden)
    with pytest.raises(ValueError, match="predecessor identity"):
        diagnosis.run_diagnosis(tmp_path)
    destination = tmp_path / config["output_root"]
    failure = json.loads((destination / "failure.json").read_text())
    ledger = json.loads((destination / "prospective_ledger.json").read_text())
    assert failure["completed_case_rows"] == 0
    assert failure["new_model_forwards"] == 0
    assert ledger["cpu_seconds"] >= failure["cpu_seconds"]
    assert ledger["gpu_wall_seconds"] == 0
    assert ledger["prior_accounting_verified"] is False
    with pytest.raises(FileExistsError):
        diagnosis.run_diagnosis(tmp_path)


def test_exact_predecessor_closure_detects_old_metric_corruption(tmp_path: Path) -> None:
    probabilities, case, target, _ = artificial_case()
    observed = np.broadcast_to(case, (48, *case.shape))
    support = np.broadcast_to(target, (48, *target.shape))
    maps = {arm: np.broadcast_to(value, observed.shape) for arm, value in probabilities.items()}
    prior = np.zeros((12, 5000), dtype=np.float64)
    prior_rows = {}
    for arm in diagnosis.ARMS:
        rows = diagnosis.learning.score_maps(maps[arm], observed, support)
        diagnosis.learning.add_spatial_prior(rows, support, prior)
        prior_rows[arm] = rows
        path = tmp_path / arm
        path.mkdir()
        diagnosis.write_new_json(
            path / "evaluation_metrics.json",
            {"cases": rows, "controls": diagnosis.learning.control_scores(maps[arm])},
        )
    main = diagnosis.learning.summarize_metrics(prior_rows["supervised"], prior_rows["permuted"])
    main["counts"] = {
        "profile_update_attempts": 10,
        "profile_updates": 10,
        "profile_inference_attempts": 2,
        "profile_inferences": 2,
    }
    for arm in diagnosis.ARMS:
        main["counts"].update(
            {
                arm + "_updates": 400,
                arm + "_update_attempts": 400,
                arm + "_inferences": 24,
                arm + "_inference_attempts": 24,
            }
        )
    arrays = {"maps": maps, "observed": observed, "support": support, "prior": prior, "main": main}
    saved = json.loads((tmp_path / "supervised/evaluation_metrics.json").read_text())
    saved["cases"][0]["focal"]["edited_ap"] += 0.001
    (tmp_path / "supervised/evaluation_metrics.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="exact prior metrics"):
        diagnosis.predecessor_closure(tmp_path, arrays, tmp_path, lambda: None)
    assert not (tmp_path / "predecessor_closure.json").exists()
