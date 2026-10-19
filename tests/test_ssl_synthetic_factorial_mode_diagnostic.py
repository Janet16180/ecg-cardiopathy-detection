"""Artificial contracts for frozen saved-map mode diagnostics; no research payloads."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ecg_experiment import ssl_synthetic_factorial_mode_diagnostic as diagnostic
from ecg_experiment import ssl_synthetic_paired_diagnostic as scope
from ecg_experiment.ssl_synthetic_metrics import paired_subject_difference


@pytest.fixture
def masks() -> tuple[np.ndarray, np.ndarray]:
    """Construct native truth in one lead and an all-lead algorithmic phase window."""
    truth = np.zeros((12, 2500), dtype=bool)
    candidate = np.zeros_like(truth)
    candidate[:, 800:1200] = True
    truth[0, 900:1100] = True
    return truth, candidate


def test_constant_scores_expected_ties_and_geometry(masks: tuple) -> None:
    """Uniform scores recover prevalence and unbiased expected top-area credit."""
    truth, candidate = masks
    geometry = diagnostic.mask_geometry(truth, candidate, [0.01, 0.05, 0.1])
    result = diagnostic.map_statistics(truth, candidate, np.full(truth.shape, 0.25), geometry)
    prevalence = 200 / 30000
    assert len(result) == 14
    assert result["full_grid_ap"] == prevalence
    assert result["lead_ap"] == 1 / 12
    assert result["temporal_ap"] == result["oracle_temporal_ap"] == 200 / 2500
    for key, area in geometry["area_budgets"].items():
        assert result[f"top_{key}_precision"] == pytest.approx(prevalence)
        assert result[f"top_{key}_recall"] == pytest.approx(area / 30000)
    assert sum(result[f"mass_{region}"] for region in "TUO") == pytest.approx(1)
    assert result["candidate_minus_outside"] == 0
    assert geometry["positive_leads"] == 1
    assert geometry["attainable_recall"] == {"1": 1.0, "5": 1.0, "10": 1.0}


def test_lead_ap_uses_candidate_cells_not_whole_record(masks: tuple) -> None:
    """High outside scores in unaffected leads cannot change candidate-only lead ranking."""
    truth, candidate = masks
    values = np.full(truth.shape, 0.1)
    values[0, candidate[0]] = 0.8
    values[1:, ~candidate[0]] = 0.99
    geometry = diagnostic.mask_geometry(truth, candidate, [0.01, 0.05, 0.1])
    result = diagnostic.map_statistics(truth, candidate, values, geometry)
    assert result["lead_ap"] == 1.0
    assert values[0].mean() < values[1].mean()


def test_oracle_is_equal_full_time_ap_of_true_leads(masks: tuple) -> None:
    """Inactive-lead scores affect temporal marginal AP while the oracle conditions on true leads."""
    truth, candidate = masks
    truth[1, 900:1100] = True
    values = np.full(truth.shape, 0.1)
    values[0, 900:1100] = 0.9
    values[1, 900:1100] = 0.01
    values[2:, 1300:1500] = 1.0
    geometry = diagnostic.mask_geometry(truth, candidate, [0.01, 0.05, 0.1])
    result = diagnostic.map_statistics(truth, candidate, values, geometry)
    assert result["oracle_temporal_ap"] == pytest.approx((1.0 + 200 / 2500) / 2)
    assert result["temporal_ap"] != result["oracle_temporal_ap"]


def test_tie_boundary_and_recall_ceiling() -> None:
    """Fractional ties allocate expected positive credit, with explicit broad-mask coverage ceiling."""
    truth = np.array([True, False, True, False])
    precision, recall = diagnostic.top_area(truth, np.array([1.0, 0.5, 0.5, 0.0]), 2)
    assert precision == recall == 0.75
    native = np.zeros((12, 2500), dtype=bool)
    candidate = np.zeros_like(native)
    native[:3, 900:1300] = True
    candidate[:, 800:1400] = True
    geometry = diagnostic.mask_geometry(native, candidate, [0.01, 0.05, 0.1])
    assert geometry["attainable_recall"]["1"] == 0.25


@pytest.mark.parametrize("defect", ["outside", "all_leads", "empty", "dtype"])
def test_invalid_truth_fails_without_filtering(masks: tuple, defect: str) -> None:
    """Invalid regions/classes stop instead of silently dropping a patient."""
    truth, candidate = masks
    if defect == "outside":
        truth[0, 0] = True
    elif defect == "all_leads":
        truth[:, 900:1100] = True
    elif defect == "empty":
        truth[:] = False
    else:
        truth = truth.astype(np.float32)
    with pytest.raises(ValueError, match="Boolean|nonempty|both"):
        diagnostic.mask_geometry(truth, candidate, [0.01, 0.05, 0.1])


def test_zero_mass_and_nonfinite_stop(masks: tuple) -> None:
    """No epsilon, null statistic or patient exclusion repairs a zero/invalid map."""
    truth, candidate = masks
    geometry = diagnostic.mask_geometry(truth, candidate, [0.01, 0.05, 0.1])
    with pytest.raises(ValueError, match="Zero"):
        diagnostic.map_statistics(truth, candidate, np.zeros(truth.shape), geometry)
    with pytest.raises(ValueError, match="finite"):
        diagnostic.map_statistics(truth, candidate, np.full(truth.shape, np.nan), geometry)


def artificial_cases() -> list[dict]:
    """Make identity-free descriptors covering both new-bank roles and all boundaries."""
    rows = []
    for local in range(192):
        for variant in range(4):
            index = 2000 + 4 * local + variant
            rows.append(
                {
                    "case_index": index,
                    "role_subject_index": local if local < 64 else local - 64,
                    "mode": ("focal", "repeated", "none", "none")[variant],
                    "role": "calibration" if local < 64 else "evaluation",
                    "target": variant < 2,
                    "mask_row": index - 2000,
                    "nuisance": False,
                    "matched_control_case_index": index // 4 * 4 + 2,
                    "baseline_sha256": str(index // 4 * 4 + 2),
                    "edited_sha256": str(index),
                }
            )
    return rows


def test_coordinate_boundaries_and_counterpart_identity() -> None:
    """Only positive evaluation rows256/257..764/765 enter the 256-case analysis."""
    cases = artificial_cases()
    positives = diagnostic.coordinates(cases)
    assert len(positives) == 256
    assert [row["case_index"] for row in positives[:2]] == [2256, 2257]
    assert [row["case_index"] for row in positives[-2:]] == [2764, 2765]
    cases[256]["matched_control_case_index"] = 2259
    with pytest.raises(ValueError, match="metadata"):
        diagnostic.coordinates(cases)


def test_shared_bootstrap_matches_frozen_patient_helper() -> None:
    """Precomputed shared draws reproduce the project's paired patient interval exactly."""
    config = {"bootstrap_seed": 20261028, "bootstrap_draws": 2000}
    values = np.arange(128, dtype=float) / 127
    indices = diagnostic.bootstrap_indices(128, config)
    result = diagnostic.interval(values, indices, config)
    expected = paired_subject_difference(np.arange(128), values, np.zeros(128), 2000, 20261028)
    assert result == expected
    with pytest.raises(ValueError, match="patient vector"):
        diagnostic.interval(values[:-1], indices, config)


def test_forbidden_payloads_are_authority_deferred_before_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    """Closed ECG/checkpoint payloads are never hashed and missing authority fails closed."""
    calls = []
    monkeypatch.setattr(scope, "sha256_file", lambda path: calls.append(path) or "same")
    declarations = {"outputs/old/backgrounds.npy": "closed", "outputs/old/model.pt": "weights"}
    exceptions = diagnostic.verify_bindings(declarations, declarations, [])
    assert len(exceptions) == 2
    assert calls == []
    with pytest.raises(ValueError, match="authority"):
        diagnostic.verify_bindings(declarations, {}, [])
    assert calls == []
    with pytest.raises(ValueError, match="changed"):
        diagnostic.verify_bindings({"docs/protocol.md": "changed"}, {}, [])
    assert len(calls) == 1


def test_failure_retained_and_overwrite_refused(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An admission failure retains costs without continuing, and its destination cannot be retried."""
    destination = tmp_path / "new"
    monkeypatch.setattr(diagnostic, "DESTINATION", destination)
    monkeypatch.setattr(diagnostic, "read_json", lambda path: {})
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    def stop(guard) -> None:
        raise TimeoutError("Artificial admission deadline")

    monkeypatch.setattr(diagnostic, "admit", stop)
    monkeypatch.setattr(diagnostic, "native_arrays", lambda: pytest.fail("Access after failed admission"))
    with pytest.raises(RuntimeError, match="retained"):
        diagnostic.run()
    failure = json.loads((destination / "failure.json").read_text())
    assert failure["failure_type"] == "TimeoutError"
    assert failure["cpu_seconds"] >= 0
    assert failure["wall_seconds"] >= 0
    assert failure["progress"] == {
        "completed": False,
        "scoring_started": False,
        "scoring_completed": False,
        "rows": 0,
    }
    assert failure["processed_waveform_reads"] == failure["checkpoint_payload_reads"] == 0
    with pytest.raises(FileExistsError):
        diagnostic.run()


def test_required_inputs_bound_before_access() -> None:
    """The six numerical inputs and consumed metadata must have explicit declarations."""
    with pytest.raises(ValueError, match="frozen binding"):
        diagnostic.require_bindings({}, ["required.json"])
    diagnostic.require_bindings({"required.json": "digest"}, ["required.json"])


def test_late_failure_keeps_completed_row_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Progress cannot claim zero computed rows after a subsequent patient's failure."""
    positives = diagnostic.coordinates(artificial_cases())[:2]
    metrics = dict.fromkeys(diagnostic.METRICS, 0.0)
    parent = {"method": "RB", "case": 2256}
    for phase in ("baseline_site", "edited"):
        for suffix in ("average_precision", "top_one_percent_precision", "top_one_percent_recall"):
            parent[f"{phase}_{suffix}"] = 0.0
    context = {
        "bindings": dict.fromkeys(diagnostic.allowed_arrays(), "digest"),
        "positives": positives,
        "parent_rows": {"responses": [parent]},
    }
    array = np.broadcast_to(np.zeros((12, 2500)), (768, 12, 2500))
    monkeypatch.setattr(diagnostic, "native_arrays", lambda: ({"RB": array}, array, array))
    monkeypatch.setattr(diagnostic, "mask_geometry", lambda *args: {})
    monkeypatch.setattr(diagnostic, "map_statistics", lambda *args: dict(metrics))
    progress = {"rows": 0}

    def guard() -> None:
        if progress["rows"]:
            raise TimeoutError("Artificial subsequent patient failure")

    with pytest.raises(TimeoutError, match="subsequent"):
        diagnostic.calculate_rows(
            context, {"global_case_offset": 2000, "area_fractions": []}, guard, progress
        )
    assert progress["rows"] == 1


def test_complete_aggregation_and_paired_direction(masks: tuple) -> None:
    """Every field/phase is retained and P minus B preserves identical patient pairing."""
    config = {
        "evaluation_patients": 2,
        "bootstrap_draws": 20,
        "bootstrap_seed": 20261028,
        "arms": ["RB", "SB", "RP", "SP"],
        "modes": ["focal", "repeated"],
    }
    geometry = diagnostic.mask_geometry(*masks, [0.01, 0.05, 0.1])
    rows = []
    for arm_index, arm in enumerate(config["arms"]):
        for mode in config["modes"]:
            for subject in range(2):
                value = float(arm_index * 10 + subject)
                rows.append(
                    {
                        "arm": arm,
                        "subject": subject,
                        "mode": mode,
                        "geometry": geometry,
                        "baseline": dict.fromkeys(diagnostic.METRICS, value),
                        "edited": dict.fromkeys(diagnostic.METRICS, value + 1),
                        "change": dict.fromkeys(diagnostic.METRICS, 1.0),
                    }
                )
    summary = diagnostic.summarize(rows, config, lambda: None)
    for arm in config["arms"]:
        for mode in config["modes"]:
            for phase in diagnostic.PHASES:
                assert set(summary["statistics"][arm][mode][phase]) == set(diagnostic.METRICS)
    for initialization in ("R", "S"):
        for mode in config["modes"]:
            contrast = summary["paired_loss_contrasts"][initialization][mode]
            assert contrast["edited"]["full_grid_ap"]["difference"] == 20.0
            assert contrast["baseline"]["lead_ap"]["ci_low"] == 20.0
            assert contrast["change"]["temporal_ap"]["ci_high"] == 0.0
    assert summary["geometry"]["focal"]["patients"] == 2
    rows[0]["subject"] = 1
    with pytest.raises(ValueError, match="alignment"):
        diagnostic.summarize(rows, config, lambda: None)
