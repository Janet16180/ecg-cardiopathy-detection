"""Artificial tests separating lead ranking from within-lead temporal ranking."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ecg_experiment import ssl_synthetic_lead_time_diagnostic as diagnostic
from ecg_experiment.intervals import patient_groups, patient_resample


def case() -> tuple[np.ndarray, np.ndarray]:
    """Create two edited leads with unequal target durations."""
    support = np.zeros((12, 5000), dtype=bool)
    support[0, :100] = True
    support[1, :1000] = True
    probability = np.full((12, 5000), 0.1, dtype=np.float32)
    probability[0] = 0.9
    probability[1] = 0.8
    return probability, support


def rows() -> list[dict[str, Any]]:
    """Create the complete fixed row grid with a shared high-variance background effect."""
    result = []
    for fit in [1, 2, 3]:
        for arm in diagnostic.ARMS:
            for offset, index in enumerate(range(2000, 2064)):
                for mode in diagnostic.MODES:
                    value = 0.2 + (offset / 64) ** 2 * (0.6 if arm == "PK" else 0.1)
                    result.append(
                        {
                            "fit_id": fit,
                            "arm": arm,
                            "index": index,
                            "seed": 2026103000 + index,
                            "mode": mode,
                            "active_leads": [0],
                            "active_lead_AP": [value],
                            "active_lead_temporal_AP": value,
                            "lead_presence_AP": value + 0.1,
                        }
                    )
    return result


def test_correct_lead_only_model_has_perfect_presence_and_prevalence_temporal_ap() -> None:
    probability, support = case()
    result = diagnostic.score_case(probability, support)
    assert result["lead_presence_AP"] == 1
    assert result["active_lead_AP"] == [100 / 5000, 1000 / 5000]
    assert result["active_lead_temporal_AP"] == pytest.approx(0.11)
    probability[:] = 0.1
    probability[support] = 0.9
    perfect = diagnostic.score_case(probability, support)
    assert perfect["active_lead_temporal_AP"] == 1
    assert perfect["lead_presence_AP"] == 1


def test_equal_lead_weighting_is_not_support_area_weighting_and_offsets_do_not_help() -> None:
    probability, support = case()
    probability[0] = 0.1
    probability[0, support[0]] = 0.9
    probability[1] = 0.8
    result = diagnostic.score_case(probability, support)
    assert result["active_lead_AP"] == [1.0, 0.2]
    assert result["active_lead_temporal_AP"] == 0.6
    assert result["active_lead_temporal_AP"] != pytest.approx((100 + 1000 * 0.2) / 1100)
    probability[1] = 0.2
    assert diagnostic.score_case(probability, support)["active_lead_temporal_AP"] == 0.6


def test_constant_ties_use_full_domain_prevalence() -> None:
    probability, support = case()
    probability[:] = 0.5
    result = diagnostic.score_case(probability, support)
    assert result["active_lead_temporal_AP"] == pytest.approx(0.11)
    assert result["lead_presence_AP"] == 2 / 12


@pytest.mark.parametrize("corruption", ["dtype", "shape", "nan", "range", "empty", "all_leads", "all_times"])
def test_geometry_and_degenerate_targets_stop_without_exclusion(corruption: str) -> None:
    probability, support = case()
    if corruption == "dtype":
        probability = probability.astype(np.float64)
    elif corruption == "shape":
        probability = probability[:11]
    elif corruption == "nan":
        probability[0, 0] = np.nan
    elif corruption == "range":
        probability[0, 0] = 1.1
    elif corruption == "empty":
        support[:] = False
    elif corruption == "all_leads":
        support[:, 0] = True
    else:
        support[0] = True
    with pytest.raises(ValueError, match="native|finite|both"):
        diagnostic.score_case(probability, support)


def test_six_crossed_contrasts_share_one_background_draw_and_preserve_fit_ranges() -> None:
    records = rows()
    summary = diagnostic.summarize(records)
    assert len(summary["descriptive_contrasts"]) == 6
    key = "PK_minus_B__active_lead_temporal_AP__pooled"
    vector = np.array(
        [
            records[(3 * 64 + offset) * 2]["active_lead_temporal_AP"]
            - records[offset * 2]["active_lead_temporal_AP"]
            for offset in range(64)
        ]
    )
    rng = np.random.default_rng(2026105601)
    groups = patient_groups(np.arange(64))
    expected = []
    for _ in range(2000):
        rng.integers(0, 3, 3)
        indices = patient_resample(groups, rng)
        expected.append(float(np.tile(vector[indices], (3, 1)).mean()))
    assert summary["descriptive_contrasts"][key]["ci_low"] == np.quantile(expected, 0.025, method="linear")
    assert summary["descriptive_contrasts"][key]["ci_high"] == np.quantile(expected, 0.975, method="linear")
    assert summary["fit_contrast_ranges"][key]["fit_means"] == [float(vector.mean())] * 3
    assert "scientific_passed" not in summary
    records[0], records[1] = records[1], records[0]
    with pytest.raises(ValueError, match="fixed identities"):
        diagnostic.summarize(records)


def test_saved_rows_preserve_offending_identity_and_completed_prefix(tmp_path: Path) -> None:
    probability, support = case()
    observed_support = np.zeros((64, 4, 12, 5000), dtype=bool)
    observed_support[0, 2] = support
    observed_support[0, 3] = support
    maps = {
        fit: {arm: np.broadcast_to(probability, (64, 4, 12, 5000)) for arm in diagnostic.ARMS}
        for fit in [1, 2, 3]
    }
    completed = []
    context = {"current_case": None}
    with pytest.raises(ValueError, match="both classes"):
        diagnostic.score_rows(observed_support, maps, tmp_path, completed, context, lambda: None)
    assert len(completed) == 2
    assert len((tmp_path / "cases.jsonl").read_text().splitlines()) == 2
    assert context["current_case"] == {
        "fit_id": 1,
        "arm": "B",
        "index": 2001,
        "seed": 2026105001,
        "mode": "focal",
    }


def test_changed_input_binding_is_rejected(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    payload = parent / "allowed.npy"
    payload.write_bytes(b"admitted map bytes")
    catalog = {payload.name: diagnostic.file_sha256(payload)}
    diagnostic.verify(tmp_path, parent, {}, catalog, lambda: None)
    payload.write_bytes(b"changed after admission")
    with pytest.raises(ValueError, match="parent input changed"):
        diagnostic.verify(tmp_path, parent, {}, catalog, lambda: None)


def test_terminal_cost_adds_only_new_cpu_and_preserves_gpu_cumulative(tmp_path: Path) -> None:
    prior = tmp_path / "prior.json"
    prior.write_text('{"verified":"artificial accounting fixture"}')
    output = tmp_path / "study"
    output.mkdir()
    (output / "summary.json").write_text("{}")
    admission = {
        "prior_ledger": {
            "path": "prior.json",
            "sha256": diagnostic.file_sha256(prior),
            "current_machine_cumulative_cpu_seconds": 123.45,
            "current_machine_cumulative_gpu_wall_seconds": 67.89,
        }
    }
    clock = {"cpu": diagnostic.time.process_time(), "wall": diagnostic.time.perf_counter()}
    diagnostic.finish_ledger(output, tmp_path, admission, "summary.json", clock, True)
    ledger = diagnostic.read_json(output / "prospective_ledger.json")
    assert ledger["current_machine_cumulative_cpu_seconds"] == 123.45 + ledger["cpu_seconds"]
    assert ledger["current_machine_cumulative_gpu_wall_seconds"] == 67.89
    assert ledger["gpu_wall_seconds"] == 0
    assert ledger["receipt_sha256"] == diagnostic.file_sha256(output / "summary.json")


def test_actual_controller_closure_failure_prevents_new_scores_and_retains_failure_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = diagnostic.read_json(diagnostic.from_stored(diagnostic.CONFIG))
    admission = {
        "prior_ledger": {
            "path": "unverified.json",
            "sha256": "unverified",
            "current_machine_cumulative_cpu_seconds": 0.0,
            "current_machine_cumulative_gpu_wall_seconds": 0.0,
        }
    }
    admission_file = tmp_path / "admission.json"
    admission_file.write_text(json.dumps(admission))
    original_path = diagnostic.from_stored
    monkeypatch.setattr(
        diagnostic,
        "from_stored",
        lambda name: admission_file if name == diagnostic.ADMISSION else original_path(name),
    )
    (tmp_path / config["output_root"]).parent.mkdir(parents=True)
    monkeypatch.setattr(diagnostic, "setup_runtime", lambda *args: None)
    monkeypatch.setattr(diagnostic, "snapshot", lambda *args: {})
    monkeypatch.setattr(diagnostic, "admit", lambda *args: tmp_path)
    probability, support = case()
    target = np.broadcast_to(support, (64, 4, 12, 5000))
    maps = {
        fit: {arm: np.broadcast_to(probability, (64, 4, 12, 5000)) for arm in diagnostic.ARMS}
        for fit in [1, 2, 3]
    }
    metrics = {
        fit: {
            arm: {
                "cases": [
                    {
                        "index": 2000 + i,
                        "seed": 2026105000 + i,
                        "focal": {"edited_ap": -1.0},
                        "repeated": {"edited_ap": -1.0},
                    }
                    for i in range(64)
                ]
            }
            for arm in diagnostic.ARMS
        }
        for fit in [1, 2, 3]
    }
    monkeypatch.setattr(diagnostic, "load_inputs", lambda *args: (target, maps, metrics))

    def forbidden(*args: Any) -> None:
        raise AssertionError("No new diagnostic endpoint before exact parent closure")

    monkeypatch.setattr(diagnostic, "score_case", forbidden)
    original_finish = diagnostic.finish_ledger
    monkeypatch.setattr(
        diagnostic,
        "finish_ledger",
        lambda destination, root, admission, receipt, clock, verified: original_finish(
            destination, root, admission, receipt, clock, False
        ),
    )
    with pytest.raises(ValueError, match="full-grid edited AP closure"):
        diagnostic.run_study(tmp_path)
    destination = tmp_path / config["output_root"]
    failure = diagnostic.read_json(destination / "failure.json")
    ledger = diagnostic.read_json(destination / "prospective_ledger.json")
    assert failure["completed_case_rows"] == 0
    assert not (destination / "cases.jsonl").exists()
    assert ledger["cpu_seconds"] >= failure["cpu_seconds"]
    assert ledger["gpu_wall_seconds"] == 0
    with pytest.raises(FileExistsError):
        diagnostic.run_study(tmp_path)


def test_diagnostic_numerical_slices_are_positive_modes_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probability, support = case()

    class PositiveOnly:
        def __init__(self, value: np.ndarray) -> None:
            self.value = value

        def __getitem__(self, key: tuple[int, int]) -> np.ndarray:
            assert key[1] in {2, 3}
            return self.value

    maps = {fit: {arm: PositiveOnly(probability) for arm in diagnostic.ARMS} for fit in [1, 2, 3]}
    original = diagnostic.score_case
    checks = []

    def score(value: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
        checks.append(1)
        return original(value, truth)

    monkeypatch.setattr(diagnostic, "score_case", score)
    completed = []
    diagnostic.score_rows(PositiveOnly(support), maps, tmp_path, completed, {}, lambda: None)
    assert len(checks) == 1536
    assert len(completed) == 1536
