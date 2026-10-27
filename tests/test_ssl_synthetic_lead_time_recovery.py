"""Artificial equivalence and raw-value preservation tests for forward-only AP recovery."""

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ecg_experiment import ssl_synthetic_lead_time_diagnostic as v1
from ecg_experiment import ssl_synthetic_lead_time_recovery as recovery
from ecg_experiment import ssl_synthetic_lead_time_recovery_run as runner


def ordinary_rows() -> list[dict[str, Any]]:
    """Create the complete fixed endpoint grid, with one active lead per artificial row."""
    rows = []
    for fit in [1, 2, 3]:
        for arm in v1.ARMS:
            for index in range(2000, 2064):
                for mode in v1.MODES:
                    value = 0.2 + (index - 2000) * 0.0001 + (0.1 if arm == "PK" else 0)
                    rows.append(
                        {
                            "fit_id": fit,
                            "arm": arm,
                            "index": index,
                            "seed": 2026103000 + index,
                            "mode": mode,
                            "active_leads": [0],
                            "active_lead_AP": [value],
                            "active_lead_temporal_AP": value,
                            "lead_presence_AP": 0.5,
                        }
                    )
    return rows


def roundoff_rows() -> list[dict[str, Any]]:
    """Add the declared one-ULP case endpoint and a separate valid per-lead-only excursion."""
    rows = ordinary_rows()
    for row in rows:
        if (row["fit_id"], row["arm"], row["index"], row["mode"]) == (1, "P", 2004, "focal"):
            value = float.fromhex("0x1.0000000000001p+0")
            row.update(
                active_leads=[8], active_lead_AP=[value], active_lead_temporal_AP=value, lead_presence_AP=1.0
            )
        if (row["fit_id"], row["arm"], row["index"], row["mode"]) == (1, "P", 2046, "focal"):
            values = [1 + 2 * np.finfo(np.float64).eps, 0.8]
            row.update(
                active_leads=[0, 2], active_lead_AP=values, active_lead_temporal_AP=float(np.mean(values))
            )
    return rows


def test_ordinary_rows_match_every_frozen_v1_aggregate_bit_for_bit() -> None:
    rows = ordinary_rows()
    before = copy.deepcopy(rows)
    assert recovery.summarize(rows) == v1.summarize(rows)
    assert rows == before


def test_one_ulp_is_admitted_without_projection_and_per_lead_excursions_are_retained() -> None:
    rows = roundoff_rows()
    before = copy.deepcopy(rows)
    with pytest.raises(ValueError, match="finite APs"):
        v1.summarize(rows)
    result = recovery.summarize(rows)
    assert rows == before
    assert result["arm_fit_means"]["1"]["P"]["focal"]["active_lead_temporal_AP"] == float(
        np.mean(
            [
                row["active_lead_temporal_AP"]
                for row in rows
                if row["fit_id"] == 1 and row["arm"] == "P" and row["mode"] == "focal"
            ]
        )
    )
    assert "scientific_passed" not in result


@pytest.mark.parametrize(
    "value", [-9 * np.finfo(np.float64).eps, 1 + 9 * np.finfo(np.float64).eps, np.nan, np.inf]
)
def test_excursions_beyond_eight_epsilon_and_nonfinite_values_are_rejected(value: float) -> None:
    rows = ordinary_rows()
    rows[0]["active_lead_AP"] = [value]
    rows[0]["active_lead_temporal_AP"] = value
    with pytest.raises(ValueError, match="eight-epsilon"):
        recovery.validate_rows(rows)


def test_exact_envelope_boundaries_are_preserved() -> None:
    rows = ordinary_rows()
    for row, value in zip(rows, [-recovery.AP_TOLERANCE, 1 + recovery.AP_TOLERANCE], strict=False):
        row["active_lead_AP"] = [value]
        row["active_lead_temporal_AP"] = value
    before = copy.deepcopy(rows)
    recovery.validate_rows(rows)
    assert rows == before


def test_identity_and_equal_lead_mean_corruption_are_rejected() -> None:
    rows = ordinary_rows()
    rows[0]["active_lead_AP"] = [0.5]
    with pytest.raises(ValueError, match="mean differs"):
        recovery.validate_rows(rows)
    rows = ordinary_rows()
    rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(ValueError, match="fixed row ordering"):
        recovery.validate_rows(rows)


def test_raw_row_copy_is_byte_identical_and_preserves_all_lead_roundoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, output = tmp_path / "failed", tmp_path / "recovery"
    parent.mkdir()
    output.mkdir()
    rows = roundoff_rows()
    payload = b"".join((json.dumps(row, allow_nan=False) + "\n").encode() for row in rows)
    (parent / "cases.jsonl").write_bytes(payload)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Recovery must not open numerical array inputs")

    monkeypatch.setattr(np, "load", forbidden)
    loaded = runner.preserve_rows(parent, output)
    assert loaded == rows
    assert (output / "cases.jsonl").read_bytes() == payload
    receipt = runner.read_json(output / "roundoff_admission.json")
    assert receipt["strict_range_offending_rows"] == 1
    assert len(receipt["strict_per_lead_outliers"]) == 2
    assert not receipt["raw_values_changed"]


def test_additional_case_endpoint_outlier_is_rejected_after_raw_copy(tmp_path: Path) -> None:
    rows = roundoff_rows()
    rows[0]["lead_presence_AP"] = 1 + np.finfo(np.float64).eps
    parent, output = tmp_path / "failed", tmp_path / "recovery"
    parent.mkdir()
    output.mkdir()
    (parent / "cases.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="exactly one original row"):
        runner.preserve_rows(parent, output)
    assert (output / "cases.jsonl").read_bytes() == (parent / "cases.jsonl").read_bytes()


def test_copied_row_mutation_is_rejected_before_aggregation(tmp_path: Path) -> None:
    parent, failed, output = tmp_path / "B", tmp_path / "failed", tmp_path / "new"
    for directory in [parent, failed, output]:
        directory.mkdir()
    (failed / "cases.jsonl").write_bytes(b"original")
    (output / "cases.jsonl").write_bytes(b"changed")
    catalog = {"parent": {}, "recovery": {"cases.jsonl": runner.file_sha256(failed / "cases.jsonl")}}
    with pytest.raises(ValueError, match="Copied original rows changed"):
        runner.verify(output, {"parent": parent, "recovery": failed}, {}, catalog, lambda: None)


def test_failure_cost_is_recorded_once_without_assuming_unverified_prior(tmp_path: Path) -> None:
    (tmp_path / "failure.json").write_text("{}")
    admission = {
        "prior_ledger": {
            "path": "unverified.json",
            "sha256": "unverified",
            "current_machine_cumulative_cpu_seconds": 410.0,
            "current_machine_cumulative_gpu_wall_seconds": 447.0,
        }
    }
    clock = {"cpu": runner.time.process_time(), "wall": runner.time.perf_counter()}
    runner.finish_ledger(tmp_path, tmp_path, admission, "failure.json", clock, False)
    ledger = runner.read_json(tmp_path / "prospective_ledger.json")
    assert ledger["cpu_seconds"] >= 0
    assert ledger["current_machine_cumulative_cpu_seconds"] is None
    assert ledger["gpu_wall_seconds"] == 0
    with pytest.raises(FileExistsError):
        runner.finish_ledger(tmp_path, tmp_path, admission, "failure.json", clock, False)


def test_actual_controller_identity_failure_stops_before_aggregation_and_preserves_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = runner.read_json(runner.from_stored(runner.CONFIG))
    destination = tmp_path / config["output_root"]
    destination.parent.mkdir(parents=True)
    monkeypatch.setattr(runner, "setup_runtime", lambda *args: None)
    monkeypatch.setattr(runner, "snapshot", lambda *args: {})

    def reject(*args: Any) -> None:
        raise ValueError("artificial recovery input identity differs")

    monkeypatch.setattr(runner, "admit", reject)

    def forbidden(*args: Any) -> None:
        raise AssertionError("Identity failure must stop before row aggregation")

    monkeypatch.setattr(runner.recovery, "summarize", forbidden)
    with pytest.raises(ValueError, match="identity differs"):
        runner.run_study(tmp_path)
    failure = runner.read_json(destination / "failure.json")
    ledger = runner.read_json(destination / "prospective_ledger.json")
    assert ledger["cpu_seconds"] >= failure["cpu_seconds"]
    assert ledger["gpu_wall_seconds"] == 0
    assert ledger["prior_accounting_verified"] is False
    assert failure["new_endpoint_computations"] == 0
    assert not (destination / "summary.json").exists()
    with pytest.raises(FileExistsError):
        runner.run_study(tmp_path)
