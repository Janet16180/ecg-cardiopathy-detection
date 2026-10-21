"""Artificial parser-first ordering and once-only failed-cost recovery checks."""

import copy

import numpy as np
import pytest

from ecg_experiment import ssl_synthetic_ludb_localizer_audit_v3 as audit
from ecg_experiment.ssl_synthetic_ludb_landmark_audit import aggregate


def fixture_rows() -> list[dict]:
    rows = []
    for identifier in range(1, 201):
        leads = [
            {"span_samples": 100, "N_samples": [19, 7, 7], "events": [{"sample": 19}, {"sample": 7}]}
            for _ in range(12)
        ]
        row = {"record_id": identifier, "leads": leads}
        if identifier % 5:
            gain = identifier / 1000
            for lead in leads:
                lead.update(
                    constant_score=False, N_mean_midrank=0.5 + gain, mean_shift_N_midrank=0.5, gain=gain
                )
            row.update(
                N_mean_midrank=0.5 + gain, shift_N_mean_midrank=[0.5] * 6, mean_shift_N_midrank=0.5, gain=gain
            )
        rows.append(row)
    return rows


def test_parser_first_normalization_closes_same_2000_draws_and_never_reorders_n_events() -> None:
    ordered = fixture_rows()
    parser_first = [row for row in ordered if not row["record_id"] % 5]
    parser_first += [row for row in ordered if row["record_id"] % 5]
    untouched = copy.deepcopy(parser_first)
    with pytest.raises(ValueError, match="all200"):
        aggregate(parser_first)
    normalized = audit.canonical_records(parser_first)
    actual, expected = aggregate(normalized), aggregate(ordered)
    assert actual == expected
    np.testing.assert_array_equal(actual["bootstrap_gains"], expected["bootstrap_gains"])
    assert len(actual["bootstrap_gains"]) == 2000
    assert parser_first == untouched
    assert normalized[0] is ordered[0]
    assert normalized[0]["leads"][0]["N_samples"] == [19, 7, 7]
    assert normalized[0]["leads"][0]["events"] == [{"sample": 19}, {"sample": 7}]
    with pytest.raises(ValueError, match="exactly once"):
        audit.canonical_records([*ordered[:-1], ordered[0]])


def test_failed32_cost_is_added_once_and_not_replaced_by_main31() -> None:
    previous = {
        "prior_ledger_sha256": "main31",
        "stages_through_main": 31,
        "cpu_through_main": 604.445725531,
        "gpu_through_main": 473.0495132012293,
    }
    spec = {"cpu_seconds": 37.394049274000004, "ledger_sha256": "failed32"}
    ledger = {
        "prior_accounting_verified": True,
        "prior_ledger_sha256": "main31",
        "current_machine_stages": 32,
        "cpu_seconds": 37.394049274000004,
        "gpu_wall_seconds": 0.0,
        "current_machine_cumulative_cpu_seconds": 641.839774805,
        "current_machine_cumulative_gpu_wall_seconds": 473.0495132012293,
    }
    result = audit.append_failure(previous, ledger, spec)
    assert result["prior_ledger_sha256"] == "failed32"
    assert result["stages_through_main"] == 32
    assert result["cpu_through_main"] == 641.839774805
    ledger["current_machine_cumulative_cpu_seconds"] += ledger["cpu_seconds"]
    with pytest.raises(ValueError, match="exactly once"):
        audit.append_failure(previous, ledger, spec)


def test_main_catalog_keeps_failed_and_current_audit_subtrees_out_of_original_bytes(tmp_path) -> None:
    import hashlib
    import json

    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({"status": "PASS", "technical_passed_before_independent_audit": True, "output_sha256": {}})
    )
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
    (tmp_path / "prospective_ledger.json").write_text(json.dumps(ledger))
    for name in ("independent_audit_v1", "independent_audit_v2", "independent_audit_v3"):
        destination = tmp_path / name / "source_snapshot"
        destination.mkdir(parents=True)
        (destination / "source.py").write_bytes(b"retained audit")
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
