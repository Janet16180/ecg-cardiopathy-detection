"""Artificial row-cause and preserved-failure cost checks; no metric calculation."""

import copy
import json
import time

import numpy as np
import pytest

from ecg_experiment.synthetic_audit_io import read_json
from scripts.validation.audit_ssl_synthetic_temporal_failure import row_cause, terminal


def saved_rows():
    rows = [
        {
            "fit_id": fit,
            "arm": arm,
            "index": index,
            "seed": 2026103000 + index,
            "mode": mode,
            "active_leads": [0],
            "active_lead_AP": [0.2],
            "active_lead_temporal_AP": 0.2,
            "lead_presence_AP": 0.3,
        }
        for fit in (1, 2, 3)
        for arm in ("B", "P", "K", "PK")
        for index in range(2000, 2064)
        for mode in ("focal", "repeated")
    ]
    selected = next(
        row
        for row in rows
        if (row["fit_id"], row["arm"], row["index"], row["mode"]) == (1, "P", 2004, "focal")
    )
    value = float(np.nextafter(1.0, np.inf))
    selected.update(active_leads=[8], active_lead_AP=[value], active_lead_temporal_AP=value)
    return rows


def test_raw_single_ulp_cause_without_clipping_or_aggregate():
    rows = saved_rows()
    snapshot = copy.deepcopy(rows)
    result = row_cause(rows)
    assert rows == snapshot
    assert result["maximum_endpoint_ULPs_above_one"] == 1
    assert result["new_AP_computations"] == 0
    assert result["new_descriptive_aggregates"] == 0
    assert len(result["strict_endpoint_outliers"]) == 1
    rows[0]["lead_presence_AP"] = float(np.nextafter(1.0, np.inf))
    with pytest.raises(ValueError, match="single-ULP endpoint"):
        row_cause(rows)
    rows = saved_rows()
    rows[-1]["index"] = 2000
    with pytest.raises(ValueError, match="identity/order"):
        row_cause(rows)
    rows = saved_rows()
    rows[0]["active_lead_AP"][0] = float("nan")
    with pytest.raises(ValueError, match="finite saved per-lead"):
        row_cause(rows)


def test_failure_ledger_cost_added_once_and_immutable(tmp_path):
    (tmp_path / "failure.json").write_text(json.dumps({"all_passed": False, "error": "artificial mismatch"}))
    costs = {
        "main_cpu_seconds": 20.793865505,
        "main_ledger_sha256": "bound",
        "current_machine_cpu_through_main": 410.948816139,
        "current_machine_gpu_wall_through_main": 447.98139213304967,
    }
    terminal(tmp_path, "failure.json", (time.process_time(), time.perf_counter()), costs)
    ledger = read_json(tmp_path / "prospective_ledger.json")
    assert ledger["current_machine_cumulative_cpu_seconds"] == 410.948816139 + ledger["cpu_seconds"]
    assert ledger["stage_main_plus_audit_cpu_seconds"] == 20.793865505 + ledger["cpu_seconds"]
    assert ledger["current_machine_cumulative_gpu_wall_seconds"] == 447.98139213304967
    assert ledger["gpu_wall_seconds"] == 0
    with pytest.raises(FileExistsError):
        terminal(tmp_path, "failure.json", (time.process_time(), time.perf_counter()), costs)
