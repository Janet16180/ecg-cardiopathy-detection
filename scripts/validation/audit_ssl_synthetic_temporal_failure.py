"""Verify only the retained temporal-v1 failure cause and costs, without new metric scores."""

from __future__ import annotations

import argparse
import json
import os
import resource
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np

from ecg_experiment.ssl_synthetic_pretrained_saved_audit import require
from ecg_experiment.synthetic_audit_io import compare_values, file_sha256, read_json, relative_file

RELATIVE = "outputs/ssl_synthetic_localization/generated_temporal_diagnosis_v1"
FAILURE_SHA = "525c05ec36b261ae847885ef664d8547e2a2ed086c87713134ed114ba94c5116"
LEDGER_SHA = "addd84989aba46a9c8a6263d9a4ae1597ffb648766915ccccba019d7db010710"
PRIOR_SHA = "705cc964022f4f1d78d08d7d1eff75a665f02c150e311dacf2e1c4503ede5496"
MAIN_SOURCE = "ecg_experiment/ssl_synthetic_lead_time_diagnostic.py"
MAIN_SHA = "191858915046faf1c1a3b00bd3901b76b412a43777ff71868c3224d215574736"


def row_cause(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate identities and raw AP ranges only; never compute an AP or case aggregate."""
    expected = [
        (fit, arm, index, mode)
        for fit in (1, 2, 3)
        for arm in ("B", "P", "K", "PK")
        for index in range(2000, 2064)
        for mode in ("focal", "repeated")
    ]
    fields = {
        "fit_id",
        "arm",
        "index",
        "seed",
        "mode",
        "active_leads",
        "active_lead_AP",
        "active_lead_temporal_AP",
        "lead_presence_AP",
    }
    require(len(rows) == len(expected) == 1536, "all saved mode rows")
    endpoint_outliers, lead_outliers = [], []
    endpoint_values, lead_values = [], []
    for row, identity in zip(rows, expected, strict=True):
        require(set(row) == fields, "complete saved row fields")
        require(
            (row["fit_id"], row["arm"], row["index"], row["mode"]) == identity,
            "exact saved row identity/order",
        )
        require(row["seed"] == 2026103000 + row["index"], "saved seed formula")
        leads, values = row["active_leads"], row["active_lead_AP"]
        require(
            isinstance(leads, list) and 0 < len(leads) < 12 and leads == sorted(set(leads)),
            "sorted distinct nonempty active leads",
        )
        require(all(type(lead) is int and 0 <= lead < 12 for lead in leads), "canonical lead indices")
        require(isinstance(values, list) and len(values) == len(leads), "one AP per active lead")
        tag = {"fit_id": identity[0], "arm": identity[1], "index": identity[2], "mode": identity[3]}
        for lead, value in zip(leads, values, strict=True):
            require(
                isinstance(value, int | float) and not isinstance(value, bool) and np.isfinite(value),
                "finite saved per-lead AP",
            )
            lead_values.append(float(value))
            if not 0 <= value <= 1:
                lead_outliers.append({**tag, "lead": lead, "value": value})
        for endpoint in ("active_lead_temporal_AP", "lead_presence_AP"):
            value = row[endpoint]
            require(
                isinstance(value, int | float) and not isinstance(value, bool) and np.isfinite(value),
                "finite saved endpoint",
            )
            endpoint_values.append(float(value))
            if not 0 <= value <= 1:
                endpoint_outliers.append({**tag, "endpoint": endpoint, "value": value})
    one_ulp = float(np.nextafter(np.float64(1), np.float64(np.inf)))
    expected_outlier = {
        "fit_id": 1,
        "arm": "P",
        "index": 2004,
        "mode": "focal",
        "endpoint": "active_lead_temporal_AP",
        "value": one_ulp,
    }
    require(endpoint_outliers == [expected_outlier], "exact claimed single-ULP endpoint failure")
    selected = rows[expected.index((1, "P", 2004, "focal"))]
    require(
        selected["active_leads"] == [8] and selected["active_lead_AP"] == [one_ulp],
        "same sole active-lead raw AP",
    )
    epsilon = np.finfo(np.float64).eps
    require(
        min(lead_values) >= -8 * epsilon and max(lead_values) <= 1 + 8 * epsilon, "per-lead roundoff envelope"
    )
    return {
        "rows": 1536,
        "unique_identities": len(expected),
        "nonfinite_values": 0,
        "strict_endpoint_outliers": endpoint_outliers,
        "strict_per_lead_outliers": lead_outliers,
        "endpoint_minimum": min(endpoint_values),
        "endpoint_maximum": max(endpoint_values),
        "per_lead_minimum": min(lead_values),
        "per_lead_maximum": max(lead_values),
        "maximum_endpoint_ULPs_above_one": (max(endpoint_values) - 1) / epsilon,
        "strict_range_predicate_rejected": True,
        "rows_changed": False,
        "new_AP_computations": 0,
        "new_descriptive_aggregates": 0,
    }


def verified_costs(root: Path, study: Path, failure: dict[str, Any], guard: Any) -> dict[str, Any]:
    """Bind the failed receipt and exact once-only prior/main accounting before row checks."""
    ledger = read_json(study / "prospective_ledger.json")
    require(
        ledger["receipt"] == "failure.json" and ledger["receipt_sha256"] == FAILURE_SHA,
        "terminal failed receipt binding",
    )
    prior_path = (
        root
        / "outputs/ssl_synthetic_localization/generated_paired_suppression_v1"
        / "independent_audit_v1/prospective_ledger.json"
    )
    require(
        file_sha256(prior_path, guard) == ledger["prior_ledger_sha256"] == PRIOR_SHA,
        "immediate audited cost anchor",
    )
    prior = read_json(prior_path)
    for field, literal in (("cpu_seconds", 390.154950634), ("gpu_wall_seconds", 447.98139213304967)):
        previous = prior[f"current_machine_cumulative_{field}"]
        require(previous == literal == ledger[f"prior_current_machine_{field}"], "literal prior cost")
        require(np.isfinite(ledger[field]) and ledger[field] >= 0, "finite failed self cost")
        require(
            ledger[f"current_machine_cumulative_{field}"] == previous + ledger[field],
            "once-only failed cost addition",
        )
    require(ledger["gpu_wall_seconds"] == 0, "zero failed diagnostic GPU cost")
    require(
        0 <= failure["cpu_seconds"] <= ledger["cpu_seconds"] < 180
        and 0 <= failure["wall_seconds"] <= ledger["wall_seconds"] < 240,
        "actual failed time limits",
    )
    require(
        all(
            ledger[key] is True
            for key in ("prior_accounting_verified", "cpu_budget_passed", "wall_budget_passed")
        ),
        "failed terminal budget flags",
    )
    files = [
        path
        for path in study.rglob("*")
        if path.is_file() and "independent_failure_audit_v1" not in path.parts
    ]
    size = sum(path.stat().st_size for path in files)
    require(
        size < 16 * 1024**2
        and size - (study / "prospective_ledger.json").stat().st_size
        == ledger["output_bytes_before_terminal_ledger"],
        "actual failed final bytes",
    )
    return {
        "main_cpu_seconds": ledger["cpu_seconds"],
        "main_wall_seconds": ledger["wall_seconds"],
        "main_gpu_wall_seconds": 0.0,
        "main_bytes": size,
        "main_ledger_sha256": LEDGER_SHA,
        "current_machine_cpu_through_main": ledger["current_machine_cumulative_cpu_seconds"],
        "current_machine_gpu_wall_through_main": ledger["current_machine_cumulative_gpu_wall_seconds"],
    }


def terminal(destination: Path, receipt: str, clock: tuple[float, float], costs: dict[str, Any]) -> None:
    """Charge this narrow audit once after its receipt hash and byte count, even on failure."""
    digest = file_sha256(destination / receipt)
    size = sum(path.stat().st_size for path in destination.rglob("*") if path.is_file())
    cpu, wall = time.process_time() - clock[0], time.perf_counter() - clock[1]
    row = {
        "receipt": receipt,
        "sha256": digest,
        "cpu_seconds": cpu,
        "wall_seconds": wall,
        "gpu_wall_seconds": 0.0,
        "main_ledger_sha256": costs.get("main_ledger_sha256"),
        "main_cpu_seconds": costs.get("main_cpu_seconds"),
        "stage_main_plus_audit_cpu_seconds": costs["main_cpu_seconds"] + cpu if costs else None,
        "current_machine_cumulative_cpu_seconds": costs["current_machine_cpu_through_main"] + cpu
        if costs
        else None,
        "current_machine_cumulative_gpu_wall_seconds": costs.get("current_machine_gpu_wall_through_main"),
        "audit_cpu_budget_passed": cpu < 60,
        "audit_wall_budget_passed": wall < 60,
        "audit_bytes_before_terminal_ledger": size,
        "historical_costs_verified": False,
        "measurement_endpoint": "after receipt hash/bytes, before ledger assembly/serialization",
    }
    with (destination / "prospective_ledger.json").open("x") as stream:
        json.dump(row, stream, indent=2, allow_nan=False)
        stream.write("\n")
    require(
        cpu < 60
        and wall < 60
        and sum(path.stat().st_size for path in destination.rglob("*") if path.is_file()) < 8 * 1024**2,
        "terminal narrow-audit caps",
    )


def run(root: Path, source_root: Path, cases_sha: str, script_sha: str) -> dict[str, Any]:
    """Run one copied failure-cause check, never a complete scientific diagnostic."""
    clock = time.process_time(), time.perf_counter()
    study = root / RELATIVE
    destination = study / "independent_failure_audit_v1"
    require(
        Path(__file__).absolute() == destination / "audit_script.py", "execute exact copied failure checker"
    )
    require(
        not (destination / "summary.json").exists() and not (destination / "failure.json").exists(),
        "narrow audit already attempted",
    )
    costs: dict[str, Any] = {}
    receipt = "summary.json"

    def guard() -> None:
        require(
            time.process_time() - clock[0] < 58 and time.perf_counter() - clock[1] < 58,
            "narrow audit deadline",
        )
        require(
            sum(path.stat().st_size for path in destination.rglob("*") if path.is_file()) < 7 * 1024**2,
            "narrow audit byte reserve",
        )

    def interrupted(signum: int, frame: Any) -> None:
        raise TimeoutError(f"failure audit timer {signum}")

    for timer, signum in ((signal.ITIMER_REAL, signal.SIGALRM), (signal.ITIMER_PROF, signal.SIGPROF)):
        require(signal.getitimer(timer) == (0, 0), "exclusive failure audit timers")
        signal.signal(signum, interrupted)
        signal.setitimer(timer, 58)
    try:
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA hidden")
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            require(os.environ.get(key) == "1", "single numerical thread")
        require(file_sha256(Path(__file__), guard) == script_sha, "admitted copied checker")
        require(
            file_sha256(source_root / "ecg_experiment/synthetic_audit_io.py", guard)
            == "f9fa5ebff570c0c99ec7a74ecdea393bdab782060d11c6855da662cd3cc0555a",
            "frozen audit I/O",
        )
        require(
            file_sha256(source_root / "ecg_experiment/ssl_synthetic_pretrained_saved_audit.py", guard)
            == "650eb6463bfe48aef307c86ee36145fa302dc3563da373ebff2abedffcef7c09",
            "frozen require helper",
        )
        require(
            file_sha256(study / "failure.json", guard) == FAILURE_SHA
            and file_sha256(study / "prospective_ledger.json", guard) == LEDGER_SHA,
            "frozen failed attempt identities",
        )
        require(file_sha256(study / "cases.jsonl", guard) == cases_sha, "admitted raw saved rows")
        failure = read_json(study / "failure.json")
        costs = verified_costs(root, study, failure, guard)
        require(
            failure["error_type"] == "ValueError"
            and failure["error"] == "Endpoint values must be complete finite APs",
            "exact failed range error",
        )
        require(failure["completed_case_rows"] == 1536, "all rows retained before range failure")
        require(failure["source_sha256"][MAIN_SOURCE] == MAIN_SHA, "frozen strict-range executable")
        source_count = 0
        for name, expected in failure["source_sha256"].items():
            for directory in (source_root, study / "source_snapshot"):
                require(
                    file_sha256(relative_file(directory, name), guard) == expected,
                    "failed source/snapshot binding",
                )
                source_count += 1
        require(
            not (study / "summary.json").exists() and not (study / "lead_and_time_diagnostics.png").exists(),
            "no completed descriptive aggregate/figure",
        )
        rows = [json.loads(line) for line in (study / "cases.jsonl").read_text().splitlines()]
        cause = row_cause(rows)
        expected_context = {key: rows[-1][key] for key in ("fit_id", "arm", "index", "seed", "mode")}
        compare_values(
            expected_context,
            failure["current_or_offending_case"],
            "last current case is separate from range offender",
        )
        for name, expected in (
            ("failure.json", FAILURE_SHA),
            ("prospective_ledger.json", LEDGER_SHA),
            ("cases.jsonl", cases_sha),
        ):
            require(file_sha256(study / name, guard) == expected, "failed artifacts unchanged")
        guard()
        result = {
            "all_passed": True,
            "mismatch_count": 0,
            "failure_sha256": FAILURE_SHA,
            "main_ledger_sha256": LEDGER_SHA,
            "cases_sha256": cases_sha,
            "script_sha256": script_sha,
            "source_bindings_verified": source_count,
            "cause": cause,
            "resources": costs,
            "last_recorded_context": expected_context,
            "new_AP_computations": 0,
            "new_descriptive_aggregates": 0,
            "model_forwards": 0,
            "optimizer_updates": 0,
            "gpu_used": False,
            "raw_or_prediction_arrays_opened": 0,
            "clinical_claim": False,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "cpu_seconds": time.process_time() - clock[0],
            "wall_seconds": time.perf_counter() - clock[1],
        }
        with (destination / receipt).open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return result
    except Exception as error:
        receipt = "failure.json"
        with (destination / receipt).open("x") as stream:
            json.dump(
                {
                    "all_passed": False,
                    "failure_type": type(error).__name__,
                    "error": str(error),
                    "cpu_seconds": time.process_time() - clock[0],
                    "wall_seconds": time.perf_counter() - clock[1],
                },
                stream,
                indent=2,
            )
            stream.write("\n")
        raise
    finally:
        terminal(destination, receipt, clock, costs)
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.setitimer(signal.ITIMER_PROF, 0)


def main() -> None:
    """Parse exact roots and admitted raw-row/copied-checker identities."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cases-sha", required=True)
    parser.add_argument("--script-sha", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root.absolute(), args.source_root.absolute(), args.cases_sha, args.script_sha)))


if __name__ == "__main__":
    main()
