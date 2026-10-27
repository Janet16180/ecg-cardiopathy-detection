"""Aggregate immutable diagnostic rows under the prospective float64 roundoff envelope."""

import json
import os
import resource
import signal
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from ecg_experiment import ROOT
from ecg_experiment import ssl_synthetic_lead_time_diagnostic as v1
from ecg_experiment import ssl_synthetic_lead_time_recovery as recovery
from ecg_experiment import ssl_synthetic_paired_suppression as parent_metrics
from ecg_experiment.paths import from_stored, to_stored
from ecg_experiment.synthetic_audit_io import compare_values, file_sha256, read_json, relative_file

CONFIG = "configs/ssl_synthetic_temporal_diagnosis_v2.json"
ADMISSION = "configs/ssl_synthetic_temporal_diagnosis_admission_v2.json"
PROTOCOL = "docs/ssl-synthetic-localization-temporal-diagnosis-v2.md"
SOURCES = tuple(
    dict.fromkeys(
        (
            *v1.SOURCES,
            CONFIG,
            ADMISSION,
            PROTOCOL,
            "docs/ssl-synthetic-localization-temporal-diagnosis-admission-v2.md",
            "docs/ssl-synthetic-localization-temporal-diagnosis-v2-clarification.md",
            "ecg_experiment/ssl_synthetic_lead_time_recovery.py",
            "ecg_experiment/ssl_synthetic_lead_time_recovery_run.py",
            "scripts/reports/recover_ssl_synthetic_lead_time.py",
            "tests/test_ssl_synthetic_lead_time_recovery.py",
        )
    )
)


def admit(
    root: Path, config: dict[str, Any], admission: dict[str, Any], inputs: dict[str, Any], guard: Any
) -> dict[str, Path]:
    """Admit only the29 B files, six failed-attempt receipts/rows and audited terminal cost base."""
    if admission["frozen_config_sha256"] != file_sha256(from_stored(CONFIG), guard) or admission[
        "frozen_protocol_sha256"
    ] != file_sha256(from_stored(PROTOCOL), guard):
        raise ValueError("Frozen recovery design identity differs")
    roots = {}
    for label, names, root_name in [
        ("parent", config["allowed_parent_files"], "parent_root"),
        ("recovery", config["recovery_inputs"], "recovery_parent_root"),
    ]:
        declaration = admission[label]
        if declaration["root"] != config[root_name] or set(declaration["sha256"]) != set(names):
            raise ValueError("Recovery input root/allowlist differs")
        roots[label] = relative_file(root, declaration["root"])
        inputs[label] = {}
        for name, expected in declaration["sha256"].items():
            if file_sha256(relative_file(roots[label], name), guard) != expected:
                raise ValueError("Admitted recovery input identity differs")
            inputs[label][name] = expected
    require_audits(roots)
    prior = admission["prior_ledger"]
    path = relative_file(root, prior["path"])
    if (
        path != roots["recovery"] / "independent_failure_audit_v1/prospective_ledger.json"
        or file_sha256(path, guard) != prior["sha256"]
    ):
        raise ValueError("Recovery audited terminal ledger identity differs")
    ledger = read_json(path)
    for key in ["current_machine_cumulative_cpu_seconds", "current_machine_cumulative_gpu_wall_seconds"]:
        if ledger[key] != prior[key]:
            raise ValueError("Recovery cumulative accounting differs")
    return roots


def require_audits(roots: dict[str, Path]) -> None:
    """Require both the B technical audit and the separately retained v1 failure audit."""
    for directory in [
        roots["parent"] / "independent_audit_v1",
        roots["recovery"] / "independent_failure_audit_v1",
    ]:
        audit = read_json(directory / "summary.json")
        if not audit["all_passed"] or audit["mismatch_count"] != 0:
            raise ValueError("Required independent parent/failure audit did not pass")


def check_design(receipt: dict[str, Any], design: dict[str, str]) -> None:
    """Require each bound parent receipt to name the frozen scientific config and protocol."""
    if any(
        receipt["source_sha256"][design[kind]] != design[kind + "_sha256"] for kind in ["config", "protocol"]
    ):
        raise ValueError("Parent frozen scientific design identity differs")


def predecessor_closure(
    roots: dict[str, Path], inputs: dict[str, Any], destination: Path, config: dict[str, Any]
) -> dict[str, Any]:
    """Close the complete B aggregate and preserve source-bound v1 full-grid closure as inherited evidence."""
    parent, failed = roots["parent"], roots["recovery"]
    main = read_json(parent / "summary.json")
    if not all(main["integrity_gates"].values()):
        raise ValueError("Parent B technical integrity did not pass")
    check_design(main, config["parent_scientific_design"])
    fits = [
        {
            "fit_id": fit,
            **{arm: read_json(parent / f"fit_{fit}" / arm / "fresh_metrics.json") for arm in v1.ARMS},
        }
        for fit in [1, 2, 3]
    ]
    aggregate = parent_metrics.aggregate(fits)
    compare_values(aggregate, main["aggregate"], "B full aggregate", tolerance=0)
    inherited = read_json(failed / "predecessor_closure.json")
    compare_values(aggregate, inherited["aggregate"], "inherited B aggregate", tolerance=0)
    if (
        not inherited["all_exact"]
        or inherited["full_grid_edited_AP_checks"] != 1536
        or inherited["parent_scientific_passed"] != aggregate["scientific_passed"]
    ):
        raise ValueError("Inherited full-grid closure evidence differs")
    failure = read_json(failed / "failure.json")
    check_design(failure, config["recovery_parent_design"])
    if (
        failure["input_sha256"] != inputs["parent"]
        or failure["completed_case_rows"] != 1536
        or failure["error_type"] != "ValueError"
        or failure["error"] != "Endpoint values must be complete finite APs"
    ):
        raise ValueError("Bound v1 failure does not match the declared recovery")
    if (failed / "summary.json").exists() or (failed / "lead_and_time_diagnostics.png").exists():
        raise ValueError("The predecessor unexpectedly contains a completed summary or figure")
    result = {
        "B_aggregate_exact": True,
        "aggregate": aggregate,
        "parent_scientific_passed": aggregate["scientific_passed"],
        "inherited_full_grid_edited_AP_checks": 1536,
        "inherited_full_grid_AP_source_bound": True,
        "v1_technical_failure_preserved": True,
    }
    v1.shared.write_new_json(destination / "predecessor_closure.json", result)
    return result


def preserve_rows(failed: Path, destination: Path) -> list[dict[str, Any]]:
    """Copy raw saved rows byte-identically and reproduce the exact single-row strict-range failure."""
    payload = (failed / "cases.jsonl").read_bytes()
    with (destination / "cases.jsonl").open("xb") as stream:
        stream.write(payload)
    rows = [json.loads(line) for line in payload.splitlines()]
    recovery.validate_rows(rows)
    offending = []
    for row in rows:
        values = [row[name] for name in recovery.ENDPOINTS]
        if any(value < 0 or value > 1 for value in values):
            offending.append(row)
    if len(offending) != 1:
        raise ValueError("Strict version1 range failure must affect exactly one original row")
    row = offending[0]
    value = float.fromhex("0x1.0000000000001p+0")
    expected = {
        "fit_id": 1,
        "arm": "P",
        "index": 2004,
        "seed": 2026105004,
        "mode": "focal",
        "active_leads": [8],
        "active_lead_AP": [value],
        "active_lead_temporal_AP": value,
        "lead_presence_AP": 1.0,
    }
    if row != expected:
        raise ValueError("Original one-ULP offending row identity or raw values differ")
    lead_outliers = [
        {**{key: item[key] for key in ["fit_id", "arm", "index", "mode"]}, "lead": lead, "value": value}
        for item in rows
        for lead, value in zip(item["active_leads"], item["active_lead_AP"], strict=True)
        if value < 0 or value > 1
    ]
    v1.shared.write_new_json(
        destination / "roundoff_admission.json",
        {
            "complete_rows": len(rows),
            "raw_rows_byte_identical": True,
            "tolerance": float(recovery.AP_TOLERANCE),
            "strict_range_offending_rows": 1,
            "strict_per_lead_outliers": lead_outliers,
            "original_offending_row": row,
            "raw_values_changed": False,
        },
    )
    return rows


def snapshot(destination: Path, guard: Any) -> dict[str, str]:
    """Retain the old and new immutable source bytes without editing any frozen file."""
    sources = {}
    for name in SOURCES:
        guard()
        target = destination / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(from_stored(name).read_bytes())
        sources[name] = file_sha256(target, guard)
    return sources


def verify(
    destination: Path, roots: dict[str, Path], sources: dict[str, str], inputs: dict[str, Any], guard: Any
) -> None:
    """Rehash all sources and exact admitted inputs, never numerically opening B arrays."""
    v1.verify(destination, roots["parent"], sources, inputs["parent"], guard)
    for name, expected in inputs["recovery"].items():
        if file_sha256(relative_file(roots["recovery"], name), guard) != expected:
            raise ValueError("Failed-attempt recovery input changed")
    copied = destination / "cases.jsonl"
    if copied.exists() and file_sha256(copied, guard) != inputs["recovery"]["cases.jsonl"]:
        raise ValueError("Copied original rows changed")


def finish_ledger(
    destination: Path,
    root: Path,
    admission: dict[str, Any],
    receipt: str,
    clock: dict[str, float],
    verified: bool,
) -> None:
    """Charge this attempt once to the audited failed-attempt chain after all receipt bookkeeping reads."""
    digest = file_sha256(destination / receipt)
    prior = admission["prior_ledger"]
    if verified and file_sha256(relative_file(root, prior["path"])) != prior["sha256"]:
        raise ValueError("Audited cost base changed before recovery accounting")
    size = v1.shared.directory_bytes(destination)
    cpu, wall = time.process_time() - clock["cpu"], time.perf_counter() - clock["wall"]
    row = {
        "stage": "generated_temporal_diagnosis_v2",
        "receipt": receipt,
        "receipt_sha256": digest,
        "cpu_seconds": cpu,
        "wall_seconds": wall,
        "gpu_wall_seconds": 0.0,
        "prior_ledger_sha256": prior["sha256"] if verified else None,
        "prior_current_machine_cpu_seconds": prior["current_machine_cumulative_cpu_seconds"]
        if verified
        else None,
        "prior_current_machine_gpu_wall_seconds": prior["current_machine_cumulative_gpu_wall_seconds"]
        if verified
        else None,
        "current_machine_cumulative_cpu_seconds": prior["current_machine_cumulative_cpu_seconds"] + cpu
        if verified
        else None,
        "current_machine_cumulative_gpu_wall_seconds": prior["current_machine_cumulative_gpu_wall_seconds"]
        if verified
        else None,
        "prior_accounting_verified": verified,
        "cpu_budget_passed": cpu < 90,
        "wall_budget_passed": wall < 120,
        "output_bytes_before_terminal_ledger": size,
        "historical_costs_verified": False,
        "measurement_endpoint": "after receipt/prior hashes and byte count, before ledger serialization",
    }
    v1.shared.write_new_json(destination / "prospective_ledger.json", row)
    if cpu >= 90 or wall >= 120 or v1.shared.directory_bytes(destination) >= 16 * 1024**2:
        raise ValueError("Terminal recovery resource ceiling exceeded")


def setup_runtime(timers: dict[int, Any], interrupted: Any) -> None:
    """Require hidden CUDA, one numerical thread and the smaller90/120-second recovery budget."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or any(
        os.environ.get(key) != "1"
        for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]
    ):
        raise ValueError("Recovery requires hidden CUDA and one numerical thread")
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    for timer, signum, seconds in [
        (signal.ITIMER_REAL, signal.SIGALRM, 118),
        (signal.ITIMER_PROF, signal.SIGPROF, 88),
    ]:
        if signal.getitimer(timer) != (0.0, 0.0):
            raise ValueError("Recovery requires exclusive process timers")
        timers[timer] = (signum, signal.getsignal(signum))
        signal.signal(signum, interrupted)
        signal.setitimer(timer, seconds)


def run_study(output_root: Path) -> dict[str, Any]:
    """Recover only aggregation from immutable saved endpoints, preserving failures and all raw floats.

    Parameters
    ----------
    output_root : Path
        Local host root containing the declared B and failed-diagnostic artifacts.

    Returns
    -------
    dict
        Unchanged descriptive endpoint comparison, inherited evidence and recovery integrity.
    """
    clock = {"cpu": time.process_time(), "wall": time.perf_counter()}
    config, admission = read_json(from_stored(CONFIG)), read_json(from_stored(ADMISSION))
    destination = output_root / config["output_root"]
    destination.mkdir(exist_ok=False)
    roots, inputs, sources = {}, {}, {}
    timers = {}
    verified = False

    def guard() -> None:
        if time.process_time() - clock["cpu"] + 2 >= 90 or time.perf_counter() - clock["wall"] + 2 >= 120:
            raise TimeoutError("Recovery CPU/wall reserve reached")
        if v1.shared.directory_bytes(destination) + 1024**2 >= 16 * 1024**2:
            raise ValueError("Recovery output receipt reserve exhausted")

    def interrupted(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Recovery process timer {signum}")

    try:
        setup_runtime(timers, interrupted)
        sources = snapshot(destination, guard)
        roots = admit(output_root, config, admission, inputs, guard)
        verified = True
        verify(destination, roots, sources, inputs, guard)
        closure = predecessor_closure(roots, inputs, destination, config)
        rows = preserve_rows(roots["recovery"], destination)
        verify(destination, roots, sources, inputs, guard)
        statistics = recovery.summarize(rows)
        figures = v1.make_figure(destination, statistics["arm_fit_means"])
        verify(destination, roots, sources, inputs, guard)
        guard()
        result = {
            "study_id": config["study_id"],
            "created_at_utc": datetime.now(UTC).isoformat(),
            "output_relative": to_stored(ROOT / config["output_root"]),
            **statistics,
            "predecessor_closure": closure,
            "integrity_gates": {
                "B_aggregate_exact": True,
                "inherited_v1_full_grid_closure": True,
                "v1_failure_preserved": True,
                "raw_rows_byte_identical": True,
                "original_one_ULP_failure_reproduced": True,
                "all_rows_admitted_without_changes": True,
                "sources_and_inputs_unchanged": True,
            },
            "technical_passed": True,
            "source_sha256": sources,
            "input_sha256": inputs,
            "figures": figures,
            "case_rows": len(rows),
            "operations": config["allowed_operations"],
            "new_endpoint_computations": 0,
            "raw_values_changed": False,
            "AP_admission_tolerance": float(recovery.AP_TOLERANCE),
            "gpu_used": False,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "runtime_environment": {
                key: os.environ.get(key)
                for key in [
                    "CUDA_VISIBLE_DEVICES",
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                ]
            },
            "oracle_conditioning": "active-lead oracle at scoring only; not deployable performance",
            "derived_lead_dependence": "equal canonical-lead weights do not imply physical independence",
            "clinical_claim": False,
            "historical_costs_verified": False,
            "cpu_seconds": time.process_time() - clock["cpu"],
            "wall_seconds": time.perf_counter() - clock["wall"],
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "output_bytes_before_receipt": v1.shared.directory_bytes(destination),
            "output_sha256": {
                path.relative_to(destination).as_posix(): file_sha256(path, guard)
                for path in destination.rglob("*")
                if path.is_file()
            },
        }
        guard()
        v1.shared.write_new_json(destination / "summary.json", result)
        guard()
        finish_ledger(destination, output_root, admission, "summary.json", clock, verified)
        return result
    except Exception as error:
        v1.shared.write_new_json(
            destination / "failure.json",
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "source_sha256": sources,
                "input_sha256": inputs,
                "cpu_seconds": time.process_time() - clock["cpu"],
                "wall_seconds": time.perf_counter() - clock["wall"],
                "operations": config["allowed_operations"],
                "new_endpoint_computations": 0,
                "raw_values_changed": False,
                "gpu_used": False,
                "historical_costs_verified": False,
            },
        )
        finish_ledger(destination, output_root, admission, "failure.json", clock, verified)
        raise
    finally:
        for timer, (signum, handler) in timers.items():
            signal.setitimer(timer, 0)
            signal.signal(signum, handler)
