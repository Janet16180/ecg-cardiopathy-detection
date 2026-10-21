"""Retry only LUDB runtime admission after a preserved pre-data source-roster failure."""

import resource
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np

from ecg_experiment import ssl_synthetic_ludb_run as original
from ecg_experiment.files import sha256_file
from ecg_experiment.paths import from_stored

CONFIG = "configs/ssl_synthetic_ludb_qualification_v2.json"
PROTOCOL = "docs/ssl-synthetic-localization-ludb-qualification-runtime-v2.md"
BASE_REQUIRED = (
    original.CONFIG, original.PROTOCOL, "ecg_experiment/ssl_synthetic_ludb_qualification.py",
    "ecg_experiment/ssl_synthetic_ludb_run.py", "ecg_experiment/ssl_synthetic_ludb_figures.py",
    "scripts/experiments/run_ssl_synthetic_ludb_qualification.py",
    "tests/test_ssl_synthetic_ludb_qualification.py", "ecg_experiment/intervals.py",
    "ecg_experiment/paths.py", "ecg_experiment/files.py", "pyproject.toml", "uv.lock",
    "ecg_experiment/__init__.py", "ecg_experiment/ssl_synthetic_lead_time_recovery.py",
    "ecg_experiment/ssl_synthetic_lead_time_diagnostic.py", "ecg_experiment/synthetic_audit_io.py",
    "ecg_experiment/ssl_synthetic_pretrained_saved_audit.py",
    "ecg_experiment/ssl_synthetic_paired_suppression.py",
    "ecg_experiment/ssl_synthetic_pretrained_generator.py",
    "ecg_experiment/ssl_synthetic_counterfactual_factorial_training.py",
    "ecg_experiment/ssl_synthetic_paired_background_training.py", "ecg_experiment/ssl_synthetic_model.py",
    "ecg_experiment/ssl_synthetic_model_v2.py", "ecg_experiment/ssl_synthetic_preparation.py",
    "ecg_experiment/ssl_synthetic_training.py", "ecg_experiment/ssl_synthetic_real_transfer.py",
    "ecg_experiment/gpu.py",
)
REQUIRED = (*BASE_REQUIRED, CONFIG, PROTOCOL, "ecg_experiment/ssl_synthetic_ludb_run_v2.py",
            "scripts/experiments/run_ssl_synthetic_ludb_qualification_v2.py",
            "tests/test_ssl_synthetic_ludb_qualification_runtime_v2.py")


def validate_semantics(base: dict[str, Any], successor: dict[str, Any]) -> None:
    """Admit only study-name and output-directory overrides to the complete original scientific config."""
    expected = {**base, "study": "ssl_synthetic_ludb_qualification_v2",
                "output_directory": "outputs/ssl_synthetic_localization/ludb_qualification_v2"}
    if successor != expected:
        raise ValueError("Runtime successor changes original scientific semantics")


def validate_roster(sources: dict[str, str]) -> None:
    """Reject missing frozen helpers explicitly before any output directory or input read."""
    missing = sorted(set(REQUIRED) - sources.keys())
    if missing:
        raise ValueError("Source preflight missing: " + ", ".join(missing))


def preflight(admission_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate complete source identities and unchanged scientific choices without accessing any dataset."""
    config = original.read_json(from_stored(CONFIG))
    admission = original.read_json(from_stored(admission_name))
    validate_semantics(original.read_json(from_stored(original.CONFIG)), config)
    sources = admission["source_sha256"]
    validate_roster(sources)
    declared = {original.CONFIG: admission["frozen_config_sha256"],
                original.PROTOCOL: admission["frozen_protocol_sha256"],
                CONFIG: admission["frozen_runtime_config_sha256"],
                PROTOCOL: admission["frozen_runtime_protocol_sha256"]}
    if any(sources[name] != digest for name, digest in declared.items()):
        raise ValueError("Original or runtime design identity declaration differs")
    for name, expected in sources.items():
        if sha256_file(from_stored(name)) != expected:
            raise ValueError("Source preflight identity differs: " + name)
    return config, admission



def validate_reconciliation(record: dict[str, Any], acquisition: dict[str, Any], failed: dict[str, Any],
                            failure: dict[str, Any], old_admission: dict[str, Any]) -> None:
    """Reconstruct the retained pre-data failure cost without changing its original null accounting."""
    missing = set(BASE_REQUIRED) - old_admission["source_sha256"].keys()
    if (missing != {"ecg_experiment/ssl_synthetic_pretrained_saved_audit.py"}
            or failure["error_type"] != "ValueError"
            or failure["error"] != "Source admission omits required scientific or runtime sources"
            or failure["scientific_result_valid"] is not False):
        raise ValueError("Retained failure must be exactly the original pre-data missing-helper guard")
    if (failed["prior_accounting_verified"] is not False or failed["prior_ledger_sha256"] is not None
            or any(failed[key] is not None for key in ["current_machine_cumulative_cpu_seconds",
                                                       "current_machine_cumulative_gpu_wall_seconds"])):
        raise ValueError("Original failed ledger null/unverified fields must remain preserved")
    if (failed["receipt"] != "failure.json"
            or failed["receipt_sha256"] != record["retained_failed_main_receipt"]["sha256"]
            or not np.isfinite(failed["cpu_seconds"]) or failed["cpu_seconds"] < 0
            or failed["gpu_wall_seconds"] != 0 or not all(failed["resource_gates"].values())):
        raise ValueError("Retained failed receipt or measured resource identity differs")
    if (acquisition["current_machine_stages"] != 22 or not all(acquisition["resource_gates"].values())
            or old_admission["prior_ledger"]["sha256"] != record["verified_acquisition_ledger"]["sha256"]):
        raise ValueError("Reconciliation must start at the verified acquisition22")
    if (record["current_machine_stages"] != 23
            or record["current_machine_cumulative_cpu_seconds"]
            != acquisition["current_machine_cumulative_cpu_seconds"] + failed["cpu_seconds"]
            or record["current_machine_cumulative_gpu_wall_seconds"]
            != acquisition["current_machine_cumulative_gpu_wall_seconds"]
            or record["failed_stage_cpu_seconds_counted_once"] != failed["cpu_seconds"]
            or record["failed_stage_wall_seconds"] != failed["wall_seconds"]
            or record["scientific_scores_computed"] is not False or record["decoded_LUDB_records"] != 0):
        raise ValueError("Derived prior23 must include the actual failed CPU exactly once, without scores")


def verify_reconciled_prior(root: Path, admission: dict[str, Any]) -> None:
    """Hash bound original receipts and verify their additive derivation inside measured runtime."""
    prior = admission["prior_ledger"]
    if (prior["path"] != "outputs/ssl_synthetic_localization/ludb_qualification_reconciliation_v1/"
            "reconciled_prior.json"
            or prior["sha256"] != "8ff30f48c3d92057be2c641dd3a168a2d009a9f23fc4206bd5ce24612fff2a24"
            or prior["current_machine_stages"] != 23):
        raise ValueError("Runtime amendment reconciliation identity differs")
    original.verify_prior(root, admission)
    record = original.read_json(original.confined(root, prior["path"]))
    names = ["verified_acquisition_ledger", "retained_failed_main_ledger", "retained_failed_main_receipt",
             "failed_main_admission"]
    inputs = []
    for name in names:
        binding = record[name]
        path = original.confined(root, binding["path"])
        if sha256_file(path) != binding["sha256"]:
            raise ValueError("Bound original reconciliation input changed: " + name)
        inputs.append(original.read_json(path))
    validate_reconciliation(record, *inputs)
    old_runtime = "ecg_experiment/ssl_synthetic_ludb_run.py"
    if inputs[3]["source_sha256"][old_runtime] != admission["source_sha256"][old_runtime]:
        raise ValueError("Frozen original failure guard source identity changed")

def numerical_stage(root: Path, destination: Path, config: dict[str, Any], admission: dict[str, Any],
                    admission_name: str, sources: dict[str, str], clock: dict[str, float],
                    environment: dict[str, Any], guard: Any) -> dict[str, Any]:
    """Reuse the complete unchanged C2 closure, parser-first execution, endpoints and fixed figures."""
    closure = original.predecessor_closure(root, destination, config)
    data = original.admit_assets(root, destination, admission)
    rows, cases = original.run_records(data, destination, guard)
    statistics = original.core.aggregate(rows)
    with (destination / "bootstrap_gains.npy").open("xb") as stream:
        np.save(stream, np.asarray(statistics.pop("bootstrap_gains")), allow_pickle=False)
    original.write_new(destination / "roles.json", {"parser_qualification": list(range(5, 201, 5)),
                       "measurement": [identifier for identifier in range(1, 201) if identifier % 5 != 0]})
    original.figures.point_figure(destination, statistics, rows)
    original.figures.overlay_figure(destination, cases)
    guard()
    original.reverify(data, admission, sources, lambda: guard(False))
    guard()
    return original.make_summary(destination, config, statistics, original.coverage_summary(rows), closure,
                                 sources, admission_name, environment, clock)


def terminal(root: Path, destination: Path, config: dict[str, Any], admission: dict[str, Any],
             clock: dict[str, float], receipt: str, verified: bool, retained: int | None) -> None:
    """Append this runtime once to the reconciled prior, preserving the original null ledger."""
    digest = sha256_file(destination / receipt)
    prior, limits = admission["prior_ledger"], config["resources"]["main"]
    if verified and sha256_file(original.confined(root, prior["path"])) != prior["sha256"]:
        raise ValueError("Reconciled accounting base changed")
    cpu, wall = time.process_time() - clock["cpu"], time.perf_counter() - clock["wall"]
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    size = retained + (destination / receipt).stat().st_size if retained is not None else None
    gates = {"cpu": cpu < limits["cpu_seconds"], "wall": wall < limits["wall_seconds"],
             "rss": rss < limits["rss_bytes"],
             "output": size is not None and size + 16384 < limits["new_output_bytes"]}
    original.write_new(destination / "prospective_ledger.json", {
        "stage": config["study"], "receipt": receipt, "receipt_sha256": digest,
        "cpu_seconds": cpu, "wall_seconds": wall, "gpu_wall_seconds": 0, "peak_rss_bytes": rss,
        "resource_gates": gates, "current_machine_stages": prior["current_machine_stages"] + 1
        if verified else None, "prior_ledger_sha256": prior["sha256"] if verified else None,
        "prior_accounting_verified": verified,
        "current_machine_cumulative_cpu_seconds": prior["current_machine_cumulative_cpu_seconds"] + cpu
        if verified else None,
        "current_machine_cumulative_gpu_wall_seconds": prior["current_machine_cumulative_gpu_wall_seconds"]
        if verified else None, "historical_costs_verified": False,
        "output_bytes_before_terminal_ledger": size,
        "measurement_endpoint": ("after guarded byte count and receipt hash "
                                 "before terminal ledger serialization"),
        "terminal_bookkeeping_reserve_seconds": 2, "terminal_bookkeeping_reserve_bytes": 16384})
    if not all(gates.values()):
        raise ValueError("Runtime-v2 resource ceiling failed")



def final_enumeration(destination: Path, result: dict[str, Any], timers: dict[int, Any]) -> int | None:
    """Retain a failure receipt even when a live deadline interrupts final output enumeration."""
    try:
        retained = original.directory_bytes(destination)
        result["retained_byte_count_exact"] = True
        return retained
    except Exception as error:
        result.update(status="FAIL", retained_byte_count_exact=False,
                      terminal_enumeration_error_type=type(error).__name__,
                      terminal_enumeration_error=str(error))
        return None
    finally:
        for timer, (signum, previous_handler) in timers.items():
            signal.setitimer(timer, 0)
            signal.signal(signum, previous_handler)

def run_study(root: Path, admission_name: str, clock: dict[str, float]) -> dict[str, Any]:
    """Execute the unchanged scientific study only after a complete no-data preflight."""
    config, admission = preflight(admission_name)
    destination = original.confined(root, config["output_directory"])
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "scores").mkdir()
    limits, timers, verified = config["resources"]["main"], {}, False
    result: dict[str, Any] = {"study": config["study"], "status": "started"}

    def guard(check_output: bool = True) -> None:
        if (time.process_time() - clock["cpu"] + 2 >= limits["cpu_seconds"]
                or time.perf_counter() - clock["wall"] + 2 >= limits["wall_seconds"]):
            raise TimeoutError("Runtime-v2 CPU/wall reserve reached")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 >= limits["rss_bytes"]:
            raise MemoryError("Runtime-v2 RSS limit reached")
        if check_output and original.directory_bytes(destination) + 1024**2 >= limits["new_output_bytes"]:
            raise ValueError("Runtime-v2 output reserve reached")

    def interrupted(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Runtime-v2 process timer: {signum}")

    try:
        environment = original.runtime_environment()
        original.setup_timers(limits, clock, interrupted, timers)
        sources = original.bind_sources(destination, admission, admission_name)
        verify_reconciled_prior(root, admission)
        verified = True
        result = numerical_stage(root, destination, config, admission, admission_name, sources, clock,
                                 environment, guard)
        result["status"] = "PASS"
    except Exception as error:
        result.update(status="FAIL", error_type=type(error).__name__, error=str(error),
                      scientific_result_valid=False)
    finally:
        retained = final_enumeration(destination, result, timers)
    receipt = "summary.json" if result["status"] == "PASS" else "failure.json"
    original.write_new(destination / receipt, result)
    terminal(root, destination, config, admission, clock, receipt, verified, retained)
    if result["status"] != "PASS":
        raise RuntimeError("Retained runtime-v2 qualification failure")
    return result
