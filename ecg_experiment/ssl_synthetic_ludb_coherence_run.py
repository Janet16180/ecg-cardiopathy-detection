"""Execute a conditional, descriptive bridge from generated to public real activity synchrony."""

import json
import os
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np

from ecg_experiment import ssl_synthetic_generated_quality as quality
from ecg_experiment import ssl_synthetic_ludb_coherence_bridge as core
from ecg_experiment import ssl_synthetic_ludb_coherence_figures as figures
from ecg_experiment import ssl_synthetic_ludb_qualification as qualification
from ecg_experiment.files import sha256_file
from ecg_experiment.paths import from_stored
from ecg_experiment.ssl_synthetic_ludb_run import (
    directory_bytes,
    read_json,
    runtime_environment,
    setup_timers,
    write_new,
)

CONFIG = "configs/ssl_synthetic_ludb_coherence_bridge_v1.json"
PROTOCOL = "docs/ssl-synthetic-localization-ludb-coherence-bridge-v1.md"


def bind_sources(destination: Path, admission: dict[str, Any], name: str) -> dict[str, str]:
    """Require committed scientific sources and preserve exact immutable snapshot bytes."""
    sources = admission["source_sha256"]
    required = {CONFIG, PROTOCOL, "ecg_experiment/ssl_synthetic_ludb_coherence_bridge.py",
                "ecg_experiment/ssl_synthetic_ludb_coherence_figures.py",
                "ecg_experiment/ssl_synthetic_ludb_coherence_run.py",
                "scripts/reports/compare_ssl_synthetic_ludb_coherence.py",
                "tests/test_ssl_synthetic_ludb_coherence_bridge.py",
                "ecg_experiment/ssl_synthetic_generated_quality.py",
                "ecg_experiment/ssl_synthetic_ludb_qualification.py",
                "ecg_experiment/ssl_synthetic_ludb_run.py", "ecg_experiment/ssl_synthetic_ludb_figures.py",
                "ecg_experiment/intervals.py", "ecg_experiment/files.py", "ecg_experiment/paths.py",
                "ecg_experiment/__init__.py", "pyproject.toml", "uv.lock",
                "ecg_experiment/ssl_synthetic_ludb_acquisition_v3.py",
                "ecg_experiment/ssl_synthetic_ludb_acquisition_v2.py",
                "ecg_experiment/ssl_synthetic_ludb_acquisition.py"}
    if not required.issubset(sources):
        raise ValueError("Bridge source roster is incomplete")
    if (sources[CONFIG] != admission["frozen_config_sha256"]
            or sources[PROTOCOL] != admission["frozen_protocol_sha256"]):
        raise ValueError("Frozen bridge design identity differs")
    for relative, expected in sources.items():
        source = from_stored(relative)
        if sha256_file(source) != expected:
            raise ValueError("Bridge source identity changed")
        target = destination / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(source.read_bytes())
    with (destination / "admission_snapshot.json").open("xb") as stream:
        stream.write(from_stored(name).read_bytes())
    return {**sources, name: sha256_file(destination / "admission_snapshot.json")}


def qualified_predecessor(admission: dict[str, Any], destination: Path) -> dict[str, Any]:
    """Require audited QRS qualification PASS and reproduce its saved primary/bootstrap exactly."""
    spec = admission["qualification"]
    root = from_stored(spec["directory"])
    names = {"summary.json": spec["summary_sha256"], "prospective_ledger.json": spec["ledger_sha256"],
             "independent_audit_v1/summary.json": spec["audit_summary_sha256"],
             "independent_audit_v1/prospective_ledger.json": spec["audit_ledger_sha256"]}
    for name, expected in names.items():
        if sha256_file(root / name) != expected:
            raise ValueError("Qualified predecessor identity differs")
    main, audit = read_json(root / "summary.json"), read_json(root / "independent_audit_v1/summary.json")
    if not (main["technical_passed_before_independent_audit"]
            and main["scientific_passed_before_independent_audit"] and audit["all_passed"]
            and audit["mismatch_count"] == 0 and audit["scientific_passed"]):
        raise ValueError("Bridge requires the complete independently audited QRS qualification PASS")
    for name in ["prospective_ledger.json", "independent_audit_v1/prospective_ledger.json"]:
        if not all(read_json(root / name)["resource_gates"].values()):
            raise ValueError("Qualified predecessor resource gates must all pass")
    for name in ["record_rows.jsonl", "bootstrap_gains.npy"]:
        if sha256_file(root / name) != main["output_sha256"][name]:
            raise ValueError("Qualified predecessor numerical rows changed")
    rows = sorted([json.loads(line) for line in (root / "record_rows.jsonl").read_text().splitlines()],
                  key=lambda row: row["record_id"])
    statistics = qualification.aggregate(rows)
    bootstrap = np.asarray(statistics.pop("bootstrap_gains"))
    if statistics != main["statistics"] or not np.array_equal(
            bootstrap, np.load(root / "bootstrap_gains.npy", allow_pickle=False)):
        raise ValueError("Exact QRS predecessor primary and2000draw closure failed")
    result = {"all_exact": True, "subject_rows": 200, "measurement_subjects": 160,
              "bootstrap_draws": 2000, "statistics": statistics, "receipt_sha256": names}
    write_new(destination / "qualified_predecessor_closure.json", result)
    return result


def admit_assets(config: dict[str, Any], admission: dict[str, Any], destination: Path,
                 guard: Any) -> tuple[Path, Path]:
    """Bind only the400 real signal/header files and four original generator-bank artifacts."""
    expected = {f"{identifier}.{extension}" for identifier in range(1, 201) for extension in ["dat", "hea"]}
    if set(admission["asset_sha256"]) != expected:
        raise ValueError("Bridge requires all400 signal/header identities")
    logical, actual = from_stored(admission["logical_data_root"]), from_stored(admission["actual_data_root"])
    for name, digest in admission["asset_sha256"].items():
        guard()
        if not os.path.samefile(logical / name, actual / name) or sha256_file(actual / name) != digest:
            raise ValueError("Read-only-use data alias or real release identity differs")
    generated = from_stored(config["generated"]["directory"])
    for name, digest in config["generated"]["allowed_files"].items():
        if sha256_file(generated / name) != digest:
            raise ValueError("Original generated-bank identity differs")
    write_new(destination / "input_identities.json", {"real": admission["asset_sha256"],
              "generated": config["generated"]["allowed_files"], "logical_data_root": admission[
                  "logical_data_root"], "actual_data_root": admission["actual_data_root"]})
    return logical, generated


def new_masks(path: Path, count: int) -> np.memmap:
    """Allocate an exclusively admitted Boolean mask bank with fixed geometry."""
    if path.exists():
        raise FileExistsError(path)
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.bool_, shape=(count, 8, 5000))


def generated_closure(root: Path, destination: Path, masks: np.memmap,
                      guard: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reproduce all128 original transformation/coherence rows and the original seed interval exactly."""
    saved = [json.loads(line) for line in (root / "cases.jsonl").read_text().splitlines()]
    raw = np.load(root / "raw.npy", allow_pickle=False, mmap_mode="r")
    if raw.shape != (128, 8, 5000) or raw.dtype != np.float32 or len(saved) != 128:
        raise ValueError("Original generator-bank geometry differs")
    if not read_json(root / "independent_audit_v1/summary.json")["all_passed"]:
        raise ValueError("Original generator bank independent audit did not pass")
    rows = []
    with (destination / "generated_rows.jsonl").open("x") as stream:
        for index in range(128):
            guard()
            row, activity = core.prepare(raw[index], index, "generated")
            if (saved[index]["index"] != index or saved[index]["seed"] != row["generator_seed"]
                    or any(saved[index][key] != row[key] for key in ["standardization", "coherence"])):
                raise ValueError("Original generated coherence or transformation closure differs")
            masks[index] = activity
            masks.flush()
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
    primary = quality.primary_interval(rows)
    if primary != read_json(root / "summary.json")["primary"]:
        raise ValueError("Original generated primary interval closure differs")
    closure = {"all_exact": True, "records": 128, "transformations": 128, "coherence_rows": 128,
               "primary": primary}
    write_new(destination / "generated_predecessor_closure.json", closure)
    return rows, closure


def real_rows(root: Path, destination: Path, masks: np.memmap, guard: Any) -> list[dict[str, Any]]:
    """Read structural-only headers and physical samples without reading any annotation stream."""
    rows = []
    with (destination / "ludb_rows.jsonl").open("x") as stream:
        for identifier in range(1, 201):
            guard()
            values, metadata = qualification.read_signal(root / str(identifier),
                                                         destination / "sanitized_headers" / str(identifier))
            raw = values[list(core.INDEPENDENT_INDICES)].astype(np.float32)
            row, activity = core.prepare(raw, identifier - 1, "ludb")
            row["signal_metadata"] = metadata
            masks[identifier - 1] = activity
            masks.flush()
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
    return rows


def verify_prior(admission: dict[str, Any]) -> None:
    """Append only the actual qualified predecessor audit ledger, preserving all earlier costs."""
    prior, spec = admission["prior_ledger"], admission["qualification"]
    if (prior["path"] != spec["directory"] + "/independent_audit_v1/prospective_ledger.json"
            or prior["sha256"] != spec["audit_ledger_sha256"]):
        raise ValueError("Bridge accounting must append the actual qualification audit")
    if sha256_file(from_stored(prior["path"])) != prior["sha256"]:
        raise ValueError("Bridge accounting base identity differs")
    saved = read_json(from_stored(prior["path"]))
    for key in ["current_machine_stages", "current_machine_cumulative_cpu_seconds",
                "current_machine_cumulative_gpu_wall_seconds"]:
        if saved[key] != prior[key]:
            raise ValueError("Bridge accounting base totals differ")


def finish(destination: Path, config: dict[str, Any], admission: dict[str, Any], summary: dict[str, Any],
           clock: dict[str, float], verified: bool, retained: int | None) -> None:
    """Write bounded success/failure receipts and charge this stage once after all guarded reads."""
    limits = config["resources"]["main"]
    summary.update(cpu_seconds=time.process_time() - clock["cpu"],
                   wall_seconds=time.perf_counter() - clock["wall"],
                   peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                   output_bytes_before_receipts=retained)
    gates = {"cpu": summary["cpu_seconds"] < limits["cpu_seconds"],
             "wall": summary["wall_seconds"] < limits["wall_seconds"],
             "rss": summary["peak_rss_bytes"] < limits["rss_bytes"],
             "output": retained is not None and retained + 1024**2 < limits["new_output_bytes"]}
    summary["resource_gates"] = gates
    if not all(gates.values()):
        summary["status"] = "FAIL"
    summary["technical_passed_before_independent_audit"] = summary["status"] == "PASS" and all(gates.values())
    write_new(destination / "summary.json", summary)
    digest = sha256_file(destination / "summary.json")
    cpu, wall = time.process_time() - clock["cpu"], time.perf_counter() - clock["wall"]
    gates.update(cpu=cpu < limits["cpu_seconds"], wall=wall < limits["wall_seconds"])
    prior = admission["prior_ledger"]
    write_new(destination / "prospective_ledger.json", {
        "stage": config["study"], "receipt": "summary.json", "receipt_sha256": digest,
        "cpu_seconds": cpu, "wall_seconds": wall, "gpu_wall_seconds": 0, "resource_gates": gates,
        "prior_ledger_sha256": prior["sha256"] if verified else None,
        "prior_accounting_verified": verified,
        "current_machine_stages": prior["current_machine_stages"] + 1 if verified else None,
        "current_machine_cumulative_cpu_seconds": prior["current_machine_cumulative_cpu_seconds"] + cpu
        if verified else None,
        "current_machine_cumulative_gpu_wall_seconds": (prior["current_machine_cumulative_gpu_wall_seconds"]
                                                       if verified else None),
        "output_bytes_before_terminal_ledger": retained + (destination / "summary.json").stat().st_size
        if retained is not None else None,
        "historical_costs_verified": False,
        "measurement_endpoint": ("after receipt hash and guarded byte count "
                                 "before terminal ledger serialization"),
        "terminal_bookkeeping_reserve_seconds": 2, "terminal_bookkeeping_reserve_bytes": 16384})
    if not summary["technical_passed_before_independent_audit"] or not all(gates.values()):
        raise RuntimeError("Retained descriptive bridge technical failure")


def verify_inputs(config: dict[str, Any], admission: dict[str, Any], sources: dict[str, str],
                  guard: Any) -> None:
    """Rehash only admitted inputs and sources after numerical work without growing-tree scans."""
    for name, digest in admission["asset_sha256"].items():
        guard()
        if sha256_file(from_stored(admission["actual_data_root"]) / name) != digest:
            raise ValueError("Real input identity changed during bridge")
    for name, digest in config["generated"]["allowed_files"].items():
        if sha256_file(from_stored(config["generated"]["directory"]) / name) != digest:
            raise ValueError("Generated input identity changed during bridge")
    for name, digest in sources.items():
        if sha256_file(from_stored(name)) != digest:
            raise ValueError("Scientific source changed during bridge")


def output_bindings(destination: Path) -> dict[str, str]:
    """Hash every regular retained artifact before writing the final result receipt."""
    return {path.relative_to(destination).as_posix(): sha256_file(path) for path in destination.rglob("*")
            if path.is_file() and not path.is_symlink()}


def run_study(admission_name: str, start_clock: dict[str, float] | None = None) -> dict[str, Any]:
    """Execute only the conditionally admitted descriptive comparison with bounded local artifacts.

    Parameters
    ----------
    admission_name : str
        Committed repository-relative source/input/accounting admission manifest.
    start_clock : dict or None
        CPU and wall clocks captured before heavy source imports.

    Returns
    -------
    dict
        Complete descriptive results awaiting independent technical verification.
    """
    from ecg_experiment.ssl_synthetic_ludb_acquisition_v3 import final_enumeration

    clock = start_clock or {"cpu": time.process_time(), "wall": time.perf_counter()}
    config, admission = read_json(from_stored(CONFIG)), read_json(from_stored(admission_name))
    destination = from_stored(config["output_directory"])
    destination.mkdir(parents=True, exist_ok=False)
    limits, timers, verified = config["resources"]["main"], {}, False
    summary: dict[str, Any] = {"study": config["study"], "status": "started",
                               "independent_audit_required": True, "clinical_quality_claim": False,
                               "physiological_fidelity_claim": False}

    def guard(check_output: bool = False) -> None:
        if (time.process_time() - clock["cpu"] + 2 >= limits["cpu_seconds"]
                or time.perf_counter() - clock["wall"] + 2 >= limits["wall_seconds"]):
            raise TimeoutError("Bridge CPU/wall reserve reached")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 >= limits["rss_bytes"]:
            raise MemoryError("Bridge RSS cap reached")
        if check_output and directory_bytes(destination) + 1024**2 >= limits["new_output_bytes"]:
            raise ValueError("Bridge output receipt reserve reached")

    def interrupted(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Bridge process timer:{signum}")

    try:
        summary["runtime"] = runtime_environment()
        setup_timers(limits, clock, interrupted, timers)
        sources = bind_sources(destination, admission, admission_name)
        summary["source_sha256"] = sources
        verify_prior(admission)
        verified = True
        predecessor = qualified_predecessor(admission, destination)
        logical, generated_root = admit_assets(config, admission, destination, guard)
        generated_masks = new_masks(destination / "generated_activity_masks.npy", 128)
        real_masks = new_masks(destination / "ludb_activity_masks.npy", 200)
        generated, old = generated_closure(generated_root, destination, generated_masks, guard)
        guard(True)
        real = real_rows(logical, destination, real_masks, guard)
        guard(True)
        statistics, bootstrap = core.summarize(generated, real)
        with (destination / "bootstrap_cohort_means_and_contrasts.npy").open("xb") as stream:
            np.save(stream, bootstrap, allow_pickle=False)
        figures.comparison_figures(destination, {"generated": generated, "LUDB": real})
        figures.raster_figure(destination, generated_masks, real_masks)
        guard(True)
        verify_inputs(config, admission, sources, guard)
        guard(True)
        summary.update(status="PASS", statistics=statistics, qualified_predecessor_closure=predecessor,
                       generated_predecessor_closure=old, output_sha256=output_bindings(destination),
                       raw_inputs_copied=0, new_annotation_stream_reads=0, waveform_record_reads=200,
                       generator_forwards=0, optimizer_updates=0, gpu_used=False,
                       activity_mask_shape={"generated": [128, 8, 5000], "ludb": [200, 8, 5000]},
                       coherence_rows=328, technical_gates={"qualified_predecessor_passed": True,
                       "old128_exact_closure": True, "all328_records_retained": True,
                       "fixed_source_input_and_seed_bindings": True, "complete2000_draws": True,
                       "zero_diagnosis_fields_or_new_annotation_reads": True})
    except Exception as error:
        summary.update(status="FAIL", error_type=type(error).__name__, error=str(error))
    finally:
        retained = final_enumeration(destination, summary, timers)
    finish(destination, config, admission, summary, clock, verified, retained)
    return summary
