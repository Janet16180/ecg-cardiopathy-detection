"""Execute one prospectively admitted, bounded public LUDB timing qualification."""

import json
import os
import platform
import resource
import signal
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn
import wfdb

from ecg_experiment import ROOT
from ecg_experiment import ssl_synthetic_ludb_figures as figures
from ecg_experiment import ssl_synthetic_ludb_qualification as core
from ecg_experiment.files import sha256_file
from ecg_experiment.paths import from_stored, to_stored

CONFIG = "configs/ssl_synthetic_ludb_qualification_v1.json"
PROTOCOL = "docs/ssl-synthetic-localization-ludb-qualification-v1.md"


def write_new(path: Path, value: Any) -> None:
    """Write strict JSON exclusively, preserving existing attempts."""
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def read_json(path: Path) -> dict[str, Any]:
    """Read admitted object metadata."""
    return json.loads(path.read_text())


def directory_bytes(path: Path) -> int:
    """Count retained regular output bytes without following raw-data symlinks."""
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file() and not item.is_symlink())


def confined(root: Path, name: str) -> Path:
    """Open only a lexical descendant of an admitted artifact root."""
    if Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("Expected relative admitted path")
    return root / name


def bind_sources(destination: Path, admission: dict[str, Any], admission_name: str) -> dict[str, str]:
    """Require and snapshot every declared committed source before opening any LUDB records."""
    sources = admission["source_sha256"]
    required = {CONFIG, PROTOCOL, "ecg_experiment/ssl_synthetic_ludb_qualification.py",
                "ecg_experiment/ssl_synthetic_ludb_run.py", "ecg_experiment/ssl_synthetic_ludb_figures.py",
                "scripts/experiments/run_ssl_synthetic_ludb_qualification.py",
                "tests/test_ssl_synthetic_ludb_qualification.py", "ecg_experiment/intervals.py",
                "ecg_experiment/paths.py", "ecg_experiment/files.py", "pyproject.toml", "uv.lock",
                "ecg_experiment/__init__.py", "ecg_experiment/ssl_synthetic_lead_time_recovery.py",
                "ecg_experiment/ssl_synthetic_lead_time_diagnostic.py",
                "ecg_experiment/synthetic_audit_io.py",
                "ecg_experiment/ssl_synthetic_pretrained_saved_audit.py",
                "ecg_experiment/ssl_synthetic_paired_suppression.py",
                "ecg_experiment/ssl_synthetic_pretrained_generator.py",
                "ecg_experiment/ssl_synthetic_counterfactual_factorial_training.py",
                "ecg_experiment/ssl_synthetic_paired_background_training.py",
                "ecg_experiment/ssl_synthetic_model.py", "ecg_experiment/ssl_synthetic_model_v2.py",
                "ecg_experiment/ssl_synthetic_preparation.py", "ecg_experiment/ssl_synthetic_training.py",
                "ecg_experiment/ssl_synthetic_real_transfer.py", "ecg_experiment/gpu.py"}
    if not required.issubset(sources):
        raise ValueError("Source admission omits required scientific or runtime sources")
    if sources[CONFIG] != admission["frozen_config_sha256"] or sources[PROTOCOL] != admission[
            "frozen_protocol_sha256"]:
        raise ValueError("Frozen scientific design bindings differ")
    for name, expected in sources.items():
        source = from_stored(name)
        if sha256_file(source) != expected:
            raise ValueError(f"Source identity differs:{name}")
        target = destination / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    target = destination / "admission_snapshot.json"
    target.write_bytes(from_stored(admission_name).read_bytes())
    return {**sources, admission_name: sha256_file(target)}


def predecessor_closure(root: Path, destination: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Reproduce every frozen C2 aggregate value exactly from admitted saved case rows.

    Parameters
    ----------
    root : Path
        Authorized local artifact root.
    destination : Path
        Fresh study output directory.
    config : dict
        Frozen scientific configuration.

    Returns
    -------
    dict
        Exact closure counts, six contrasts and predecessor identities.
    """
    from ecg_experiment import ssl_synthetic_lead_time_recovery as recovery
    from ecg_experiment.synthetic_audit_io import compare_values

    spec = config["predecessor"]
    parent = confined(root, spec["directory"])
    names = {"summary.json": spec["summary_sha256"],
             "independent_audit_v1/summary.json": spec["independent_summary_sha256"],
             "independent_audit_v1/prospective_ledger.json": spec["latest_ledger_sha256"]}
    for name, expected in names.items():
        if sha256_file(parent / name) != expected:
            raise ValueError("Frozen C2 receipt identity differs")
    saved = read_json(parent / "summary.json")
    if not read_json(parent / "independent_audit_v1/summary.json")["all_passed"]:
        raise ValueError("Frozen C2 independent audit failed")
    cases = parent / "cases.jsonl"
    if sha256_file(cases) != saved["output_sha256"]["cases.jsonl"]:
        raise ValueError("Frozen C2 saved rows identity differs")
    result = recovery.summarize([json.loads(line) for line in cases.read_text().splitlines()])
    checks = sum(compare_values(value, saved[key], key, tolerance=0) for key, value in result.items())
    closure = {"all_exact": True, "case_rows": 1536, "scalar_checks": checks,
               "six_contrasts": result["descriptive_contrasts"],
               "input_sha256": {**names, "cases.jsonl": sha256_file(cases)}}
    write_new(destination / "predecessor_closure.json", closure)
    return closure


def admit_assets(root: Path, destination: Path, admission: dict[str, Any]) -> Path:
    """Verify the complete2800-file release identity manifest without parsing diagnosis fields."""
    expected = {f"{identifier}.{extension}" for identifier in range(1, 201)
                for extension in ["dat", "hea", *core.LEADS]}
    if set(admission["asset_sha256"]) != expected:
        raise ValueError("Asset admission must contain exactly200x14release files")
    for key in ["acquisition_summary", "acquisition_ledger"]:
        binding = admission[key]
        if sha256_file(confined(root, binding["path"])) != binding["sha256"]:
            raise ValueError("Acquisition receipt identity differs")
    data = confined(root, admission["data_root"])
    for name, digest in admission["asset_sha256"].items():
        if sha256_file(data / name) != digest:
            raise ValueError(f"Release file identity differs:{name}")
    write_new(destination / "release_file_identities.json", admission["asset_sha256"])
    return data


def coverage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe every admitted lead envelope and wave count without treating tails as negative."""
    leads = [lead for row in rows for lead in row["leads"]]
    return {"records": len(rows), "lead_streams": len(leads), "omitted_records": 0, "omitted_leads": 0,
            "coverage_samples_quantiles": np.percentile([row["coverage_samples"] for row in leads],
                                                        [0, 25, 50, 75, 100]).tolist(),
            "first_onset_quantiles": np.percentile([row["start"] for row in leads],
                                                   [0, 25, 50, 75, 100]).tolist(),
            "last_offset_quantiles": np.percentile([row["end_exclusive"] - 1 for row in leads],
                                                   [0, 25, 50, 75, 100]).tolist(),
            "wave_counts": {name: sum(row["wave_counts"][name] for row in leads)
                            for name in ["P", "QRS", "T"]}}


def run_records(data: Path, destination: Path, guard: Any) -> tuple[list[dict[str, Any]], dict[int, Any]]:
    """Finish all40 parser-only subjects before scoring the160 fixed measurement subjects."""
    rows, cases = [], {}
    order = [identifier for identifier in range(1, 201) if identifier % 5 == 0]
    order += [identifier for identifier in range(1, 201) if identifier % 5 != 0]
    with (destination / "record_rows.jsonl").open("x") as stream:
        for identifier in order:
            guard()
            row, values, proxies = core.evaluate_record(data / str(identifier),
                                                        destination / "sanitized_headers" / str(identifier))
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            rows.append(row)
            if identifier in [5, 10, 15, 20]:
                cases[identifier] = (row, values)
            if proxies is not None:
                with (destination / "scores" / f"{identifier}.npy").open("xb") as handle:
                    np.save(handle, proxies, allow_pickle=False)
    return sorted(rows, key=lambda row: row["record_id"]), cases


def runtime_environment() -> dict[str, Any]:
    """Require hidden CUDA and single-threaded numerical libraries before scientific work."""
    keys = ["CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS"]
    if os.environ.get(keys[0]) != "" or any(os.environ.get(key) != "1" for key in keys[1:]):
        raise ValueError("Require hidden CUDA and one numerical thread")
    return {"environment": {key: os.environ.get(key) for key in keys}, "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            "wfdb": wfdb.__version__, "gpu_used": False, "training_updates": 0}


def terminal_ledger(root: Path, destination: Path, admission: dict[str, Any], clock: dict[str, float],
                    receipt: str, verified: bool, budgets: dict[str, Any]) -> None:
    """Charge the retained attempt once, including receipt hashing and retained-byte enumeration."""
    digest = sha256_file(destination / receipt)
    prior = admission["prior_ledger"]
    if verified and sha256_file(confined(root, prior["path"])) != prior["sha256"]:
        raise ValueError("Prior cost base identity changed")
    size = directory_bytes(destination)
    cpu, wall = time.process_time() - clock["cpu"], time.perf_counter() - clock["wall"]
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    limits = {"cpu": cpu < budgets["cpu_seconds"], "wall": wall < budgets["wall_seconds"],
              "rss": rss < budgets["rss_bytes"], "output": size + 16384 < budgets["new_output_bytes"]}
    write_new(destination / "prospective_ledger.json", {
        "stage": "ssl_synthetic_ludb_qualification_v1", "receipt": receipt, "receipt_sha256": digest,
        "cpu_seconds": cpu, "wall_seconds": wall, "gpu_wall_seconds": 0.0, "peak_rss_bytes": rss,
        "output_bytes_before_terminal_ledger": size, "resource_gates": limits,
        "prior_ledger_sha256": prior["sha256"] if verified else None,
        "prior_accounting_verified": verified,
        "current_machine_cumulative_cpu_seconds": prior["current_machine_cumulative_cpu_seconds"] + cpu
        if verified else None,
        "current_machine_cumulative_gpu_wall_seconds": prior["current_machine_cumulative_gpu_wall_seconds"]
        if verified else None,
        "historical_costs_verified": False,
        "measurement_endpoint": ("after receipt hashing and byte enumeration, "
                                 "before terminal ledger serialization"),
        "terminal_bookkeeping_reserve_seconds": 2, "terminal_bookkeeping_reserve_bytes": 16384})
    if not all(limits.values()):
        raise ValueError("Terminal resource ceiling exceeded")



def reverify(data: Path, admission: dict[str, Any], sources: dict[str, str], guard: Any) -> None:
    """Recheck source and raw-file identities after figures and numerical work finish."""
    for name, expected in admission["asset_sha256"].items():
        guard()
        if sha256_file(data / name) != expected:
            raise ValueError("Release inputs changed during execution")
    for name, expected in sources.items():
        if sha256_file(from_stored(name)) != expected:
            raise ValueError("Scientific source changed during execution")


def make_summary(destination: Path, config: dict[str, Any], statistics: dict[str, Any],
                 coverage: dict[str, Any], closure: dict[str, Any], sources: dict[str, str],
                 admission_name: str, environment: dict[str, Any], clock: dict[str, float]) -> dict[str, Any]:
    """Assemble complete output bindings and explicitly provisional decision flags."""
    return {"study": config["study"], "created_at_utc": datetime.now(UTC).isoformat(),
                  "output_relative": to_stored(ROOT / config["output_directory"]),
                  "statistics": statistics, "coverage": coverage, "predecessor_closure": closure,
                  "source_sha256": sources, "admission_sha256": sha256_file(from_stored(admission_name)),
                  "runtime": environment, "technical_passed_before_independent_audit": True,
                  "scientific_passed_before_independent_audit": statistics["scientific_passed"],
                  "independent_audit_required": True, "clinical_claim": False,
                  "parser_activity_scores": 8, "measurement_activity_scores": 1920,
                  "measurement_AP_values": 13440, "parser_AP_values": 0,
                  "cpu_seconds_before_summary": time.process_time() - clock["cpu"],
                  "wall_seconds_before_summary": time.perf_counter() - clock["wall"],
                  "output_sha256": {item.relative_to(destination).as_posix(): sha256_file(item)
                                    for item in destination.rglob("*")
                                    if item.is_file() and not item.is_symlink()}}



def setup_timers(budgets: dict[str, Any], clock: dict[str, float], interrupted: Any,
                 timers: dict[int, Any]) -> None:
    """Install process-level CPU and wall timers including elapsed import time."""
    for timer, signum, limit, current in [
        (signal.ITIMER_REAL, signal.SIGALRM, budgets["wall_seconds"], time.perf_counter() - clock["wall"]),
        (signal.ITIMER_PROF, signal.SIGPROF, budgets["cpu_seconds"], time.process_time() - clock["cpu"]),
    ]:
        if signal.getitimer(timer) != (0., 0.):
            raise ValueError("Require exclusive process timers")
        timers[timer] = (signum, signal.getsignal(signum))
        signal.signal(signum, interrupted)
        signal.setitimer(timer, max(.01, limit - current - 2))


def verify_prior(root: Path, admission: dict[str, Any]) -> None:
    """Verify the audited acquisition accounting base before any numerical continuation."""
    prior = admission["prior_ledger"]
    ledger = read_json(confined(root, prior["path"]))
    if sha256_file(confined(root, prior["path"])) != prior["sha256"] or any(
            ledger[key] != prior[key] for key in ["current_machine_cumulative_cpu_seconds",
                                                  "current_machine_cumulative_gpu_wall_seconds"]):
        raise ValueError("Prospective accounting base differs")

def run_study(root: Path, admission_name: str, start_clock: dict[str, float] | None = None) -> dict[str, Any]:
    """Run exactly the frozen public-wave timing study, retaining failures without overwriting outputs.

    Parameters
    ----------
    root : Path
        Artifact checkout with local output symlinks.
    admission_name : str
        Repository-relative committed identity-only manifest.
    start_clock : dict or None
        CPU and wall start times captured before heavy library imports.

    Returns
    -------
    dict
        Saved provisional results awaiting independent reconstruction.
    """
    clock = start_clock or {"cpu": time.process_time(), "wall": time.perf_counter()}
    config, admission = read_json(from_stored(CONFIG)), read_json(from_stored(admission_name))
    destination = confined(root, config["output_directory"])
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "scores").mkdir()
    budgets, timers, verified = config["resources"]["main"], {}, False

    def guard() -> None:
        if (time.process_time() - clock["cpu"] + 2 >= budgets["cpu_seconds"]
                or time.perf_counter() - clock["wall"] + 2 >= budgets["wall_seconds"]):
            raise TimeoutError("Main CPU/wall reserve reached")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 >= budgets["rss_bytes"]:
            raise MemoryError("Main RSS limit reached")
        if directory_bytes(destination) + 1024**2 >= budgets["new_output_bytes"]:
            raise ValueError("Main output receipt reserve reached")

    def interrupted(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Main process timer:{signum}")

    try:
        environment = runtime_environment()
        setup_timers(budgets, clock, interrupted, timers)
        sources = bind_sources(destination, admission, admission_name)
        verify_prior(root, admission)
        verified = True
        closure = predecessor_closure(root, destination, config)
        data = admit_assets(root, destination, admission)
        rows, cases = run_records(data, destination, guard)
        statistics = core.aggregate(rows)
        with (destination / "bootstrap_gains.npy").open("xb") as stream:
            np.save(stream, np.asarray(statistics.pop("bootstrap_gains")), allow_pickle=False)
        write_new(destination / "roles.json", {"parser_qualification": list(range(5, 201, 5)),
                  "measurement": [identifier for identifier in range(1, 201) if identifier % 5 != 0]})
        coverage = coverage_summary(rows)
        figures.point_figure(destination, statistics, rows)
        figures.overlay_figure(destination, cases)
        reverify(data, admission, sources, guard)
        guard()
        result = make_summary(destination, config, statistics, coverage, closure, sources,
                              admission_name, environment, clock)
        write_new(destination / "summary.json", result)
        terminal_ledger(root, destination, admission, clock, "summary.json", verified, budgets)
        return result
    except Exception as error:
        if not (destination / "failure.json").exists():
            write_new(destination / "failure.json", {"error_type": type(error).__name__, "error": str(error),
                      "cpu_seconds_before_failure": time.process_time() - clock["cpu"],
                      "wall_seconds_before_failure": time.perf_counter() - clock["wall"],
                      "scientific_result_valid": False})
        if not (destination / "prospective_ledger.json").exists():
            terminal_ledger(root, destination, admission, clock, "failure.json", verified, budgets)
        raise
    finally:
        for timer, (signum, previous) in timers.items():
            signal.setitimer(timer, 0)
            signal.signal(signum, previous)
