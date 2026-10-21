"""Inspect only the twelve public record95 annotation streams without waveform or AP computation."""

import json
import os
import resource
import signal
import time
from collections import Counter
from pathlib import Path
from typing import Any

import wfdb

from ecg_experiment.files import sha256_file
from ecg_experiment.paths import from_stored
from ecg_experiment.ssl_synthetic_ludb_raw_events import decode

CONFIG = "configs/ssl_synthetic_ludb_annotation_diagnosis_v1.json"
PROTOCOL = "docs/ssl-synthetic-localization-ludb-annotation-diagnosis-v1.md"
SOURCES = (CONFIG, PROTOCOL, "ecg_experiment/ssl_synthetic_ludb_annotation_diagnosis.py",
           "scripts/reports/diagnose_ssl_synthetic_ludb_annotations.py",
           "tests/test_ssl_synthetic_ludb_annotation_diagnosis.py",
           "ecg_experiment/ssl_synthetic_ludb_raw_events.py", "tests/test_ssl_synthetic_ludb_raw_events.py",
           "ecg_experiment/files.py", "ecg_experiment/paths.py", "ecg_experiment/__init__.py",
           "pyproject.toml", "uv.lock")


def write_new(path: Path, value: Any) -> None:
    """Write strict JSON exclusively without overwriting any retained attempt."""
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def grammar(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe every frozen triplet violation without sorting, repairing or dropping events."""
    bad = set()
    for index, event in enumerate(events):
        if not 0 <= event["sample"] < 5000 or event["symbol"] not in ["(", ")", "p", "N", "t"]:
            bad.add(index)
        if index and event["sample"] < events[index - 1]["sample"]:
            bad.update([index - 1, index])
    for index in range(0, len(events), 3):
        part = events[index:index + 3]
        if (len(part) != 3 or [event["symbol"] for event in part][::2] != ["(", ")"]
                or part[1]["symbol"] not in ["p", "N", "t"]
                or not part[0]["sample"] <= part[1]["sample"] <= part[2]["sample"]
                or part[0]["sample"] == part[2]["sample"]):
            bad.update(range(index, min(index + 3, len(events))))
    cross = [index for index in range(3, len(events) - 2, 3)
             if events[index]["sample"] < events[index - 1]["sample"]]
    for index in cross:
        bad.update([index - 1, index])
    counts = Counter((event["sample"], event["type_code"]) for event in events)
    return {"empty": not events, "event_count": len(events), "event_count_modulo_3": len(events) % 3,
            "symbol_counts": dict(Counter(event["symbol"] for event in events)),
            "type_code_counts": dict(Counter(str(event["type_code"]) for event in events)),
            "cross_wave_overlap_indices": cross,
            "decreasing_indices": [index for index in range(1, len(events))
                                   if events[index]["sample"] < events[index - 1]["sample"]],
            "out_of_bounds_indices": [index for index, event in enumerate(events)
                                      if not 0 <= event["sample"] < 5000],
            "duplicate_sample_code_counts": [{"sample": key[0], "type_code": key[1], "count": value}
                                             for key, value in counts.items() if value > 1],
            "unknown_code_indices": [index for index, event in enumerate(events)
                                     if event["symbol"].startswith("unknown_code_")],
            "all_offending_indices": sorted(bad),
            "neighborhoods_first20": [events[max(0, index - 3):min(len(events), index + 4)]
                                      for index in sorted(bad)[:20]],
            "count_issue_first6": events[:6] if len(events) % 3 else [],
            "count_issue_last6": events[-6:] if len(events) % 3 else []}


def library_events(record: Path, extension: str) -> list[dict[str, Any]]:
    """Read WFDB only beside a sanitized structural header; retain AUX lengths without text."""
    annotation = wfdb.rdann(str(record), extension, return_label_elements=["symbol", "label_store"])
    return [{"event_index": index, "sample": int(sample), "symbol": str(annotation.symbol[index]),
             "type_code": int(annotation.label_store[index]), "subtype": int(annotation.subtype[index]),
             "channel": int(annotation.chan[index]), "num": int(annotation.num[index]),
             "aux_length": len(annotation.aux_note[index] or "")}
            for index, sample in enumerate(annotation.sample)]


def agreement(raw: dict[str, Any], library: list[dict[str, Any]]) -> bool:
    """Require all numeric fields and all known wave symbols to agree, flagging unmapped symbols."""
    fields = ["sample", "type_code", "subtype", "channel", "num", "aux_length"]
    return (not raw["errors"] and len(raw["events"]) == len(library)
            and all(all(first[key] == second[key] for key in fields)
                    and (first["symbol"].startswith("unknown_code_") or first["symbol"] == second["symbol"])
                    for first, second in zip(raw["events"], library, strict=True)))


def run(admission_name: str, clock: dict[str, float]) -> dict[str, Any]:
    """Execute exactly one bounded annotation-only diagnosis after prospective source admission."""
    config = json.loads(from_stored(CONFIG).read_text())
    admission = json.loads(from_stored(admission_name).read_text())
    destination = from_stored(config["output_directory"])
    destination.mkdir(parents=True, exist_ok=False)
    limits, prior, verified = config["resources"], config["prior_ledger"], False
    summary: dict[str, Any] = {"study": config["study"], "status": "started", "streams": {},
                               "waveform_reads": 0, "AP_computations": 0, "measurement_records_read": 0,
                               "aux_text_emitted": False, "gpu_used": False}
    previous_handlers = {}

    def guard() -> None:
        if (time.process_time() - clock["cpu"] + 2 >= limits["cpu_seconds"]
                or time.perf_counter() - clock["wall"] + 2 >= limits["wall_seconds"]):
            raise TimeoutError("Annotation diagnosis reached its receipt reserve")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 >= limits["rss_bytes"]:
            raise MemoryError("Annotation diagnosis RSS ceiling")

    def interrupted(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Annotation diagnosis process timer {signum}")

    try:
        setup_runtime(limits, clock, interrupted, previous_handlers)
        snapshot(destination, admission, admission_name)
        ledger = json.loads(from_stored(prior["path"]).read_text())
        if sha256_file(from_stored(prior["path"])) != prior["sha256"] or any(
                ledger[key] != prior[key] for key in ["current_machine_stages",
                "current_machine_cumulative_cpu_seconds", "current_machine_cumulative_gpu_wall_seconds"]):
            raise ValueError("Actual failed24 accounting identity differs")
        verified = True
        failure = config["failed_receipt"]
        if sha256_file(from_stored(failure["path"])) != failure["sha256"]:
            raise ValueError("Actual failed qualification receipt changed")
        inspect_record(config, destination, summary, guard)
        summary.update(status="PASS", all_12_dual_decoders_agree=all(
            row["dual_decoder_agreement"] for row in summary["streams"].values()),
            wfdb_version=wfdb.__version__)
        guard()
    except Exception as error:
        summary.update(status="FAIL", error_type=type(error).__name__, error=str(error))
    finish(destination, config, summary, clock, verified, previous_handlers)
    return summary



def inspect_record(config: dict[str, Any], destination: Path, summary: dict[str, Any], guard: Any) -> None:
    """Inspect only admitted record95 streams through a sanitized structural-only header."""
    source = from_stored(config["logical_data_root"])
    for name, expected in config["asset_sha256"].items():
        if (not os.path.samefile(source / name, from_stored(config["actual_data_target"]) / name)
                or sha256_file(source / name) != expected):
            raise ValueError("Admitted record95 alias or source identity differs")
    scratch = destination / "sanitized"
    scratch.mkdir()
    with (source / "95.hea").open() as stream:
        header = [stream.readline().rstrip("\n") for _ in range(13)]
    if any(not line or line.startswith("#") for line in header):
        raise ValueError("Structural thirteen-line header is invalid")
    (scratch / "95.hea").write_text("\n".join(header) + "\n")
    for extension in config["annotation_extensions"]:
        guard()
        name = "95." + extension
        (scratch / name).symlink_to(source / name)
        raw = decode((source / name).read_bytes())
        try:
            library = library_events(scratch / "95", extension)
            row = {"raw": raw, "wfdb_events": library, "dual_decoder_agreement": agreement(raw, library),
                   "raw_grammar": grammar(raw["events"]), "wfdb_grammar": grammar(library)}
        except Exception as error:
            row = {"raw": raw, "dual_decoder_agreement": False,
                   "wfdb_error_type": type(error).__name__, "raw_grammar": grammar(raw["events"])}
        write_new(destination / (extension + "_events.json"), row)
        summary["streams"][extension] = {"dual_decoder_agreement": row["dual_decoder_agreement"],
            "event_count": len(raw["events"]), "raw_grammar": row["raw_grammar"]}

    for name, expected in config["asset_sha256"].items():
        if sha256_file(source / name) != expected:
            raise ValueError("Record95 sources changed during diagnosis")


def setup_runtime(limits: dict[str, Any], clock: dict[str, float], interrupted: Any,
                  previous_handlers: dict[int, Any]) -> None:
    """Require hidden CUDA and bound the actual single-threaded diagnostic process."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or any(os.environ.get(key) != "1" for key in [
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]):
        raise ValueError("Require hidden CUDA and one numerical thread")
    for timer, signum, seconds in [(signal.ITIMER_REAL, signal.SIGALRM, limits["wall_seconds"]),
                                   (signal.ITIMER_PROF, signal.SIGPROF, limits["cpu_seconds"])]:
        previous_handlers[timer] = (signum, signal.getsignal(signum))
        signal.signal(signum, interrupted)
        elapsed = time.perf_counter() - clock["wall"] if timer == signal.ITIMER_REAL else (
            time.process_time() - clock["cpu"])
        signal.setitimer(timer, max(.01, seconds - elapsed - 2))

def snapshot(destination: Path, admission: dict[str, Any], name: str) -> None:
    """Require the complete exported source roster and preserve exact bytes before admitted reads."""
    sources = admission["source_sha256"]
    if not set(SOURCES).issubset(sources):
        raise ValueError("Annotation diagnosis source roster incomplete")
    for path, expected in sources.items():
        if sha256_file(from_stored(path)) != expected:
            raise ValueError("Annotation diagnosis source identity changed")
        target = destination / "source_snapshot" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(from_stored(path).read_bytes())
    runtime = admission["wfdb_runtime"]
    if runtime["version"] != wfdb.__version__:
        raise ValueError("Installed WFDB version differs")
    for label, module in [("annotation", wfdb.io.annotation), ("record", wfdb.io.record)]:
        path = Path(module.__file__)
        if sha256_file(path) != runtime[label + "_source_sha256"]:
            raise ValueError("Installed WFDB reader source identity differs")
        (destination / ("wfdb_" + label + "_source.py")).write_bytes(path.read_bytes())
    (destination / "admission_snapshot.json").write_bytes(from_stored(name).read_bytes())


def finish(destination: Path, config: dict[str, Any], summary: dict[str, Any], clock: dict[str, float],
           verified: bool, timers: dict[int, Any]) -> None:
    """Keep guards live through receipt hashing and enumeration, retaining a bounded failure fallback."""
    prior, limits = config["prior_ledger"], config["resources"]
    receipt, size = "summary.json", None
    summary.update(cpu_seconds=time.process_time() - clock["cpu"],
                   wall_seconds=time.perf_counter() - clock["wall"],
                   peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                   decoder_agreement_scope=("all numeric fields and known wave symbols; "
                                            "unmapped labels retained"))
    try:
        write_new(destination / receipt, summary)
        digest = sha256_file(destination / receipt)
        size = sum(path.stat().st_size for path in destination.rglob("*")
                   if path.is_file() and not path.is_symlink())
    except Exception as error:
        summary.update(status="FAIL", terminal_error_type=type(error).__name__, terminal_bytes_exact=False)
        receipt = "finalization_failure.json"
    finally:
        for timer, (signum, handler) in timers.items():
            signal.setitimer(timer, 0)
            signal.signal(signum, handler)
    if receipt == "finalization_failure.json":
        write_new(destination / receipt, summary)
        digest = sha256_file(destination / receipt)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    cpu, wall = time.process_time() - clock["cpu"], time.perf_counter() - clock["wall"]
    gates = {"cpu": cpu < limits["cpu_seconds"], "wall": wall < limits["wall_seconds"],
             "rss": rss < limits["rss_bytes"],
             "output": size is not None and size + 16384 < limits["new_output_bytes"]}
    passed = summary["status"] == "PASS" and all(gates.values()) and summary["all_12_dual_decoders_agree"]
    write_new(destination / "prospective_ledger.json", {"stage": config["study"], "receipt": receipt,
        "receipt_sha256": digest, "cpu_seconds": cpu, "wall_seconds": wall, "gpu_wall_seconds": 0,
        "peak_rss_bytes": rss,
        "resource_gates": gates, "technical_passed": passed,
        "current_machine_stages": 25 if verified else None,
        "prior_ledger_sha256": prior["sha256"] if verified else None,
        "current_machine_cumulative_cpu_seconds": prior["current_machine_cumulative_cpu_seconds"] + cpu
        if verified else None,
        "current_machine_cumulative_gpu_wall_seconds": prior["current_machine_cumulative_gpu_wall_seconds"]
        if verified else None, "prior_accounting_verified": verified, "historical_costs_verified": False,
        "output_bytes_before_terminal_ledger": size,
        "measurement_endpoint": "after receipt hash and byte count before terminal ledger serialization",
        "terminal_bookkeeping_reserve_seconds": 2, "terminal_bookkeeping_reserve_bytes": 16384})
    if not passed:
        raise RuntimeError("Retained annotation-only diagnostic technical failure")
