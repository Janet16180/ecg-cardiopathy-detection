"""Resume the final opaque LUDB suffix with a separately bounded socket-idle amendment."""

import json
import resource
import signal
import time
import urllib.request
from pathlib import Path
from typing import Any

from ecg_experiment import ssl_synthetic_ludb_acquisition_v2 as transport
from ecg_experiment.files import sha256_file
from ecg_experiment.paths import from_stored, to_stored

CONFIG = "configs/ssl_synthetic_ludb_acquisition_v3.json"
PROTOCOL = "docs/ssl-synthetic-localization-ludb-acquisition-v3.md"


def snapshot(root: Path, config: dict[str, Any], admission: dict[str, Any], name: str) -> dict[str, str]:
    """Bind the new transport amendment and every reused frozen helper before network access."""
    required = {CONFIG, PROTOCOL, "ecg_experiment/ssl_synthetic_ludb_acquisition_v3.py",
                "scripts/data/acquire_ssl_synthetic_ludb_v3.py",
                "tests/test_ssl_synthetic_ludb_acquisition_v3.py",
                "ecg_experiment/ssl_synthetic_ludb_acquisition_v2.py",
                "ecg_experiment/ssl_synthetic_ludb_acquisition.py", "ecg_experiment/files.py",
                "ecg_experiment/paths.py", "ecg_experiment/__init__.py", "pyproject.toml", "uv.lock",
                config["unchanged_scientific_protocol"], config["unchanged_scientific_config"]}
    sources = admission["source_sha256"]
    if not required.issubset(sources):
        raise ValueError("Incomplete successor transport source roster")
    if (sources[CONFIG] != admission["frozen_config_sha256"]
            or sources[PROTOCOL] != admission["frozen_protocol_sha256"]):
        raise ValueError("Frozen transport amendment identity differs")
    for relative, expected in sources.items():
        source = from_stored(relative)
        if sha256_file(source) != expected:
            raise ValueError("Admitted successor source changed")
        target = root / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(source.read_bytes())
    with (root / "admission_snapshot.json").open("xb") as stream:
        stream.write(from_stored(name).read_bytes())
    return {**sources, name: sha256_file(root / "admission_snapshot.json")}


def verify_lineage(config: dict[str, Any], admission: dict[str, Any]) -> dict[str, Any]:
    """Require retained failure19, narrow audit20 and socket-idle failure21 charged exactly once."""
    prior = admission["prior_ledger"]
    literal = config["source_predecessor_ledger"]
    if prior != literal or sha256_file(from_stored(prior["path"])) != prior["sha256"]:
        raise ValueError("Successor must directly append the actual stage21 failure")
    latest = json.loads(from_stored(prior["path"]).read_text())
    for key in ["current_machine_stages", "current_machine_cumulative_cpu_seconds",
                "current_machine_cumulative_gpu_wall_seconds"]:
        if latest[key] != prior[key]:
            raise ValueError("Stage21 literal cumulative accounting differs")
    bindings = config["preserved_bindings"]
    for name, expected in bindings.items():
        if sha256_file(from_stored(name)) != expected:
            raise ValueError("Preserved stage19/20 chain identity differs")
    ledgers = {json.loads(from_stored(name).read_text())["current_machine_stages"]:
               json.loads(from_stored(name).read_text()) for name in bindings
               if name.endswith("prospective_ledger.json")}
    if set(ledgers) != {19, 20}:
        raise ValueError("Preserved transport chain requires exact stages19 and20")
    check_additive(ledgers[19], ledgers[20], config["required_v1_failure_ledger_sha256"])
    check_additive(ledgers[20], latest, config["required_narrow_failure_audit_ledger_sha256"])
    if (latest["current_machine_stages"] != 21 or not all(latest["resource_gates"].values())
            or latest["sha256"] != config["source_summary_sha256"]):
        raise ValueError("Stage21 failure must retain successful resource accounting")
    audits = [json.loads(from_stored(name).read_text()) for name in bindings
              if name.endswith("independent_failure_audit_v1/summary.json")]
    if len(audits) != 1 or not audits[0]["all_passed"] or audits[0]["mismatch_count"] != 0:
        raise ValueError("Preserved narrow failure audit must pass without mismatches")
    return latest


def check_additive(previous: dict[str, Any], current: dict[str, Any], digest: str) -> None:
    """Require an adjacent zero-GPU transport stage with exact cumulative CPU arithmetic."""
    if (current["current_machine_stages"] != previous["current_machine_stages"] + 1
            or current["prior_ledger_sha256"] != digest or current["gpu_wall_seconds"] != 0
            or current["current_machine_cumulative_cpu_seconds"]
            != previous["current_machine_cumulative_cpu_seconds"] + current["cpu_seconds"]
            or current["current_machine_cumulative_gpu_wall_seconds"]
            != previous["current_machine_cumulative_gpu_wall_seconds"]):
        raise ValueError("Preserved transport costs must remain adjacent and additive exactly once")


def verify_inputs(config: dict[str, Any]) -> dict[str, str]:
    """Verify opaque v2 partial bytes and unchanged qualification design without opening ECG content."""
    old = from_stored(config["source_directory"])
    bindings = {"summary.json": config["source_summary_sha256"],
                "prospective_ledger.json": config["source_predecessor_ledger"]["sha256"],
                config["resume"]["prefix_file"]: config["resume"]["prefix_sha256"],
                config["checksum_manifest"]["file"]: config["checksum_manifest"]["sha256"]}
    for name, expected in bindings.items():
        if sha256_file(old / name) != expected:
            raise ValueError("Preserved socket-timeout attempt identity differs")
    for group, filename in [("resume", "prefix_file"), ("checksum_manifest", "file")]:
        size_key = "prefix_bytes" if group == "resume" else "bytes"
        if (old / config[group][filename]).stat().st_size != config[group][size_key]:
            raise ValueError("Preserved opaque source length differs")
    failure = json.loads((old / "summary.json").read_text())
    if (failure["status"] != "FAIL" or failure["error_type"] != "TimeoutError"
            or failure["response"]["status"] != 206 or failure["received_payload_bytes"] != 4325376):
        raise ValueError("Stage21 is not the admitted idle-timeout transport failure")
    for name in ["unchanged_scientific_protocol", "unchanged_scientific_config"]:
        if sha256_file(from_stored(config[name])) != config[name + "_sha256"]:
            raise ValueError("Scientific qualification protocol changed")
    return bindings


def copy_inputs(root: Path, config: dict[str, Any], bindings: dict[str, str]) -> None:
    """Copy only verified opaque predecessor bytes into a fresh successor directory."""
    old = from_stored(config["source_directory"])
    for name in bindings:
        target_name = "v2_" + name if name in ["summary.json", "prospective_ledger.json"] else name
        with (root / target_name).open("xb") as stream:
            stream.write((old / name).read_bytes())


def storage_bound(root: Path, config: dict[str, Any]) -> dict[str, int]:
    """Prove a conservative retained-byte ceiling once, without repeated directory scans."""
    base = transport.directory_bytes(root)
    remaining = config["resume"]["expected_complete_bytes"] - config["resume"]["prefix_bytes"]
    bound = base + remaining + config["resources"]["selective_extract_bytes"] + 1024**2
    if bound >= config["resources"]["new_output_bytes"]:
        raise ValueError("Conservative retained-byte bound exceeds successor budget")
    return {"base_bytes": base, "remaining_archive_bytes": remaining,
            "maximum_selective_extract_bytes": config["resources"]["selective_extract_bytes"],
            "receipt_reserve_bytes": 1024**2, "conservative_final_bytes": bound}


def receive(root: Path, config: dict[str, Any], summary: dict[str, Any], guard: Any) -> Path:
    """Append one exact accepted suffix with the prospective180-second socket idle timeout.

    Parameters
    ----------
    root : Path
        Exclusive output directory containing the copied opaque prefix.
    config : dict
        Frozen range, timeout, source and resource declarations.
    summary : dict
        Mutable transport counters retained on failure.
    guard : callable
        Live CPU, wall and RSS guard without directory scans.

    Returns
    -------
    Path
        Complete byte-counted release archive, retained locally.
    """
    resume = config["resume"]
    request = urllib.request.Request(config["request_url"], headers={
        "User-Agent": "ECG-local-research/1.0", "Range": resume["range_header"],
        "If-Range": '"' + resume["if_range_etag"] + '"'})
    summary["requests"] = 1
    partial = root / resume["prefix_file"]
    opener = urllib.request.build_opener(transport.NoRedirect())
    with opener.open(request, timeout=config["socket_timeout_seconds"]) as response:
        remaining = transport.verify_response(response, config)
        summary["response"] = {"status": response.status, "final_url": response.url,
                               "content_range": response.headers.get("Content-Range"),
                               "content_length": response.headers.get("Content-Length"),
                               "etag": response.headers.get("ETag")}
        with partial.open("ab") as stream:
            while True:
                guard()
                chunk = response.read(min(65536, remaining - summary["received_payload_bytes"] + 1))
                if not chunk:
                    break
                summary["received_payload_bytes"] += len(chunk)
                if (summary["received_payload_bytes"] > remaining or summary["received_payload_bytes"]
                        > config["resources"]["received_payload_bytes"]):
                    raise ValueError("Suffix exceeds exact admitted byte count")
                stream.write(chunk)
    if summary["received_payload_bytes"] != remaining or partial.stat().st_size != resume[
            "expected_complete_bytes"]:
        raise ValueError("Suffix or completed archive is shorter than exact admission")
    archive = partial.rename(root / "release.zip")
    summary["archive"] = {"bytes": archive.stat().st_size, "sha256": sha256_file(archive)}
    return archive


def reverify(config: dict[str, Any], old: dict[str, str], sources: dict[str, str]) -> None:
    """Require predecessor and source bytes to remain unchanged after attempted transport."""
    bindings = {**config["preserved_bindings"], **sources}
    for name, expected in bindings.items():
        if sha256_file(from_stored(name)) != expected:
            raise ValueError("Preserved chain or source changed during successor")
    for name, expected in old.items():
        if sha256_file(from_stored(config["source_directory"]) / name) != expected:
            raise ValueError("Preserved socket-timeout input changed during successor")


def finish(root: Path, config: dict[str, Any], admission: dict[str, Any], summary: dict[str, Any],
           clock: dict[str, float], verified: bool, retained_bytes: int | None) -> None:
    """Charge measured work after one guarded byte enumeration and a bounded bookkeeping tail."""
    limits = config["resources"]
    summary.update(cpu_seconds=time.process_time() - clock["cpu"],
                   wall_seconds=time.monotonic() - clock["wall"],
                   peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                   output_bytes_before_receipts=retained_bytes)
    gates = {"received_payload": summary["received_payload_bytes"] <= limits["received_payload_bytes"],
             "cpu": summary["cpu_seconds"] < limits["cpu_seconds"],
             "wall": summary["wall_seconds"] < limits["wall_seconds"],
             "rss": summary["peak_rss_bytes"] < limits["rss_bytes"],
             "output": retained_bytes is not None
             and retained_bytes + 1024**2 < limits["new_output_bytes"]}
    summary["resource_gates"] = gates
    if not all(gates.values()):
        summary["status"] = "FAIL"
    transport.write_new(root / "summary.json", summary)
    digest = sha256_file(root / "summary.json")
    if retained_bytes is not None:
        retained_bytes += (root / "summary.json").stat().st_size
    cpu, wall = time.process_time() - clock["cpu"], time.monotonic() - clock["wall"]
    gates.update(cpu=cpu < limits["cpu_seconds"], wall=wall < limits["wall_seconds"])
    prior = admission["prior_ledger"]
    ledger = {"receipt": to_stored(root / "summary.json"), "sha256": digest, "cpu_seconds": cpu,
              "wall_seconds": wall, "gpu_wall_seconds": 0, "resource_gates": gates,
              "current_machine_stages": prior["current_machine_stages"] + 1 if verified else None,
              "prior_ledger_sha256": prior["sha256"] if verified else None,
              "current_machine_cumulative_cpu_seconds": prior["current_machine_cumulative_cpu_seconds"] + cpu
              if verified else None,
              "current_machine_cumulative_gpu_wall_seconds": (
                  prior["current_machine_cumulative_gpu_wall_seconds"] if verified else None),
              "prior_accounting_verified": verified, "historical_costs_verified": False,
              "output_bytes_before_terminal_ledger": retained_bytes,
              "measurement_endpoint": "after one guarded byte count and receipt hash before terminal ledger",
              "terminal_bookkeeping_reserve_seconds": 2, "terminal_bookkeeping_reserve_bytes": 16384}
    transport.write_new(root / "prospective_ledger.json", ledger)
    if not all(gates.values()) or summary["status"] != "PASS":
        raise RuntimeError("Retained transport successor failure; no silent retry")


def acquire(admission_name: str, start_clock: dict[str, float] | None = None) -> dict[str, Any]:
    """Execute one separately frozen final-suffix attempt without changing scientific endpoints.

    Parameters
    ----------
    admission_name : str
        Repository-relative committed source and prior-ledger identity manifest.
    start_clock : dict or None
        CPU and monotonic wall clocks captured before importing this module.

    Returns
    -------
    dict
        Completed local acquisition receipt; retained failures are raised.
    """
    clock = start_clock or {"cpu": time.process_time(), "wall": time.monotonic()}
    config = json.loads(from_stored(CONFIG).read_text())
    admission = json.loads(from_stored(admission_name).read_text())
    root = from_stored(config["output_directory"])
    root.mkdir(parents=True, exist_ok=False)
    timers, sources, old = {}, {}, {}
    verified = False
    summary: dict[str, Any] = {"study": config["study"], "status": "started", "requests": 0,
                               "received_payload_bytes": 0, "gpu_seconds": 0,
                               "url": config["url"], "request_url": config["request_url"],
                               "socket_timeout_seconds": config["socket_timeout_seconds"]}

    def guard() -> None:
        limits = config["resources"]
        if (time.process_time() - clock["cpu"] + 2 >= limits["cpu_seconds"]
                or time.monotonic() - clock["wall"] + 2 >= limits["wall_seconds"]):
            raise TimeoutError("Transport successor reached reserved time boundary")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 >= limits["rss_bytes"]:
            raise MemoryError("Transport successor RSS cap reached")

    def interrupted(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Transport successor process timer:{signum}")

    try:
        transport.require_runtime()
        transport.setup_timers(config["resources"], clock, interrupted, timers)
        sources = snapshot(root, config, admission, admission_name)
        summary["source_sha256"] = sources
        verify_lineage(config, admission)
        verified = True
        old = verify_inputs(config)
        summary["source_input_sha256"] = old
        copy_inputs(root, config, old)
        summary["storage_bound"] = storage_bound(root, config)
        guard()
        archive = receive(root, config, summary, guard)
        summary.update(files=transport.extract(root, archive, config, guard), record_files=2800, records=200)
        reverify(config, old, sources)
        guard()
        summary["status"] = "PASS"
    except Exception as error:
        summary.update(status="FAIL", error_type=type(error).__name__, error=str(error))
    finally:
        retained_bytes = final_enumeration(root, summary, timers)
    finish(root, config, admission, summary, clock, verified, retained_bytes)
    return summary


def final_enumeration(root: Path, summary: dict[str, Any], timers: dict[int, Any]) -> int | None:
    """Retain a failure ledger even if a live deadline interrupts the single final byte enumeration."""
    try:
        retained = transport.directory_bytes(root)
        summary["retained_byte_count_exact"] = True
        return retained
    except Exception as error:
        summary.update(status="FAIL", terminal_enumeration_error_type=type(error).__name__,
                       terminal_enumeration_error=str(error), retained_byte_count_exact=False,
                       retained_output_upper_bound_bytes=summary.get("storage_bound", {}).get(
                           "conservative_final_bytes"))
        return None
    finally:
        for timer, (signum, previous) in timers.items():
            signal.setitimer(timer, 0)
            signal.signal(signum, previous)
