"""Prepare and verify a single external public LUDB suffix transfer with split resource clocks."""

import json
import re
import resource
import signal
import time
from pathlib import Path
from typing import Any

from ecg_experiment import ssl_synthetic_ludb_acquisition_v2 as transport
from ecg_experiment import ssl_synthetic_ludb_acquisition_v3 as previous
from ecg_experiment.files import sha256_file
from ecg_experiment.paths import from_stored, to_stored

CONFIG = "configs/ssl_synthetic_ludb_acquisition_v4.json"
PROTOCOL = "docs/ssl-synthetic-localization-ludb-acquisition-v4.md"


def elapsed_seconds(value: str) -> float:
    """Decode GNUtime verbose elapsed h:mm:ss or m:ss without assuming wall-clock continuity."""
    fields = value.split(":")
    if len(fields) not in [2, 3]:
        raise ValueError("GNUtime elapsed format must be m:ss or h:mm:ss")
    seconds = 0.
    for field in fields:
        seconds = seconds * 60 + float(field)
    if seconds < 0 or not seconds < float("inf"):
        raise ValueError("GNUtime elapsed duration must be finite and nonnegative")
    return seconds


def transfer_metrics(path: Path) -> dict[str, Any]:
    """Read authoritative child user/system/wall/exit/RSS fields from one GNUtime-v report."""
    fields = {}
    text = path.read_text()
    for line in text.splitlines():
        if ": " in line:
            key, value = line.strip().split(": ", 1)
            if key in fields:
                raise ValueError("Duplicate GNUtime verbose field")
            fields[key] = value
    user, system = float(fields["User time (seconds)"]), float(fields["System time (seconds)"])
    wall = elapsed_seconds(fields["Elapsed (wall clock) time (h:mm:ss or m:ss)"])
    rss = int(fields["Maximum resident set size (kbytes)"]) * 1024
    status = int(fields["Exit status"])
    if (not all(0 <= value < float("inf") for value in [user, system, wall])
            or rss < 0 or not 0 <= status <= 255):
        raise ValueError("Invalid GNUtime resource fields")
    return {"user_cpu_seconds": user, "system_cpu_seconds": system, "cpu_seconds": user + system,
            "wall_seconds": wall, "peak_rss_bytes": rss, "exit_status": status,
            "terminated_by_signal": bool(re.search(r"Command terminated by signal", text)),
            "measurement": "GNUtime verbose child clocks; printed precision retained"}


def response_headers(path: Path) -> dict[str, Any]:
    """Read the final HTTP block while permitting only a preceding proxy CONNECT tunnel block."""
    text = path.read_text(encoding="iso-8859-1").replace("\r\n", "\n")
    blocks = [block for block in text.split("\n\n") if block.strip()]
    parsed = []
    for block in blocks:
        lines = block.strip().splitlines()
        match = re.fullmatch(r"HTTP/[0-9.]+\s+([0-9]{3})(?:\s+(.*))?", lines[0])
        if match is None:
            raise ValueError("Invalid HTTP response status line")
        headers = {}
        for line in lines[1:]:
            key, value = line.split(":", 1)
            key = key.lower().strip()
            if key in headers:
                raise ValueError("Duplicate response header")
            headers[key] = value.strip()
        parsed.append({"status": int(match[1]), "reason": match[2] or "", "headers": headers})
    final = [block for block in parsed if block["status"] >= 200]
    if not final or any(
            block["status"] >= 200 and (block["status"] != 200
            or block["reason"].lower() != "connection established") for block in parsed[:-1]):
        raise ValueError("Unexpected multiple HTTP responses or redirect")
    return final[-1]


def verify_transfer(root: Path, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require successful bounded child execution, exact206 range/ETag and exact suffix bytes."""
    external, resume = config["curl"], config["resume"]
    metrics = transfer_metrics(root / external["resource_log"])
    response = response_headers(root / external["response_headers"])
    headers = response["headers"]
    remaining = resume["expected_complete_bytes"] - resume["prefix_bytes"]
    if metrics["exit_status"] != 0 or metrics["terminated_by_signal"]:
        raise ValueError("External curl did not exit successfully")
    if (response["status"] != 206 or headers.get("content-range") != resume["required_content_range"]
            or headers.get("etag") != '"' + resume["if_range_etag"] + '"'):
        raise ValueError("External suffix response identity or exact range differs")
    if headers.get("content-length") is not None and headers["content-length"] != str(remaining):
        raise ValueError("External suffix Content-Length differs")
    if (root / external["suffix_file"]).stat().st_size != remaining:
        raise ValueError("External suffix length differs from exact admission")
    limits = config["resources"]
    if (metrics["cpu_seconds"] >= limits["external_curl"]["cpu_seconds"]
            or metrics["wall_seconds"] >= limits["external_curl"]["wall_seconds"]
            or metrics["peak_rss_bytes"] >= limits["total"]["rss_bytes"]):
        raise ValueError("External suffix exceeded its separately metered resource budget")
    return metrics, response


def snapshot(root: Path, config: dict[str, Any], admission: dict[str, Any], name: str) -> dict[str, str]:
    """Validate the split-transport design, then reuse the frozen full source-snapshot implementation."""
    sources = admission["source_sha256"]
    required = {CONFIG, PROTOCOL, "ecg_experiment/ssl_synthetic_ludb_acquisition_v4.py",
                "scripts/data/acquire_ssl_synthetic_ludb_v4.py",
                "tests/test_ssl_synthetic_ludb_acquisition_v4.py"}
    if not required.issubset(sources) or sources[CONFIG] != admission["frozen_config_sha256"] or sources[
            PROTOCOL] != admission["frozen_protocol_sha256"]:
        raise ValueError("Frozen split-transport source admission differs")
    inherited = {**admission, "frozen_config_sha256": sources[previous.CONFIG],
                 "frozen_protocol_sha256": sources[previous.PROTOCOL]}
    return previous.snapshot(root, config, inherited, name)


def storage_bound(root: Path, config: dict[str, Any]) -> dict[str, int]:
    """Count the base once and bound both retained suffix copies plus admitted extraction."""
    base = transport.directory_bytes(root)
    remaining = config["resume"]["expected_complete_bytes"] - config["resume"]["prefix_bytes"]
    bound = base + 2 * remaining + config["resources"]["total"]["selective_extract_bytes"] + 1024**2
    if bound >= config["resources"]["total"]["new_output_bytes"]:
        raise ValueError("Split-transport retained-byte bound exceeds budget")
    return {"base_bytes": base, "two_remaining_suffix_copies_bytes": 2 * remaining,
            "maximum_extract_bytes": config["resources"]["total"]["selective_extract_bytes"],
            "receipt_reserve_bytes": 1024**2, "conservative_final_bytes": bound}


def phase_limits(config: dict[str, Any], preparation: dict[str, Any] | None = None) -> dict[str, Any]:
    """Apply the shared Python-phase budget after subtracting the recorded preparation cost."""
    limits, preparation = config["resources"], preparation or {"cpu_seconds": 0, "wall_seconds": 0}
    offline = limits["offline_prepare_and_finish_combined"]
    return {"cpu_seconds": offline["cpu_seconds"] - preparation["cpu_seconds"],
            "wall_seconds": offline["active_wall_seconds"] - preparation["wall_seconds"],
            "rss_bytes": limits["total"]["rss_bytes"]}


def prepare_data(admission_name: str, config: dict[str, Any], root: Path) -> dict[str, Any]:
    """Create a new local attempt and expose the exact one-command transfer for separate approval."""
    admission = json.loads(from_stored(admission_name).read_text())
    source = snapshot(root, config, admission, admission_name)
    previous.verify_lineage(config, admission)
    inputs = previous.verify_inputs(config)
    previous.copy_inputs(root, config, inputs)
    bound = storage_bound(root, config)
    for name in [config["curl"][key] for key in ["resource_log", "response_headers", "suffix_file"]]:
        if (root / name).exists():
            raise FileExistsError(root / name)
    receipt = {"status": "PASS", "study": config["study"], "source_sha256": source,
               "source_input_sha256": inputs, "admission_sha256": sha256_file(from_stored(admission_name)),
               "storage_bound": bound,
               "external_argv": [argument.format(output_directory=config["output_directory"])
                                 for argument in config["curl"]["argv"]],
               "approval_wait_measured": False, "transfer_executed": False}
    for binary in admission["executables"].values():
        if sha256_file(Path(binary["path"])) != binary["sha256"]:
            raise ValueError("Prospectively admitted system executable identity differs")
    receipt["external_command_argv"] = [admission["executables"]["time"]["path"], "-v", "-o",
        config["output_directory"] + "/" + config["curl"]["resource_log"],
        admission["executables"]["curl"]["path"], *receipt["external_argv"]]
    receipt["external_environment"] = {"LC_ALL": "C"}
    return receipt


def assemble(root: Path, config: dict[str, Any], guard: Any) -> Path:
    """Append only a verified complete suffix to the new local copied prefix; retain external bytes."""
    partial = root / config["resume"]["prefix_file"]
    if sha256_file(partial) != config["resume"]["prefix_sha256"]:
        raise ValueError("Copied prefix changed before offline assembly")
    suffix = root / config["curl"]["suffix_file"]
    with suffix.open("rb") as source, partial.open("ab") as target:
        transport.copy_member(source, target, suffix.stat().st_size, guard)
    if partial.stat().st_size != config["resume"]["expected_complete_bytes"]:
        raise ValueError("Offline assembled archive length differs")
    return partial.rename(root / "release.zip")


def disarm(timers: dict[int, Any]) -> None:
    """Restore process timers even when offline preparation or final enumeration fails."""
    for timer, (signum, previous_handler) in timers.items():
        signal.setitimer(timer, 0)
        signal.signal(signum, previous_handler)


def prepare(admission_name: str, clock: dict[str, float]) -> dict[str, Any]:
    """Measure offline preparation and retain an explicit state before any external transfer."""
    config = json.loads(from_stored(CONFIG).read_text())
    root = from_stored(config["output_directory"])
    root.mkdir(parents=True, exist_ok=False)
    timers = {}

    def interrupted(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Preparation process timer:{signum}")

    try:
        transport.require_runtime()
        transport.setup_timers(phase_limits(config), clock, interrupted, timers)
        receipt = prepare_data(admission_name, config, root)
    except Exception as error:
        receipt = {"status": "FAIL", "error_type": type(error).__name__, "error": str(error),
                   "transfer_executed": False}
    finally:
        disarm(timers)
    transport.write_new(root / "prepare.json", receipt)
    digest = sha256_file(root / "prepare.json")
    costs = {"cpu_seconds": time.process_time() - clock["cpu"],
             "wall_seconds": time.monotonic() - clock["wall"],
             "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
             "prepare_sha256": digest,
             "measurement_endpoint": "after preparation receipt hash before phase-cost serialization",
             "phase_cost_serialization_reserve_seconds": 2}
    transport.write_new(root / "prepare_costs.json", costs)
    offline = config["resources"]["offline_prepare_and_finish_combined"]
    if (receipt["status"] != "PASS" or costs["cpu_seconds"] + 2 >= offline["cpu_seconds"]
            or costs["wall_seconds"] + 2 >= offline["active_wall_seconds"]
            or costs["peak_rss_bytes"] >= config["resources"]["total"]["rss_bytes"]):
        raise RuntimeError("Retained split-transport preparation failure; do not start curl")
    return receipt


def completion(root: Path, config: dict[str, Any], admission: dict[str, Any], summary: dict[str, Any],
               preparation: dict[str, Any], child: dict[str, Any] | None, clock: dict[str, float],
               verified: bool, retained: int | None) -> None:
    """Append exactly once using three separately measured phases and disclosed excluded waiting."""
    offline = {"cpu_seconds": time.process_time() - clock["cpu"],
               "wall_seconds": time.monotonic() - clock["wall"],
               "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    resources, limits = config["resources"], config["resources"]["total"]
    python_cpu = preparation["cpu_seconds"] + offline["cpu_seconds"]
    python_wall = preparation["wall_seconds"] + offline["wall_seconds"]
    cpu = python_cpu + child["cpu_seconds"] if child is not None else None
    wall = python_wall + child["wall_seconds"] if child is not None else None
    rss = max(preparation["peak_rss_bytes"], offline["peak_rss_bytes"],
              child["peak_rss_bytes"] if child is not None else 0)
    gates = {"python_cpu": python_cpu < resources["offline_prepare_and_finish_combined"]["cpu_seconds"],
             "python_active_wall": python_wall < resources["offline_prepare_and_finish_combined"][
                 "active_wall_seconds"], "total_cpu": cpu is not None and cpu < limits["cpu_seconds"],
             "summed_active_wall": wall is not None and wall < limits["summed_active_wall_seconds"],
             "rss": rss < limits["rss_bytes"],
             "curl_cpu": (child is not None
                          and child["cpu_seconds"] < resources["external_curl"]["cpu_seconds"]),
             "curl_wall": (child is not None
                           and child["wall_seconds"] < resources["external_curl"]["wall_seconds"]),
             "output": retained is not None and retained + 1024**2 < limits["new_output_bytes"]}
    if not all(gates.values()):
        summary["status"] = "FAIL"
    summary.update(resource_gates=gates, phase_measurements={"prepare": preparation, "curl": child,
                   "finish_before_receipts": offline}, cpu_seconds=cpu, summed_active_wall_seconds=wall,
                   peak_rss_bytes=rss, approval_and_interphase_wait="excluded, unmeasured",
                   uninterrupted_whole_wall_claim=False)
    transport.write_new(root / "summary.json", summary)
    digest = sha256_file(root / "summary.json")
    finish_cpu = time.process_time() - clock["cpu"]
    finish_wall = time.monotonic() - clock["wall"]
    known_cpu = preparation["cpu_seconds"] + finish_cpu
    cpu = known_cpu + child["cpu_seconds"] if child is not None else None
    wall = preparation["wall_seconds"] + finish_wall + child["wall_seconds"] if child is not None else None
    gates.update(python_cpu=known_cpu < resources["offline_prepare_and_finish_combined"]["cpu_seconds"],
                 python_active_wall=preparation["wall_seconds"] + finish_wall
                 < resources["offline_prepare_and_finish_combined"]["active_wall_seconds"],
                 total_cpu=cpu is not None and cpu < limits["cpu_seconds"],
                 summed_active_wall=wall is not None and wall < limits["summed_active_wall_seconds"])
    prior = admission["prior_ledger"]
    transport.write_new(root / "prospective_ledger.json", {
        "receipt": to_stored(root / "summary.json"), "sha256": digest, "cpu_seconds": cpu,
        "wall_seconds": wall, "wall_measurement": "sum of active phases; approval waiting excluded",
        "gpu_wall_seconds": 0, "known_python_cpu_seconds": known_cpu,
        "finish_cpu_seconds": finish_cpu, "finish_wall_seconds": finish_wall,
        "resource_gates": gates, "current_machine_stages": 22 if verified else None,
        "prior_ledger_sha256": prior["sha256"] if verified else None,
        "current_machine_cumulative_cpu_seconds": prior["current_machine_cumulative_cpu_seconds"] + cpu
        if verified and cpu is not None else None,
        "known_current_machine_cumulative_cpu_seconds": (
            prior["current_machine_cumulative_cpu_seconds"] + known_cpu if verified else None),
        "current_machine_cumulative_gpu_wall_seconds": prior["current_machine_cumulative_gpu_wall_seconds"]
        if verified else None, "prior_accounting_verified": verified, "historical_costs_verified": False,
        "output_bytes_before_terminal_ledger": retained + (root / "summary.json").stat().st_size
        if retained is not None else None,
        "measurement_endpoint": "after receipt hash and single guarded byte count before terminal ledger",
        "terminal_bookkeeping_reserve_seconds": 2, "terminal_bookkeeping_reserve_bytes": 16384})
    if summary["status"] != "PASS" or not all(gates.values()):
        raise RuntimeError("Retained split-transport acquisition failure")


def finish(admission_name: str, clock: dict[str, float],
           transfer_not_started: bool = False) -> dict[str, Any]:
    """Verify the external evidence offline and extract only official checksum-matching release assets."""
    config = json.loads(from_stored(CONFIG).read_text())
    admission = json.loads(from_stored(admission_name).read_text())
    root = from_stored(config["output_directory"])
    if (root / "summary.json").exists() or (root / "prospective_ledger.json").exists():
        raise FileExistsError("This split acquisition attempt was already finalized")
    preparation = json.loads((root / "prepare_costs.json").read_text())
    prepared = json.loads((root / "prepare.json").read_text())
    timers, child, verified = {}, None, False
    limits = phase_limits(config, preparation)
    summary: dict[str, Any] = {"study": config["study"], "status": "started", "requests": 0,
                               "received_payload_bytes": 0, "gpu_seconds": 0,
                               "source_sha256": prepared.get("source_sha256", {}),
                               "storage_bound": prepared.get("storage_bound", {})}

    def guard() -> None:
        if (time.process_time() - clock["cpu"] + 2 >= limits["cpu_seconds"]
                or time.monotonic() - clock["wall"] + 2 >= limits["wall_seconds"]):
            raise TimeoutError("Offline finish reached its shared Python phase reserve")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 >= limits["rss_bytes"]:
            raise MemoryError("Offline finish RSS limit reached")

    def interrupted(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Offline finish process timer:{signum}")

    try:
        transport.require_runtime()
        transport.setup_timers(limits, clock, interrupted, timers)
        previous.verify_lineage(config, admission)
        verified = True
        if sha256_file(root / "prepare.json") != preparation["prepare_sha256"]:
            raise ValueError("Preparation receipt identity differs")
        if prepared["status"] != "PASS":
            child = {"cpu_seconds": 0., "wall_seconds": 0., "peak_rss_bytes": 0, "started": False}
            raise ValueError("Preparation failed; no transfer was authorized")
        external_paths = [root / config["curl"][key]
                          for key in ["resource_log", "response_headers", "suffix_file"]]
        if transfer_not_started:
            child = proven_unstarted(external_paths)
            raise ValueError("External transfer was proven not started")
        summary.update(requests=None, received_payload_bytes=(external_paths[2].stat().st_size
                       if external_paths[2].exists() else None))
        child = transfer_metrics(external_paths[0])
        summary.update(requests=1, received_payload_bytes=external_paths[2].stat().st_size,
                       external_sha256={path.name: sha256_file(path) for path in external_paths})
        child, response = verify_transfer(root, config)
        summary["response"] = {"status": response["status"], "content_range": response["headers"].get(
            "content-range"), "content_length": response["headers"].get("content-length"),
            "etag": response["headers"].get("etag"), "final_url": config["request_url"]}
        archive = assemble(root, config, guard)
        summary["archive"] = {"bytes": archive.stat().st_size, "sha256": sha256_file(archive)}
        flat = {**config, "resources": config["resources"]["total"]}
        summary.update(files=transport.extract(root, archive, flat, guard), record_files=2800, records=200)
        previous.reverify(config, prepared["source_input_sha256"], prepared["source_sha256"])
        guard()
        summary["status"] = "PASS"
    except Exception as error:
        summary.update(status="FAIL", error_type=type(error).__name__, error=str(error))
    finally:
        retained = previous.final_enumeration(root, summary, timers)
    completion(root, config, admission, summary, preparation, child, clock, verified, retained)
    return summary


def proven_unstarted(paths: list[Path]) -> dict[str, Any]:
    """Accept a pre-start declaration only when no external transfer artifact exists."""
    if any(path.exists() for path in paths):
        raise ValueError("Cannot declare an unstarted transfer when external files exist")
    return {"cpu_seconds": 0., "wall_seconds": 0., "peak_rss_bytes": 0, "started": False}
