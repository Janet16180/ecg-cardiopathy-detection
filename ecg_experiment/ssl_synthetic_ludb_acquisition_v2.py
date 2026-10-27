"""Resume one opaque public LUDB archive prefix under a new prospective transport budget."""

import json
import os
import resource
import signal
import time
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ecg_experiment.files import sha256_file
from ecg_experiment.paths import from_stored, to_stored
from ecg_experiment.ssl_synthetic_ludb_acquisition import admitted_members

CONFIG = "configs/ssl_synthetic_ludb_acquisition_v2.json"
PROTOCOL = "docs/ssl-synthetic-localization-ludb-acquisition-v2.md"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Reject redirects so the prospective single-request budget counts actual wire GETs."""

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any,
                         newurl: str) -> None:
        """Stop instead of silently issuing a second archive request."""
        raise ValueError("Archive redirect is outside the single-GET admission")


def write_new(path: Path, value: Any) -> None:
    """Write a strict JSON receipt exclusively, retaining every previous attempt."""
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def directory_bytes(path: Path) -> int:
    """Count all retained local bytes without following symlinks."""
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file() and not item.is_symlink())


def confined(root: Path, name: str) -> Path:
    """Confine identity declarations to their repository-relative artifact root."""
    if Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("Expected repository-relative admitted path")
    return root / name


def snapshot(root: Path, admission: dict[str, Any], admission_name: str) -> dict[str, str]:
    """Bind unchanged scientific files and the new transport-only source before the request.

    Parameters
    ----------
    root : Path
        Fresh successor output directory.
    admission : dict
        Prospectively committed source identities.
    admission_name : str
        Repository-relative identity-only manifest.

    Returns
    -------
    dict
        Exact source and admission identities retained locally.
    """
    names = {CONFIG, PROTOCOL, "ecg_experiment/ssl_synthetic_ludb_acquisition_v2.py",
             "scripts/data/acquire_ssl_synthetic_ludb_v2.py",
             "tests/test_ssl_synthetic_ludb_acquisition_v2.py",
             "ecg_experiment/ssl_synthetic_ludb_acquisition.py", "ecg_experiment/paths.py",
             "ecg_experiment/files.py", "ecg_experiment/__init__.py", "pyproject.toml", "uv.lock"}
    sources = admission["source_sha256"]
    config = json.loads(from_stored(CONFIG).read_text())
    names.update([config["unchanged_scientific_protocol"], config["unchanged_scientific_config"]])
    if not names.issubset(sources):
        raise ValueError("Successor source admission is incomplete")
    if sources[CONFIG] != admission["frozen_config_sha256"] or sources[PROTOCOL] != admission[
            "frozen_protocol_sha256"]:
        raise ValueError("Frozen transport design identity differs")
    for name, expected in sources.items():
        source = from_stored(name)
        if sha256_file(source) != expected:
            raise ValueError(f"Admitted transport source changed:{name}")
        target = root / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(source.read_bytes())
    target = root / "admission_snapshot.json"
    with target.open("xb") as stream:
        stream.write(from_stored(admission_name).read_bytes())
    return {**sources, admission_name: sha256_file(target)}


def verify_old(root: Path, config: dict[str, Any], admission: dict[str, Any]) -> dict[str, str]:
    """Verify opaque failed-attempt bytes and the separately audited cumulative cost base."""
    old = from_stored(config["source_directory"])
    bindings = {"summary.json": config["source_summary_sha256"],
                "prospective_ledger.json": config["source_predecessor_ledger"]["sha256"],
                config["resume"]["prefix_file"]: config["resume"]["prefix_sha256"],
                config["checksum_manifest"]["file"]: config["checksum_manifest"]["sha256"]}
    for name, expected in bindings.items():
        if sha256_file(old / name) != expected:
            raise ValueError("Preserved failed-attempt identity differs")
    if (old / config["resume"]["prefix_file"]).stat().st_size != config["resume"]["prefix_bytes"]:
        raise ValueError("Preserved archive prefix length differs")
    if (old / config["checksum_manifest"]["file"]).stat().st_size != config["checksum_manifest"]["bytes"]:
        raise ValueError("Preserved official manifest length differs")
    for name, key in [("unchanged_scientific_protocol", "unchanged_scientific_protocol_sha256"),
                      ("unchanged_scientific_config", "unchanged_scientific_config_sha256")]:
        if sha256_file(from_stored(config[name])) != config[key]:
            raise ValueError("Scientific qualification design changed")
    ledger, receipt = verify_audit(admission)
    verify_lineage(config, admission, ledger, receipt)
    return bindings





def copy_prefix(root: Path, config: dict[str, Any], bindings: dict[str, str]) -> None:
    """Copy opaque verified v1 files exclusively after admitting the cumulative cost base."""
    old = from_stored(config["source_directory"])
    for name in bindings:
        target_name = "v1_" + name if name in ["summary.json", "prospective_ledger.json"] else name
        with (root / target_name).open("xb") as stream:
            stream.write((old / name).read_bytes())

def verify_audit(admission: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the immutable narrow-audit receipt and cumulative terminal values."""
    prior = admission["prior_ledger"]
    if sha256_file(from_stored(prior["path"])) != prior["sha256"]:
        raise ValueError("Audited terminal accounting identity differs")
    ledger = json.loads(from_stored(prior["path"]).read_text())
    for key in ["current_machine_stages", "current_machine_cumulative_cpu_seconds",
                "current_machine_cumulative_gpu_wall_seconds"]:
        if ledger[key] != prior[key]:
            raise ValueError("Audited terminal accounting values differ")
    audit = admission["failure_audit_summary"]
    if sha256_file(from_stored(audit["path"])) != audit["sha256"]:
        raise ValueError("Independent transport-failure audit identity differs")
    receipt = json.loads(from_stored(audit["path"]).read_text())
    if not receipt["all_passed"] or receipt["mismatch_count"] != 0:
        raise ValueError("Independent transport-failure audit did not pass")
    return ledger, receipt

def verify_lineage(config: dict[str, Any], admission: dict[str, Any], ledger: dict[str, Any],
                   audit: dict[str, Any]) -> None:
    """Require direct source-failure then narrow-audit accounting with no reset or skipped cost."""
    source = config["source_predecessor_ledger"]
    binding = admission["failure_audit_summary"]
    if (ledger["current_machine_stages"] != 20 or source["current_machine_stages"] != 19
            or ledger["prior_ledger_sha256"] != source["sha256"]
            or ledger["receipt"] != "summary.json" or ledger["sha256"] != binding["sha256"]):
        raise ValueError("Narrow failure audit must directly precede successor acquisition")
    if (ledger["current_machine_cumulative_cpu_seconds"]
            != source["current_machine_cumulative_cpu_seconds"] + ledger["cpu_seconds"]
            or ledger["current_machine_cumulative_gpu_wall_seconds"]
            != source["current_machine_cumulative_gpu_wall_seconds"]
            or ledger["gpu_wall_seconds"] != 0 or not all(ledger["resource_gates"].values())):
        raise ValueError("Failure audit costs must be charged exactly once within frozen caps")
    if (audit["input_sha256"]["prospective_ledger.json"] != source["sha256"]
            or not audit["acquisition_failure_preserved"]):
        raise ValueError("Failure audit does not preserve the admitted failed acquisition")

def verify_response(response: Any, config: dict[str, Any]) -> int:
    """Require an exact partial response before any response body is appended.

    Parameters
    ----------
    response : HTTPResponse
        The one admitted archive response.
    config : dict
        Frozen range and complete-length declarations.

    Returns
    -------
    int
        Exact number of required new response payload bytes.
    """
    resume = config["resume"]
    remaining = resume["expected_complete_bytes"] - resume["prefix_bytes"]
    if response.status != 206:
        raise ValueError("Archive resume requires HTTP206; no200fallback")
    if response.headers.get("Content-Range") != resume["required_content_range"]:
        raise ValueError("Archive Content-Range differs from exact admitted suffix")
    declared = response.headers.get("Content-Length")
    if declared is not None and declared != str(remaining):
        raise ValueError("Archive Content-Length differs from exact admitted suffix")
    return remaining


def receive(root: Path, config: dict[str, Any], summary: dict[str, Any], guard: Callable[[], None]) -> Path:
    """Append exactly one accepted response to the copied prefix, never modifying failed-attempt bytes."""
    resume = config["resume"]
    request = urllib.request.Request(config["request_url"], headers={
        "User-Agent": "ECG-local-research/1.0", "Range": resume["range_header"],
        "If-Range": '"' + resume["if_range_etag"] + '"'})
    summary["requests"] = 1
    archive = root / resume["prefix_file"]
    opener = urllib.request.build_opener(NoRedirect())
    with opener.open(request, timeout=30) as response:
        remaining = verify_response(response, config)
        summary["response"] = {"status": response.status, "final_url": response.url,
                               "content_range": response.headers.get("Content-Range"),
                               "content_length": response.headers.get("Content-Length"),
                               "etag": response.headers.get("ETag")}
        with archive.open("ab") as stream:
            while True:
                guard()
                chunk = response.read(min(65536, remaining - summary["received_payload_bytes"] + 1))
                if not chunk:
                    break
                summary["received_payload_bytes"] += len(chunk)
                if (summary["received_payload_bytes"] > remaining
                        or summary["received_payload_bytes"] > config["resources"]["received_payload_bytes"]):
                    raise ValueError("Archive suffix exceeds admitted exact length or byte cap")
                stream.write(chunk)
    if summary["received_payload_bytes"] != remaining:
        raise ValueError("Archive suffix is shorter than admitted exact length")
    if archive.stat().st_size != resume["expected_complete_bytes"]:
        raise ValueError("Completed archive byte length differs")
    archive = archive.rename(root / "release.zip")
    summary["archive"] = {"bytes": archive.stat().st_size, "sha256": sha256_file(archive)}
    return archive


def checksum_manifest(path: Path) -> dict[str, str]:
    """Parse only the admitted official file-identity manifest, never record contents."""
    result = {}
    for line in path.read_text(encoding="ascii").splitlines():
        checksum, name = line.split(maxsplit=1)
        name = name.removeprefix("*").removeprefix("./")
        if (len(checksum) != 64 or any(value not in "0123456789abcdef" for value in checksum)
                or name in result):
            raise ValueError("Invalid or duplicate official checksum declaration")
        result[name] = checksum
    return result



def copy_member(source: Any, target: Any, expected_bytes: int, guard: Callable[[], None]) -> None:
    """Stream one bounded archive member while enforcing live byte and process limits."""
    count = 0
    while True:
        guard()
        block = source.read(min(65536, expected_bytes - count + 1))
        if not block:
            break
        count += len(block)
        if count > expected_bytes:
            raise ValueError("Archive member exceeds declared uncompressed length")
        target.write(block)
    if count != expected_bytes:
        raise ValueError("Archive member is shorter than declared uncompressed length")

def extract(root: Path, archive_path: Path, config: dict[str, Any],
            guard: Callable[[], None]) -> dict[str, Any]:
    """Extract only admitted files with new guards and the frozen safe-member contract."""
    checksums = checksum_manifest(root / config["checksum_manifest"]["file"])
    identities, total = {}, 0
    with zipfile.ZipFile(archive_path) as archive:
        for name, member in admitted_members(archive).items():
            guard()
            if name == "SHA256SUMS.txt":
                continue
            total += member.file_size
            if total > config["resources"]["selective_extract_bytes"]:
                raise ValueError("Selective extraction exceeds successor byte budget")
            target = confined(root, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("xb") as stream:
                copy_member(source, stream, member.file_size, guard)
            if target.stat().st_size != member.file_size:
                raise ValueError("Extracted member length differs")
            digest = sha256_file(target)
            if name not in checksums or checksums[name] != digest:
                raise ValueError("Admitted release member differs from official checksum")
            identities[name] = {"bytes": member.file_size, "sha256": digest}
    if sum(name.startswith("data/") for name in identities) != 2800:
        raise ValueError("Require exact2800record files")
    return identities


def finish(root: Path, config: dict[str, Any], admission: dict[str, Any], summary: dict[str, Any],
           clock: dict[str, float], verified: bool) -> None:
    """Save success or retained failure and append measured successor costs exactly once."""
    resources = config["resources"]
    summary.update(cpu_seconds=time.process_time() - clock["cpu"],
                   wall_seconds=time.monotonic() - clock["wall"],
                   peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    summary["resource_gates"] = {
        "cpu": summary["cpu_seconds"] < resources["cpu_seconds"],
        "wall": summary["wall_seconds"] < resources["wall_seconds"],
        "rss": summary["peak_rss_bytes"] < resources["rss_bytes"],
        "received_bytes": summary["received_payload_bytes"] <= resources["received_payload_bytes"],
        "output_bytes": directory_bytes(root) + 16384 < resources["new_output_bytes"]}
    if not all(summary["resource_gates"].values()):
        summary["status"] = "FAIL"
    write_new(root / "summary.json", summary)
    receipt_sha = sha256_file(root / "summary.json")
    size = directory_bytes(root)
    cpu, wall = time.process_time() - clock["cpu"], time.monotonic() - clock["wall"]
    terminal_gates = {**summary["resource_gates"], "cpu": cpu < resources["cpu_seconds"],
                      "wall": wall < resources["wall_seconds"],
                      "output_bytes": size + 16384 < resources["new_output_bytes"]}
    prior = admission["prior_ledger"]
    write_new(root / "prospective_ledger.json", {
        "receipt": to_stored(root / "summary.json"), "sha256": receipt_sha,
        "cpu_seconds": cpu, "wall_seconds": wall, "gpu_wall_seconds": 0,
        "current_machine_stages": prior["current_machine_stages"] + 1 if verified else None,
        "prior_ledger_sha256": prior["sha256"] if verified else None,
        "current_machine_cumulative_cpu_seconds": prior["current_machine_cumulative_cpu_seconds"] + cpu
        if verified else None,
        "current_machine_cumulative_gpu_wall_seconds": prior["current_machine_cumulative_gpu_wall_seconds"]
        if verified else None,
        "prior_accounting_verified": verified, "historical_costs_verified": False,
        "output_bytes_before_terminal_ledger": size, "resource_gates": terminal_gates,
        "measurement_endpoint": ("after receipt hashing and byte enumeration "
                                 "before terminal ledger serialization"),
        "terminal_bookkeeping_reserve_seconds": 2, "terminal_bookkeeping_reserve_bytes": 16384})
    if (cpu >= resources["cpu_seconds"] or wall >= resources["wall_seconds"]
            or directory_bytes(root) >= resources["new_output_bytes"] or summary["status"] != "PASS"):
        raise RuntimeError("Acquisition successor failed; retained all local attempt bytes")


def acquire(admission_name: str, start_clock: dict[str, float] | None = None) -> dict[str, Any]:
    """Run one prospectively admitted range request without retries or fallback.

    Parameters
    ----------
    admission_name : str
        Committed repository-relative identity-only manifest.
    start_clock : dict or None
        CPU/monotonic times captured before importing acquisition source.

    Returns
    -------
    dict
        A completed local release receipt; failures remain and are raised.
    """
    clock = start_clock or {"cpu": time.process_time(), "wall": time.monotonic()}
    config = json.loads(from_stored(CONFIG).read_text())
    admission = json.loads(from_stored(admission_name).read_text())
    root = from_stored(config["output_directory"])
    root.mkdir(parents=True, exist_ok=False)
    resources, verified, timers = config["resources"], False, {}
    summary: dict[str, Any] = {"study": config["study"], "status": "started", "requests": 0,
                               "received_payload_bytes": 0, "gpu_seconds": 0,
                               "url": config["url"], "request_url": config["request_url"]}

    def guard() -> None:
        if (time.process_time() - clock["cpu"] + 2 >= resources["cpu_seconds"]
                or time.monotonic() - clock["wall"] + 2 >= resources["wall_seconds"]):
            raise TimeoutError("Successor acquisition reached reserved receipt boundary")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 >= resources["rss_bytes"]:
            raise MemoryError("Successor acquisition RSS budget exceeded")
        if directory_bytes(root) + 16384 >= resources["new_output_bytes"]:
            raise ValueError("Successor acquisition retained-output cap reached")

    def interrupted(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Acquisition successor process timer:{signum}")

    try:
        require_runtime()
        setup_timers(resources, clock, interrupted, timers)
        sources = snapshot(root, admission, admission_name)
        summary["source_sha256"] = sources
        old = verify_old(root, config, admission)
        verified = True
        summary["v1_input_sha256"] = old
        copy_prefix(root, config, old)
        guard()
        archive = receive(root, config, summary, guard)
        summary.update(files=extract(root, archive, config, guard), record_files=2800, records=200)
        reverify(config, old, sources)
        guard()
        summary["status"] = "PASS"
    except Exception as error:
        summary.update(status="FAIL", error_type=type(error).__name__, error=str(error))
    finally:
        for timer, (signum, previous) in timers.items():
            signal.setitimer(timer, 0)
            signal.signal(signum, previous)
    finish(root, config, admission, summary, clock, verified)
    return summary



def reverify(config: dict[str, Any], old: dict[str, str], sources: dict[str, str]) -> None:
    """Rehash unchanged failed-attempt bytes and all admitted sources before terminal accounting."""
    for name, expected in old.items():
        if sha256_file(from_stored(config["source_directory"]) / name) != expected:
            raise ValueError("Preserved v1input changed during successor")
    for name, expected in sources.items():
        if sha256_file(from_stored(name)) != expected:
            raise ValueError("Admitted source changed during successor")

def require_runtime() -> None:
    """Require hidden CUDA and one numerical thread without importing numerical libraries."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or any(os.environ.get(name) != "1" for name in [
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]):
        raise ValueError("Acquisition requires hidden CUDA and one numerical thread")


def setup_timers(resources: dict[str, Any], clock: dict[str, float], handler: Any,
                 timers: dict[int, Any]) -> None:
    """Bound total elapsed CPU and wall time including source-import overhead."""
    for timer, signum, duration, elapsed in [
        (signal.ITIMER_REAL, signal.SIGALRM, resources["wall_seconds"], time.monotonic() - clock["wall"]),
        (signal.ITIMER_PROF, signal.SIGPROF, resources["cpu_seconds"], time.process_time() - clock["cpu"]),
    ]:
        if signal.getitimer(timer) != (0., 0.):
            raise ValueError("Acquisition requires exclusive process timers")
        timers[timer] = (signum, signal.getsignal(signum))
        signal.signal(signum, handler)
        signal.setitimer(timer, max(.01, duration - elapsed - 2))
