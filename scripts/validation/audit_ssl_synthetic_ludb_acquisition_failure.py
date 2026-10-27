"""Audit retained LUDB transfer failure metadata without opening any ECG or ZIP member."""

import argparse
import hashlib
import json
import os
import resource
import signal
import time
from pathlib import Path
from typing import Any

RELATIVE = "outputs/ssl_synthetic_localization/ludb_acquisition_v1"
PROTOCOL = "docs/ssl-synthetic-localization-ludb-acquisition-failure-audit-v1.md"
INPUT_HASHES = {
    "summary.json": "f59fc938a23930ecbd30043c15f28a49499a1a8135f7ac35ead68016526de78b",
    "prospective_ledger.json": "e505549b8110b2e4e7dadeca285209b115ffc6e50121d9c6088cbced054130f6",
    "release.zip.partial": "1bf23aa1ac52cd6ec1205baca4b819c9e2c90842ecd0314c80dd4a48c1b4158f",
    "SHA256SUMS.txt": "cccef1f3529519db8f26a333c97a6872a7d0e3e5c64a1448b76b161fd87fb75f",
}


def digest(path: Path) -> str:
    """Hash allowed bytes without interpreting container or clinical content."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition: bool, message: str) -> None:
    """Fail explicitly when a prospective metadata requirement is violated."""
    if not condition:
        raise ValueError(message)


def write_new(path: Path, value: Any) -> None:
    """Write a strict exclusive receipt, preserving failed attempts."""
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def check_receipts(summary: dict[str, Any], ledger: dict[str, Any], prior: dict[str, Any]) -> None:
    """Verify immutable failure semantics, resource clocks and exactly-once stage accounting."""
    require(summary["status"] == "FAIL" and summary["error_type"] == "TimeoutError"
            and summary["error"] == "Acquisition wall deadline", "retained declared transfer failure")
    require(summary["downloads"] == 2 and summary["gpu_seconds"] == 0, "only two requests, no GPU")
    require("files" not in summary and "archive" not in summary, "no completed release declaration")
    require(0 <= summary["cpu_seconds"] <= ledger["cpu_seconds"] < 60, "terminal acquisition CPU")
    require(598 <= summary["wall_seconds"] <= ledger["wall_seconds"] < 600, "reserved wall deadline")
    require(summary["peak_rss_bytes"] < 2 * 1024**3 and all(summary["resource_gates"].values()),
            "acquisition actual resource gates")
    require(ledger["current_machine_stages"] == 19, "failed attempt is stage19")
    for suffix in ("cpu_seconds", "gpu_wall_seconds"):
        field = "current_machine_cumulative_" + suffix
        require(ledger[field] == prior[field] + ledger[suffix], "once-only acquisition cost arithmetic")
    require(ledger["gpu_wall_seconds"] == 0, "zero GPU cost")


def run(root: Path, source: Path, checker_sha: str) -> dict[str, Any]:
    """Execute one independently frozen narrow audit and retain terminal success or failure."""
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    directory = root / RELATIVE
    destination = directory / "independent_failure_audit_v1"
    destination.mkdir(exist_ok=False)
    prior_costs: dict[str, Any] = {}

    def interrupted(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Narrow acquisition audit deadline:{signum}")

    signal.signal(signal.SIGALRM, interrupted)
    signal.signal(signal.SIGPROF, interrupted)
    signal.setitimer(signal.ITIMER_REAL, 58)
    signal.setitimer(signal.ITIMER_PROF, 28)
    receipt = "failure.json"
    try:
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "hidden CUDA")
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            require(os.environ.get(key) == "1", "single numerical thread")
        require(digest(Path(__file__)) == checker_sha, "literal checker source identity")
        for name, expected in INPUT_HASHES.items():
            require(digest(directory / name) == expected, "literal retained input identity")
        summary = json.loads((directory / "summary.json").read_text())
        ledger = json.loads((directory / "prospective_ledger.json").read_text())
        prior_binding = summary["prior_ledger"]
        require(digest(root / prior_binding["path"]) == prior_binding["sha256"]
                == ledger["prior_ledger_sha256"], "unchanged audited predecessor")
        prior = json.loads((root / prior_binding["path"]).read_text())
        check_receipts(summary, ledger, prior)
        require(ledger["sha256"] == INPUT_HASHES["summary.json"], "failed receipt terminal binding")
        prior_costs = ledger
        require((directory / "release.zip.partial").stat().st_size == 6291456, "retained prefix size")
        require((directory / "SHA256SUMS.txt").stat().st_size == 215460, "retained manifest size")
        require(not (directory / "data").exists() and not (directory / "release.zip").exists(),
                "no extraction or complete archive")
        for name, expected in summary["source_sha256"].items():
            require(digest(source / name) == expected, "frozen acquisition source binding")
        snapshot = destination / "audit_script.py"
        snapshot.write_bytes(Path(__file__).read_bytes())
        (destination / "audit_protocol.md").write_bytes((source / PROTOCOL).read_bytes())
        result = {"all_passed": True, "mismatch_count": 0, "input_sha256": INPUT_HASHES,
                  "checker_sha256": checker_sha, "protocol_sha256": digest(source / PROTOCOL),
                  "source_bindings": summary["source_sha256"], "retained_partial_bytes": 6291456,
                  "new_waveform_reads": 0, "new_annotation_reads": 0, "zip_member_reads": 0,
                  "new_AP_computations": 0, "downloads": 0, "gpu_used": False,
                  "acquisition_failure_preserved": True,
                  "acquisition_cpu_seconds": ledger["cpu_seconds"],
                  "acquisition_wall_seconds": ledger["wall_seconds"]}
        receipt = "summary.json"
        write_new(destination / receipt, result)
        return result
    except Exception as error:
        write_new(destination / receipt, {"all_passed": False, "error_type": type(error).__name__,
                                         "error": str(error)})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.setitimer(signal.ITIMER_PROF, 0)
        receipt_sha = digest(destination / receipt)
        size = sum(path.stat().st_size for path in destination.rglob("*") if path.is_file())
        cpu, wall = time.process_time() - cpu_start, time.perf_counter() - wall_start
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        gates = {"cpu": cpu < 30, "wall": wall < 60, "rss": rss < 2 * 1024**3,
                 "output": size + 65536 < 16 * 1024**2}
        write_new(destination / "prospective_ledger.json", {
            "receipt": receipt, "sha256": receipt_sha, "cpu_seconds": cpu, "wall_seconds": wall,
            "gpu_wall_seconds": 0., "peak_rss_bytes": rss, "resource_gates": gates,
            "current_machine_stages": 20, "prior_ledger_sha256": INPUT_HASHES["prospective_ledger.json"],
            "current_machine_cumulative_cpu_seconds": prior_costs["current_machine_cumulative_cpu_seconds"]
            + cpu if prior_costs else None,
            "current_machine_cumulative_gpu_wall_seconds": prior_costs.get(
                "current_machine_cumulative_gpu_wall_seconds"),
            "historical_costs_verified": False, "bytes_before_terminal_ledger": size,
            "measurement_endpoint": ("after receipt hashing/byte enumeration "
                                     "before terminal ledger serialization"),
            "terminal_bookkeeping_reserve_seconds": 2, "terminal_bookkeeping_reserve_bytes": 65536})
        require(all(gates.values()), "narrow audit terminal resource ceilings")


def main() -> None:
    """Require explicit source roots and checker identity for the one authorized audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checker-sha", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.source_root, args.checker_sha)))


if __name__ == "__main__":
    main()
