"""Acquire only the prospectively admitted public LUDB release, keeping all bytes local."""

import hashlib
import json
import resource
import signal
import stat
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from ecg_experiment.paths import from_stored, to_stored

URL = (
    "https://physionet.org/static/published-projects/ludb/"
    "lobachevsky-university-electrocardiography-database-1.0.1.zip"
)
EXTENSIONS = ("dat", "hea", "i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6")
EXPECTED_BYTES = 14474484
BYTE_CAP = 64 * 1024**2
OUTPUT = "outputs/ssl_synthetic_localization/ludb_acquisition_v1"
PRIOR = (
    "outputs/ssl_synthetic_localization/generated_temporal_diagnosis_v2/"
    "independent_audit_v1/prospective_ledger.json"
)
PRIOR_SHA = "a5e28f417f3d4d4ea445176149e2812a5efa48f0b7680a4a515cd94d7b518daf"


def digest(path: Path) -> str:
    """Hash local bytes without interpreting record content.

    Parameters
    ----------
    path : Path
        Local admitted asset.

    Returns
    -------
    str
        SHA-256 hexadecimal digest.
    """
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def admitted_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    """Validate exact record membership and bounded selective extraction.

    Parameters
    ----------
    archive : ZipFile
        Downloaded official release archive.

    Returns
    -------
    dict
        Release-relative selected members; header comments remain uninterpreted bytes.
    """
    selected = {}
    seen = set()
    expected = {f"data/{record}.{ext}" for record in range(1, 201) for ext in EXTENSIONS}
    metadata = {"RECORDS", "ANNOTATORS", "LICENSE.txt", "SHA256SUMS.txt"}
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if path.is_absolute() or ".." in path.parts or member.filename in seen:
            raise ValueError("Unsafe archive member path")
        seen.add(member.filename)
        if stat.S_ISLNK(member.external_attr >> 16):
            raise ValueError("Archive symlinks are forbidden")
        if member.is_dir():
            continue
        relative = PurePosixPath(*path.parts[1:]).as_posix()
        if relative not in expected | metadata:
            continue
        if relative in selected or member.file_size > BYTE_CAP:
            raise ValueError("Duplicate or excessive archive member")
        selected[relative] = member
    if expected - selected.keys():
        raise ValueError("Official archive lacks an admitted record file")
    if sum(member.file_size for member in selected.values()) > BYTE_CAP:
        raise ValueError("Selective extraction exceeds resource budget")
    return selected


def _guard(start_cpu: float, start_wall: float) -> None:
    if time.process_time() - start_cpu > 58 or time.monotonic() - start_wall > 598:
        raise TimeoutError("Acquisition reached reserved receipt boundary")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 2 * 1024**3:
        raise MemoryError("Acquisition RSS budget exceeded")


def _timeout(_signum: int, _frame: Any) -> None:
    raise TimeoutError("Acquisition wall deadline")


def _manifest(root: Path, summary: dict[str, Any]) -> tuple[dict[str, str], int]:
    url = "https://physionet.org/files/ludb/1.0.1/SHA256SUMS.txt"
    summary["downloads"] += 1
    with urllib.request.urlopen(url, timeout=30) as response:
        received = response.read(1024**2 + 1)
    if len(received) > 1024**2:
        raise ValueError("Official checksum manifest exceeds one MiB")
    path = root / "SHA256SUMS.txt"
    path.write_bytes(received)
    checksums = {}
    for line in received.decode("ascii").splitlines():
        checksum, name = line.split(maxsplit=1)
        name = name.removeprefix("*").removeprefix("./")
        if len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
            raise ValueError("Invalid official checksum declaration")
        if name in checksums:
            raise ValueError("Duplicate official checksum declaration")
        checksums[name] = checksum
    summary["checksum_manifest"] = {"url": url, "bytes": len(received), "sha256": digest(path)}
    return checksums, len(received)


def _extract(
    root: Path, archive_path: Path, checksums: dict[str, str], start_cpu: float, start_wall: float
) -> dict[str, Any]:
    identities = {}
    with zipfile.ZipFile(archive_path) as archive:
        for name, member in admitted_members(archive).items():
            if name == "SHA256SUMS.txt":
                continue
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("xb") as destination:
                destination.write(source.read(BYTE_CAP + 1))
            if target.stat().st_size != member.file_size:
                raise ValueError("Extracted member length differs")
            received_sha = digest(target)
            if name not in checksums or checksums[name] != received_sha:
                raise ValueError("Release member differs from official SHA256SUMS")
            identities[name] = {"bytes": member.file_size, "sha256": received_sha}
            _guard(start_cpu, start_wall)
    return identities


def acquire() -> dict[str, Any]:
    """Run one bounded download and extract only declared assets, preserving failure bytes.

    Returns
    -------
    dict
        Actual local acquisition receipt; failures are also written and re-raised.
    """
    start_cpu, start_wall = time.process_time(), time.monotonic()
    root = from_stored(OUTPUT)
    root.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(598)
    summary: dict[str, Any] = {"url": URL, "status": "started", "downloads": 0, "gpu_seconds": 0}
    error = None
    try:
        if digest(from_stored(PRIOR)) != PRIOR_SHA:
            raise ValueError("Prior audited ledger identity differs")
        checksums, manifest_length = _manifest(root, summary)
        _guard(start_cpu, start_wall)
        request = urllib.request.Request(URL, headers={"User-Agent": "ECG-local-research/1.0"})
        archive_path = root / "release.zip.partial"
        summary["downloads"] += 1
        with urllib.request.urlopen(request, timeout=30) as response, archive_path.open("xb") as stream:
            summary["final_url"] = response.url
            count = 0
            while chunk := response.read(1024 * 1024):
                count += len(chunk)
                if count + manifest_length > BYTE_CAP:
                    raise ValueError("Download exceeds admitted size cap")
                stream.write(chunk)
                _guard(start_cpu, start_wall)
        if count != EXPECTED_BYTES:
            raise ValueError("Release archive length differs from public HEAD declaration")
        archive_path = archive_path.rename(root / "release.zip")
        summary["archive"] = {"bytes": count, "sha256": digest(archive_path)}
        identities = _extract(root, archive_path, checksums, start_cpu, start_wall)
        summary.update(status="PASS", files=identities, record_files=2800, records=200)
    except Exception as caught:
        summary.update(status="FAIL", error_type=type(caught).__name__, error=str(caught))
        error = caught
    finally:
        signal.alarm(0)
        summary.update(
            cpu_seconds=time.process_time() - start_cpu,
            wall_seconds=time.monotonic() - start_wall,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            prior_ledger={"path": PRIOR, "sha256": PRIOR_SHA},
            source_sha256={
                name: digest(from_stored(name))
                for name in [
                    "ecg_experiment/ssl_synthetic_ludb_acquisition.py",
                    "scripts/data/acquire_ssl_synthetic_ludb.py",
                    "docs/ssl-synthetic-localization-ludb-qualification-v1.md",
                    "configs/ssl_synthetic_ludb_qualification_v1.json",
                ]
            },
        )
        summary["resource_gates"] = {
            "cpu": summary["cpu_seconds"] <= 60,
            "wall": summary["wall_seconds"] <= 600,
            "rss": summary["peak_rss_bytes"] <= 2 * 1024**3,
        }
        if not all(summary["resource_gates"].values()):
            summary["status"] = "FAIL"
            error = error or RuntimeError("Acquisition exceeded a resource cap")
        (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        prior = json.loads(from_stored(PRIOR).read_text())
        ledger = {
            "receipt": to_stored(root / "summary.json"),
            "sha256": digest(root / "summary.json"),
            "cpu_seconds": time.process_time() - start_cpu,
            "wall_seconds": time.monotonic() - start_wall,
            "gpu_wall_seconds": 0,
            "current_machine_stages": 19,
            "prior_ledger_sha256": PRIOR_SHA,
            "current_machine_cumulative_cpu_seconds": prior["current_machine_cumulative_cpu_seconds"],
            "current_machine_cumulative_gpu_wall_seconds": (
                prior["current_machine_cumulative_gpu_wall_seconds"]
            ),
            "historical_costs_verified": False,
            "measurement_endpoint": "after receipt hash before ledger assembly/serialization",
        }
        ledger["current_machine_cumulative_cpu_seconds"] += ledger["cpu_seconds"]
        (root / "prospective_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    if error is not None:
        raise error
    return summary
