"""Pin a fully hashed NPY cache for cheap checks within a local session.

A seal is an integrity aid for an unchanged local file, not a fresh full-file
hash or a defense against a writer able to forge both the file and the seal.
Consumers should pin the seal's SHA-256 in their own executable manifest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .files import sha256_file, write_json_atomic

SEAL_VERSION = 1
BLOCK_SIZE = 64 * 1024
RANDOM_BLOCKS = 6


@dataclass(frozen=True)
class CacheFileSpec:
    """One NPY file and its independently pinned source receipt."""

    name: str
    path: Path
    source_receipt_path: Path
    source_receipt_sha256: str
    source_sha256_key: str


def _digest(value: Any) -> str:
    """Hash a canonical JSON value for tamper detection."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _check_hex(value: Any, label: str) -> str:
    """Require an ordinary lowercase SHA-256 hexadecimal digest."""
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise ValueError(f"Invalid {label} SHA-256")
    return value


def _stat(path: Path) -> dict[str, int]:
    """Capture the fields that identify the current local file instance."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Cache path must be a regular, non-symlink file: {path}")
    item = path.stat()
    return {"size": item.st_size, "dev": item.st_dev, "inode": item.st_ino,
            "mtime_ns": item.st_mtime_ns, "ctime_ns": item.st_ctime_ns}


def _npy_header(path: Path) -> dict[str, Any]:
    """Read the NPY header without mapping or loading the array payload."""
    with path.open("rb") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version in ((2, 0), (3, 0)):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"Unsupported NPY format {version}: {path}")
    if dtype.hasobject:
        raise ValueError(f"Object NPY arrays are not accepted: {path}")
    return {"shape": list(shape), "dtype": dtype.str,
            "fortran_order": bool(fortran), "npy_version": list(version)}


def _offsets(size: int, full_sha256: str) -> list[int]:
    """Choose deterministic first, last and dispersed payload byte windows."""
    width = min(BLOCK_SIZE, size)
    last = size - width
    offsets = {0, last}
    for index in range(RANDOM_BLOCKS):
        seed = hashlib.sha256(f"cache-seal-v1:{full_sha256}:{size}:{index}".encode()).digest()
        offsets.add(int.from_bytes(seed[:8], "big") % (last + 1))
    return sorted(offsets)


def _blocks(path: Path, full_sha256: str, size: int) -> list[dict[str, Any]]:
    """Hash at most eight bounded windows and reject a concurrent file change."""
    before = _stat(path)
    if before["size"] != size:
        raise ValueError(f"Cache file size changed: {path}")
    width = min(BLOCK_SIZE, size)
    result = []
    with path.open("rb") as handle:
        for offset in _offsets(size, full_sha256):
            handle.seek(offset)
            chunk = handle.read(width)
            if len(chunk) != width:
                raise ValueError(f"Short cache block read: {path}")
            result.append({"offset": offset, "length": width,
                           "sha256": hashlib.sha256(chunk).hexdigest()})
    if _stat(path) != before:
        raise ValueError(f"Cache file changed during block verification: {path}")
    return result


def _receipt_digest(path: Path, expected_sha256: str, key: str) -> str:
    """Verify the small source receipt and obtain its pinned payload digest."""
    pinned = _check_hex(expected_sha256, "source receipt")
    if sha256_file(path) != pinned:
        raise ValueError(f"Source receipt SHA-256 mismatch: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if sha256_file(path) != pinned:
        raise ValueError(f"Source receipt changed while reading: {path}")
    if not isinstance(data, dict) or key not in data:
        raise ValueError(f"Source receipt lacks {key}: {path}")
    return _check_hex(data[key], f"source receipt {key}")


def create_seal(seal_path: Path, specs: list[CacheFileSpec]) -> dict[str, Any]:
    """Write a versioned seal only after fresh full SHA-256 checks pass.

    The historical receipt supplies the expected digest but cannot by itself
    establish the current file's stat identity. Each call rehashes every NPY.
    """
    seal_path = Path(seal_path)
    if seal_path.exists():
        raise FileExistsError(f"Cache seal already exists: {seal_path}")
    if not specs or len({spec.name for spec in specs}) != len(specs):
        raise ValueError("Cache seal requires uniquely named files")
    entries = {}
    for spec in specs:
        if not spec.name or "/" in spec.name:
            raise ValueError(f"Invalid cache seal name: {spec.name}")
        path = Path(spec.path).absolute()
        receipt = Path(spec.source_receipt_path).resolve(strict=True)
        expected = _receipt_digest(receipt, spec.source_receipt_sha256,
                                   spec.source_sha256_key)
        before = _stat(path)
        header = _npy_header(path)
        actual = sha256_file(path)
        if _stat(path) != before:
            raise ValueError(f"Cache file changed during full SHA-256: {path}")
        if actual != expected:
            raise ValueError(f"Cache file full SHA-256 mismatch: {path}")
        blocks = _blocks(path, expected, before["size"])
        if _stat(path) != before:
            raise ValueError(f"Cache file changed during seal creation: {path}")
        entries[spec.name] = {
            "path": str(path), "full_sha256": expected, "stat": before,
            "npy": header, "blocks": blocks,
            "source_receipt": {"path": str(receipt),
                               "sha256": spec.source_receipt_sha256,
                               "sha256_key": spec.source_sha256_key},
        }
    seal = {"version": SEAL_VERSION, "creation_verification": "fresh_full_sha256",
            "files": entries}
    seal["seal_sha256"] = _digest(seal)
    write_json_atomic(seal_path, seal, sort_keys=True)
    return seal


def _external_queue_hashes(
    pre_stats_path: Path, profile_path: Path, queue_path: Path, status_path: Path,
    root: Path,
) -> tuple[dict[str, Path], dict[str, str], dict[str, Any], dict[str, Any]]:
    """Check completed 011 profile provenance before adopting its full hashes."""
    paths = {"pre_hash_stats": pre_stats_path, "profile": profile_path,
             "queue": queue_path, "status": status_path}
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    pre = json.loads(pre_stats_path.read_text(encoding="utf-8"))
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if (pre.get("version") != 1 or not isinstance(pre.get("files"), dict)
            or pre_stats_path.stat().st_mtime_ns >= profile_path.stat().st_mtime_ns):
        raise ValueError("Pre-hash snapshot was not captured before the profile")
    if (status.get("state") != "complete" or status.get("returncode") != 0
            or status.get("queue_manifest_sha256") != hashes["queue"]):
        raise ValueError("External profile queue did not complete successfully")
    jobs = queue.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 1:
        raise ValueError("External profile queue must contain one verified job")
    stages = jobs[0].get("stages")
    if not isinstance(stages, list) or len(stages) != 1:
        raise ValueError("External queue must contain one profile stage")
    command = stages[0].get("command", [])
    if (stages[0].get("name") != "profile" or len(command) < 4
            or command[-3:] != ["scripts.experiments.run_delta_memory25k011",
                                   "--stage", "profile"]
            or str(profile_path) not in [str((root / item).resolve())
                                         for item in jobs[0].get("required_artifacts", [])]):
        raise ValueError("External queue does not identify the 011 full-hash profile")
    identity_hashes = profile.get("identity", {}).get("hashes")
    if not isinstance(identity_hashes, dict):
        raise ValueError("External profile lacks full-SHA input identity")
    sources_path = (root / jobs[0]["sources"]).resolve(strict=True)
    source_hash = sha256_file(sources_path)
    if source_hash != jobs[0].get("sources_sha256"):
        raise ValueError("External queue source map changed")
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    runner = "scripts/experiments/run_delta_memory25k011.py"
    if sources.get(runner) != identity_hashes.get(runner):
        raise ValueError("External profile runner source hash disagrees with queue")
    paths["sources"] = sources_path
    hashes["sources"] = source_hash
    return paths, hashes, pre, identity_hashes


def adopt_external_profile_seal(
    seal_path: Path, specs: list[CacheFileSpec], *, pre_stats_path: Path,
    pre_stats_root: Path, profile_path: Path, queue_path: Path, status_path: Path,
    profile_hash_keys: dict[str, str], profile_receipt_hash_keys: dict[str, str],
    pre_stat_keys: dict[str, str],
) -> dict[str, Any]:
    """Adopt an already completed full-SHA profile without rereading NPY payloads.

    The pre-hash snapshot must have been written before the profile receipt.
    The completed queue must identify the exact runner that generated that
    profile. A missing or changed field fails closed; only bounded cache blocks
    are read here. This path records *external* full-SHA evidence explicitly.
    """
    seal_path = Path(seal_path)
    if seal_path.exists():
        raise FileExistsError(f"Cache seal already exists: {seal_path}")
    names = {spec.name for spec in specs}
    if (not specs or len(names) != len(specs) or names != set(profile_hash_keys)
            or names != set(profile_receipt_hash_keys) or names != set(pre_stat_keys)):
        raise ValueError("External profile mappings must cover unique cache files")
    pre_stats_path = Path(pre_stats_path).resolve(strict=True)
    profile_path = Path(profile_path).resolve(strict=True)
    queue_path = Path(queue_path).resolve(strict=True)
    status_path = Path(status_path).resolve(strict=True)
    evidence_paths, evidence_sha, pre, hashes = _external_queue_hashes(
        pre_stats_path, profile_path, queue_path, status_path, Path(pre_stats_root))
    entries = {}
    for spec in specs:
        path = Path(spec.path).absolute()
        receipt = Path(spec.source_receipt_path).resolve(strict=True)
        expected = _receipt_digest(receipt, spec.source_receipt_sha256,
                                   spec.source_sha256_key)
        if hashes.get(profile_receipt_hash_keys[spec.name]) != spec.source_receipt_sha256:
            raise ValueError(f"External profile source receipt SHA-256 disagrees: {spec.name}")
        if hashes.get(profile_hash_keys[spec.name]) != expected:
            raise ValueError(f"External profile full SHA-256 disagrees: {spec.name}")
        snapshot = pre["files"].get(pre_stat_keys[spec.name], {})
        snapshot_path = (Path(pre_stats_root) / snapshot.get("path", "")).resolve()
        snapshot_stat = {"size": snapshot.get("size"), "dev": snapshot.get("device"),
                         "inode": snapshot.get("inode"), "mtime_ns": snapshot.get("mtime_ns"),
                         "ctime_ns": snapshot.get("ctime_ns")}
        current = _stat(path)
        if snapshot_path != path.resolve() or snapshot_stat != current:
            raise ValueError(f"Cache stat changed since pre-hash snapshot: {spec.name}")
        header = _npy_header(path)
        blocks = _blocks(path, expected, current["size"])
        if _stat(path) != current:
            raise ValueError(f"Cache changed during external evidence adoption: {spec.name}")
        entries[spec.name] = {
            "path": str(path), "full_sha256": expected, "stat": current,
            "npy": header, "blocks": blocks,
            "source_receipt": {"path": str(receipt),
                               "sha256": spec.source_receipt_sha256,
                               "sha256_key": spec.source_sha256_key},
        }
    if any(sha256_file(path) != evidence_sha[name] for name, path in evidence_paths.items()):
        raise ValueError("External verification evidence changed during adoption")
    seal = {"version": SEAL_VERSION,
            "creation_verification": "external_completed_full_sha256_profile",
            "files": entries,
            "external_verification": {
                "evidence": {name: {"path": str(path), "sha256": evidence_sha[name]}
                             for name, path in evidence_paths.items()},
                "profile_hash_keys": profile_hash_keys,
                "profile_receipt_hash_keys": profile_receipt_hash_keys,
                "pre_stat_keys": pre_stat_keys,
            }}
    seal["seal_sha256"] = _digest(seal)
    write_json_atomic(seal_path, seal, sort_keys=True)
    return seal


def _load_seal(seal_path: Path, expected_seal_sha256: str | None) -> dict[str, Any]:
    """Validate seal syntax, digest and any pinned external evidence."""
    seal = json.loads(Path(seal_path).read_text(encoding="utf-8"))
    if not isinstance(seal, dict) or seal.get("version") != SEAL_VERSION:
        raise ValueError("Unsupported cache seal version")
    recorded = _check_hex(seal.get("seal_sha256"), "cache seal")
    content = {key: value for key, value in seal.items() if key != "seal_sha256"}
    if _digest(content) != recorded:
        raise ValueError("Cache seal contents changed")
    if expected_seal_sha256 is not None and recorded != _check_hex(
        expected_seal_sha256, "expected cache seal"
    ):
        raise ValueError("Cache seal does not match pinned SHA-256")
    mode = seal.get("creation_verification")
    if mode == "external_completed_full_sha256_profile":
        evidence = seal["external_verification"]["evidence"]
        for name, item in evidence.items():
            if sha256_file(item["path"]) != _check_hex(item["sha256"], name):
                raise ValueError(f"External verification evidence changed: {name}")
    elif mode != "fresh_full_sha256":
        raise ValueError("Cache seal lacks full SHA-256 creation evidence")
    return seal


def _validate_entry(name: str, entry: dict[str, Any]) -> None:
    """Check one cache file without computing its complete SHA-256."""
    path = Path(entry["path"])
    receipt = entry["source_receipt"]
    expected = _receipt_digest(Path(receipt["path"]), receipt["sha256"],
                               receipt["sha256_key"])
    if expected != _check_hex(entry["full_sha256"], f"{name} full cache"):
        raise ValueError(f"Cache and source receipt digests disagree: {name}")
    before = _stat(path)
    if before != entry["stat"]:
        raise ValueError(f"Cache file stat identity changed: {name}")
    if _npy_header(path) != entry["npy"]:
        raise ValueError(f"Cache NPY header changed: {name}")
    if _blocks(path, expected, before["size"]) != entry["blocks"]:
        raise ValueError(f"Cache bounded blocks changed: {name}")
    if _stat(path) != before:
        raise ValueError(f"Cache file changed during validation: {name}")


def validate_seal(seal_path: Path, *, expected_seal_sha256: str | None = None,
                  ) -> dict[str, Any]:
    """Check a pinned local cache using stat, NPY header and bounded blocks.

    This check does not recompute the full cache SHA-256. A caller requiring
    standalone cryptographic content verification must hash the full file.
    """
    seal = _load_seal(seal_path, expected_seal_sha256)
    files = seal.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Cache seal has no files")
    for name, entry in files.items():
        _validate_entry(name, entry)
    return {"status": "passed", "verification_mode": "cached_stat_and_bounded_blocks",
            "full_sha256_recomputed": False, "seal_sha256": seal["seal_sha256"],
            "creation_verification": seal["creation_verification"],
            "full_sha256_from_creation": {name: entry["full_sha256"]
                                          for name, entry in files.items()}}
