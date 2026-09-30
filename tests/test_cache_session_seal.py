"""Creation and bounded validation of local NPY cache session seals."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from ecg_experiment import cache_session_seal as seal
from ecg_experiment.files import sha256_file


def _fixture(tmp_path: Path) -> tuple[seal.CacheFileSpec, Path]:
    """Build one NPY and a pinned, small source receipt."""
    path = tmp_path / "signals.npy"
    np.save(path, np.arange(2048, dtype=np.float32).reshape(128, 16))
    receipt = tmp_path / "complete.json"
    receipt.write_text(json.dumps({"signals_sha256": sha256_file(path)}))
    spec = seal.CacheFileSpec("training_signals", path, receipt, sha256_file(receipt),
                              "signals_sha256")
    return spec, tmp_path / "seal.json"


def test_fresh_seal_checks_full_hash_then_bounded_validation(tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    """Creation uses full SHA; subsequent checks avoid hashing the NPY payload."""
    spec, target = _fixture(tmp_path)
    created = seal.create_seal(target, [spec])
    assert created["creation_verification"] == "fresh_full_sha256"
    assert created["files"][spec.name]["npy"]["shape"] == [128, 16]
    assert created["files"][spec.name]["npy"]["dtype"] == "<f4"
    original = seal.sha256_file

    def small_only(path: str | Path) -> str:
        if str(path).endswith(".npy"):
            raise AssertionError("validation unexpectedly computed a full NPY SHA")
        return original(path)

    monkeypatch.setattr(seal, "sha256_file", small_only)
    checked = seal.validate_seal(target, expected_seal_sha256=created["seal_sha256"])
    assert checked["verification_mode"] == "cached_stat_and_bounded_blocks"
    assert checked["full_sha256_recomputed"] is False
    assert checked["full_sha256_from_creation"][spec.name] == json.loads(
        spec.source_receipt_path.read_text())["signals_sha256"]


def test_seal_rejects_changed_cache_receipt_and_seal(tmp_path: Path) -> None:
    """File instance changes, source changes and seal edits all fail closed."""
    spec, target = _fixture(tmp_path)
    created = seal.create_seal(target, [spec])
    with spec.path.open("r+b") as handle:
        handle.seek(256)
        handle.write(b"CHANGED!")
    with pytest.raises(ValueError, match="stat identity|bounded blocks"):
        seal.validate_seal(target)

    np.save(spec.path, np.arange(2048, dtype=np.float32).reshape(128, 16))
    spec.source_receipt_path.write_text("{}")
    with pytest.raises(ValueError, match="receipt SHA-256"):
        seal.validate_seal(target)

    spec.source_receipt_path.write_text(json.dumps({"signals_sha256": sha256_file(spec.path)}))
    value = json.loads(target.read_text())
    value["files"][spec.name]["stat"]["size"] += 1
    target.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="seal contents"):
        seal.validate_seal(target)
    with pytest.raises(FileExistsError, match="already exists"):
        seal.create_seal(target, [spec])
    assert created["seal_sha256"]


def test_fresh_creation_needs_matching_receipt_and_full_hash(tmp_path: Path) -> None:
    """A historical SHA receipt is never enough to seal changed current bytes."""
    spec, target = _fixture(tmp_path)
    with spec.path.open("r+b") as handle:
        handle.seek(512)
        handle.write(b"different")
    with pytest.raises(ValueError, match="full SHA-256 mismatch"):
        seal.create_seal(target, [spec])
    assert not target.exists()


def _external_evidence(tmp_path: Path, spec: seal.CacheFileSpec) -> dict[str, Path]:
    """Build the frozen profile and queue evidence around a pre-hash snapshot."""
    stat = spec.path.stat()
    pre = tmp_path / "pre.json"
    pre.write_text(json.dumps({"version": 1, "files": {"train": {
        "path": str(spec.path), "size": stat.st_size, "device": stat.st_dev,
        "inode": stat.st_ino, "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns,
    }}}))
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"identity": {"hashes": {
        "cache_signals": sha256_file(spec.path),
        "cache_receipt": spec.source_receipt_sha256,
        "scripts/experiments/run_delta_memory25k011.py": "a" * 64,
    }}}))
    os.utime(pre, ns=(1_000_000_000, 1_000_000_000))
    queue = tmp_path / "queue.json"
    sources = tmp_path / "sources.json"
    sources.write_text(json.dumps({"scripts/experiments/run_delta_memory25k011.py": "a" * 64}))
    queue.write_text(json.dumps({"jobs": [{"stages": [{"name": "profile", "command": [
        "python", "-m", "scripts.experiments.run_delta_memory25k011", "--stage", "profile",
    ]}], "required_artifacts": [str(profile)], "sources": str(sources),
        "sources_sha256": sha256_file(sources)}]}))
    status = tmp_path / "status.json"
    status.write_text(json.dumps({"state": "complete", "returncode": 0,
                                  "queue_manifest_sha256": sha256_file(queue)}))
    return {"pre_stats_path": pre, "profile_path": profile,
            "queue_path": queue, "status_path": status}


def test_external_profile_adoption_is_labeled_and_bounded(tmp_path: Path,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """A completed external full-SHA profile can be adopted without another full read."""
    spec, target = _fixture(tmp_path)
    evidence = _external_evidence(tmp_path, spec)
    original = seal.sha256_file

    def small_only(path: str | Path) -> str:
        if str(path).endswith(".npy"):
            raise AssertionError("external adoption unexpectedly hashed an NPY")
        return original(path)

    monkeypatch.setattr(seal, "sha256_file", small_only)
    created = seal.adopt_external_profile_seal(
        target, [spec], **evidence, pre_stats_root=tmp_path,
        profile_hash_keys={spec.name: "cache_signals"},
        profile_receipt_hash_keys={spec.name: "cache_receipt"},
        pre_stat_keys={spec.name: "train"},
    )
    assert created["creation_verification"] == "external_completed_full_sha256_profile"
    assert seal.validate_seal(target)["creation_verification"] == created["creation_verification"]
    evidence["status_path"].write_text("{}")
    with pytest.raises(ValueError, match="External verification evidence changed"):
        seal.validate_seal(target)


def test_external_adoption_requires_unchanged_stat_and_completed_queue(tmp_path: Path) -> None:
    """Evidence cannot be adopted after a file replacement or queue failure."""
    spec, target = _fixture(tmp_path)
    evidence = _external_evidence(tmp_path, spec)
    status = evidence["status_path"]
    value = json.loads(status.read_text())
    value["state"] = "failed"
    status.write_text(json.dumps(value))
    kwargs = {**evidence, "pre_stats_root": tmp_path,
              "profile_hash_keys": {spec.name: "cache_signals"},
              "profile_receipt_hash_keys": {spec.name: "cache_receipt"},
              "pre_stat_keys": {spec.name: "train"}}
    with pytest.raises(ValueError, match="did not complete"):
        seal.adopt_external_profile_seal(target, [spec], **kwargs)
    value["state"] = "complete"
    status.write_text(json.dumps(value))
    replacement = tmp_path / "new.npy"
    np.save(replacement, np.arange(2048, dtype=np.float32).reshape(128, 16))
    os.replace(replacement, spec.path)
    with pytest.raises(ValueError, match="stat changed"):
        seal.adopt_external_profile_seal(target, [spec], **kwargs)
    assert not target.exists()
