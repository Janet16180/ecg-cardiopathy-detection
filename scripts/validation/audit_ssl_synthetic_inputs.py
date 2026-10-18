"""Audit every permitted PTB-XL train/validation input, retaining protected roles unread."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from typing import Any

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file
from ecg_experiment.paths import to_stored
from ecg_experiment.ssl_synthetic_checksum_audit import audit_duplicates, read_checksums, verify_partition
from ecg_experiment.ssl_synthetic_inputs import audit_record, write_audit
from ecg_experiment.ssl_synthetic_readiness import checked_relative

OUTPUT = ROOT / "outputs/ssl_synthetic_localization/input_audit_v1"
SOURCE_FILES = (
    "ecg_experiment/ssl_synthetic_inputs.py",
    "scripts/validation/audit_ssl_synthetic_inputs.py",
    "docs/ssl-synthetic-localization-input-audit.md",
    "ecg_experiment/ecg_quality.py",
    "ecg_experiment/waveforms.py",
    "ecg_experiment/files.py",
    "ecg_experiment/paths.py",
    "ecg_experiment/ssl_synthetic_checksum_audit.py",
    "ecg_experiment/ssl_synthetic_partition.py",
    "ecg_experiment/ssl_synthetic_readiness.py",
    "pyproject.toml",
    "uv.lock",
)


def load_bound_inputs() -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    """
    Replay immutable partition and duplicate evidence before opening any waveform.

    Returns
    -------
    tuple[dict, dict, dict]
        Manifest, official checksums and aggregate input binding.
    """
    base = ROOT / "outputs/ssl_synthetic_localization"
    summary_path = base / "partition_summary.json"
    duplicate_path = base / "partition_checksum_audit.json"
    summary = json.loads(summary_path.read_bytes())
    previous = json.loads(duplicate_path.read_bytes())
    manifest_path = ROOT / checked_relative(summary["manifest"])
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    metadata_path = ROOT / checked_relative(manifest["metadata"])
    verify_partition(manifest, summary, manifest_bytes, metadata_path.read_bytes())
    sources = manifest["source_sha256"] | previous["source_sha256"]
    for name, expected in sources.items():
        if sha256_file(ROOT / checked_relative(name)) != expected:
            raise ValueError("Frozen partition source changed")
    checksum_path = ROOT / checked_relative(previous["checksums"])
    checksum_bytes = checksum_path.read_bytes()
    if hashlib.sha256(checksum_bytes).hexdigest() != previous["checksums_sha256"]:
        raise ValueError("Official checksum source changed")
    if sha256_file(summary_path) != previous["partition_summary_sha256"]:
        raise ValueError("Original partition summary changed")
    if (
        previous["manifest_sha256"] != summary["manifest_sha256"]
        or previous["metadata_sha256"] != manifest["metadata_sha256"]
        or previous["manifest"] != summary["manifest"]
    ):
        raise ValueError("Duplicate receipt partition binding differs")
    checksums = read_checksums(checksum_bytes.decode())
    if checksums["ptbxl_database.csv"] != manifest["metadata_sha256"]:
        raise ValueError("Metadata differs from official table")
    replay = audit_duplicates(manifest["records"], checksums)
    if any(previous[key] != value for key, value in replay.items()) or not replay["duplicate_gate_passed"]:
        raise ValueError("Declared duplicate audit does not reproduce")
    binding = {
        "input_sha256": {
            to_stored(p): sha256_file(p)
            for p in (summary_path, duplicate_path, manifest_path, metadata_path, checksum_path)
        },
        "source_sha256": {name: sha256_file(ROOT / name) for name in SOURCE_FILES},
        "duplicate_audit_reproduced": True,
        "original_partition_counts": manifest["counts"],
    }
    return manifest, checksums, binding


def main() -> int:
    """
    Stream permitted records and emit source-bound local audit artifacts.

    Returns
    -------
    int
        Zero for a passing input integrity gate, two for recorded hard failures.
    """
    argparse.ArgumentParser(description=__doc__).parse_args()
    required = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    if any(os.environ.get(name) != "1" for name in required) or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Set numerical thread limits to 1 and CUDA_VISIBLE_DEVICES to empty")
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise FileExistsError("Input audit output directory must be new")
    started = datetime.now(UTC).isoformat()
    wall, cpu = time.perf_counter(), time.process_time()
    manifest, checksums, binding = load_bound_inputs()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    raw = ROOT / "data/raw/ptb-xl/1.0.3"
    records = []
    with (OUTPUT / "records.jsonl").open("x") as stream:
        for record in manifest["records"]:
            result = audit_record(record, raw, checksums)
            records.append(result)
            stream.write(json.dumps(result) + "\n")
            if len(records) % 2000 == 0:
                stream.flush()
                print(json.dumps({"records_accounted": len(records)}), flush=True)
    report = write_audit(records, OUTPUT, binding)
    report.update(binding)
    report.update(
        schema_version=1,
        audit_id="ssl_synthetic_input_audit_v1",
        started_at_utc=started,
        finished_at_utc=datetime.now(UTC).isoformat(),
        wall_seconds=time.perf_counter() - wall,
        cpu_seconds=time.process_time() - cpu,
        numerical_threads=1,
        waveform_cache_created=False,
        versions={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "wfdb")},
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        outputs={
            name: to_stored(OUTPUT / name) for name in ("records.jsonl", "eligibility.json", "summary.json")
        },
        not_verified=[
            "clinical normality or diagnostic labels",
            "biological identity and near duplicates",
            "generator qualification",
            "executable training protocol and actual GPU profile",
            "protected engineering-test and excluded waveform integrity/quality",
        ],
    )
    with (OUTPUT / "summary.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "summary": to_stored(OUTPUT / "summary.json"),
                "input_integrity_gate_passed": report["input_integrity_gate_passed"],
            }
        )
    )
    return 0 if report["input_integrity_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
