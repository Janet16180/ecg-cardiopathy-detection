"""Check declared signal duplicates while keeping evaluation waveform payloads unread."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from ecg_experiment import ROOT
from ecg_experiment.paths import to_stored
from ecg_experiment.ssl_synthetic_checksum_audit import audit_duplicates, read_checksums, verify_partition
from ecg_experiment.ssl_synthetic_readiness import checked_relative


def main() -> int:
    """
    Verify the frozen partition and write a new aggregate checksum audit receipt.

    Returns
    -------
    int
        Zero for a passing duplicate gate, two for cross-patient/role duplicates.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partition-summary",
        type=Path,
        default=ROOT / "outputs/ssl_synthetic_localization/partition_summary.json",
    )
    parser.add_argument("--checksums", type=Path, default=ROOT / "data/raw/ptb-xl/1.0.3/SHA256SUMS.txt")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/ssl_synthetic_localization/partition_checksum_audit.json",
    )
    args = parser.parse_args()
    stored = {key: to_stored(value) for key, value in vars(args).items()}
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Audit output must be new")
    summary_bytes = args.partition_summary.read_bytes()
    summary = json.loads(summary_bytes)
    manifest_path = ROOT / checked_relative(summary["manifest"])
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    metadata_path = ROOT / checked_relative(manifest["metadata"])
    verify_partition(manifest, summary, manifest_bytes, metadata_path.read_bytes())
    for relative, expected in manifest["source_sha256"].items():
        if hashlib.sha256((ROOT / checked_relative(relative)).read_bytes()).hexdigest() != expected:
            raise ValueError("A frozen partition source changed")
    checksum_bytes = args.checksums.read_bytes()
    checksums = read_checksums(checksum_bytes.decode())
    if checksums["ptbxl_database.csv"] != manifest["metadata_sha256"]:
        raise ValueError("Partition metadata differs from official checksum")
    report = audit_duplicates(manifest["records"], checksums)
    report.update(
        created_at_utc=datetime.now(UTC).isoformat(),
        partition_summary=stored["partition_summary"],
        partition_summary_sha256=hashlib.sha256(summary_bytes).hexdigest(),
        manifest=summary["manifest"],
        manifest_sha256=summary["manifest_sha256"],
        checksums=stored["checksums"],
        checksums_sha256=hashlib.sha256(checksum_bytes).hexdigest(),
        metadata_sha256=manifest["metadata_sha256"],
        source_sha256={
            to_stored(ROOT / relative): hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "ecg_experiment/ssl_synthetic_checksum_audit.py",
                "scripts/validation/audit_ssl_synthetic_checksums.py",
                "docs/ssl-synthetic-localization-checksum-audit.md",
                "ecg_experiment/ssl_synthetic_partition.py",
                "ecg_experiment/ssl_synthetic_readiness.py",
            )
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": stored["output"], "duplicate_gate_passed": report["duplicate_gate_passed"]}))
    return 0 if report["duplicate_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
