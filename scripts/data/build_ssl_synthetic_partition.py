"""Write a local patient manifest and an aggregate SSL localization partition receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from ecg_experiment import ROOT
from ecg_experiment.paths import to_stored
from ecg_experiment.ssl_synthetic_partition import build_partition


def main() -> None:
    """
    Build the frozen metadata-only partition, refusing to overwrite either output.

    Returns
    -------
    None
        Writes a local record-level manifest and prints aggregate counts only.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=ROOT / "data/raw/ptb-xl/1.0.3/ptbxl_database.csv")
    parser.add_argument("--policy", type=Path, default=ROOT / "configs/ssl_synthetic_partition_v1.json")
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data/processed/ssl_synthetic_localization_v1/partition.json"
    )
    parser.add_argument(
        "--summary", type=Path, default=ROOT / "outputs/ssl_synthetic_localization/partition_summary.json"
    )
    args = parser.parse_args()
    stored = {key: to_stored(value) for key, value in vars(args).items()}
    if args.manifest.absolute() == args.summary.absolute():
        raise ValueError("Manifest and summary must use different paths")
    for path in (args.manifest, args.summary):
        if path.exists() or path.is_symlink():
            raise FileExistsError("Partition outputs must be new; no output was overwritten")
    policy_bytes = args.policy.read_bytes()
    manifest = build_partition(args.metadata.read_bytes(), json.loads(policy_bytes), args.metadata.parent)
    sources = {
        stored["policy"]: hashlib.sha256(policy_bytes).hexdigest(),
        **{
            to_stored(ROOT / relative): hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "ecg_experiment/ssl_synthetic_partition.py",
                "scripts/data/build_ssl_synthetic_partition.py",
                "docs/ssl-synthetic-localization-partition.md",
            )
        },
    }
    manifest["source_sha256"] = sources
    manifest["metadata"] = stored["metadata"]
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode()
    summary = {key: value for key, value in manifest.items() if key != "records"}
    summary.update(
        created_at_utc=datetime.now(UTC).isoformat(),
        manifest=stored["manifest"],
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        scope="Metadata patient separation and eligible path existence only; no waveform payload opened",
    )
    for path in (args.manifest, args.summary):
        path.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("xb") as stream:
        stream.write(manifest_bytes)
    with args.summary.open("x") as stream:
        stream.write(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"summary": stored["summary"], "counts": summary["counts"], "training_admitted": False}))


if __name__ == "__main__":
    main()
