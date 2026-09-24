"""Verify preserved scientific sources against the user-requested pause archive.

This is a CPU-only provenance check. It neither trains nor schedules experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(root: Path) -> dict:
    pause_dir = root / "outputs/refactor_pause"
    pause = json.loads((pause_dir / "pause.json").read_text())
    source_hashes = json.loads((pause_dir / "source_hashes.json").read_text())
    archive = root / pause["source_archive"]
    if sha256(archive) != pause["source_archive_sha256"]:
        raise ValueError("Historical source archive hash mismatch")
    archived_count = 0
    with tarfile.open(archive, "r:gz") as handle:
        for name, expected in source_hashes.items():
            member = handle.extractfile(name)
            if member is None or hashlib.sha256(member.read()).hexdigest() != expected:
                raise ValueError(f"Archived source mismatch: {name}")
            archived_count += 1
    checked = []
    archived_environments = []
    archive_names = {"pyproject.toml": "pyproject.pre-refactor.toml",
                     "uv.lock": "uv.pre-refactor.lock"}
    for name, expected in source_hashes.items():
        if name.startswith(("ecg_experiment/", "scripts/", "tests/")):
            if sha256(root / name) != expected:
                raise ValueError(f"Frozen scientific source changed: {name}")
            checked.append(name)
        elif name in archive_names or name.startswith("requirements"):
            saved = root / "environments/archive" / archive_names.get(name, name)
            if sha256(saved) != expected:
                raise ValueError(f"Historical environment receipt changed: {name}")
            archived_environments.append(str(saved.relative_to(root)))
    queue = json.loads((root / "docs/experiment-queue.json").read_text())
    if queue["scheduling"]["state"] != "paused_by_user":
        raise ValueError("This maintenance verification requires the experiment pause")
    return {
        "status": "verified",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "repository_maintenance_not_performance_evaluation",
        "source_archive_sha256": pause["source_archive_sha256"],
        "archive_entries_verified": archived_count,
        "unchanged_scientific_sources": checked,
        "archived_environment_receipts": archived_environments,
        "training_launched": False,
        "scheduling_state": "paused_by_user",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/repository_refactor/verification.json"))
    args = parser.parse_args()
    result = verify(ROOT)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "receipt": str(output),
                      "scientific_sources": len(result["unchanged_scientific_sources"])}))


if __name__ == "__main__":
    main()
