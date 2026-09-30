"""Freeze one verified, sequential NLP-inspired 25k experiment queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.provenance import utc_now
from scripts.coordination.run_priority_queue import coordinator_sha256

ROOT = Path(__file__).resolve().parents[2]


def _stage(value: str) -> dict[str, object]:
    """Parse ``name:module:stage`` into an explicit queue command."""
    parts = value.split(":")
    if len(parts) != 3 or not parts[1].startswith("scripts.") or not all(parts):
        raise ValueError(f"Invalid stage specification: {value}")
    name, module, action = parts
    return {
        "name": name,
        "command": [str(ROOT / ".venv/bin/python"), "-u", "-m", module,
                    "--stage", action],
    }


def _relative_path(value: str) -> str:
    """Require a repository-relative source or artifact path."""
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Expected repository-relative path: {value}")
    return path.as_posix()


def main() -> None:
    """Write immutable source and queue JSON files in a new output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--predecessor-status", required=True)
    parser.add_argument("--stage", action="append", required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--artifact", action="append", required=True)
    args = parser.parse_args()

    directory = ROOT / _relative_path(str(args.directory))
    sources_path = directory / "sources.json"
    queue_path = directory / "queue.json"
    if sources_path.exists() or queue_path.exists():
        raise FileExistsError("Queue files already exist; use a new versioned directory")
    predecessor_path = ROOT / _relative_path(args.predecessor_status)
    predecessor = json.loads(predecessor_path.read_text())
    if predecessor.get("state") != "complete" or predecessor.get("returncode") != 0:
        raise ValueError("Predecessor queue did not complete")

    sources = {_relative_path(name): sha256_file(ROOT / _relative_path(name))
               for name in args.source}
    artifacts = [_relative_path(name) for name in args.artifact]
    stages = [_stage(value) for value in args.stage]
    directory.mkdir(parents=True, exist_ok=True)
    write_json_atomic(sources_path, dict(sorted(sources.items())))
    manifest = {
        "created_at": utc_now(),
        "reason": args.reason,
        "coordinator_sha256": coordinator_sha256(),
        "predecessor": {
            "pid": 0,
            "identity": None,
            "status_path": _relative_path(args.predecessor_status),
        },
        "legacy": None,
        "jobs": [{
            "name": args.name,
            "output_dir": (directory / "job").relative_to(ROOT).as_posix(),
            "sources": sources_path.relative_to(ROOT).as_posix(),
            "sources_sha256": sha256_file(sources_path),
            "stages": stages,
            "required_artifacts": artifacts,
        }],
    }
    write_json_atomic(queue_path, manifest)
    print(json.dumps({"manifest": queue_path.relative_to(ROOT).as_posix(),
                      "manifest_sha256": sha256_file(queue_path),
                      "sources_sha256": sha256_file(sources_path)}), flush=True)


if __name__ == "__main__":
    main()
