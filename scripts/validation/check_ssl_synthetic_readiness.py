"""Write a payload-free availability report for SSL and synthetic localization work."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from ecg_experiment import ROOT
from ecg_experiment.paths import to_stored
from ecg_experiment.ssl_synthetic_readiness import inspect_prerequisites, load_spec


def main() -> int:
    """
    Inspect prerequisite paths and save a new local report.

    Returns
    -------
    int
        Zero when required paths exist, two when incomplete. Invalid configuration raises.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=ROOT / "configs/ssl_synthetic_readiness.json")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/ssl_synthetic_localization/readiness.json"
    )
    args = parser.parse_args()
    specification = to_stored(args.spec)
    output = to_stored(args.output)
    report = inspect_prerequisites(load_spec(args.spec))
    report["created_at_utc"] = datetime.now(UTC).isoformat()
    report["specification"] = specification
    sources = [
        args.spec,
        ROOT / "ecg_experiment/ssl_synthetic_readiness.py",
        ROOT / "scripts/validation/check_ssl_synthetic_readiness.py",
    ]
    report["source_sha256"] = {
        to_stored(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": output, "missing_required": report["missing_required"]}))
    return 0 if report["required_paths_present"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
