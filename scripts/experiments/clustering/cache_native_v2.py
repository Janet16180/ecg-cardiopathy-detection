"""Execute the canonical-lead-compatible CLUSTER-03 profile or cache extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecg_experiment import ROOT
from ecg_experiment.clustering.native_cache_v2 import run_stage


def main() -> None:
    """Parse a local stage request and delegate all execution to the reusable module."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("profile", "extract"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    args = parser.parse_args()
    if args.stage == "extract" and args.profile is None:
        parser.error("extraction requires a completed --profile result.json")
    output = args.output if args.output.is_absolute() else ROOT / args.output
    profile_path = args.profile
    if profile_path is not None and not profile_path.is_absolute():
        profile_path = ROOT / profile_path
    result = run_stage(args.stage, output, profile_path)
    print(json.dumps({"stage": args.stage, "status": "completed", "seconds": result["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
