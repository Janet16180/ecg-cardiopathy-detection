"""Run one HEADS-01 stage: qualify, smoke, profile, extract, evaluate or export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from threadpoolctl import threadpool_limits

from ecg_experiment.cuda_runtime20261028 import configure_cuda_runtime
from ecg_experiment.heads01_study import OUTPUT, evaluate, export, extract, profile, qualify, smoke


def main() -> None:
    """Parse the stage and run it with at most two numerical threads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--stage", required=True,
                        choices=("qualify", "smoke", "profile", "extract", "evaluate", "export"))
    parser.add_argument("--part", choices=("training", "evaluation"))
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.stage in ("profile", "extract") and args.part is None:
        parser.error("--part is required for profile and extract")
    with threadpool_limits(limits=2):
        if args.stage in ("smoke", "profile", "extract"):
            configure_cuda_runtime()
        if args.stage == "qualify":
            result = qualify(args.input_root, args.output)
        elif args.stage == "smoke":
            result = smoke(args.input_root, args.output, args.device)
        elif args.stage == "profile":
            result = profile(args.input_root, args.output, args.part, args.device)
        elif args.stage == "extract":
            result = extract(args.input_root, args.output, args.part, args.device)
        elif args.stage == "evaluate":
            result = evaluate(args.input_root, args.output)
        else:
            result = export(args.input_root, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "source_sha256"}, indent=2,
                     default=str))


if __name__ == "__main__":
    main()
