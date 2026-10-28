"""Run the registered corrective study using explicit fresh raw-derived dependencies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from threadpoolctl import threadpool_limits

from ecg_experiment.pr74_fresh_baseline import OUTPUT as BASELINE
from ecg_experiment.pr74_raw_study import COHORTS, OUTPUT, STRICT, evaluate, extract, profile, verify


def main() -> None:
    """Keep cold inputs explicit and warm extraction reuse an opt-in recorded alternative."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--cohorts", type=Path, default=COHORTS)
    parser.add_argument("--strict", type=Path, default=STRICT)
    parser.add_argument("--reuse-training-features", type=Path)
    parser.add_argument("--stage", choices=("profile", "extract", "evaluate", "verify"), required=True)
    args = parser.parse_args()
    if args.reuse_training_features is not None and args.stage != "extract":
        parser.error("Warm extraction reuse applies only to the extraction stage")
    with threadpool_limits(limits=2):
        common = (args.input_root, args.output, args.cohorts, args.strict)
        if args.stage == "verify":
            result = verify(args.input_root, args.output)
        elif args.stage == "profile":
            result = profile(*common)
        elif args.stage == "extract":
            result = extract(*common, args.reuse_training_features)
        else:
            result = evaluate(*common, args.baseline)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
