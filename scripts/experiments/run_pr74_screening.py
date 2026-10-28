"""Run an immutable stage of corrective development screening evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from threadpoolctl import threadpool_limits

from ecg_experiment.pr74_screening_features import extract, profile
from ecg_experiment.pr74_screening_io import OUTPUT
from ecg_experiment.pr74_screening_replay import qualify


def main() -> None:
    """Require explicit local inputs and keep training/evaluation behind predecessor and cohort gates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--stage", choices=("qualify", "profile", "extract", "evaluate"), required=True)
    args = parser.parse_args()
    if args.stage == "evaluate":
        from ecg_experiment.pr74_screening_study import evaluate

        runner = evaluate
    else:
        runner = {"qualify": qualify, "profile": profile, "extract": extract}[args.stage]
    with threadpool_limits(limits=2):
        result = runner(args.input_root, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
