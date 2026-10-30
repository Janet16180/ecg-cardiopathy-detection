"""Run CLF-01: integrity check, fixed classifier fits and external SPH comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from threadpoolctl import threadpool_limits

from ecg_experiment.clf01_study import OUTPUT, run


def main() -> None:
    """Parse the input root and run the study with at most two numerical threads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        result = run(args.input_root, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
