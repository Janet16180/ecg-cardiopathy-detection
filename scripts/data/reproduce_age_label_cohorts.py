"""Reproduce verified recovery profiles, cleaning, references and fixed-seed age cohorts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecg_experiment.cohort_reproduction import DERIVED_OUTPUT, OUTPUT, reproduce, stages


def main() -> None:
    """Run or inspect the full bounded-CPU local reconstruction dependency graph."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--size", type=int, default=25000)
    parser.add_argument("--derived-output", type=Path, default=DERIVED_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    function = stages if args.plan else reproduce
    result = function(args.input_root, args.output, args.size, args.seed, args.workers, args.derived_output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
