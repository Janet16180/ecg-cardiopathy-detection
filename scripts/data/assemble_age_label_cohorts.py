"""Assemble new label-first younger and all-age supervised training cohorts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecg_experiment.age_cohort_assembly import OUTPUT, assemble


def main() -> None:
    """Require an explicit shared input root and refuse incomplete recovery evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--size", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(assemble(args.input_root, args.output, args.size, args.seed), indent=2))


if __name__ == "__main__":
    main()
