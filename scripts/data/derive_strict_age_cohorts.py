"""Create strict 18–25 and same-size general cohorts from the completed admitted training pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecg_experiment.strict_age_cohorts import OUTPUT, PARENT, derive


def main() -> None:
    """Use a fixed explicit seed and write only a fresh derived-cohort directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parent", type=Path, default=PARENT)
    args = parser.parse_args()
    print(json.dumps(derive(args.input_root, args.output, args.seed, args.parent), indent=2))


if __name__ == "__main__":
    main()
