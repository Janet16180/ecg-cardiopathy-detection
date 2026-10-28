"""Select new cohort receipts using raw administrative roles and cold reference provenance."""

import argparse
import json
from pathlib import Path

from ecg_experiment.cold_cohort_assembly import assemble, strict


def main() -> None:
    """Select either 25k cohorts or their strict source/evidence/target matched derivative."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["cohorts", "strict"])
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    function = assemble if args.stage == "cohorts" else strict
    print(json.dumps(function(args.input_root, args.seed), indent=2))


if __name__ == "__main__":
    main()
