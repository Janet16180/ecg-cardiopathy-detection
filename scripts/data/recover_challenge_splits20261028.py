"""Mechanically reconstruct the original Challenge split, without model evaluation."""

import argparse
import json
from pathlib import Path

from ecg_experiment.challenge_recovery20261028 import recover


def main() -> None:
    """Parse explicit source and output paths and run the recovery."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--profile-records", type=int, default=0)
    args = parser.parse_args()
    result = recover(args.source_root, args.output_dir, args.workers, args.profile_records)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
