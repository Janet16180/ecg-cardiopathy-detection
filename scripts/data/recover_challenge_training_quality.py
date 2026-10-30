"""Stage Challenge training views and conservative supervised label eligibility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecg_experiment.challenge_training_recovery import OUTPUT, recover


def main() -> None:
    """Parse arguments; the recovery refuses any unverified original split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(recover(args.input_root, args.output, args.workers, args.limit), indent=2))


if __name__ == "__main__":
    main()
