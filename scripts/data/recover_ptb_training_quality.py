"""Recover PTB-XL training-only quality from verified local raw files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecg_experiment.ptb_training_recovery import RECOVERY, recover


def main() -> None:
    """Parse recovery arguments and print the executed receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=RECOVERY / "quality")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(recover(args.input_root, args.output, args.workers, args.limit), indent=2))


if __name__ == "__main__":
    main()
