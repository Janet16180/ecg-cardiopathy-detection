"""Recover mechanical canonical hash references without model fitting or performance evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecg_experiment.heldout_window_recovery import REFERENCE_ROOT, recover


def main() -> None:
    """Require completed original split recovery and write a new reference artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(recover(args.input_root, args.output, args.workers, args.limit), indent=2))


if __name__ == "__main__":
    main()
