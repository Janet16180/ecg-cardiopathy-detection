"""Build fresh PTB/SPH administrative role tables without historical scientific-stage prerequisites."""

import argparse
import json
from pathlib import Path

from ecg_experiment.raw_role_bootstrap import OUTPUT, bootstrap


def main() -> None:
    """Run a raw resource profile or publish full administrative evidence under a new receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--profile-limit", type=int)
    args = parser.parse_args()
    print(json.dumps(bootstrap(args.input_root, args.output, args.workers, args.profile_limit), indent=2))


if __name__ == "__main__":
    main()
