"""Reproduce all cohort data stages from local raw releases, without historical scientific artifacts."""

import argparse
import json
from pathlib import Path

from ecg_experiment.cold_start_pipeline import plan, reproduce


def main() -> None:
    """Inspect or execute the raw-only dependency graph with fixed scientific seeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-verified-caches", action="store_true")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    function = plan if args.plan else reproduce
    result = function(args.input_root, args.workers, args.seed, args.reuse_verified_caches)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
