"""Run the raw-only scientific baseline bootstrap without any old score or feature artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from threadpoolctl import threadpool_limits

from ecg_experiment.pr74_fresh_baseline import OUTPUT, ROLES, features, fit, profile


def main() -> None:
    """Require explicit local raw inputs and completed fresh administrative roles."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--roles", type=Path, default=ROLES)
    parser.add_argument("--device", type=int, choices=(0, 1), default=0)
    parser.add_argument("--stage", choices=("profile", "features", "fit"), required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        result = (fit(args.input_root, args.output) if args.stage == "fit" else
                  {"profile": profile, "features": features}[args.stage](
                      args.input_root, args.output, args.roles, args.device))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
