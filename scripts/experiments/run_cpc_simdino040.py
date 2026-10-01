"""Execute one accounted stage of the prospective CPC/SimDINO objective study."""

from __future__ import annotations

import argparse
import json

from ecg_experiment import ROOT
from ecg_experiment import simdino_study040 as study


def main() -> None:
    """Run the selected fixed-objective stage with its original receipt identity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=study.STAGES)
    parser.add_argument("--objective", required=True, choices=study.OBJECTIVES)
    parser.add_argument("--seed", required=True, type=int, choices=study.SEEDS)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    print(json.dumps(study.execute(ROOT, args.objective, args.seed, args.stage, args.device), indent=2))


if __name__ == "__main__":
    main()
