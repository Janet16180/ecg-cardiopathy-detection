"""Execute one frozen Experiment 039 encoder/context factorial cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecg_experiment import ROOT
from ecg_experiment import encoder_context_study039 as study


def main() -> None:
    """Parse and execute one stage or one complete matched-context cell."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=study.STAGES + ("run",), required=True)
    parser.add_argument("--tier", type=int, choices=study.TIERS, required=True)
    parser.add_argument("--encoder", choices=study.ENCODERS, required=True)
    parser.add_argument("--seed", type=int, choices=study.SEEDS, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.stage == "run":
        result = study.run(args.root, args.tier, args.encoder, args.seed, args.device)
    else:
        result = study.execute(args.stage, args.root, args.tier, args.encoder, args.seed, args.device)
    print(json.dumps({"stage": args.stage, "tier": args.tier, "encoder": args.encoder,
                      "seed": args.seed, "result": result}, default=str), flush=True)


if __name__ == "__main__":
    main()
