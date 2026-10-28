"""Run one NBR-03 stage: checks, profile, select, prepare, encode, parity, score, analyze or export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from threadpoolctl import threadpool_limits

from ecg_experiment.cuda_runtime20261028 import configure_cuda_runtime
from ecg_experiment.localization import nbr03_export, nbr03_study

STAGES = ("checks", "profile", "select", "prepare", "encode", "parity", "score", "analyze", "export")


def main() -> None:
    """Parse the stage and run it with at most two numerical threads in the parent process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    root = args.input_root
    with threadpool_limits(limits=2):
        if args.stage in ("profile", "encode", "parity"):
            configure_cuda_runtime()
        if args.stage == "checks":
            result = nbr03_study.checks(root, args.workers)
        elif args.stage == "profile":
            result = nbr03_study.profile(root, args.device, args.workers)
        elif args.stage == "select":
            result = nbr03_study.select(root)
        elif args.stage == "prepare":
            result = nbr03_study.prepare(root, args.workers)
        elif args.stage == "encode":
            result = nbr03_study.encode(root, args.device, args.shard, args.shards)
        elif args.stage == "parity":
            result = nbr03_study.parity(root, args.device)
        elif args.stage == "score":
            result = nbr03_study.score(root)
        elif args.stage == "analyze":
            result = nbr03_study.analyze(root)
        else:
            result = nbr03_export.export(root)
    print(json.dumps(result["details"], indent=2, default=str))


if __name__ == "__main__":
    main()
