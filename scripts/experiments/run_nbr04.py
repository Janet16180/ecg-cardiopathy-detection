"""Run one NBR-04 stage: fitsets, profile, prepare, encode, score, analyze or export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from threadpoolctl import threadpool_limits

from ecg_experiment.cuda_runtime20261028 import configure_cuda_runtime
from ecg_experiment.localization import nbr04_export, nbr04_study

STAGES = ("fitsets", "profile", "prepare", "encode", "score", "analyze", "export")


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
        if args.stage in ("profile", "encode"):
            configure_cuda_runtime()
        if args.stage == "fitsets":
            result = nbr04_study.fitsets(root, args.workers)
        elif args.stage == "profile":
            result = nbr04_study.profile(root, args.device, args.workers)
        elif args.stage == "prepare":
            result = nbr04_study.prepare(root, args.workers)
        elif args.stage == "encode":
            result = nbr04_study.encode(root, args.device, args.shard, args.shards)
        elif args.stage == "score":
            result = nbr04_study.score(root)
        elif args.stage == "analyze":
            result = nbr04_study.analyze(root)
        else:
            result = nbr04_export.export(root)
    print(json.dumps(result["details"], indent=2, default=str))


if __name__ == "__main__":
    main()
