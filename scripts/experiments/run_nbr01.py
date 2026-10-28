"""Run one NBR-01 stage: counterpart, profile, select, prepare, encode, score, analyze or export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from threadpoolctl import threadpool_limits

from ecg_experiment.cuda_runtime20261028 import configure_cuda_runtime
from ecg_experiment.localization import nbr01_export, nbr01_study


def main() -> None:
    """Parse the stage and run it with at most two numerical threads in the parent process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--stage", required=True, choices=("counterpart", "profile", "select", "prepare",
                                                           "encode", "parity", "score", "analyze", "export"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        if args.stage in ("profile", "encode", "parity"):
            configure_cuda_runtime()
        if args.stage == "counterpart":
            result = nbr01_study.counterpart(args.input_root, args.workers)
        elif args.stage == "profile":
            result = nbr01_study.profile(args.input_root, args.device, args.workers)
        elif args.stage == "select":
            result = nbr01_study.select(args.input_root)
        elif args.stage == "prepare":
            result = nbr01_study.prepare(args.input_root, args.workers)
        elif args.stage == "encode":
            result = nbr01_study.encode(args.input_root, args.device, args.shard, args.shards)
        elif args.stage == "parity":
            result = nbr01_study.parity(args.input_root, args.device)
        elif args.stage == "score":
            result = nbr01_study.score(args.input_root)
        elif args.stage == "analyze":
            result = nbr01_study.analyze(args.input_root)
        else:
            result = nbr01_export.export(args.input_root)
    shown = {key: value for key, value in result.items() if key not in ("source_sha256", "artifacts")}
    print(json.dumps(shown, indent=2, default=str))


if __name__ == "__main__":
    main()
