"""Run candidate F of MLC-02 (fine-tuned xECG) on train and validation records only.

Steps: ``prepare`` (CPU), then ``profile``, ``train`` and ``finalize`` on one GPU under its project lock. No
step reads a test role. ``--smoke`` runs on a few records into a scratch directory; its outputs are not
results.

Examples
--------
PYTHONPATH=. .venv/bin/python -m scripts.experiments.run_mlc02_finetune prepare --workers 8
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python -m scripts.experiments.run_mlc02_finetune profile \
    --gpu 0 --microbatch 8
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python -m scripts.experiments.run_mlc02_finetune train --gpu 0
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python -m scripts.experiments.run_mlc02_finetune finalize \
    --gpu 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ecg_experiment.mlc02_finetune import CANDIDATE, DIRECTORY
from ecg_experiment.mlc02_finetune_data import WORK
from ecg_experiment.mlc02_finetune_runs import (
    PROFILE_UPDATES,
    run_finalize,
    run_prepare,
    run_profile,
    run_train,
)


def main(argv: list[str] | None = None) -> None:
    """
    Parse the command line and run one step.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or None for ``sys.argv``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("prepare", "profile", "train", "finalize"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--microbatch", type=int, default=8)
    parser.add_argument("--updates", type=int, default=PROFILE_UPDATES)
    parser.add_argument("--smoke", type=Path, help="scratch directory of a smoke test on a few records")
    parser.add_argument("--smoke-train", type=int, default=256)
    parser.add_argument("--smoke-validation", type=int, default=1200)
    args = parser.parse_args(argv)
    command = sys.argv if argv is None else ["run_mlc02_finetune", *argv]
    limits = {"train": None, "validation": None}
    output, work = DIRECTORY, WORK
    if args.smoke is not None:
        limits = {"train": args.smoke_train, "validation": args.smoke_validation}
        output, work = args.smoke / "candidates" / CANDIDATE, args.smoke / "work"
    if args.step == "prepare":
        result = run_prepare(args.workers, command)
    elif args.step == "profile":
        result = run_profile(args.gpu, args.microbatch, args.updates, command, limits, output, work)
    elif args.step == "train":
        result = run_train(args.gpu, command, limits, output, work)
    else:
        result = run_finalize(args.gpu, command, limits, output, work)
    print(json.dumps(result, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
