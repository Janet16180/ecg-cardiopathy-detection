"""Run the MLC-02 Stage 2 feature-based candidates on train and validation records only.

Logistic fits use the machine's default OpenBLAS kernel (``OPENBLAS_CORETYPE`` unset); networks run on GPU 1
with deterministic cuBLAS. No step reads a test role.

Examples
--------
PYTHONPATH=. .venv/bin/python -m scripts.experiments.run_mlc02_feature_candidates logistic --candidate P0 \
    --workers 8 --threads 2
PYTHONPATH=. .venv/bin/python -m scripts.experiments.run_mlc02_feature_candidates boosting \
    --workers 4 --threads 4
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=. .venv/bin/python \
    -m scripts.experiments.run_mlc02_feature_candidates network --candidate H-X --profile-epochs 3
PYTHONPATH=. .venv/bin/python -m scripts.experiments.run_mlc02_feature_candidates lofo --candidate P0
PYTHONPATH=. .venv/bin/python -m scripts.experiments.run_mlc02_feature_candidates source-id --threads 16
"""

from __future__ import annotations

import argparse
import json
import sys

from ecg_experiment.mlc02_feature_runs import (
    run_boosting,
    run_lofo,
    run_logistic,
    run_network,
    run_source_identification,
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
    parser.add_argument("step", choices=("logistic", "boosting", "network", "lofo", "source-id"))
    parser.add_argument("--candidate", choices=("P0", "L-J", "L-C", "H-X", "H-J", "H-C", "H-All"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--profile-epochs", type=int, help="network: profile this many epochs only")
    args = parser.parse_args(argv)
    command = sys.argv if argv is None else ["run_mlc02_feature_candidates", *argv]
    if args.step == "logistic":
        result = run_logistic(args.candidate, args.workers, args.threads, command)
    elif args.step == "boosting":
        result = run_boosting(args.workers, args.threads, command)
    elif args.step == "network":
        result = run_network(args.candidate, args.device, args.threads, args.profile_epochs, command)
    elif args.step == "lofo":
        result = run_lofo(args.candidate, args.workers, args.threads, args.device, command)
    else:
        result = run_source_identification(args.threads, command)
    summary = result.get("validation", result) if isinstance(result, dict) else result
    print(json.dumps(summary, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
