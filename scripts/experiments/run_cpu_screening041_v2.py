"""Run Experiment 041 amended execution with CPU thread limits set before scientific imports."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    """Parse the output location and fixed CPU budget, then run the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/experiment041_cpu_screening_v2"))
    parser.add_argument("--protocol", type=Path, default=Path("docs/experiment-041-cpu-screening.md"))
    parser.add_argument(
        "--amendment", type=Path, default=Path("docs/experiment-041-cpu-screening-execution-v2.md")
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.threads <= 4:
        parser.error("--threads must be between 1 and 4")
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[name] = str(args.threads)
    from ecg_experiment.cpu_screening041_v2 import run_experiment

    result = run_experiment(args.output, args.protocol, args.amendment, args.threads)
    print(result["decision"]["outcome"])


if __name__ == "__main__":
    main()
