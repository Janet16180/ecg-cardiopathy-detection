"""Audit saved Experiment 041 and 042 arithmetic without fits or inference."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    """Parse saved artifact locations and run one exclusive arithmetic audit."""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    from ecg_experiment.measurement_screening_audit import run_audit

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment041", required=True, type=Path)
    parser.add_argument("--experiment042", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run_audit(args.experiment041, args.experiment042, args.output)
    print(f"{result['status']}: {result['check_count']} checks; {len(result['mismatches'])} mismatches")
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
