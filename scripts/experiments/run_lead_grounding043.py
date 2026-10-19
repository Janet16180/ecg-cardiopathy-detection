"""Run the frozen finding-conditioned lead grounding comparison on CPU."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    """Set numerical limits before importing scientific libraries and execute the fixed study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/experiment043_lead_grounding_v1"))
    parser.add_argument("--protocol", type=Path, default=Path("docs/experiment-043-lead-grounding.md"))
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
    from ecg_experiment.lead_grounding043 import run_experiment

    result = run_experiment(args.output, args.protocol, args.threads)
    print(result["decision"]["outcome"])


if __name__ == "__main__":
    main()
