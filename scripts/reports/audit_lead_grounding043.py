"""Verify saved grounding arithmetic without models, inference or annotation parsing."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    """Parse the existing output directory and write one exclusive independent audit."""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    from ecg_experiment.lead_grounding_audit043 import run_audit

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=Path("outputs/experiment043_lead_grounding_v1"))
    parser.add_argument("--output", type=Path, default=Path("outputs/experiment043_lead_grounding_audit_v1"))
    args = parser.parse_args()
    result = run_audit(args.experiment, args.output)
    print(f"{result['status']}: {result['check_count']} checks; {len(result['mismatches'])} mismatches")
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
