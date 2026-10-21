"""Qualify the frozen real-background introduced-voltage data without model inference or training."""

from __future__ import annotations

import argparse
import json
import os

from ecg_experiment.paths import to_stored
from ecg_experiment.ssl_synthetic_real_background_preparation import OUTPUT, prepare


def main() -> int:
    """Run the new fixed qualification once with one numerical CPU thread and hidden CUDA."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    required = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    if any(os.environ.get(name) != "1" for name in required) or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Use one numerical CPU thread and hide CUDA")
    report = prepare()
    print(
        json.dumps(
            {
                "summary": to_stored(OUTPUT / "summary.json"),
                "qualification_gate_passed": report["qualification_gate_passed"],
                "backgrounds": report["backgrounds"],
                "cases": report["cases"],
                "failed_cases": report["failed_cases"],
            }
        )
    )
    return 0 if report["qualification_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
