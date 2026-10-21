"""Qualify every fixed pilot seed and condition before model training."""

from __future__ import annotations

import argparse
import json

from ecg_experiment import ROOT
from ecg_experiment.ssl_synthetic_generator_pilot_qualification import qualify_pilot_domain


def main() -> None:
    """Retain full admission/failure receipts for the declared generator version.

    Returns
    -------
    None
        Exit nonzero after retaining any failed admission.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", choices=("v1", "v2"), default="v1")
    args = parser.parse_args()
    output = ROOT / f"outputs/ssl_synthetic_localization/generator_pilot_{args.generator}"
    receipt = qualify_pilot_domain(output, args.generator)
    print(
        json.dumps(
            {
                "all_passed": receipt["all_passed"],
                "count": receipt["case_admission"]["count"],
                "failed_cases": receipt["case_admission"]["failed_cases"],
                "wall_seconds": receipt["wall_seconds"],
            }
        )
    )
    raise SystemExit(0 if receipt["all_passed"] else 1)


if __name__ == "__main__":
    main()
