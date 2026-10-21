"""Evaluate the qualified and profiled matched-control real-background training study."""

import argparse
import json

from ecg_experiment.paths import to_stored
from ecg_experiment.ssl_synthetic_paired_background_evaluation import EVALUATION, evaluate_paired_background


def main() -> None:
    """Execute the fixed R/S evaluation once, printing only aggregate completion and decision."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    report = evaluate_paired_background()
    print(
        json.dumps(
            {
                "summary": to_stored(EVALUATION / "summary.json"),
                "evaluation_completed": report["evaluation_completed"],
                "primary": report["s_minus_r_response"]["all"],
                "decision": report["decision"],
            }
        )
    )


if __name__ == "__main__":
    main()
