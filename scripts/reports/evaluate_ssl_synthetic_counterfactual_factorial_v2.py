"""Evaluate the frozen factorial using its explicit CPU scoring correction."""

import argparse
import json

from ecg_experiment.ssl_synthetic_counterfactual_factorial_evaluation_v2 import evaluate_factorial_v2


def main() -> None:
    """Emit aggregate execution status and the unchanged twelve-gate decision."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    result = evaluate_factorial_v2()
    print(
        json.dumps(
            {
                "completed": result["evaluation_completed"],
                "wall_seconds": result["wall_seconds"],
                "decision": result["decision"],
            }
        )
    )


if __name__ == "__main__":
    main()
