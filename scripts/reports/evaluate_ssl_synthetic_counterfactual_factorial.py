"""Run only the frozen four-arm counterfactual-loss evaluation."""

import argparse
import json

from ecg_experiment.ssl_synthetic_counterfactual_factorial_evaluation import evaluate_factorial


def main() -> None:
    """Emit aggregate execution status and the unchanged prospective conjunction."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    result = evaluate_factorial()
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
