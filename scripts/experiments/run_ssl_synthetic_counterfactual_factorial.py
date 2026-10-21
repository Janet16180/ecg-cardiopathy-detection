"""Profile or fit the frozen four-arm counterfactual-loss factorial."""

import argparse
import json

from ecg_experiment.ssl_synthetic_counterfactual_factorial_training import profile_factorial, train_factorial


def main() -> None:
    """Keep actual resource admission separate from the fixed scientific fits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("profile", "train"), required=True)
    stage = parser.parse_args().stage
    result = profile_factorial() if stage == "profile" else train_factorial()
    print(
        json.dumps(
            {
                "stage": stage,
                "wall_seconds": result["wall_seconds"],
                "passed": result["profile_gate_passed"]
                if stage == "profile"
                else result["training_completed"],
            }
        )
    )


if __name__ == "__main__":
    main()
