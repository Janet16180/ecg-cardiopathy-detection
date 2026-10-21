"""Profile or fit only the prospectively frozen paired real-background localizers."""

import argparse
import json

from ecg_experiment.ssl_synthetic_paired_background_training import (
    profile_paired_background,
    train_paired_background,
)


def main() -> None:
    """Keep actual resource admission and scientific fitting as separately authorized stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("profile", "train"), required=True)
    stage = parser.parse_args().stage
    result = profile_paired_background() if stage == "profile" else train_paired_background()
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
