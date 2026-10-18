"""Run only the prospectively frozen original pilot after measured resource admission."""

import argparse
import json

from ecg_experiment.ssl_synthetic_training import train


def main() -> None:
    """Execute all fixed fits, keeping evaluation as a separate calibrated stage."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    receipt = train()
    print(
        json.dumps(
            {"training_completed": receipt["training_completed"], "wall_seconds": receipt["wall_seconds"]}
        )
    )


if __name__ == "__main__":
    main()
