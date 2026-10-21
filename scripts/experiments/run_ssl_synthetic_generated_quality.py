"""Run the frozen CUDA public-generator background qualification."""

import argparse
import json

from ecg_experiment.ssl_synthetic_generated_quality_run import run_study


def main() -> None:
    """Run one immutable study and print its aggregate operational result."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    result = run_study()
    print(
        json.dumps(
            {
                key: result[key]
                for key in [
                    "operational_support_qualified",
                    "primary",
                    "cpu_seconds",
                    "wall_seconds",
                    "gpu_lock_wall_seconds",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
