"""Execute one source-frozen generated-background suppression factorial."""

import argparse
import json
from pathlib import Path

from ecg_experiment.ssl_synthetic_paired_suppression_run import run_study


def main() -> None:
    """Read the local output host and print the aggregate completion verdict."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run_study(args.output_root)
    print(
        json.dumps(
            {
                "all_passed": result["all_passed"],
                "scientific_passed": result["aggregate"]["scientific_passed"],
            }
        )
    )


if __name__ == "__main__":
    main()
