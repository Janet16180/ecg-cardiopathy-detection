"""Run the committed randomized-support generated-background learning study once."""

import argparse
import json
from pathlib import Path

from ecg_experiment.ssl_synthetic_randomized_localization_run import run_study


def main() -> None:
    """Require the local output repository root and print aggregate scientific verdict."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run_study(args.output_root)
    print(json.dumps({key: result[key] for key in ["all_passed", "scientific_gates", "integrity_gates"]}))


if __name__ == "__main__":
    main()
