"""Execute one prospectively bounded public-generator fit replication."""

import argparse
import json
from pathlib import Path

from ecg_experiment.ssl_synthetic_fit_replication_run import run_study


def main() -> None:
    """Parse the local output-host root and print only aggregate completion fields."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run_study(args.output_root)
    print(json.dumps({key: result[key] for key in ["all_passed", "scientific_passed", "contrasts"]}))


if __name__ == "__main__":
    main()
