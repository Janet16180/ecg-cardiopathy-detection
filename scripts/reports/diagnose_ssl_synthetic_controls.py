"""Execute one source-frozen saved-map background-hotspot diagnosis."""

import argparse
import json
from pathlib import Path

from ecg_experiment.ssl_synthetic_control_diagnosis import run_diagnosis


def main() -> None:
    """Require the local output repository root and print aggregate completion status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run_diagnosis(args.output_root)
    print(json.dumps({key: result[key] for key in ["integrity_completed", "primary", "case_rows"]}))


if __name__ == "__main__":
    main()
