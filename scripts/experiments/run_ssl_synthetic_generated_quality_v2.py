"""Execute one reviewed prospective generated-quality v2 attempt."""

import argparse
import json
from pathlib import Path

from ecg_experiment.ssl_synthetic_generated_quality_v2 import run_study


def main() -> None:
    """Require the explicit local output repository root and print aggregate status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run_study(args.output_root)
    print(json.dumps({key: result[key] for key in ["operational_support_qualified", "primary", "gates"]}))


if __name__ == "__main__":
    main()
