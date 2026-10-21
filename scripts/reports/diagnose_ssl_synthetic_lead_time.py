"""Run the frozen saved-map lead/time diagnostic with no model calls."""

import argparse
import json
from pathlib import Path

from ecg_experiment.ssl_synthetic_lead_time_diagnostic import run_study


def main() -> None:
    """Parse the explicit local output-host root and display descriptive contrasts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run_study(args.output_root)
    print(
        json.dumps(
            {
                "technical_passed": result["technical_passed"],
                "descriptive_contrasts": result["descriptive_contrasts"],
            }
        )
    )


if __name__ == "__main__":
    main()
