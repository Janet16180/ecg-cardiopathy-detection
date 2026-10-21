"""Run the frozen saved-map-only control and phase diagnostic once."""

import argparse
import json

from ecg_experiment.ssl_synthetic_paired_diagnostic import diagnose_paired_background


def main() -> None:
    """Keep the runner thin and emit only aggregate execution status."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    result = diagnose_paired_background()
    print(json.dumps({"completed": result["diagnostic_completed"], "wall_seconds": result["wall_seconds"]}))


if __name__ == "__main__":
    main()
