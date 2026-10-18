"""Run the forward exact-accounting saved-map diagnostic once."""

import argparse
import json

from ecg_experiment.ssl_synthetic_paired_diagnostic_v2 import diagnose_paired_background_v2


def main() -> None:
    """Emit only aggregate execution status; all diagnostic work remains in the library."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    result = diagnose_paired_background_v2()
    print(json.dumps({"completed": result["diagnostic_completed"], "wall_seconds": result["wall_seconds"]}))


if __name__ == "__main__":
    main()
