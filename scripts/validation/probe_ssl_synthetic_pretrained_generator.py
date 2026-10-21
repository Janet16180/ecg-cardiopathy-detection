"""Run the source-frozen CPU-only Pulse2Pulse technical feasibility probe."""

import argparse
import json

from ecg_experiment.ssl_synthetic_pretrained_generator import run_probe


def main() -> None:
    """Print aggregate technical status only."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    result = run_probe()
    print(
        json.dumps(
            {
                "technical_generation_feasible": result["technical_generation_feasible"],
                "gates": result["gates"],
                "cpu_seconds": result["cpu_seconds"],
                "physical_units_verified": result["physical_units_verified"],
            }
        )
    )


if __name__ == "__main__":
    main()
