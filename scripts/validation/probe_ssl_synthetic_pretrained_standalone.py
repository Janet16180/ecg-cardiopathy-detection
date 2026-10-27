"""Run the reviewed standalone CPU-only public Pulse2Pulse feasibility attempt."""

import argparse
import json
from pathlib import Path

from ecg_experiment.ssl_synthetic_pretrained_standalone import run_probe


def main() -> None:
    """Require an explicit output repository root and print aggregate technical status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run_probe(args.output_root)
    print(
        json.dumps(
            {
                name: result[name]
                for name in [
                    "technical_generation_feasible",
                    "gates",
                    "cpu_seconds",
                    "wall_seconds",
                    "physical_units_verified",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
