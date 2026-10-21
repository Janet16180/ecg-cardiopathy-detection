"""Execute the prospective pure-simulation intervention qualification grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecg_experiment import ROOT
from ecg_experiment.ssl_synthetic_generator_qualification import qualify_generator


def main() -> None:
    """Write aggregate qualification and local paired examples.

    Returns
    -------
    None
        Exit nonzero after retaining a failed gate receipt.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory", type=Path, default=ROOT / "outputs/ssl_synthetic_localization/generator_v1"
    )
    args = parser.parse_args()
    receipt = qualify_generator(args.output_directory)
    print(
        json.dumps(
            {
                "all_passed": receipt["all_passed"],
                "wall_seconds": receipt["wall_seconds"],
                "positive_count": receipt["positive_grid"]["count"],
                "negative_count": receipt["negative_grid"]["count"],
            }
        )
    )
    raise SystemExit(0 if receipt["all_passed"] else 1)


if __name__ == "__main__":
    main()
