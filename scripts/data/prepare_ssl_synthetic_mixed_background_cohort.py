"""Declare the fixed mixed-background supervised cohort using metadata only."""

import argparse
from pathlib import Path

from ecg_experiment.ssl_synthetic_mixed_background_cohort import prepare


def main() -> None:
    """Write one local prospective cohort manifest without selected waveform access."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.root, args.destination))


if __name__ == "__main__":
    main()
