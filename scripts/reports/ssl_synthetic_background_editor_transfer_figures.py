"""Render only fixed saved background/editor examples under the coordinator's admitted wrapper."""

import argparse
from pathlib import Path

from ecg_experiment.ssl_synthetic_background_editor_transfer_figures import plot


def main() -> None:
    """Read the declared main output and save the two immutable fixed-example figures.

    Returns
    -------
    None
        Prints the new local PNG basenames; performs no model inference or metric computation.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Completed main-study output directory")
    parser.add_argument("--destination", type=Path, required=True)
    arguments = parser.parse_args()
    print(plot(arguments.root, arguments.destination))


if __name__ == "__main__":
    main()
