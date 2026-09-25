"""Run the immutable CPU-only compact-CPC local-readout study."""

from __future__ import annotations

import argparse
from pathlib import Path

from ecg_experiment.cpc_local_readout import run


def main() -> None:
    """Parse the frozen manifest identity and launch the study."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    run(root, args.manifest, args.manifest_sha256)


if __name__ == "__main__":
    main()
