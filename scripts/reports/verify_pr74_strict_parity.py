"""Verify every saved PR74 evaluation payload and the complete primary comparison exactly."""

import argparse
import json
from pathlib import Path

from ecg_experiment import ROOT
from ecg_experiment.pr74_strict_parity_verifier import verify_saved

OUTPUT = Path("reports/pr74-corrective-evaluation-2026-10-28/strict-parity.json")


def main() -> None:
    """Write a new aggregate-only receipt after both warm and fresh saved comparisons pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    args = parser.parse_args()
    result = verify_saved(args.input_root)
    with (ROOT / OUTPUT).open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
