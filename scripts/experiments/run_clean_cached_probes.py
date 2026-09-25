"""Refit JEPA and ordinary CPC cached probes on the clean original PTB-XL labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecg_experiment.clean_cached_probes import ROOT, run


def main() -> None:
    """Parse the output path and run the clean CPU-only development refits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/clean_cached_probe_rerun_v1")
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir)), flush=True)


if __name__ == "__main__":
    main()
