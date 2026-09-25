"""Repeat the 014 cached-probe fusion development screen on clean labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecg_experiment.clean_probe_fusion import ROOT, run


def main() -> None:
    """Screen both clean label budgets and print a compact result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes-dir", type=Path, default=ROOT / "outputs/clean_cached_probe_rerun_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/clean_probe_fusion_v1")
    args = parser.parse_args()
    print(json.dumps(run(args.probes_dir, args.output_dir)), flush=True)


if __name__ == "__main__":
    main()
