"""Build the MLC-01 condition dataset manifest, count tables and receipt; no model is fitted or scored.

Run from the repository root with ``PYTHONPATH=. OPENBLAS_CORETYPE=Haswell OMP_NUM_THREADS=1``:
``.venv/bin/python -m scripts.data.build_mlc01_dataset --workers 16``.
"""

import argparse
import json
import sys
from pathlib import Path

from ecg_experiment import ROOT
from ecg_experiment.mlc01_build import build


def main() -> None:
    """Parse arguments and run the build."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/mlc01_condition_dataset_20261110")
    parser.add_argument("--cache-dir", type=Path, default=Path("/var/tmp/janetrivera/mlc01_scan"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="records per source, for a profile run")
    args = parser.parse_args()
    result = build(args.output_dir, args.cache_dir, args.workers, args.limit, sys.argv)
    print(json.dumps(result["summary"], indent=2, default=str))


if __name__ == "__main__":
    main()
