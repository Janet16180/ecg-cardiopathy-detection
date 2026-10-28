"""Build cold-start reference/QC stages using fresh raw administrative provenance."""

import argparse
import json
from pathlib import Path

from ecg_experiment.cold_data_stages import quality, references


def main() -> None:
    """Profiles always read raw; optional verified caches can accelerate full stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["references", "quality"])
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--reuse-verified-caches", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or (args.limit is not None and args.limit < 1):
        raise ValueError("Use one to four workers and a positive resource-profile limit")
    function = references if args.stage == "references" else quality
    result = function(args.input_root, args.workers, args.limit, args.reuse_verified_caches)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
