"""Reproduce the final admitted PTB microbatch with its historical full batch shape."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from threadpoolctl import threadpool_limits

from ecg_experiment.pr74_batch_shape_reproduction import OUTPUT, SOURCE, reproduce


def main() -> None:
    """Keep failed/completed parents and write only a new receipt-bound feature directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        result = reproduce(args.input_root, args.source, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
