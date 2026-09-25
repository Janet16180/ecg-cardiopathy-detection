"""Publish a CPU-only clean original-cohort rerun preflight (no training)."""

from __future__ import annotations

import argparse
from pathlib import Path

from ecg_experiment.clean_rerun import prepare


def main(argv: list[str] | None = None) -> None:
    """Audit pinned data and publish a new manifest-only preflight directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = prepare(args.output_dir)
    print(f"Clean-rerun preflight passed: {receipt['train_count']} train, "
          f"{sum(receipt['heldout_counts'].values())} held-out; "
          f"{args.output_dir / 'receipt.json'}")


if __name__ == "__main__":
    main()
