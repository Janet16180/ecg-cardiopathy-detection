"""Execute the prospective CLUSTER-05 matched dimension, pooling and scale controls."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from threadpoolctl import threadpool_limits

from ecg_experiment import ROOT
from ecg_experiment.clustering.matched_controls import run
from ecg_experiment.files import sha256_file, write_json_atomic


def main() -> None:
    """Preserve one source-bound execution and typed failure without changing settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/cluster05_matched_controls"))
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output / "status.json", {"status": "running"})
    try:
        torch.set_num_threads(2)
        with threadpool_limits(limits=2):
            result = run(output)
        write_json_atomic(
            output / "status.json",
            {"status": "completed", "result_sha256": sha256_file(output / "result.json")},
        )
        passed = result["decision"]["primary_matched_geometric_concordance_passed"]
        print(f"Completed in {result['seconds']:.1f}s; primary={passed}")
    except Exception as error:
        write_json_atomic(output / "failure.json", {"type": type(error).__name__, "message": str(error)})
        write_json_atomic(output / "status.json", {"status": "failed"})
        raise


if __name__ == "__main__":
    main()
