"""Execute the fixed CLUSTER-06 bank averaging and scale-control study."""

import argparse
from pathlib import Path

import torch
from threadpoolctl import threadpool_limits

from ecg_experiment import ROOT
from ecg_experiment.clustering.bank_scale import run
from ecg_experiment.files import sha256_file, write_json_atomic


def main() -> None:
    """Preserve a single bounded execution and typed failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/cluster06_bank_scale"))
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
        print(result["decision"], flush=True)
    except Exception as error:
        write_json_atomic(output / "failure.json", {"type": type(error).__name__, "message": str(error)})
        write_json_atomic(output / "status.json", {"status": "failed"})
        raise


if __name__ == "__main__":
    main()
