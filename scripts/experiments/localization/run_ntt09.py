"""Run source-frozen fixed-contrast reference contribution stability."""

import argparse
from pathlib import Path

from threadpoolctl import threadpool_limits

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.localization.reference_contribution_stability import run


def main() -> None:
    """Preserve one bounded execution and any typed failure without changing settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/ntt09_reference_contribution_stability"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(args.output / "status.json", {"status": "running"})
    try:
        with threadpool_limits(limits=2):
            result = run(args.output)
    except Exception as error:
        write_json_atomic(args.output / "failure.json", {"type": type(error).__name__, "message": str(error)})
        write_json_atomic(args.output / "status.json", {"status": "failed"})
        raise
    write_json_atomic(
        args.output / "status.json",
        {"status": "completed", "result_sha256": sha256_file(args.output / "result.json")},
    )
    print(result["decision"], flush=True)


if __name__ == "__main__":
    main()
