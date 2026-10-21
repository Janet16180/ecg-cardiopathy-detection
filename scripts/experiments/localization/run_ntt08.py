"""Run the frozen contextual representation contribution study on admitted cached data."""

from __future__ import annotations

import argparse
from pathlib import Path

from threadpoolctl import threadpool_limits

from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.localization.contribution_study import run


def main() -> None:
    """Create a new local study directory and preserve typed failures without retrying."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/ntt08_contextual_contributions"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(args.output / "status.json", {"status": "running"})
    try:
        with threadpool_limits(limits=2):
            result = run(args.output)
    except Exception as error:
        write_json_atomic(args.output / "failure.json", {"type": type(error).__name__, "message": str(error)})
        write_json_atomic(
            args.output / "status.json", {"status": "failed", "failure_type": type(error).__name__}
        )
        raise
    write_json_atomic(
        args.output / "status.json",
        {"status": "completed", "result_sha256": sha256_file(args.output / "result.json")},
    )
    print(f"NTT08 completed: engineering gate={result['primary_engineering_gate_passed']}", flush=True)


if __name__ == "__main__":
    main()
