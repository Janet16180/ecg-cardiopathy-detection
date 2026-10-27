"""Run the prospectively admitted independent public QRS-landmark timing audit once."""

import argparse
import time
from pathlib import Path


def main() -> None:
    """Capture full-stage clocks before importing the scientific independent checker."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source-root", "admission", "main-sha", "script-sha", "runner-sha"):
        parser.add_argument("--" + name, required=True, type=Path if name.endswith("root") else str)
    args = parser.parse_args()
    clock = {"cpu": time.process_time(), "wall": time.perf_counter()}
    from ecg_experiment.ssl_synthetic_ludb_landmark_audit_run import run
    from ecg_experiment.synthetic_audit_io import file_sha256

    if file_sha256(Path(__file__)) != args.script_sha:
        raise ValueError("Explicit independent entrypoint identity differs")
    result = run(
        args.root, args.source_root, args.admission, args.main_sha, args.script_sha, args.runner_sha, clock
    )
    print({"all_passed": result["all_passed"], "mismatch_count": result["mismatch_count"]})


if __name__ == "__main__":
    main()
