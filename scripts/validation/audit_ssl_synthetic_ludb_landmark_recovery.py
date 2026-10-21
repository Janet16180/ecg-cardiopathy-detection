"""Run metadata-only setup preflight or one prospectively admitted landmark-audit recovery."""

import argparse
import time
from pathlib import Path


def main() -> None:
    """Capture actual-stage clocks before imports and distinguish setup from data reconstruction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    for name in ("main-sha", "script-sha", "runner-sha", "config-sha"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    clock = {"cpu": time.process_time(), "wall": time.perf_counter()}
    from ecg_experiment.ssl_synthetic_ludb_landmark_audit_recovery import RUNNER, metadata_preflight, run
    from ecg_experiment.synthetic_audit_io import file_sha256

    if (
        file_sha256(Path(__file__)) != args.script_sha
        or file_sha256(args.source_root / RUNNER) != args.runner_sha
    ):
        raise ValueError("Explicit new entrypoint identity differs")
    if args.preflight_only:
        _, prior = metadata_preflight(args.root, args.source_root, args.config_sha)
        print(
            {
                "metadata_preflight_passed": True,
                "raw_files_read": 0,
                "score_arrays_read": 0,
                "derived_prior_stages": prior["current_machine_stages"],
            }
        )
        return
    result = run(
        args.root, args.source_root, args.main_sha, args.script_sha, args.runner_sha, args.config_sha, clock
    )
    print({"all_passed": result["all_passed"], "mismatch_count": result["mismatch_count"]})


if __name__ == "__main__":
    main()
