"""Independently recover frozen-localizer audit after preserving the actual timing-order failure."""

import argparse
import time
from pathlib import Path


def main() -> None:
    """Check original/recovery identities before metadata-only setup or one measured recovery."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    for name in ("admission", "recovery", "recovery-sha", "main-sha", "script-sha", "runner-sha"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    clock = {"cpu": time.process_time(), "wall": time.perf_counter()}
    from ecg_experiment.ssl_synthetic_ludb_localizer_audit_v4 import RUNNER, metadata_setup, run
    from ecg_experiment.synthetic_audit_io import file_sha256

    if (
        file_sha256(Path(__file__)) != args.script_sha
        or file_sha256(args.source_root / RUNNER) != args.runner_sha
    ):
        raise ValueError("Literal independent recovery executable identities differ")
    if args.preflight_only:
        metadata_setup(args.root, args.source_root, args.admission, args.recovery, args.recovery_sha)
        print({"metadata_preflight_passed": True, "new_waveform_reads": 0, "new_map_scores": 0})
        return
    result = run(
        args.root,
        args.source_root,
        args.admission,
        args.main_sha,
        args.script_sha,
        args.runner_sha,
        args.recovery,
        args.recovery_sha,
        clock,
    )
    print({"all_passed": result["all_passed"], "mismatch_count": result["mismatch_count"]})


if __name__ == "__main__":
    main()
