"""Preflight actual artifact schemas or execute the separately admitted mixed-training audit."""

import argparse
import time
from pathlib import Path


def main() -> None:
    """Keep metadata setup separate from one measured independent numerical reconstruction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    for name in ("main-admission", "audit-admission", "script-sha", "runner-sha", "recovery-sha"):
        parser.add_argument("--" + name, required=True)
    for name in ("audit-admission-sha", "main-sha"):
        parser.add_argument("--" + name)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    clock = {"cpu": time.process_time(), "wall": time.perf_counter()}
    from ecg_experiment.ssl_synthetic_mixed_background_audit_run_v2 import (
        RECOVERY,
        RUNNER,
        metadata_setup,
        run,
    )
    from ecg_experiment.synthetic_audit_io import file_sha256

    if (
        file_sha256(Path(__file__)) != args.script_sha
        or file_sha256(args.source_root / RUNNER) != args.runner_sha
        or file_sha256(args.source_root / RECOVERY) != args.recovery_sha
    ):
        raise ValueError("Literal independent runtime/CLI identities differ")
    if args.preflight_only:
        metadata_setup(args.root, args.source_root, args.main_admission, args.audit_admission)
        print({"metadata_preflight_passed": True, "new_score_reconstructions": 0, "new_model_forwards": 0})
        return
    if not all((args.audit_admission_sha, args.main_sha)):
        raise ValueError("Actual audit requires the separate admission and literal actual main identities")
    result = run(
        args.root,
        args.source_root,
        args.main_admission,
        args.audit_admission,
        args.main_sha,
        args.audit_admission_sha,
        args.runner_sha,
        args.script_sha,
        clock,
    )
    print(
        {
            "all_passed": result["all_passed"],
            "mismatch_count": result["mismatch_count"],
            "scientific_passed": result["scientific_passed"],
        }
    )


if __name__ == "__main__":
    main()
