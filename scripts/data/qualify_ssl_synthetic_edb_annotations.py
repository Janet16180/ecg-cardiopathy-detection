"""Preflight or execute the separately admitted EDB annotation qualification."""

import time

STARTED = (time.perf_counter(), time.process_time())


def main() -> None:
    """Distinguish raw-byte metadata checks from one measured annotation decode phase."""
    import argparse
    from pathlib import Path

    from ecg_experiment.ssl_synthetic_edb_qualification_run import preflight, run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        preflight(args.root, args.admission)
        print({"preflight_passed": True, "annotation_decodes": 0, "new_model_scores": 0})
        return
    if not run(args.root, args.admission, STARTED)["technical_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
