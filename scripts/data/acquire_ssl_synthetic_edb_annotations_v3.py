"""Preflight or run the admitted released-roster EDB acquisition recovery."""

import time

STARTED = (time.perf_counter(), time.process_time())


def main() -> None:
    """Separate source/history metadata preflight from measured record-asset acquisition."""
    import argparse
    from pathlib import Path

    from ecg_experiment.ssl_synthetic_edb_acquisition_v3 import acquire, preflight

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        preflight(args.root, args.admission)
        print({"preflight_passed": True, "http_requests": 0, "annotation_decodes": 0})
        return
    if not acquire(args.root, args.admission, STARTED)["technical_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
