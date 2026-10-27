"""Run the prospectively admitted, annotation-only EDB acquisition."""

import time

STARTED = (time.perf_counter(), time.process_time())


def main() -> None:
    """Parse local paths and execute the source-admitted acquisition."""
    import argparse
    from pathlib import Path

    from ecg_experiment.ssl_synthetic_edb_acquisition import acquire

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    args = parser.parse_args()
    ledger = acquire(args.root, args.admission, STARTED)
    if not ledger["technical_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
