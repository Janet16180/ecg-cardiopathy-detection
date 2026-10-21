"""Prepare or finish one transparent, externally measured public LUDB suffix transfer."""

import argparse
import time


def main() -> None:
    """Capture phase clocks before importing offline source; never issue a network request here."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["prepare", "finish"], required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--transfer-not-started", action="store_true")
    arguments = parser.parse_args()
    clock = {"cpu": time.process_time(), "wall": time.monotonic()}
    from ecg_experiment.ssl_synthetic_ludb_acquisition_v4 import finish, prepare
    if arguments.phase == "prepare":
        result = prepare(arguments.admission, clock)
    else:
        result = finish(arguments.admission, clock, arguments.transfer_not_started)
    print(result["status"])


if __name__ == "__main__":
    main()
