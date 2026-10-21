"""Resume only the prospectively admitted opaque public LUDB archive suffix."""

import argparse
import time


def main() -> None:
    """Capture full-stage clocks before importing the bounded successor acquisition."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", required=True)
    arguments = parser.parse_args()
    clock = {"cpu": time.process_time(), "wall": time.monotonic()}
    from ecg_experiment.ssl_synthetic_ludb_acquisition_v2 import acquire
    result = acquire(arguments.admission, clock)
    print(result["status"])


if __name__ == "__main__":
    main()
