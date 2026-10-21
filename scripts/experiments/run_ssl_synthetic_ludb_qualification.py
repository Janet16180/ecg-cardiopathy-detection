"""Run the prospectively frozen public LUDB wave-timing qualification."""

import argparse
import time
from pathlib import Path


def main() -> None:
    """Parse artifact-root and identity-only admission arguments before starting the bounded study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--admission", required=True)
    arguments = parser.parse_args()
    clock = {"cpu": time.process_time(), "wall": time.perf_counter()}
    from ecg_experiment.ssl_synthetic_ludb_run import run_study
    result = run_study(arguments.root, arguments.admission, clock)
    print(result["statistics"])


if __name__ == "__main__":
    main()
