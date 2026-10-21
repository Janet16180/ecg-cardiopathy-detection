"""Run the conditionally admitted, descriptive real/generated activity comparison."""

import argparse
import time
from pathlib import Path


def main() -> None:
    """Capture stage clocks before importing the scientific runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    clock = {"cpu": time.process_time(), "wall": time.perf_counter()}
    from ecg_experiment.ssl_synthetic_ludb_coherence_run_v2 import preflight, run
    if arguments.preflight_only:
        preflight(arguments.admission)
        print("Complete source/design preflight passed; no recordings read or outputs created")
    else:
        print(run(arguments.root, arguments.admission, clock)["statistics"]["primary_descriptive"])


if __name__ == "__main__":
    main()
