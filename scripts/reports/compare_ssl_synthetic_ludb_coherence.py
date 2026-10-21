"""Run only the conditionally admitted descriptive LUDB/generated synchrony comparison."""

import argparse
import time


def main() -> None:
    """Capture complete stage clocks before importing the bounded scientific runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", required=True)
    arguments = parser.parse_args()
    clock = {"cpu": time.process_time(), "wall": time.perf_counter()}
    from ecg_experiment.ssl_synthetic_ludb_coherence_run import run_study
    result = run_study(arguments.admission, clock)
    print(result["statistics"]["primary_descriptive"])


if __name__ == "__main__":
    main()
