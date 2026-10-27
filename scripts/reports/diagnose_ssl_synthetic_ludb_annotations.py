"""Run the prospective record95-only annotation diagnosis without ECG samples or AP scores."""

import argparse
import time


def main() -> None:
    """Capture total-stage clocks before importing the bounded annotation reader."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", required=True)
    arguments = parser.parse_args()
    clock = {"cpu": time.process_time(), "wall": time.perf_counter()}
    from ecg_experiment.ssl_synthetic_ludb_annotation_diagnosis import run
    result = run(arguments.admission, clock)
    print({lead: row["event_count"] for lead, row in result["streams"].items()})


if __name__ == "__main__":
    main()
