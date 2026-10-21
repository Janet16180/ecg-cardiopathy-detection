"""Run only the prospectively admitted frozen-localizer landmark diagnostic."""

import argparse
import time
from pathlib import Path


def main() -> None:
    """Capture full-stage clocks before importing scientific dependencies.

    Returns
    -------
    None
        No return value; writes only the declared local artifacts or raises on failure.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    clock = {"cpu": time.process_time(), "wall": time.perf_counter()}
    from ecg_experiment.ssl_synthetic_ludb_localizer_run import preflight, run

    if arguments.preflight_only:
        preflight(arguments.admission)
        print("Source and identity preflight passed; no GPU operations, scores or recordings read")
    else:
        print(run(arguments.root, arguments.admission, clock)["statistics"]["primary_descriptive"])


if __name__ == "__main__":
    main()
