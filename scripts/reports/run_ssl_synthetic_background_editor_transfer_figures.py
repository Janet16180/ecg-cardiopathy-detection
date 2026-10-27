"""Measure the admitted fixed saved-output illustration step with zero forwards or scores."""

import argparse
import time
from pathlib import Path


def main() -> None:
    """Capture actual whole-stage clocks before importing the measured illustration runtime.

    Returns
    -------
    None
        Prints the new PNG identities after the terminal resource ledger passes.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    clock = {"cpu": time.process_time(), "wall": time.perf_counter()}
    from ecg_experiment.ssl_synthetic_background_editor_transfer_figures_run import preflight, run

    if arguments.preflight_only:
        preflight(arguments.admission)
        print("Fixed illustration metadata/source preflight passed; no arrays, plots or scores")
    else:
        print(run(arguments.root, arguments.admission, clock)["outputs"])


if __name__ == "__main__":
    main()
