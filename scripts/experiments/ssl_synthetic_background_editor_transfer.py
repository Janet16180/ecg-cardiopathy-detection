"""Run the admitted no-training background/editor transfer study on CUDA0."""

import argparse
import time
from pathlib import Path


def main() -> None:
    """Capture full-stage clocks before scientific imports and execute only admitted work.

    Returns
    -------
    None
        Prints a preflight result or the complete fixed primary statistics.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    clock = {"cpu": time.process_time(), "wall": time.perf_counter()}
    from ecg_experiment.ssl_synthetic_background_editor_transfer_run import preflight, run

    if arguments.preflight_only:
        preflight(arguments.admission)
        print("Source/input roster preflight passed; no arrays, GPU forwards or scores read")
    else:
        print(run(arguments.root, arguments.admission, clock)["statistics"]["primary"])


if __name__ == "__main__":
    main()
