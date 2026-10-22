"""Run the admitted matched generated/mixed-background B-localizer training study."""

import argparse
import time
from pathlib import Path


def main() -> None:
    """Capture the whole-stage clocks before scientific imports and execute only admitted work.

    Returns
    -------
    None
        Prints the fixed primary result after its actual resource ledger passes.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    clock = {"cpu": time.process_time(), "wall": time.perf_counter()}
    from ecg_experiment.ssl_synthetic_mixed_background_run import preflight, run

    if args.preflight_only:
        preflight(args.admission)
        print("Mixed-background metadata/source preflight passed; no signals, training or scores")
    else:
        print(run(args.root, args.admission, clock)["statistics"]["primary"])


if __name__ == "__main__":
    main()
