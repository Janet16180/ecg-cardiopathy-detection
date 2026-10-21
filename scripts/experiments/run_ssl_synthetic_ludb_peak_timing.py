"""Preflight or run the prospective explicit QRS-marker timing qualification."""

import argparse
import time
from pathlib import Path


def main() -> None:
    """Capture full-stage clocks before importing frozen scientific helpers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    clock = {"cpu": time.process_time(), "wall": time.perf_counter()}
    from ecg_experiment.ssl_synthetic_ludb_peak_run import preflight, run
    if arguments.preflight_only:
        _, admission = preflight(arguments.admission)
        print({"preflight_passed": True, "source_bindings": len(admission["source_sha256"]),
               "dataset_files_opened": 0})
        return
    result = run(arguments.root, arguments.admission, clock)
    print(result["statistics"])


if __name__ == "__main__":
    main()
