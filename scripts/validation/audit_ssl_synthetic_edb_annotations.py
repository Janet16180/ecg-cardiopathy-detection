"""Execute a prospectively source-admitted independent EDB annotation audit locally."""

import time

START_CPU = time.process_time()
START_WALL = time.monotonic()

import argparse  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

for THREAD_VARIABLE in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[THREAD_VARIABLE] = "1"


def main() -> None:
    """Require explicit source, artifact and independent admission paths for one execution."""
    from ecg_experiment.ssl_synthetic_edb_annotation_audit_run import run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--admission", required=True, type=Path)
    arguments = parser.parse_args()
    result = run(arguments.root, arguments.source, arguments.admission, START_CPU, START_WALL)
    print(result["status"], result["record_decodes_completed"], result["mismatch_count"])
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
