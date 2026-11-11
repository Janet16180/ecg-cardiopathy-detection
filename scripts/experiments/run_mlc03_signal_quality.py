"""Run MLC-03 Part A (signal quality) on the train and validation records, and once on the test roles.

Examples
--------
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python \
    -m scripts.experiments.run_mlc03_signal_quality measure --workers 16
PYTHONPATH=. .venv/bin/python -m scripts.experiments.run_mlc03_signal_quality measure --limit 40 --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from ecg_experiment.mlc03_signal_runs import (
    DEVELOPMENT_ROLES,
    OUTPUT,
    WORK,
    assemble,
    code_hashes,
    frame_sha256,
    input_hashes,
    read_identity,
    read_index,
    run_full_records,
    run_measures,
    write_receipt,
)
from ecg_experiment.paths import to_stored

MEASURE_CODE = ["ecg_experiment/mlc03_signal_quality.py", "ecg_experiment/mlc03_signal_runs.py",
                "ecg_experiment/measurement_extractor042.py", "ecg_experiment/ecg_quality.py",
                "scripts/experiments/run_mlc03_signal_quality.py"]


def measure(args: argparse.Namespace, command: list[str]) -> None:
    """
    Measure every train and validation window (or a smoke sample) and publish the table.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    command : list[str]
        Command line, for the receipt.
    """
    name = "smoke" if args.smoke else "measures"
    directory = WORK / f"{name}_chunks"
    log = WORK / f"{name}.log"
    WORK.mkdir(parents=True, exist_ok=True)
    summary = run_measures(directory, DEVELOPMENT_ROLES, args.workers, args.limit, log=log)
    index = read_index(DEVELOPMENT_ROLES)
    if args.limit is not None:
        index = index.groupby("block", sort=False).head(args.limit).reset_index(drop=True)
    frame = assemble(directory, index)
    target = (WORK if args.smoke else OUTPUT) / f"{name}_train_validation.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)
    write_receipt(target.with_suffix(".receipt.json"), {
        "step": "measure", "command": command, "smoke": args.smoke, "limit": args.limit,
        "roles": list(DEVELOPMENT_ROLES), "records": len(frame), "columns": len(frame.columns),
        "summary": summary, "inputs": input_hashes(), "code": code_hashes(MEASURE_CODE),
        "table": {"path": str(target), "csv_sha256": frame_sha256(frame)}})
    print(json.dumps({**summary, "table": str(target)}, indent=1))


def full_records(args: argparse.Namespace, command: list[str]) -> None:
    """
    Measure the whole raw recording of every train and validation record longer than 10 s.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    command : list[str]
        Command line, for the receipt.
    """
    start = time.perf_counter()
    identity = read_identity(DEVELOPMENT_ROLES)
    frame = run_full_records(identity, args.workers)
    target = OUTPUT / "full_records_train_validation.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)
    write_receipt(target.with_suffix(".receipt.json"), {
        "step": "full-records", "command": command, "roles": list(DEVELOPMENT_ROLES),
        "records": len(frame), "wall_seconds": time.perf_counter() - start, "inputs": input_hashes(),
        "code": code_hashes(MEASURE_CODE),
        "table": {"path": to_stored(target), "csv_sha256": frame_sha256(frame)}})
    print(json.dumps({"records": len(frame), "table": str(target)}))


def main(argv: list[str] | None = None) -> None:
    """
    Parse the command line and run one step.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or None for ``sys.argv``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("measure", "full-records"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, help="measure: first records of each cache block only")
    parser.add_argument("--smoke", action="store_true", help="measure: write to the scratch directory")
    args = parser.parse_args(argv)
    command = sys.argv if argv is None else ["run_mlc03_signal_quality", *argv]
    if args.step == "measure":
        if args.limit is not None and not args.smoke:
            raise SystemExit("--limit is for smoke tests only; add --smoke")
        measure(args, command)
    elif args.step == "full-records":
        full_records(args, command)


if __name__ == "__main__":
    main()
