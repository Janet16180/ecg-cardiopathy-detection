"""Run MLC-02 Stage 3: selection record, operating points, test predictions and evaluation.

``select`` and ``fix`` read validation predictions and validation labels only. ``test-predict`` scores the
test roles once and refuses to start until the selection record is committed (see
``ecg_experiment/mlc02_stage3_predict.py``); ``--dry-run-role validation`` runs its code path on validation
into a scratch directory instead. ``evaluate`` reads saved test predictions only;
``--dry-run-validation-standin`` runs it on validation stand-ins into a scratch directory.

Examples
--------
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 PYTHONPATH=. .venv/bin/python \
    -m scripts.experiments.run_mlc02_stage3 select
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 PYTHONPATH=. .venv/bin/python \
    -m scripts.experiments.run_mlc02_stage3 fix
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 PYTHONPATH=. .venv/bin/python \
    -m scripts.experiments.run_mlc02_stage3 test-predict --device cuda --i-am-the-final-run
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python \
    -m scripts.experiments.run_mlc02_stage3 evaluate --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ecg_experiment.gpu import GPU_LOCK_PATH
from ecg_experiment.mlc02_stage3 import (
    BASE_CANDIDATES,
    CANDIDATES_DIR,
    RECORD_DOC,
    STAGE3,
    run_fix,
    run_select,
)


def main(argv: list[str] | None = None) -> None:
    """
    Parse the command line and run one step.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or None for ``sys.argv``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("select", "fix", "test-predict", "evaluate"))
    parser.add_argument("--output", type=Path, default=STAGE3, help="Stage 3 directory")
    parser.add_argument("--doc", type=Path, default=RECORD_DOC, help="Markdown selection record")
    parser.add_argument("--candidates", type=Path, default=CANDIDATES_DIR, help="Stage 2 candidate directory")
    parser.add_argument(
        "--allow-missing",
        nargs="*",
        default=[],
        choices=BASE_CANDIDATES,
        help="select: candidates that may be absent (the record is then provisional)",
    )
    parser.add_argument(
        "--overwrite-before-tests",
        action="store_true",
        help="select, fix: replace an existing file while no test prediction exists",
    )
    parser.add_argument(
        "--uncommitted", action="store_true", help="select, fix: allow uncommitted code (scratch runs only)"
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), help="test-predict: device of S-R and F")
    parser.add_argument("--gpu-lock", type=Path, default=GPU_LOCK_PATH, help="test-predict: GPU lock file")
    parser.add_argument(
        "--i-am-the-final-run", action="store_true", help="test-predict: score the test roles"
    )
    parser.add_argument("--dry-run-role", choices=("validation",), help="test-predict: dry run on this role")
    parser.add_argument("--scratch", type=Path, help="dry runs: scratch directory")
    parser.add_argument(
        "--waveform-records", type=int, help="test-predict dry run: S-R and F on this many records"
    )
    parser.add_argument(
        "--dry-run-validation-standin", action="store_true", help="evaluate: dry run on validation stand-ins"
    )
    parser.add_argument("--draws", type=int, default=2000, help="evaluate: bootstrap resamples")
    parser.add_argument("--workers", type=int, default=8, help="evaluate: threads")
    args = parser.parse_args(argv)
    command = sys.argv if argv is None else ["run_mlc02_stage3", *argv]
    if args.step == "select":
        record = run_select(
            args.output,
            args.doc,
            args.candidates,
            allow_missing=tuple(args.allow_missing),
            overwrite_before_tests=args.overwrite_before_tests,
            command=command,
            require_committed=not args.uncommitted,
        )
        summary = {
            key: record[key]
            for key in ("provisional", "missing_candidates", "selected", "highest", "highest_metric", "tied")
        }
        summary["table"] = {row["id"]: row["selection_metric"] for row in record["candidates"]}
        summary["ensemble"] = record["ensemble"]["members"]
    elif args.step == "fix":
        points = run_fix(
            args.output,
            overwrite_before_tests=args.overwrite_before_tests,
            command=command,
            require_committed=not args.uncommitted,
        )
        summary = {name: entry["thresholds"] for name, entry in points["candidates"].items()}
    elif args.step == "test-predict":
        from ecg_experiment.mlc02_stage3_predict import run_dry_run, run_test_predict

        if args.device is None:
            parser.error("test-predict needs --device")
        if args.dry_run_role is not None:
            if args.scratch is None or args.i_am_the_final_run:
                parser.error("a dry run needs --scratch and must not carry --i-am-the-final-run")
            summary = run_dry_run(
                args.scratch,
                args.device,
                args.dry_run_role,
                args.output,
                args.candidates,
                waveform_records=args.waveform_records,
                lock=args.gpu_lock,
                command=command,
            )["candidates"]
        else:
            if args.waveform_records is not None:
                parser.error("--waveform-records is for dry runs only")
            summary = run_test_predict(
                args.i_am_the_final_run,
                args.device,
                args.output,
                args.doc,
                args.candidates,
                lock=args.gpu_lock,
                command=command,
            )["files"]
    else:
        from ecg_experiment.mlc02_stage3_evaluate import run_evaluate, run_evaluate_dry

        if args.dry_run_validation_standin:
            if args.scratch is None:
                parser.error("a dry run needs --scratch")
            summary = run_evaluate_dry(
                args.scratch, args.output, draws=args.draws, workers=args.workers, command=command
            )
        else:
            summary = run_evaluate(args.output, draws=args.draws, workers=args.workers, command=command)
    print(json.dumps(summary, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
