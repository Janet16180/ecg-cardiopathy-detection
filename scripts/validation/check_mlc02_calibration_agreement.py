"""
Agreement of the damped-Newton calibration fit with the frozen library fit and with sklearn, on validation.

For every candidate of a selection record, the logistic recalibration ``logit P(y = 1) = a + b logit(p)`` is
fitted on each validation cell where the condition has at least 20 positives and 20 negatives: the pooled
validation records (the Platt fit of ``fix``) and each validation source (the calibration slope of a set).
Each cell is fitted three ways:

- ``mlc02_evaluation.logistic_fit``, the library's undamped Newton;
- ``mlc02_calibration.damped_logistic_fit``, primary under Addendum 2 of the MLC-02 protocol;
- sklearn's unpenalized ``LogisticRegression`` (lbfgs, tol 1e-14), as an outside reference.

Only validation predictions and validation labels are read. Run from the repository root:
``OPENBLAS_NUM_THREADS=8 PYTHONPATH=. .venv/bin/python -m scripts.validation.check_mlc02_calibration_agreement
--output <stage3 directory> --report <file.json>``.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from ecg_experiment.files import write_json_atomic
from ecg_experiment.mlc02_calibration import damped_logistic_fit
from ecg_experiment.mlc02_data import CONDITIONS, VALIDATION_SOURCES
from ecg_experiment.mlc02_evaluation import MIN_SUPPORT, logistic_fit, logit
from ecg_experiment.mlc02_stage3 import (
    STAGE3,
    code_state,
    json_ready,
    read_record,
    validation_data,
    validation_probabilities,
)

SKLEARN_TOLERANCE = 1e-14
SKLEARN_ITERATIONS = 100000


def cell_fits(y: np.ndarray, x: np.ndarray) -> dict[str, float]:
    """
    Intercept and slope of one cell by the library, the damped fit and sklearn.

    Parameters
    ----------
    y : np.ndarray
        Defined binary states.
    x : np.ndarray
        Logits of the probabilities.

    Returns
    -------
    dict[str, float]
        ``library_a``, ``library_b``, ``damped_a``, ``damped_b``, ``sklearn_a`` and ``sklearn_b``; NaN where a
        fit did not converge.
    """
    library = logistic_fit(y, x)
    damped = damped_logistic_fit(y, x)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reference = LogisticRegression(C=np.inf, tol=SKLEARN_TOLERANCE, max_iter=SKLEARN_ITERATIONS)
        reference.fit(x[:, None], y)
    return {
        "library_a": float(library[0][0]),
        "library_b": float(library[1][0]),
        "damped_a": float(damped[0][0]),
        "damped_b": float(damped[1][0]),
        "sklearn_a": float(reference.intercept_[0]),
        "sklearn_b": float(reference.coef_[0, 0]),
    }


def agreement(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Summary of the cell fits.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        One ``cell_fits`` result per cell, with ``candidate``, ``scope`` and ``condition``.

    Returns
    -------
    dict[str, Any]
        Counts of cells and of converged fits, the library failures, and the largest absolute differences of
        the damped fit from the library (where the library converged) and from sklearn.
    """

    def largest(pairs: list[dict[str, Any]], other: str) -> float:
        return max(
            max(abs(row["damped_a"] - row[f"{other}_a"]), abs(row["damped_b"] - row[f"{other}_b"]))
            for row in pairs
        )

    damped_ok = [row for row in rows if np.isfinite(row["damped_b"])]
    library_ok = [row for row in damped_ok if np.isfinite(row["library_b"])]
    failures = [row for row in rows if not np.isfinite(row["library_b"])]
    pooled = [row for row in damped_ok if row["scope"] == "pooled"]
    return {
        "cells": len(rows),
        "pooled_cells": sum(row["scope"] == "pooled" for row in rows),
        "source_cells": sum(row["scope"] != "pooled" for row in rows),
        "damped_converged": len(damped_ok),
        "library_converged": len(rows) - len(failures),
        "library_failures_pooled": [
            f"{row['candidate']} {row['condition']}" for row in failures if row["scope"] == "pooled"
        ],
        "library_failures_by_source": len([row for row in failures if row["scope"] != "pooled"]),
        "max_abs_damped_minus_library": largest(library_ok, "library"),
        "max_abs_damped_minus_sklearn": largest(damped_ok, "sklearn"),
        "max_abs_damped_minus_sklearn_pooled": largest(pooled, "sklearn"),
        "max_abs_damped_minus_sklearn_library_failures": largest(
            [row for row in failures if np.isfinite(row["damped_b"])], "sklearn"
        )
        if failures
        else None,
    }


def main(argv: list[str] | None = None) -> None:
    """
    Fit every validation cell three ways and write the agreement report.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or None for ``sys.argv``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--output", type=Path, default=STAGE3, help="Stage 3 directory with the selection record"
    )
    parser.add_argument("--report", type=Path, required=True, help="JSON report to write")
    args = parser.parse_args(argv)
    started = time.time()
    record, record_sha = read_record(args.output)
    data = validation_data()
    probabilities = validation_probabilities(record, data)
    rows = []
    for name, probability in probabilities.items():
        scopes = {"pooled": np.ones(len(probability), dtype=bool)} | {
            source: data["source"] == source for source in VALIDATION_SOURCES
        }
        for scope, mask in scopes.items():
            for column, condition in enumerate(CONDITIONS):
                cell = mask & ~np.isnan(data["labels"][:, column])
                y = data["labels"][cell, column]
                if MIN_SUPPORT <= y.sum() <= len(y) - MIN_SUPPORT:
                    fits = cell_fits(y, logit(probability[cell, column]))
                    rows.append({"candidate": name, "scope": scope, "condition": condition, **fits})
    summary = agreement(rows)
    report = {
        "summary": summary,
        "selection_record_sha256": record_sha,
        "selection_record_provisional": record["provisional"],
        "candidates": list(probabilities),
        "command": sys.argv if argv is None else ["check_mlc02_calibration_agreement", *argv],
        "git": code_state(require_committed=False),
        "wall_seconds": time.time() - started,
        "test_roles_opened": False,
        "cells": rows,
    }
    write_json_atomic(args.report, json_ready(report))
    print({key: value for key, value in summary.items() if key != "library_failures_pooled"})
    print("library failures, pooled:", summary["library_failures_pooled"])


if __name__ == "__main__":
    main()
