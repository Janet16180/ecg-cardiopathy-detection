"""Experiment 044, post hoc: sensitivity of pipelines v2, v3 and v3b at a matched false-referral rate.

In the prespecified run, v3 referred slightly more evaluation normals than v2 at the same nominal budget. This
exploratory check sets every pipeline's thresholds on the evaluation normals themselves (an oracle that a site
cannot use), so that all pipelines refer the same share of the evaluation normals, and compares what they
catch. It also reports the evaluation rate when the thresholds come from the whole local pool (6,923
normals). No decision attaches to it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.finding_screen import split_thresholds
from ecg_experiment.pipeline_v2 import percentile_interval, resample_counts
from scripts.experiments.run_finding_screen033 import RULES as RULES033
from scripts.experiments.run_finding_screen033 import load_rows
from scripts.experiments.run_pipeline_v2_037 import (
    BOOTSTRAP_SEED,
    BUDGETS,
    RESAMPLES,
    RULES,
    outcome_masks,
    receipt_checked,
)

ROOT = Path(__file__).resolve().parents[2]
PRIOR044 = ROOT / "outputs/experiment044_pipeline_v3_v1"
OUTPUT = ROOT / "outputs/experiment044_rate_matched_v1"
PIPELINES = ("v2", "v3", "v3b")
CONTRASTS = (("v3_minus_v2", "v3", "v2"), ("v3b_minus_v2", "v3b", "v2"), ("v3b_minus_v3", "v3b", "v3"))
OUTCOMES = ("normal", "composite", "binary_positive", "pvc", "other")
SOURCES = ("scripts/experiments/rate_matched_044.py", "scripts/experiments/run_finding_screen033.py",
           "scripts/experiments/run_pipeline_v2_037.py", "ecg_experiment/finding_screen.py",
           "ecg_experiment/pipeline_v2.py")


def pipeline_matrices(saved: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """
    Return the score matrix of every pipeline and main rule from 044's saved predictions.

    Parameters
    ----------
    saved : dict[str, np.ndarray]
        Arrays of 044's ``predictions.npz``.

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        Per pipeline and rule, the ``(n, 1 + F)`` matrix of ``finding_screen.split_thresholds``.
    """
    findings = {"v2": "v2", "v3": "v2", "v3b": "v3b"}
    binary = {"v2": "v2", "v3": "v3", "v3b": "v3"}
    matrices = {}
    for pipeline in PIPELINES:
        scores = {"binary": saved[f"sph_{binary[pipeline]}_binary"],
                  "combined": saved[f"sph_{findings[pipeline]}_z_combined"]}
        matrices[pipeline] = {rule: np.column_stack([scores["binary"], *(scores[name]
                                                                         for name in RULES033[rule][0])])
                              for rule in RULES}
    return matrices


def weighted_screen(matrix: np.ndarray, counts: np.ndarray, normal: np.ndarray, masks: dict[str, np.ndarray],
                    per_mille: int, shares: tuple[int, ...]) -> dict[str, float]:
    """
    Set thresholds on the (resampled) evaluation normals and return each outcome's referral rate.

    Parameters
    ----------
    matrix : np.ndarray
        Scores of the evaluation ECGs.
    counts : np.ndarray
        How often each ECG enters (1 for the observed data).
    normal : np.ndarray
        Mask of the evaluation normals.
    masks : dict[str, np.ndarray]
        Outcome masks.
    per_mille : int
        Budget in thousandths.
    shares : tuple[int, ...]
        Finding shares of the rule.

    Returns
    -------
    dict[str, float]
        Referral rate per outcome, weighted by ``counts``.
    """
    normals = np.repeat(matrix[normal], counts[normal].astype(np.int64), axis=0)
    thresholds = split_thresholds(normals, per_mille, shares)
    referred = (matrix > thresholds).any(axis=1)
    return {name: float((counts * referred * mask).sum() / (counts * mask).sum())
            for name, mask in masks.items()}


def main() -> None:
    """Run the check once and write its result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(f"Refusing to overwrite {arguments.output}")
    started = time.perf_counter()
    receipt = receipt_checked(PRIOR044, ("predictions.npz",))
    with np.load(PRIOR044 / "predictions.npz") as saved:
        arrays = {key: saved[key] for key in saved.files}
    rows, _, local_normal = load_rows()
    if not np.array_equal(rows["ecg_id"].to_numpy(dtype=str), arrays["sph_ecg_ids"]):
        raise ValueError("SPH rows differ from 044's")
    evaluation = rows["evaluation"].to_numpy()
    if not np.array_equal(evaluation, arrays["sph_evaluation"]):
        raise ValueError("Evaluation half differs from 044's")
    masks = {name: mask for name, mask in outcome_masks(rows, evaluation).items() if name in OUTCOMES}
    matrices = pipeline_matrices(arrays)
    patients = rows.loc[evaluation, "patient_id"].to_numpy(dtype=str)
    counts = resample_counts(patients, RESAMPLES, BOOTSTRAP_SEED)
    ones = np.ones(int(evaluation.sum()))
    normal = masks["normal"]
    oracle, pooled = [], []
    for rule in RULES:
        shares = RULES033[rule][1]
        for budget in BUDGETS:
            observed, boot = {}, {}
            for pipeline in PIPELINES:
                matrix = matrices[pipeline][rule]
                observed[pipeline] = weighted_screen(matrix[evaluation], ones, normal, masks, budget, shares)
                boot[pipeline] = [weighted_screen(matrix[evaluation], row, normal, masks, budget, shares)
                                  for row in counts]
                thresholds = split_thresholds(matrix[local_normal], budget, shares)
                referred = (matrix[evaluation] > thresholds).any(axis=1)
                pooled.append({"rule": rule, "budget": budget / 1000, "pipeline": pipeline,
                               **{name: float(referred[mask].mean()) for name, mask in masks.items()}})
            row: dict[str, Any] = {"rule": rule, "budget": budget / 1000, "observed": observed,
                                   "contrasts": {}}
            for label, first, second in CONTRASTS:
                pairs = list(zip(boot[first], boot[second], strict=True))
                row["contrasts"][label] = {
                    name: {"difference": observed[first][name] - observed[second][name],
                           "ci": percentile_interval(np.array([a[name] - b[name] for a, b in pairs]))}
                    for name in OUTCOMES}
            oracle.append(row)
            seconds = round(time.perf_counter() - started)
            print(json.dumps({"rule": rule, "budget": budget, "seconds": seconds}), flush=True)
    arguments.output.mkdir(parents=True)
    inputs = {f"outputs/{PRIOR044.name}/{name}": digest for name, digest in receipt.items()}
    write_json_atomic(arguments.output / "result.json", {
        "status": "complete", "post_hoc": True,
        "identity": {"inputs": inputs, "sources": {name: sha256_file(ROOT / name) for name in SOURCES}},
        "oracle_evaluation_thresholds": oracle, "whole_local_pool_thresholds": pooled,
        "local_normals": int(len(local_normal)), "evaluation_normals": int(masks["normal"].sum()),
        "resamples": RESAMPLES, "bootstrap_seed": BOOTSTRAP_SEED, "seconds": time.perf_counter() - started})


if __name__ == "__main__":
    main()
