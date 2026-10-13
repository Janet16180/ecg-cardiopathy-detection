"""Experiment 044: pipeline v3, pipeline v2 with the binary readout refitted on xECG + JEPA features.

Fits v2's readout R and the xECG PVC and WPW heads again (to check them against Experiment 037), the
concatenated-feature readout of Experiment 043 (v3) and the concatenated PVC and WPW heads (v3b), and
recomputes 037's operating numbers on SPH for the three pipelines over 037's identical draws and resamples.

The row logic lives in pinned runners (Experiments 030, 032, 033, 037 and 043), which are imported as
documented in the protocol.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

import ecg_experiment
from ecg_experiment.ann_heads import validation_mask
from ecg_experiment.files import sha256_file, write_json_atomic, write_npz_atomic
from ecg_experiment.finding_screen import clipped_logits, finding_z
from ecg_experiment.full_development import fit_logistic, predict
from ecg_experiment.hybrid_score import normal_standardizer
from ecg_experiment.intervals import paired_auroc_difference
from ecg_experiment.multisource_readout import fit_readout
from ecg_experiment.paths import to_stored
from ecg_experiment.pipeline_v2 import (
    head_parameters,
    percentile_interval,
    resample_counts,
    resampled_rates,
    score_parameters,
)
from ecg_experiment.pipeline_v3 import local_normal_plans, pipeline_reading, screen_draws
from ecg_experiment.provenance import git_head
from ecg_experiment.referral_budget import referrals_per_1000
from ecg_experiment.rhythm_findings import GROUPS, clear_normal
from scripts.experiments.run_ann_heads043 import (
    check_against_037,
    check_comparator_aurocs,
    evaluation_sets,
    logits_of,
    set_scores,
    smoke_split,
)
from scripts.experiments.run_ann_heads043 import training_data as training_data043
from scripts.experiments.run_finding_screen033 import EXPECTED as EXPECTED033
from scripts.experiments.run_finding_screen033 import RULES as RULES033
from scripts.experiments.run_finding_screen033 import counts_of, load_rows, load_scores
from scripts.experiments.run_lead_wave_maps042 import blas_architectures
from scripts.experiments.run_pipeline_v2_037 import (
    BOOTSTRAP_SEED,
    BUDGETS,
    DRAW_SEED,
    DRAWS,
    FINDING_HEADS,
    LOCAL_OUTCOMES,
    MAX_BINARY_COST,
    PREVALENCE,
    PRIMARY_BUDGET,
    PRIMARY_RULE,
    PRIMARY_SIZE,
    RESAMPLES,
    RULES,
    SIZES,
    WORSE_MARGIN,
    largest_difference,
    outcome_masks,
    receipt_checked,
)
from scripts.experiments.run_referral_budget030 import checked_csv
from scripts.experiments.run_rhythm_findings032 import challenge_table

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/experiment044_pipeline_v3_v1"
PRIOR035 = ROOT / "outputs/experiment035_hard_subset_v1"
PRIOR037 = ROOT / "outputs/experiment037_pipeline_v2_v1"
PRIOR043 = ROOT / "outputs/experiment043_ann_heads_v1/stage1"
PROTOCOL = "docs/experiment-044-pipeline-v3.md"
SOURCES = (
    "ecg_experiment/pipeline_v3.py", "ecg_experiment/pipeline_v2.py", "ecg_experiment/finding_screen.py",
    "ecg_experiment/intervals.py", "ecg_experiment/referral_budget.py", "ecg_experiment/hard_subset.py",
    "ecg_experiment/multisource_readout.py", "ecg_experiment/full_development.py",
    "ecg_experiment/hybrid_score.py", "ecg_experiment/rhythm_findings.py", "ecg_experiment/ann_heads.py",
    "scripts/experiments/run_pipeline_v3_044.py", "scripts/experiments/run_ann_heads043.py",
    "scripts/experiments/run_pipeline_v2_037.py", "scripts/experiments/run_finding_screen033.py",
    "scripts/experiments/run_rhythm_findings032.py", "scripts/experiments/run_referral_budget030.py",
    "scripts/experiments/run_lead_wave_maps042.py", "pyproject.toml", "uv.lock", PROTOCOL,
)
PIPELINES = ("v2", "v3", "v3b")
CONTRASTS = (("v3_minus_v2", "v3", "v2"), ("v3b_minus_v2", "v3b", "v2"), ("v3b_minus_v3", "v3b", "v3"))
HEAD_FEATURES = {"v2": "xecg", "v3": "concat", "pvc": "xecg", "wpw": "xecg", "v3b_pvc": "concat",
                 "v3b_wpw": "concat"}
SAVED_HEADS = ("v3", "pvc", "wpw", "v3b_pvc", "v3b_wpw")
FINDINGS = {"pvc": ("pvc", "v3b_pvc"), "wpw": ("wpw", "v3b_wpw")}
TOLERANCE = 1e-12
LOGIT_TOLERANCE = 1e-10
THREADS = 4
SMOKE_SIZES = (50, 100)
SMOKE_PRIMARY_SIZE = 100
SMOKE_RESAMPLES = 200
SMOKE_NORMALS = 2000
SMOKE_OUTCOMES = ("normal", "binary_positive", "composite", "pvc", "wpw", "finding_only")
LOG = logging.getLogger("experiment044")


def feature_sets(data: dict[str, Any], calibration_x: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    """
    Return the xECG and concatenated (xECG first, then JEPA) features of every part.

    Parameters
    ----------
    data : dict[str, Any]
        Output of Experiment 043's ``training_data``.
    calibration_x : dict[str, np.ndarray]
        Challenge calibration features per encoder.

    Returns
    -------
    dict[str, dict[str, Any]]
        Per feature set (``xecg``, ``concat``): ``stacked`` training rows, ``sph``, ``development`` and
        ``calibration``.
    """
    parts = {"stacked": data["x"], "sph": data["sph_x"], "development": data["development_x"],
             "calibration": calibration_x}
    return {"xecg": {part: values["xecg"] for part, values in parts.items()},
            "concat": {part: np.concatenate([values["xecg"], values["jepa"]], axis=1)
                       for part, values in parts.items()}}


def calibration_inputs() -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    """
    Return 032's Challenge calibration rows, their features per encoder and the clear normals among them.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]
        Rows (as Experiment 037's ``training_data``), features per encoder and the source-normal mask.
    """
    rows, challenge_x = challenge_table()
    in_calibration = (rows["split"] == "calibration").to_numpy()
    calibration = rows[in_calibration].reset_index(drop=True)
    standard = np.where(calibration["evaluable"].astype(bool), calibration["primary"], np.nan)
    source_normal = clear_normal(standard, calibration[list(GROUPS)])
    return calibration, {name: values[in_calibration] for name, values in challenge_x.items()}, source_normal


def head_plan(data: dict[str, Any], keep: tuple[np.ndarray, np.ndarray] | None = None
              ) -> dict[str, tuple[Any, ...]]:
    """
    Return the training rows, targets and weights of every head, as positions into the stacked rows.

    Parameters
    ----------
    data : dict[str, Any]
        Output of Experiment 043's ``training_data``.
    keep : tuple[np.ndarray, np.ndarray] | None
        Positions into the binary rows that the readouts may use, and a mask over the stacked rows that the
        finding heads may use (smoke mode); ``None`` keeps every row.

    Returns
    -------
    dict[str, tuple[Any, ...]]
        Per head: stacked positions, targets and weights (``None`` for the unweighted finding heads).
    """
    readouts, design = data["readouts"], data["design"]
    binary = np.arange(len(design["y"])) if keep is None else keep[0]
    readout = (readouts["binary"]["rows"][binary], design["y"][binary], design["weights"][binary])
    plan: dict[str, tuple[Any, ...]] = {"v2": readout, "v3": readout}
    for name, group in FINDING_HEADS.items():
        allowed = np.ones(len(readouts[group]["rows"]), dtype=bool) if keep is None else keep[1][
            readouts[group]["rows"]]
        for head in FINDINGS[name]:
            plan[head] = (readouts[group]["rows"][allowed], readouts[group]["y"][allowed], None)
    return plan


def fit_heads(sets: dict[str, dict[str, Any]], plan: dict[str, tuple[Any, ...]], parts: tuple[str, ...]
              ) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]], dict[str, int]]:
    """
    Fit every head with one BLAS thread and score the named parts.

    Parameters
    ----------
    sets : dict[str, dict[str, Any]]
        Output of ``feature_sets`` (or the smoke equivalent).
    plan : dict[str, tuple[Any, ...]]
        Output of ``head_plan``.
    parts : tuple[str, ...]
        Parts to score.

    Returns
    -------
    tuple[dict[str, Any], dict[str, dict[str, np.ndarray]], dict[str, int]]
        Heads, probabilities per head and part, and iterations per head.
    """
    heads, scores, iterations = {}, {}, {}
    with threadpool_limits(limits=1):
        for name, (rows, y, weights) in plan.items():
            began = time.perf_counter()
            x = sets[HEAD_FEATURES[name]]["stacked"][rows]
            heads[name] = fit_logistic(x, y) if weights is None else fit_readout(x, y, weights)
            scores[name] = {part: predict(heads[name], sets[HEAD_FEATURES[name]][part]) for part in parts}
            model: LogisticRegression = heads[name][1]
            iterations[name] = int(model.n_iter_[0])
            LOG.info("fit %s on %s rows x %s: %s iterations, %.1f s", name, len(rows), x.shape[1],
                     iterations[name], time.perf_counter() - began)
    return heads, scores, iterations


def standardization(scores: dict[str, dict[str, np.ndarray]], normal_part: str, normal: np.ndarray | None,
                    scored_part: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Standardize every finding head on its source normals, as Experiments 033 and 037 do.

    Parameters
    ----------
    scores : dict[str, dict[str, np.ndarray]]
        Output of ``fit_heads``.
    normal_part : str
        Part that holds the source normals.
    normal : np.ndarray | None
        Mask of the source normals in that part, or ``None`` for all of it.
    scored_part : str
        Part to standardize.

    Returns
    -------
    tuple[dict[str, np.ndarray], dict[str, np.ndarray]]
        Constants (``{head}_logit_mean`` and ``{head}_logit_sd``) and z-scores per finding head.
    """
    constants, z = {}, {}
    for heads in FINDINGS.values():
        for name in heads:
            source = scores[name][normal_part] if normal is None else scores[name][normal_part][normal]
            mean, spread = normal_standardizer(clipped_logits(source)[0])
            constants[f"{name}_logit_mean"] = np.array([mean])
            constants[f"{name}_logit_sd"] = np.array([spread])
            z[name] = finding_z(scores[name][scored_part], source)
    return constants, z


def check_v2(data: dict[str, Any], scores: dict[str, dict[str, np.ndarray]], z: dict[str, np.ndarray],
             constants: dict[str, np.ndarray], sets: dict[str, dict[str, Any]]) -> dict[str, float]:
    """
    Require the refitted v2 readout and xECG finding heads to reproduce Experiment 037's saved outputs.

    Parameters
    ----------
    data : dict[str, Any]
        Output of Experiment 043's ``training_data``.
    scores : dict[str, dict[str, np.ndarray]]
        Output of ``fit_heads``.
    z : dict[str, np.ndarray]
        SPH z-scores from ``standardization``.
    constants : dict[str, np.ndarray]
        Standardization constants from ``standardization``.
    sets : dict[str, dict[str, Any]]
        Output of ``feature_sets``.

    Returns
    -------
    dict[str, float]
        Largest absolute difference per check.

    Raises
    ------
    ValueError
        If a row order differs or a difference exceeds ``TOLERANCE``.
    """
    with np.load(PRIOR037 / "predictions.npz") as saved:
        if not (np.array_equal(saved["sph_ecg_ids"], data["sph"]["ecg_id"].to_numpy(dtype=str))
                and np.array_equal(saved["development_record_ids"],
                                   data["development"].index.to_numpy(dtype=str))):
            raise ValueError("037 prediction rows differ")
        differences = {f"v2_{part}": float(np.abs(scores["v2"][part] - saved[f"{part}_v2_binary"]).max())
                       for part in ("sph", "development")}
        for name in FINDING_HEADS:
            differences[f"z_{name}"] = float(np.abs(z[name] - saved[f"sph_z_{name}"]).max())
        combined = np.maximum(z["pvc"], z["wpw"])
        differences["z_combined"] = float(np.abs(combined - saved["sph_z_combined"]).max())
    with np.load(PRIOR037 / "pipeline_v2_heads.npz") as saved:
        heads037 = {key: saved[key] for key in saved.files}
    for key, value in constants.items():
        if key in heads037:
            differences[key] = float(np.abs(value - heads037[key]).max())
    for name in ("v2", *FINDING_HEADS):
        differences[f"{name}_vs_037_parameters"] = float(np.abs(
            score_parameters(heads037, name, sets["xecg"]["sph"]) - scores[name]["sph"]).max())
    if max(differences.values()) > TOLERANCE:
        raise ValueError(f"v2 does not reproduce 037: {differences}")
    return differences


def check_v3(data: dict[str, Any], heads: dict[str, Any], sets: dict[str, dict[str, Any]]
             ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """
    Require the v3 readout's logits to reproduce Experiment 043 Stage 1's saved ``logistic_concat``.

    Parameters
    ----------
    data : dict[str, Any]
        Output of Experiment 043's ``training_data``.
    heads : dict[str, Any]
        Output of ``fit_heads``.
    sets : dict[str, dict[str, Any]]
        Output of ``feature_sets``.

    Returns
    -------
    tuple[dict[str, np.ndarray], dict[str, float]]
        v3 logits per part and the largest absolute difference per part.

    Raises
    ------
    ValueError
        If a row order differs or a difference exceeds ``LOGIT_TOLERANCE``.
    """
    logits = {part: logits_of(heads["v3"], sets["concat"][part]) for part in ("sph", "development")}
    with np.load(PRIOR043 / "predictions.npz") as saved:
        if not (np.array_equal(saved["sph_ecg_ids"], data["sph"]["ecg_id"].to_numpy(dtype=str))
                and np.array_equal(saved["development_record_ids"],
                                   data["development"].index.to_numpy(dtype=str))):
            raise ValueError("043 prediction rows differ")
        differences = {part: float(np.abs(logits[part] - saved[f"{part}_logistic_concat"]).max())
                       for part in logits}
    if max(differences.values()) > LOGIT_TOLERANCE:
        raise ValueError(f"v3 does not reproduce 043's logistic_concat: {differences}")
    return logits, differences


def saved_heads(heads: dict[str, Any], constants: dict[str, np.ndarray], scores: dict[str, dict[str, Any]],
                sets: dict[str, dict[str, Any]]) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """
    Collect the parameters of v3's readout and the finding heads, and require them to reproduce the fits.

    Parameters
    ----------
    heads : dict[str, Any]
        Output of ``fit_heads``.
    constants : dict[str, np.ndarray]
        Output of ``standardization``.
    scores : dict[str, dict[str, Any]]
        Output of ``fit_heads``.
    sets : dict[str, dict[str, Any]]
        Output of ``feature_sets``.

    Returns
    -------
    tuple[dict[str, np.ndarray], dict[str, float]]
        Arrays for ``pipeline_v3_heads.npz`` and the largest difference per head.

    Raises
    ------
    ValueError
        If a difference exceeds ``TOLERANCE``.
    """
    parameters = {key: value for name in SAVED_HEADS
                  for key, value in head_parameters(heads[name], name).items()}
    differences = {name: float(np.abs(score_parameters(parameters, name, sets[HEAD_FEATURES[name]]["sph"])
                                      - scores[name]["sph"]).max()) for name in SAVED_HEADS}
    if max(differences.values()) > TOLERANCE:
        raise ValueError(f"Saved parameters do not reproduce the heads: {differences}")
    return {**parameters, **constants}, differences


def run_draws(pipelines: dict[str, dict[str, np.ndarray]], evaluation: np.ndarray,
              masks: dict[str, np.ndarray], local_masks: dict[str, np.ndarray], local_normal: np.ndarray,
              sizes: tuple[int, ...], primary_size: int
              ) -> tuple[pd.DataFrame, dict[tuple[str, str, int, int], np.ndarray]]:
    """
    Thresholds and outcomes of every pipeline, rule, size, budget and draw, as Experiment 037's ``run_draws``.

    Parameters
    ----------
    pipelines : dict[str, dict[str, np.ndarray]]
        Per pipeline the ``binary``, ``pvc``, ``wpw`` and ``combined`` scores of every row.
    evaluation : np.ndarray
        Evaluation mask over the rows.
    masks : dict[str, np.ndarray]
        Outcome masks over the evaluation rows.
    local_masks : dict[str, np.ndarray]
        Outcome masks over the local-pool rows.
    local_normal : np.ndarray
        Row positions of the local normals, in their fixed order.
    sizes : tuple[int, ...]
        Numbers of local normals.
    primary_size : int
        Size at which the other 033 rules run (with the primary budget).

    Returns
    -------
    tuple[pd.DataFrame, dict[tuple[str, str, int, int], np.ndarray]]
        One row per pipeline, rule, size, budget and draw; and, for the main rules, the referral share over
        draws of each evaluation row.
    """
    plans = local_normal_plans(local_normal, sizes, DRAWS, DRAW_SEED)
    jobs = [(pipeline, rule, m, budget) for pipeline in pipelines for rule in RULES for m in sizes
            for budget in BUDGETS if rule == "binary" or budget * m // 1000 >= 1]
    jobs += [(pipeline, rule, primary_size, PRIMARY_BUDGET) for pipeline in pipelines for rule in RULES033
             if rule not in RULES]
    records, shares = [], {}
    for pipeline, rule, m, budget in jobs:
        columns, rule_shares = RULES033[rule]
        scores = pipelines[pipeline]
        matrix = np.column_stack([scores["binary"], *(scores[column] for column in columns)])
        found = screen_draws(matrix, plans[m], budget, rule_shares, evaluation, masks, local_masks)
        if rule in RULES:
            shares[(pipeline, rule, m, budget)] = found["shares"]
        for draw in range(DRAWS):
            records.append({
                "pipeline": pipeline, "rule": rule, "m": m, "budget": budget / 1000, "draw": draw,
                **{f"threshold_{index}": float(value)
                   for index, value in enumerate(found["thresholds"][draw])},
                **{name: float(values[draw]) for name, values in found["outcomes"].items()},
                **{f"local_{name}": float(values[draw]) for name, values in found["local"].items()}})
    return pd.DataFrame(records), shares


def check_v2_draws(draws: pd.DataFrame) -> dict[str, Any]:
    """
    Require the v2 draws to equal the v2 rows of Experiment 037's ``draws.csv`` exactly.

    Parameters
    ----------
    draws : pd.DataFrame
        Output of ``run_draws``.

    Returns
    -------
    dict[str, Any]
        Rows compared and the largest difference.

    Raises
    ------
    ValueError
        If any value differs.
    """
    prior = checked_csv(PRIOR037, "draws.csv", float_precision="round_trip")
    prior = prior[prior["pipeline"] == "v2"].drop(columns="pipeline")
    ours = draws[draws["pipeline"] == "v2"].drop(columns="pipeline")
    keys = ["rule", "m", "budget", "draw"]
    if sorted(ours.columns) != sorted(prior.columns):
        raise ValueError("Draw columns differ from 037's")
    columns = [column for column in prior.columns if column not in keys]
    largest = largest_difference(ours, prior, keys, [(column, f"{column}_prior") for column in columns])
    if largest != 0.0:
        raise ValueError(f"The v2 draws differ from 037's by {largest}")
    return {"rows": len(ours), "largest_difference": largest}


def bootstrap(patients: np.ndarray, shares: dict[tuple[str, str, int, int], np.ndarray],
              masks: dict[str, np.ndarray], resamples: int
              ) -> tuple[dict[str, np.ndarray], list[tuple[str, str, int, int]], np.ndarray, dict[str, int]]:
    """
    Resampled referral rates of every screen and outcome over 037's whole-patient resamples.

    Parameters
    ----------
    patients : np.ndarray
        Patient of each evaluation row.
    shares : dict[tuple[str, str, int, int], np.ndarray]
        Output of ``run_draws``.
    masks : dict[str, np.ndarray]
        Outcome masks over the evaluation rows.
    resamples : int
        Number of resamples.

    Returns
    -------
    tuple[dict[str, np.ndarray], list[tuple[str, str, int, int]], np.ndarray, dict[str, int]]
        ``(resamples, keys)`` rates per outcome, the keys in column order, the share matrix and the number
        of resamples without an ECG of each outcome.
    """
    keys = list(shares)
    matrix = np.column_stack([shares[key] for key in keys])
    counts = resample_counts(patients, resamples, BOOTSTRAP_SEED)
    boot = {name: resampled_rates(counts, matrix, mask) for name, mask in masks.items()}
    empty = {name: int(np.isnan(values[:, 0]).sum()) for name, values in boot.items()}
    return boot, keys, matrix, empty


def summarize(draws: pd.DataFrame, boot: dict[str, np.ndarray], keys: list[tuple[str, str, int, int]]
              ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Per-screen summaries and the paired pipeline and rule minus ``binary`` contrasts, as Experiment 037.

    Parameters
    ----------
    draws : pd.DataFrame
        Output of ``run_draws``.
    boot : dict[str, np.ndarray]
        ``(resamples, keys)`` rates per outcome.
    keys : list[tuple[str, str, int, int]]
        Keys in the column order of ``boot``.

    Returns
    -------
    tuple[list[dict[str, Any]], list[dict[str, Any]]]
        Summary rows and contrast rows.
    """
    groups = draws.groupby(["pipeline", "rule", "m", "budget"], sort=False)
    column = {key: index for index, key in enumerate(keys)}
    outcomes = list(boot)
    summary, contrasts = [], []
    for key in keys:
        pipeline, rule, m, budget = key
        group = groups.get_group((pipeline, rule, m, budget / 1000))
        row: dict[str, Any] = {"pipeline": pipeline, "rule": rule, "m": m, "budget": budget / 1000,
                               "threshold_0_mean": float(group["threshold_0"].mean()),
                               "share_rate_within_1pp": float(((group["normal"] - budget / 1000).abs()
                                                               <= 0.01 + 1e-12).mean())}
        for name in outcomes:
            values = group[name]
            row[name] = {"mean": float(values.mean()), "p5": float(values.quantile(0.05)),
                         "p95": float(values.quantile(0.95)),
                         "ci": percentile_interval(boot[name][:, column[key]])}
        for label in ("composite", "binary_positive"):
            per_draw = referrals_per_1000(group[label], group["normal"], PREVALENCE)
            resampled = referrals_per_1000(boot[label][:, column[key]], boot["normal"][:, column[key]],
                                           PREVALENCE)
            row[f"per_1000_{label}"] = {
                "referrals": float(per_draw.mean()), "referrals_ci": percentile_interval(resampled),
                "caught": float(1000 * PREVALENCE * group[label].mean()),
                "caught_ci": percentile_interval(1000 * PREVALENCE * boot[label][:, column[key]])}
        summary.append(row)
        references = [(label, (second, rule, m, budget)) for label, first, second in CONTRASTS
                      if first == pipeline and (second, rule, m, budget) in column]
        if rule != "binary":
            references.append((f"{pipeline}_{rule}_minus_binary", (pipeline, "binary", m, budget)))
        for label, reference in references:
            base = groups.get_group((reference[0], reference[1], m, budget / 1000))
            for name in outcomes:
                difference = boot[name][:, column[key]] - boot[name][:, column[reference]]
                contrasts.append({"contrast": label, "rule": rule, "m": m, "budget": budget / 1000,
                                  "outcome": name,
                                  "difference": float(group[name].mean() - base[name].mean()),
                                  "ci": percentile_interval(difference)})
    return summary, contrasts


def local_selection(draws: pd.DataFrame, primary_size: int) -> dict[str, Any]:
    """
    033's local-pool selection step at the primary size and budget, for each pipeline (as Experiment 037).

    Parameters
    ----------
    draws : pd.DataFrame
        Output of ``run_draws``.
    primary_size : int
        Number of local normals.

    Returns
    -------
    dict[str, Any]
        Per pipeline, the table of local composite sensitivity and binary cost, and the selected rule.
    """
    at = draws[(draws["m"] == primary_size) & (draws["budget"] == PRIMARY_BUDGET / 1000)]
    means = at.groupby(["pipeline", "rule"])[["local_binary_positive", "local_composite"]].mean()
    selection = {}
    for pipeline in at["pipeline"].unique():
        base = means.loc[(pipeline, "binary"), "local_binary_positive"]
        table = [{"rule": rule, "local_composite": float(means.loc[(pipeline, rule), "local_composite"]),
                  "local_binary_cost": float(base - means.loc[(pipeline, rule), "local_binary_positive"])}
                 for rule in RULES033 if rule != "binary"]
        eligible = [row for row in table if row["local_binary_cost"] <= MAX_BINARY_COST + TOLERANCE]
        if eligible:
            best = max(row["local_composite"] for row in eligible)
            selected = next(row["rule"] for row in eligible if row["local_composite"] >= best - TOLERANCE)
        else:
            cheapest = min(row["local_binary_cost"] for row in table)
            selected = next(row["rule"] for row in table if row["local_binary_cost"] <= cheapest + TOLERANCE)
        selection[str(pipeline)] = {"table": table, "selected": selected}
    return selection


def decide(contrasts: list[dict[str, Any]], primary_size: int) -> dict[str, Any]:
    """
    Apply the pre-registered reading to the primary contrast, and report the v3b contrasts beside it.

    Parameters
    ----------
    contrasts : list[dict[str, Any]]
        Output of ``summarize``.
    primary_size : int
        Number of local normals of the primary screen.

    Returns
    -------
    dict[str, Any]
        The primary contrast, the decision, and the v3b readings (no decision).
    """
    def contrast(label: str, outcome: str) -> dict[str, Any]:
        (row,) = [row for row in contrasts if row["contrast"] == label and row["rule"] == PRIMARY_RULE
                  and row["m"] == primary_size and row["budget"] == PRIMARY_BUDGET / 1000
                  and row["outcome"] == outcome]
        return row

    primary = contrast("v3_minus_v2", "composite")
    found = {"primary": primary, "decision": pipeline_reading(primary["ci"], WORSE_MARGIN),
             "margin": WORSE_MARGIN}
    for label in ("v3b_minus_v2", "v3b_minus_v3"):
        found[f"{label}_reading"] = pipeline_reading(contrast(label, "composite")["ci"], WORSE_MARGIN)
    return found


def check_shares(draws: pd.DataFrame, matrix: np.ndarray, keys: list[tuple[str, str, int, int]],
                 composite: np.ndarray) -> None:
    """
    Require every screen's mean referral share of the composite rows to equal its draw mean.

    Raises
    ------
    ValueError
        If a screen differs by more than 1e-9.
    """
    for index, key in enumerate(keys):
        group = draws[(draws["pipeline"] == key[0]) & (draws["rule"] == key[1]) & (draws["m"] == key[2])
                      & (draws["budget"] == key[3] / 1000)]
        if abs(matrix[composite, index].mean() - group["composite"].mean()) > 1e-9:
            raise ValueError(f"Referral shares do not reproduce the draw mean for {key}")


def auroc_tables(data: dict[str, Any], v2: dict[str, np.ndarray], v3: dict[str, np.ndarray],
                 v3_logits: dict[str, np.ndarray], rows: pd.DataFrame,
                 scores: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    """
    AUROC of v2 and v3 on Experiment 037's sets, and of the xECG and concatenated finding heads at SPH.

    Parameters
    ----------
    data : dict[str, Any]
        Output of Experiment 043's ``training_data``.
    v2, v3 : dict[str, np.ndarray]
        Binary probabilities per part (``development``, ``sph``).
    v3_logits : dict[str, np.ndarray]
        v3 logits per part.
    rows : pd.DataFrame
        033's ``load_rows`` rows.
    scores : dict[str, dict[str, np.ndarray]]
        Output of ``fit_heads``.

    Returns
    -------
    dict[str, Any]
        ``binary`` per set and ``findings`` per finding, with paired differences and the checks.

    Raises
    ------
    ValueError
        If v2's AUROCs differ from 037's or v3's from 043's.
    """
    sets = evaluation_sets(data)
    checks = {"v2_vs_037": check_comparator_aurocs(sets, v2)}
    prior = json.loads((PRIOR043 / "result.json").read_text())["detection"]["logistic_concat"]["sets"]
    binary = {}
    for name, spec in sets.items():
        found = {pipeline: float(roc_auc_score(spec["y"], set_scores(spec, values)))
                 for pipeline, values in (("v2", v2), ("v3", v3))}
        from_logits = float(roc_auc_score(spec["y"], set_scores(spec, v3_logits)))
        if max(abs(found["v3"] - prior[name]["auroc"]), abs(from_logits - prior[name]["auroc"])) > TOLERANCE:
            raise ValueError(f"v3's {name} AUROC differs from 043's")
        binary[name] = {"records": len(spec["y"]), "positives": int(spec["y"].sum()),
                        "patients": int(len(np.unique(spec["patients"]))), "auroc": found,
                        "v3_minus_v2": paired_auroc_difference(
                            spec["patients"], spec["y"], set_scores(spec, v3), set_scores(spec, v2),
                            RESAMPLES, BOOTSTRAP_SEED)}
    checks["v3_vs_043"] = {name: binary[name]["auroc"]["v3"] for name in binary}
    findings = {}
    for name, group in FINDING_HEADS.items():
        defined = rows[group].notna().to_numpy()
        y = (rows.loc[defined, group] == 1).to_numpy(dtype=np.int64)
        patients = rows.loc[defined, "patient_id"].to_numpy(dtype=str)
        xecg, concat = (scores[head]["sph"][defined] for head in FINDINGS[name])
        findings[name] = {"records": int(defined.sum()), "positives": int(y.sum()),
                          "auroc": {"xecg": float(roc_auc_score(y, xecg)),
                                    "concat": float(roc_auc_score(y, concat))},
                          "concat_minus_xecg": paired_auroc_difference(patients, y, concat, xecg, RESAMPLES,
                                                                       BOOTSTRAP_SEED)}
    return {"binary": binary, "findings": findings, "checks": checks}


def identity(receipts: dict[str, Any], data_identity: dict[str, Any]) -> dict[str, Any]:
    """
    Hash every source and the protocol, with the input receipts and the run environment.

    Parameters
    ----------
    receipts : dict[str, Any]
        Hashes of earlier outputs, checked against their receipts, keyed by repository-relative path.
    data_identity : dict[str, Any]
        Input identity from Experiment 043's ``training_data``.

    Returns
    -------
    dict[str, Any]
        The identity block of ``result.json``.

    Raises
    ------
    RuntimeError
        If ``ecg_experiment`` is imported from outside this checkout.
    """
    if Path(ecg_experiment.__file__).resolve().parents[1] != ROOT:
        raise RuntimeError(f"ecg_experiment is imported from {ecg_experiment.__file__}, not {ROOT}")
    return {"inputs": receipts, "data": data_identity,
            "sources": {name: sha256_file(ROOT / name) for name in SOURCES}, "git_head": git_head(ROOT),
            "openblas_architectures": blas_architectures()}


def input_receipts() -> dict[str, str]:
    """Return the hashes of the earlier outputs this run reads, each checked against its receipt."""
    found = {}
    for directory, names in ((PRIOR037, ("predictions.npz", "draws.csv", "pipeline_v2_heads.npz")),
                             (PRIOR043, ("predictions.npz",)), (PRIOR035, ("predictions.npz",))):
        found.update({to_stored(directory / name): digest
                      for name, digest in receipt_checked(directory, names).items()})
    return found


def run_full(data: dict[str, Any], partial: Path) -> dict[str, Any]:
    """
    Run the experiment: checks, the three pipelines over 037's draws and resamples, and the AUROCs.

    Parameters
    ----------
    data : dict[str, Any]
        Output of Experiment 043's ``training_data``.
    partial : Path
        Output folder.

    Returns
    -------
    dict[str, Any]
        Result fields.
    """
    began = time.perf_counter()
    checks: dict[str, Any] = {"training_rows_vs_037": check_against_037(data)}
    calibration, calibration_x, source_normal = calibration_inputs()
    sets = feature_sets(data, calibration_x)
    LOG.info("features %s, calibration normals %s at %.1f s",
             {name: list(values["stacked"].shape) for name, values in sets.items()}, int(source_normal.sum()),
             time.perf_counter() - began)
    heads, scores, iterations = fit_heads(sets, head_plan(data), ("sph", "development", "calibration"))
    constants, z = standardization(scores, "calibration", source_normal, "sph")
    checks["v2_vs_037"] = check_v2(data, scores, z, constants, sets)
    v3_logits, checks["v3_vs_043_logits"] = check_v3(data, heads, sets)
    parameters, checks["saved_parameters"] = saved_heads(heads, constants, scores, sets)
    checks["source_normals"] = int(source_normal.sum())
    LOG.info("heads reproduce: %s", checks)

    rows, site, local_normal = load_rows()
    counts = counts_of(rows)
    if json.loads(json.dumps(counts)) != EXPECTED033:
        raise ValueError("Counts differ from 033's")
    if not np.array_equal(rows["ecg_id"].to_numpy(dtype=str), data["sph"]["ecg_id"].to_numpy(dtype=str)):
        raise ValueError("033's SPH rows differ from 032's")
    with threadpool_limits(limits=THREADS):
        scores033, checks["033_load_scores"] = load_scores(rows, site)
        with np.load(PRIOR037 / "predictions.npz") as saved:
            frozen_v2 = {part: saved[f"{part}_v2_binary"] for part in ("sph", "development")}
            checks["z_033_vs_037"] = max(float(np.abs(scores033[name] - saved[f"sph_z_{name}"]).max())
                                         for name in ("pvc", "wpw", "combined"))
        if checks["z_033_vs_037"] > TOLERANCE:
            raise ValueError(f"033's z-scores differ from 037's by {checks['z_033_vs_037']}")
        findings033 = {name: scores033[name] for name in ("pvc", "wpw", "combined")}
        pipelines = {"v2": {"binary": frozen_v2["sph"], **findings033},
                     "v3": {"binary": scores["v3"]["sph"], **findings033},
                     "v3b": {"binary": scores["v3"]["sph"], "pvc": z["v3b_pvc"], "wpw": z["v3b_wpw"],
                             "combined": np.maximum(z["v3b_pvc"], z["v3b_wpw"])}}
        evaluation = rows["evaluation"].to_numpy()
        masks = outcome_masks(rows, evaluation)
        local_masks = {name: rows.loc[~evaluation, name].to_numpy() for name in LOCAL_OUTCOMES}
        draws, shares = run_draws(pipelines, evaluation, masks, local_masks, local_normal, SIZES,
                                  PRIMARY_SIZE)
        checks["v2_draws_vs_037"] = check_v2_draws(draws)
        LOG.info("draws done at %.1f s: %s", time.perf_counter() - began, checks["v2_draws_vs_037"])
        boot, keys, matrix, empty = bootstrap(rows.loc[evaluation, "patient_id"].to_numpy(dtype=str), shares,
                                              masks, RESAMPLES)
    check_shares(draws, matrix, keys, masks["composite"])
    summary, contrasts = summarize(draws, boot, keys)
    reading = decide(contrasts, PRIMARY_SIZE)
    selection = local_selection(draws, PRIMARY_SIZE)
    if selection["v2"]["selected"] != PRIMARY_RULE:
        raise ValueError("The v2 local selection does not reproduce 037's")
    aurocs = auroc_tables(data, frozen_v2, {part: scores["v3"][part] for part in ("sph", "development")},
                          v3_logits, rows, scores)
    LOG.info("summaries done at %.1f s: %s", time.perf_counter() - began, reading)

    draws.to_csv(partial / "draws.csv", index=False)
    write_npz_atomic(partial / "pipeline_v3_heads.npz", **parameters)
    write_npz_atomic(
        partial / "predictions.npz",
        sph_ecg_ids=rows["ecg_id"].to_numpy(dtype=str),
        sph_patient_ids=rows["patient_id"].to_numpy(dtype=str),
        sph_evaluation=evaluation, development_record_ids=data["development"].index.to_numpy(dtype=str),
        sph_v2_binary=frozen_v2["sph"], sph_v3_binary=scores["v3"]["sph"],
        **{f"sph_{pipeline}_z_{name}": pipelines[pipeline][name] for pipeline in ("v2", "v3b")
           for name in ("pvc", "wpw", "combined")},
        development_v2_binary=frozen_v2["development"], development_v3_binary=scores["v3"]["development"],
        **{f"{part}_v3_logit": v3_logits[part] for part in ("sph", "development")})
    written = ("draws.csv", "pipeline_v3_heads.npz", "predictions.npz")
    return {"checks": checks, "iterations": iterations, "counts": counts,
            "standardization": {name: float(values[0]) for name, values in constants.items()},
            "empty_resamples": empty, "summary": summary, "contrasts": contrasts,
            "local_selection": selection,
            "auroc": aurocs, "reading": reading, "resamples": RESAMPLES, "sizes": list(SIZES),
            "outputs_sha256": {name: sha256_file(partial / name) for name in written}}


def run_smoke(data: dict[str, Any], partial: Path) -> dict[str, Any]:
    """
    Run every code path on training rows only: no development, calibration or SPH ECG is scored.

    The readouts are fitted on 043's smoke fit rows and the finding heads on their rows outside the held-out
    part; the finding heads are standardized on readout-training normals. The held-out part is split by group
    into a local pool and an evaluation half.

    Parameters
    ----------
    data : dict[str, Any]
        Output of Experiment 043's ``training_data``.
    partial : Path
        Output folder.

    Returns
    -------
    dict[str, Any]
        Result fields.
    """
    split = smoke_split(data)
    binary_rows, y = data["readouts"]["binary"]["rows"], data["design"]["y"]
    held = binary_rows[split["held_out"]]
    allowed = np.ones(len(data["x"]["xecg"]), dtype=bool)
    allowed[held] = False
    fit_normals = binary_rows[split["fit"][y[split["fit"]] == 0]][:SMOKE_NORMALS]
    stacked = {"xecg": data["x"]["xecg"], "concat": np.concatenate([data["x"]["xecg"], data["x"]["jepa"]],
                                                                   axis=1)}
    sets = {name: {"stacked": values, "held_out": values[held], "normals": values[fit_normals]}
            for name, values in stacked.items()}
    plan = head_plan(data, (split["fit"], allowed))
    heads, scores, iterations = fit_heads(sets, plan, ("held_out", "normals"))
    constants, z = standardization(scores, "normals", None, "held_out")

    labels = {}
    for name, group in FINDING_HEADS.items():
        values = np.zeros(len(stacked["xecg"]), dtype=bool)
        readout = data["readouts"][group]
        values[readout["rows"][readout["y"] == 1]] = True
        labels[name] = values[held]
    positive = y[split["held_out"]] == 1
    composite = positive | labels["pvc"] | labels["wpw"]
    outcomes = {"normal": ~positive, "binary_positive": positive, "composite": composite,
                "pvc": labels["pvc"], "wpw": labels["wpw"], "finding_only": composite & ~positive}
    groups = data["groups"][split["held_out"]]
    evaluation = validation_mask(groups, 0.5, DRAW_SEED)
    masks = {name: values[evaluation] for name, values in outcomes.items()}
    local_masks = {name: outcomes[name][~evaluation] for name in LOCAL_OUTCOMES}
    local_normal = np.flatnonzero(~evaluation & outcomes["normal"])
    xecg_z = {name: z[name] for name in FINDING_HEADS}
    pipelines = {"v2": {"binary": scores["v2"]["held_out"], **xecg_z,
                        "combined": np.maximum(xecg_z["pvc"], xecg_z["wpw"])},
                 "v3": {"binary": scores["v3"]["held_out"], **xecg_z,
                        "combined": np.maximum(xecg_z["pvc"], xecg_z["wpw"])},
                 "v3b": {"binary": scores["v3"]["held_out"], "pvc": z["v3b_pvc"], "wpw": z["v3b_wpw"],
                         "combined": np.maximum(z["v3b_pvc"], z["v3b_wpw"])}}
    with threadpool_limits(limits=THREADS):
        draws, shares = run_draws(pipelines, evaluation, masks, local_masks, local_normal, SMOKE_SIZES,
                                  SMOKE_PRIMARY_SIZE)
        boot, keys, matrix, empty = bootstrap(groups[evaluation], shares, masks, SMOKE_RESAMPLES)
    check_shares(draws, matrix, keys, masks["composite"])
    summary, contrasts = summarize(draws, boot, keys)
    reading = decide(contrasts, SMOKE_PRIMARY_SIZE)
    parameters = {key: value for name in SAVED_HEADS
                  for key, value in head_parameters(heads[name], name).items()}
    reproduced = {name: float(np.abs(score_parameters(parameters, name, sets[HEAD_FEATURES[name]]["held_out"])
                                     - scores[name]["held_out"]).max()) for name in SAVED_HEADS}
    held_y = y[split["held_out"]][evaluation]
    auroc = {"v2": float(roc_auc_score(held_y, scores["v2"]["held_out"][evaluation])),
             "v3": float(roc_auc_score(held_y, scores["v3"]["held_out"][evaluation])),
             "v3_minus_v2": paired_auroc_difference(groups[evaluation], held_y,
                                                    scores["v3"]["held_out"][evaluation],
                                                    scores["v2"]["held_out"][evaluation], SMOKE_RESAMPLES,
                                                    BOOTSTRAP_SEED)}
    draws.to_csv(partial / "draws.csv", index=False)
    write_npz_atomic(partial / "predictions.npz", held_out_rows=split["held_out"],
                     **{f"held_out_{name}": values["held_out"] for name, values in scores.items()})
    return {"split": {key: int(len(value)) for key, value in split.items()},
            "held_out": {"evaluation": int(evaluation.sum()), "local": int((~evaluation).sum()),
                         "local_normals": int(len(local_normal)),
                         **{name: int(values.sum()) for name, values in masks.items()}},
            "iterations": iterations,
            "standardization": {key: float(value[0]) for key, value in constants.items()},
            "saved_parameters": reproduced, "empty_resamples": empty, "summary": summary,
            "contrasts": contrasts, "local_selection": local_selection(draws, SMOKE_PRIMARY_SIZE),
            "reading": reading, "auroc": auroc, "resamples": SMOKE_RESAMPLES, "sizes": list(SMOKE_SIZES),
            "outputs_sha256": {name: sha256_file(partial / name)
                               for name in ("draws.csv", "predictions.npz")},
            "note": "smoke: training rows only; the readings are not results"}


def run(smoke: bool, output: Path) -> None:
    """
    Run the experiment (or its smoke test) end to end and write its outputs.

    Parameters
    ----------
    smoke : bool
        Training-only smoke test.
    output : Path
        Final output folder; it must not exist.
    """
    started = time.perf_counter()
    partial = output.with_name(output.name + ".partial")
    partial.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(partial / "run.log")])
    receipts = input_receipts()
    data = training_data043()
    LOG.info("loaded %s at %.1f s", data["training_counts"], time.perf_counter() - started)
    run_identity = identity(receipts, data["identity"])
    found = run_smoke(data, partial) if smoke else run_full(data, partial)
    write_json_atomic(partial / "result.json", {
        "status": "smoke" if smoke else "complete", "identity": run_identity,
        "training_counts": data["training_counts"], "pipelines": {
            "v2": "037 pipeline v2 (frozen scores)",
            "v3": "v2 with the readout on xECG + JEPA (043 logistic_concat)",
            "v3b": "v3 with the PVC and WPW heads on xECG + JEPA"},
        "feature_order": "concat = xECG (1,024) then JEPA (768)",
        "rules": {name: {"columns": list(RULES033[name][0]), "shares_per_mille": list(RULES033[name][1])}
                  for name in RULES033},
        "draw_seed": DRAW_SEED, "bootstrap_seed": BOOTSTRAP_SEED, "draws": DRAWS,
        "budgets": [budget / 1000 for budget in BUDGETS], "prevalence": PREVALENCE, **found,
        "total_seconds": time.perf_counter() - started,
        "challenge_test_read": False, "ptbxl_calibration_read": False, "ptbxl_test_read": False,
    })
    partial.rename(output)
    LOG.info("done in %.1f s", time.perf_counter() - started)


def main() -> None:
    """Parse arguments and run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="training-only smoke test")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    if arguments.output.exists() or arguments.output.with_name(arguments.output.name + ".partial").exists():
        raise FileExistsError(f"Refusing to overwrite {arguments.output}")
    run(arguments.smoke, arguments.output)


if __name__ == "__main__":
    main()
