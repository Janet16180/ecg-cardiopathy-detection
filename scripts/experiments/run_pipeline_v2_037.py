"""Experiment 037: the candidate screening pipeline (v2 readout, 030 budget threshold, 033 finding rule).

Refits the 022b pooled readout (v1) and 035's ``dropped_upweighted`` readout (v2) on xECG features, refits
032's PVC and pre-excitation heads, checks every reproduction of the protocol, and recomputes 030's and 033's
operating numbers on SPH for both readouts.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from ecg_experiment.files import sha256_file, write_json_atomic, write_npz_atomic
from ecg_experiment.finding_screen import clipped_logits, finding_z, referred_matrix, split_thresholds
from ecg_experiment.full_development import fit_logistic, predict
from ecg_experiment.hard_subset import ARM_SPECS, arm_design
from ecg_experiment.hybrid_score import normal_standardizer
from ecg_experiment.intervals import paired_auroc_difference
from ecg_experiment.multisource_readout import fit_readout
from ecg_experiment.pipeline_v2 import (
    head_parameters,
    percentile_interval,
    resample_counts,
    resampled_rates,
    score_parameters,
    v2_is_worse,
)
from ecg_experiment.referral_budget import referrals_per_1000
from ecg_experiment.rhythm_findings import GROUPS, clear_normal
from scripts.experiments.run_finding_screen033 import BUDGETS as BUDGETS033
from scripts.experiments.run_finding_screen033 import EXPECTED as EXPECTED033
from scripts.experiments.run_finding_screen033 import RULES as RULES033
from scripts.experiments.run_finding_screen033 import SIZES as SIZES033
from scripts.experiments.run_finding_screen033 import counts_of, load_rows, load_scores
from scripts.experiments.run_multisource_calibration027b import feature_identity
from scripts.experiments.run_referral_budget030 import EXPECTED_EVALUATION, checked_csv
from scripts.experiments.run_rhythm_findings032 import (
    challenge_table,
    check_binary_rows,
    ptb_inputs,
    sph_inputs,
    training_sets,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/experiment037_pipeline_v2_v1"
PRIOR022B = ROOT / "outputs/experiment022b_multisource_readout_v1"
PRIOR030 = ROOT / "outputs/experiment030_referral_budget_v1"
PRIOR032 = ROOT / "outputs/experiment032_rhythm_findings_v1"
PRIOR033 = ROOT / "outputs/experiment033_finding_heads_screen_v1"
PRIOR035 = ROOT / "outputs/experiment035_hard_subset_v1"
ENCODER = "xecg"
PIPELINES = ("v1", "v2")
RULES = ("binary", "combined_50")
FINDING_HEADS = {"pvc": "ventricular_ectopy", "wpw": "preexcitation"}
BUDGETS = (10, 20, 50, 100)
SIZES = (50, 100, 200, 500, 1000, 2000)
PRIMARY_BUDGET, PRIMARY_SIZE, PRIMARY_RULE = 50, 200, "combined_50"
DRAWS = 200
DRAW_SEED = 30030
BOOTSTRAP_SEED = 41041
RESAMPLES = 2000
PREVALENCE = 0.05
WORSE_MARGIN = 0.005
MAX_BINARY_COST = 0.010
MAX_RATE_INCREASE = 0.0025
TOLERANCE = 1e-12
SUPERCLASSES = ("MI", "STTC", "CD", "HYP")
OUTCOMES = ("normal", "binary_positive", "composite", "pvc", "frequent_pvc", "wpw", "af_flutter",
            "high_grade_av_block", "long_qt", "finding_only", "other", *SUPERCLASSES)
LOCAL_OUTCOMES = ("binary_positive", "composite")
EXPECTED_TRAINING = {"binary": [39577, 27360], "dropped": 1724, "ptbxl_binary": 17083}
EXPECTED_DEVELOPMENT = {"hard": [266, 41], "ordinary": [1306, 843], "full": [1572, 884]}
SOURCES = (
    "ecg_experiment/pipeline_v2.py", "ecg_experiment/intervals.py", "ecg_experiment/finding_screen.py",
    "ecg_experiment/referral_budget.py", "ecg_experiment/hard_subset.py",
    "ecg_experiment/multisource_readout.py", "ecg_experiment/full_development.py",
    "ecg_experiment/hybrid_score.py", "ecg_experiment/rhythm_findings.py",
    "ecg_experiment/local_adaptation.py", "ecg_experiment/sph.py",
    "scripts/experiments/run_pipeline_v2_037.py", "scripts/experiments/run_finding_screen033.py",
    "scripts/experiments/run_referral_budget030.py", "scripts/experiments/run_rhythm_findings032.py",
    "scripts/experiments/run_hard_subset035.py", "scripts/experiments/run_multisource_readout022b.py",
    "scripts/experiments/run_multisource_manifold026b.py", "scripts/experiments/run_local_adaptation029.py",
    "pyproject.toml", "uv.lock", "docs/experiment-037-pipeline-v2.md",
)


def receipt_checked(directory: Path, names: tuple[str, ...]) -> dict[str, str]:
    """
    Hashes of an earlier run's output files, each checked against that run's receipt.

    Parameters
    ----------
    directory : Path
        Output directory with a ``result.json``.
    names : tuple[str, ...]
        Output files to check.

    Returns
    -------
    dict[str, str]
        SHA-256 per file, and of ``result.json``.

    Raises
    ------
    ValueError
        If a file differs from its receipt.
    """
    receipt = json.loads((directory / "result.json").read_text())
    hashes = {name: sha256_file(directory / name) for name in names}
    for name, digest in hashes.items():
        if digest != receipt["outputs_sha256"][name]:
            raise ValueError(f"{directory.name}/{name} differs from its receipt")
    return {**hashes, "result.json": sha256_file(directory / "result.json")}


def training_data() -> dict[str, Any]:
    """
    032's PTB-XL, SPH and Challenge rows and xECG features, with 032's saved training quality reasons.

    Returns
    -------
    dict[str, Any]
        Stacked training features, readout training sets, 035's ``dropped_upweighted`` design, SPH rows and
        features, development rows and features, calibration rows and features, and identity hashes.

    Raises
    ------
    ValueError
        If a row order or count differs from 032, 022b or the protocol.
    """
    feature_metadata, feature_hashes = feature_identity()
    ptb, ptb_x, ptb_identity = ptb_inputs()
    sph, sph_x, sph_hashes = sph_inputs()
    rows, challenge_x = challenge_table()
    in_train = (rows["split"] == "train").to_numpy()
    in_calibration = (rows["split"] == "calibration").to_numpy()
    train = rows[in_train].copy()
    saved = checked_csv(PRIOR032, "training_rows.csv", dtype={"record": str}, keep_default_na=False)
    if not all(np.array_equal(saved[column].to_numpy(dtype=str), train[column].to_numpy(dtype=str))
               for column in ("source", "record")):
        raise ValueError("Challenge training rows differ from 032's")
    train["reasons"] = saved["reasons"].to_numpy(dtype=str)
    check_binary_rows(train)
    kept = (train["reasons"] == "").to_numpy()
    readouts = training_sets(ptb["train"], train[kept])
    x = np.concatenate([ptb_x[ENCODER]["train"], challenge_x[ENCODER][in_train][kept]])
    families = np.concatenate([np.full(len(ptb["train"]), "ptbxl"),
                               train.loc[kept, "family"].to_numpy(dtype=str)])
    dropped = np.concatenate([ptb["train"]["target"].isna().to_numpy(),
                              np.zeros(int(kept.sum()), dtype=bool)])
    binary_rows = readouts["binary"]["rows"]
    design = arm_design(ARM_SPECS["dropped_upweighted"], families[binary_rows],
                        {"primary": readouts["binary"]["y"].astype(np.float64)}, dropped[binary_rows])
    found = {"binary": [len(binary_rows), int(readouts["binary"]["y"].sum())],
             "dropped": int(dropped[binary_rows].sum()),
             "ptbxl_binary": int((families[binary_rows] == "ptbxl").sum())}
    if found != EXPECTED_TRAINING or not design["selected"].all():
        raise ValueError(f"Training counts differ from the protocol: {found}")
    calibration = rows[in_calibration].reset_index(drop=True)
    identity = {**ptb_identity, "sph_manifest": sph_hashes, "challenge_features": feature_hashes,
                "challenge_features_identity": feature_metadata["identity"]}
    return {"x": x, "readouts": readouts, "design": design, "sph": sph, "sph_x": sph_x[ENCODER],
            "development": ptb["development"], "development_x": ptb_x[ENCODER]["development"],
            "calibration": calibration, "calibration_x": challenge_x[ENCODER][in_calibration],
            "identity": identity, "training_counts": found}


def fit_heads(data: dict[str, Any]
              ) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]], dict[str, int]]:
    """
    Fit the v1 and v2 binary readouts and the PVC and WPW heads, and score SPH, development and calibration.

    Parameters
    ----------
    data : dict[str, Any]
        Output of ``training_data``.

    Returns
    -------
    tuple[dict[str, Any], dict[str, dict[str, np.ndarray]], dict[str, int]]
        Heads, probabilities per head and set, and iterations per head.
    """
    readouts, design, x = data["readouts"], data["design"], data["x"]
    binary = readouts["binary"]
    heads, iterations = {}, {}
    with threadpool_limits(limits=1):
        heads["v1"] = fit_readout(x[binary["rows"]], design["y"])
        heads["v2"] = fit_readout(x[binary["rows"]], design["y"], design["weights"])
        for name, group in FINDING_HEADS.items():
            heads[name] = fit_logistic(x[readouts[group]["rows"]], readouts[group]["y"])
        inputs = {"sph": data["sph_x"], "development": data["development_x"],
                  "calibration": data["calibration_x"]}
        scores = {name: {key: predict(head, values) for key, values in inputs.items()}
                  for name, head in heads.items()}
    for name, head in heads.items():
        model: LogisticRegression = head[1]
        iterations[name] = int(model.n_iter_[0])
        print(json.dumps({"stage": f"fit:{name}", "iterations": iterations[name]}), flush=True)
    return heads, scores, iterations


def reproduce_heads(data: dict[str, Any], scores: dict[str, dict[str, np.ndarray]],
                    parameters: dict[str, np.ndarray]) -> dict[str, float]:
    """
    Require the refitted heads to reproduce 022b, 032 and 035, and the saved parameters the heads.

    Parameters
    ----------
    data : dict[str, Any]
        Output of ``training_data``.
    scores : dict[str, dict[str, np.ndarray]]
        Output of ``fit_heads``.
    parameters : dict[str, np.ndarray]
        Saved head parameters.

    Returns
    -------
    dict[str, float]
        Largest absolute difference per check.

    Raises
    ------
    ValueError
        If a row order differs or a difference exceeds ``TOLERANCE``.
    """
    sph_ids = data["sph"]["ecg_id"].to_numpy(dtype=str)
    differences = {}
    with np.load(PRIOR022B / "predictions.npz") as saved:
        positions = pd.Index(sph_ids).get_indexer(saved["sph_ecg_ids"])
        if (positions < 0).any():
            raise ValueError("022b SPH rows are missing")
        differences["v1_sph_vs_022b"] = float(np.abs(scores["v1"]["sph"][positions]
                                                     - saved[f"sph_{ENCODER}_pooled"]).max())
    with np.load(PRIOR032 / "predictions.npz") as saved:
        calibration = data["calibration"]
        records = (calibration["source"] + ":" + calibration["record"]).to_numpy(dtype=str)
        if not (np.array_equal(saved["sph_ecg_ids"], sph_ids)
                and np.array_equal(saved["calibration_records"], records)):
            raise ValueError("032 row orders differ")
        differences["v1_sph_vs_032"] = float(np.abs(
            scores["v1"]["sph"] - saved[f"sph_{ENCODER}_binary"]).max())
        for name, group in FINDING_HEADS.items():
            for key in ("sph", "calibration"):
                differences[f"{name}_{key}_vs_032"] = float(np.abs(
                    scores[name][key] - saved[f"{key}_{ENCODER}_{group}"]).max())
    with np.load(PRIOR035 / "predictions.npz") as saved:
        positions = pd.Index(sph_ids).get_indexer(saved["sph_ecg_ids"])
        development = pd.Index(data["development"].index.to_numpy(dtype=str)).get_indexer(
            saved["development_record_ids"])
        if (positions < 0).any() or (development < 0).any():
            raise ValueError("035 rows are missing")
        differences["v2_sph_vs_035"] = float(np.abs(
            scores["v2"]["sph"][positions] - saved[f"sph_{ENCODER}_dropped_upweighted"]).max())
        differences["v2_development_vs_035"] = float(np.abs(
            scores["v2"]["development"][development]
            - saved[f"development_{ENCODER}_dropped_upweighted"]).max())
        differences["v1_development_vs_035"] = float(np.abs(
            scores["v1"]["development"][development] - saved[f"development_{ENCODER}_pooled"]).max())
    for name in ("v1", "v2", *FINDING_HEADS):
        differences[f"{name}_saved_parameters"] = float(np.abs(
            score_parameters(parameters, name, data["sph_x"]) - scores[name]["sph"]).max())
    if max(differences.values()) > TOLERANCE:
        raise ValueError(f"A head does not reproduce: {differences}")
    return differences


def finding_standardization(data: dict[str, Any], scores: dict[str, dict[str, np.ndarray]]
                            ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """
    Standardization constants of the finding heads on 032's Challenge calibration clear normals.

    Parameters
    ----------
    data : dict[str, Any]
        Output of ``training_data``.
    scores : dict[str, dict[str, np.ndarray]]
        Output of ``fit_heads``.

    Returns
    -------
    tuple[dict[str, np.ndarray], dict[str, float]]
        Saved constants (``{head}_logit_mean`` and ``{head}_logit_sd``) and the number of source normals.
    """
    calibration = data["calibration"]
    standard = np.where(calibration["evaluable"].astype(bool), calibration["primary"], np.nan)
    source_normal = clear_normal(standard, calibration[list(GROUPS)])
    constants = {}
    for name in FINDING_HEADS:
        mean, spread = normal_standardizer(clipped_logits(scores[name]["calibration"][source_normal])[0])
        constants[f"{name}_logit_mean"] = np.array([mean])
        constants[f"{name}_logit_sd"] = np.array([spread])
    return constants, {"source_normals": int(source_normal.sum())}


def pipeline_scores(rows: pd.DataFrame, scores033: dict[str, np.ndarray],
                    scores: dict[str, dict[str, np.ndarray]], data: dict[str, Any]
                    ) -> tuple[dict[str, dict[str, np.ndarray]], float]:
    """
    Binary and combined finding scores of both pipelines on 033's SPH rows.

    Parameters
    ----------
    rows : pd.DataFrame
        033's ``load_rows`` rows.
    scores033 : dict[str, np.ndarray]
        033's ``load_scores`` scores (the v1 pipeline).
    scores : dict[str, dict[str, np.ndarray]]
        Output of ``fit_heads``.
    data : dict[str, Any]
        Output of ``training_data``.

    Returns
    -------
    tuple[dict[str, dict[str, np.ndarray]], float]
        Per pipeline the ``binary``, ``pvc``, ``wpw`` and ``combined`` scores; and the largest difference
        of the refitted z-scores from 033's.

    Raises
    ------
    ValueError
        If the rows differ or the refitted z-scores differ from 033's.
    """
    if not np.array_equal(rows["ecg_id"].to_numpy(dtype=str), data["sph"]["ecg_id"].to_numpy(dtype=str)):
        raise ValueError("033's SPH rows differ from 032's")
    calibration = data["calibration"]
    standard = np.where(calibration["evaluable"].astype(bool), calibration["primary"], np.nan)
    source_normal = clear_normal(standard, calibration[list(GROUPS)])
    z = {name: finding_z(scores[name]["sph"], scores[name]["calibration"][source_normal])
         for name in FINDING_HEADS}
    difference = max(float(np.abs(z[name] - scores033[name]).max()) for name in FINDING_HEADS)
    difference = max(difference, float(np.abs(scores["v1"]["sph"] - scores033["binary"]).max()))
    if difference > TOLERANCE:
        raise ValueError(f"Refitted scores differ from 033's by {difference}")
    findings = {"pvc": scores033["pvc"], "wpw": scores033["wpw"], "combined": scores033["combined"]}
    return {"v1": {"binary": scores033["binary"], **findings},
            "v2": {"binary": scores["v2"]["sph"], **findings}}, difference


def outcome_masks(rows: pd.DataFrame, evaluation: np.ndarray) -> dict[str, np.ndarray]:
    """
    Outcome masks of the evaluation half, with the superclasses on 030's rows.

    Parameters
    ----------
    rows : pd.DataFrame
        033's ``load_rows`` rows.
    evaluation : np.ndarray
        Evaluation mask.

    Returns
    -------
    dict[str, np.ndarray]
        One boolean mask per name in ``OUTCOMES``, over the evaluation ECGs.

    Raises
    ------
    ValueError
        If a superclass count differs from 030's.
    """
    part = rows[evaluation]
    masks = {name: part[name].to_numpy(dtype=bool) for name in OUTCOMES if name not in SUPERCLASSES}
    for name in SUPERCLASSES:
        masks[name] = (part["in030"] & (part[name] == 1)).to_numpy()
    found = {name: int(masks[name].sum()) for name in SUPERCLASSES}
    if found != {name: EXPECTED_EVALUATION[name] for name in SUPERCLASSES}:
        raise ValueError(f"Superclass counts differ from 030's: {found}")
    return masks


def run_draws(pipelines: dict[str, dict[str, np.ndarray]], rows: pd.DataFrame, local_normal: np.ndarray
              ) -> tuple[pd.DataFrame, dict[tuple[str, str, int, int], np.ndarray]]:
    """
    Thresholds and outcomes of every pipeline, rule, size, budget and draw.

    The two main rules run at every size and budget; the other 033 rules run at the primary size and budget
    only, for the local-pool selection step.

    Parameters
    ----------
    pipelines : dict[str, dict[str, np.ndarray]]
        Output of ``pipeline_scores``.
    rows : pd.DataFrame
        033's ``load_rows`` rows.
    local_normal : np.ndarray
        Row positions of the local normals, in 030's order.

    Returns
    -------
    tuple[pd.DataFrame, dict[tuple[str, str, int, int], np.ndarray]]
        One row per pipeline, rule, size, budget and draw; and, for the main rules, the referral share over
        draws of each evaluation ECG.
    """
    evaluation = rows["evaluation"].to_numpy()
    masks = outcome_masks(rows, evaluation)
    local_masks = {name: rows.loc[~evaluation, name].to_numpy() for name in LOCAL_OUTCOMES}
    plans = {m: [local_normal[np.sort(np.random.default_rng([DRAW_SEED, m, draw])
                                      .choice(len(local_normal), m, replace=False))]
                 for draw in range(DRAWS)] for m in SIZES}
    jobs = [(pipeline, rule, m, budget) for pipeline in PIPELINES for rule in RULES for m in SIZES
            for budget in BUDGETS if rule == "binary" or budget * m // 1000 >= 1]
    jobs += [(pipeline, rule, PRIMARY_SIZE, PRIMARY_BUDGET) for pipeline in PIPELINES for rule in RULES033
             if rule not in RULES]
    records, shares = [], {}
    for pipeline, rule, m, budget in jobs:
        columns, rule_shares = RULES033[rule]
        scores = pipelines[pipeline]
        matrix = np.column_stack([scores["binary"], *(scores[column] for column in columns)])
        thresholds = np.array([split_thresholds(matrix[positions], budget, rule_shares)
                               for positions in plans[m]])
        referred = referred_matrix(matrix[evaluation], thresholds)
        local = referred_matrix(matrix[~evaluation], thresholds)
        if rule in RULES:
            shares[(pipeline, rule, m, budget)] = referred.mean(axis=1)
        found = {name: referred[mask].mean(axis=0) for name, mask in masks.items()}
        found_local = {name: local[mask].mean(axis=0) for name, mask in local_masks.items()}
        for draw in range(DRAWS):
            records.append({
                "pipeline": pipeline, "rule": rule, "m": m, "budget": budget / 1000, "draw": draw,
                **{f"threshold_{index}": float(value) for index, value in enumerate(thresholds[draw])},
                **{name: float(values[draw]) for name, values in found.items()},
                **{f"local_{name}": float(values[draw]) for name, values in found_local.items()}})
        print(json.dumps({"stage": f"draws:{pipeline}:{rule}:{m}:{budget}"}), flush=True)
    return pd.DataFrame(records), shares


def largest_difference(ours: pd.DataFrame, prior: pd.DataFrame, keys: list[str],
                       pairs: list[tuple[str, str]]) -> float:
    """
    Largest absolute difference of paired columns after a one-to-one merge, with matching NaN patterns.

    Parameters
    ----------
    ours, prior : pd.DataFrame
        Tables to compare.
    keys : list[str]
        Merge keys.
    pairs : list[tuple[str, str]]
        (our column, prior column) pairs.

    Returns
    -------
    float
        The largest difference.

    Raises
    ------
    ValueError
        If a row is unmatched or a NaN pattern differs.
    """
    merged = ours.merge(prior, on=keys, validate="one_to_one", suffixes=("", "_prior"))
    if len(merged) != len(ours) or len(merged) != len(prior):
        raise ValueError("Draws do not match one to one")
    largest = 0.0
    for mine, theirs in pairs:
        first, second = merged[mine].to_numpy(dtype=float), merged[theirs].to_numpy(dtype=float)
        if not np.array_equal(np.isnan(first), np.isnan(second)):
            raise ValueError(f"NaN patterns differ for {mine}")
        defined = ~np.isnan(first)
        if defined.any():
            largest = max(largest, float(np.abs(first[defined] - second[defined]).max()))
    return largest


def check_prior_draws(draws: pd.DataFrame) -> dict[str, float]:
    """
    Require the v1 draws to equal 030's pooled xECG draws and 033's draws.

    Parameters
    ----------
    draws : pd.DataFrame
        Output of ``run_draws``.

    Returns
    -------
    dict[str, float]
        Largest difference from 030 and from 033.

    Raises
    ------
    ValueError
        If a difference exceeds ``TOLERANCE``.
    """
    prior030 = checked_csv(PRIOR030, "draws.csv", float_precision="round_trip")
    prior030 = prior030[(prior030["score"] == "pooled") & (prior030["encoder"] == ENCODER)
                        & prior030["m"].isin(SIZES)]
    ours = draws[(draws["pipeline"] == "v1") & (draws["rule"] == "binary")]
    pairs030 = [("threshold_0", "threshold"), ("normal", "rate"), ("binary_positive", "sensitivity"),
                *((name, f"sensitivity_{name}") for name in SUPERCLASSES)]
    found = {"030": largest_difference(ours, prior030, ["m", "budget", "draw"], pairs030)}
    keys = ["rule", "m", "budget", "draw"]
    prior033 = checked_csv(PRIOR033, "draws.csv", float_precision="round_trip")
    ours = draws[draws["pipeline"] == "v1"].drop(columns="pipeline")
    prior033 = prior033.merge(ours[keys], on=keys)
    ours = ours.merge(prior033[keys], on=keys)
    shared = 2 * len(BUDGETS033) * len(SIZES033) * DRAWS + (len(RULES033) - len(RULES)) * DRAWS
    if len(ours) != shared or len(prior033) != shared:
        raise ValueError("033's draws are not all present")
    columns = [column for column in prior033.columns if column not in keys]
    pairs033 = [(column, f"{column}_prior") for column in columns]
    found["033"] = largest_difference(ours, prior033, keys, pairs033)
    if max(found.values()) > TOLERANCE:
        raise ValueError(f"The v1 draws differ from 030 or 033: {found}")
    return found


def summarize(draws: pd.DataFrame, boot: dict[str, np.ndarray], keys: list[tuple[str, str, int, int]]
              ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Per-screen summaries and the paired v2 minus v1 and rule minus ``binary`` contrasts.

    Parameters
    ----------
    draws : pd.DataFrame
        Output of ``run_draws``.
    boot : dict[str, np.ndarray]
        ``(RESAMPLES, keys)`` rates per outcome.
    keys : list[tuple[str, str, int, int]]
        Keys in the column order of ``boot``.

    Returns
    -------
    tuple[list[dict[str, Any]], list[dict[str, Any]]]
        Summary rows and contrast rows.
    """
    groups = draws.groupby(["pipeline", "rule", "m", "budget"], sort=False)
    column = {key: index for index, key in enumerate(keys)}
    summary, contrasts = [], []
    for key in keys:
        pipeline, rule, m, budget = key
        group = groups.get_group((pipeline, rule, m, budget / 1000))
        row: dict[str, Any] = {"pipeline": pipeline, "rule": rule, "m": m, "budget": budget / 1000,
                               "threshold_0_mean": float(group["threshold_0"].mean()),
                               "share_rate_within_1pp": float(((group["normal"] - budget / 1000).abs()
                                                               <= 0.01 + 1e-12).mean())}
        for name in OUTCOMES:
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
        references = []
        if pipeline == "v2":
            references.append(("v2_minus_v1", ("v1", rule, m, budget)))
        if rule != "binary":
            references.append((f"{pipeline}_{rule}_minus_binary", (pipeline, "binary", m, budget)))
        for label, reference in references:
            base = groups.get_group((reference[0], reference[1], m, budget / 1000))
            for name in OUTCOMES:
                difference = boot[name][:, column[key]] - boot[name][:, column[reference]]
                contrasts.append({"contrast": label, "rule": rule, "m": m, "budget": budget / 1000,
                                  "outcome": name,
                                  "difference": float(group[name].mean() - base[name].mean()),
                                  "ci": percentile_interval(difference)})
    return summary, contrasts


def local_selection(draws: pd.DataFrame) -> dict[str, Any]:
    """
    033's local-pool selection step at the primary size and budget, for each pipeline.

    Parameters
    ----------
    draws : pd.DataFrame
        Output of ``run_draws``.

    Returns
    -------
    dict[str, Any]
        Per pipeline, the table of local composite sensitivity and binary cost, and the selected rule.
    """
    at = draws[(draws["m"] == PRIMARY_SIZE) & (draws["budget"] == PRIMARY_BUDGET / 1000)]
    means = at.groupby(["pipeline", "rule"])[["local_binary_positive", "local_composite"]].mean()
    selection = {}
    for pipeline in PIPELINES:
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
        selection[pipeline] = {"table": table, "selected": selected}
    return selection


def auroc_sets(data: dict[str, Any], scores: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    """
    AUROC of v1 and v2 on PTB-XL development (hard, ordinary, full) and SPH, with paired v2 - v1 intervals.

    Parameters
    ----------
    data : dict[str, Any]
        Output of ``training_data``.
    scores : dict[str, dict[str, np.ndarray]]
        Output of ``fit_heads``.

    Returns
    -------
    dict[str, Any]
        Per set: counts, AUROC per pipeline and the paired difference.

    Raises
    ------
    ValueError
        If a set's counts differ from the protocol.
    """
    with np.load(PRIOR035 / "predictions.npz") as saved:
        record_ids = saved["development_record_ids"]
        labels, original = saved["development_labels"], saved["development_original"]
    development = data["development"]
    positions = pd.Index(development.index.to_numpy(dtype=str)).get_indexer(record_ids)
    patients = development["patient_id"].to_numpy(dtype=str)[positions]
    selections = {"hard": ~original, "ordinary": original, "full": np.ones(len(original), dtype=bool)}
    sets = {name: {"y": labels[mask], "patients": patients[mask],
                   "v1": scores["v1"]["development"][positions][mask],
                   "v2": scores["v2"]["development"][positions][mask]}
            for name, mask in selections.items()}
    found = {name: [len(spec["y"]), int(spec["y"].sum())] for name, spec in sets.items()}
    if found != EXPECTED_DEVELOPMENT:
        raise ValueError(f"Development counts differ from 035: {found}")
    sph = data["sph"]
    labeled = sph["primary"].notna().to_numpy()
    sets["sph"] = {"y": sph.loc[labeled, "primary"].to_numpy(dtype=np.int64),
                   "patients": sph.loc[labeled, "patient_id"].to_numpy(dtype=str),
                   "v1": scores["v1"]["sph"][labeled], "v2": scores["v2"]["sph"][labeled]}
    return {name: {"records": len(spec["y"]), "positives": int(spec["y"].sum()),
                   "patients": len(np.unique(spec["patients"])),
                   "auroc": {pipeline: float(roc_auc_score(spec["y"], spec[pipeline]))
                             for pipeline in PIPELINES},
                   "v2_minus_v1": paired_auroc_difference(spec["patients"], spec["y"], spec["v2"], spec["v1"],
                                                          RESAMPLES, BOOTSTRAP_SEED)}
            for name, spec in sets.items()}


def decide(summary: list[dict[str, Any]], contrasts: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Apply the pre-registered decision to the primary contrast, and 033's adoption conditions to v2.

    Parameters
    ----------
    summary : list[dict[str, Any]]
        Output of ``summarize``.
    contrasts : list[dict[str, Any]]
        Output of ``summarize``.

    Returns
    -------
    dict[str, Any]
        The primary contrast, the decision, the non-inferiority reading and 033's conditions for v2.
    """
    def contrast(label: str, outcome: str) -> dict[str, Any]:
        (row,) = [row for row in contrasts if row["contrast"] == label and row["rule"] == PRIMARY_RULE
                  and row["m"] == PRIMARY_SIZE and row["budget"] == PRIMARY_BUDGET / 1000
                  and row["outcome"] == outcome]
        return row

    primary = contrast("v2_minus_v1", "composite")
    worse = v2_is_worse(primary["ci"], WORSE_MARGIN)
    gain = contrast(f"v2_{PRIMARY_RULE}_minus_binary", "composite")
    cost = -contrast(f"v2_{PRIMARY_RULE}_minus_binary", "binary_positive")["difference"]
    rate = contrast(f"v2_{PRIMARY_RULE}_minus_binary", "normal")["difference"]
    holds = gain["ci"][0] > 0 and cost <= MAX_BINARY_COST and rate <= MAX_RATE_INCREASE
    conditions = {"composite_gain": gain["difference"], "composite_gain_ci": gain["ci"], "binary_cost": cost,
                  "rate_difference": rate, "all_hold": bool(holds)}
    return {"primary": primary, "v2_worse": worse, "decision": "keep_v1" if worse else "adopt_v2",
            "lower_bound_above_minus_margin": bool(primary["ci"][0] > -WORSE_MARGIN),
            "v2_033_conditions": conditions}


def main() -> None:
    """Run the experiment once and write the aggregate result."""
    if (OUTPUT / "result.json").exists():
        raise FileExistsError("Experiment 037 v1 has already run")
    started = time.monotonic()
    receipts = {"experiment022b": receipt_checked(PRIOR022B, ("predictions.npz", "training_rows.csv")),
                "experiment030": receipt_checked(PRIOR030, ("draws.csv",)),
                "experiment032": receipt_checked(PRIOR032, ("predictions.npz", "training_rows.csv",
                                                            "calibration_rows.csv")),
                "experiment033": receipt_checked(PRIOR033, ("draws.csv",)),
                "experiment035": receipt_checked(PRIOR035, ("predictions.npz", "training_rows.csv"))}
    data = training_data()
    print(json.dumps({"stage": "loaded", **data["training_counts"],
                      "seconds": round(time.monotonic() - started)}), flush=True)
    heads, scores, iterations = fit_heads(data)
    constants, standardization = finding_standardization(data, scores)
    parameters = {name: values for head, head_values in heads.items()
                  for name, values in head_parameters(head_values, head).items()}
    checks: dict[str, Any] = {"heads": reproduce_heads(data, scores, parameters), **standardization}
    print(json.dumps({"stage": "heads", **checks, "seconds": round(time.monotonic() - started)}), flush=True)

    rows, site, local_normal = load_rows()
    counts = counts_of(rows)
    if json.loads(json.dumps(counts)) != EXPECTED033:
        raise ValueError("Counts differ from 033's")
    with threadpool_limits(limits=3):
        scores033, checks["033_load_scores"] = load_scores(rows, site)
        pipelines, checks["z_difference_from_033"] = pipeline_scores(rows, scores033, scores, data)
        draws, shares = run_draws(pipelines, rows, local_normal)
        checks["draws_difference"] = check_prior_draws(draws)
        print(json.dumps({"stage": "draws", **checks["draws_difference"],
                          "seconds": round(time.monotonic() - started)}), flush=True)
        evaluation = rows["evaluation"].to_numpy()
        masks = outcome_masks(rows, evaluation)
        keys = list(shares)
        matrix = np.column_stack([shares[key] for key in keys])
        counts_matrix = resample_counts(rows.loc[evaluation, "patient_id"].to_numpy(dtype=str), RESAMPLES,
                                        BOOTSTRAP_SEED)
        boot = {name: resampled_rates(counts_matrix, matrix, mask) for name, mask in masks.items()}
        empty = {name: int(np.isnan(values[:, 0]).sum()) for name, values in boot.items()}
        del counts_matrix
        print(json.dumps({"stage": "bootstrap", "empty_resamples": empty,
                          "seconds": round(time.monotonic() - started)}), flush=True)
    for index, key in enumerate(keys):
        group = draws[(draws["pipeline"] == key[0]) & (draws["rule"] == key[1]) & (draws["m"] == key[2])
                      & (draws["budget"] == key[3] / 1000)]
        if abs(matrix[masks["composite"], index].mean() - group["composite"].mean()) > 1e-9:
            raise ValueError(f"Referral shares do not reproduce the draw mean for {key}")
    summary, contrasts = summarize(draws, boot, keys)
    reading = decide(summary, contrasts)
    selection = local_selection(draws)
    if selection["v1"]["selected"] != PRIMARY_RULE:
        raise ValueError("The v1 local selection does not reproduce 033's")
    aurocs = auroc_sets(data, scores)
    print(json.dumps({"stage": "auroc", "seconds": round(time.monotonic() - started)}), flush=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    draws.to_csv(OUTPUT / "draws.csv", index=False)
    write_npz_atomic(OUTPUT / "pipeline_v2_heads.npz", **parameters, **constants)
    development_ids = data["development"].index.to_numpy(dtype=str)
    write_npz_atomic(
        OUTPUT / "predictions.npz",
        sph_ecg_ids=rows["ecg_id"].to_numpy(dtype=str),
        sph_patient_ids=rows["patient_id"].to_numpy(dtype=str),
        sph_evaluation=evaluation, development_record_ids=development_ids,
        **{f"sph_{pipeline}_binary": pipelines[pipeline]["binary"] for pipeline in PIPELINES},
        **{f"sph_z_{name}": pipelines["v2"][name] for name in ("pvc", "wpw", "combined")},
        **{f"development_{pipeline}_binary": scores[pipeline]["development"] for pipeline in PIPELINES})
    written = ("draws.csv", "pipeline_v2_heads.npz", "predictions.npz")
    result = {
        "status": "complete",
        "identity": {"receipts": receipts, **data["identity"],
                     "sources": {name: sha256_file(ROOT / name) for name in SOURCES}},
        "encoder": ENCODER, "pipelines": {"v1": "022b pooled", "v2": "035 dropped_upweighted"},
        "rules": {name: {"columns": list(RULES033[name][0]), "shares_per_mille": list(RULES033[name][1])}
                  for name in RULES033},
        "draw_seed": DRAW_SEED, "bootstrap_seed": BOOTSTRAP_SEED, "resamples": RESAMPLES, "draws": DRAWS,
        "sizes": list(SIZES), "budgets": [budget / 1000 for budget in BUDGETS], "prevalence": PREVALENCE,
        "training_counts": data["training_counts"], "iterations": iterations, "counts": counts,
        "weights": {"dropped": float(data["design"]["weights"][data["design"]["weights"] > 1][0]),
                    "other": float(data["design"]["weights"][data["design"]["weights"] < 1][0])},
        "standardization": {name: float(values[0]) for name, values in constants.items()},
        "checks": checks, "empty_resamples": empty, "summary": summary, "contrasts": contrasts,
        "local_selection": selection, "auroc": aurocs, "reading": reading,
        "outputs_sha256": {name: sha256_file(OUTPUT / name) for name in written},
        "challenge_test_read": False, "ptbxl_calibration_read": False, "ptbxl_test_read": False,
        "runtime_seconds": time.monotonic() - started,
    }
    write_json_atomic(OUTPUT / "result.json", result)
    print(json.dumps({"stage": "complete", "runtime_seconds": result["runtime_seconds"],
                      "decision": reading["decision"], "primary": reading["primary"]}), flush=True)


if __name__ == "__main__":
    main()
