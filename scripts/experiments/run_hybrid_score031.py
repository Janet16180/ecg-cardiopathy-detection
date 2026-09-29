"""Experiment 031: a hybrid of the supervised readout and the distance from normal at a referral budget."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.full_development import cohorts, predict, ptb_table
from ecg_experiment.hybrid_score import (
    fit_stack,
    normal_standardizer,
    referred,
    rule_thresholds,
    standardize,
)
from ecg_experiment.multisource_readout import fit_readout
from ecg_experiment.referral_budget import bootstrap_counts, group_sums, referrals_per_1000
from ecg_experiment.screening_threshold import head_logits
from scripts.experiments.run_calibrated_threshold027 import prior022_identity
from scripts.experiments.run_multisource_calibration027b import feature_identity
from scripts.experiments.run_multisource_manifold026b import challenge_features, split_table
from scripts.experiments.run_normal_manifold026 import record_classes
from scripts.experiments.run_referral_budget030 import checked_csv, load_heads, load_site
from scripts.experiments.run_sph_external022 import open_caches, ptb_encoder_features

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/experiment031_hybrid_score_v1"
PRIOR022 = ROOT / "outputs/experiment022_sph_external_v3"
PRIOR022B = ROOT / "outputs/experiment022b_multisource_readout_v1"
PRIOR026B = ROOT / "outputs/experiment026b_multisource_manifold_v1"
PRIOR029 = ROOT / "outputs/experiment029_local_adaptation_v1"
PRIOR030 = ROOT / "outputs/experiment030_referral_budget_v1"
ENCODERS = ("xecg", "jepa")
SUPERCLASSES = ("MI", "STTC", "CD", "HYP")
ONLY = tuple(f"{name}_only" for name in SUPERCLASSES)
GROUPS = ("positive", "normal", *SUPERCLASSES, *ONLY)
RULES = ("readout", "distance", "zmean", "stack", "either")
HYBRIDS = ("zmean", "stack", "either")
BUDGETS = (10, 20, 50, 100)
SIZES = (200, 1000)
DRAWS = 200
SEED = 30030
BOOTSTRAP_SEED = 35035
RESAMPLES = 2000
PREVALENCES = (0.02, 0.05)
BAND = 0.01
PRIMARY = {"encoder": "xecg", "m": 200, "budget": 50}
ADOPT_MINIMUM = 0.010
RATE_MAXIMUM = 0.005
REPRODUCTION_TOLERANCE = 1e-9
EXPECTED_ONLY = {"MI_only": 37, "STTC_only": 2194, "CD_only": 995, "HYP_only": 27}
EXPECTED_CALIBRATION = (7612, 5905, 1707)
EXPECTED_TRAINING = (39577, 27360)
EXPECTED_WITHOUT = {"MI": (34319, 22102), "STTC": (22719, 10502), "CD": (29763, 17546), "HYP": (35540, 23323)}
EXPECTED_CALIBRATION_WITHOUT = {"MI": 7332, "STTC": 3358, "CD": 5581, "HYP": 7014}
SOURCES = (
    "ecg_experiment/hybrid_score.py", "ecg_experiment/referral_budget.py",
    "ecg_experiment/multisource_readout.py", "ecg_experiment/screening_threshold.py",
    "scripts/experiments/run_hybrid_score031.py", "scripts/experiments/run_referral_budget030.py",
    "scripts/experiments/run_multisource_readout022b.py",
    "scripts/experiments/run_multisource_manifold026b.py",
    "scripts/experiments/run_normal_manifold026.py", "scripts/experiments/run_sph_external022.py",
    "pyproject.toml", "uv.lock", "docs/experiment-031-hybrid-screening-score.md",
)


def group_masks(masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Add the single-superclass (S-only) groups to 030's evaluation masks.

    Parameters
    ----------
    masks : dict[str, np.ndarray]
        030's masks: ``positive``, ``normal`` and the four superclasses.

    Returns
    -------
    dict[str, np.ndarray]
        Masks of every name in ``GROUPS``.

    Raises
    ------
    ValueError
        If an S-only count differs from the protocol.
    """
    carried = np.sum([masks[name] for name in SUPERCLASSES], axis=0)
    result = {**masks, **{f"{name}_only": masks[name] & (carried == 1) for name in SUPERCLASSES}}
    found = {name: int(result[name].sum()) for name in ONLY}
    if found != EXPECTED_ONLY:
        raise ValueError(f"S-only counts differ from the protocol: {found}")
    return result


def calibration_flags(arrays: dict[str, np.ndarray], challenge: pd.DataFrame) -> pd.DataFrame:
    """
    Superclass flags of the Challenge calibration records, in 026b's order.

    Parameters
    ----------
    arrays : dict[str, np.ndarray]
        022b's arrays.
    challenge : pd.DataFrame
        026b's Challenge calibration scores, whose order 030's ``load_heads`` checked.

    Returns
    -------
    pd.DataFrame
        Boolean ``MI``, ``STTC``, ``CD`` and ``HYP`` per record.

    Raises
    ------
    ValueError
        If the calibration counts differ from the protocol.
    """
    y = challenge["primary"].to_numpy(dtype=np.int64)
    found = (len(y), int(y.sum()), int((y == 0).sum()))
    if found != EXPECTED_CALIBRATION:
        raise ValueError(f"Calibration counts differ from the protocol: {found}")
    rows = split_table().set_index("record")
    rows = rows[rows["split"] == "calibration"]
    flags = rows.loc[challenge["record"].to_numpy(dtype=str), list(SUPERCLASSES)]
    return flags.fillna(0).astype(bool).reset_index(drop=True)


def component_scores(arrays: dict[str, np.ndarray], challenge: pd.DataFrame
                     ) -> dict[str, dict[str, np.ndarray]]:
    """
    Readout logits and log distances of SPH and the Challenge calibration records per encoder.

    Parameters
    ----------
    arrays : dict[str, np.ndarray]
        022b's arrays.
    challenge : pd.DataFrame
        026b's Challenge calibration scores.

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        Per encoder: ``sph_r``, ``sph_d``, ``cal_r`` and ``cal_d``.
    """
    sph_scores = checked_csv(PRIOR026B, "sph_scores.csv", dtype={"ecg_id": str}, float_precision="round_trip")
    outside_ptbxl = arrays["calibration_families"] != "ptbxl"
    return {name: {"sph_r": head_logits(arrays[f"sph_{name}_pooled"]),
                   "sph_d": np.log(sph_scores[f"{name}_pooled"].to_numpy(dtype=np.float64)),
                   "cal_r": head_logits(arrays[f"calibration_{name}_pooled"][outside_ptbxl]),
                   "cal_d": np.log(challenge[f"{name}_pooled"].to_numpy(dtype=np.float64))}
            for name in ENCODERS}


def build_rules(sph_r: np.ndarray, sph_d: np.ndarray, cal_r: np.ndarray, cal_d: np.ndarray,
                cal_y: np.ndarray, stack_rows: np.ndarray) -> tuple[dict[str, dict[str, np.ndarray]],
                                                                    dict[str, Any]]:
    """
    Every rule's SPH scores and source-normal scores from one readout and the distance.

    Parameters
    ----------
    sph_r, sph_d : np.ndarray
        Readout logit and log distance of the SPH ECGs.
    cal_r, cal_d : np.ndarray
        The same of the Challenge calibration records.
    cal_y : np.ndarray
        Standard label of the calibration records.
    stack_rows : np.ndarray
        Calibration records the stack is fitted on.

    Returns
    -------
    tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]
        Per rule in ``RULES``: ``sph`` and ``source`` scores (two columns for ``either``); and the fitted
        standardization and stack coefficients.
    """
    normal = cal_y == 0
    r_stats, d_stats = normal_standardizer(cal_r[normal]), normal_standardizer(cal_d[normal])
    sph_z = np.column_stack([standardize(sph_r, r_stats), standardize(sph_d, d_stats)])
    cal_z = np.column_stack([standardize(cal_r, r_stats), standardize(cal_d, d_stats)])
    stack = fit_stack(cal_z[stack_rows], cal_y[stack_rows])
    sph = {"readout": sph_r, "distance": sph_d, "zmean": sph_z.mean(axis=1),
           "stack": stack.decision_function(sph_z), "either": np.column_stack([sph_r, sph_d])}
    source = {"readout": cal_r, "distance": cal_d, "zmean": cal_z.mean(axis=1),
              "stack": stack.decision_function(cal_z), "either": np.column_stack([cal_r, cal_d])}
    fitted = {"readout_mean_sd": r_stats, "distance_mean_sd": d_stats,
              "stack_coefficients": [float(value) for value in stack.coef_[0]],
              "stack_intercept": float(stack.intercept_[0]), "stack_rows": int(stack_rows.sum()),
              "stack_iterations": int(stack.n_iter_[0])}
    return {rule: {"sph": sph[rule], "source": source[rule][normal]} for rule in RULES}, fitted


def heldout_readouts(arrays: dict[str, np.ndarray]) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]],
                                                             dict[str, Any]]:
    """
    Refit the pooled readout on all rows and without each superclass; check the full refit against 022b.

    Parameters
    ----------
    arrays : dict[str, np.ndarray]
        022b's arrays.

    Returns
    -------
    tuple[dict[str, dict[str, dict[str, np.ndarray]]], dict[str, Any]]
        Per encoder and superclass: SPH and Challenge calibration probabilities (``sph``, ``cal``); and the
        checks (hashes, counts, reproduction differences, fit iterations).

    Raises
    ------
    ValueError
        If an input differs from its receipt, a row order or count differs, or the full refit does not
        reproduce 022b.
    """
    hashes022 = prior022_identity()
    prior022 = json.loads((PRIOR022 / "result.json").read_text())
    caches, cache_hashes = open_caches()
    if cache_hashes != prior022["identity"]["feature_caches"]:
        raise ValueError("Feature caches differ from Experiment 022")
    _, feature_hashes = feature_identity()
    groups = cohorts(ptb_table())
    with np.load(PRIOR022 / "ptb_features.npz") as saved:
        extracted = {name: saved[name] for name in saved.files}
    encoded = ptb_encoder_features(groups, caches, extracted)
    train = groups["train"]
    standard = train["standard"].notna().to_numpy()
    ptb_y = train.loc[standard, "standard"].to_numpy(dtype=np.int64)
    classes = record_classes().loc[train.loc[standard, "ecg_id"].to_numpy(dtype=np.int64)]
    ptb_flags = np.column_stack([classes["superclasses"].apply(lambda found, s=name: s in found).to_numpy()
                                 for name in SUPERCLASSES])

    rows, challenge_x = challenge_features(split_table())
    in_train = (rows["split"] == "train").to_numpy()
    in_calibration = (rows["split"] == "calibration").to_numpy()
    saved_rows = checked_csv(PRIOR022B, "training_rows.csv", dtype={"record": str})
    loaded = rows[in_train]
    same = [np.array_equal(saved_rows[column].to_numpy(dtype=str), loaded[column].to_numpy(dtype=str))
            for column in ("record", "source")]
    if not all(same):
        raise ValueError("Challenge training rows differ from 022b's training_rows.csv")
    kept = saved_rows["kept"].to_numpy(dtype=bool)
    outside_ptbxl = arrays["calibration_families"] != "ptbxl"
    if not np.array_equal(rows.loc[in_calibration, "record"].to_numpy(dtype=str),
                          arrays["calibration_records"][outside_ptbxl]):
        raise ValueError("Challenge calibration rows differ from 022b's order")
    kept_rows = rows[in_train][kept]
    y = np.concatenate([ptb_y, kept_rows["primary"].to_numpy(dtype=np.int64)])
    flags = np.concatenate([ptb_flags, kept_rows[list(SUPERCLASSES)].fillna(0).to_numpy(dtype=bool)])
    counts = {"all": (len(y), int(y.sum())),
              **{name: (int((~flags[:, i]).sum()), int(y[~flags[:, i]].sum()))
                 for i, name in enumerate(SUPERCLASSES)}}
    if counts != {"all": EXPECTED_TRAINING, **EXPECTED_WITHOUT}:
        raise ValueError(f"Training counts differ from the protocol: {counts}")

    if sha256_file(PRIOR022 / "features.npz") != prior022["outputs_sha256"]["features.npz"]:
        raise ValueError("Experiment 022 SPH features differ from its receipt")
    with np.load(PRIOR022 / "features.npz") as saved:
        positions = pd.Index(saved["ecg_ids"].astype(str)).get_indexer(arrays["sph_ecg_ids"])
        if (positions < 0).any():
            raise ValueError("Experiment 022 SPH features lack a scored row")
        sph_x = {name: saved[name][positions] for name in ENCODERS}

    readouts, differences, iterations = {}, {}, {}
    with threadpool_limits(limits=1):
        for name in ENCODERS:
            x = np.concatenate([encoded[name]["train"][standard], challenge_x[name][in_train][kept]])
            cal_x = challenge_x[name][in_calibration]
            readouts[name] = {}
            for held_out in ("all", *SUPERCLASSES):
                began = time.monotonic()
                keep = np.ones(len(y), dtype=bool) if held_out == "all" else \
                    ~flags[:, SUPERCLASSES.index(held_out)]
                head = fit_readout(x[keep], y[keep])
                readouts[name][held_out] = {"sph": predict(head, sph_x[name]), "cal": predict(head, cal_x)}
                tag = f"{name}:{held_out}"
                iterations[tag] = int(head[1].n_iter_[0])
                print(json.dumps({"stage": f"fit:{tag}", "iterations": iterations[tag],
                                  "seconds": round(time.monotonic() - began, 1)}), flush=True)
            differences[name] = {
                "sph": float(np.abs(readouts[name]["all"]["sph"] - arrays[f"sph_{name}_pooled"]).max()),
                "calibration": float(np.abs(readouts[name]["all"]["cal"]
                                            - arrays[f"calibration_{name}_pooled"][outside_ptbxl]).max())}
            if max(differences[name].values()) > REPRODUCTION_TOLERANCE:
                raise ValueError(f"The pooled refit does not reproduce 022b for {name}: {differences[name]}")
    checks = {"experiment022": hashes022, "feature_caches": cache_hashes,
              "challenge_features": feature_hashes, "training_counts": counts,
              "reproduction_max_abs_difference": differences, "iterations": iterations}
    return readouts, checks


def score_rule(rule: dict[str, np.ndarray], evaluation: np.ndarray, local_normal: np.ndarray,
               masks: dict[str, np.ndarray], plans: dict[int, list[np.ndarray]], label: dict[str, str]
               ) -> tuple[list[dict[str, Any]], dict[tuple[int, int], np.ndarray]]:
    """
    Thresholds, achieved rates and group sensitivities of one rule for every size, draw and budget.

    Parameters
    ----------
    rule : dict[str, np.ndarray]
        ``sph`` and ``source`` scores of the rule.
    evaluation : np.ndarray
        Evaluation mask per SPH ECG.
    local_normal : np.ndarray
        SPH positions of the local pool's normals.
    masks : dict[str, np.ndarray]
        Group masks of the evaluation half.
    plans : dict[int, list[np.ndarray]]
        Positions into ``local_normal`` per size and draw.
    label : dict[str, str]
        ``score`` and ``encoder`` of the rows.

    Returns
    -------
    tuple[list[dict[str, Any]], dict[tuple[int, int], np.ndarray]]
        One row per size, draw and budget (size 0 is the source normals), and the referral share of each
        evaluation ECG per (size, budget).
    """
    evaluation_scores = rule["sph"][evaluation]
    local_scores = rule["sph"][local_normal]
    samples = {0: [rule["source"]], **{m: [local_scores[positions] for positions in draws]
                                       for m, draws in plans.items()}}
    rows, shares = [], {}
    for m, draws in samples.items():
        for budget in BUDGETS:
            thresholds = [rule_thresholds(normals, budget) for normals in draws]
            matrix = np.column_stack([referred(evaluation_scores, values) for values in thresholds])
            shares[(m, budget)] = matrix.mean(axis=1)
            rates = {group: matrix[mask].mean(axis=0) for group, mask in masks.items()}
            for draw, values in enumerate(thresholds):
                rows.append({**label, "m": m, "draw": draw, "budget": budget / 1000,
                             "threshold": float(values[0]),
                             "threshold_distance": float(values[1]) if len(values) > 1 else np.nan,
                             "rate": float(rates["normal"][draw]),
                             "sensitivity": float(rates["positive"][draw]),
                             **{f"sensitivity_{group}": float(rates[group][draw])
                                for group in (*SUPERCLASSES, *ONLY)}})
    return rows, shares


def reproduce_030(draws: pd.DataFrame) -> dict[str, float]:
    """
    Require the readout and distance alone to reproduce 030's draws.

    Parameters
    ----------
    draws : pd.DataFrame
        This run's draw rows.

    Returns
    -------
    dict[str, float]
        Largest differences in threshold (on this run's scale) and in any rate.

    Raises
    ------
    ValueError
        If a row differs.
    """
    prior = checked_csv(PRIOR030, "draws.csv", float_precision="round_trip")
    columns = ["rate", "sensitivity", *(f"sensitivity_{name}" for name in SUPERCLASSES)]
    worst = {"threshold": 0.0, "rates": 0.0}
    for rule, score, transform in (("readout", "pooled", head_logits), ("distance", "normal_ref", np.log)):
        for name in ENCODERS:
            ours = draws[(draws["score"] == rule) & (draws["encoder"] == name) & draws["m"].isin(SIZES)]
            theirs = prior[(prior["score"] == score) & (prior["encoder"] == name) & prior["m"].isin(SIZES)]
            keys = ["m", "budget", "draw"]
            ours, theirs = ours.sort_values(keys), theirs.sort_values(keys)
            if len(ours) != len(theirs) or not np.array_equal(ours[keys].to_numpy(), theirs[keys].to_numpy()):
                raise ValueError(f"Draw rows differ from 030 for {rule}:{name}")
            threshold = np.abs(ours["threshold"].to_numpy() - transform(theirs["threshold"].to_numpy())).max()
            rates = np.abs(ours[columns].to_numpy() - theirs[columns].to_numpy()).max()
            worst = {"threshold": max(worst["threshold"], float(threshold)),
                     "rates": max(worst["rates"], float(rates))}
    if worst["threshold"] > 0 or worst["rates"] > 1e-12:
        raise ValueError(f"The single scores do not reproduce 030: {worst}")
    return worst


def bootstrap(shares: dict[tuple[str, str, int, int], np.ndarray], masks: dict[str, np.ndarray],
              patients: np.ndarray) -> dict[str, np.ndarray]:
    """
    Patient-bootstrap distribution of the draw-averaged referral share of every group and key.

    Parameters
    ----------
    shares : dict[tuple[str, str, int, int], np.ndarray]
        Referral share of each evaluation ECG per (score, encoder, size, budget).
    masks : dict[str, np.ndarray]
        Group masks of the evaluation half.
    patients : np.ndarray
        Patient ID of each evaluation ECG.

    Returns
    -------
    dict[str, np.ndarray]
        Per group, a ``(RESAMPLES, keys)`` array in the order of ``shares``.

    Raises
    ------
    ValueError
        If a resample holds no ECG of a group.
    """
    unique, codes = np.unique(patients, return_inverse=True)
    counts = bootstrap_counts(len(unique), RESAMPLES, BOOTSTRAP_SEED)
    matrix = np.column_stack(list(shares.values()))
    result = {}
    for group, mask in masks.items():
        denominator = counts @ group_sums(codes, mask[:, None].astype(np.float64), len(unique))
        if (denominator == 0).any():
            raise ValueError(f"A resample holds no {group} ECG")
        result[group] = counts @ group_sums(codes, matrix * mask[:, None], len(unique)) / denominator
    return result


def summarize(draws: pd.DataFrame, boot: dict[str, np.ndarray], keys: list[tuple[str, str, int, int]]
              ) -> list[dict[str, Any]]:
    """
    Summary per key: means and 5th-95th percentiles over draws, with patient-bootstrap intervals.

    Parameters
    ----------
    draws : pd.DataFrame
        Every draw row.
    boot : dict[str, np.ndarray]
        Output of ``bootstrap``.
    keys : list[tuple[str, str, int, int]]
        Keys in the column order of ``boot``.

    Returns
    -------
    list[dict[str, Any]]
        One summary row per key.
    """
    columns = {"normal": "rate", "positive": "sensitivity",
               **{group: f"sensitivity_{group}" for group in (*SUPERCLASSES, *ONLY)}}
    grouped = draws.groupby(["score", "encoder", "m", "budget"], sort=False)
    summary = []
    for index, (score, name, m, budget) in enumerate(keys):
        group = grouped.get_group((score, name, m, budget / 1000))
        target = budget / 1000
        row = {"score": score, "encoder": name, "m": m, "budget": target, "draws": len(group),
               "share_rate_within_1pp": float(((group["rate"] - target).abs() <= BAND + 1e-12).mean())}
        for label, column in columns.items():
            low, high = np.percentile(boot[label][:, index], [2.5, 97.5])
            row.update({f"{column}_mean": float(group[column].mean()),
                        f"{column}_p5": float(group[column].quantile(0.05)),
                        f"{column}_p95": float(group[column].quantile(0.95)),
                        f"{column}_ci": [float(low), float(high)]})
        for prevalence in PREVALENCES:
            tag = f"{round(prevalence * 100)}pct"
            row[f"referrals_per_1000_at_{tag}"] = float(
                referrals_per_1000(group["sensitivity"], group["rate"], prevalence).mean())
            row[f"caught_per_1000_at_{tag}"] = float(1000 * prevalence * group["sensitivity"].mean())
        summary.append(row)
    return summary


def contrasts(summary: list[dict[str, Any]], boot: dict[str, np.ndarray],
              keys: list[tuple[str, str, int, int]]) -> list[dict[str, Any]]:
    """
    Paired differences between rules on the same draws and resamples.

    Parameters
    ----------
    summary : list[dict[str, Any]]
        Output of ``summarize``.
    boot : dict[str, np.ndarray]
        Output of ``bootstrap``.
    keys : list[tuple[str, str, int, int]]
        Keys in the column order of ``boot``.

    Returns
    -------
    list[dict[str, Any]]
        One row per pair, encoder, size, budget and group.
    """
    column_of = {key: index for index, key in enumerate(keys)}
    mean_column = {"normal": "rate_mean", "positive": "sensitivity_mean",
                   **{group: f"sensitivity_{group}_mean" for group in (*SUPERCLASSES, *ONLY)}}
    pairs = [(rule, "readout", GROUPS) for rule in (*HYBRIDS, "distance")]
    for name in SUPERCLASSES:
        outcomes = ("positive", "normal", name, f"{name}_only")
        pairs += [(f"{rule}_without_{name}", f"readout_without_{name}", outcomes) for rule in HYBRIDS]
        pairs += [(f"readout_without_{name}", "readout", outcomes),
                  (f"zmean_without_{name}", "readout", outcomes)]
    rows = []
    for first, second, outcomes in pairs:
        for name in ENCODERS:
            for m in (0, *SIZES):
                for budget in BUDGETS:
                    a, b = column_of[(first, name, m, budget)], column_of[(second, name, m, budget)]
                    for group in outcomes:
                        difference = boot[group][:, a] - boot[group][:, b]
                        low, high = np.percentile(difference, [2.5, 97.5])
                        column = mean_column[group]
                        rows.append({"first": first, "second": second, "encoder": name, "m": m,
                                     "budget": budget / 1000, "group": group,
                                     "difference": summary[a][column] - summary[b][column],
                                     "ci": [float(low), float(high)]})
    return rows


def side(interval: list[float]) -> str:
    """Where an interval lies relative to 0."""
    if interval[0] > 0:
        return "above 0"
    if interval[1] < 0:
        return "below 0"
    return "includes 0"


def meets_adoption(sensitivity: dict[str, Any], rate: dict[str, Any]) -> bool:
    """
    Whether a hybrid-minus-readout contrast meets the pre-registered adoption criteria.

    Parameters
    ----------
    sensitivity : dict[str, Any]
        Sensitivity contrast with ``difference`` and ``ci``.
    rate : dict[str, Any]
        Achieved-rate contrast on the same keys.

    Returns
    -------
    bool
        Interval above 0, difference at least ``ADOPT_MINIMUM`` and rate difference at most ``RATE_MAXIMUM``.
    """
    return (sensitivity["ci"][0] > 0 and sensitivity["difference"] >= ADOPT_MINIMUM
            and rate["difference"] <= RATE_MAXIMUM)


def reading(table: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Apply the pre-registered primary decision and the held-out readings.

    Parameters
    ----------
    table : list[dict[str, Any]]
        Output of ``contrasts``.

    Returns
    -------
    dict[str, Any]
        Primary contrast, rate contrast, decision, and per-superclass held-out contrasts.
    """
    def pick(first: str, second: str, group: str, encoder: str = PRIMARY["encoder"]) -> dict[str, Any]:
        (row,) = [row for row in table if row["first"] == first and row["second"] == second
                  and row["group"] == group and row["encoder"] == encoder and row["m"] == PRIMARY["m"]
                  and row["budget"] == PRIMARY["budget"] / 1000]
        return row

    primary, rate = pick("zmean", "readout", "positive"), pick("zmean", "readout", "normal")
    decision = "not distinguished; the readout alone stays"
    if meets_adoption(primary, rate):
        decision = "adopt the hybrid"
    elif primary["ci"][1] < 0:
        decision = "the hybrid hurts; the readout alone stays"
    secondary = {}
    for encoder in ENCODERS:
        for rule in HYBRIDS:
            row = pick(rule, "readout", "positive", encoder)
            rate_row = pick(rule, "readout", "normal", encoder)
            secondary[f"{rule}:{encoder}"] = {"sensitivity": row, "rate": rate_row, "side": side(row["ci"]),
                                              "meets_adoption_criteria": meets_adoption(row, rate_row)}
    heldout = {}
    for name in SUPERCLASSES:
        for encoder in ENCODERS:
            on_s = pick(f"zmean_without_{name}", f"readout_without_{name}", name, encoder)
            heldout[f"{name}:{encoder}"] = {
                "zmean_minus_readout_on_S": on_s, "helps": on_s["ci"][0] > 0,
                "zmean_minus_readout_on_S_only": pick(f"zmean_without_{name}", f"readout_without_{name}",
                                                      f"{name}_only", encoder),
                "readout_without_minus_full_on_S": pick(f"readout_without_{name}", "readout", name, encoder),
                "zmean_without_minus_full_on_S": pick(f"zmean_without_{name}", "readout", name, encoder)}
    return {"primary": primary, "primary_side": side(primary["ci"]), "primary_rate": rate,
            "decision": decision, "secondary": secondary, "heldout": heldout}


def main() -> None:
    """Run the experiment once and write the aggregate result."""
    if (OUTPUT / "result.json").exists():
        raise FileExistsError("Experiment 031 v1 has already run")
    started = time.monotonic()
    arrays, evaluation, masks030, split_counts = load_site()
    masks = group_masks(masks030)
    _, challenge = load_heads(arrays)
    cal_y = challenge["primary"].to_numpy(dtype=np.int64)
    flags = calibration_flags(arrays, challenge)
    stack_counts = {name: int((~flags[name]).sum()) for name in SUPERCLASSES}
    if stack_counts != EXPECTED_CALIBRATION_WITHOUT:
        raise ValueError(f"Calibration counts without S differ from the protocol: {stack_counts}")
    components = component_scores(arrays, challenge)
    y = arrays["sph_labels"]
    local_normal = np.flatnonzero(~evaluation & (y == 0))
    plans = {m: [np.sort(np.random.default_rng([SEED, m, draw]).choice(len(local_normal), m, replace=False))
                 for draw in range(DRAWS)] for m in SIZES}
    print(json.dumps({"stage": "loaded", "split": split_counts, "local_normals": len(local_normal)}),
          flush=True)

    readouts, heldout_checks = heldout_readouts(arrays)
    print(json.dumps({"stage": "readouts", "seconds": round(time.monotonic() - started)}), flush=True)

    with threadpool_limits(limits=4):
        rules, fitted = {}, {}
        everything = np.ones(len(cal_y), dtype=bool)
        for name in ENCODERS:
            parts = components[name]
            built, fitted[f"{name}:full"] = build_rules(parts["sph_r"], parts["sph_d"], parts["cal_r"],
                                                        parts["cal_d"], cal_y, everything)
            rules.update({(rule, name): values for rule, values in built.items()})
            for held_out in SUPERCLASSES:
                sph_r = head_logits(readouts[name][held_out]["sph"])
                cal_r = head_logits(readouts[name][held_out]["cal"])
                built, fitted[f"{name}:without_{held_out}"] = build_rules(
                    sph_r, parts["sph_d"], cal_r, parts["cal_d"], cal_y, ~flags[held_out].to_numpy())
                rules.update({(f"{rule}_without_{held_out}", name): values for rule, values in built.items()
                              if rule != "distance"})
        rows, shares = [], {}
        for (score, name), rule in rules.items():
            rule_rows, rule_shares = score_rule(rule, evaluation, local_normal, masks, plans,
                                                {"score": score, "encoder": name})
            rows.extend(rule_rows)
            shares.update({(score, name, m, budget): values for (m, budget), values in rule_shares.items()})
        draws = pd.DataFrame(rows)
        reproduction030 = reproduce_030(draws)
        print(json.dumps({"stage": "scored", "rules": len(rules), "reproduction030": reproduction030,
                          "seconds": round(time.monotonic() - started)}), flush=True)
        keys = list(shares)
        boot = bootstrap(shares, masks, arrays["sph_patient_ids"][evaluation])
        print(json.dumps({"stage": "bootstrap", "seconds": round(time.monotonic() - started)}), flush=True)
    summary = summarize(draws, boot, keys)
    for row, key in zip(summary, keys, strict=True):
        if abs(shares[key][masks["positive"]].mean() - row["sensitivity_mean"]) > 1e-9:
            raise ValueError(f"Referral shares do not reproduce the draw mean for {key}")
    table = contrasts(summary, boot, keys)
    answers = reading(table)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    draws.to_csv(OUTPUT / "draws.csv", index=False)
    result = {
        "status": "complete",
        "identity": {"experiment022b_result_sha256": sha256_file(PRIOR022B / "result.json"),
                     "experiment022b_predictions_sha256": sha256_file(PRIOR022B / "predictions.npz"),
                     "experiment026b_result_sha256": sha256_file(PRIOR026B / "result.json"),
                     "experiment029_result_sha256": sha256_file(PRIOR029 / "result.json"),
                     "experiment030_result_sha256": sha256_file(PRIOR030 / "result.json"),
                     "sources": {name: sha256_file(ROOT / name) for name in SOURCES}},
        "seed": SEED, "bootstrap_seed": BOOTSTRAP_SEED, "resamples": RESAMPLES, "draws": DRAWS,
        "sizes": [0, *SIZES], "budgets": [b / 1000 for b in BUDGETS], "encoders": list(ENCODERS),
        "split": split_counts, "group_counts": {name: int(mask.sum()) for name, mask in masks.items()},
        "calibration_counts_without": stack_counts, "local_normals": len(local_normal),
        "heldout_checks": heldout_checks, "reproduction030_max_difference": reproduction030,
        "fitted": fitted, "summary": summary, "contrasts": table, "reading": answers,
        "outputs_sha256": {"draws.csv": sha256_file(OUTPUT / "draws.csv")},
        "challenge_test_read": False, "ptbxl_calibration_read": False, "ptbxl_test_read": False,
        "runtime_seconds": time.monotonic() - started,
    }
    write_json_atomic(OUTPUT / "result.json", result)
    print(json.dumps({"stage": "complete", "runtime_seconds": result["runtime_seconds"],
                      "decision": answers["decision"], "primary": answers["primary"]}), flush=True)


if __name__ == "__main__":
    main()
