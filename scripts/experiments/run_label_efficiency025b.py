"""Experiment 025b: 025's label-efficiency draws on pooled PTB-XL and Challenge labels, read on SPH."""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from ecg_experiment.external_readout import prior_cpc_features, prior_identity
from ecg_experiment.files import sha256_file, write_json_atomic, write_npz_atomic
from ecg_experiment.full_development import cohorts, fit_logistic, predict, ptb_table
from ecg_experiment.label_efficiency import draw_subset, paired_summary, reading, smallest_budget, summarize
from ecg_experiment.label_efficiency_multisource import (
    bootstrap_mean_auroc,
    challenge_units,
    decision,
    interval,
    plus_positions,
    resample_counts,
    verdict,
)
from ecg_experiment.multisource_readout import arm_members
from scripts.experiments.run_calibrated_threshold027 import prior022_identity, sph_rows
from scripts.experiments.run_label_efficiency025 import (
    BUDGETS,
    DRAWS,
    READ_BUDGETS,
    draw_plans,
    load_features,
    select_rows,
)
from scripts.experiments.run_label_efficiency025 import OUTPUT as PRIOR025
from scripts.experiments.run_label_efficiency025 import SEED as DRAW_SEED
from scripts.experiments.run_label_efficiency025 import identity as identity025
from scripts.experiments.run_multisource_calibration027b import feature_identity
from scripts.experiments.run_multisource_manifold026b import challenge_features, interval_side, split_table
from scripts.experiments.run_multisource_readout022b import NINGBO, PRIOR022, SPLITS, sph_features, stacked
from scripts.experiments.run_multisource_readout022b import OUTPUT as PRIOR022B
from scripts.experiments.run_sph_external022 import open_caches, ptb_encoder_features

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/experiment025b_label_efficiency_multisource_v1"
ENCODERS = ("xecg", "jepa", "cpc")
PRIMARY_ENCODER = "xecg"
ARMS = ("ptbxl", "pooled", "ptbxl_plus_challenge")
ARM_CONTRASTS = (("pooled", "ptbxl"), ("ptbxl_plus_challenge", "ptbxl"))
ENCODER_CONTRASTS = (("jepa", "cpc"), ("xecg", "cpc"), ("xecg", "jepa"))
RANKING_SETS = ("sph", "development_original")
BOOTSTRAP_SEED = 32032
BOOTSTRAP_DRAWS = 2000
REPRODUCTION_TOLERANCE = 1e-9
NEGLIGIBLE_AUROC = 0.005
FAMILIES = ("chapman_ningbo", "georgia", "cpsc")
EXPECTED = {
    "ptbxl_pool": (15359, 9487), "pooled_pool": (37853, 27007), "pooled_units": 35845,
    "challenge_training": {"ningbo": (9952, 7371), "chapman_shaoxing": (3050, 2231), "georgia": (5136, 4113),
                           "cpsc_2018": (2600, 2049), "cpsc_2018_extra": (1756, 1756)},
    "sets": {"sph": (21008, 7190), "development_original": (1306, 843), "development_added": (266, 41),
             "family:chapman_ningbo": (4432, 3254), "family:georgia": (1718, 1372),
             "family:cpsc": (1462, 1279)},
}
SOURCES = (
    "ecg_experiment/label_efficiency_multisource.py", "ecg_experiment/label_efficiency.py",
    "ecg_experiment/multisource_readout.py", "ecg_experiment/normal_manifold.py",
    "ecg_experiment/full_development.py", "ecg_experiment/external_readout.py",
    "ecg_experiment/challenge_features.py", "ecg_experiment/challenge_splits.py",
    "scripts/experiments/run_label_efficiency025b.py", "scripts/experiments/run_label_efficiency025.py",
    "scripts/experiments/run_multisource_readout022b.py", "scripts/experiments/run_sph_external022.py",
    "scripts/experiments/run_calibrated_threshold027.py",
    "scripts/experiments/run_multisource_calibration027b.py",
    "scripts/experiments/run_multisource_manifold026b.py", "pyproject.toml", "uv.lock",
    "docs/experiment-025b-label-efficiency-multisource.md",
)

Plan = tuple[str, str, int, int, np.ndarray]


def prior_receipts() -> dict[str, str]:
    """
    Require 025's draws and predictions and 022b's training rows to match their results.

    Returns
    -------
    dict[str, str]
        SHA-256 of each artifact.

    Raises
    ------
    ValueError
        If an artifact differs from the hash its result recorded.
    """
    result025 = json.loads((PRIOR025 / "result.json").read_text())
    result022b = json.loads((PRIOR022B / "result.json").read_text())
    hashes = {"025/draws.csv": sha256_file(PRIOR025 / "draws.csv"),
              "025/all_budget_predictions.npz": sha256_file(PRIOR025 / "all_budget_predictions.npz"),
              "025/result.json": sha256_file(PRIOR025 / "result.json"),
              "022b/training_rows.csv": sha256_file(PRIOR022B / "training_rows.csv"),
              "022b/result.json": sha256_file(PRIOR022B / "result.json")}
    expected = {"025/draws.csv": result025["draws_sha256"],
                "025/all_budget_predictions.npz": result025["predictions_sha256"],
                "022b/training_rows.csv": result022b["outputs_sha256"]["training_rows.csv"]}
    if any(hashes[name] != digest for name, digest in expected.items()):
        raise ValueError("A 025 or 022b artifact differs from its result")
    return hashes


def development_features(groups: dict[str, pd.DataFrame], caches: Any, evaluation: pd.DataFrame,
                         reference: dict[str, tuple[np.ndarray, np.ndarray]]
                         ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, pd.DataFrame]]:
    """
    Ordinary and hard added development features, with the ordinary rows checked against 025's.

    Parameters
    ----------
    groups : dict[str, pd.DataFrame]
        Output of ``full_development.cohorts``.
    caches : Any
        Output of 022's ``open_caches``.
    evaluation : pd.DataFrame
        025's evaluation rows.
    reference : dict[str, tuple[np.ndarray, np.ndarray]]
        025's pool and evaluation features per encoder.

    Returns
    -------
    tuple[dict[str, dict[str, np.ndarray]], dict[str, pd.DataFrame]]
        Features per encoder keyed by ``development_original`` and ``development_added``, and those rows.

    Raises
    ------
    ValueError
        If the ordinary rows or their features differ from 025's.
    """
    with np.load(PRIOR022 / "ptb_features.npz") as saved:
        extracted = {name: saved[name] for name in saved.files}
    encoded = ptb_encoder_features(groups, caches, extracted)
    development = {"cpc": prior_cpc_features()["development"], "jepa": encoded["jepa"]["development"],
                   "xecg": encoded["xecg"]["development"]}
    rows = groups["development"]
    labeled = rows["standard"].notna().to_numpy()
    original = rows["original"].to_numpy(dtype=bool)
    masks = {"development_original": labeled & original, "development_added": labeled & ~original}
    if not rows[masks["development_original"]].index.equals(evaluation.index):
        raise ValueError("Ordinary development rows differ from Experiment 025")
    features = {name: {key: values[mask] for key, mask in masks.items()}
                for name, values in development.items()}
    for name in ENCODERS:
        if not np.array_equal(features[name]["development_original"], reference[name][1]):
            raise ValueError(f"022's development features differ from 025's for {name}")
    return features, {key: rows[mask] for key, mask in masks.items()}


def challenge_rows() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    022b's kept Challenge training rows and the Challenge calibration rows with their features.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], np.ndarray, np.ndarray]
        Kept training rows, calibration rows, all features per encoder, and the training and calibration
        selections of those features.

    Raises
    ------
    ValueError
        If the loaded training rows differ from 022b's.
    """
    rows, features = challenge_features(split_table())
    in_train = (rows["split"] == "train").to_numpy()
    in_calibration = (rows["split"] == "calibration").to_numpy()
    train = rows[in_train].reset_index(drop=True)
    saved = pd.read_csv(PRIOR022B / "training_rows.csv", dtype={"record": str})
    same = all(np.array_equal(saved[column].to_numpy(dtype=str), train[column].to_numpy(dtype=str))
               for column in ("source", "record"))
    if not same:
        raise ValueError("Challenge training rows differ from 022b")
    kept = saved["kept"].to_numpy(dtype=bool)
    selected = np.flatnonzero(in_train)[kept]
    return (train[kept].reset_index(drop=True), rows[in_calibration].reset_index(drop=True), features,
            selected, in_calibration)


def plans_for(pool: pd.DataFrame, units: np.ndarray, y: np.ndarray, members: dict[str, np.ndarray]
              ) -> list[Plan]:
    """
    Arm, budget, draw, seed and stacked positions of every fit.

    Parameters
    ----------
    pool : pd.DataFrame
        025's PTB-XL pool, the first rows of the stacked table.
    units : np.ndarray
        Patient unit of each stacked row.
    y : np.ndarray
        Label of each stacked row.
    members : dict[str, np.ndarray]
        022b's ``arm_members`` of the stacked rows.

    Returns
    -------
    list[Plan]
        Plans of ``ptbxl`` (025's draws), ``pooled`` and ``ptbxl_plus_challenge``.

    Raises
    ------
    ValueError
        If the PTB-XL rows are not the first rows of the stacked table.
    """
    ptb = np.flatnonzero(members["ptbxl"])
    if not np.array_equal(ptb, np.arange(len(pool))):
        raise ValueError("PTB-XL rows must come first in the stacked table")
    pooled = np.flatnonzero(members["pooled"])
    reference = draw_plans(pool)
    plans: list[Plan] = [("ptbxl", budget, draw, seed, ptb[positions])
                         for budget, draw, seed, positions in reference]
    for size in BUDGETS:
        for draw in range(DRAWS):
            positions = draw_subset(units[pooled], y[pooled], size, DRAW_SEED + draw)
            plans.append(("pooled", str(size), draw, DRAW_SEED + draw, pooled[positions]))
    plans.append(("pooled", "all", 0, DRAW_SEED, pooled))
    plans += [("ptbxl_plus_challenge", budget, draw, seed, plus_positions(ptb[positions], len(pool), len(y)))
              for budget, draw, seed, positions in reference if budget != "all"]
    return plans


def plan_table(plans: list[Plan], ids: np.ndarray, y: np.ndarray, units: np.ndarray,
               families: np.ndarray) -> pd.DataFrame:
    """
    Size, positives, source mix and subset hash of every fit.

    Parameters
    ----------
    plans : list[Plan]
        Output of ``plans_for``.
    ids, y, units, families : np.ndarray
        Record ID, label, patient unit and family of each stacked row.

    Returns
    -------
    pd.DataFrame
        One row per plan.
    """
    rows = []
    for arm, budget, draw, seed, positions in plans:
        mix = pd.Series(families[positions]).value_counts()
        rows.append({"arm": arm, "budget": budget, "draw": draw, "seed": seed, "records": len(positions),
                     "positives": int(y[positions].sum()), "patients": len(np.unique(units[positions])),
                     **{f"records_{family}": int(mix.get(family, 0)) for family in ("ptbxl", *FAMILIES)},
                     "subset_sha256": hashlib.sha256("\n".join(ids[positions]).encode()).hexdigest()})
    return pd.DataFrame(rows)


def evaluation_sets(sph: pd.DataFrame, development: dict[str, pd.DataFrame],
                    calibration: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """
    Labels, bootstrap units and row selections of every evaluation set.

    Parameters
    ----------
    sph : pd.DataFrame
        SPH rows with ``primary`` and ``patient_id``.
    development : dict[str, pd.DataFrame]
        Ordinary and added development rows.
    calibration : pd.DataFrame
        Challenge calibration rows.

    Returns
    -------
    dict[str, dict[str, Any]]
        Per set: the scored input it reads (``scores_of``), its ``selection``, ``y`` and ``units``.
    """
    sets = {"sph": {"scores_of": "sph", "selection": np.ones(len(sph), dtype=bool),
                    "y": sph["primary"].to_numpy(dtype=np.int64),
                    "units": sph["patient_id"].to_numpy(dtype=str)}}
    for name, rows in development.items():
        sets[name] = {"scores_of": name, "selection": np.ones(len(rows), dtype=bool),
                      "y": rows["standard"].to_numpy(dtype=np.int64),
                      "units": rows["patient_id"].to_numpy(dtype=str)}
    family = calibration["family"].to_numpy(dtype=str)
    for name in FAMILIES:
        selected = family == name
        sets[f"family:{name}"] = {"scores_of": "calibration", "selection": selected,
                                  "y": calibration.loc[selected, "primary"].to_numpy(dtype=np.int64),
                                  "units": calibration.loc[selected, "record"].to_numpy(dtype=str)}
    return sets


def check_counts(pool: pd.DataFrame, train: pd.DataFrame, y: np.ndarray, units: np.ndarray,
                 sets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Require the pool and evaluation counts of the protocol.

    Parameters
    ----------
    pool : pd.DataFrame
        025's PTB-XL pool.
    train : pd.DataFrame
        Kept Challenge training rows.
    y, units : np.ndarray
        Labels and patient units of the stacked rows.
    sets : dict[str, dict[str, Any]]
        Output of ``evaluation_sets``.

    Returns
    -------
    dict[str, Any]
        The counts found.

    Raises
    ------
    ValueError
        If a count differs from the protocol.
    """
    found = {
        "ptbxl_pool": (len(pool), int(pool["standard"].sum())), "pooled_pool": (len(y), int(y.sum())),
        "pooled_units": len(np.unique(units)),
        "challenge_training": {source: (len(part), int(part["primary"].sum()))
                               for source, part in train.groupby("source")},
        "sets": {name: (len(spec["y"]), int(spec["y"].sum())) for name, spec in sets.items()},
    }
    if found != EXPECTED:
        raise ValueError(f"Counts differ from the protocol: {found}")
    return found


def evaluate_encoder(name: str, x: np.ndarray, y: np.ndarray, plans: list[Plan],
                     inputs: dict[str, np.ndarray], sets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Fit every plan on one encoder, score every set, and bootstrap the draw-mean AUROC of each arm and budget.

    Parameters
    ----------
    name : str
        Encoder name.
    x, y : np.ndarray
        Stacked training features and labels.
    plans : list[Plan]
        Output of ``plans_for``.
    inputs : dict[str, np.ndarray]
        Features of each scored input.
    sets : dict[str, dict[str, Any]]
        Output of ``evaluation_sets``.

    Returns
    -------
    dict[str, Any]
        Per-draw metrics, N = all scores, bootstrap values, skipped resamples and fit diagnostics.
    """
    started = time.monotonic()
    rows, scores, iterations = [], {}, []
    with threadpool_limits(limits=1):
        for arm, budget, draw, _, positions in plans:
            head = fit_logistic(x[positions], y[positions])
            iterations.append(int(head[1].n_iter_[0]))
            predicted = {key: predict(head, values) for key, values in inputs.items()}
            for set_name, spec in sets.items():
                values = predicted[spec["scores_of"]][spec["selection"]]
                scores.setdefault(set_name, {}).setdefault((arm, budget), []).append(values)
                rows.append({"encoder": name, "arm": arm, "budget": budget, "draw": draw, "set": set_name,
                             "auroc": float(roc_auc_score(spec["y"], values)),
                             "average_precision": float(average_precision_score(spec["y"], values))})
            if draw == DRAWS - 1 or budget == "all":
                print(json.dumps({"stage": f"{name}:fit:{arm}:{budget}", "records": len(positions),
                                  "seconds": round(time.monotonic() - started, 1)}), flush=True)
        bootstrap, invalid = {}, {}
        for set_name, spec in sets.items():
            counts, invalid[set_name] = resample_counts(spec["units"], spec["y"], BOOTSTRAP_DRAWS,
                                                        BOOTSTRAP_SEED)
            bootstrap[set_name] = {
                f"{arm}:{budget}": bootstrap_mean_auroc(spec["y"], np.vstack(vectors), counts)
                for (arm, budget), vectors in scores[set_name].items()}
            print(json.dumps({"stage": f"{name}:bootstrap:{set_name}",
                              "seconds": round(time.monotonic() - started, 1)}), flush=True)
    final = {set_name: {arm: by_key[(arm, "all")][0] for arm in ("ptbxl", "pooled")}
             for set_name, by_key in scores.items()}
    return {"rows": rows, "final": final, "bootstrap": bootstrap, "invalid": invalid,
            "iterations": {"min": min(iterations), "max": max(iterations)},
            "seconds": time.monotonic() - started}


def reproduce_025(table: pd.DataFrame, plans: pd.DataFrame, final: dict[str, np.ndarray],
                  evaluation: pd.DataFrame) -> dict[str, float]:
    """
    Require the ``ptbxl`` arm to reproduce 025's primary draws and N = all probabilities.

    Parameters
    ----------
    table : pd.DataFrame
        Per-draw metrics of every encoder.
    plans : pd.DataFrame
        Output of ``plan_table``.
    final : dict[str, np.ndarray]
        N = all ordinary development probabilities of the ``ptbxl`` arm per encoder.
    evaluation : pd.DataFrame
        025's evaluation rows.

    Returns
    -------
    dict[str, float]
        Largest absolute AUROC, AP and probability differences.

    Raises
    ------
    ValueError
        If a subset, count or value differs from 025's.
    """
    prior = pd.read_csv(PRIOR025 / "draws.csv", dtype={"budget": str})
    prior = prior[(prior["readout"] == "primary") & prior["encoder"].isin(ENCODERS)]
    ours = table[(table["arm"] == "ptbxl") & (table["set"] == "development_original")]
    keys = ["encoder", "budget", "draw"]
    merged = prior.merge(ours, on=keys, suffixes=("_025", ""), validate="one_to_one")
    merged = merged.merge(plans[plans["arm"] == "ptbxl"], on=["budget", "draw"], suffixes=("", "_plan"))
    if len(merged) != len(prior) or len(prior) != len(ENCODERS) * (len(BUDGETS) * DRAWS + 1):
        raise ValueError("025's draws do not match the ptbxl arm")
    same = ((merged["subset_sha256"] == merged["subset_sha256_plan"]).all()
            and (merged["records"] == merged["records_plan"]).all()
            and (merged["positives"] == merged["positives_plan"]).all())
    differences = {"auroc": float((merged["auroc"] - merged["auroc_025"]).abs().max()),
                   "average_precision": float((merged["average_precision"]
                                               - merged["average_precision_025"]).abs().max())}
    with np.load(PRIOR025 / "all_budget_predictions.npz", allow_pickle=True) as saved:
        if not np.array_equal(saved["record_ids"], evaluation.index.to_numpy()):
            raise ValueError("025's saved development order differs")
        differences["all_probability"] = max(float(np.abs(final[name] - saved[f"primary_{name}"]).max())
                                             for name in ENCODERS)
    if not same or max(differences.values()) > REPRODUCTION_TOLERANCE:
        raise ValueError(f"The ptbxl arm does not reproduce Experiment 025: {differences}, subsets {same}")
    return differences


def analyse(table: pd.DataFrame, results: dict[str, Any]) -> dict[str, Any]:
    """
    Per-budget summaries, arm and encoder contrasts, and the prespecified readings.

    Parameters
    ----------
    table : pd.DataFrame
        Per-draw metrics of every encoder.
    results : dict[str, Any]
        Output of ``evaluate_encoder`` per encoder.

    Returns
    -------
    dict[str, Any]
        Summaries, contrasts, primary verdicts and decision, ranking readings and SPH label efficiency.
    """
    budgets = [str(size) for size in BUDGETS]
    wide = {metric: table.pivot_table(index=["set", "arm", "budget", "draw"], columns="encoder",
                                      values=metric) for metric in ("auroc", "average_precision")}
    summaries, arm_contrasts, encoder_contrasts = {}, {}, {}
    for (set_name, arm, budget), frame in wide["auroc"].groupby(level=["set", "arm", "budget"]):
        ap = wide["average_precision"].loc[(set_name, arm, budget)]
        summaries.setdefault(set_name, {}).setdefault(arm, {})[budget] = {
            name: {"auroc": summarize(frame[name].to_numpy()) if budget != "all"
                   else float(frame[name].iloc[0]),
                   "average_precision": summarize(ap[name].to_numpy()) if budget != "all"
                   else float(ap[name].iloc[0])} for name in ENCODERS}
        pairs = {}
        for first, second in ENCODER_CONTRASTS:
            difference = frame[first].to_numpy() - frame[second].to_numpy()
            entry: dict[str, Any] = {"mean": float(np.mean(difference))}
            if budget == "all":
                entry["bootstrap"] = interval(
                    difference[0], results[first]["bootstrap"][set_name][f"{arm}:all"],
                    results[second]["bootstrap"][set_name][f"{arm}:all"])
            else:
                entry["draws"] = paired_summary(frame[first].to_numpy(), frame[second].to_numpy())
                entry["reading"] = reading(first, second, entry["draws"])
            pairs[f"{first}_minus_{second}"] = entry
        encoder_contrasts.setdefault(set_name, {}).setdefault(arm, {})[budget] = pairs
    for set_name in summaries:
        for name in ENCODERS:
            by_budget = {}
            for budget in (*budgets, "all"):
                pairs = ARM_CONTRASTS if budget != "all" else (("pooled", "ptbxl"),)
                entries = {}
                for first, second in pairs:
                    first_values = wide["auroc"].loc[(set_name, first, budget), name].to_numpy()
                    second_values = wide["auroc"].loc[(set_name, second, budget), name].to_numpy()
                    bounds = interval(np.mean(first_values) - np.mean(second_values),
                                      results[name]["bootstrap"][set_name][f"{first}:{budget}"],
                                      results[name]["bootstrap"][set_name][f"{second}:{budget}"])
                    entry: dict[str, Any] = {"bootstrap": bounds, "side": interval_side(bounds),
                                             "negligible": abs(bounds["difference"]) < NEGLIGIBLE_AUROC}
                    if budget != "all":
                        entry["draws"] = paired_summary(first_values, second_values)
                        entry["verdict"] = verdict(first, second, entry["draws"], bounds)
                    entries[f"{first}_minus_{second}"] = entry
                by_budget[budget] = entries
            arm_contrasts.setdefault(set_name, {})[name] = by_budget
    primary = {budget: arm_contrasts["sph"][PRIMARY_ENCODER][budget]["pooled_minus_ptbxl"]
               for budget in READ_BUDGETS}
    ranking = {set_name: {budget: {arm: {pair: values["reading"]
                                         for pair, values in encoder_contrasts[set_name][arm][budget].items()}
                                   for arm in ("ptbxl", "pooled")} for budget in READ_BUDGETS}
               for set_name in RANKING_SETS}
    changes = any(by_arm["ptbxl"] != by_arm["pooled"] for by_budget in ranking.values()
                  for by_arm in by_budget.values())
    efficiency = {}
    for name in ENCODERS:
        target = summaries["sph"]["ptbxl"]["all"][name]["auroc"]
        efficiency[name] = {"ptbxl_all_auroc": target, **{arm: smallest_budget(
            {int(budget): summaries["sph"][arm][budget][name]["auroc"]["mean"] for budget in budgets}, target)
            for arm in ARMS}}
    return {"summaries": summaries, "arm_contrasts": arm_contrasts, "encoder_contrasts": encoder_contrasts,
            "primary": primary,
            "primary_verdicts": {budget: value["verdict"] for budget, value in primary.items()},
            "decision": decision([value["verdict"] for value in primary.values()]),
            "ranking": ranking, "ranking_changes": changes, "sph_label_efficiency": efficiency}


def key(set_name: str) -> str:
    """
    Array name of an evaluation set in the saved npz.

    Parameters
    ----------
    set_name : str
        Evaluation set, such as ``family:georgia``.

    Returns
    -------
    str
        The name with ``:`` replaced by ``_``.
    """
    return set_name.replace(":", "_")


def main() -> None:
    """Fit every arm's label draws once and score SPH, PTB-XL development and the Challenge families."""
    if (OUTPUT / "result.json").exists():
        raise FileExistsError("Experiment 025b v1 has already run")
    started = time.monotonic()
    identity = {"experiment025": identity025(), "priors": prior_receipts(), "experiment020": prior_identity(),
                "experiment022": prior022_identity()}
    prior_result = json.loads((PRIOR022 / "result.json").read_text())
    caches, cache_hashes = open_caches()
    if cache_hashes != prior_result["identity"]["feature_caches"]:
        raise ValueError("Feature caches differ from Experiment 022")
    feature_metadata, identity["challenge_features"] = feature_identity()
    prior022b = json.loads((PRIOR022B / "result.json").read_text())
    if identity["challenge_features"] != prior022b["identity"]["challenge_features"]:
        raise ValueError("Challenge features differ from Experiment 022b")
    identity["challenge_split"] = {"rows.csv": sha256_file(SPLITS / "rows.csv"),
                                   "metadata.json": sha256_file(SPLITS / "metadata.json"),
                                   "ningbo_rows.csv": sha256_file(NINGBO / "rows.csv")}

    groups = cohorts(ptb_table())
    pool, evaluation, counts025 = select_rows(groups["train"], groups["development"])
    reference = load_features(groups["train"], groups["development"], pool, evaluation)
    development_x, development = development_features(groups, caches, evaluation, reference)
    sph, _, identity["sph_manifest"] = sph_rows(prior_result)
    sph_x = sph_features(sph, prior_result)
    train, calibration, challenge_x, train_rows, calibration_rows = challenge_rows()

    families, sources = stacked(len(pool), train)
    members = arm_members(families, sources)
    y = np.concatenate([pool["standard"].to_numpy(dtype=np.int64), train["primary"].to_numpy(dtype=np.int64)])
    units = np.concatenate([pool["patient_id"].to_numpy(dtype=str),
                            challenge_units(train["source"].to_numpy(dtype=str),
                                            train["record"].to_numpy(dtype=str))])
    ids = np.concatenate([pool.index.to_numpy(dtype=str), units[len(pool):]])
    sets = evaluation_sets(sph, development, calibration)
    counts = check_counts(pool, train, y, units, sets)
    plans = plans_for(pool, units, y, members)
    plan_rows = plan_table(plans, ids, y, units, families)
    print(json.dumps({"stage": "counts", "fits_per_encoder": len(plans), **counts}), flush=True)

    with ProcessPoolExecutor(max_workers=len(ENCODERS)) as executor:
        futures = {}
        for name in ENCODERS:
            x = np.concatenate([reference[name][0], challenge_x[name][train_rows]])
            inputs = {"sph": sph_x[name], **development_x[name],
                      "calibration": challenge_x[name][calibration_rows]}
            futures[name] = executor.submit(evaluate_encoder, name, x, y, plans, inputs, sets)
        results = {name: future.result() for name, future in futures.items()}
    table = pd.DataFrame([row for output in results.values() for row in output["rows"]])
    reproduction = reproduce_025(table, plan_rows, {name: output["final"]["development_original"]["ptbxl"]
                                                    for name, output in results.items()}, evaluation)
    print(json.dumps({"stage": "reproduced_025", **reproduction}), flush=True)
    analysis = analyse(table, results)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    table.merge(plan_rows, on=["arm", "budget", "draw"], validate="many_to_one").to_csv(
        OUTPUT / "draws.csv", index=False)
    record_ids = {"sph": sph["ecg_id"].to_numpy(dtype=str),
                  **{key: rows.index.to_numpy(dtype=str) for key, rows in development.items()},
                  **{f"family:{name}": calibration.loc[sets[f"family:{name}"]["selection"], "record"]
                     .to_numpy(dtype=str) for name in FAMILIES}}
    write_npz_atomic(OUTPUT / "all_budget_predictions.npz",
                     **{f"{key(set_name)}_ids": values for set_name, values in record_ids.items()},
                     **{f"{key(set_name)}_labels": spec["y"] for set_name, spec in sets.items()},
                     **{f"{key(set_name)}_{name}_{arm}": values for name, output in results.items()
                        for set_name, by_arm in output["final"].items() for arm, values in by_arm.items()})
    mix_columns = [f"records_{family}" for family in ("ptbxl", *FAMILIES)]
    source_mix = plan_rows.groupby(["arm", "budget"])[mix_columns].mean().reset_index().to_dict("records")
    result = {
        "status": "complete",
        "identity": {**identity, "feature_caches": cache_hashes,
                     "challenge_features_identity": feature_metadata["identity"],
                     "sources": {name: sha256_file(ROOT / name) for name in SOURCES}},
        "counts": counts, "counts_025": counts025, "encoders": list(ENCODERS), "arms": list(ARMS),
        "budgets": list(BUDGETS), "draws": DRAWS, "draw_seed": DRAW_SEED, "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "invalid_bootstrap_draws": {name: output["invalid"] for name, output in results.items()},
        "fit_iterations": {name: output["iterations"] for name, output in results.items()},
        "encoder_seconds": {name: output["seconds"] for name, output in results.items()},
        "source_mix": source_mix,
        "reproduction_025": reproduction, **analysis,
        "outputs_sha256": {name: sha256_file(OUTPUT / name)
                           for name in ("draws.csv", "all_budget_predictions.npz")},
        "challenge_test_read": False, "ptbxl_calibration_read": False, "ptbxl_test_read": False,
        "runtime_seconds": time.monotonic() - started,
    }
    write_json_atomic(OUTPUT / "result.json", result)
    print(json.dumps({"stage": "complete", "runtime_seconds": result["runtime_seconds"],
                      "decision": result["decision"], "primary_verdicts": result["primary_verdicts"],
                      "ranking_changes": result["ranking_changes"]}), flush=True)


if __name__ == "__main__":
    main()
