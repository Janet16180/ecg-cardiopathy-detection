"""Experiment 022b: the 022 readout fitted on PTB-XL plus the Challenge training groups, read on SPH."""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from ecg_experiment.external_readout import Head, prior_cpc_features, prior_identity
from ecg_experiment.files import sha256_file, write_json_atomic, write_npz_atomic
from ecg_experiment.full_development import cohorts, predict, ptb_table
from ecg_experiment.multisource_calibration import (
    bootstrap_rates,
    family_weights,
    fit_arm,
    summarize,
    verdict,
    weighted_rates,
)
from ecg_experiment.multisource_readout import (
    ARMS,
    CHALLENGE_FAMILIES,
    WEIGHTED_ARMS,
    arm_members,
    fit_readout,
    probability_quality,
)
from ecg_experiment.normal_manifold import bootstrap_metrics, contrast, metrics, patient_resamples
from ecg_experiment.screening_threshold import calibrate, head_logits, operating_point
from scripts.experiments.run_calibrated_threshold027 import (
    HEADS,
    calibration_rows,
    prior022_identity,
    sph_rows,
)
from scripts.experiments.run_multisource_calibration027b import feature_identity, prior027_identity
from scripts.experiments.run_multisource_manifold026b import (
    challenge_features,
    interval_side,
    quality_reasons,
    split_table,
)
from scripts.experiments.run_sph_external022 import open_caches, ptb_encoder_features

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/experiment022b_multisource_readout_v1"
PRIOR022 = ROOT / "outputs/experiment022_sph_external_v3"
PRIOR027 = ROOT / "outputs/experiment027_calibrated_threshold_v1"
SPLITS = ROOT / "data/processed/challenge_splits_v1"
NINGBO = ROOT / "data/processed/ningbo_clean_v1"
ENCODERS = ("xecg", "jepa", "cpc")
PRIMARY_ENCODER = "xecg"
SEED = 31031
BOOTSTRAP_DRAWS = 2000
REPRODUCTION_TOLERANCE = 1e-9
NEGLIGIBLE_AUROC = 0.005
EXPECTED_TRAINING = {"ptbxl": (17083, 9840), "ningbo": (9952, 7371), "chapman_shaoxing": (3050, 2231),
                     "georgia": (5136, 4113), "cpsc_2018": (2600, 2049), "cpsc_2018_extra": (1756, 1756)}
EXPECTED_EXCLUDED = {"ningbo": 286, "chapman_shaoxing": 4, "georgia": 9, "cpsc_2018": 13,
                     "cpsc_2018_extra": 18}
EXPECTED_ARM_TRAINING = {"ptbxl": 17083, "pooled": 39577, "balanced": 39577, "ptbxl_chapman_ningbo": 30085,
                         "ptbxl_ningbo": 27035, "loso_chapman_ningbo": 26575, "loso_georgia": 34441,
                         "loso_cpsc": 35221}
EXPECTED_ARM_CALIBRATION = {"ptbxl": 564, "pooled": 8176, "balanced": 8176, "ptbxl_chapman_ningbo": 4996,
                            "ptbxl_ningbo": 3978, "loso_chapman_ningbo": 3744, "loso_georgia": 6458,
                            "loso_cpsc": 6714}
EXPECTED_SETS = {"sph": (21008, 7190), "development_full": (1572, 884), "development_original": (1306, 843),
                 "development_added": (266, 41), "family:chapman_ningbo": (4432, 3254),
                 "family:chapman_ningbo_without_zero_leads": (4362, 3207), "family:georgia": (1718, 1372),
                 "family:cpsc": (1462, 1279)}
SOURCES = (
    "ecg_experiment/multisource_readout.py", "ecg_experiment/multisource_calibration.py",
    "ecg_experiment/screening_threshold.py", "ecg_experiment/evaluation.py",
    "ecg_experiment/normal_manifold.py", "ecg_experiment/external_readout.py",
    "ecg_experiment/full_development.py", "ecg_experiment/ecg_quality.py",
    "ecg_experiment/challenge_features.py", "ecg_experiment/challenge_splits.py",
    "scripts/experiments/run_multisource_readout022b.py", "scripts/experiments/run_sph_external022.py",
    "scripts/experiments/run_calibrated_threshold027.py",
    "scripts/experiments/run_multisource_calibration027b.py",
    "scripts/experiments/run_multisource_manifold026b.py", "pyproject.toml", "uv.lock",
    "docs/experiment-022b-multisource-readout.md",
)


def sph_features(sph: pd.DataFrame, prior_result: dict[str, Any]) -> dict[str, np.ndarray]:
    """
    Experiment 022's saved SPH features of the scored rows, in their order.

    Parameters
    ----------
    sph : pd.DataFrame
        Output of 027's ``sph_rows``.
    prior_result : dict[str, Any]
        Experiment 022 ``result.json``.

    Returns
    -------
    dict[str, np.ndarray]
        Features per encoder.

    Raises
    ------
    ValueError
        If the feature file differs from 022's receipt or lacks a scored row.
    """
    if sha256_file(PRIOR022 / "features.npz") != prior_result["outputs_sha256"]["features.npz"]:
        raise ValueError("Experiment 022 SPH features differ from its receipt")
    with np.load(PRIOR022 / "features.npz") as saved:
        positions = pd.Index(saved["ecg_ids"]).get_indexer(sph["ecg_id"].to_numpy(dtype=str))
        if (positions < 0).any():
            raise ValueError("Experiment 022 SPH features lack a scored row")
        return {name: saved[name][positions] for name in ENCODERS}


def ptb_inputs(groups: dict[str, pd.DataFrame], calibration: pd.DataFrame, caches: Any
               ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray], pd.DataFrame]:
    """
    PTB-XL training, development and calibration features in 022's and 027's orders.

    Parameters
    ----------
    groups : dict[str, pd.DataFrame]
        Output of ``full_development.cohorts``.
    calibration : pd.DataFrame
        Output of 027's ``calibration_rows``.
    caches : Any
        Output of 022's ``open_caches``.

    Returns
    -------
    tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray], pd.DataFrame]
        Features per encoder keyed by ``train``, ``development`` and ``calibration`` (labeled rows only),
        the training and development labels, and the labeled development rows.

    Raises
    ------
    ValueError
        If 027's calibration features differ from the calibration rows.
    """
    with np.load(PRIOR022 / "ptb_features.npz") as saved:
        extracted = {name: saved[name] for name in saved.files}
    encoded = ptb_encoder_features(groups, caches, extracted)
    cpc = prior_cpc_features()
    by_encoder = {"cpc": cpc, "jepa": encoded["jepa"], "xecg": encoded["xecg"]}
    train, development = groups["train"], groups["development"]
    standard = train["standard"].notna().to_numpy()
    labeled = development["standard"].notna().to_numpy()
    with np.load(PRIOR027 / "calibration_features.npz") as saved:
        if not np.array_equal(saved["ecg_ids"], calibration["ecg_id"].to_numpy(dtype=np.int64)):
            raise ValueError("027 calibration features differ from the calibration rows")
        calibration_x = {name: saved[name] for name in ENCODERS}
    features = {name: {"train": by_encoder[name]["train"][standard],
                       "development": by_encoder[name]["development"][labeled],
                       "calibration": calibration_x[name]}
                for name in ENCODERS}
    labels = {"train": train.loc[standard, "standard"].to_numpy(dtype=np.int64),
              "development": development.loc[labeled, "standard"].to_numpy(dtype=np.int64)}
    return features, labels, development[labeled]


def stacked(ptb_count: int, part: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Family and source of PTB-XL rows followed by Challenge rows.

    Parameters
    ----------
    ptb_count : int
        Number of PTB-XL rows, which come first.
    part : pd.DataFrame
        Challenge rows with ``family`` and ``source``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Families and sources.
    """
    families = np.concatenate([np.full(ptb_count, "ptbxl"), part["family"].to_numpy(dtype=str)])
    sources = np.concatenate([np.full(ptb_count, "ptbxl"), part["source"].to_numpy(dtype=str)])
    return families, sources


def check_counts(train: pd.DataFrame, ptb_y: np.ndarray, training: dict[str, np.ndarray],
                 calibration: dict[str, np.ndarray], sets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Require the training, calibration and evaluation counts of the protocol.

    Parameters
    ----------
    train : pd.DataFrame
        Challenge training rows with ``reasons``.
    ptb_y : np.ndarray
        PTB-XL training labels.
    training, calibration : dict[str, np.ndarray]
        Arm membership of the stacked training and calibration rows.
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
    kept = train[train["reasons"] == ""]
    found = {
        "training": {"ptbxl": (len(ptb_y), int(ptb_y.sum())),
                     **{source: (len(part), int(part["primary"].sum()))
                        for source, part in kept.groupby("source")}},
        "excluded": {source: int((part["reasons"] != "").sum()) for source, part in train.groupby("source")},
        "arm_training": {arm: int(selected.sum()) for arm, selected in training.items()},
        "arm_calibration": {arm: int(selected.sum()) for arm, selected in calibration.items()},
        "sets": {name: (len(spec["y"]), int(spec["y"].sum())) for name, spec in sets.items()},
    }
    expected = {"training": EXPECTED_TRAINING, "excluded": EXPECTED_EXCLUDED,
                "arm_training": EXPECTED_ARM_TRAINING, "arm_calibration": EXPECTED_ARM_CALIBRATION,
                "sets": EXPECTED_SETS}
    if found != expected:
        raise ValueError(f"Counts differ from the protocol: {found}")
    return found


def evaluation_sets(sph: pd.DataFrame, development: pd.DataFrame, challenge: pd.DataFrame,
                    ptb_calibration_count: int) -> dict[str, dict[str, Any]]:
    """
    Labels, bootstrap units, contrasts and row selections of every evaluation set.

    Parameters
    ----------
    sph : pd.DataFrame
        SPH rows with ``primary`` and ``patient_id``.
    development : pd.DataFrame
        Labeled PTB-XL development rows with ``standard``, ``original`` and ``patient_id``.
    challenge : pd.DataFrame
        Challenge calibration rows, which follow the PTB-XL calibration rows in the stacked set.
    ptb_calibration_count : int
        Number of PTB-XL calibration rows.

    Returns
    -------
    dict[str, dict[str, Any]]
        Per set: the scored set it reads (``scores_of``), its ``selection`` within it, ``y``, ``units`` and
        the contrast ``pairs`` as (first, second).
    """
    versus_reference = [(arm, "ptbxl") for arm in ARMS[1:]]
    sets = {"sph": {"scores_of": "sph", "selection": np.ones(len(sph), dtype=bool),
                    "y": sph["primary"].to_numpy(dtype=np.int64),
                    "units": sph["patient_id"].to_numpy(dtype=str),
                    "pairs": [*versus_reference, ("pooled", "loso_chapman_ningbo")]}}
    original = development["original"].to_numpy(dtype=bool)
    for name, selected in (("full", np.ones(len(development), dtype=bool)), ("original", original),
                           ("added", ~original)):
        sets[f"development_{name}"] = {
            "scores_of": "development", "selection": selected,
            "y": development.loc[selected, "standard"].to_numpy(dtype=np.int64),
            "units": development.loc[selected, "patient_id"].to_numpy(dtype=str), "pairs": versus_reference}
    family = challenge["family"].to_numpy()
    nonzero = ~challenge["zero_lead"].to_numpy(dtype=bool)
    choices = {name: (family == name, name) for name in CHALLENGE_FAMILIES}
    choices["chapman_ningbo_without_zero_leads"] = ((family == "chapman_ningbo") & nonzero, "chapman_ningbo")
    for name, (selected, held_out) in choices.items():
        within = np.concatenate([np.zeros(ptb_calibration_count, dtype=bool), selected])
        sets[f"family:{name}"] = {
            "scores_of": "calibration", "selection": within, "family": held_out,
            "y": challenge.loc[selected, "primary"].to_numpy(dtype=np.int64),
            "units": challenge.loc[selected, "record"].to_numpy(dtype=str),
            "pairs": [*versus_reference, ("pooled", f"loso_{held_out}")]}
    return sets


def evaluate_encoder(name: str, x: np.ndarray, y: np.ndarray, members: dict[str, np.ndarray],
                     weights: np.ndarray, inputs: dict[str, np.ndarray], sets: dict[str, dict[str, Any]]
                     ) -> dict[str, Any]:
    """
    Fit every arm's readout on one encoder and rank every evaluation set with paired bootstrap contrasts.

    Parameters
    ----------
    name : str
        Encoder name.
    x, y : np.ndarray
        Stacked training features and labels (PTB-XL first).
    members : dict[str, np.ndarray]
        Training rows of each arm.
    weights : np.ndarray
        Family weights of the stacked training rows, used by ``WEIGHTED_ARMS``.
    inputs : dict[str, np.ndarray]
        Features of each scored set: ``sph``, ``development`` and ``calibration``.
    sets : dict[str, dict[str, Any]]
        Output of ``evaluation_sets``.

    Returns
    -------
    dict[str, Any]
        Probabilities per scored set and arm, fit diagnostics, and each set's metrics and contrasts.
    """
    started = time.monotonic()
    with threadpool_limits(limits=1):
        heads: dict[str, Head] = {}
        fits = {}
        for arm, selected in members.items():
            began = time.monotonic()
            arm_weights = weights[selected] if arm in WEIGHTED_ARMS else None
            heads[arm] = fit_readout(x[selected], y[selected], arm_weights)
            model: LogisticRegression = heads[arm][1]
            fits[arm] = {"seconds": time.monotonic() - began, "iterations": int(model.n_iter_[0])}
            print(json.dumps({"stage": f"{name}:fit:{arm}", **fits[arm]}), flush=True)
        scores = {set_name: {arm: predict(head, values) for arm, head in heads.items()}
                  for set_name, values in inputs.items()}
        evaluations = {}
        for set_name, spec in sets.items():
            arm_scores = {arm: values[spec["selection"]] for arm, values in scores[spec["scores_of"]].items()}
            observed = {arm: metrics(spec["y"], values) for arm, values in arm_scores.items()}
            resamples, invalid = patient_resamples(spec["units"], spec["y"], BOOTSTRAP_DRAWS, SEED)
            resampled = bootstrap_metrics(spec["y"], arm_scores, resamples)
            evaluations[set_name] = {
                "records": len(spec["y"]), "positives": int(spec["y"].sum()),
                "units": len(np.unique(spec["units"])), "metrics": observed,
                "raw_probability": {arm: probability_quality(spec["y"], values)
                                    for arm, values in arm_scores.items()},
                "contrasts": {f"{first}_minus_{second}": contrast(observed, resampled, first, second)
                              for first, second in spec["pairs"]},
                "invalid_draws": invalid}
            print(json.dumps({"stage": f"{name}:{set_name}", "auroc": {
                arm: round(value["auroc"], 4) for arm, value in observed.items()}}), flush=True)
    return {"scores": scores, "fits": fits, "evaluations": evaluations, "seconds": time.monotonic() - started}


def reproduce_022(results: dict[str, Any], development_mask: np.ndarray,
                  saved_sph: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """
    Require the ``ptbxl`` arm to reproduce 022's SPH and development probabilities and SPH AUROC.

    Parameters
    ----------
    results : dict[str, Any]
        Output of ``evaluate_encoder`` per encoder.
    development_mask : np.ndarray
        Labeled rows among 022's saved development rows.
    saved_sph : dict[str, np.ndarray]
        022's SPH probabilities per head name, from 027's ``sph_rows``.

    Returns
    -------
    dict[str, dict[str, float]]
        Largest absolute probability differences and the SPH AUROC difference per encoder.

    Raises
    ------
    ValueError
        If a probability differs by more than ``REPRODUCTION_TOLERANCE`` or an SPH AUROC differs.
    """
    prior = json.loads((PRIOR022 / "result.json").read_text())["by_label"]["primary"]
    differences = {}
    with np.load(PRIOR022 / "development_predictions.npz") as saved:
        for name, output in results.items():
            head = HEADS[name]
            auroc = output["evaluations"]["sph"]["metrics"]["ptbxl"]["auroc"]
            differences[name] = {
                "sph": float(np.abs(output["scores"]["sph"]["ptbxl"] - saved_sph[head]).max()),
                "development": float(np.abs(output["scores"]["development"]["ptbxl"]
                                            - saved[head][development_mask]).max()),
                "sph_auroc": auroc - prior[head]["auroc"]}
    worst = max(max(value["sph"], value["development"]) for value in differences.values())
    if worst > REPRODUCTION_TOLERANCE or any(value["sph_auroc"] != 0 for value in differences.values()):
        raise ValueError(f"The ptbxl arm does not reproduce Experiment 022: {differences}")
    return differences


def fit_operating_points(calibration_scores: dict[str, dict[str, np.ndarray]], y: np.ndarray,
                         families: np.ndarray, members: dict[str, np.ndarray]
                         ) -> tuple[dict[str, dict[str, LogisticRegression]], dict[str, dict[str, float]]]:
    """
    Fit each arm's Platt mapping and 95%-sensitivity threshold on its own calibration ECGs.

    Parameters
    ----------
    calibration_scores : dict[str, dict[str, np.ndarray]]
        Stacked calibration probabilities keyed by encoder, then arm.
    y, families : np.ndarray
        Labels and families of the stacked calibration ECGs.
    members : dict[str, np.ndarray]
        Calibration ECGs of each arm.

    Returns
    -------
    tuple[dict[str, dict[str, LogisticRegression]], dict[str, dict[str, float]]]
        Calibrators and thresholds keyed by arm, then encoder.
    """
    calibrators, thresholds = {}, {}
    for arm, selected in members.items():
        weights = family_weights(families[selected]) if arm in WEIGHTED_ARMS else None
        calibrators[arm], thresholds[arm] = {}, {}
        for name, scores in calibration_scores.items():
            fitted = fit_arm(head_logits(scores[arm][selected]), y[selected], weights)
            calibrators[arm][name], thresholds[arm][name] = fitted
    return calibrators, thresholds


def reproduce_027(calibrators: dict[str, LogisticRegression], thresholds: dict[str, float],
                  sph_calibrated: dict[str, np.ndarray]) -> dict[str, float]:
    """
    Require the ``ptbxl`` arm's operating point to reproduce 027's.

    Parameters
    ----------
    calibrators, thresholds : dict
        The ``ptbxl`` arm, keyed by encoder.
    sph_calibrated : dict[str, np.ndarray]
        The ``ptbxl`` arm's calibrated SPH probabilities per encoder.

    Returns
    -------
    dict[str, float]
        Largest absolute difference per encoder over the Platt coefficients, threshold and SPH probabilities.

    Raises
    ------
    ValueError
        If a difference exceeds ``REPRODUCTION_TOLERANCE``.
    """
    result027 = json.loads((PRIOR027 / "result.json").read_text())
    differences = {}
    with np.load(PRIOR027 / "calibrated_predictions.npz") as saved:
        for name, model in calibrators.items():
            head = HEADS[name]
            platt = result027["platt"][head]
            differences[name] = max(abs(model.coef_[0, 0] - platt["slope"]),
                                    abs(model.intercept_[0] - platt["intercept"]),
                                    abs(thresholds[name] - result027["thresholds"][head]),
                                    float(np.abs(sph_calibrated[name] - saved[f"sph_{head}"]).max()))
    if max(differences.values()) > REPRODUCTION_TOLERANCE:
        raise ValueError(f"The ptbxl arm does not reproduce Experiment 027: {differences}")
    return differences


def operating_set(units: np.ndarray, y: np.ndarray, scores: dict[str, dict[str, np.ndarray]],
                  calibrators: dict[str, dict[str, LogisticRegression]],
                  thresholds: dict[str, dict[str, float]],
                  ) -> tuple[dict[str, Any], np.ndarray, list[tuple[str, str]]]:
    """
    Operating points of every arm and encoder on one set, with paired bootstrap rates.

    Parameters
    ----------
    units : np.ndarray
        Bootstrap unit of each ECG.
    y : np.ndarray
        Labels.
    scores : dict[str, dict[str, np.ndarray]]
        Raw probabilities of the set keyed by encoder, then arm.
    calibrators, thresholds : dict
        Output of ``fit_operating_points``.

    Returns
    -------
    tuple[dict[str, Any], np.ndarray, list[tuple[str, str]]]
        ``operating_points[arm][encoder]`` and ``bootstrap``, the calibrated matrix and its columns.
    """
    columns = [(arm, name) for arm in ARMS for name in ENCODERS]
    matrix = np.column_stack([calibrate(calibrators[arm][name], head_logits(scores[name][arm]))
                              for arm, name in columns])
    cutoffs = np.array([thresholds[arm][name] for arm, name in columns])
    points = {}
    for index, (arm, name) in enumerate(columns):
        points.setdefault(arm, {})[name] = operating_point(y, matrix[:, index], cutoffs[index])
    observed = weighted_rates(np.ones(len(y)), y, matrix, cutoffs)
    drawn = bootstrap_rates(units, y, matrix, cutoffs, BOOTSTRAP_DRAWS, SEED)
    return ({"operating_points": points, "bootstrap": summarize(columns, observed, drawn, "ptbxl")},
            matrix, columns)


def reading(results: dict[str, Any], sph_operating: dict[str, Any]) -> dict[str, Any]:
    """
    Apply the prespecified decision rule, the Ningbo answer and the operating-point reading.

    Parameters
    ----------
    results : dict[str, Any]
        Output of ``evaluate_encoder`` per encoder.
    sph_operating : dict[str, Any]
        SPH output of ``operating_set``.

    Returns
    -------
    dict[str, Any]
        Primary contrast, side and decision; the Ningbo contrasts and answer; deviation verdicts and
        transfer per arm and encoder; and the side of every AUROC and AP contrast.
    """
    sph = results[PRIMARY_ENCODER]["evaluations"]["sph"]["contrasts"]
    primary = sph["pooled_minus_ptbxl"]["auroc"]
    side = interval_side(primary)
    decision = {"above 0": "adopt_pooled", "below 0": "pooling_hurts"}.get(side, "not_distinguished")
    ningbo = {"ptbxl_chapman_ningbo_minus_ptbxl": sph["ptbxl_chapman_ningbo_minus_ptbxl"]["auroc"],
              "pooled_minus_loso_chapman_ningbo": sph["pooled_minus_loso_chapman_ningbo"]["auroc"]}
    ningbo_sides = [interval_side(values) for values in ningbo.values()]
    answer = "not_established"
    if ningbo_sides == ["above 0", "above 0"]:
        answer = "helps"
    elif ningbo_sides == ["below 0", "below 0"]:
        answer = "hurts"
    differences = sph_operating["bootstrap"]["differences"]
    deviation = {arm: {name: verdict(values["deviation"]) for name, values in heads.items()}
                 for arm, heads in differences.items()}
    transfers = {arm: {name: rates["sensitivity"]["ci_low"] <= 0.95 <= rates["sensitivity"]["ci_high"]
                       for name, rates in heads.items()}
                 for arm, heads in sph_operating["bootstrap"]["rates"].items()}
    sides = {name: {set_name: {pair: {metric: interval_side(values) for metric, values in pair_values.items()}
                               for pair, pair_values in evaluation["contrasts"].items()}
                    for set_name, evaluation in output["evaluations"].items()}
             for name, output in results.items()}
    return {"primary_contrast": primary, "primary_side": side, "decision": decision,
            "primary_negligible_in_practice": abs(primary["difference"]) < NEGLIGIBLE_AUROC,
            "ningbo_contrasts": ningbo, "ningbo_sides": ningbo_sides, "ningbo_answer": answer,
            "deviation_verdicts": deviation, "threshold_transfers_to_sph": transfers, "interval_sides": sides}


def main() -> None:
    """Fit every readout arm once and score SPH, PTB-XL development and the held-out Challenge families."""
    if (OUTPUT / "result.json").exists():
        raise FileExistsError("Experiment 022b v1 has already run")
    started = time.monotonic()
    feature_metadata, feature_hashes = feature_identity()
    _, hashes027 = prior027_identity()
    prior_result = json.loads((PRIOR022 / "result.json").read_text())
    hashes022 = prior022_identity()
    caches, cache_hashes = open_caches()
    if cache_hashes != prior_result["identity"]["feature_caches"]:
        raise ValueError("Feature caches differ from Experiment 022")
    table = ptb_table()
    groups = cohorts(table)
    calibration, references_sha256 = calibration_rows(table, groups)
    sph, saved_sph, sph_hashes = sph_rows(prior_result)
    sph_x = sph_features(sph, prior_result)
    ptb_x, ptb_y, development = ptb_inputs(groups, calibration, caches)
    development_mask = groups["development"]["standard"].notna().to_numpy()

    rows, challenge_x = challenge_features(split_table())
    in_train = (rows["split"] == "train").to_numpy()
    in_calibration = (rows["split"] == "calibration").to_numpy()
    train = rows[in_train].copy()
    began = time.monotonic()
    train["reasons"] = quality_reasons(train)
    quality_seconds = time.monotonic() - began
    kept = (train["reasons"] == "").to_numpy()
    challenge = rows[in_calibration].reset_index(drop=True)

    train_families, train_sources = stacked(len(ptb_y["train"]), train[kept])
    training = arm_members(train_families, train_sources)
    y_train = np.concatenate([ptb_y["train"], train.loc[kept, "primary"].to_numpy(dtype=np.int64)])
    weights = family_weights(train_families)
    ptb_calibration_y = calibration["standard"].to_numpy(dtype=np.int64)
    calibration_families, calibration_sources = stacked(len(ptb_calibration_y), challenge)
    calibration_members = arm_members(calibration_families, calibration_sources)
    y_calibration = np.concatenate([ptb_calibration_y, challenge["primary"].to_numpy(dtype=np.int64)])
    sets = evaluation_sets(sph, development, challenge, len(ptb_calibration_y))
    counts = check_counts(train, ptb_y["train"], training, calibration_members, sets)
    print(json.dumps({"stage": "counts", "quality_seconds": quality_seconds, **counts}), flush=True)

    with ProcessPoolExecutor(max_workers=len(ENCODERS)) as executor:
        futures = {}
        for name in ENCODERS:
            x = np.concatenate([ptb_x[name]["train"], challenge_x[name][in_train][kept]])
            inputs = {"sph": sph_x[name], "development": ptb_x[name]["development"],
                      "calibration": np.concatenate([ptb_x[name]["calibration"],
                                                     challenge_x[name][in_calibration]])}
            futures[name] = executor.submit(evaluate_encoder, name, x, y_train, training, weights, inputs,
                                            sets)
        results = {name: future.result() for name, future in futures.items()}
    reproduction022 = reproduce_022(results, development_mask, saved_sph)

    with threadpool_limits(limits=1):
        calibrators, thresholds = fit_operating_points(
            {name: output["scores"]["calibration"] for name, output in results.items()},
            y_calibration, calibration_families, calibration_members)
        sph_operating, sph_matrix, columns = operating_set(
            sets["sph"]["units"], sets["sph"]["y"],
            {name: output["scores"]["sph"] for name, output in results.items()}, calibrators, thresholds)
        reference = {name: sph_matrix[:, columns.index(("ptbxl", name))] for name in ENCODERS}
        reproduction027 = reproduce_027(calibrators["ptbxl"], thresholds["ptbxl"], reference)
        family_operating = {}
        for set_name, spec in sets.items():
            if not set_name.startswith("family:"):
                continue
            selected = spec["selection"]
            family_operating[set_name], _, _ = operating_set(
                spec["units"], spec["y"],
                {name: {arm: values[selected] for arm, values in output["scores"]["calibration"].items()}
                 for name, output in results.items()}, calibrators, thresholds)
        in_sample = {arm: {name: operating_point(
            y_calibration[selected], calibrate(calibrators[arm][name], head_logits(
                results[name]["scores"]["calibration"][arm][selected])), thresholds[arm][name])
            for name in ENCODERS} for arm, selected in calibration_members.items()}

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_npz_atomic(
        OUTPUT / "predictions.npz",
        sph_ecg_ids=sph["ecg_id"].to_numpy(dtype=str), sph_patient_ids=sph["patient_id"].to_numpy(dtype=str),
        sph_labels=sets["sph"]["y"], development_record_ids=development.index.to_numpy(dtype=str),
        development_labels=sets["development_full"]["y"],
        calibration_records=np.concatenate([calibration["ecg_id"].to_numpy(dtype=str),
                                            challenge["record"].to_numpy(dtype=str)]),
        calibration_families=calibration_families, calibration_labels=y_calibration,
        sph_calibrated=sph_matrix,
        sph_calibrated_columns=np.array([f"{arm}:{name}" for arm, name in columns]),
        **{f"{set_name}_{name}_{arm}": values for name, output in results.items()
           for set_name, by_arm in output["scores"].items() for arm, values in by_arm.items()})
    train.assign(kept=kept).to_csv(OUTPUT / "training_rows.csv", index=False)
    platt = {arm: {name: {"slope": float(model.coef_[0, 0]), "intercept": float(model.intercept_[0])}
                   for name, model in models.items()}
             for arm, models in calibrators.items()}
    result = {
        "status": "complete",
        "identity": {
            "experiment022": hashes022,
            "experiment022_features_sha256": sha256_file(PRIOR022 / "features.npz"),
            "experiment020": prior_identity(), "experiment027": hashes027, "sph_manifest": sph_hashes,
            "heldout_references": references_sha256, "feature_caches": cache_hashes,
            "challenge_features": feature_hashes, "challenge_features_identity": feature_metadata["identity"],
            "challenge_split": {"rows.csv": sha256_file(SPLITS / "rows.csv"),
                                "metadata.json": sha256_file(SPLITS / "metadata.json")},
            "ningbo_rows_sha256": sha256_file(NINGBO / "rows.csv"),
            "sources": {name: sha256_file(ROOT / name) for name in SOURCES},
        },
        "counts": counts, "arms": list(ARMS), "encoders": list(ENCODERS), "seed": SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "ptbxl_arm_difference_from_022": reproduction022, "ptbxl_arm_difference_from_027": reproduction027,
        "ranking": {name: {key: value for key, value in output.items() if key != "scores"}
                    for name, output in results.items()},
        "platt": platt, "thresholds": thresholds, "sph_operating": sph_operating,
        "family_operating": family_operating, "in_sample_operating": in_sample,
        "reading": reading(results, sph_operating),
        "outputs_sha256": {name: sha256_file(OUTPUT / name)
                           for name in ("predictions.npz", "training_rows.csv")},
        "challenge_test_read": False, "ptbxl_calibration_as_test": False, "ptbxl_test_read": False,
        "quality_seconds": quality_seconds, "runtime_seconds": time.monotonic() - started,
    }
    write_json_atomic(OUTPUT / "result.json", result)
    print(json.dumps({"stage": "complete", "runtime_seconds": result["runtime_seconds"],
                      "decision": result["reading"]["decision"],
                      "ningbo_answer": result["reading"]["ningbo_answer"],
                      "primary": result["reading"]["primary_contrast"]}), flush=True)


if __name__ == "__main__":
    main()
