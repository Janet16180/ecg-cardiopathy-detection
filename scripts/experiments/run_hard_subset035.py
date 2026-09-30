"""Experiment 035: why the pooled PTB-XL plus Challenge readout loses on the PTB-XL hard added subset.

``--counts`` builds every arm's training rows, runs the training quality policy on the Challenge rows 022b
did not check, and stops before any readout is fitted. The default stage fits every arm of
``hard_subset.ARM_SPECS`` for the three frozen encoders, requires the arms 022b also fitted to reproduce its
probabilities, cross-fits the xECG arms on the PTB-XL training ECGs the project label dropped, and scores the
hard subset, ordinary PTB-XL development and SPH.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from ecg_experiment.external_readout import prior_cpc_features, prior_identity
from ecg_experiment.files import sha256_file, write_json_atomic, write_npz_atomic
from ecg_experiment.full_development import cohorts, predict, ptb_table
from ecg_experiment.hard_subset import (
    ARM_SPECS,
    CHALLENGE_FAMILIES,
    LABEL_RULES,
    arm_design,
    arm_verdict,
    patient_folds,
    recovery,
    rule_labels,
)
from ecg_experiment.intervals import paired_auroc_difference
from ecg_experiment.multisource_readout import fit_readout
from scripts.experiments.describe_hard_subset035 import KEPT_DUPLICATES, challenge_codes
from scripts.experiments.run_calibrated_threshold027 import prior022_identity, sph_rows
from scripts.experiments.run_multisource_calibration027b import feature_identity
from scripts.experiments.run_multisource_manifold026b import (
    FEATURES,
    NINGBO,
    SOURCES_ORDER,
    SPLIT_ROWS_SHA256,
    SPLITS,
    interval_side,
    quality_reasons,
)
from scripts.experiments.run_multisource_readout022b import OUTPUT as PRIOR022B
from scripts.experiments.run_multisource_readout022b import PRIOR022, sph_features
from scripts.experiments.run_sph_external022 import open_caches, ptb_encoder_features

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/experiment035_hard_subset_v1"
ENCODERS = ("xecg", "jepa", "cpc")
PRIMARY_ENCODER = "xecg"
SEED = 39039
DRAWS = 2000
FOLDS = 5
REPRODUCTION_TOLERANCE = 1e-9
SPH_TOLERANCE = 0.005
MINIMUM_RECOVERY = 0.5
ARMS = tuple(ARM_SPECS)
REUSED_022B = {"ptbxl": "ptbxl", "pooled": "pooled", "loso_chapman_ningbo": "loso_chapman_ningbo",
               "loso_georgia": "loso_georgia", "loso_cpsc": "loso_cpsc",
               "ptbxl_chapman_ningbo": "ptbxl_chapman_ningbo"}
CANDIDATES = tuple(arm for arm in ARMS if arm not in ("ptbxl", "pooled", "without_dropped"))
SETS = ("hard", "ordinary", "full", "sph")
HARD_NEGATIVE_CATEGORIES = (
    ("st_t_or_pr_form", {"STD_", "STE_", "INVT", "TAB_", "NT_", "LOWT", "LPR"}),
    ("sinus_variant", {"SBRAD", "STACH", "SARRH"}),
    ("ectopy", {"PVC", "PAC", "PRC(S)", "BIGU", "TRIGU", "SVARR"}),
    ("voltage_or_q", {"LVOLT", "VCLVH", "HVOLT", "QWAVE"}),
)
HARD_POSITIVE_CATEGORIES = (("irbbb", {"IRBBB"}), ("other_cd", {"IVCD", "1AVB", "LAFB", "LPFB"}))
EXPECTED = {
    "ptbxl_train": [17083, 9840], "ptbxl_dropped": [1724, 353],
    "development": {"hard": [266, 41], "ordinary": [1306, 843]}, "sph": [21008, 7190],
    "challenge_rows": {"022b": {"ningbo": 10238, "georgia": 5145, "chapman_shaoxing": 3054, "cpsc_2018": 2613,
                                "cpsc_2018_extra": 1774},
                       "035": {"ningbo": 8223, "chapman_shaoxing": 2192, "georgia": 638,
                               "cpsc_2018_extra": 1}},
    "challenge_excluded": {"022b": {"ningbo": 286, "cpsc_2018_extra": 18, "cpsc_2018": 13, "georgia": 9,
                                    "chapman_shaoxing": 4},
                           "035": {"ningbo": 607, "chapman_shaoxing": 3, "georgia": 1}},
}
EXPECTED_ARMS = {
    "ptbxl": [17083, 9840], "pooled": [39577, 27360], "loso_chapman_ningbo": [26575, 17758],
    "loso_georgia": [34441, 23247], "loso_cpsc": [35221, 23555], "ptbxl_chapman_ningbo": [30085, 19442],
    "ptbxl_georgia": [22219, 13953], "ptbxl_cpsc": [21439, 13645], "secondary_negative": [47889, 27360],
    "ptbxl_negative": [50020, 27360], "form_ignored": [35316, 23099], "ptbxl_rule": [47809, 23099],
    "source_indicator": [39577, 27360], "prevalence_matched": [39577, 27360], "ptbxl_half": [39577, 27360],
    "dropped_upweighted": [39577, 27360], "without_dropped": [37853, 27007],
}
SOURCES = (
    "ecg_experiment/hard_subset.py", "ecg_experiment/intervals.py", "ecg_experiment/multisource_readout.py",
    "ecg_experiment/external_readout.py", "ecg_experiment/full_development.py",
    "ecg_experiment/challenge_labels.py", "ecg_experiment/multisource_manifold.py",
    "ecg_experiment/ecg_quality.py", "scripts/experiments/run_hard_subset035.py",
    "scripts/experiments/describe_hard_subset035.py", "scripts/experiments/run_multisource_readout022b.py",
    "scripts/experiments/run_multisource_manifold026b.py", "scripts/experiments/run_sph_external022.py",
    "scripts/experiments/run_calibrated_threshold027.py", "pyproject.toml", "uv.lock",
    "docs/experiment-035-hard-subset.md",
)


def prior022b() -> dict[str, Any]:
    """
    022b's result, with its saved predictions and training rows checked against it.

    Returns
    -------
    dict[str, Any]
        022b ``result.json``.

    Raises
    ------
    ValueError
        If a saved 022b file differs from its receipt.
    """
    result = json.loads((PRIOR022B / "result.json").read_text())
    for name in ("predictions.npz", "training_rows.csv"):
        if sha256_file(PRIOR022B / name) != result["outputs_sha256"][name]:
            raise ValueError(f"022b {name} differs from its receipt")
    return result


def ptbxl_rows() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, np.ndarray]], dict[str, str]]:
    """
    PTB-XL training and development rows with a standard label and their features, in 022's order.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, np.ndarray]], dict[str, str]]
        Training rows, development rows, features per encoder keyed by ``train`` and ``development``, and
        the JEPA and xECG cache hashes.

    Raises
    ------
    ValueError
        If the caches differ from Experiment 022's.
    """
    prior_result = json.loads((PRIOR022 / "result.json").read_text())
    caches, cache_hashes = open_caches()
    if cache_hashes != prior_result["identity"]["feature_caches"]:
        raise ValueError("Feature caches differ from Experiment 022")
    groups = cohorts(ptb_table())
    with np.load(PRIOR022 / "ptb_features.npz") as saved:
        extracted = {name: saved[name] for name in saved.files}
    encoded = ptb_encoder_features(groups, caches, extracted)
    by_encoder = {"cpc": prior_cpc_features(), "jepa": encoded["jepa"], "xecg": encoded["xecg"]}
    train, development = groups["train"], groups["development"]
    standard = train["standard"].notna().to_numpy()
    labeled = development["standard"].notna().to_numpy()
    features = {name: {"train": values["train"][standard], "development": values["development"][labeled]}
                for name, values in by_encoder.items()}
    return train[standard], development[labeled], features, cache_hashes


def challenge_universe() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """
    Challenge training-group rows with features, a kept duplicate status and a label under some rule.

    Rows are in source then record order. The training quality policy is 022b's saved result for the rows
    022b checked and 026b's ``quality_reasons`` for the others.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, np.ndarray]]
        Rows with ``codes``, one label column per rule, ``reasons`` and ``checked_by`` (``022b`` or
        ``035``), and their features per encoder.

    Raises
    ------
    ValueError
        If the split differs from its receipt, a rule disagrees with the split's labels, or the rows with a
        primary label differ from 022b's training rows.
    """
    if sha256_file(SPLITS / "rows.csv") != SPLIT_ROWS_SHA256:
        raise ValueError("Challenge split differs from its receipt")
    codes = challenge_codes()
    rows = pd.read_csv(SPLITS / "rows.csv", dtype={"record": str})
    rows = rows[(rows["split"] == "train") & rows["duplicate_status"].isin(KEPT_DUPLICATES)]
    ningbo = pd.read_csv(NINGBO / "rows.csv", dtype={"record": str}, usecols=["record", "use_training"])
    rows = rows.merge(ningbo, on="record", how="left")
    parts, features = [], {name: [] for name in ENCODERS}
    for source in SOURCES_ORDER:
        with np.load(FEATURES / f"{source}.npz") as saved:
            records = pd.Index(saved["record"].astype(str))
            part = rows[(rows["source"] == source) & rows["record"].isin(records)].sort_values("record")
            positions = records.get_indexer(part["record"])
            part = part.assign(window_start=saved["window_start"][positions])
            part["codes"] = [codes[(source, record)] for record in part["record"]]
            labels = {rule: rule_labels(part["codes"], rule) for rule in LABEL_RULES}
            keep = np.any([~np.isnan(values) for values in labels.values()], axis=0)
            for name in ENCODERS:
                features[name].append(saved[name][positions][keep])
        for rule in ("primary", "secondary"):
            if not np.array_equal(labels[rule], part[rule].to_numpy(dtype=np.float64), equal_nan=True):
                raise ValueError(f"The {rule} rule differs from the split for {source}")
        parts.append(part.assign(**labels)[keep])
    universe = pd.concat(parts, ignore_index=True)
    saved_rows = pd.read_csv(PRIOR022B / "training_rows.csv", dtype={"record": str}, keep_default_na=False)
    with_primary = universe[universe["primary"].notna()]
    same = all(np.array_equal(with_primary[column].to_numpy(dtype=str),
                              saved_rows[column].to_numpy(dtype=str)) for column in ("source", "record"))
    if not same:
        raise ValueError("Challenge rows with a primary label differ from 022b's training rows")
    reasons = saved_rows.set_index(["source", "record"])["reasons"]
    key = pd.MultiIndex.from_frame(universe[["source", "record"]])
    universe["checked_by"] = np.where(key.isin(reasons.index), "022b", "035")
    universe["reasons"] = reasons.reindex(key).to_numpy()
    unchecked = universe["checked_by"] == "035"
    universe.loc[unchecked, "reasons"] = quality_reasons(universe[unchecked]).to_numpy()
    return universe, {name: np.concatenate(values) for name, values in features.items()}


def stacked_universe(train: pd.DataFrame, universe: pd.DataFrame
                     ) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Families, labels per rule, dropped flags, patients and quality selection of PTB-XL then Challenge rows.

    Parameters
    ----------
    train : pd.DataFrame
        PTB-XL training rows with a standard label.
    universe : pd.DataFrame
        Output of ``challenge_universe``.

    Returns
    -------
    tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]
        Family per row, label per rule, PTB-XL rows without a project label, PTB-XL patient IDs (empty text
        for Challenge rows), and the Challenge rows that pass the quality policy.
    """
    count = len(train)
    passing = (universe["reasons"] == "").to_numpy()
    part = universe[passing]
    families = np.concatenate([np.full(count, "ptbxl"), part["family"].to_numpy(dtype=str)])
    standard = train["standard"].to_numpy(dtype=np.float64)
    labels = {rule: np.concatenate([standard, part[rule].to_numpy(dtype=np.float64)]) for rule in LABEL_RULES}
    dropped = np.concatenate([train["target"].isna().to_numpy(), np.zeros(len(part), dtype=bool)])
    patients = np.concatenate([train["patient_id"].to_numpy(dtype=str), np.full(len(part), "")])
    return families, labels, dropped, patients, passing


def design_counts(families: np.ndarray, labels: dict[str, np.ndarray], dropped: np.ndarray
                  ) -> dict[str, dict[str, list[int]]]:
    """
    Training rows and positives of every arm, per family.

    Parameters
    ----------
    families, labels, dropped : np.ndarray
        Output of ``stacked_universe``.

    Returns
    -------
    dict[str, dict[str, list[int]]]
        ``[rows, positives]`` per family and in ``all``, per arm.
    """
    counts = {}
    for arm, spec in ARM_SPECS.items():
        design = arm_design(spec, families, labels, dropped)
        chosen = families[design["selected"]]
        counts[arm] = {family: [int(np.count_nonzero(chosen == family)),
                                int(design["y"][chosen == family].sum())]
                       for family in ("ptbxl", *CHALLENGE_FAMILIES)}
        counts[arm]["all"] = [len(design["y"]), int(design["y"].sum())]
    return counts


def fit_design(x: np.ndarray, design: dict[str, np.ndarray | None]) -> tuple[Any, float]:
    """
    Fit one arm's readout.

    Parameters
    ----------
    x : np.ndarray
        Features of the selected rows.
    design : dict[str, np.ndarray | None]
        Output of ``arm_design`` for the same rows.

    Returns
    -------
    tuple[Any, float]
        The fitted head and the fit's iteration count.
    """
    if design["indicators"] is not None:
        x = np.column_stack([x, design["indicators"]])
    head = fit_readout(x, design["y"], design["weights"])
    model: LogisticRegression = head[1]
    return head, int(model.n_iter_[0])


def score(head: Any, x: np.ndarray, indicator: bool) -> np.ndarray:
    """
    Probabilities of a head, scoring every ECG with all source indicators at 0 when the arm has them.

    Parameters
    ----------
    head : Any
        Output of ``fit_design``.
    x : np.ndarray
        Features to score.
    indicator : bool
        Whether the arm was fitted with source indicators.

    Returns
    -------
    np.ndarray
        One probability per row.
    """
    if indicator:
        x = np.column_stack([x, np.zeros((len(x), len(CHALLENGE_FAMILIES)))])
    return predict(head, x)


def set_summary(y: np.ndarray, patients: np.ndarray, scores: dict[str, np.ndarray]) -> dict[str, Any]:
    """
    AUROC and AP of every arm and each arm's paired bootstrap AUROC difference from ``pooled``.

    Parameters
    ----------
    y : np.ndarray
        Labels.
    patients : np.ndarray
        Patient ID of each ECG.
    scores : dict[str, np.ndarray]
        Probabilities per arm.

    Returns
    -------
    dict[str, Any]
        Counts, ``auroc`` and ``average_precision`` per arm, and ``minus_pooled`` per other arm.
    """
    return {
        "records": len(y), "positives": int(y.sum()), "patients": len(np.unique(patients)),
        "auroc": {arm: float(roc_auc_score(y, values)) for arm, values in scores.items()},
        "average_precision": {arm: float(average_precision_score(y, values))
                              for arm, values in scores.items()},
        "minus_pooled": {arm: paired_auroc_difference(patients, y, values, scores["pooled"], DRAWS, SEED)
                         for arm, values in scores.items() if arm != "pooled"},
    }


def evaluate_encoder(name: str, x: np.ndarray, families: np.ndarray, labels: dict[str, np.ndarray],
                     dropped: np.ndarray, inputs: dict[str, np.ndarray],
                     sets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Fit every arm on one encoder and score the evaluation sets.

    Parameters
    ----------
    name : str
        Encoder name.
    x : np.ndarray
        Stacked training features (PTB-XL, then the Challenge rows that pass the quality policy).
    families, labels, dropped : np.ndarray
        Output of ``stacked_universe``.
    inputs : dict[str, np.ndarray]
        Features of ``development`` (all 1,572 labeled ECGs) and ``sph``.
    sets : dict[str, dict[str, Any]]
        Output of ``evaluation_sets``.

    Returns
    -------
    dict[str, Any]
        Probabilities per scored input and arm, iterations per arm, and each set's summary.
    """
    started = time.monotonic()
    with threadpool_limits(limits=1):
        scores: dict[str, dict[str, np.ndarray]] = {key: {} for key in inputs}
        iterations = {}
        for arm, spec in ARM_SPECS.items():
            design = arm_design(spec, families, labels, dropped)
            head, iterations[arm] = fit_design(x[design["selected"]], design)
            for key, values in inputs.items():
                scores[key][arm] = score(head, values, design["indicators"] is not None)
            print(json.dumps({"stage": f"{name}:fit:{arm}", "iterations": iterations[arm]}), flush=True)
        summaries = {}
        for set_name, spec in sets.items():
            chosen = {arm: values[spec["selection"]] for arm, values in scores[spec["scores_of"]].items()}
            summaries[set_name] = set_summary(spec["y"], spec["patients"], chosen)
            print(json.dumps({"stage": f"{name}:{set_name}", "done": True}), flush=True)
    return {"scores": scores, "iterations": iterations, "sets": summaries,
            "seconds": time.monotonic() - started}


def crossfit_fold(fold: int, x: np.ndarray, families: np.ndarray, labels: dict[str, np.ndarray],
                  dropped: np.ndarray, folds: np.ndarray) -> dict[str, Any]:
    """
    Fit every arm without one fold's PTB-XL patients and score that fold's dropped training ECGs.

    Parameters
    ----------
    fold : int
        Held-out fold.
    x : np.ndarray
        Stacked training features of the primary encoder.
    families, labels, dropped : np.ndarray
        Output of ``stacked_universe``.
    folds : np.ndarray
        Fold of each PTB-XL row, -1 for Challenge rows.

    Returns
    -------
    dict[str, Any]
        Positions of the scored rows, probabilities per arm and iterations per arm.
    """
    held_out = folds == fold
    kept = np.flatnonzero(~held_out)
    scored = np.flatnonzero(held_out & dropped)
    sub_labels = {rule: values[kept] for rule, values in labels.items()}
    probabilities, iterations = {}, {}
    with threadpool_limits(limits=1):
        for arm, spec in ARM_SPECS.items():
            design = arm_design(spec, families[kept], sub_labels, dropped[kept])
            head, iterations[arm] = fit_design(x[kept[design["selected"]]], design)
            probabilities[arm] = score(head, x[scored], design["indicators"] is not None)
        print(json.dumps({"stage": f"crossfit:fold{fold}", "scored": len(scored)}), flush=True)
    return {"positions": scored, "scores": probabilities, "iterations": iterations}


def evaluation_sets(development: pd.DataFrame, sph: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """
    Labels, patients and row selections of the hard, ordinary, full development and SPH sets.

    Parameters
    ----------
    development : pd.DataFrame
        Labeled PTB-XL development rows with ``original``.
    sph : pd.DataFrame
        SPH rows with ``primary`` and ``patient_id``.

    Returns
    -------
    dict[str, dict[str, Any]]
        Per set: ``scores_of``, ``selection``, ``y`` and ``patients``.
    """
    original = development["original"].to_numpy(dtype=bool)
    choices = {"hard": ~original, "ordinary": original, "full": np.ones(len(development), dtype=bool)}
    sets = {name: {"scores_of": "development", "selection": selected,
                   "y": development.loc[selected, "standard"].to_numpy(dtype=np.int64),
                   "patients": development.loc[selected, "patient_id"].to_numpy(dtype=str)}
            for name, selected in choices.items()}
    sets["sph"] = {"scores_of": "sph", "selection": np.ones(len(sph), dtype=bool),
                   "y": sph["primary"].to_numpy(dtype=np.int64),
                   "patients": sph["patient_id"].to_numpy(dtype=str)}
    return sets


def reproduce_022b(results: dict[str, Any]) -> dict[str, dict[str, float]]:
    """
    Require the arms 022b also fitted to reproduce its development and SPH probabilities.

    Parameters
    ----------
    results : dict[str, Any]
        Output of ``evaluate_encoder`` per encoder.

    Returns
    -------
    dict[str, dict[str, float]]
        Largest absolute difference per encoder and arm.

    Raises
    ------
    ValueError
        If a difference exceeds ``REPRODUCTION_TOLERANCE``.
    """
    differences = {}
    with np.load(PRIOR022B / "predictions.npz") as saved:
        for name, output in results.items():
            differences[name] = {
                arm: max(float(np.abs(output["scores"][key][arm] - saved[f"{key}_{name}_{prior}"]).max())
                         for key in ("development", "sph"))
                for arm, prior in REUSED_022B.items()}
    worst = max(max(values.values()) for values in differences.values())
    if worst > REPRODUCTION_TOLERANCE:
        raise ValueError(f"Arms do not reproduce 022b: {differences}")
    return differences


def percentile_against(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    Share of reference scores strictly below each value.

    Parameters
    ----------
    reference : np.ndarray
        Scores of the reference ECGs.
    values : np.ndarray
        Scores to place.

    Returns
    -------
    np.ndarray
        One share in [0, 1] per value.
    """
    ordered = np.sort(reference)
    return np.searchsorted(ordered, values, side="left") / len(ordered)


def category_of(codes: set[str], categories: tuple[tuple[str, set[str]], ...]) -> str:
    """
    First matching category of a record's SCP codes, or ``other``.

    Parameters
    ----------
    codes : set[str]
        The record's SCP codes.
    categories : tuple[tuple[str, set[str]], ...]
        Names and code sets, in priority order.

    Returns
    -------
    str
        Category name.
    """
    return next((name for name, members in categories if codes & members), "other")


def error_analysis(development: pd.DataFrame, scores: dict[str, np.ndarray]) -> dict[str, Any]:
    """
    Where the hard ECGs rank among ordinary development normals under each arm, by code group and device.

    Parameters
    ----------
    development : pd.DataFrame
        Labeled development rows with ``original``, ``standard``, ``device`` and ``ecg_id``.
    scores : dict[str, np.ndarray]
        Primary-encoder development probabilities per arm.

    Returns
    -------
    dict[str, Any]
        Mean percentile per arm for each hard-negative and hard-positive category, and for ``pooled`` and
        ``ptbxl`` per device with at least 20 hard ECGs.
    """
    from ecg_experiment.eda.ptbxl import load_metadata

    meta = load_metadata()
    original = development["original"].to_numpy(dtype=bool)
    standard = development["standard"].to_numpy(dtype=np.int64)
    reference = original & (standard == 0)
    hard = ~original
    codes = [set(meta.loc[int(ecg_id), "scp_codes"]) for ecg_id in development["ecg_id"]]
    category = np.array([
        category_of(item, HARD_NEGATIVE_CATEGORIES if label == 0 else HARD_POSITIVE_CATEGORIES)
        for item, label in zip(codes, standard, strict=True)])
    percentiles = {arm: percentile_against(values[reference], values) for arm, values in scores.items()}
    groups = {}
    for label, prefix in ((0, "negative"), (1, "positive")):
        for name in np.unique(category[hard & (standard == label)]):
            rows = hard & (standard == label) & (category == name)
            groups[f"{prefix}:{name}"] = {"records": int(rows.sum()), "mean_percentile": {
                arm: float(values[rows].mean()) for arm, values in percentiles.items()}}
    devices = development["device"].to_numpy(dtype=str)
    by_device = {}
    for device in np.unique(devices[hard]):
        for label, prefix in ((0, "negative"), (1, "positive")):
            rows = hard & (devices == device) & (standard == label)
            if np.count_nonzero(hard & (devices == device)) >= 20 and rows.any():
                by_device[f"{device}:{prefix}"] = {"records": int(rows.sum()), "mean_percentile": {
                    arm: float(percentiles[arm][rows].mean()) for arm in ("ptbxl", "pooled")}}
    return {"categories": groups, "devices": by_device}


def reading(results: dict[str, Any], analogues: dict[str, Any]) -> dict[str, Any]:
    """
    Apply the prespecified rule to every candidate arm and every encoder.

    Parameters
    ----------
    results : dict[str, Any]
        Output of ``evaluate_encoder`` per encoder.
    analogues : dict[str, Any]
        Cross-fitted summary of the primary encoder on the dropped training ECGs.

    Returns
    -------
    dict[str, Any]
        Per encoder: the hard-subset loss, each candidate's recovery, contrasts, verdict and interval
        sides; the preferred primary-encoder arm, the decision and whether the analogues confirm it.
    """
    by_encoder = {}
    for name, output in results.items():
        hard, sph = output["sets"]["hard"], output["sets"]["sph"]
        arms = {}
        for arm in CANDIDATES:
            share = recovery(hard["auroc"][arm], hard["auroc"]["pooled"], hard["auroc"]["ptbxl"])
            arms[arm] = {
                "recovery": share, "hard": hard["minus_pooled"][arm], "sph": sph["minus_pooled"][arm],
                "ordinary": output["sets"]["ordinary"]["minus_pooled"][arm],
                "verdict": arm_verdict(hard["minus_pooled"][arm], sph["minus_pooled"][arm], share,
                                       MINIMUM_RECOVERY, SPH_TOLERANCE),
                "sides": {set_name: interval_side(output["sets"][set_name]["minus_pooled"][arm])
                          for set_name in SETS}}
        by_encoder[name] = {"loss": hard["minus_pooled"]["ptbxl"], "arms": arms}
    primary = by_encoder[PRIMARY_ENCODER]["arms"]
    sph_auroc = results[PRIMARY_ENCODER]["sets"]["sph"]["auroc"]
    fixes = [arm for arm in CANDIDATES if primary[arm]["verdict"] == "fix"]
    preferred = max(fixes, key=lambda arm: sph_auroc[arm]) if fixes else None
    confirmed = None
    if preferred:
        confirmed = analogues["minus_pooled"][preferred]["ci_low"] > 0
    return {"by_encoder": by_encoder, "fixes": fixes, "preferred": preferred,
            "decision": "adopt_preferred_arm" if preferred else "no_fix",
            "trade_offs": [arm for arm in CANDIDATES if primary[arm]["verdict"] == "trade_off"],
            "suggestive": [arm for arm in CANDIDATES if primary[arm]["verdict"] == "suggestive"],
            "analogues_confirm_preferred": confirmed}


def prepare() -> dict[str, Any]:
    """
    Load and check every input and build the stacked training rows, without fitting anything.

    Returns
    -------
    dict[str, Any]
        Rows, features, labels, counts and receipts.
    """
    receipts = {"experiment022b": prior022b()["outputs_sha256"], "experiment022": prior022_identity(),
                "experiment020": prior_identity(), "challenge_features": feature_identity()[1]}
    train, development, ptb_x, cache_hashes = ptbxl_rows()
    prior_result = json.loads((PRIOR022 / "result.json").read_text())
    sph, _, sph_hashes = sph_rows(prior_result)
    sph_x = sph_features(sph, prior_result)
    began = time.monotonic()
    universe, challenge_x = challenge_universe()
    quality_seconds = time.monotonic() - began
    families, labels, dropped, patients, passing = stacked_universe(train, universe)
    counts = {
        "ptbxl_train": [len(train), int(train["standard"].sum())],
        "ptbxl_dropped": [int(dropped.sum()), int(train.loc[train["target"].isna(), "standard"].sum())],
        "development": {name: [int(mask.sum()), int(development.loc[mask, "standard"].sum())]
                        for name, mask in (("hard", ~development["original"]),
                                           ("ordinary", development["original"]))},
        "sph": [len(sph), int(sph["primary"].sum())],
        "challenge_rows": {checker: {source: int(count) for source, count in part.value_counts().items()}
                           for checker, part in universe.groupby("checked_by")["source"]},
        "challenge_excluded": {checker: {source: int(count) for source, count in part.value_counts().items()}
                               for checker, part in universe[~passing].groupby("checked_by")["source"]},
        "arms": design_counts(families, labels, dropped),
    }
    return {"receipts": receipts, "train": train, "development": development, "ptb_x": ptb_x,
            "cache_hashes": cache_hashes, "sph": sph, "sph_x": sph_x, "sph_hashes": sph_hashes,
            "universe": universe, "challenge_x": challenge_x, "passing": passing, "families": families,
            "labels": labels, "dropped": dropped, "patients": patients, "counts": counts,
            "quality_seconds": quality_seconds}


def main() -> None:
    """Run the counts stage or the full experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", action="store_true", help="stop after the training rows are built")
    arguments = parser.parse_args()
    if (OUTPUT / "result.json").exists():
        raise FileExistsError("Experiment 035 v1 has already run")
    started = time.monotonic()
    data = prepare()
    print(json.dumps({"stage": "counts", "quality_seconds": data["quality_seconds"], **data["counts"]}),
          flush=True)
    if arguments.counts:
        return
    counts = data["counts"]
    if {key: counts[key] for key in EXPECTED} != EXPECTED or \
            {arm: values["all"] for arm, values in counts["arms"].items()} != EXPECTED_ARMS:
        raise ValueError("Counts differ from the protocol")

    sets = evaluation_sets(data["development"], data["sph"])
    folds = np.full(len(data["families"]), -1, dtype=np.int64)
    ptbxl = data["families"] == "ptbxl"
    folds[ptbxl] = patient_folds(data["patients"][ptbxl], FOLDS, SEED)
    stacked_x = {name: np.concatenate([data["ptb_x"][name]["train"],
                                       data["challenge_x"][name][data["passing"]]])
                 for name in ENCODERS}
    common = (data["families"], data["labels"], data["dropped"])
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = {name: executor.submit(evaluate_encoder, name, stacked_x[name], *common,
                                         {"development": data["ptb_x"][name]["development"],
                                          "sph": data["sph_x"][name]}, sets)
                   for name in ENCODERS}
        fold_futures = [executor.submit(crossfit_fold, fold, stacked_x[PRIMARY_ENCODER], *common, folds)
                        for fold in range(FOLDS)]
        results = {name: future.result() for name, future in futures.items()}
        fold_results = [future.result() for future in fold_futures]
    reproduction = reproduce_022b(results)

    positions = np.concatenate([item["positions"] for item in fold_results])
    order = np.argsort(positions)
    analogue_scores = {arm: np.concatenate([item["scores"][arm] for item in fold_results])[order]
                       for arm in ARMS}
    positions = positions[order]
    analogue_y = data["labels"]["primary"][positions].astype(np.int64)
    analogues = set_summary(analogue_y, data["patients"][positions], analogue_scores)
    analysis = error_analysis(data["development"], results[PRIMARY_ENCODER]["scores"]["development"])

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_npz_atomic(
        OUTPUT / "predictions.npz",
        development_record_ids=data["development"].index.to_numpy(dtype=str),
        development_labels=data["development"]["standard"].to_numpy(dtype=np.int64),
        development_original=data["development"]["original"].to_numpy(dtype=bool),
        sph_ecg_ids=data["sph"]["ecg_id"].to_numpy(dtype=str), sph_labels=sets["sph"]["y"],
        analogue_record_ids=data["train"].index.to_numpy(dtype=str)[positions], analogue_labels=analogue_y,
        **{f"{key}_{name}_{arm}": values for name, output in results.items()
           for key, by_arm in output["scores"].items() for arm, values in by_arm.items()},
        **{f"analogues_{PRIMARY_ENCODER}_{arm}": values for arm, values in analogue_scores.items()})
    data["universe"].drop(columns="codes").assign(codes=data["universe"]["codes"].apply(";".join)).to_csv(
        OUTPUT / "training_rows.csv", index=False)
    result = {
        "status": "complete",
        "identity": {
            **data["receipts"],
            "experiment022b_result_sha256": sha256_file(PRIOR022B / "result.json"),
            "feature_caches": data["cache_hashes"], "sph_manifest": data["sph_hashes"],
            "challenge_split": sha256_file(SPLITS / "rows.csv"),
            "ningbo_rows_sha256": sha256_file(NINGBO / "rows.csv"),
            "descriptive_sha256": sha256_file(OUTPUT / "descriptive.json"),
            "sources": {name: sha256_file(ROOT / name) for name in SOURCES},
        },
        "counts": data["counts"], "arms": {arm: dict(spec) for arm, spec in ARM_SPECS.items()},
        "encoders": list(ENCODERS), "seed": SEED, "draws": DRAWS, "folds": FOLDS,
        "arms_difference_from_022b": reproduction,
        "ranking": {name: {"iterations": output["iterations"], "sets": output["sets"],
                           "seconds": output["seconds"]} for name, output in results.items()},
        "analogues": {"encoder": PRIMARY_ENCODER, **analogues,
                      "iterations": [item["iterations"] for item in fold_results]},
        "error_analysis": analysis,
        "reading": reading(results, analogues),
        "outputs_sha256": {name: sha256_file(OUTPUT / name) for name in ("predictions.npz",
                                                                           "training_rows.csv")},
        "ptbxl_calibration_read": False, "ptbxl_test_read": False, "challenge_test_read": False,
        "quality_seconds": data["quality_seconds"], "runtime_seconds": time.monotonic() - started,
    }
    write_json_atomic(OUTPUT / "result.json", result)
    print(json.dumps({"stage": "complete", "runtime_seconds": result["runtime_seconds"],
                      "decision": result["reading"]["decision"],
                      "preferred": result["reading"]["preferred"]}), flush=True)


if __name__ == "__main__":
    main()
