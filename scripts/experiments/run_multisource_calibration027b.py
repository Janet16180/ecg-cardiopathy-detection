"""Experiment 027b: Platt calibration and the 95%-sensitivity threshold fitted on several hospitals."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from ecg_experiment.challenge_splits import FAMILIES
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
from ecg_experiment.screening_threshold import calibrate, head_logits, operating_point
from scripts.experiments.run_calibrated_threshold027 import (
    HEADS,
    calibration_rows,
    prior022_identity,
    reproduce_heads,
    sph_rows,
)
from scripts.experiments.run_sph_external022 import open_caches

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/experiment027b_multisource_calibration_v1"
PRIOR027 = ROOT / "outputs/experiment027_calibrated_threshold_v1"
PRIOR022 = ROOT / "outputs/experiment022_sph_external_v3"
SPLITS = ROOT / "data/processed/challenge_splits_v1"
FEATURES = ROOT / "outputs/features_challenge_v1"
NINGBO = ROOT / "data/processed/ningbo_clean_v1"
SPLIT_ROWS_SHA256 = "5ecd82154fe9ed574b9706d9b7efbe417a578197845cf2a6a48615ca54959cc8"
CHALLENGE_FAMILIES = ("chapman_ningbo", "georgia", "cpsc")
ARMS = ("ptbxl", "pooled", "balanced", *(f"loso_{family}" for family in CHALLENGE_FAMILIES))
EXPECTED_CALIBRATION = {"ptbxl": (564, 348), "chapman_shaoxing": (1018, 745), "ningbo": (3414, 2509),
                        "georgia": (1718, 1372), "cpsc_2018": (872, 689), "cpsc_2018_extra": (590, 590)}
REPRODUCTION_TOLERANCE = 1e-9
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 29029
THREADS = 4
SOURCES = (
    "ecg_experiment/multisource_calibration.py", "ecg_experiment/screening_threshold.py",
    "ecg_experiment/evaluation.py", "ecg_experiment/challenge_splits.py",
    "ecg_experiment/external_readout.py", "ecg_experiment/full_development.py",
    "scripts/experiments/run_calibrated_threshold027.py",
    "scripts/experiments/run_sph_external022.py", "scripts/experiments/run_multisource_calibration027b.py",
    "pyproject.toml", "uv.lock", "docs/experiment-027b-multisource-calibration.md",
)


def prior027_identity() -> tuple[dict[str, object], dict[str, str]]:
    """
    Load 027's result and check the artifacts this rerun reuses against it.

    Returns
    -------
    tuple[dict[str, object], dict[str, str]]
        027's ``result.json`` and the SHA-256 of its reused files.

    Raises
    ------
    ValueError
        If an artifact differs from 027's receipt.
    """
    result = json.loads((PRIOR027 / "result.json").read_text())
    names = ("calibration_features.npz", "calibrated_predictions.npz")
    hashes = {name: sha256_file(PRIOR027 / name) for name in names}
    changed = any(hashes[name] != result["outputs_sha256"][name] for name in names)
    if result["status"] != "complete" or changed:
        raise ValueError("Experiment 027 artifacts differ from its receipt")
    return result, {**hashes, "result.json": sha256_file(PRIOR027 / "result.json")}


def feature_identity() -> tuple[dict[str, object], dict[str, str]]:
    """
    Require complete Challenge features whose npz files match their receipt.

    Returns
    -------
    tuple[dict[str, object], dict[str, str]]
        The feature ``metadata.json`` and the hashes of it and every npz.

    Raises
    ------
    ValueError
        If the extraction is incomplete or a file differs from its receipt.
    """
    metadata = json.loads((FEATURES / "metadata.json").read_text())
    if metadata["status"] != "complete" or set(metadata["sources"]) != set(FAMILIES):
        raise ValueError("Challenge features are not complete")
    hashes = {source: sha256_file(FEATURES / f"{source}.npz") for source in FAMILIES}
    if any(hashes[source] != metadata["sources"][source]["npz_sha256"] for source in FAMILIES):
        raise ValueError("A Challenge feature file differs from its receipt")
    return metadata, {**hashes, "metadata.json": sha256_file(FEATURES / "metadata.json")}


def split_rows() -> pd.DataFrame:
    """
    Read the evaluable calibration-group rows of the frozen Challenge split.

    Returns
    -------
    pd.DataFrame
        Rows with ``source``, ``family``, ``record`` and ``primary``, and ``zero_leads`` for Ningbo records.

    Raises
    ------
    ValueError
        If the split or the Ningbo manifest differs from its receipt.
    """
    metadata = json.loads((SPLITS / "metadata.json").read_text())
    digest = sha256_file(SPLITS / "rows.csv")
    if digest != metadata["rows_sha256"] or digest != SPLIT_ROWS_SHA256:
        raise ValueError("Challenge split differs from its receipt")
    if sha256_file(NINGBO / "rows.csv") != metadata["inputs_sha256"]["ningbo_rows_csv"]:
        raise ValueError("Ningbo manifest differs from the split's input")
    rows = pd.read_csv(SPLITS / "rows.csv", dtype={"record": str})
    rows = rows[rows["evaluable"] & (rows["split"] == "calibration")]
    zero = pd.read_csv(NINGBO / "rows.csv", dtype={"record": str}, usecols=["record", "zero_leads"])
    rows = rows.merge(zero, on="record", how="left")
    rows["zero_lead"] = rows["zero_leads"].fillna("") != ""
    return rows


def challenge_scores(rows: pd.DataFrame, heads: dict[str, object],
                     ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """
    Head probabilities of the calibration-group ECGs that have frozen-encoder features.

    Parameters
    ----------
    rows : pd.DataFrame
        Output of ``split_rows``.
    heads : dict[str, object]
        Refitted heads keyed by encoder.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, np.ndarray]]
        The scored rows in source then record order, and probabilities per head name.
    """
    frames, scores = [], {name: [] for name in HEADS.values()}
    for source in FAMILIES:
        with np.load(FEATURES / f"{source}.npz") as saved:
            records = pd.Index(saved["record"].astype(str))
            part = rows[(rows["source"] == source) & rows["record"].isin(records)].sort_values("record")
            positions = records.get_indexer(part["record"])
            for encoder, head in heads.items():
                scores[HEADS[encoder]].append(predict(head, saved[encoder][positions]))
        frames.append(part)
    joined = {name: np.concatenate(values) for name, values in scores.items()}
    return pd.concat(frames, ignore_index=True), joined


def check_counts(ptb_y: np.ndarray, challenge: pd.DataFrame) -> dict[str, dict[str, int]]:
    """
    Require the calibration counts of the protocol.

    Parameters
    ----------
    ptb_y : np.ndarray
        PTB-XL calibration labels.
    challenge : pd.DataFrame
        Scored Challenge calibration rows.

    Returns
    -------
    dict[str, dict[str, int]]
        ECGs and positives per source.

    Raises
    ------
    ValueError
        If a count differs from ``EXPECTED_CALIBRATION``.
    """
    found = {"ptbxl": (len(ptb_y), int(ptb_y.sum()))}
    for source, part in challenge.groupby("source"):
        found[source] = (len(part), int(part["primary"].sum()))
    if found != EXPECTED_CALIBRATION:
        raise ValueError(f"Calibration counts differ from the protocol: {found}")
    return {name: {"ecgs": value[0], "positives": value[1]} for name, value in found.items()}


def arm_members(families: np.ndarray) -> dict[str, np.ndarray]:
    """
    Calibration ECGs of each arm.

    Parameters
    ----------
    families : np.ndarray
        Family of each pooled calibration ECG (``ptbxl`` or a Challenge family).

    Returns
    -------
    dict[str, np.ndarray]
        Boolean selection per arm.
    """
    members = {"ptbxl": families == "ptbxl", "pooled": np.ones(len(families), dtype=bool),
               "balanced": np.ones(len(families), dtype=bool)}
    for family in CHALLENGE_FAMILIES:
        members[f"loso_{family}"] = families != family
    return members


def fit_arms(y: np.ndarray, families: np.ndarray, scores: dict[str, np.ndarray],
             ) -> tuple[dict[str, dict[str, LogisticRegression]], dict[str, dict[str, float]]]:
    """
    Fit every arm's Platt mapping and threshold per head.

    Parameters
    ----------
    y : np.ndarray
        Labels of the pooled calibration ECGs.
    families : np.ndarray
        Their families.
    scores : dict[str, np.ndarray]
        Head probabilities per head name.

    Returns
    -------
    tuple[dict[str, dict[str, LogisticRegression]], dict[str, dict[str, float]]]
        Calibrators and thresholds keyed by arm, then head.
    """
    calibrators, thresholds = {}, {}
    for arm, selected in arm_members(families).items():
        weights = family_weights(families) if arm == "balanced" else None
        calibrators[arm], thresholds[arm] = {}, {}
        for head, values in scores.items():
            fitted = fit_arm(head_logits(values[selected]), y[selected], weights)
            calibrators[arm][head], thresholds[arm][head] = fitted
    return calibrators, thresholds


def reproduce_027(result027: dict[str, object], calibrators: dict[str, LogisticRegression],
                  thresholds: dict[str, float], sph_calibrated: dict[str, np.ndarray]) -> dict[str, float]:
    """
    Require the ``ptbxl`` arm to reproduce 027's Platt fits, thresholds and SPH calibrated probabilities.

    Parameters
    ----------
    result027 : dict[str, object]
        027's ``result.json``.
    calibrators, thresholds : dict
        The ``ptbxl`` arm, keyed by head.
    sph_calibrated : dict[str, np.ndarray]
        The ``ptbxl`` arm's calibrated SPH probabilities per head.

    Returns
    -------
    dict[str, float]
        Largest absolute difference per head over every compared value.

    Raises
    ------
    ValueError
        If any difference exceeds ``REPRODUCTION_TOLERANCE``.
    """
    differences = {}
    with np.load(PRIOR027 / "calibrated_predictions.npz") as saved:
        for head, model in calibrators.items():
            platt = result027["platt"][head]
            differences[head] = max(abs(model.coef_[0, 0] - platt["slope"]),
                                    abs(model.intercept_[0] - platt["intercept"]),
                                    abs(thresholds[head] - result027["thresholds"][head]),
                                    float(np.abs(sph_calibrated[head] - saved[f"sph_{head}"]).max()))
    if max(differences.values()) > REPRODUCTION_TOLERANCE:
        raise ValueError(f"The ptbxl arm does not reproduce Experiment 027: {differences}")
    return differences


def calibrated_matrix(scores: dict[str, np.ndarray], calibrators: dict[str, dict[str, LogisticRegression]],
                      thresholds: dict[str, dict[str, float]], arms: tuple[str, ...],
                      ) -> tuple[list[tuple[str, str]], np.ndarray, np.ndarray]:
    """
    Calibrated probabilities of one set under several arms, one column per arm and head.

    Parameters
    ----------
    scores : dict[str, np.ndarray]
        Head probabilities of the set.
    calibrators, thresholds : dict
        Output of ``fit_arms``.
    arms : tuple[str, ...]
        Arms to apply.

    Returns
    -------
    tuple[list[tuple[str, str]], np.ndarray, np.ndarray]
        Columns, the ``(n, k)`` calibrated probabilities and the ``(k,)`` thresholds.
    """
    columns = [(arm, head) for arm in arms for head in scores]
    matrix = np.column_stack([calibrate(calibrators[arm][head], head_logits(scores[head]))
                              for arm, head in columns])
    return columns, matrix, np.array([thresholds[arm][head] for arm, head in columns])


def evaluate_set(units: np.ndarray, y: np.ndarray, scores: dict[str, np.ndarray],
                 calibrators: dict[str, dict[str, LogisticRegression]],
                 thresholds: dict[str, dict[str, float]], arms: tuple[str, ...],
                 ) -> tuple[dict[str, object], np.ndarray]:
    """
    Operating points and paired bootstrap summary of one evaluation set.

    Parameters
    ----------
    units : np.ndarray
        Bootstrap unit of each ECG.
    y : np.ndarray
        Labels.
    scores : dict[str, np.ndarray]
        Head probabilities.
    calibrators, thresholds : dict
        Output of ``fit_arms``.
    arms : tuple[str, ...]
        Arms to evaluate; the first is the reference.

    Returns
    -------
    tuple[dict[str, object], np.ndarray]
        ``operating_points[arm][head]`` and ``bootstrap``, and the calibrated matrix.
    """
    columns, matrix, cutoffs = calibrated_matrix(scores, calibrators, thresholds, arms)
    points = {}
    for index, (arm, head) in enumerate(columns):
        points.setdefault(arm, {})[head] = operating_point(y, matrix[:, index], cutoffs[index])
    observed = weighted_rates(np.ones(len(y)), y, matrix, cutoffs)
    drawn = bootstrap_rates(units, y, matrix, cutoffs, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    return {"records": len(y), "positives": int(y.sum()), "operating_points": points,
            "bootstrap": summarize(columns, observed, drawn, arms[0])}, matrix


def in_sample(y: np.ndarray, families: np.ndarray, scores: dict[str, np.ndarray],
              calibrators: dict[str, dict[str, LogisticRegression]],
              thresholds: dict[str, dict[str, float]]) -> dict[str, dict[str, dict[str, object]]]:
    """
    Each arm's operating point on its own calibration ECGs.

    Parameters
    ----------
    y, families : np.ndarray
        Labels and families of the pooled calibration ECGs.
    scores : dict[str, np.ndarray]
        Their head probabilities.
    calibrators, thresholds : dict
        Output of ``fit_arms``.

    Returns
    -------
    dict[str, dict[str, dict[str, object]]]
        Operating points keyed by arm, then head.
    """
    points = {}
    for arm, selected in arm_members(families).items():
        points[arm] = {head: operating_point(y[selected],
                                             calibrate(calibrators[arm][head], head_logits(values[selected])),
                                             thresholds[arm][head])
                       for head, values in scores.items()}
    return points


def reading(sph: dict[str, object]) -> dict[str, object]:
    """
    Apply the prespecified primary comparison, decision rule and 027 transfer rule.

    Parameters
    ----------
    sph : dict[str, object]
        The SPH result of ``evaluate_set``.

    Returns
    -------
    dict[str, object]
        Verdict per head and arm, the decision, and whether each arm's threshold transfers.
    """
    differences = sph["bootstrap"]["differences"]
    verdicts = {arm: {head: verdict(values["deviation"]) for head, values in heads.items()}
                for arm, heads in differences.items()}
    primary = list(verdicts["pooled"].values())
    adopt = primary.count("better") >= 2 and "worse" not in primary
    transfers = {arm: {head: rates["sensitivity"]["ci_low"] <= 0.95 <= rates["sensitivity"]["ci_high"]
                       for head, rates in heads.items()}
                 for arm, heads in sph["bootstrap"]["rates"].items()}
    return {"deviation_verdicts": verdicts, "primary_verdicts": verdicts["pooled"],
            "adopt_pooled_calibration": adopt, "threshold_transfers_to_sph": transfers}


def heldout_sets(challenge: pd.DataFrame) -> dict[str, tuple[np.ndarray, str]]:
    """
    Select the ECGs of each held-out-family readout.

    Parameters
    ----------
    challenge : pd.DataFrame
        Scored Challenge calibration rows.

    Returns
    -------
    dict[str, tuple[np.ndarray, str]]
        Boolean selection within ``challenge`` and the held-out family, per readout name.
    """
    family = challenge["family"].to_numpy()
    sets = {family_name: (family == family_name, family_name) for family_name in CHALLENGE_FAMILIES}
    nonzero = ~challenge["zero_lead"].to_numpy()
    sets["chapman_ningbo_without_zero_leads"] = ((family == "chapman_ningbo") & nonzero, "chapman_ningbo")
    return sets


def main() -> None:
    """Fit every calibration arm and read it once on SPH and on the held-out Challenge families."""
    started = time.monotonic()
    feature_metadata, feature_hashes = feature_identity()
    result027, hashes027 = prior027_identity()
    prior_result = json.loads((PRIOR022 / "result.json").read_text())
    caches, cache_hashes = open_caches()
    if cache_hashes != prior_result["identity"]["feature_caches"]:
        raise ValueError("Feature caches differ from Experiment 022")
    table = ptb_table()
    groups = cohorts(table)
    calibration, references_sha256 = calibration_rows(table, groups)
    sph, sph_scores, sph_hashes = sph_rows(prior_result)
    heads, head_differences = reproduce_heads(groups, caches)

    with threadpool_limits(limits=THREADS), np.load(PRIOR027 / "calibration_features.npz") as saved:
        if not np.array_equal(saved["ecg_ids"], calibration["ecg_id"].to_numpy(dtype=np.int64)):
            raise ValueError("027 calibration features differ from the calibration rows")
        ptb_scores = {HEADS[name]: predict(head, saved[name]) for name, head in heads.items()}
        challenge, scored = challenge_scores(split_rows(), heads)
    ptb_y = calibration["standard"].to_numpy(dtype=np.int64)
    counts = check_counts(ptb_y, challenge)

    y = np.concatenate([ptb_y, challenge["primary"].to_numpy(dtype=np.int64)])
    families = np.concatenate([np.full(len(ptb_y), "ptbxl"), challenge["family"].to_numpy(dtype=str)])
    pool_scores = {head: np.concatenate([ptb_scores[head], scored[head]]) for head in ptb_scores}
    with threadpool_limits(limits=THREADS):
        calibrators, thresholds = fit_arms(y, families, pool_scores)
        sph_y = sph["primary"].to_numpy(dtype=np.int64)
        sph_result, sph_matrix = evaluate_set(sph["patient_id"].to_numpy(dtype=str), sph_y, sph_scores,
                                              calibrators, thresholds, ARMS)
        reference = {head: sph_matrix[:, index] for index, head in enumerate(sph_scores)}
        reproduction = reproduce_027(result027, calibrators["ptbxl"], thresholds["ptbxl"], reference)
        heldout, heldout_matrices = {}, {}
        for name, (selected, family) in heldout_sets(challenge).items():
            part = challenge[selected]
            heldout[name], heldout_matrices[name] = evaluate_set(
                part["record"].to_numpy(dtype=str), part["primary"].to_numpy(dtype=np.int64),
                {head: values[selected] for head, values in scored.items()}, calibrators, thresholds,
                ("ptbxl", f"loso_{family}"))
        own = in_sample(y, families, pool_scores, calibrators, thresholds)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_npz_atomic(OUTPUT / "challenge_calibration_scores.npz",
                     record=challenge["record"].to_numpy(dtype=str),
                     source=challenge["source"].to_numpy(dtype=str),
                     label=challenge["primary"].to_numpy(dtype=np.int64), **scored)
    write_npz_atomic(OUTPUT / "calibrated_predictions.npz", sph=sph_matrix, sph_labels=sph_y,
                     **{f"heldout_{name}": matrix for name, matrix in heldout_matrices.items()})
    platt = {arm: {head: {"slope": float(model.coef_[0, 0]), "intercept": float(model.intercept_[0])}
                   for head, model in models.items()}
             for arm, models in calibrators.items()}
    result = {
        "status": "complete",
        "identity": {
            "experiment027": hashes027, "experiment022": prior022_identity(), "sph_manifest": sph_hashes,
            "heldout_references": references_sha256, "feature_caches": cache_hashes,
            "challenge_features": feature_hashes, "challenge_features_identity": feature_metadata["identity"],
            "challenge_split": {"rows.csv": sha256_file(SPLITS / "rows.csv"),
                                "metadata.json": sha256_file(SPLITS / "metadata.json")},
            "sources": {name: sha256_file(ROOT / name) for name in SOURCES},
        },
        "head_max_abs_difference_from_022": head_differences,
        "ptbxl_arm_max_abs_difference_from_027": reproduction,
        "calibration_counts": counts, "arms": list(ARMS), "platt": platt, "thresholds": thresholds,
        "sph": sph_result, "heldout_families": heldout, "in_sample": own, "reading": reading(sph_result),
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED}, "challenge_test_read": False,
        "ptbxl_test_read": False,
        "outputs_sha256": {name: sha256_file(OUTPUT / name)
                           for name in ("challenge_calibration_scores.npz", "calibrated_predictions.npz")},
        "runtime_seconds": time.monotonic() - started,
    }
    write_json_atomic(OUTPUT / "result.json", result)
    print(json.dumps({"stage": "complete", "runtime_seconds": result["runtime_seconds"],
                      "reading": result["reading"]["primary_verdicts"]}), flush=True)


if __name__ == "__main__":
    main()
