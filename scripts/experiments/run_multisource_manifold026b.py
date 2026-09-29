"""Experiment 026b: the distance-from-normal score fitted on normal ECGs from several hospitals."""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ecg_experiment.challenge_features import RAW_ROOTS, read_verified
from ecg_experiment.downloads import parse_checksums
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.full_development import cohorts, ptb_table
from ecg_experiment.multisource_manifold import equal_family_subsample, window_reasons
from ecg_experiment.normal_manifold import (
    bootstrap_metrics,
    contrast,
    fit_mahalanobis,
    mahalanobis_scores,
    metrics,
    patient_resamples,
)
from scripts.experiments import run_label_efficiency025 as label_efficiency
from scripts.experiments import run_normal_manifold026 as manifold026
from scripts.experiments.run_multisource_calibration027b import feature_identity

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/experiment026b_multisource_manifold_v1"
PRIOR026 = ROOT / "outputs/experiment026_normal_manifold_v1"
SPLITS = ROOT / "data/processed/challenge_splits_v1"
FEATURES = ROOT / "outputs/features_challenge_v1"
NINGBO = ROOT / "data/processed/ningbo_clean_v1"
SPLIT_ROWS_SHA256 = "5ecd82154fe9ed574b9706d9b7efbe417a578197845cf2a6a48615ca54959cc8"
ENCODERS = ("cpc", "jepa", "xecg")
SOURCES_ORDER = ("ningbo", "chapman_shaoxing", "georgia", "cpsc_2018", "cpsc_2018_extra")
CHALLENGE_FAMILIES = ("chapman_ningbo", "georgia", "cpsc")
ARMS = ("ptbxl", "pooled", "balanced", *(f"loso_{family}" for family in CHALLENGE_FAMILIES))
PRIMARY_ENCODER = "xecg"
SEED = 30030
BOOTSTRAP_DRAWS = 2000
REPRODUCTION_TOLERANCE = 1e-9
EXPECTED_FIT = {"ptbxl": 5872, "ningbo": 2581, "chapman_shaoxing": 819, "georgia": 1023, "cpsc_2018": 551}
EXPECTED_EXCLUDED = {"ningbo": 126, "chapman_shaoxing": 1, "georgia": 4, "cpsc_2018": 0}
EXPECTED_ARMS = {"ptbxl": 5872, "pooled": 10846, "balanced": 2204, "loso_chapman_ningbo": 7446,
                 "loso_georgia": 9823, "loso_cpsc": 10295}
EXPECTED_FAMILY_SETS = {"chapman_ningbo": (4432, 3254), "chapman_ningbo_without_zero_leads": (4362, 3190),
                        "georgia": (1718, 1372), "cpsc": (1462, 1279)}
SOURCES = (
    "ecg_experiment/multisource_manifold.py", "ecg_experiment/normal_manifold.py",
    "ecg_experiment/ecg_quality.py", "ecg_experiment/challenge_features.py",
    "ecg_experiment/challenge_splits.py", "ecg_experiment/full_development.py",
    "scripts/experiments/run_multisource_manifold026b.py", "scripts/experiments/run_normal_manifold026.py",
    "scripts/experiments/run_label_efficiency025.py",
    "scripts/experiments/run_multisource_calibration027b.py", "pyproject.toml", "uv.lock",
    "docs/experiment-026b-multisource-normal-manifold.md",
)


def prior_scores() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Load 026's result and its saved development and SPH scores, checked against its receipt.

    Returns
    -------
    tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]
        026's ``result.json``, ``development_scores.csv`` and ``sph_scores.csv``.

    Raises
    ------
    ValueError
        If a score file differs from 026's recorded hash.
    """
    result = json.loads((PRIOR026 / "result.json").read_text())
    for name in ("development_scores", "sph_scores"):
        if sha256_file(PRIOR026 / f"{name}.csv") != result[f"{name}_sha256"]:
            raise ValueError(f"Experiment 026 {name}.csv differs from its receipt")
    development = pd.read_csv(PRIOR026 / "development_scores.csv", index_col=0, float_precision="round_trip")
    sph = pd.read_csv(PRIOR026 / "sph_scores.csv", dtype={"ecg_id": str}, float_precision="round_trip")
    return result, development, sph


def split_table() -> pd.DataFrame:
    """
    Read the evaluable train and calibration rows of the frozen Challenge split with Ningbo quality.

    Returns
    -------
    pd.DataFrame
        Split rows with ``use_training`` (Ningbo only, else NaN) and a boolean ``zero_lead``.

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
    rows = rows[rows["evaluable"] & rows["split"].isin(["train", "calibration"])]
    ningbo = pd.read_csv(NINGBO / "rows.csv", dtype={"record": str},
                         usecols=["record", "zero_leads", "use_training"])
    rows = rows.merge(ningbo, on="record", how="left")
    rows["zero_lead"] = rows["zero_leads"].fillna("") != ""
    return rows.drop(columns="zero_leads")


def challenge_features(rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """
    Keep the rows that have frozen-encoder features and load those features.

    Parameters
    ----------
    rows : pd.DataFrame
        Output of ``split_table``.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, np.ndarray]]
        The rows in source then record order with their ``window_start``, and features per encoder.
    """
    frames, features = [], {name: [] for name in ENCODERS}
    for source in SOURCES_ORDER:
        with np.load(FEATURES / f"{source}.npz") as saved:
            records = pd.Index(saved["record"].astype(str))
            part = rows[(rows["source"] == source) & rows["record"].isin(records)].sort_values("record")
            positions = records.get_indexer(part["record"])
            part = part.assign(window_start=saved["window_start"][positions])
            for name in ENCODERS:
                features[name].append(saved[name][positions])
        frames.append(part)
    return pd.concat(frames, ignore_index=True), {name: np.concatenate(v) for name, v in features.items()}


def quality_reasons(normals: pd.DataFrame) -> pd.Series:
    """
    Training quality-policy exclusion reasons of each Challenge training normal.

    Ningbo uses the manifest's ``use_training``; the other sources are checked on the saved feature window.

    Parameters
    ----------
    normals : pd.DataFrame
        Train-group primary negatives with ``source``, ``path``, ``window_start`` and ``use_training``.

    Returns
    -------
    pd.Series
        ``;``-joined reasons per row, empty when the record passes.
    """
    checksums = {source: parse_checksums((RAW_ROOTS[source] / "SHA256SUMS.txt").read_text())
                 for source in normals["source"].unique()}
    reasons = []
    for row in normals.itertuples(index=False):
        if row.source == "ningbo":
            reasons.append("" if row.use_training else "ningbo_use_training_false")
            continue
        signal, _, names = read_verified(RAW_ROOTS[row.source], row.path, checksums[row.source])
        reasons.append(";".join(window_reasons(signal, names, int(row.window_start))))
    return pd.Series(reasons, index=normals.index)


def arm_positions(families: np.ndarray) -> dict[str, np.ndarray]:
    """
    Fit-set rows of each arm.

    Parameters
    ----------
    families : np.ndarray
        Family of each fit-set candidate (``ptbxl`` rows first, in 026's order).

    Returns
    -------
    dict[str, np.ndarray]
        Sorted positions per arm.
    """
    positions = {"ptbxl": np.flatnonzero(families == "ptbxl"), "pooled": np.arange(len(families)),
                 "balanced": equal_family_subsample(families, SEED)}
    for family in CHALLENGE_FAMILIES:
        positions[f"loso_{family}"] = np.flatnonzero(families != family)
    return positions


def family_sets(challenge: pd.DataFrame) -> dict[str, tuple[np.ndarray, str]]:
    """
    Calibration-group ECGs of each family readout.

    Parameters
    ----------
    challenge : pd.DataFrame
        Challenge calibration rows with features.

    Returns
    -------
    dict[str, tuple[np.ndarray, str]]
        Boolean selection and the family, per readout name.
    """
    family = challenge["family"].to_numpy()
    sets = {name: (family == name, name) for name in CHALLENGE_FAMILIES}
    sets["chapman_ningbo_without_zero_leads"] = ((family == "chapman_ningbo")
                                                 & ~challenge["zero_lead"].to_numpy(), "chapman_ningbo")
    return sets


def check_counts(fit: pd.DataFrame, positions: dict[str, np.ndarray], challenge: pd.DataFrame
                 ) -> dict[str, Any]:
    """
    Require the fit-set and evaluation counts of the protocol.

    Parameters
    ----------
    fit : pd.DataFrame
        Every fit-set candidate with ``source`` and ``reasons``.
    positions : dict[str, np.ndarray]
        Output of ``arm_positions`` on the kept candidates.
    challenge : pd.DataFrame
        Challenge calibration rows with features.

    Returns
    -------
    dict[str, Any]
        Fit ECGs and exclusions per source, ECGs per arm, and ECGs and positives per family readout.

    Raises
    ------
    ValueError
        If a count differs from the protocol.
    """
    kept = fit["reasons"] == ""
    found = {
        "fit": {source: int(size) for source, size in fit[kept].groupby("source").size().items()},
        "excluded": fit[fit["source"] != "ptbxl"].groupby("source")["reasons"].apply(
            lambda reasons: int((reasons != "").sum())).to_dict(),
        "arms": {arm: len(rows) for arm, rows in positions.items()},
        "family_sets": {name: (int(selected.sum()), int(challenge.loc[selected, "primary"].sum()))
                        for name, (selected, _) in family_sets(challenge).items()},
    }
    expected = {"fit": EXPECTED_FIT, "excluded": EXPECTED_EXCLUDED, "arms": EXPECTED_ARMS,
                "family_sets": EXPECTED_FAMILY_SETS}
    if found != expected:
        raise ValueError(f"Counts differ from the protocol: {found}")
    return found


def evaluate_set(y: np.ndarray, units: np.ndarray, scores: dict[str, np.ndarray],
                 pairs: list[tuple[str, str]]) -> dict[str, Any]:
    """
    Metrics of every arm and paired bootstrap contrasts on one evaluation set.

    Parameters
    ----------
    y : np.ndarray
        Binary labels.
    units : np.ndarray
        Bootstrap unit of each ECG (patient or record).
    scores : dict[str, np.ndarray]
        Distance score per arm.
    pairs : list[tuple[str, str]]
        Contrasts as (first, second), first minus second.

    Returns
    -------
    dict[str, Any]
        Counts, ``metrics`` per arm, ``contrasts`` and the number of skipped draws.
    """
    observed = {arm: metrics(y, values) for arm, values in scores.items()}
    resamples, invalid = patient_resamples(units, y, BOOTSTRAP_DRAWS, SEED)
    resampled = bootstrap_metrics(y, scores, resamples)
    return {"records": len(y), "positives": int(y.sum()), "units": len(np.unique(units)), "metrics": observed,
            "contrasts": {f"{first}_minus_{second}": contrast(observed, resampled, first, second)
                          for first, second in pairs},
            "invalid_draws": invalid}


def evaluate_encoder(name: str, fit_x: np.ndarray, positions: dict[str, np.ndarray],
                     inputs: dict[str, np.ndarray], sets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Fit every arm on one encoder's features and evaluate it on SPH, development and the families.

    Parameters
    ----------
    name : str
        Encoder name.
    fit_x : np.ndarray
        Features of every kept fit-set candidate.
    positions : dict[str, np.ndarray]
        Output of ``arm_positions``.
    inputs : dict[str, np.ndarray]
        Features of each scored set: ``sph``, ``development`` and ``challenge``.
    sets : dict[str, dict[str, Any]]
        Output of ``evaluation_sets``.

    Returns
    -------
    dict[str, Any]
        Explained variance per arm, scores per scored set and arm, and each set's evaluation.
    """
    start = time.monotonic()
    with threadpool_limits(limits=1):
        models = {arm: fit_mahalanobis(fit_x[rows]) for arm, rows in positions.items()}
        scores = {set_name: {arm: mahalanobis_scores(model, x) for arm, model in models.items()}
                  for set_name, x in inputs.items()}
        evaluations = {}
        for set_name, spec in sets.items():
            arm_scores = {arm: values[spec["selection"]] for arm, values in scores[spec["scores_of"]].items()}
            evaluations[set_name] = evaluate_set(spec["y"], spec["units"], arm_scores, spec["pairs"])
            print(json.dumps({"stage": f"{name}:{set_name}", "auroc": {
                arm: round(value["auroc"], 4) for arm, value in evaluations[set_name]["metrics"].items()}}),
                flush=True)
    return {"pca_explained_variance": {arm: float(model[1].explained_variance_ratio_.sum())
                                       for arm, model in models.items()},
            "scores": scores, "evaluations": evaluations, "seconds": time.monotonic() - start}


def reproduce_026(results: dict[str, Any], development: pd.DataFrame, sph: pd.DataFrame) -> dict[str, float]:
    """
    Require the ``ptbxl`` arm to reproduce 026's saved Mahalanobis scores.

    Parameters
    ----------
    results : dict[str, Any]
        Output of ``evaluate_encoder`` per encoder.
    development, sph : pd.DataFrame
        026's saved score tables, in the same row order as this run's sets.

    Returns
    -------
    dict[str, float]
        Largest relative difference per encoder over development and SPH.

    Raises
    ------
    ValueError
        If any difference exceeds ``REPRODUCTION_TOLERANCE``.
    """
    differences = {}
    for name, output in results.items():
        pairs = ((output["scores"]["development"]["ptbxl"], development[f"{name}_mahalanobis"].to_numpy()),
                 (output["scores"]["sph"]["ptbxl"], sph[f"{name}_mahalanobis"].to_numpy()))
        differences[name] = max(float(np.max(np.abs(new - old) / np.maximum(1.0, np.abs(old))))
                                for new, old in pairs)
    if max(differences.values()) > REPRODUCTION_TOLERANCE:
        raise ValueError(f"The ptbxl arm does not reproduce Experiment 026: {differences}")
    return differences


def interval_side(values: dict[str, float]) -> str:
    """
    Where a contrast interval lies relative to 0.

    Parameters
    ----------
    values : dict[str, float]
        ``ci_low`` and ``ci_high``.

    Returns
    -------
    str
        ``above 0``, ``below 0`` or ``includes 0``.
    """
    side = "includes 0"
    if values["ci_low"] > 0:
        side = "above 0"
    elif values["ci_high"] < 0:
        side = "below 0"
    return side


def reading(results: dict[str, Any], probes: dict[str, Any]) -> dict[str, Any]:
    """
    Apply the prespecified decision rule and describe every contrast by its interval.

    Parameters
    ----------
    results : dict[str, Any]
        Output of ``evaluate_encoder`` per encoder.
    probes : dict[str, Any]
        026's record of the Experiment 022 SPH probes per encoder.

    Returns
    -------
    dict[str, Any]
        The primary side and decision, the share of the SPH gap closed, and the side of every contrast.
    """
    primary = results[PRIMARY_ENCODER]["evaluations"]["sph"]["contrasts"]["pooled_minus_ptbxl"]["auroc"]
    side = interval_side(primary)
    decision = {"above 0": "adopt_pooled", "below 0": "pooling_hurts"}.get(side, "not_distinguished")
    gap_closed = {}
    for name, output in results.items():
        sph = output["evaluations"]["sph"]["metrics"]
        gap = probes[name]["primary"]["auroc"] - sph["ptbxl"]["auroc"]
        gap_closed[name] = {arm: (value["auroc"] - sph["ptbxl"]["auroc"]) / gap for arm, value in sph.items()}
    sides = {name: {set_name: {pair: {metric: interval_side(values) for metric, values in contrasts.items()}
                               for pair, contrasts in evaluation["contrasts"].items()}
                    for set_name, evaluation in output["evaluations"].items()}
             for name, output in results.items()}
    return {"primary_contrast": primary, "primary_side": side, "decision": decision,
            "share_of_sph_gap_to_probe_closed": gap_closed, "interval_sides": sides}


def score_tables(frames: dict[str, pd.DataFrame], results: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """
    Per-row scores of every encoder and arm, one table per scored set.

    Parameters
    ----------
    frames : dict[str, pd.DataFrame]
        Identifying columns per scored set.
    results : dict[str, Any]
        Output of ``evaluate_encoder`` per encoder.

    Returns
    -------
    dict[str, pd.DataFrame]
        Tables keyed by scored set.
    """
    tables = {}
    for set_name, frame in frames.items():
        table = frame.copy()
        for name, output in results.items():
            for arm, values in output["scores"][set_name].items():
                table[f"{name}_{arm}"] = values
        tables[set_name] = table
    return tables


def evaluation_sets(evaluation: pd.DataFrame, sph: pd.DataFrame, challenge: pd.DataFrame
                    ) -> dict[str, dict[str, Any]]:
    """
    Labels, units, contrasts and row selections of every evaluation set.

    Parameters
    ----------
    evaluation : pd.DataFrame
        026's development rows.
    sph : pd.DataFrame
        SPH evaluation rows.
    challenge : pd.DataFrame
        Challenge calibration rows with features.

    Returns
    -------
    dict[str, dict[str, Any]]
        Per set: the scored set it reads (``scores_of``), its ``selection`` within it, ``y``, ``units`` and
        the contrast ``pairs``.
    """
    versus_reference = [(arm, "ptbxl") for arm in ARMS[1:]]
    sets = {"sph": {"scores_of": "sph", "selection": np.ones(len(sph), dtype=bool),
                    "y": sph["primary"].to_numpy(dtype=np.int64),
                    "units": sph["patient_id"].to_numpy(dtype=str), "pairs": versus_reference},
            "development": {"scores_of": "development", "selection": np.ones(len(evaluation), dtype=bool),
                            "y": evaluation["standard"].to_numpy(dtype=np.int64),
                            "units": evaluation["patient_id"].to_numpy(dtype=str), "pairs": versus_reference}}
    for set_name, (selected, family) in family_sets(challenge).items():
        part = challenge[selected]
        sets[f"family:{set_name}"] = {
            "scores_of": "challenge", "selection": selected, "y": part["primary"].to_numpy(dtype=np.int64),
            "units": part["record"].to_numpy(dtype=str),
            "pairs": [(f"loso_{family}", "ptbxl"), ("pooled", f"loso_{family}")]}
    return sets


def main() -> None:
    """Fit every normal reference once and score SPH, PTB-XL development and the Challenge families."""
    if (OUTPUT / "result.json").exists():
        raise FileExistsError("Experiment 026b v1 has already run")
    started = time.monotonic()
    identity026 = manifold026.identity()
    features_metadata, feature_hashes = feature_identity()
    result026, prior_development, prior_sph = prior_scores()
    groups = cohorts(ptb_table())
    pool, evaluation, _ = label_efficiency.select_rows(groups["train"], groups["development"])
    ptb_features = label_efficiency.load_features(groups["train"], groups["development"], pool, evaluation)
    sph, sph_features = manifold026.load_sph()
    if not (np.array_equal(prior_development["ecg_id"].to_numpy(), evaluation["ecg_id"].to_numpy())
            and np.array_equal(prior_sph["ecg_id"].to_numpy(), sph["ecg_id"].to_numpy())):
        raise ValueError("Row order differs from Experiment 026")

    rows, challenge_x = challenge_features(split_table())
    normal = ((rows["split"] == "train") & (rows["primary"] == 0)).to_numpy()
    calibration = (rows["split"] == "calibration").to_numpy()
    normals = rows[normal].copy()
    normals["reasons"] = quality_reasons(normals)
    ptb_fit = (pool["standard"] == 0).to_numpy()
    fit = pd.concat([pd.DataFrame({"family": "ptbxl", "source": "ptbxl", "record": pool["ecg_id"][ptb_fit]
                                   .astype(str).to_numpy(), "reasons": ""}),
                     normals[["family", "source", "record", "reasons"]]], ignore_index=True)
    kept = (fit["reasons"] == "").to_numpy()
    challenge_kept = kept[ptb_fit.sum():]
    positions = arm_positions(fit["family"].to_numpy()[kept])
    challenge = rows[calibration].reset_index(drop=True)
    counts = check_counts(fit, positions, challenge)
    print(json.dumps({"stage": "counts", **counts}), flush=True)

    sets = evaluation_sets(evaluation, sph, challenge)
    with ProcessPoolExecutor(max_workers=len(ENCODERS)) as executor:
        futures = {}
        for name in ENCODERS:
            challenge_fit = challenge_x[name][normal][challenge_kept]
            fit_x = np.concatenate([ptb_features[name][0][ptb_fit], challenge_fit])
            inputs = {"sph": sph_features[name], "development": ptb_features[name][1],
                      "challenge": challenge_x[name][calibration]}
            futures[name] = executor.submit(evaluate_encoder, name, fit_x, positions, inputs, sets)
        results = {name: future.result() for name, future in futures.items()}
    reproduction = reproduce_026(results, prior_development, prior_sph)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    frames = {"development": evaluation[["ecg_id", "patient_id", "standard"]].reset_index(drop=True),
              "sph": sph[["ecg_id", "patient_id", "primary"]],
              "challenge": challenge[["source", "family", "record", "primary", "zero_lead"]]}
    for set_name, table in score_tables(frames, results).items():
        table.to_csv(OUTPUT / f"{set_name}_scores.csv", index=False)
    fit_table = fit.assign(kept=kept)
    for arm, rows_of_arm in positions.items():
        member = np.zeros(int(kept.sum()), dtype=bool)
        member[rows_of_arm] = True
        fit_table.loc[kept, arm] = member
    fit_table.to_csv(OUTPUT / "fit_sets.csv", index=False)
    written = ("development_scores.csv", "sph_scores.csv", "challenge_scores.csv", "fit_sets.csv")
    result = {
        "status": "complete",
        "identity": {"experiment026": identity026, "experiment026_result_sha256": sha256_file(
                         PRIOR026 / "result.json"),
                     "challenge_features": feature_hashes,
                     "challenge_features_identity": features_metadata["identity"],
                     "challenge_split": {"rows.csv": sha256_file(SPLITS / "rows.csv"),
                                         "metadata.json": sha256_file(SPLITS / "metadata.json")},
                     "ningbo_rows_sha256": sha256_file(NINGBO / "rows.csv"),
                     "sources": {name: sha256_file(ROOT / name) for name in SOURCES}},
        "counts": counts, "arms": list(ARMS), "seed": SEED, "bootstrap_draws": BOOTSTRAP_DRAWS,
        "ptbxl_arm_max_relative_difference_from_026": reproduction,
        "encoders": {name: {key: value for key, value in output.items() if key != "scores"}
                     for name, output in results.items()},
        "reading": reading(results, result026["sph_probes_experiment022"]),
        "sph_probes_experiment022": result026["sph_probes_experiment022"],
        "outputs_sha256": {name: sha256_file(OUTPUT / name) for name in written},
        "challenge_test_read": False, "ptbxl_calibration_test_read": False,
        "runtime_seconds": time.monotonic() - started,
    }
    write_json_atomic(OUTPUT / "result.json", result)
    print(json.dumps({"stage": "complete", "runtime_seconds": result["runtime_seconds"],
                      "decision": result["reading"]["decision"],
                      "primary": result["reading"]["primary_contrast"]}), flush=True)


if __name__ == "__main__":
    main()
