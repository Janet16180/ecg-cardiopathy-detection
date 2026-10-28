"""
NBR-02 fit sets, rank-0 and rank-6 population models, personalized templates and arm counterparts.

Everything numerical is 062's machinery called unchanged: ``record_features``, ``design``,
``fit_population``, ``subspace_coefficients``, ``preprocess`` and ``delineate``, and 070's
``template_timeline``. The rank-6 template of a record is its median beat minus 062's
``population_residual``: ``covariates @ coefficients + scale * (basis @ coordinates)``.
"""

from __future__ import annotations

import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .. import normative_map062 as nm
from .. import replacement070 as rp
from ..files import sha256_file
from ..locbench_cache import open_cache
from ..locbench_data import load_partition
from ..pr74_screening_features import read_training_signal

COHORT = Path("outputs/cohort_age_label_priority_20261028/all_age.csv")
CHALLENGE_SPLIT = Path("outputs/cohort25_recovery_20261028/challenge/full/rows.csv")
CHALLENGE_SPLIT_SHA256 = "5ecd82154fe9ed574b9706d9b7efbe417a578197845cf2a6a48615ca54959cc8"
SPH_ROWS = Path("outputs/pr74_fresh_baseline_exact_20261028/features/sph_rows.csv")
ARMS = ("P0", "P6", "C0", "C6")
NEW_ARMS = ("P6", "C0", "C6")
SEX_CODE = {"M": 0.0, "F": 1.0}

_STATE: dict[str, Any] = {}


def _limited_features(signal: np.ndarray) -> nm.RecordFeatures | None:
    with threadpool_limits(limits=1):
        return nm.record_features(signal)


def _locbench_job(position: int) -> nm.RecordFeatures | None:
    return _limited_features(np.asarray(_STATE["signals"][position], dtype=float))


def _cohort_job(position: int) -> nm.RecordFeatures | None:
    row = _STATE["rows"].iloc[position]
    with threadpool_limits(limits=1):
        return nm.record_features(read_training_signal(_STATE["input_root"], row))


def stack(features: list[nm.RecordFeatures | None], age: np.ndarray,
          sex: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Medians and covariates of the records 062 could delineate, as ``fit_normative`` builds them.

    Parameters
    ----------
    features : list[nm.RecordFeatures | None]
        Per-record features, None where fewer than two complete beats were found.
    age, sex : np.ndarray
        Covariates in the same order.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Kept positions, flattened medians and the 062 design rows.
    """
    kept = np.array([position for position, value in enumerate(features) if value is not None],
                    dtype=int)
    intervals = np.stack([features[position].intervals for position in kept])
    covariates = nm.design(intervals[:, 0], age[kept], sex[kept])
    medians = np.stack([features[position].median.ravel() for position in kept])
    return kept, medians, covariates


def locbench_fit_set(workers: int) -> dict[str, Any]:
    """
    Build the P fit set: NBR-01's 2,666 LocBench ``fit`` clean normals from the LocBench cache.

    Parameters
    ----------
    workers : int
        Forked single-thread processes.

    Returns
    -------
    dict[str, Any]
        ``medians``, ``covariates``, ``counts`` and the number of records without two complete beats.
    """
    partition = load_partition()
    ids = partition[(partition.partition == "fit") & partition.clean_normal].ecg_id.to_numpy().astype(int)
    signals, index = open_cache()
    row_of = {int(e): i for i, e in enumerate(index.ecg_id)}
    _STATE.update(signals=signals)
    with get_context("fork").Pool(workers) as pool:
        features = pool.map(_locbench_job, [row_of[int(e)] for e in ids], chunksize=16)
    meta = index.set_index("ecg_id").loc[ids]
    kept, medians, covariates = stack(features, meta.age.to_numpy(), meta.sex.to_numpy())
    return {"medians": medians, "covariates": covariates, "records": len(ids), "used": len(kept),
            "too_few_beats": len(ids) - len(kept), "counts": {"ptbxl": len(kept)}}


def cohort_rows(input_root: Path) -> pd.DataFrame:
    """Return the all-age 25k cohort's normal rows (``target == 0``) in file order."""
    rows = pd.read_csv(input_root / COHORT, dtype=str, keep_default_na=False)
    return rows.loc[rows["target"].eq("0")].reset_index(drop=True)


def cohort_checks(input_root: Path, rows: pd.DataFrame) -> dict[str, Any]:
    """
    Prove from identities and hashes alone that no cohort normal is a closed or SPH record.

    Parameters
    ----------
    input_root : Path
        Checkout holding outputs.
    rows : pd.DataFrame
        Output of ``cohort_rows``.

    Returns
    -------
    dict[str, Any]
        Counts of every check; ``passed`` is true only if all overlaps are zero.
    """
    split_path = input_root / CHALLENGE_SPLIT
    if sha256_file(split_path) != CHALLENGE_SPLIT_SHA256:
        raise ValueError("Challenge split table differs from docs/challenge-splits-v1.md")
    split = pd.read_csv(split_path, dtype=str, keep_default_na=False)
    closed = split.loc[split["split"].isin(["test", "calibration"])]
    challenge = rows.loc[rows["source"].ne("ptbxl")]
    merged = challenge.merge(split[["source", "record", "split", "signal_sha256"]], on=["source", "record"],
                             how="left", suffixes=("", "_split"))
    closed_hashes = set(closed["signal_sha256"]) | (set(closed["window_sha256"]) - {""})
    cohort_hashes = set(rows["signal_sha256"]) | set(rows["full_signal_sha256"])
    sph = pd.read_csv(input_root / SPH_ROWS, dtype=str, keep_default_na=False)
    result = {
        "rows": len(rows), "challenge_rows": len(challenge),
        "ptbxl_rows": int(rows["source"].eq("ptbxl").sum()),
        "challenge_rows_matched_in_split": int(merged["split_split"].notna().sum()),
        "challenge_rows_not_train": int(merged["split_split"].ne("train").sum()),
        "challenge_full_hash_equals_split_hash": int(
            merged["full_signal_sha256"].eq(merged["signal_sha256_split"]).sum()),
        "hash_overlap_with_challenge_test_or_calibration": len(cohort_hashes & closed_hashes),
        "record_overlap_with_challenge_test_or_calibration": int(challenge.merge(
            closed[["source", "record"]], on=["source", "record"]).shape[0]),
        "sph_source_rows": int(rows["source"].str.contains("sph").sum()),
        "sph_paths": int(rows["waveform_path"].str.startswith("data/raw/sph").sum()),
        "hash_overlap_with_sph": len(cohort_hashes & set(sph["signal_sha256"])),
        "ptbxl_folds": sorted(rows.loc[rows["source"].eq("ptbxl"), "strat_fold"].unique().tolist()),
        "forbidden_sources": int(rows["source"].str.contains("echonext|mimic|code15|code_15").sum()),
    }
    zero = ("challenge_rows_not_train", "hash_overlap_with_challenge_test_or_calibration",
            "record_overlap_with_challenge_test_or_calibration", "sph_source_rows", "sph_paths",
            "hash_overlap_with_sph", "forbidden_sources")
    result["passed"] = all(result[key] == 0 for key in zero) and set(result["ptbxl_folds"]) <= set("1234567")
    return result


def cohort_fit_set(input_root: Path, rows: pd.DataFrame, workers: int) -> dict[str, Any]:
    """
    Build the C fit set: cohort normals with known age and sex that 062 can delineate.

    Parameters
    ----------
    input_root : Path
        Checkout holding the raw sources.
    rows : pd.DataFrame
        Output of ``cohort_rows``, already checked by ``cohort_checks``.
    workers : int
        Forked single-thread processes.

    Returns
    -------
    dict[str, Any]
        ``medians``, ``covariates``, exclusion counts and used counts by source.
    """
    known = rows["age"].ne("") & rows["age_missing"].eq("False")
    sexed = rows["sex_normalized"].isin(list(SEX_CODE))
    eligible = rows.loc[known & sexed].reset_index(drop=True)
    _STATE.update(rows=eligible, input_root=input_root)
    with get_context("fork").Pool(workers) as pool:
        features = pool.map(_cohort_job, range(len(eligible)), chunksize=8)
    age = eligible["age"].astype(float).to_numpy()
    sex = eligible["sex_normalized"].map(SEX_CODE).to_numpy()
    kept, medians, covariates = stack(features, age, sex)
    used = eligible.iloc[kept]
    return {"medians": medians, "covariates": covariates, "records": len(rows),
            "missing_age": int((~known).sum()), "unknown_sex": int((known & ~sexed).sum()),
            "too_few_beats": len(eligible) - len(kept), "used": len(kept),
            "counts": used["source"].value_counts().sort_index().to_dict()}


def fit_models(fit_set: dict[str, Any]) -> dict[int, nm.PopulationModel]:
    """Fit 062's rank-0 and rank-6 population models on one fit set."""
    return {rank: nm.fit_population(fit_set["medians"], fit_set["covariates"], rank)
            for rank in (0, nm.SUBSPACE)}


def record_template(features: nm.RecordFeatures, age: float, sex: float,
                    model: nm.PopulationModel) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the population (rank 0) or personalized (rank 6) template of one record on the 062 grid.

    Parameters
    ----------
    features : nm.RecordFeatures
        The record's 062 features.
    age, sex : float
        Covariates, sex coded 0 male and 1 female.
    model : nm.PopulationModel
        Fitted model; its basis width sets the rank.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (12, BINS) template in mV and the (rank,) subspace coordinates.
    """
    covariates = nm.design(features.intervals[:1], np.array([age]), np.array([sex]))[0]
    mean = covariates @ model.coefficients
    standard = (features.median.ravel() - mean) / model.scale
    coordinates = nm.subspace_coefficients(model, standard)
    template = mean + model.scale * (model.basis @ coordinates)
    return template.reshape(-1, nm.BINS), coordinates


def arm_differences(raw: np.ndarray, age: float, sex: float,
                    models: dict[str, nm.PopulationModel]) -> dict[str, Any] | None:
    """
    Counterpart-minus-record differences of every arm for one record, with template diagnostics.

    Parameters
    ----------
    raw : np.ndarray
        (12, 5000) signal in mV at 500 Hz.
    age, sex : float
        Covariates, sex coded 0 male and 1 female.
    models : dict[str, nm.PopulationModel]
        Model of each arm name.

    Returns
    -------
    dict[str, Any] or None
        Per arm: float32 ``difference`` (12, 5000), ``coordinates`` and the grid ``template``; plus
        ``knots``. None when 062 finds fewer than two complete beats.
    """
    raw = np.asarray(raw, np.float64)
    features = nm.record_features(raw)
    if features is None:
        return None
    clean = nm.preprocess(raw, nm.delineate(raw)["qon"])
    result: dict[str, Any] = {"knots": features.knots}
    for arm, model in models.items():
        template, coordinates = record_template(features, age, sex, model)
        timeline = rp.template_timeline(template, features.knots, raw.shape[1])
        result[arm] = {"difference": (timeline - clean).astype(np.float32), "coordinates": coordinates,
                       "template": template, "clipped": clipped(coordinates, model)}
    return result


def clipped(coordinates: np.ndarray, model: nm.PopulationModel) -> int:
    """Count subspace coordinates that sit on 062's ``BOX`` clip bound."""
    if not len(coordinates):
        return 0
    bound = nm.BOX * np.sqrt(model.eigenvalues)
    return int(np.sum(np.isclose(np.abs(coordinates), bound, rtol=0.0, atol=1e-12)))


def fit_summary(started: float, **sets: dict[str, Any]) -> dict[str, Any]:
    """Summarize each fit set without its arrays."""
    return {name: {key: value for key, value in fit.items() if key not in ("medians", "covariates")}
            for name, fit in sets.items()} | {"seconds": time.perf_counter() - started}
