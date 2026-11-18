"""
Features, frozen models and thresholds of MLC-04.

Features of a role are extracted from the role's waveform cache by the MLC-02 extractors
(``mlc02_features.run_encoder`` and ``run_measurements``) and published in cache order. The frozen models are
scored with their own code and artefacts, checked against their ``model.json`` hashes:

- ``P0-v5`` and ``H-All-v5``: trained on dataset v5 (MLC-03 Part D, ``effect/candidates``);
- ``P0-v4`` and ``H-All-v4``: trained on dataset v4 (MLC-02, ``candidates``);
- ``K6``: the KEY-05 key, standardized pooled xECG centred on its train source's mean, dot product with the
  key (``key05_keys.score_single``); every Challenge source has a train mean.

The thresholds of ``P0-v5`` follow the MLC-02 rule (``mlc02_evaluation.specificity_thresholds``: 90%
specificity on all validation records where the condition is defined, pooled over validation sources), on the
v5 validation labels and P0-v5's stored validation probabilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ecg_experiment import ROOT
from ecg_experiment import key05_keys as kk
from ecg_experiment import mlc03_dataset_v5 as mv
from ecg_experiment.files import sha256_file
from ecg_experiment.mlc02_data import CONDITIONS, load_manifest, load_predictions, subset
from ecg_experiment.mlc02_evaluation import specificity_thresholds, threshold_rates
from ecg_experiment.mlc02_feature_models import CANDIDATES, load_candidate, predict_loaded, standardize
from ecg_experiment.mlc02_features import assemble, run_encoder, run_measurements
from ecg_experiment.mlc04_data import OUTPUT, WORK, check_role, feature_positions

FEATURE_SETS = ("xecg", "jepa", "cpc", "measurements")
EFFECT_CANDIDATES = mv.QUALITY / "effect" / "candidates"
KEYS = ROOT / "outputs/key05_condition_keys_20261111/keys.npz"
KEY_SCORES = ROOT / "outputs/key05_condition_keys_20261111/scores_validation.npz"
MODELS = ("P0-v5", "P0-v4", "H-All-v5", "H-All-v4", "K6")
CANDIDATE_MODELS = {
    "P0-v5": ("P0", EFFECT_CANDIDATES),
    "P0-v4": ("P0", CANDIDATES),
    "H-All-v5": ("H-All", EFFECT_CANDIDATES),
    "H-All-v4": ("H-All", CANDIDATES),
}
STORED_VALIDATION = {
    "P0-v5": EFFECT_CANDIDATES / "P0" / "validation__seeds-42.npz",
    "P0-v4": CANDIDATES / "P0" / "validation__seeds-42.npz",
    "H-All-v5": EFFECT_CANDIDATES / "H-All" / "validation__seeds-20261110-20261111-20261112.npz",
    "H-All-v4": CANDIDATES / "H-All" / "validation__seeds-20261110-20261111-20261112.npz",
}
STATUS_COLUMNS = ["status", "implausible", "flat_leads"]


def extraction_directory(role: str, name: str) -> Path:
    """
    Shard directory of one extractor for one role.

    Parameters
    ----------
    role : str
        ``validation`` or ``challenge_test``.
    name : str
        One of ``FEATURE_SETS``.

    Returns
    -------
    Path
        ``WORK/<role>/features/<name>``.
    """
    return WORK / role / "features" / name


def extract(role: str, name: str, gpu: int | None, workers: int) -> dict[str, Any]:
    """
    Run one MLC-02 extractor on every cached record of a role.

    Parameters
    ----------
    role : str
        ``validation`` or ``challenge_test``.
    name : str
        One of ``FEATURE_SETS``.
    gpu : int | None
        Physical GPU of an encoder (``CUDA_VISIBLE_DEVICES`` must name it alone).
    workers : int
        Processes of the measurement extractor.

    Returns
    -------
    dict[str, Any]
        The extractor's run record.
    """
    role = check_role(role)
    cache = WORK / role / "cache"
    directory = extraction_directory(role, name)
    if name == "measurements":
        return run_measurements(directory, workers, None, cache=cache, index_name="index.csv")
    return run_encoder(name, directory, int(gpu), None, cache=cache, index_name="index.csv")


def publish_features(role: str, target: Path) -> dict[str, str]:
    """
    Concatenate a role's finished extractions in cache order and write them with their rows.

    Parameters
    ----------
    role : str
        ``validation`` or ``challenge_test``.
    target : Path
        Directory for ``<name>.npy``, ``measurements_status.csv`` and ``rows.csv``.

    Returns
    -------
    dict[str, str]
        SHA-256 of every written file.

    Raises
    ------
    ValueError
        If an extraction does not cover the cache.
    """
    role = check_role(role)
    index = pd.read_csv(WORK / role / "cache" / "index.csv", dtype={"record_key": str, "raw_path": str})
    target.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in FEATURE_SETS:
        values, status, _ = assemble(name, extraction_directory(role, name))
        if len(values) != len(index):
            raise ValueError(f"{name} covers {len(values)} of {len(index)} records")
        np.save(target / f"{name}.npy", values)
        hashes[f"{name}.npy"] = sha256_file(target / f"{name}.npy")
        if status is not None:
            status.insert(0, "record_key", index["record_key"].to_numpy())
            status.to_csv(target / "measurements_status.csv", index=False)
            hashes["measurements_status.csv"] = sha256_file(target / "measurements_status.csv")
    index[["position", "record_key", "source", "role", "sha256_10s", "sha256_f32"]].to_csv(
        target / "rows.csv", index=False
    )
    hashes["rows.csv"] = sha256_file(target / "rows.csv")
    return hashes


def load_features(role: str, keys: np.ndarray, directory: Path | None = None) -> dict[str, np.ndarray]:
    """
    Float64 features of some records of a role, from its published arrays.

    Parameters
    ----------
    role : str
        ``validation`` or ``challenge_test``.
    keys : np.ndarray
        Record keys.
    directory : Path | None
        Published feature directory; None uses ``OUTPUT/<role>/features``.

    Returns
    -------
    dict[str, np.ndarray]
        One ``[records, width]`` matrix per feature set, aligned with ``keys``.
    """
    role = check_role(role)
    directory = OUTPUT / role / "features" if directory is None else Path(directory)
    rows = pd.read_csv(directory / "rows.csv", dtype={"record_key": str})
    positions = feature_positions(rows, keys)
    return {
        name: np.asarray(np.load(directory / f"{name}.npy", mmap_mode="r")[positions], dtype=np.float64)
        for name in FEATURE_SETS
    }


def load_measurement_status(role: str, keys: np.ndarray, directory: Path | None = None) -> pd.Series:
    """
    MLC-02 measurement extractor status of some records of a role.

    Parameters
    ----------
    role : str
        ``validation`` or ``challenge_test``.
    keys : np.ndarray
        Record keys.
    directory : Path | None
        Published feature directory; None uses ``OUTPUT/<role>/features``.

    Returns
    -------
    pd.Series
        Status text aligned with ``keys``.
    """
    role = check_role(role)
    directory = OUTPUT / role / "features" if directory is None else Path(directory)
    status = pd.read_csv(directory / "measurements_status.csv", dtype=str).set_index("record_key")
    return status.loc[np.asarray(keys), "status"].reset_index(drop=True)


def load_models() -> dict[str, Any]:
    """
    Load every frozen model with its artefact hashes checked.

    Returns
    -------
    dict[str, Any]
        ``load_candidate`` output per candidate model, and ``K6`` (key, source means and standardizer).
    """
    models: dict[str, Any] = {
        name: load_candidate(candidate, directory)
        for name, (candidate, directory) in CANDIDATE_MODELS.items()
    }
    with np.load(KEYS, allow_pickle=False) as archive:
        content = {name: archive[name] for name in archive.files}
    models["K6"] = {
        "key": content["K6"],
        "means": {
            name.removeprefix("source_mean__"): value
            for name, value in content.items()
            if name.startswith("source_mean__")
        },
        "scaler": {"mean": content["scaler_mean"], "scale": content["scaler_scale"]},
    }
    return models


def key_scores(model: dict[str, Any], xecg: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """
    K6 scores of raw pooled xECG rows, centred on each record's train source mean.

    Parameters
    ----------
    model : dict[str, Any]
        ``load_models()["K6"]``.
    xecg : np.ndarray
        ``[records, 1024]`` raw embeddings.
    sources : np.ndarray
        Source of each record.

    Returns
    -------
    np.ndarray
        ``[records, 14]`` scores (dot products, unbounded).

    Raises
    ------
    KeyError
        If a source has no train mean.
    """
    sources = np.asarray(sources, dtype=str)
    missing = sorted(set(sources) - set(model["means"]))
    if missing:
        raise KeyError(f"no train source mean for {missing}")
    x = standardize(xecg, model["scaler"])
    return kk.score_single("K6", x, model["key"], sources, model["means"])


def predict_all(
    models: dict[str, Any], features: dict[str, np.ndarray], sources: np.ndarray
) -> dict[str, np.ndarray]:
    """
    Probabilities of the four candidate models and K6 scores of the same records.

    Parameters
    ----------
    models : dict[str, Any]
        Output of ``load_models``.
    features : dict[str, np.ndarray]
        Raw feature matrices of the records.
    sources : np.ndarray
        Source of each record.

    Returns
    -------
    dict[str, np.ndarray]
        ``[records, 14]`` per model id of ``MODELS``.
    """
    out = {name: predict_loaded(models[name], features) for name in CANDIDATE_MODELS}
    out["K6"] = key_scores(models["K6"], features["xecg"], sources)
    return out


def stored_validation(name: str, keys: np.ndarray) -> np.ndarray:
    """
    Read the stored validation output of one frozen model for some of its validation records.

    Parameters
    ----------
    name : str
        Model id of ``MODELS``.
    keys : np.ndarray
        Record keys, all held by the stored file.

    Returns
    -------
    np.ndarray
        ``[records, 14]`` probabilities (K6: raw scores), aligned with ``keys``.
    """
    if name == "K6":
        with np.load(KEY_SCORES, allow_pickle=False) as archive:
            stored_keys, values = archive["record_key"], archive["K6"]
    else:
        with np.load(STORED_VALIDATION[name], allow_pickle=False) as archive:
            stored_keys = archive["record_key"]
        values = load_predictions(STORED_VALIDATION[name], stored_keys)["probability"]
    position = pd.Series(np.arange(len(stored_keys)), index=np.asarray(stored_keys, dtype=str))
    return np.asarray(values, dtype=np.float64)[position.loc[np.asarray(keys, dtype=str)].to_numpy()]


def v5_validation() -> dict[str, np.ndarray]:
    """
    Arrays of the included v5 validation records (all six sources), in manifest order.

    Returns
    -------
    dict[str, np.ndarray]
        ``mlc02_data.load_manifest`` arrays of dataset v5, validation role only.
    """
    data = load_manifest(mv.V5 / "manifest.csv")
    return subset(data, data["role"] == "validation")


def fit_thresholds() -> dict[str, Any]:
    """
    Thresholds of P0-v5 by the MLC-02 rule on the v5 validation records, with their validation rates.

    Returns
    -------
    dict[str, Any]
        ``thresholds`` (condition to value), ``validation_at_threshold`` (pooled sensitivity, specificity,
        positives and negatives), ``records`` and the SHA-256 of the inputs.
    """
    data = v5_validation()
    probability = load_predictions(STORED_VALIDATION["P0-v5"], data["record_key"])["probability"]
    thresholds = specificity_thresholds(data["labels"], probability)
    rates = {}
    for column, name in enumerate(CONDITIONS):
        defined = ~np.isnan(data["labels"][:, column])
        y = data["labels"][defined, column]
        rates[name] = threshold_rates(y, probability[defined, column], thresholds[column]) | {
            "positives": int(y.sum()),
            "negatives": int(len(y) - y.sum()),
        }
    return {
        "rule": "mlc02_evaluation.specificity_thresholds: lowest observed score with specificity >= 0.90 "
        "on all v5 validation records where the condition is defined, pooled over the six sources",
        "model": "P0-v5",
        "records": len(data["record_key"]),
        "thresholds": dict(zip(CONDITIONS, thresholds.tolist(), strict=True)),
        "validation_at_threshold": rates,
        "inputs_sha256": {
            "v5_manifest": sha256_file(mv.V5 / "manifest.csv"),
            "p0_v5_validation_predictions": sha256_file(STORED_VALIDATION["P0-v5"]),
        },
    }
