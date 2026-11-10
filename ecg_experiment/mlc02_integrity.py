"""Integrity checks of MLC-02 Stage 1: new features against earlier caches, and KEY-04's SPH macro AUROC.

Earlier caches are matched to the new features by record key on the records they share with the MLC-01
manifest. Where a cache stored the float32 window hash (``public_sources.signal_sha256``), it is compared with
the hash of the cached MLC-02 window. No label of the MLC-01 dataset is read and no classifier is fitted on
it;
the only fit is KEY-04's own refit on KEY-04's cached features, which the protocol names.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd

from ecg_experiment import ROOT
from ecg_experiment.cpu_screening041 import RAW_FEATURES
from ecg_experiment.files import sha256_file

HEADS = "outputs/heads01_token_superclass_20261028"
PR74 = "outputs/pr74_fresh_baseline_exact_20261028/features"
SCREEN = "outputs/screen01_matched_transfer_20261027/cuda_runtime_execution"
AVANCE4 = "outputs/avance4_modelos/features"
KEY04 = "outputs/condition_classifiers_20261106"
KEY04_MACRO_AUROC = 0.9608446027429329
KEY04_C = 10.0
TOLERANCE = 1e-4
CACHES = (
    {
        "cache": "HEADS-01 training",
        "extractor": "xecg",
        "features": f"{HEADS}/extract_training/pooled.npy",
        "rows": f"{HEADS}/extract_training/rows.csv",
        "key": "record_id",
    },
    {
        "cache": "HEADS-01 evaluation",
        "extractor": "xecg",
        "features": f"{HEADS}/extract_evaluation/pooled.npy",
        "rows": f"{HEADS}/extract_evaluation/rows.csv",
        "key": "record_id",
    },
    {
        "cache": "PR-74 PTB-XL",
        "extractor": "xecg",
        "features": f"{PR74}/ptb_xecg.npy",
        "rows": f"{PR74}/ptb_rows.csv",
        "key": "record_id",
        "hash": "signal_sha256",
    },
    {
        "cache": "PR-74 SPH",
        "extractor": "xecg",
        "features": f"{PR74}/sph_xecg.npy",
        "rows": f"{PR74}/sph_rows.csv",
        "key": "record_id",
        "hash": "signal_sha256",
    },
    {
        "cache": "PR-74 PTB-XL",
        "extractor": "measurements",
        "features": f"{PR74}/ptb_measurements.csv",
        "rows": f"{PR74}/ptb_rows.csv",
        "key": "record_id",
        "hash": "signal_sha256",
    },
    {
        "cache": "PR-74 SPH",
        "extractor": "measurements",
        "features": f"{PR74}/sph_measurements.csv",
        "rows": f"{PR74}/sph_rows.csv",
        "key": "record_id",
        "hash": "signal_sha256",
    },
    {
        "cache": "SCREEN-01 PTB-XL",
        "extractor": "xecg",
        "features": f"{SCREEN}/qualify/ptb_xecg.npy",
        "rows": f"{SCREEN}/qualify/ptb_rows.csv",
        "key": "ecg_id",
        "prefix": "ptbxl:",
    },
    {
        "cache": "SCREEN-01 PTB-XL",
        "extractor": "jepa",
        "features": f"{SCREEN}/qualify/ptb_ecg_jepa.npy",
        "rows": f"{SCREEN}/qualify/ptb_rows.csv",
        "key": "ecg_id",
        "prefix": "ptbxl:",
    },
    {
        "cache": "SCREEN-01 SPH",
        "extractor": "xecg",
        "features": f"{SCREEN}/extract/sph_xecg.npy",
        "rows": f"{SCREEN}/qualify/sph_rows.csv",
        "key": "ecg_id",
        "prefix": "sph:",
        "hash": "signal_sha256",
    },
    {
        "cache": "SCREEN-01 SPH",
        "extractor": "jepa",
        "features": f"{SCREEN}/extract/sph_ecg_jepa.npy",
        "rows": f"{SCREEN}/qualify/sph_rows.csv",
        "key": "ecg_id",
        "prefix": "sph:",
        "hash": "signal_sha256",
    },
    {
        "cache": "Avance 4 PTB-XL",
        "extractor": "xecg",
        "features": f"{AVANCE4}/xecg/features.npy",
        "rows": f"{AVANCE4}/xecg/ecg_ids.npy",
        "prefix": "ptbxl:",
    },
    {
        "cache": "Avance 4 PTB-XL",
        "extractor": "jepa",
        "features": f"{AVANCE4}/ecg_jepa/features.npy",
        "rows": f"{AVANCE4}/ecg_jepa/ecg_ids.npy",
        "prefix": "ptbxl:",
    },
    {
        "cache": "Avance 4 PTB-XL",
        "extractor": "cpc",
        "features": f"{AVANCE4}/ecg_cpc/features.npy",
        "rows": f"{AVANCE4}/ecg_cpc/ecg_ids.npy",
        "prefix": "ptbxl:",
    },
)


def cache_keys(spec: dict[str, str]) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Record keys of a cache's rows and, when stored, their float32 window hashes.

    Parameters
    ----------
    spec : dict[str, str]
        One entry of ``CACHES``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray | None]
        Keys in the manifest's ``source:record`` form, and hashes or None.
    """
    path = ROOT / spec["rows"]
    if path.suffix == ".npy":
        return np.array([spec["prefix"] + str(int(value)) for value in np.load(path)]), None
    columns = [spec["key"]] + ([spec["hash"]] if "hash" in spec else [])
    rows = pd.read_csv(path, usecols=columns, dtype=str)
    keys = (spec.get("prefix", "") + rows[spec["key"]]).to_numpy()
    return keys, (rows[spec["hash"]].to_numpy() if "hash" in spec else None)


def cache_values(spec: dict[str, str]) -> np.ndarray:
    """
    Feature rows of a cache as float64.

    Parameters
    ----------
    spec : dict[str, str]
        One entry of ``CACHES``.

    Returns
    -------
    np.ndarray
        ``[rows, dimension]``; measurement CSVs in ``RAW_FEATURES`` order.
    """
    path = ROOT / spec["features"]
    if path.suffix == ".csv":
        return pd.read_csv(path)[list(RAW_FEATURES)].to_numpy(dtype=np.float64)
    return np.load(path, mmap_mode="r").astype(np.float64)


def six_digits(values: np.ndarray) -> np.ndarray:
    """
    Values after the six-significant-digit CSV round trip used by the PR-74 measurement cache.

    Parameters
    ----------
    values : np.ndarray
        Float ``[rows, 130]`` measurements.

    Returns
    -------
    np.ndarray
        The same values written with ``%.6g`` and read back.
    """
    frame = pd.DataFrame(values, columns=list(RAW_FEATURES))
    return pd.read_csv(io.StringIO(frame.to_csv(index=False, float_format="%.6g")))[
        list(RAW_FEATURES)
    ].to_numpy()


def compare_vectors(new: np.ndarray, old: np.ndarray) -> dict[str, object]:
    """
    Differences between new and cached encoder features on shared records.

    The relative difference of a record is its largest absolute difference divided by the largest absolute
    cached value of that record.

    Parameters
    ----------
    new, old : np.ndarray
        Aligned ``[records, dimension]`` features.

    Returns
    -------
    dict[str, object]
        ``max_abs``, ``median_abs``, ``max_rel``, ``records_rel_above_tolerance``, ``bit_identical_records``
        and ``worst_rows`` (row numbers of the five largest relative differences).
    """
    new, old = np.asarray(new, dtype=np.float64), np.asarray(old, dtype=np.float64)
    difference = np.abs(new - old)
    relative = difference.max(axis=1) / np.abs(old).max(axis=1)
    return {
        "max_abs": float(difference.max()),
        "median_abs": float(np.median(difference)),
        "max_rel": float(relative.max()),
        "records_rel_above_tolerance": int((relative > TOLERANCE).sum()),
        "bit_identical_records": int((difference == 0).all(axis=1).sum()),
        "worst_rows": np.argsort(-relative)[:5].tolist(),
    }


def compare_measurements(new: np.ndarray, old: np.ndarray) -> dict[str, object]:
    """
    Differences between new and cached measurements on shared records, element by element.

    Parameters
    ----------
    new, old : np.ndarray
        Aligned ``[records, 130]`` measurements; the cache holds six significant digits.

    Returns
    -------
    dict[str, object]
        ``max_abs`` and ``max_rel`` over elements finite in both (relative to the cached value, nonzero
        cached values only), ``missing_pattern_mismatches``, ``zero_mismatches`` (cached 0, new not),
        ``records_rel_above_tolerance``, ``records_equal_after_six_digits`` and the worst records and columns.
    """
    new, old = np.asarray(new, dtype=np.float64), np.asarray(old, dtype=np.float64)
    both = np.isfinite(new) & np.isfinite(old)
    pattern = np.isfinite(new) != np.isfinite(old)
    difference = np.where(both, np.abs(new - old), 0.0)
    nonzero = both & (old != 0)
    relative = np.where(nonzero, difference / np.where(nonzero, np.abs(old), 1.0), 0.0)
    zero = both & (old == 0) & (new != 0)
    bad = (relative > TOLERANCE) | pattern | zero
    rounded = six_digits(new)
    equal = ((rounded == old) | (np.isnan(rounded) & np.isnan(old))).all(axis=1)
    counts = bad.sum(axis=0)
    return {
        "max_abs": float(difference.max()),
        "max_rel": float(relative.max()),
        "missing_pattern_mismatches": int(pattern.sum()),
        "zero_mismatches": int(zero.sum()),
        "records_rel_above_tolerance": int(bad.any(axis=1).sum()),
        "records_equal_after_six_digits": int(equal.sum()),
        "worst_rows": np.flatnonzero(bad.any(axis=1))[:20].tolist(),
        "columns_above_tolerance": {RAW_FEATURES[i]: int(counts[i]) for i in np.flatnonzero(counts)},
    }


def check_cache(spec: dict[str, str], index: pd.DataFrame, features: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """
    Compare one cache with the new features of the same extractor.

    Parameters
    ----------
    spec : dict[str, str]
        One entry of ``CACHES``.
    index : pd.DataFrame
        MLC-02 cache index (``record_key``, ``source``, ``sha256_f32``).
    features : np.ndarray
        New features of the extractor in index order.

    Returns
    -------
    tuple[dict, pd.DataFrame]
        Summary row, and the records whose difference exceeds the tolerance (key and source).
    """
    keys, hashes = cache_keys(spec)
    values = cache_values(spec)
    if len(keys) != len(values):
        raise ValueError(f"{spec['features']}: {len(values)} rows, {len(keys)} keys")
    position = pd.Series(index.index.to_numpy(), index=index["record_key"].to_numpy())
    shared = np.flatnonzero(pd.Index(position.index).get_indexer(keys) >= 0)
    where = position.loc[keys[shared]].to_numpy()
    row = {
        "cache": spec["cache"],
        "extractor": spec["extractor"],
        "cache_rows": len(keys),
        "shared": len(shared),
        "sources": ",".join(sorted(set(index["source"].iloc[where]))),
        "features_sha256": sha256_file(ROOT / spec["features"]),
    }
    if hashes is not None:
        row["window_hash_equal"] = int((index["sha256_f32"].to_numpy()[where] == hashes[shared]).sum())
    if spec["extractor"] == "measurements":
        stats = compare_measurements(features[where], values[shared])
    else:
        stats = compare_vectors(features[where], values[shared])
    worst = index.iloc[where[stats.pop("worst_rows")]][["record_key", "source"]]
    row.update(stats)
    return row, worst.assign(cache=spec["cache"], extractor=spec["extractor"])


def key04_reproduction() -> dict[str, object]:
    """
    KEY-04's SPH macro AUROC, read-only from its saved predictions and from a refit on its cached features.

    Returns
    -------
    dict[str, object]
        Saved value, the recomputation from ``predictions.npz``, the refit's value, their differences, the
        largest score difference of the refit and the number of matching decimal digits.
    """
    from threadpoolctl import threadpool_limits

    from ecg_experiment.condition_keys import ranking_metrics
    from ecg_experiment.key_classifier_models import fit_package, predict_package
    from ecg_experiment.key_classifier_study import qualify

    result = json.loads((ROOT / KEY04 / "result.json").read_text())
    saved = result["metrics"]["independent_logistic"]["macro_auroc"]
    predictions = np.load(ROOT / KEY04 / "predictions.npz", allow_pickle=False)
    recomputed = ranking_metrics(predictions["flags"], predictions["score_independent_logistic"])[
        "macro_auroc"
    ]
    with threadpool_limits(2):
        data, qualification = qualify()
        if not np.array_equal(data["query_flags"], predictions["flags"]):
            raise ValueError("KEY-04 query flags differ from its saved predictions")
        package = fit_package(data["train_pooled"], data["train_flags"], "independent_logistic", KEY04_C)
        scores, _ = predict_package(package, data["query_pooled"])
    refit = ranking_metrics(data["query_flags"], scores)["macro_auroc"]
    difference = abs(refit - saved)
    digits = int(np.floor(-np.log10(difference))) if difference > 0 else 16
    return {
        "saved_macro_auroc": saved,
        "expected": KEY04_MACRO_AUROC,
        "recomputed_from_predictions": recomputed,
        "recomputed_equal": recomputed == saved,
        "refit_macro_auroc": refit,
        "refit_abs_difference": difference,
        "refit_matching_decimals": digits,
        "refit_max_abs_score_difference": float(
            np.max(np.abs(scores - predictions["score_independent_logistic"]))
        ),
        "refit_iterations": package["iterations"],
        "chosen_c": KEY04_C,
        "query_records": len(data["query_flags"]),
        "train_records": len(data["train_flags"]),
        "qualification_baseline_max_score_error": qualification["baseline_max_score_error"],
        "predictions_sha256": sha256_file(ROOT / KEY04 / "predictions.npz"),
        "result_sha256": sha256_file(ROOT / KEY04 / "result.json"),
    }
