"""
The dry run of MLC-04 on the Challenge validation (calibration) groups: what the runner must reproduce.

Step 1 of ``docs/mlc04-final-test-protocol-2026-11-11.md``. The role ``validation`` is run through the same
path as the test groups, and its outputs are compared with what the frozen builds stored:

- the MLC-01 scan and v4 manifest rows, cell by cell (as text);
- the waveform cache, bit by bit, against the MLC-02 cache;
- the Part A measures and flags, method B, the window summaries, method A and the analysis table of MLC-03;
- the v5 manifest rows, cell by cell, and the flow of rules C1-C12;
- the re-extracted features against the stored MLC-02 features (largest absolute difference at most
  ``FEATURE_TOLERANCE``);
- the stored validation outputs of every frozen model, from the stored features (at most
  ``PREDICTION_TOLERANCE``, relative for K6) and from the re-extracted ones (reported, at most
  ``REEXTRACTED_TOLERANCE``);
- the stored validation metric of P0-v5, from the stored features exactly and with the re-extracted Challenge
  rows within ``FEATURE_TOLERANCE``;
- the frozen thresholds, refitted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ecg_experiment import mlc03_dataset_v5 as mv
from ecg_experiment.files import read_json
from ecg_experiment.mlc02_data import CHALLENGE_SOURCES, CONDITIONS
from ecg_experiment.mlc02_evaluation import selection_metric
from ecg_experiment.mlc02_feature_data import PUBLISHED, load_split
from ecg_experiment.mlc02_waveforms import CACHE, load_cache, windows
from ecg_experiment.mlc04_data import OUTPUT, WORK, cell_mismatches, read_text
from ecg_experiment.mlc04_evaluation import VALIDATION_METRIC, VALIDATION_MINIMUM, source_metric
from ecg_experiment.mlc04_models import (
    CANDIDATE_MODELS,
    FEATURE_SETS,
    MODELS,
    fit_thresholds,
    load_features,
    load_models,
    predict_all,
    stored_validation,
    v5_validation,
)

SIGNAL = mv.QUALITY / "signal"
FEATURE_TOLERANCE = 1e-4
PREDICTION_TOLERANCE = 1e-12
REEXTRACTED_TOLERANCE = 1e-3
BLOCK = 512


def compare_rows(mine: pd.DataFrame, stored: pd.DataFrame, drop: tuple[str, ...] = ()) -> dict[str, Any]:
    """
    Compare a table of the dry run with the stored table on the same records and common columns.

    Parameters
    ----------
    mine : pd.DataFrame
        Dry-run table with ``record_key``.
    stored : pd.DataFrame
        Stored table with ``record_key`` (a superset of the records).
    drop : tuple[str, ...]
        Columns left out of the comparison.

    Returns
    -------
    dict[str, Any]
        ``records``, ``missing`` (records absent from the stored table), ``mismatches`` (column to differing
        cells), ``only_mine`` and ``only_stored`` (columns), and ``passed``.
    """
    stored = stored.drop_duplicates("record_key").set_index("record_key")
    missing = int((~mine["record_key"].isin(stored.index)).sum())
    if missing:
        return {"records": len(mine), "missing": missing, "passed": False}
    aligned = stored.loc[mine["record_key"]].reset_index()
    common = [name for name in mine.columns if name in aligned.columns and name not in drop]
    mismatches = cell_mismatches(mine[common].reset_index(drop=True), aligned[common])
    return {
        "records": len(mine),
        "missing": 0,
        "mismatches": mismatches,
        "only_mine": sorted(set(mine.columns) - set(aligned.columns) - set(drop)),
        "only_stored": sorted(set(aligned.columns) - set(mine.columns) - set(drop)),
        "passed": not mismatches,
    }


def manifest_check(name: str, stored: Path) -> dict[str, Any]:
    """
    Compare the dry run's manifest rows with the stored manifest, as text, records in the same order.

    Parameters
    ----------
    name : str
        ``manifest_v4.csv`` or ``manifest_v5.csv``.
    stored : Path
        The stored build's ``manifest.csv``.

    Returns
    -------
    dict[str, Any]
        ``records``, ``same_records_in_order``, ``columns``, ``cells``, ``mismatches`` and ``passed``.
    """
    mine = read_text(OUTPUT / "validation" / name)
    every = read_text(stored)
    expected = every[(every["role"] == "validation") & every["source"].isin(CHALLENGE_SOURCES)]
    expected = expected.reset_index(drop=True)
    same = mine["record_key"].tolist() == expected["record_key"].tolist()
    mismatches = cell_mismatches(mine, expected) if same else {"record_keys": -1}
    return {
        "records": len(mine),
        "stored_records": len(expected),
        "same_records_in_order": same,
        "columns": len(mine.columns),
        "cells": int(mine.size),
        "mismatches": mismatches,
        "passed": same and not mismatches and list(mine.columns) == list(expected.columns),
    }


def cache_check(directory: Path = WORK / "validation" / "cache", reference: Path = CACHE) -> dict[str, Any]:
    """
    Compare every cached window of the dry run bit by bit with the MLC-02 cache.

    Parameters
    ----------
    directory : Path
        The dry run's cache.
    reference : Path
        The MLC-02 cache (``index_all.csv``).

    Returns
    -------
    dict[str, Any]
        ``records``, ``differing`` and ``passed``.
    """
    index, blocks = load_cache(directory, "index.csv")
    old_index, old_blocks = load_cache(reference, "index_all.csv")
    old = pd.Series(old_index["position"].to_numpy(), index=old_index["record_key"].to_numpy())
    differing = 0
    for start in range(0, len(index), BLOCK):
        part = index.iloc[start : start + BLOCK]
        mine = windows(index, blocks, part["position"].to_numpy())
        stored = windows(old_index, old_blocks, old.loc[part["record_key"]].to_numpy())
        differing += int(
            (mine.view(np.uint32) != stored.view(np.uint32)).reshape(len(part), -1).any(axis=1).sum()
        )
    return {"records": len(index), "differing": differing, "passed": differing == 0}


def stored_rows(directory: Path, keys: np.ndarray, name: str) -> np.ndarray:
    """
    Rows of a stored MLC-02 feature array for some record keys.

    Parameters
    ----------
    directory : Path
        Stored feature directory with ``rows.csv``.
    keys : np.ndarray
        Record keys.
    name : str
        Feature set.

    Returns
    -------
    np.ndarray
        Float64 rows aligned with ``keys``.
    """
    rows = pd.read_csv(directory / "rows.csv", usecols=["record_key"], dtype=str)
    position = pd.Series(np.arange(len(rows)), index=rows["record_key"].to_numpy())
    array = np.load(directory / f"{name}.npy", mmap_mode="r")
    return np.asarray(array[position.loc[keys].to_numpy()], dtype=np.float64)


def feature_difference(mine: np.ndarray, stored: np.ndarray) -> dict[str, Any]:
    """
    Differences between re-extracted and stored feature rows.

    Parameters
    ----------
    mine, stored : np.ndarray
        Aligned ``[records, width]`` arrays.

    Returns
    -------
    dict[str, Any]
        ``max_abs``, ``max_rel`` (per record: largest absolute difference over largest stored absolute value),
        ``records_above_1e-4_rel``, ``bit_identical_records``, ``same_missing`` and ``passed``.
    """
    same_missing = bool(np.array_equal(np.isnan(mine), np.isnan(stored)))
    difference = np.abs(np.nan_to_num(mine) - np.nan_to_num(stored))
    per_record = difference.max(axis=1)
    scale = np.nanmax(np.abs(stored), axis=1)
    relative = np.divide(per_record, scale, out=np.zeros_like(per_record), where=scale > 0)
    identical = (np.nan_to_num(mine, nan=1e300) == np.nan_to_num(stored, nan=1e300)).all(axis=1)
    max_abs = float(per_record.max()) if len(per_record) else 0.0
    return {
        "records": len(mine),
        "max_abs": max_abs,
        "max_rel": float(relative.max()) if len(relative) else 0.0,
        "records_above_1e-4_rel": int((relative > 1e-4).sum()),
        "bit_identical_records": int(identical.sum()),
        "same_missing": same_missing,
        "passed": same_missing and max_abs <= FEATURE_TOLERANCE,
    }


def features_check() -> dict[str, Any]:
    """
    Compare the dry run's re-extracted features with the stored MLC-02 features.

    Returns
    -------
    dict[str, Any]
        One ``feature_difference`` per feature set.
    """
    rows = pd.read_csv(OUTPUT / "validation" / "features" / "rows.csv", dtype={"record_key": str})
    keys = rows["record_key"].to_numpy()
    mine = load_features("validation", keys)
    return {name: feature_difference(mine[name], stored_rows(PUBLISHED, keys, name)) for name in FEATURE_SETS}


def output_difference(name: str, found: np.ndarray, expected: np.ndarray, tolerance: float) -> dict[str, Any]:
    """
    Largest difference between a model's outputs and its stored validation outputs.

    Parameters
    ----------
    name : str
        Model id; K6 scores are compared relative to the largest stored absolute score.
    found, expected : np.ndarray
        Aligned ``[records, 14]`` outputs.
    tolerance : float
        Largest accepted difference.

    Returns
    -------
    dict[str, Any]
        ``records``, ``max_abs``, ``max_rel`` (K6), and ``passed``.
    """
    difference = float(np.abs(found - expected).max())
    relative = difference / float(np.abs(expected).max()) if name == "K6" else difference
    return {
        "records": len(found),
        "max_abs": difference,
        "max_rel": relative,
        "passed": relative <= tolerance,
    }


def stored_feature_outputs(models: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """
    Score every frozen model on the stored features of all v4 validation records.

    Parameters
    ----------
    models : dict[str, Any]
        Output of ``load_models``.

    Returns
    -------
    tuple[dict[str, np.ndarray], dict[str, Any]]
        Model to outputs, and ``data`` (``load_split`` arrays of v4 validation).
    """
    data, features = load_split(("validation",), FEATURE_SETS, directory=PUBLISHED)
    return predict_all(models, features, data["source"]), data


def model_keys(name: str, v4_keys: np.ndarray, v5_keys: np.ndarray) -> np.ndarray:
    """
    Select the records a model's stored validation file holds.

    Parameters
    ----------
    name : str
        Model id.
    v4_keys, v5_keys : np.ndarray
        v4-included and v5-included validation keys.

    Returns
    -------
    np.ndarray
        ``v4_keys`` for the v4-trained models, ``v5_keys`` otherwise.
    """
    return v4_keys if name.endswith("-v4") else v5_keys


def outputs_check(
    models: dict[str, Any], stored: dict[str, np.ndarray], data: dict[str, np.ndarray]
) -> dict[str, Any]:
    """
    Compare every model's outputs with its stored validation outputs, from stored and re-extracted features.

    Parameters
    ----------
    models : dict[str, Any]
        Output of ``load_models``.
    stored : dict[str, np.ndarray]
        Outputs on the stored features of all v4 validation records.
    data : dict[str, np.ndarray]
        Their ``load_split`` arrays.

    Returns
    -------
    dict[str, Any]
        ``stored_features`` and ``reextracted_features``: one ``output_difference`` per model, and the
        re-extracted outputs on the Challenge records (``outputs``, ``keys``).
    """
    position = pd.Series(np.arange(len(data["record_key"])), index=data["record_key"])
    v5_keys = v5_validation()["record_key"]
    from_stored = {}
    for name in MODELS:
        keys = model_keys(name, data["record_key"], v5_keys)
        found = stored[name][position.loc[keys].to_numpy()]
        from_stored[name] = output_difference(
            name, found, stored_validation(name, keys), PREDICTION_TOLERANCE
        )
    mine4 = pd.read_csv(
        OUTPUT / "validation" / "manifest_v4.csv", usecols=["record_key", "source", "included"], dtype=str
    )
    mine4 = mine4[mine4["included"] == "True"]
    features = load_features("validation", mine4["record_key"].to_numpy())
    outputs = predict_all(models, features, mine4["source"].to_numpy())
    mine_position = pd.Series(np.arange(len(mine4)), index=mine4["record_key"].to_numpy())
    challenge5 = np.array([key for key in v5_keys if key in mine_position.index])
    from_mine = {}
    for name in MODELS:
        keys = model_keys(name, mine4["record_key"].to_numpy(), challenge5)
        found = outputs[name][mine_position.loc[keys].to_numpy()]
        from_mine[name] = output_difference(name, found, stored_validation(name, keys), REEXTRACTED_TOLERANCE)
    return {
        "stored_features": from_stored,
        "reextracted_features": from_mine,
        "outputs": outputs,
        "keys": mine4["record_key"].to_numpy(),
    }


def metric_check(
    stored: dict[str, np.ndarray], data: dict[str, np.ndarray], mine: dict[str, Any]
) -> dict[str, Any]:
    """
    Reproduce the stored validation metric of P0-v5, and the agreement of ``source_metric``.

    Parameters
    ----------
    stored : dict[str, np.ndarray]
        Outputs on the stored features of all v4 validation records.
    data : dict[str, np.ndarray]
        Their ``load_split`` arrays.
    mine : dict[str, Any]
        Output of ``outputs_check``.

    Returns
    -------
    dict[str, Any]
        The metric from the stored features and with the re-extracted Challenge rows, their differences
        from the stored value, and ``source_metric`` against ``selection_metric`` on the Challenge sources.
    """
    v5 = v5_validation()
    position = pd.Series(np.arange(len(data["record_key"])), index=data["record_key"])
    probability = stored["P0-v5"][position.loc[v5["record_key"]].to_numpy()]
    exact = selection_metric(v5["labels"], probability, v5["source"])["selection_metric"]
    mine_position = pd.Series(np.arange(len(mine["keys"])), index=mine["keys"])
    replaced = probability.copy()
    challenge = np.isin(v5["source"], CHALLENGE_SOURCES)
    replaced[challenge] = mine["outputs"]["P0-v5"][mine_position.loc[v5["record_key"][challenge]].to_numpy()]
    mixed = selection_metric(v5["labels"], replaced, v5["source"])["selection_metric"]
    labels, sources = v5["labels"][challenge], v5["source"][challenge]
    agreement = (
        source_metric(labels, replaced[challenge], sources, VALIDATION_MINIMUM)["metric"]
        - selection_metric(labels, replaced[challenge], sources)["selection_metric"]
    )
    return {
        "stored_value": VALIDATION_METRIC,
        "from_stored_features": exact,
        "difference_stored_features": exact - VALIDATION_METRIC,
        "with_reextracted_challenge_rows": mixed,
        "difference_reextracted": mixed - VALIDATION_METRIC,
        "source_metric_minus_selection_metric": agreement,
        "passed": exact == VALIDATION_METRIC
        and abs(mixed - VALIDATION_METRIC) <= FEATURE_TOLERANCE
        and agreement == 0.0,
    }


def measure_checks(parts: dict[str, Any]) -> dict[str, Any]:
    """
    Compare the dry run's MLC-03 inputs with the stored Part A and Part B tables.

    Parameters
    ----------
    parts : dict[str, Any]
        ``records``, ``a``, ``b``, ``table``, ``flags`` and ``windows`` of the validation role.

    Returns
    -------
    dict[str, Any]
        One ``compare_rows`` per table.
    """
    target = OUTPUT / "validation"
    measures = pd.read_parquet(target / "part_a_measures.parquet")
    keys = parts["records"]["record_key"].to_numpy()
    return {
        "scan": compare_rows(
            pd.read_parquet(target / "scan.parquet"),
            pd.read_parquet(mv.V4 / "waveform_scan.parquet"),
            drop=("worker_max_rss_kib",),
        ),
        "part_a_measures": compare_rows(
            measures, pd.read_parquet(SIGNAL / "measures_train_validation.parquet"), drop=("position",)
        ),
        "part_a_flags": compare_rows(
            parts["flags"], pd.read_parquet(SIGNAL / "flags_train_validation.parquet")
        ),
        "method_b": compare_rows(parts["b"], pd.read_parquet(mv.MEASUREMENTS_B)),
        "windows": compare_rows(parts["windows"], pd.read_csv(mv.WINDOWS, dtype={"record_key": str})),
        "method_a_against_stored_features": compare_rows(parts["a"], mv.measurements_a(keys)),
        "analysis": compare_rows(parts["table"].drop(columns=["codes"]), pd.read_parquet(mv.ANALYSIS)),
    }


def flow_check() -> dict[str, Any]:
    """
    Compare the dry run's rule flow with the v5 build's flow for the Challenge validation sources.

    Returns
    -------
    dict[str, Any]
        ``rows``, ``mismatches`` and ``passed``.
    """
    mine = pd.read_csv(OUTPUT / "validation" / "rule_flow.csv", keep_default_na=False, dtype=str)
    every = pd.read_csv(mv.V5 / "rule_flow.csv", keep_default_na=False, dtype=str)
    stored = every[(every["role"] == "validation") & every["source"].isin(CHALLENGE_SOURCES)]
    keys = ["rule", "condition", "source"]
    stored = stored.set_index(keys).loc[pd.MultiIndex.from_frame(mine[keys])].reset_index()[mine.columns]
    mismatches = cell_mismatches(mine, stored)
    return {"rows": len(mine), "mismatches": mismatches, "passed": not mismatches}


def thresholds_check(path: Path) -> dict[str, Any]:
    """
    Refit the thresholds and compare them with the frozen file.

    Parameters
    ----------
    path : Path
        Frozen ``thresholds.json``.

    Returns
    -------
    dict[str, Any]
        ``equal`` and ``passed``.
    """
    frozen = read_json(path)["thresholds"]
    refit = fit_thresholds()["thresholds"]
    equal = all(frozen[name] == refit[name] for name in CONDITIONS)
    return {"equal": equal, "passed": equal}


def run_checks(parts: dict[str, Any], thresholds: Path) -> dict[str, Any]:
    """
    Every reproduction of the dry run, with the names of the failed ones.

    Parameters
    ----------
    parts : dict[str, Any]
        Tables of the validation role (``role_tables`` of the runner).
    thresholds : Path
        Frozen ``thresholds.json``.

    Returns
    -------
    dict[str, Any]
        One entry per check, ``failed`` (names) and the candidate models checked.
    """
    models = load_models()
    stored, data = stored_feature_outputs(models)
    outputs = outputs_check(models, stored, data)
    checks = {
        "manifest_v4": manifest_check("manifest_v4.csv", mv.V4 / "manifest.csv"),
        "manifest_v5": manifest_check("manifest_v5.csv", mv.V5 / "manifest.csv"),
        "cache": cache_check(),
        "rule_flow": flow_check(),
        **measure_checks(parts),
        "features": features_check(),
        "metric": metric_check(stored, data, outputs),
        "thresholds": thresholds_check(thresholds),
    }
    checks["outputs_stored_features"] = outputs["stored_features"]
    checks["outputs_reextracted_features"] = outputs["reextracted_features"]
    failed = [
        name
        for name, value in checks.items()
        if isinstance(value.get("passed"), bool) and not value["passed"]
    ]
    failed += [f"features:{name}" for name, value in checks["features"].items() if not value["passed"]]
    for group in ("outputs_stored_features", "outputs_reextracted_features"):
        failed += [f"{group}:{name}" for name, value in checks[group].items() if not value["passed"]]
    return {**checks, "failed": failed, "models": [*CANDIDATE_MODELS, "K6"]}
