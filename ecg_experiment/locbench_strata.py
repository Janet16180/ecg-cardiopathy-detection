"""
QRS-stratified ECGAD reading for LocBench v1 (Experiment 061 amendment 2).

Experiment 061 found that a disease-blind QRS indicator (``qrs_prior``) is the best baseline on A2, because
the cardiologist marks concentrate on QRS complexes. This reading separates the two questions: within the QRS
complexes, does a map pick the marked complexes and leads (``A2_qrs``)? Outside them, does it find the marked
P, PR, ST and T regions (``A2_nonqrs``)? The QRS mask is the 059 hybrid QRS of ``qrs_prior`` (value >= 0.5).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ecg_experiment import ROOT
from ecg_experiment.ecgad import load_labels, load_split
from ecg_experiment.locbench import DRAWS, SEED, START, STOP, _bootstrap, _mean, auroc, load_tables

QRS_TABLES = ROOT / "outputs/experiment061_locbench_v1/tables/qrs_prior"
STRATA = ("A2_qrs", "A2_nonqrs")


def qrs_masks(folder: Path = QRS_TABLES) -> tuple[np.ndarray, np.ndarray]:
    """
    QRS masks of the ECGAD development records from the saved ``qrs_prior`` maps.

    Parameters
    ----------
    folder : Path
        ``qrs_prior`` tables folder of Experiment 061.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``ecg_id`` order and boolean masks (n, 12, 5000).
    """
    tables = load_tables(folder, maps=True)
    return tables["ecgad"]["ecg_id"].to_numpy(), np.asarray(tables["ecgad_maps"]) >= 0.5


def strata_table(maps: np.ndarray, ecg_ids: np.ndarray) -> pd.DataFrame:
    """
    Per-record ``A2_qrs`` and ``A2_nonqrs`` for the ECGAD development records.

    Parameters
    ----------
    maps : np.ndarray
        Maps (n, 12, 5000) of a method.
    ecg_ids : np.ndarray
        ``ecg_id`` of each map row; must contain every ECGAD development record.

    Returns
    -------
    pd.DataFrame
        ``ecg_id``, ``patient_id`` and the two strata (NaN where a stratum lacks marked or unmarked samples),
        in the ``qrs_prior`` table order.
    """
    order, masks = qrs_masks()
    split = load_split().set_index("ecg_id")
    labels = load_labels()
    position = {int(ecg_id): k for k, ecg_id in enumerate(ecg_ids)}
    rows = []
    for k, ecg_id in enumerate(order):
        marks = np.asarray(labels[int(split.loc[ecg_id, "bench_index"])]).T.astype(bool)[:, START:STOP]
        values = np.asarray(maps[position[int(ecg_id)]], dtype=np.float64)[:, START:STOP]
        qrs = masks[k][:, START:STOP]
        row = {"ecg_id": int(ecg_id), "patient_id": int(split.loc[ecg_id, "patient_id"])}
        for name, inside in (("A2_qrs", qrs), ("A2_nonqrs", ~qrs)):
            y = marks[inside]
            row[name] = auroc(y, values[inside]) if y.any() and not y.all() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def strata_summary(table: pd.DataFrame, draws: int = DRAWS, seed: int = SEED) -> dict[str, Any]:
    """
    Mean of each stratum with a whole-patient bootstrap interval.

    Parameters
    ----------
    table : pd.DataFrame
        Output of ``strata_table``.
    draws, seed : int
        Bootstrap settings (LocBench defaults).

    Returns
    -------
    dict[str, Any]
        Per stratum: ``value``, ``ci_low``, ``ci_high``, ``records``.
    """
    summary = {}
    for name in STRATA:
        frame = table[table[name].notna()]
        values, _ = _bootstrap(frame["patient_id"].to_numpy(), [_mean(frame[name].to_numpy())], draws, seed)
        low, high = np.percentile(values[:, 0], [2.5, 97.5])
        summary[name] = {
            "value": float(frame[name].mean()),
            "ci_low": float(low),
            "ci_high": float(high),
            "records": len(frame),
        }
    return summary


def strata_paired(
    table_a: pd.DataFrame, table_b: pd.DataFrame, name: str, draws: int = DRAWS, seed: int = SEED
) -> dict[str, Any]:
    """
    Stratum mean of method A minus method B on records where both are defined, paired patient bootstrap.

    Parameters
    ----------
    table_a, table_b : pd.DataFrame
        Outputs of ``strata_table`` for the two methods.
    name : str
        ``A2_qrs`` or ``A2_nonqrs``.
    draws, seed : int
        Bootstrap settings.

    Returns
    -------
    dict[str, Any]
        ``value``, ``ci_low``, ``ci_high``, ``a``, ``b``, ``records``.
    """
    merged = table_a.merge(table_b, on=["ecg_id", "patient_id"], suffixes=("_a", "_b"))
    merged = merged[merged[f"{name}_a"].notna() & merged[f"{name}_b"].notna()]
    a, b = merged[f"{name}_a"].to_numpy(), merged[f"{name}_b"].to_numpy()
    values, _ = _bootstrap(merged["patient_id"].to_numpy(), [_mean(a), _mean(b)], draws, seed)
    low, high = np.percentile(values[:, 0] - values[:, 1], [2.5, 97.5])
    return {
        "value": float(a.mean() - b.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "a": float(a.mean()),
        "b": float(b.mean()),
        "records": len(merged),
    }
