"""
Lead-standardized noise reading for LocBench v1 (Experiment 061 amendment 3).

C1 compares raw lead scores of noisy and clean leads within a record. The noise annotations fall mostly
on limb leads, whose signals are small, so a map with a lead-level bias moves C1 away from 0.5 without
reacting to noise. ``C1_std`` first standardizes each lead score by the same lead's median and
interquartile range over the 1,400 ``evaluation`` clean normals, then takes the within-record AUROC
against the named noisy leads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ecg_experiment.locbench import DRAWS, SEED, _bootstrap, _mean, auroc
from ecg_experiment.locbench_data import load_partition
from ecg_experiment.waveforms import LEADS

LEAD_COLUMNS = [f"lead_{lead}" for lead in LEADS]


def standardized_noise_table(folder: Path) -> pd.DataFrame:
    """
    Per noise-set record ``C1_std`` from a method's saved LocBench tables.

    Parameters
    ----------
    folder : Path
        Tables folder written by ``locbench.save_tables``.

    Returns
    -------
    pd.DataFrame
        ``ecg_id``, ``patient_id``, ``C1_std`` for the noise-set records.
    """
    records = pd.read_csv(folder / "records.csv").set_index("ecg_id")
    control = pd.read_csv(folder / "noise_control.csv")
    clean = records.loc[control.loc[control["y"] == 0, "ecg_id"], LEAD_COLUMNS].to_numpy(np.float64)
    median = np.median(clean, axis=0)
    spread = np.subtract(*np.percentile(clean, [75, 25], axis=0))
    spread = np.where(spread > 0, spread, 1.0)
    noisy_leads = load_partition().set_index("ecg_id")["noisy_leads"]
    rows = []
    for row in control[control["y"] == 1].itertuples(index=False):
        z = (records.loc[row.ecg_id, LEAD_COLUMNS].to_numpy(np.float64) - median) / spread
        mask = np.isin(LEADS, noisy_leads.loc[row.ecg_id].split(";"))
        rows.append({"ecg_id": int(row.ecg_id), "patient_id": int(row.patient_id), "C1_std": auroc(mask, z)})
    return pd.DataFrame(rows)


def noise_summary(table: pd.DataFrame, draws: int = DRAWS, seed: int = SEED) -> dict[str, Any]:
    """
    Mean ``C1_std`` with a whole-patient bootstrap interval.

    Parameters
    ----------
    table : pd.DataFrame
        Output of ``standardized_noise_table``.
    draws, seed : int
        Bootstrap settings.

    Returns
    -------
    dict[str, Any]
        ``value``, ``ci_low``, ``ci_high``, ``records``.
    """
    values, _ = _bootstrap(table["patient_id"].to_numpy(), [_mean(table["C1_std"].to_numpy())], draws, seed)
    low, high = np.percentile(values[:, 0], [2.5, 97.5])
    return {
        "value": float(table["C1_std"].mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "records": len(table),
    }
