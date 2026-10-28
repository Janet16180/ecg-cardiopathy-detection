"""
ECGAD cardiologist localization benchmark (Jiang et al., MICCAI 2023) on PTB-XL fold 10.

The benchmark ships 400 PTB-XL ECGs (one record twice) without record IDs. The IDs were recovered
by signal matching (``data/raw/ecgad/ecgad_ptbxl_match.csv``); see ``docs/ecgad-benchmark-v1.md``.
Labels are 0/1 per sample and lead, aligned to the 500 Hz PTB-XL record.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ecg_experiment import ROOT

ECGAD_DIR = ROOT / "data/raw/ecgad"
SPLIT_PATH = ROOT / "docs/ecgad-split-v1.json"
PTBXL_DATABASE = ROOT / "data/raw/ptb-xl/1.0.3/ptbxl_database.csv"
SPLIT_SEED = 61061


def load_match() -> pd.DataFrame:
    """
    Benchmark index to PTB-XL record table, with patient IDs and fold.

    Returns
    -------
    pd.DataFrame
        One row per benchmark index: ``bench_index``, ``ecg_id``, ``similarity``, ``patient_id``,
        ``strat_fold``.
    """
    match = pd.read_csv(ECGAD_DIR / "ecgad_ptbxl_match.csv")
    meta = pd.read_csv(PTBXL_DATABASE, index_col="ecg_id")
    match["patient_id"] = meta.loc[match.ecg_id, "patient_id"].to_numpy().astype(int)
    match["strat_fold"] = meta.loc[match.ecg_id, "strat_fold"].to_numpy()
    return match


def load_labels() -> np.ndarray:
    """
    Point labels, shape (400, 5000, 12), memory-mapped float64 in {0, 1}.

    Returns
    -------
    np.ndarray
        Labels in benchmark order; leads in PTB-XL order (I, II, III, aVR, aVL, aVF, V1-V6).
    """
    return np.load(ECGAD_DIR / "benchmark_label.npy", mmap_mode="r")


def build_split(match: pd.DataFrame, labeled: np.ndarray, seed: int = SPLIT_SEED) -> pd.DataFrame:
    """
    Patient-level half split into ``development`` and ``holdout``, stratified by marks.

    Each benchmark row appears once. The duplicated record keeps only its first benchmark index;
    its second index gets partition ``duplicate``. Patients are stratified by whether any of their
    records carries a mark, shuffled with ``seed``, and alternately assigned.

    Parameters
    ----------
    match : pd.DataFrame
        Output of ``load_match``.
    labeled : np.ndarray
        Boolean per benchmark index, true if the record has any marked sample.
    seed : int
        Shuffle seed.

    Returns
    -------
    pd.DataFrame
        ``bench_index``, ``ecg_id``, ``patient_id``, ``labeled``, ``partition``.
    """
    table = match[["bench_index", "ecg_id", "patient_id"]].copy()
    table["labeled"] = labeled
    table["partition"] = ""
    first = ~table.ecg_id.duplicated(keep="first")
    table.loc[~first, "partition"] = "duplicate"
    patients = table[first].groupby("patient_id").labeled.any()
    rng = np.random.default_rng(seed)
    assignment = {}
    for flag in (False, True):
        ids = np.sort(patients.index[patients == flag].to_numpy())
        rng.shuffle(ids)
        for k, pid in enumerate(ids):
            assignment[pid] = "development" if k % 2 == 0 else "holdout"
    table.loc[first, "partition"] = table.loc[first, "patient_id"].map(assignment)
    return table


def load_split() -> pd.DataFrame:
    """
    Frozen split table from ``docs/ecgad-split-v1.json``.

    Returns
    -------
    pd.DataFrame
        ``bench_index``, ``ecg_id``, ``patient_id``, ``labeled``, ``partition``.
    """
    return pd.read_json(SPLIT_PATH, orient="split")
