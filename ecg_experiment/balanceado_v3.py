"""Balanced dataset v3 (notebook 43): v2 without the ECGs that have a cardiopathy outside the 43 kept ones.

The partition, label budgets and metrics are those of ``balanceado_v2``; only the table changes. v3 has no
``cardiopatias_generales`` column: the general diagnoses were the assistant's own grouping, without sources.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATASET = Path("data/processed/balanceado_jovenes_v3")


def load_table(root: Path) -> pd.DataFrame:
    """
    Table of the balanced dataset v3, without reading the signals.

    Parameters
    ----------
    root : Path
        Repository root.

    Returns
    -------
    pd.DataFrame
        One row per ECG indexed by ``fila``, with ``cardiopatias`` as a list, ``y`` (1 = con cardiopatía) and
        ``grupo`` (patient ID, or the record itself in the Challenge datasets that have no patient ID).
    """
    ecgs = pd.read_csv(root / DATASET / "ecgs.csv", index_col="fila", dtype={"registro": str})
    ecgs["cardiopatias"] = ecgs["cardiopatias"].apply(lambda text: text.split("; ") if isinstance(text, str) else [])
    ecgs["y"] = (ecgs["clase"] == "con cardiopatía").astype(int)
    ecgs["grupo"] = ecgs["paciente"].fillna(ecgs["dataset"] + " " + ecgs["registro"])
    return ecgs


def load(root: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Table and memory-mapped float16 signals ``(20000, 5000, 12)``; row ``i`` of the signals is ECG ``i``.

    Parameters
    ----------
    root : Path
        Repository root.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        Output of ``load_table`` and the signals.
    """
    return load_table(root), np.load(root / DATASET / "senales.npy", mmap_mode="r")
