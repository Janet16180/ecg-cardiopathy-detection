"""Common pieces of experiments 057 to 062 on the balanced dataset v2 (notebook 29).

Loading, the patient-grouped partition, the nested label budgets and the metrics that every model of
the line reports, so all of them are compared on the same ECGs with the same code.
Protocol: ``docs/experiment-057-062-protocolo-v2.md``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold

from .intervals import paired_auroc_difference, patient_groups, two_class_resamples

DATASET = Path("data/processed/balanceado_jovenes_v2")
FS = 500
CROP = 4800                      # 9.6 s: drops the almost flat end of PTB-XL
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
BUDGETS = (250, 500, 1000, 2000)
SEEDS = (0, 1, 2, 3, 4)
SPLIT_SEED = 42


def load(root: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Table and signals of the balanced dataset v2.

    Parameters
    ----------
    root : Path
        Repository root.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        One row per ECG indexed by ``fila`` with ``y`` (1 = con cardiopatía), ``grupo`` (patient ID, or
        the record itself in the Challenge datasets that have no patient ID) and list columns
        ``cardiopatias`` and ``cardiopatias_generales``; and the memory-mapped float16 signals
        ``(20000, 5000, 12)`` where row ``i`` is ECG ``i``.
    """
    ecgs = load_table(root)
    signals = np.load(root / DATASET / "senales.npy", mmap_mode="r")
    return ecgs, signals


def load_table(root: Path) -> pd.DataFrame:
    """
    Table of the balanced dataset v2, without reading the signals.

    Parameters
    ----------
    root : Path
        Repository root.

    Returns
    -------
    pd.DataFrame
        The table returned by ``load``.
    """
    ecgs = pd.read_csv(root / DATASET / "ecgs.csv", index_col="fila", dtype={"registro": str})
    for column in ("cardiopatias", "cardiopatias_generales"):
        ecgs[column] = ecgs[column].apply(lambda text: text.split("; ") if isinstance(text, str) else [])
    ecgs["y"] = (ecgs["clase"] == "con cardiopatía").astype(int)
    ecgs["grupo"] = ecgs["paciente"].fillna(ecgs["dataset"] + " " + ecgs["registro"])
    return ecgs


def partition(ecgs: pd.DataFrame, seed: int = SPLIT_SEED) -> pd.Series:
    """
    Train, validation and test split grouped by patient and stratified by class and dataset of origin.

    Ten ``StratifiedGroupKFold`` parts: parts 0 and 1 are the test set (20%), part 2 validation (10%)
    and the other seven training (70%).

    Parameters
    ----------
    ecgs : pd.DataFrame
        Output of ``load``.
    seed : int
        Seed of the shuffle.

    Returns
    -------
    pd.Series
        ``"train"``, ``"val"`` or ``"test"`` per ECG, aligned with ``ecgs``.
    """
    strata = ecgs["clase"] + " | " + ecgs["dataset"]
    split = pd.Series("train", index=ecgs.index)
    folds = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    for part, (_, rows) in enumerate(folds.split(ecgs, strata, ecgs["grupo"])):
        if part < 3:
            split.iloc[rows] = "test" if part < 2 else "val"
    return split


def budgets(train: pd.DataFrame, sizes: tuple[int, ...] = BUDGETS, seeds: tuple[int, ...] = SEEDS) -> pd.DataFrame:
    """
    Nested labelled subsets of the training set, half of each class.

    Parameters
    ----------
    train : pd.DataFrame
        Training ECGs with column ``y``.
    sizes : tuple[int, ...]
        Labelled ECGs of each budget.
    seeds : tuple[int, ...]
        One ordering of each class per seed; smaller budgets are prefixes of larger ones.

    Returns
    -------
    pd.DataFrame
        Boolean, one column ``"{size}_{seed}"`` per budget and seed, indexed like ``train``.
    """
    columns = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        order = {label: rng.permutation(train.index[train["y"] == label]) for label in (0, 1)}
        for size in sizes:
            chosen = np.concatenate([order[label][: size // 2] for label in (0, 1)])
            columns[f"{size}_{seed}"] = train.index.isin(chosen)
    return pd.DataFrame(columns, index=train.index)


def specificity_at(y: np.ndarray, score: np.ndarray, sensitivity: float) -> float:
    """
    Highest specificity among the thresholds that reach a sensitivity.

    Parameters
    ----------
    y : np.ndarray
        Binary targets.
    score : np.ndarray
        Higher means more likely positive.
    sensitivity : float
        Required sensitivity, between 0 and 1.

    Returns
    -------
    float
        Specificity.
    """
    fpr, tpr, _ = roc_curve(y, score)
    return float(1 - fpr[tpr >= sensitivity].min())


def summary(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """
    AUROC, AUPRC and specificity at 95% and 99% sensitivity.

    Parameters
    ----------
    y : np.ndarray
        Binary targets.
    score : np.ndarray
        Model scores.

    Returns
    -------
    dict[str, float]
        ``AUROC``, ``AUPRC``, ``esp_sens95`` and ``esp_sens99``.
    """
    return {
        "AUROC": float(roc_auc_score(y, score)),
        "AUPRC": float(average_precision_score(y, score)),
        "esp_sens95": specificity_at(y, score, 0.95),
        "esp_sens99": specificity_at(y, score, 0.99),
    }


def auroc_interval(y: np.ndarray, score: np.ndarray, groups: np.ndarray, draws: int = 1000, seed: int = 2026
                   ) -> tuple[float, float]:
    """
    95% interval of the AUROC by whole-group bootstrap.

    Parameters
    ----------
    y : np.ndarray
        Binary targets.
    score : np.ndarray
        Model scores.
    groups : np.ndarray
        Patient (or record) of each ECG.
    draws : int
        Bootstrap draws, including skipped single-class ones.
    seed : int
        Seed of ``numpy.random.default_rng``.

    Returns
    -------
    tuple[float, float]
        2.5 and 97.5 percentiles.
    """
    rng = np.random.default_rng(seed)
    values = [roc_auc_score(y[rows], score[rows])
              for rows in two_class_resamples(patient_groups(np.asarray(groups)), np.asarray(y), draws, rng)]
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def auroc_by_dataset(y: np.ndarray, score: np.ndarray, datasets: np.ndarray) -> pd.Series:
    """
    AUROC inside each dataset of origin that has both classes.

    Parameters
    ----------
    y : np.ndarray
        Binary targets.
    score : np.ndarray
        Model scores.
    datasets : np.ndarray
        Dataset of origin of each ECG.

    Returns
    -------
    pd.Series
        AUROC per dataset; datasets with a single class are left out.
    """
    frame = pd.DataFrame({"y": y, "score": score, "dataset": datasets})
    return frame.groupby("dataset").apply(
        lambda part: roc_auc_score(part["y"], part["score"]) if part["y"].nunique() == 2 else np.nan,
        include_groups=False).dropna().rename("AUROC")


def compare(y: np.ndarray, first: np.ndarray, second: np.ndarray, groups: np.ndarray, draws: int = 1000
            ) -> dict[str, float | int]:
    """
    Paired AUROC difference, first minus second, with a whole-group bootstrap interval.

    Parameters
    ----------
    y : np.ndarray
        Binary targets.
    first, second : np.ndarray
        Scores of the two models on the same ECGs.
    groups : np.ndarray
        Patient (or record) of each ECG.
    draws : int
        Bootstrap draws.

    Returns
    -------
    dict[str, float | int]
        Output of ``intervals.paired_auroc_difference``.
    """
    return paired_auroc_difference(np.asarray(groups), np.asarray(y), np.asarray(first), np.asarray(second),
                                   draws=draws)
