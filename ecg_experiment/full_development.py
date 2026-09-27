"""Cohorts, labels, baselines and statistics for the Experiment 020 full-development readout."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .eda.ptbxl import diagnostic_classes, load_metadata, load_statements, superclass_label

ROOT = Path(__file__).resolve().parents[1]
UNION = ROOT / "data/processed/training_union_500hz_v1"
CLEAN = ROOT / "outputs/data_quality/clean_rerun_preflight_v1"
SINUS_RATE_CODES = ("SBRAD", "STACH")
PROBE_DEVICE = "CS100 3"


def ptb_table() -> pd.DataFrame:
    """
    One row per PTB-XL record with the fields the readout needs.

    Returns
    -------
    pd.DataFrame
        Indexed by ``record_id`` (``ptbxl:<ecg_id>``), with ``ecg_id``,
        ``patient_id`` (``ptbxl:<id>``), ``strat_fold``, ``age`` (NaN for the 300
        placeholder), ``male``, ``device`` (whitespace-normalized),
        ``standard`` label (1, 0 or NaN), ``sinus_rate`` (SBRAD or STACH listed)
        and ``filename_hr``.
    """
    meta = load_metadata()
    classes = diagnostic_classes(load_statements())
    columns = {
        "ecg_id": meta.index.astype(int),
        "patient_id": "ptbxl:" + meta["patient_id"].astype(str),
        "strat_fold": meta["strat_fold"].astype(int),
        "age": meta["age_capped"].astype(float),
        "male": (meta["sex"] == 0).astype(float),
        "device": meta["device"].str.split().str.join(" "),
        "standard": meta["scp_codes"].apply(lambda codes: superclass_label(codes, classes)),
        "sinus_rate": meta["scp_codes"].apply(lambda codes: any(code in codes for code in SINUS_RATE_CODES)),
        "filename_hr": meta["filename_hr"],
    }
    table = pd.DataFrame({name: np.asarray(values) for name, values in columns.items()},
                         index="ptbxl:" + meta.index.astype(str))
    table.index.name = "record_id"
    return table


def cohorts(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Training, original development and full development rows.

    Parameters
    ----------
    table : pd.DataFrame
        Output of ``ptb_table``.

    Returns
    -------
    dict[str, pd.DataFrame]
        ``train`` (clean union PTB training rows, with the project ``target``
        where one exists and ``limited`` membership), ``development`` (full
        development, with ``original`` and project ``target`` for original rows)
        and ``closed`` (fold-9 rows of calibration patients, never scored).

    Raises
    ------
    ValueError
        If partitions overlap or the historical counts changed.
    """
    references = pd.read_csv(CLEAN / "heldout_references.csv", dtype=str)
    full = pd.read_csv(CLEAN / "labels_fraction1.csv", dtype=str)
    limited = set(pd.read_csv(CLEAN / "labels_fraction0.1.csv", dtype=str)["record_id"])
    union = pd.read_csv(UNION / "train_manifest.csv", dtype=str, keep_default_na=False)

    train = table.loc[union.loc[union["source"] == "ptbxl", "record_id"]].copy()
    train["target"] = full.set_index("record_id")["target"].astype(float).reindex(train.index)
    train["limited"] = train.index.isin(limited)

    original = references[references["split"] == "development"].set_index("record_id")
    calibration_patients = set(references.loc[references["split"] == "calibration", "patient_id"])
    fold9 = table[table["strat_fold"] == 9]
    unassigned = fold9[~fold9.index.isin(references["record_id"])]
    in_calibration = unassigned["patient_id"].isin(calibration_patients)
    added, closed = unassigned[~in_calibration], unassigned[in_calibration]
    development = pd.concat([table.loc[original.index], added])
    development["original"] = development.index.isin(original.index)
    development["target"] = original["target"].astype(float).reindex(development.index)

    if len(train) != 17417 or train["target"].notna().sum() != 15359 or train["limited"].sum() != 1518:
        raise ValueError("Training selection changed")
    if len(original) != 1306 or original["patient_id"].nunique() != 1173:
        raise ValueError("Original development selection changed")
    if set(train["patient_id"]) & set(development["patient_id"]):
        raise ValueError("Training and development patients overlap")
    if set(development["patient_id"]) & calibration_patients:
        raise ValueError("A calibration patient entered development")
    if (development["strat_fold"] != 9).any():
        raise ValueError("Development must come from fold 9 only")
    return {"train": train, "development": development, "closed": closed}


def demographics(frame: pd.DataFrame, median_age: float) -> np.ndarray:
    """
    Age per decade, male and a missing-age indicator.

    Parameters
    ----------
    frame : pd.DataFrame
        Rows with ``age`` and ``male``.
    median_age : float
        Training median used to fill missing ages.

    Returns
    -------
    np.ndarray
        Float64 array of shape ``(rows, 3)``.
    """
    missing = frame["age"].isna().to_numpy(dtype=np.float64)
    age = frame["age"].fillna(median_age).to_numpy(dtype=np.float64) / 10
    return np.column_stack([age, frame["male"].to_numpy(dtype=np.float64), missing])


def fit_logistic(x: np.ndarray, y: np.ndarray) -> tuple[StandardScaler, LogisticRegression]:
    """
    Fit Experiment 018's fixed scaler and L2 logistic head on training rows.

    Parameters
    ----------
    x : np.ndarray
        Training inputs.
    y : np.ndarray
        Binary targets.

    Returns
    -------
    tuple[StandardScaler, LogisticRegression]
        Fitted scaler and classifier.

    Raises
    ------
    RuntimeError
        If the solver does not converge.
    """
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=0.01, penalty="l2", fit_intercept=True, solver="lbfgs", max_iter=5000,
                               tol=1e-8, class_weight=None, random_state=42)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(scaler.transform(x), y)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught) or model.n_iter_[0] >= 5000:
        raise RuntimeError("Logistic fit did not converge")
    return scaler, model


def predict(head: tuple[StandardScaler, LogisticRegression], x: np.ndarray) -> np.ndarray:
    """
    Positive-class probabilities of a fitted head.

    Parameters
    ----------
    head : tuple[StandardScaler, LogisticRegression]
        Output of ``fit_logistic``.
    x : np.ndarray
        Inputs.

    Returns
    -------
    np.ndarray
        One probability per row.
    """
    scaler, model = head
    return model.predict_proba(scaler.transform(x))[:, 1]


def patient_bootstrap(patients: np.ndarray, y: np.ndarray, first: np.ndarray, second: np.ndarray,
                      draws: int = 2000, seed: int = 20020) -> dict[str, float | int]:
    """
    Paired whole-patient bootstrap of an AUROC difference.

    Parameters
    ----------
    patients : np.ndarray
        Patient ID of each ECG.
    y : np.ndarray
        Binary targets.
    first, second : np.ndarray
        Scores of the two models on the same ECGs.
    draws : int
        Number of bootstrap draws.
    seed : int
        Random seed.

    Returns
    -------
    dict[str, float | int]
        Observed ``difference`` (first minus second), 95% interval and the
        number of invalid single-class draws.
    """
    unique, codes = np.unique(patients, return_inverse=True)
    members = [np.flatnonzero(codes == index) for index in range(len(unique))]
    rng = np.random.default_rng(seed)
    differences, invalid = [], 0
    for _ in range(draws):
        rows = np.concatenate([members[i] for i in rng.integers(0, len(unique), len(unique))])
        if len(np.unique(y[rows])) < 2:
            invalid += 1
            continue
        differences.append(roc_auc_score(y[rows], first[rows]) - roc_auc_score(y[rows], second[rows]))
    low, high = np.percentile(differences, [2.5, 97.5])
    return {"difference": float(roc_auc_score(y, first) - roc_auc_score(y, second)),
            "ci_low": float(low), "ci_high": float(high), "invalid_draws": invalid}
