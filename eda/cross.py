"""Put the per-record summaries of all sources into one table and compare them."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from eda.challenge import challenge_summary, load_headers
from eda.code15 import code15_summary, load_exams
from eda.mimic import load_records, mimic_summary
from eda.ptbxl import load_metadata
from eda.ptbxl_signals import ptbxl_summary

SIGNAL_COLUMNS = ["duration_s", "std", "peak_abs_mv", "baseline_fraction", "high_frequency_fraction",
                  "powerline_50_fraction", "powerline_60_fraction", "heart_rate", "zero_fraction",
                  "flat_leads", "constant_leads", "large_leads", "nan_leads", "einthoven_residual"]
FINGERPRINT_COLUMNS = ["std", "baseline_fraction", "high_frequency_fraction", "powerline_50_fraction",
                       "powerline_60_fraction", "zero_fraction"]


def combined_summary() -> pd.DataFrame:
    """
    One row per record across all six sources, with age and sex where known.

    Returns
    -------
    pd.DataFrame
        ``SIGNAL_COLUMNS`` plus ``source``, ``age`` and ``male``.
    """
    ptbxl = ptbxl_summary(load_metadata())
    ptbxl = ptbxl.assign(source="ptbxl", age=ptbxl["age"].where(ptbxl["age"] < 300),
                         male=ptbxl["sex"] == 0)
    challenge = challenge_summary(load_headers())
    challenge = challenge.assign(age=challenge["age_years"].where(challenge["age_years"] >= 0),
                                 male=challenge["sex"].eq("Male"))
    mimic = mimic_summary(load_records()).assign(source="mimic", age=np.nan, male=np.nan)
    exams = load_exams()
    code15 = code15_summary().join(exams[["age", "is_male"]], on="exam_id")
    code15 = code15.assign(source="code15", male=code15["is_male"])
    columns = [*SIGNAL_COLUMNS, "source", "age", "male"]
    sources = (("ptbxl", ptbxl), ("challenge", challenge), ("mimic", mimic), ("code15", code15))
    parts = [frame[columns].rename(index=lambda value, name=name: f"{name}:{value}")
             for name, frame in sources]
    return pd.concat(parts)


def source_fingerprint(table: pd.DataFrame, per_source: int = 5000,
                       seed: int = 0) -> tuple[float, pd.DataFrame]:
    """
    How well the source can be predicted from amplitude and noise statistics alone.

    A balanced sample per source is split 70/30; a gradient-boosted classifier
    sees only ``FINGERPRINT_COLUMNS``, none of which describe cardiac content.

    Parameters
    ----------
    table : pd.DataFrame
        Output of ``combined_summary``.
    per_source : int
        Records sampled from each source.
    seed : int
        Sampling and split seed.

    Returns
    -------
    tuple[float, pd.DataFrame]
        Balanced accuracy on the held-out part (chance is one over the number
        of sources), and the row-normalized confusion matrix.
    """
    sample = table.groupby("source").sample(per_source, random_state=seed, replace=False)
    train, test = train_test_split(sample, test_size=0.3, random_state=seed, stratify=sample["source"])
    model = HistGradientBoostingClassifier(random_state=seed)
    model.fit(train[FINGERPRINT_COLUMNS], train["source"])
    predicted = model.predict(test[FINGERPRINT_COLUMNS])
    labels = sorted(sample["source"].unique())
    matrix = confusion_matrix(test["source"], predicted, labels=labels, normalize="true")
    accuracy = balanced_accuracy_score(test["source"], predicted)
    return accuracy, pd.DataFrame(matrix, index=labels, columns=labels)
