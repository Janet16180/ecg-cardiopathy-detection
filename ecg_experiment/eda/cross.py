"""Put the per-record summaries of all sources into one table and compare them."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from ecg_experiment.eda.challenge import challenge_summary, load_headers
from ecg_experiment.eda.code15 import code15_summary, load_exams
from ecg_experiment.eda.mimic import load_machine_measurements, load_records, mimic_summary
from ecg_experiment.eda.ptbxl import load_metadata
from ecg_experiment.eda.ptbxl_signals import ptbxl_summary

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


CONDITIONS = ["atrial fibrillation", "right bundle branch block", "left bundle branch block",
              "1st degree AV block", "sinus bradycardia", "sinus tachycardia"]
PTBXL_CODES = {"atrial fibrillation": ["AFIB"], "right bundle branch block": ["CRBBB"],
               "left bundle branch block": ["CLBBB"], "1st degree AV block": ["1AVB"],
               "sinus bradycardia": ["SBRAD"], "sinus tachycardia": ["STACH"]}
# The Challenge 2021 scores 713427006 as 59118001 and 733534002 as 164909002 (dx_mapping_scored.csv).
SNOMED_CODES = {"atrial fibrillation": ["164889003"], "right bundle branch block": ["59118001", "713427006"],
                "left bundle branch block": ["164909002", "733534002"], "1st degree AV block": ["270492004"],
                "sinus bradycardia": ["426177001"], "sinus tachycardia": ["427084000"]}
CODE15_COLUMNS = {"atrial fibrillation": "AF", "right bundle branch block": "RBBB",
                  "left bundle branch block": "LBBB", "1st degree AV block": "1dAVb",
                  "sinus bradycardia": "SB", "sinus tachycardia": "ST"}
MIMIC_PATTERNS = {"atrial fibrillation": r"atrial fibrillation",
                  "right bundle branch block": r"(?<!incomplete )right bundle branch block",
                  "left bundle branch block": r"(?<!incomplete )left bundle branch block",
                  "1st degree AV block": r"(?<!borderline )(?:1st|first) degree a-v block",
                  "sinus bradycardia": r"sinus bradycardia", "sinus tachycardia": r"sinus tachycardia"}
NOT_LABELED = {"cpsc_2018": ["sinus bradycardia", "sinus tachycardia"]}


def _any_listed(listed: pd.Series, codes: dict[str, list[str]]) -> pd.DataFrame:
    """
    Mark which conditions appear in each record's list of codes.

    Parameters
    ----------
    listed : pd.Series
        Codes of each record, as a list or dict keyed by code.
    codes : dict[str, list[str]]
        Codes that count as each condition.

    Returns
    -------
    pd.DataFrame
        One float column (1.0 or 0.0) per condition.
    """
    flags = {}
    for name, wanted in codes.items():
        flags[name] = listed.apply(lambda found, wanted=wanted: float(bool(set(wanted) & set(found))))
    return pd.DataFrame(flags)


def condition_flags() -> pd.DataFrame:
    """
    Six common findings on one vocabulary across all sources.

    PTB-XL uses its SCP codes, the Challenge sources their SNOMED codes, CODE-15
    its six flags and MIMIC its machine report text (all 800,035 studies). The
    sources were annotated differently (cardiologists, other sites' readers, an
    automatic system), so prevalences are comparable only approximately.
    Findings a source never annotated are NaN rather than 0.

    Returns
    -------
    pd.DataFrame
        One row per record with ``source``, ``age``, ``male`` and one float
        column (1, 0 or NaN) per condition.
    """
    ptbxl = load_metadata()
    ptbxl_flags = _any_listed(ptbxl["scp_codes"], PTBXL_CODES).set_index(ptbxl.index)
    ptbxl_flags = ptbxl_flags.assign(source="ptbxl", age=ptbxl["age_capped"], male=ptbxl["sex"].eq(0))
    headers = load_headers()
    challenge = _any_listed(headers["dx_codes"], SNOMED_CODES).set_index(headers.index)
    challenge = challenge.assign(source=headers["source"], male=headers["sex"].eq("Male"),
                                 age=headers["age_years"].where(headers["age_years"] >= 0))
    for source, missing in NOT_LABELED.items():
        challenge.loc[challenge["source"] == source, missing] = np.nan
    exams = load_exams()
    code15 = exams[list(CODE15_COLUMNS.values())].astype(float)
    code15.columns = list(CODE15_COLUMNS)
    code15 = code15.assign(source="code15", age=exams["age"], male=exams["is_male"])
    reports = load_machine_measurements()["report"].str.lower()
    mimic = pd.DataFrame({name: reports.str.contains(pattern, regex=True).astype(float)
                          for name, pattern in MIMIC_PATTERNS.items()})
    mimic = mimic.assign(source="mimic", age=np.nan, male=np.nan)
    return pd.concat([ptbxl_flags, challenge, code15, mimic], ignore_index=True)
