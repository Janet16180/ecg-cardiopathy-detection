"""Read the quality receipt of the clean cohorts and compare it with earlier cohorts."""

import json

import pandas as pd

from ecg_experiment.clean_cohorts import ROOT, candidate_pool
from ecg_experiment.ecg_quality import EXCLUSION_REASONS, REVIEW_FLAGS

QUALITY_DIR = ROOT / "outputs/data_quality/clean_cohorts_v1"
SIZES = ("25k", "50k", "100k")


def load_receipt() -> dict[str, object]:
    """
    Load the aggregate receipt written by ``scripts.data.build_clean_cohorts``.

    Returns
    -------
    dict[str, object]
        Counts by source, quality policy and cohort composition.
    """
    return json.loads((QUALITY_DIR / "receipt.json").read_text())


def quality_rows() -> pd.DataFrame:
    """
    Per-record quality results joined with each record's storage pointer.

    Returns
    -------
    pd.DataFrame
        One row per labeled or candidate record, with one boolean column per
        exclusion reason and review flag, plus ``excluded`` and ``labeled``.
    """
    labeled, candidates, _ = candidate_pool()
    rows = pd.concat([labeled.assign(labeled=True), candidates.assign(labeled=False)], ignore_index=True)
    scores = pd.read_csv(QUALITY_DIR / "record_quality.csv.gz", dtype=str, keep_default_na=False)
    rows = rows.merge(scores[["record_id", "exclusion", "review_flags"]], on="record_id", validate="1:1")
    for name in EXCLUSION_REASONS:
        rows[name] = rows["exclusion"].str.split(";").apply(lambda values, name=name: name in values)
    for name in REVIEW_FLAGS:
        rows[name] = rows["review_flags"].str.split(";").apply(lambda values, name=name: name in values)
    rows["excluded"] = rows["exclusion"] != ""
    return rows


def cohort_manifest(name: str) -> pd.DataFrame:
    """
    Read the training manifest of a published cohort.

    Parameters
    ----------
    name : str
        Directory name under ``data/processed``.

    Returns
    -------
    pd.DataFrame
        Manifest rows as text.
    """
    path = ROOT / "data/processed" / name / "train_manifest.csv"
    return pd.read_csv(path, dtype=str, keep_default_na=False)
