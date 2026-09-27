"""PTB-XL specific signal analyses built on ``eda.signals``."""

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from eda.ptbxl import OUTPUT_DIR, QUALITY_COLUMNS, read_signal
from eda.signals import compute_features, record_summary

FEATURE_CACHE = OUTPUT_DIR / "features" / "ptbxl_500hz.parquet"


def ptbxl_summary(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Per-record signal summary of every 500 Hz record, joined with metadata.

    Parameters
    ----------
    meta : pd.DataFrame
        PTB-XL metadata indexed by ``ecg_id``.

    Returns
    -------
    pd.DataFrame
        ``record_summary`` columns plus metadata, indexed by ``ecg_id``.
    """
    items = [(str(ecg_id), path) for ecg_id, path in meta["filename_hr"].items()]
    summary = record_summary(compute_features(items, read_signal, FEATURE_CACHE))
    summary.index = summary.index.astype(int)
    return summary.join(meta)


def lead_features(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Per-lead features of every record, with the record's metadata columns.

    Parameters
    ----------
    meta : pd.DataFrame
        PTB-XL metadata indexed by ``ecg_id``, with ``project`` and ``device``.

    Returns
    -------
    pd.DataFrame
        One row per record and lead.
    """
    items = [(str(ecg_id), path) for ecg_id, path in meta["filename_hr"].items()]
    features = compute_features(items, read_signal, FEATURE_CACHE)
    features["ecg_id"] = features["record_id"].astype(int)
    return features.merge(meta[["project", "device", "split"]], left_on="ecg_id", right_index=True)


def annotation_agreement(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Compare PTB-XL's human noise annotations with the measured statistics.

    Parameters
    ----------
    summary : pd.DataFrame
        Output of ``ptbxl_summary``.

    Returns
    -------
    pd.DataFrame
        Median measured statistics for records with and without each annotation.
    """
    metrics = ["baseline_fraction", "high_frequency_fraction", "powerline_50_fraction"]
    rows = []
    for column in QUALITY_COLUMNS[:4]:
        flagged = summary[column].notna()
        for value, subset in ((True, summary[flagged]), (False, summary[~flagged])):
            rows.append({"annotation": column, "annotated": value, "records": len(subset),
                         **subset[metrics].median().round(4).to_dict()})
    return pd.DataFrame(rows)


def low_rate_consistency(meta: pd.DataFrame, records: int = 300, seed: int = 0) -> pd.DataFrame:
    """
    Check that each 100 Hz file is a decimated copy of its 500 Hz file.

    Parameters
    ----------
    meta : pd.DataFrame
        PTB-XL metadata indexed by ``ecg_id``.
    records : int
        Number of randomly sampled records.
    seed : int
        Sampling seed.

    Returns
    -------
    pd.DataFrame
        Per record: correlation and largest absolute difference (mV) between
        the 100 Hz file and the 500 Hz file downsampled by 5.
    """
    rows = []
    for ecg_id, row in meta.sample(records, random_state=seed).iterrows():
        high, _ = read_signal(row["filename_hr"])
        low, _ = read_signal(row["filename_lr"])
        downsampled = resample_poly(high, up=1, down=5, axis=0)
        rows.append({"ecg_id": ecg_id,
                     "correlation": np.corrcoef(low.ravel(), downsampled.ravel())[0, 1],
                     "max_abs_difference_mv": np.abs(low - downsampled).max(),
                     "median_abs_difference_mv": np.median(np.abs(low - downsampled))})
    return pd.DataFrame(rows).set_index("ecg_id")
