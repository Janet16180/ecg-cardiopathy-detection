"""Load the local MIMIC-IV-ECG subset, its machine measurements and waveforms."""

import numpy as np
import pandas as pd
import wfdb

from eda.ptbxl import OUTPUT_DIR, ROOT
from eda.signals import LEADS, canonical_order, compute_features, record_summary

MIMIC_DIR = ROOT / "data/raw/mimic-iv-ecg/1.0"
SELECTION_DIR = ROOT / "data/processed/mimic_ssl_200k"
FEATURE_CACHE = OUTPUT_DIR / "features" / "mimic.parquet"
MEASUREMENT_CACHE = OUTPUT_DIR / "features" / "mimic_measurements.parquet"
RECORD_CACHE = OUTPUT_DIR / "features" / "mimic_records.parquet"
MISSING_CODE = 29999
MISSING_THRESHOLD_MS = 29000
REPORT_COLUMNS = [f"report_{i}" for i in range(18)]
INTERVAL_COLUMNS = ["rr_interval", "p_onset", "p_end", "qrs_onset", "qrs_end", "t_end",
                    "p_axis", "qrs_axis", "t_axis"]


def load_records() -> pd.DataFrame:
    """
    Load the official record list and mark which records are on local disk, cached.

    Returns
    -------
    pd.DataFrame
        One row per official record indexed by ``study_id``, with
        ``downloaded`` and ``project_selected`` flags.
    """
    if RECORD_CACHE.exists():
        return pd.read_parquet(RECORD_CACHE)
    records = pd.read_csv(MIMIC_DIR / "record_list.csv", parse_dates=["ecg_time"]).set_index("study_id")
    local = {path.stem for path in (MIMIC_DIR / "files").rglob("*.hea")}
    selected = pd.read_csv(SELECTION_DIR / "selected_records.csv")["study_id"]
    records["downloaded"] = records.index.astype(str).isin(local)
    records["project_selected"] = records.index.isin(selected)
    RECORD_CACHE.parent.mkdir(parents=True, exist_ok=True)
    records.to_parquet(RECORD_CACHE)
    return records


def load_machine_measurements() -> pd.DataFrame:
    """
    Load machine measurements with the free-text report lines joined, cached.

    Returns
    -------
    pd.DataFrame
        One row per study indexed by ``study_id``. ``report`` joins the report
        lines, and ``overall`` holds the machine's normal/borderline/abnormal
        statement when one exists. Interval values keep the release's 29999
        missing-value code; see ``MISSING_CODE``.
    """
    if MEASUREMENT_CACHE.exists():
        return pd.read_parquet(MEASUREMENT_CACHE)
    table = pd.read_csv(MIMIC_DIR / "machine_measurements.csv", low_memory=False).set_index("study_id")
    lines = table[REPORT_COLUMNS].stack().dropna().astype(str).str.strip()
    table["report"] = lines[lines != ""].groupby(level=0).agg(" | ".join).reindex(table.index).fillna("")
    overall = table["report"].str.extract(r"(Abnormal ECG|Borderline ECG|Normal ECG|Otherwise normal ECG)",
                                          expand=False)
    table["overall"] = overall.fillna("no statement")
    table = table.drop(columns=REPORT_COLUMNS)
    MEASUREMENT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(MEASUREMENT_CACHE)
    return table


def read_signal(path: str) -> tuple[np.ndarray, int]:
    """
    Read one MIMIC record in mV, reordering leads by their stored names.

    Parameters
    ----------
    path : str
        Relative record path from ``record_list.csv``.

    Returns
    -------
    tuple[np.ndarray, int]
        Array of shape ``(samples, 12)`` in ``LEADS`` order, and the rate.
    """
    signal, fields = wfdb.rdsamp(str(MIMIC_DIR / path))
    return canonical_order(signal, fields["sig_name"]), int(fields["fs"])


def read_signal_by_position(path: str) -> tuple[np.ndarray, int]:
    """
    Read one MIMIC record assuming columns are already in standard order.

    This deliberately ignores the stored lead names, to show what a loader
    that stacks channels by position would produce.

    Parameters
    ----------
    path : str
        Relative record path from ``record_list.csv``.

    Returns
    -------
    tuple[np.ndarray, int]
        Array of shape ``(samples, 12)`` in file order, and the rate.
    """
    signal, fields = wfdb.rdsamp(str(MIMIC_DIR / path))
    return signal, int(fields["fs"])


def stored_lead_orders(records: pd.DataFrame, sample: int = 2000, seed: int = 0) -> pd.Series:
    """
    Count the lead orders written in a random sample of headers.

    Parameters
    ----------
    records : pd.DataFrame
        Output of ``load_records``.
    sample : int
        Number of downloaded headers to read.
    seed : int
        Sampling seed.

    Returns
    -------
    pd.Series
        Number of headers per comma-joined lead order.
    """
    paths = records.loc[records["downloaded"], "path"].sample(sample, random_state=seed)
    orders = [",".join(wfdb.rdheader(str(MIMIC_DIR / path)).sig_name) for path in paths]
    return pd.Series(orders).value_counts()


def nan_profile(paths: list[str]) -> pd.DataFrame:
    """
    Describe where non-finite samples occur in the given records.

    Parameters
    ----------
    paths : list[str]
        Relative record paths.

    Returns
    -------
    pd.DataFrame
        Per record: NaN samples per lead and the first and last NaN index.
    """
    rows = []
    for path in paths:
        signal, _ = read_signal(path)
        missing = np.isnan(signal)
        positions = np.flatnonzero(missing.any(axis=1))
        rows.append({"path": path, **dict(zip(LEADS, missing.sum(axis=0), strict=True)),
                     "first_nan": positions.min(), "last_nan": positions.max()})
    return pd.DataFrame(rows).set_index("path")


def mimic_summary(records: pd.DataFrame) -> pd.DataFrame:
    """
    Per-record signal summary of every downloaded record, joined with the record list.

    Parameters
    ----------
    records : pd.DataFrame
        Output of ``load_records``.

    Returns
    -------
    pd.DataFrame
        ``record_summary`` columns plus record-list fields, indexed by ``study_id``.
    """
    local = records[records["downloaded"]]
    items = [(str(study), path) for study, path in local["path"].items()]
    summary = record_summary(compute_features(items, read_signal, FEATURE_CACHE, workers=7))
    summary.index = summary.index.astype(int)
    return summary.join(records)


def measured_intervals(measurements: pd.DataFrame) -> pd.DataFrame:
    """
    Machine interval and axis columns with missing and impossible values set to NaN.

    The release marks unmeasured values with 29999 (a few nearby values also
    occur). Timings of 29000 ms or more, RR intervals of zero or less, and axes
    beyond +/-360 degrees are treated as missing.

    Parameters
    ----------
    measurements : pd.DataFrame
        Output of ``load_machine_measurements``.

    Returns
    -------
    pd.DataFrame
        ``INTERVAL_COLUMNS`` as floats, plus derived ``heart_rate`` (bpm),
        ``qrs_duration``, ``pr_interval``, ``qt_interval`` and ``qtc`` (Bazett),
        all in ms. Derived values are not otherwise cleaned, so impossible
        durations remain visible.
    """
    timing_columns, axis_columns = INTERVAL_COLUMNS[:6], INTERVAL_COLUMNS[6:]
    timing = measurements[timing_columns].astype(float)
    timing = timing.mask(timing >= MISSING_THRESHOLD_MS)
    timing["rr_interval"] = timing["rr_interval"].mask(timing["rr_interval"] <= 0)
    axes = measurements[axis_columns].astype(float)
    intervals = timing.join(axes.mask(axes.abs() > 360))
    intervals["heart_rate"] = 60000 / intervals["rr_interval"]
    intervals["qrs_duration"] = intervals["qrs_end"] - intervals["qrs_onset"]
    intervals["pr_interval"] = intervals["qrs_onset"] - intervals["p_onset"]
    intervals["qt_interval"] = intervals["t_end"] - intervals["qrs_onset"]
    intervals["qtc"] = intervals["qt_interval"] / np.sqrt(intervals["rr_interval"] / 1000)
    return intervals


def project_accepted(name: str = "mimic_ssl_200k") -> pd.Index:
    """
    Study IDs the project accepted into one of its MIMIC SSL manifests.

    Parameters
    ----------
    name : str
        ``mimic_ssl_200k`` or ``mimic_ssl_40k_cpc``.

    Returns
    -------
    pd.Index
        Integer study IDs.
    """
    manifest = pd.read_csv(ROOT / "data/processed" / name / "ssl_manifest.csv")
    return pd.Index(manifest["ecg_id"].str.split(":").str[1].astype(int))
