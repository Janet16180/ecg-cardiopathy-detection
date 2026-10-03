"""Dataset-independent profiling helpers: structure, univariate summaries, association measures."""

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.stats import chi2_contingency


def column_profile(frame: pd.DataFrame, scales: dict[str, str]) -> pd.DataFrame:
    """
    Describe the type, measurement scale and completeness of every column.

    Parameters
    ----------
    frame : pd.DataFrame
        One row per record. Columns holding dicts or lists are counted as text.
    scales : dict[str, str]
        Measurement scale or role per column (for example nominal, ratio,
        identifier). Columns not listed get an empty scale.

    Returns
    -------
    pd.DataFrame
        One row per column with ``dtype``, ``scale``, ``non_null``,
        ``missing_share``, ``unique`` and ``example``.
    """
    rows = []
    for column in frame.columns:
        values = frame[column]
        hashable = values.dropna().map(str) if values.dtype == object else values.dropna()
        rows.append({
            "column": column,
            "dtype": str(values.dtype),
            "scale": scales.get(column, ""),
            "non_null": int(values.notna().sum()),
            "missing_share": float(values.isna().mean()),
            "unique": int(hashable.nunique()),
            "example": str(values.dropna().iloc[0])[:40] if values.notna().any() else "",
        })
    return pd.DataFrame(rows).set_index("column")


def numeric_summary(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Descriptive statistics, shape and Tukey outliers of numeric columns.

    Outliers are values beyond 1.5 interquartile ranges from the quartiles.

    Parameters
    ----------
    frame : pd.DataFrame
        One row per record.
    columns : list[str]
        Numeric columns; missing values are ignored.

    Returns
    -------
    pd.DataFrame
        One row per column with the ``describe`` statistics, ``skew``,
        ``kurtosis``, ``outliers`` and ``outlier_share``.
    """
    table = frame[columns].astype(float).describe().T
    table["skew"] = frame[columns].astype(float).skew()
    table["kurtosis"] = frame[columns].astype(float).kurt()
    spread = table["75%"] - table["25%"]
    low = table["25%"] - 1.5 * spread
    high = table["75%"] + 1.5 * spread
    outside = frame[columns].astype(float).lt(low) | frame[columns].astype(float).gt(high)
    table["outliers"] = outside.sum()
    table["outlier_share"] = table["outliers"] / table["count"]
    return table


def category_name(value: object) -> str:
    """
    Display name of a category, with float-coded integers shown without decimals.

    Parameters
    ----------
    value : object
        One categorical value.

    Returns
    -------
    str
        ``(missing)`` for gaps, otherwise the value as text.
    """
    if pd.isna(value):
        return "(missing)"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def frequencies(values: pd.Series, top: int = 10) -> pd.DataFrame:
    """
    Count and share of each category, with missing values as their own row.

    Parameters
    ----------
    values : pd.Series
        Categorical values.
    top : int
        Number of most frequent categories kept; the rest are summed into
        ``(other)``.

    Returns
    -------
    pd.DataFrame
        ``records`` and ``share`` per category.
    """
    counts = values.map(category_name).value_counts()
    if len(counts) > top:
        counts = pd.concat([counts.iloc[:top], pd.Series({"(other)": counts.iloc[top:].sum()})])
    return pd.DataFrame({"records": counts, "share": counts / len(values)})


def missing_correlation(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Correlation between the missingness indicators of partially missing columns.

    Parameters
    ----------
    frame : pd.DataFrame
        One row per record.

    Returns
    -------
    pd.DataFrame
        Pearson correlation of the ``isna`` indicators of every column that is
        missing in some but not all rows.
    """
    missing = frame.isna()
    partial = missing.columns[missing.any() & ~missing.all()]
    return missing[partial].astype(float).corr()


def cramers_v(first: pd.Series, second: pd.Series) -> float:
    """
    Bias-corrected Cramér's V between two categorical variables.

    Parameters
    ----------
    first, second : pd.Series
        Categorical values; rows where either is missing are dropped.

    Returns
    -------
    float
        Association strength from 0 (independent) to 1.
    """
    table = pd.crosstab(first, second)
    total = table.to_numpy().sum()
    phi2 = chi2_contingency(table, correction=False)[0] / total
    rows, columns = table.shape
    phi2 = max(0.0, phi2 - (rows - 1) * (columns - 1) / (total - 1))
    rows = rows - (rows - 1) ** 2 / (total - 1)
    columns = columns - (columns - 1) ** 2 / (total - 1)
    return float(np.sqrt(phi2 / min(rows - 1, columns - 1)))


def cramers_v_matrix(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Pairwise bias-corrected Cramér's V between categorical columns.

    Parameters
    ----------
    frame : pd.DataFrame
        One row per record.
    columns : list[str]
        Categorical columns.

    Returns
    -------
    pd.DataFrame
        Symmetric matrix with ones on the diagonal.
    """
    matrix = pd.DataFrame(1.0, index=columns, columns=columns)
    for i, first in enumerate(columns):
        for second in columns[i + 1:]:
            value = cramers_v(frame[first].astype(str), frame[second].astype(str))
            matrix.loc[first, second] = matrix.loc[second, first] = value
    return matrix


def log_skew(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Skewness of positive columns before and after a log transform.

    Parameters
    ----------
    frame : pd.DataFrame
        One row per record.
    columns : list[str]
        Strictly positive numeric columns; missing values are ignored.

    Returns
    -------
    pd.DataFrame
        ``raw_skew`` and ``log_skew`` per column.
    """
    values = frame[columns].astype(float)
    return pd.DataFrame({"raw_skew": values.skew(), "log_skew": np.log10(values).skew()})


def remove_baseline(signal: np.ndarray, fs: int, cutoff_hz: float = 0.5) -> np.ndarray:
    """
    Remove baseline wander with a zero-phase Butterworth high-pass filter.

    Parameters
    ----------
    signal : np.ndarray
        Array of shape ``(samples, leads)``.
    fs : int
        Sampling rate in Hz.
    cutoff_hz : float
        High-pass cutoff in Hz.

    Returns
    -------
    np.ndarray
        Filtered signal with the same shape and unit.
    """
    sos = butter(2, cutoff_hz, btype="highpass", fs=fs, output="sos")
    return sosfiltfilt(sos, signal, axis=0)


def standardize_leads(signal: np.ndarray) -> np.ndarray:
    """
    Scale every lead to zero mean and unit standard deviation.

    Parameters
    ----------
    signal : np.ndarray
        Array of shape ``(samples, leads)``.

    Returns
    -------
    np.ndarray
        Unitless standardized signal.
    """
    return (signal - signal.mean(axis=0)) / signal.std(axis=0)
