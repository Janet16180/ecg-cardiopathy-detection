"""Dataset-independent signal features, feature caching and ECG plotting."""

from collections.abc import Callable
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, welch

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
Reader = Callable[[str], tuple[np.ndarray, int]]


def canonical_order(signal: np.ndarray, names: list[str]) -> np.ndarray:
    """
    Reorder columns to the standard twelve-lead order using the stored lead names.

    Parameters
    ----------
    signal : np.ndarray
        Array of shape ``(samples, 12)``.
    names : list[str]
        Lead name of each column, in any case (``AVL``, ``aVL`` or ``DI``).

    Returns
    -------
    np.ndarray
        Columns in ``LEADS`` order.

    Raises
    ------
    ValueError
        If the names are not exactly the twelve standard leads.
    """
    aliases = {"DI": "I", "DII": "II", "DIII": "III"}
    upper = [aliases.get(name.upper(), name.upper()) for name in names]
    wanted = [lead.upper() for lead in LEADS]
    if sorted(upper) != sorted(wanted):
        raise ValueError(f"Unexpected lead names {names}")
    return signal[:, [upper.index(lead) for lead in wanted]]


def longest_constant_run(lead: np.ndarray) -> int:
    """
    Length of the longest run of identical consecutive samples.

    Parameters
    ----------
    lead : np.ndarray
        One-dimensional signal.

    Returns
    -------
    int
        Number of samples in the longest constant stretch.
    """
    changes = np.flatnonzero(np.diff(lead) != 0)
    boundaries = np.concatenate(([-1], changes, [len(lead) - 1]))
    return int(np.diff(boundaries).max())


def band_fraction(frequencies: np.ndarray, power: np.ndarray, low: float, high: float) -> np.ndarray:
    """
    Fraction of total power inside a frequency band, per lead.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency grid from ``welch``.
    power : np.ndarray
        Power spectral density with shape ``(leads, frequencies)``.
    low, high : float
        Band edges in Hz, inclusive of ``low`` and exclusive of ``high``.

    Returns
    -------
    np.ndarray
        One fraction per lead; NaN for leads with zero power.
    """
    in_band = (frequencies >= low) & (frequencies < high)
    with np.errstate(invalid="ignore", divide="ignore"):
        return power[:, in_band].sum(axis=1) / power.sum(axis=1)


def heart_rate(signal: np.ndarray, fs: int) -> float:
    """
    Rough heart rate from QRS energy summed over all leads.

    The signal is band-passed to 8-20 Hz, where QRS complexes dominate, and
    peaks at least 0.3 s apart are counted. It is a screening estimate, not a
    clinical measurement, and it is unreliable for very noisy records.

    Parameters
    ----------
    signal : np.ndarray
        Array of shape ``(samples, 12)``.
    fs : int
        Sampling rate in Hz.

    Returns
    -------
    float
        Beats per minute from the median RR interval, or NaN with fewer than
        three detected beats.
    """
    b, a = butter(2, [8, 20], btype="band", fs=fs)
    energy = np.abs(filtfilt(b, a, np.nan_to_num(signal), axis=0)).sum(axis=1)
    peaks, _ = find_peaks(energy, distance=int(0.3 * fs), height=0.35 * np.percentile(energy, 99))
    rate = np.nan
    if len(peaks) >= 3:
        rate = 60 / np.median(np.diff(peaks) / fs)
    return float(rate)


def signal_features(signal: np.ndarray, fs: int) -> dict[str, np.ndarray]:
    """
    Per-lead and per-record features of one canonical twelve-lead ECG.

    Parameters
    ----------
    signal : np.ndarray
        Array of shape ``(samples, 12)`` in ``LEADS`` order and mV.
    fs : int
        Sampling rate in Hz.

    Returns
    -------
    dict[str, np.ndarray]
        Arrays of length 12. Record-level values (limb-lead residuals, heart
        rate, length) are repeated for each lead.
    """
    finite = np.nan_to_num(signal)
    leads = finite.T
    frequencies, power = welch(leads, fs=fs, nperseg=min(2048, signal.shape[0]))
    lead = {name: finite[:, i] for i, name in enumerate(LEADS)}
    record = {
        "samples": signal.shape[0],
        "fs": fs,
        "einthoven_residual": np.abs(lead["II"] - lead["I"] - lead["III"]).max(),
        "avf_residual": np.abs(lead["aVF"] - (lead["II"] + lead["III"]) / 2).max(),
        "avl_residual": np.abs(lead["aVL"] - (lead["I"] - lead["III"]) / 2).max(),
        "avr_residual": np.abs(lead["aVR"] + (lead["I"] + lead["II"]) / 2).max(),
        "heart_rate": heart_rate(finite, fs),
    }
    per_lead = {
        "lead": np.array(LEADS),
        "nan_samples": np.isnan(signal).sum(axis=0),
        "zero_fraction": (finite == 0).mean(axis=0),
        "minimum": leads.min(axis=1),
        "maximum": leads.max(axis=1),
        "mean": leads.mean(axis=1),
        "std": leads.std(axis=1),
        "p01": np.percentile(leads, 1, axis=1),
        "p99": np.percentile(leads, 99, axis=1),
        "longest_constant": np.array([longest_constant_run(values) for values in leads]),
        "baseline_fraction": band_fraction(frequencies, power, 0, 0.7),
        "powerline_50_fraction": band_fraction(frequencies, power, 48, 52),
        "powerline_60_fraction": band_fraction(frequencies, power, 58, 62),
        "high_frequency_fraction": band_fraction(frequencies, power, 40, 150),
    }
    return per_lead | {name: np.full(len(LEADS), value) for name, value in record.items()}


def _features_for(item: tuple[str, str], reader: Reader) -> dict[str, np.ndarray]:
    record_id, path = item
    signal, fs = reader(path)
    features = signal_features(signal, fs)
    features["record_id"] = np.full(len(LEADS), record_id, dtype=object)
    return features


def compute_features(items: list[tuple[str, str]], reader: Reader, cache: Path,
                     workers: int = 6) -> pd.DataFrame:
    """
    Compute features for many records in parallel, or load them from cache.

    Parameters
    ----------
    items : list[tuple[str, str]]
        Record identifier and the path handed to ``reader``.
    reader : Reader
        Top-level function returning ``(signal, fs)`` in canonical order.
    cache : Path
        Parquet file written after the first computation.
    workers : int
        Number of processes.

    Returns
    -------
    pd.DataFrame
        One row per record and lead.
    """
    if cache.exists():
        return pd.read_parquet(cache)
    with Pool(workers) as pool:
        parts = pool.map(partial(_features_for, reader=reader), items, chunksize=64)
    table = pd.DataFrame({key: np.concatenate([part[key] for part in parts]) for key in parts[0]})
    table["record_id"] = table["record_id"].astype(str)
    cache.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(cache)
    return table


def record_summary(features: pd.DataFrame, flat_seconds: float = 1.0,
                   large_mv: float = 10.0) -> pd.DataFrame:
    """
    Collapse per-lead features into one row per record with quality flags.

    Parameters
    ----------
    features : pd.DataFrame
        Output of ``compute_features``.
    flat_seconds : float
        A lead is flat when it has a constant run at least this long.
    large_mv : float
        A lead is large when any sample reaches this absolute amplitude.

    Returns
    -------
    pd.DataFrame
        One row per record indexed by ``record_id``.
    """
    features = features.assign(
        flat=features["longest_constant"] >= flat_seconds * features["fs"],
        fully_constant=features["longest_constant"] >= features["samples"],
        large=np.maximum(features["maximum"].abs(), features["minimum"].abs()) >= large_mv,
        any_nan=features["nan_samples"] > 0,
        peak=np.maximum(features["maximum"].abs(), features["minimum"].abs()),
    )
    grouped = features.groupby("record_id", sort=False)
    first = grouped[["samples", "fs", "einthoven_residual", "avf_residual", "avl_residual",
                     "avr_residual", "heart_rate"]].first()
    medians = grouped[["std", "baseline_fraction", "powerline_50_fraction", "powerline_60_fraction",
                       "high_frequency_fraction", "zero_fraction"]].median()
    counts = grouped[["flat", "fully_constant", "large", "any_nan"]].sum()
    counts.columns = ["flat_leads", "constant_leads", "large_leads", "nan_leads"]
    summary = first.join(medians).join(counts)
    summary["peak_abs_mv"] = grouped["peak"].max()
    summary["duration_s"] = summary["samples"] / summary["fs"]
    return summary


def problem_counts(summary: pd.DataFrame, residual_mv: float = 0.05) -> pd.Series:
    """
    Count records affected by each integrity problem.

    Parameters
    ----------
    summary : pd.DataFrame
        Output of ``record_summary``.
    residual_mv : float
        Largest acceptable limb-lead identity violation in mV.

    Returns
    -------
    pd.Series
        Record count per problem.
    """
    limb = summary[["einthoven_residual", "avf_residual", "avl_residual", "avr_residual"]].max(axis=1)
    return pd.Series({
        "records": len(summary),
        "any_nan": int((summary["nan_leads"] > 0).sum()),
        "fully_constant_lead": int((summary["constant_leads"] > 0).sum()),
        "lead_flat_for_1s": int((summary["flat_leads"] > 0).sum()),
        "lead_over_10mv": int((summary["large_leads"] > 0).sum()),
        "limb_identity_violated": int((limb > residual_mv).sum()),
        "heart_rate_not_estimated": int(summary["heart_rate"].isna().sum()),
    })


def plot_ecg(signal: np.ndarray, fs: int, title: str, seconds: float = 10.0) -> plt.Figure:
    """
    Plot a twelve-lead ECG as a 6x2 grid.

    Parameters
    ----------
    signal : np.ndarray
        Array of shape ``(samples, 12)`` in ``LEADS`` order and mV.
    fs : int
        Sampling rate in Hz.
    title : str
        Figure title.
    seconds : float
        Length of the plotted window from the start.

    Returns
    -------
    plt.Figure
        The figure.
    """
    shown = signal[: int(seconds * fs)]
    time = np.arange(len(shown)) / fs
    figure, axes = plt.subplots(6, 2, figsize=(15, 9), sharex=True)
    for i, axis in enumerate(axes.T.flat):
        axis.plot(time, shown[:, i], linewidth=0.6, color="black")
        axis.set_ylabel(LEADS[i], rotation=0, labelpad=18)
        axis.grid(alpha=0.3)
    for axis in axes[-1]:
        axis.set_xlabel("Seconds")
    figure.suptitle(title)
    figure.tight_layout()
    return figure
