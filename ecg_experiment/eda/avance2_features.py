"""Clinical ECG measurements and derived features for the Avance 2 PTB-XL notebook."""

import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.stats import boxcox, yeojohnson
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ecg_experiment.eda.ptbxl import read_signal
from ecg_experiment.eda.signals import LEADS

BEFORE_S, AFTER_S = 0.3, 0.65
ST_OFFSET_S = 0.06
T_INVERSION_LEADS = ["I", "II", "aVL", "aVF", "V2", "V3", "V4", "V5", "V6"]


def qrs_peaks(signal: np.ndarray, fs: int) -> np.ndarray:
    """
    Locate QRS complexes from the 8-20 Hz energy summed over all leads.

    Parameters
    ----------
    signal : np.ndarray
        Array of shape ``(samples, 12)`` in mV.
    fs : int
        Sampling rate in Hz.

    Returns
    -------
    np.ndarray
        Sample index of each detected QRS complex.
    """
    b, a = butter(2, [8, 20], btype="band", fs=fs)
    energy = np.abs(filtfilt(b, a, signal, axis=0)).sum(axis=1)
    peaks, _ = find_peaks(energy, distance=int(0.3 * fs), height=0.35 * np.percentile(energy, 99))
    return peaks


def median_beat(filtered: np.ndarray, peaks: np.ndarray, fs: int) -> tuple[np.ndarray, float]:
    """
    Median of all complete beats aligned on their QRS complexes.

    Parameters
    ----------
    filtered : np.ndarray
        Band-passed signal of shape ``(samples, 12)``.
    peaks : np.ndarray
        QRS sample indices.
    fs : int
        Sampling rate in Hz.

    Returns
    -------
    tuple[np.ndarray, float]
        Median beat of shape ``(window, 12)``, and the median correlation of
        each beat with it (1 means every beat has the same shape).
    """
    before, after = int(BEFORE_S * fs), int(AFTER_S * fs)
    keep = peaks[(peaks >= before) & (peaks + after <= len(filtered))]
    beats = np.stack([filtered[p - before:p + after] for p in keep])
    beat = np.median(beats, axis=0)
    flat = beat.ravel() - beat.mean()
    corr = [np.corrcoef(b.ravel() - b.mean(), flat)[0, 1] for b in beats]
    return beat, float(np.median(corr))


def qrs_bounds(beat: np.ndarray, fs: int, fraction: float = 0.05, hold_s: float = 0.01) -> tuple[int, int]:
    """
    QRS onset and offset where the spatial velocity of the median beat stays low.

    Parameters
    ----------
    beat : np.ndarray
        Median beat of shape ``(window, 12)`` with the QRS at ``BEFORE_S``.
    fs : int
        Sampling rate in Hz.
    fraction : float
        Share of the peak velocity below which the signal counts as quiet.
    hold_s : float
        Seconds the signal must stay quiet to end the QRS.

    Returns
    -------
    tuple[int, int]
        Onset and offset sample indices inside the median beat.
    """
    velocity = np.abs(np.diff(beat, axis=0)).sum(axis=1)
    center = int(BEFORE_S * fs)
    near = slice(center - int(0.06 * fs), center + int(0.06 * fs))
    peak = near.start + int(np.argmax(velocity[near]))
    hold = int(hold_s * fs)
    quiet = np.convolve(velocity < fraction * velocity[peak], np.ones(hold), "valid") == hold
    first, last = peak - int(0.15 * fs), peak + int(0.2 * fs)
    before = quiet[first - hold + 1:peak - hold + 1][::-1]
    after = quiet[peak:last]
    onset = peak - 1 - int(np.argmax(before)) if before.any() else first
    offset = peak + int(np.argmax(after)) if after.any() else last
    return onset, offset


def t_wave(corrected: np.ndarray, offset: int, rr_samples: float, fs: int) -> tuple[int, int] | None:
    """
    T-wave peak and end from the spatial magnitude after the QRS.

    The end follows the tangent method: the steepest descending point after
    the peak is projected to the lowest magnitude before the next beat.

    Parameters
    ----------
    corrected : np.ndarray
        Median beat minus the isoelectric baseline of each lead.
    offset : int
        QRS offset index.
    rr_samples : float
        Median RR interval in samples, used to stop before the next beat.
    fs : int
        Sampling rate in Hz.

    Returns
    -------
    tuple[int, int] | None
        T-peak and T-end indices inside the median beat, or None when the
        next beat starts before a T wave can fit.
    """
    magnitude = np.sqrt((corrected ** 2).sum(axis=1))
    stop = min(len(corrected), int(BEFORE_S * fs + rr_samples - 0.1 * fs))
    start = offset + int(0.08 * fs)
    if start >= stop:
        return None
    peak = start + int(np.argmax(magnitude[start:stop]))
    floor = magnitude[peak:stop].min()
    slope = np.diff(magnitude[peak:min(stop, peak + int(0.2 * fs))])
    if len(slope) == 0 or slope.min() >= 0:
        return peak, stop
    steep = int(np.argmin(slope))
    end = peak + steep + (magnitude[peak + steep] - floor) / -slope[steep]
    return peak, int(min(round(end), stop))


def beat_landmarks(signal: np.ndarray, fs: int, peaks: np.ndarray) -> dict:
    """
    Median beat of one ECG with its QRS, ST and T landmarks.

    Parameters
    ----------
    signal : np.ndarray
        Array of shape ``(samples, 12)`` in mV, canonical lead order.
    fs : int
        Sampling rate in Hz.
    peaks : np.ndarray
        QRS sample indices from ``qrs_peaks``, at least three.

    Returns
    -------
    dict
        ``corrected`` (median beat minus each lead's isoelectric level, in
        mV), ``consistency``, and the indices ``onset``, ``offset``, ``st``,
        ``t_peak`` and ``t_end`` (the last two None without a T wave).
    """
    b, a = butter(2, [0.5, 40], btype="band", fs=fs)
    beat, consistency = median_beat(filtfilt(b, a, signal, axis=0), peaks, fs)
    onset, offset = qrs_bounds(beat, fs)
    corrected = beat - beat[onset - int(0.03 * fs):onset - int(0.01 * fs)].mean(axis=0)
    t_found = t_wave(corrected, offset, np.median(np.diff(peaks)), fs)
    return {"corrected": corrected, "consistency": consistency, "onset": onset, "offset": offset,
            "st": min(offset + int(ST_OFFSET_S * fs), len(beat) - 1),
            "t_peak": t_found[0] if t_found else None, "t_end": t_found[1] if t_found else None}


def ecg_measurements(signal: np.ndarray, fs: int) -> dict:
    """
    Rhythm, interval and per-lead amplitude measurements of one 12-lead ECG.

    Parameters
    ----------
    signal : np.ndarray
        Array of shape ``(samples, 12)`` in mV, canonical lead order.
    fs : int
        Sampling rate in Hz.

    Returns
    -------
    dict
        Raw measurements; NaN when fewer than three beats are found.
    """
    row = {"constant_leads": int((np.ptp(signal, axis=0) == 0).sum()),
           "amp_II": float(np.ptp(signal[:, 1])), "max_abs": float(np.abs(signal).max()),
           "signal_std": float(signal.std())}
    freqs, power = welch(signal, fs=fs, nperseg=2 * fs, axis=0)
    row["baseline_fraction"] = float(power[freqs < 0.7].sum() / power.sum())
    peaks = qrs_peaks(signal, fs)
    row["n_beats"] = len(peaks)
    if len(peaks) < 3:
        return row
    rr = np.diff(peaks) / fs
    row.update(hr_bpm=60 / np.median(rr), rr_std_ms=1000 * rr.std(), rr_mean_ms=1000 * rr.mean())
    marks = beat_landmarks(signal, fs, peaks)
    corrected, onset, offset = marks["corrected"], marks["onset"], marks["offset"]
    qrs = corrected[onset:offset + 1]
    has_t = marks["t_peak"] is not None
    row.update(beat_consistency=marks["consistency"], qrs_ms=1000 * (offset - onset) / fs,
               qt_ms=1000 * (marks["t_end"] - onset) / fs if has_t else np.nan)
    for j, lead in enumerate(LEADS):
        row[f"r_amp_{lead}"] = float(max(qrs[:, j].max(), 0))
        row[f"s_amp_{lead}"] = float(max(-qrs[:, j].min(), 0))
        row[f"st_{lead}"] = float(corrected[marks["st"], j])
        row[f"t_amp_{lead}"] = float(corrected[marks["t_peak"], j]) if has_t else np.nan
    return row


def record_measurements(item: tuple[int, str]) -> dict:
    """
    Read one PTB-XL record and measure it.

    Parameters
    ----------
    item : tuple[int, str]
        ``ecg_id`` and the 500 Hz relative file name.

    Returns
    -------
    dict
        Measurements with the ``ecg_id``.
    """
    ecg_id, filename = item
    signal, fs = read_signal(filename)
    return {"ecg_id": ecg_id, **ecg_measurements(signal, fs)}


def measurement_table(meta: pd.DataFrame, workers: int = 16) -> pd.DataFrame:
    """
    Measure every record of the metadata table in parallel.

    Parameters
    ----------
    meta : pd.DataFrame
        PTB-XL metadata indexed by ``ecg_id`` with ``filename_hr``.
    workers : int
        Number of processes.

    Returns
    -------
    pd.DataFrame
        One row per ECG indexed by ``ecg_id``.
    """
    items = list(meta["filename_hr"].items())
    with Pool(workers) as pool:
        rows = pool.map(record_measurements, items, chunksize=64)
    return pd.DataFrame(rows).set_index("ecg_id").loc[meta.index]


def derived_features(measures: pd.DataFrame, female: pd.Series) -> pd.DataFrame:
    """
    Combine raw measurements into clinically motivated features.

    Parameters
    ----------
    measures : pd.DataFrame
        Output of ``measurement_table``.
    female : pd.Series
        1 for women and 0 for men, aligned with ``measures``.

    Returns
    -------
    pd.DataFrame
        New features indexed like ``measures``.
    """
    m = measures
    net_i = m["r_amp_I"] - m["s_amp_I"]
    net_avf = m["r_amp_aVF"] - m["s_amp_aVF"]
    axis = np.degrees(np.arctan2(net_avf, net_i))
    sokolow = m["s_amp_V1"] + np.maximum(m["r_amp_V5"], m["r_amp_V6"])
    cornell = m["r_amp_aVL"] + m["s_amp_V3"]
    st = m[[f"st_{lead}" for lead in LEADS]]
    t_inverted = m[[f"t_amp_{lead}" for lead in T_INVERSION_LEADS]].lt(-0.1)
    missing_t = m["qt_ms"].isna()
    out = pd.DataFrame({
        "qtc_ms": m["qt_ms"] / np.sqrt(m["rr_mean_ms"] / 1000),
        "rr_cv": m["rr_std_ms"] / m["rr_mean_ms"],
        "qrs_axis_deg": axis,
        "axis_sin": np.sin(np.radians(axis)),
        "axis_cos": np.cos(np.radians(axis)),
        "sokolow_lyon_mv": sokolow,
        "cornell_mv": cornell,
        "lvh_sokolow": sokolow.gt(3.5).astype(float),
        "lvh_cornell": cornell.gt(np.where(female.eq(1), 2.0, 2.8)).astype(float),
        "n_st_elevated": st.gt(0.1).sum(axis=1).astype(float),
        "n_st_depressed": st.lt(-0.05).sum(axis=1).astype(float),
        "n_t_inverted": t_inverted.sum(axis=1).astype(float),
    }, index=m.index)
    out.loc[missing_t, "n_t_inverted"] = np.nan
    return out


def axis_category(axis_deg: pd.Series) -> pd.Series:
    """
    Name the frontal QRS axis range.

    Parameters
    ----------
    axis_deg : pd.Series
        Axis in degrees between -180 and 180.

    Returns
    -------
    pd.Series
        ``normal`` (-30 to 90), ``izquierda`` (-90 to -30), ``derecha``
        (90 to 180) or ``extrema`` (-180 to -90); NaN stays NaN.
    """
    bins = [-180.001, -90, -30, 90, 180.001]
    return pd.cut(axis_deg, bins, labels=["extrema", "izquierda", "normal", "derecha"]).astype("object")


def skew_table(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Skewness before and after the usual transformations, per column.

    Parameters
    ----------
    frame : pd.DataFrame
        Numeric columns without missing values.

    Returns
    -------
    pd.DataFrame
        Skewness of the original values and after log1p (only for
        non-negative columns), square root, Box-Cox (only for strictly
        positive columns) and Yeo-Johnson.
    """
    rows = {}
    for col in frame:
        x = frame[col].to_numpy(dtype=float)
        nonneg, positive = (x >= 0).all(), (x > 0).all()
        rows[col] = {
            "original": pd.Series(x).skew(),
            "log1p": pd.Series(np.log1p(x)).skew() if nonneg else np.nan,
            "raiz_cuadrada": pd.Series(np.sqrt(x)).skew() if nonneg else np.nan,
            "box_cox": pd.Series(boxcox(x)[0]).skew() if positive else np.nan,
            "yeo_johnson": pd.Series(yeojohnson(x)[0]).skew(),
        }
    return pd.DataFrame(rows).T


def eta_squared(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """
    ANOVA effect size of each numeric feature for each binary label.

    Parameters
    ----------
    features : pd.DataFrame
        Numeric features without missing values.
    labels : pd.DataFrame
        Binary label columns aligned with ``features``.

    Returns
    -------
    pd.DataFrame
        Eta squared, features by labels. It is the share of the feature's
        variance explained by the label, from 0 to 1.
    """
    n = len(features)
    out = {}
    for label in labels:
        f, _ = f_classif(features, labels[label])
        out[label] = f / (f + n - 2)
    return pd.DataFrame(out, index=features.columns)


def drop_redundant(
    corr: pd.DataFrame, relevance: pd.Series, threshold: float,
) -> tuple[list[str], pd.DataFrame]:
    """
    Remove one feature from every highly correlated pair, keeping the more relevant.

    Parameters
    ----------
    corr : pd.DataFrame
        Absolute correlation matrix.
    relevance : pd.Series
        Relevance score per feature; larger is kept.
    threshold : float
        Absolute correlation above which two features count as redundant.

    Returns
    -------
    tuple[list[str], pd.DataFrame]
        Kept features in relevance order, and one row per removed feature.
    """
    kept, removed = [], []
    for feature in relevance.sort_values(ascending=False).index:
        partner = next((k for k in kept if corr.loc[feature, k] > threshold), None)
        if partner is None:
            kept.append(feature)
        else:
            removed.append({"eliminada": feature, "se_conserva": partner,
                            "correlacion_abs": corr.loc[feature, partner]})
    return kept, pd.DataFrame(removed)


def validation_auc(features: pd.DataFrame, labels: pd.DataFrame, train: pd.Series,
                   validation: pd.Series) -> dict:
    """
    Fit one logistic regression per label on train and score it on validation.

    Parameters
    ----------
    features : pd.DataFrame
        Scaled features without missing values.
    labels : pd.DataFrame
        Binary label columns aligned with ``features``.
    train, validation : pd.Series
        Boolean masks aligned with ``features``.

    Returns
    -------
    dict
        Number of features, fitting seconds, ROC AUC per label and their mean.
    """
    start = time.perf_counter()
    aucs = {}
    for label in labels:
        model = LogisticRegression(max_iter=3000).fit(features[train], labels.loc[train, label])
        scores = model.predict_proba(features[validation])[:, 1]
        aucs[label] = roc_auc_score(labels.loc[validation, label], scores)
    return {"n_caracteristicas": features.shape[1], "segundos": time.perf_counter() - start,
            **aucs, "AUC_macro": float(np.mean(list(aucs.values())))}
