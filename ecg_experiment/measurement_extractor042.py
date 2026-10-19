"""Forward port of Avance2.8 raw measurements with unchanged arithmetic.

The six original functions bind cache ee39c8f6f7. The array entry point only
replaces its raw-file read; no measurement or plausibility rule is changed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import stats
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, find_peaks, sosfiltfilt, welch

FS = 500
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
INDEPENDENT = [0, 1, 6, 7, 8, 9, 10, 11]
PRE, POST = int(0.35 * FS), int(0.55 * FS)
SOS_CLEAN = butter(3, [0.5, 40], btype="band", fs=FS, output="sos")
SOS_QRS = butter(2, [5, 15], btype="band", fs=FS, output="sos")
PLAUSIBLE = {"pr_ms": (60, 400), "qrs_dur_ms": (40, 220), "qt_ms": (200, 650)}
FEATURE_NAMES = (
    [
        "noise_drift_frac",
        "noise_hf_frac",
        "noise_powerline_frac",
        "sp_low_rel",
        "sp_mid_rel",
        "sp_high_rel",
        "sp_entropy",
        "sp_centroid_hz",
    ]
    + [f"{measure}_{lead}" for lead in LEADS for measure in ("std", "ptp", "kurt")]
    + [
        "n_beats",
        "beat_similarity",
        "rr_mean_ms",
        "rr_sdnn_ms",
        "rr_rmssd_ms",
        "rr_range_rel",
        "hr_bpm",
        "qrs_dur_ms",
        "qt_ms",
        "pq_peak_ms",
        "p_rms_mv",
        "t_peak_rms_mv",
        "pr_ms",
        "p_amp_II",
    ]
    + [
        f"{measure}_{lead}"
        for lead in LEADS
        for measure in ("r_amp", "s_depth", "q_depth", "qrs_area", "st", "t_amp")
    ]
)


def read_ecg(record_path: Path) -> np.ndarray:
    """Read PTB-XL format16 raw integers with original mV scaling.

    Parameters
    ----------
    record_path : Path
        WFDB record path without its .dat suffix.

    Returns
    -------
    np.ndarray
        Float64 samples by 12 leads in mV.
    """
    return np.fromfile(f"{record_path}.dat", dtype="<i2").reshape(-1, 12) / 1000.0


def detect_r_peaks(clean: np.ndarray) -> np.ndarray:
    """Find R peaks from slope energy in eight independent leads.

    Parameters
    ----------
    clean : np.ndarray
        Filtered samples by canonical leads.

    Returns
    -------
    np.ndarray
        Ascending sample locations, excluding truncated edge complexes.
    """
    qrs_band = sosfiltfilt(SOS_QRS, clean[:, INDEPENDENT], axis=0)
    slope_energy = np.sum(np.gradient(qrs_band, axis=0) ** 2, axis=1)
    envelope = np.sqrt(uniform_filter1d(slope_energy, size=int(0.12 * FS)))
    candidates, _ = find_peaks(envelope, height=0.35 * np.percentile(envelope, 98), distance=int(0.25 * FS))
    magnitude = np.sqrt(np.sum(qrs_band**2, axis=1))
    half = int(0.06 * FS)
    peaks = np.unique(
        [max(c - half, 0) + int(np.argmax(magnitude[max(c - half, 0) : c + half])) for c in candidates]
    )
    edge = int(0.15 * FS)
    return peaks[(peaks >= edge) & (peaks < len(clean) - edge)]


def median_beat(clean: np.ndarray, peaks: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the median beat and average correlation to its template.

    Parameters
    ----------
    clean : np.ndarray
        Filtered samples by canonical leads.
    peaks : np.ndarray
        R-peak sample locations.

    Returns
    -------
    tuple[np.ndarray, float]
        Median beat and correlation; missing when fewer than two beats fit.
    """
    inside = peaks[(peaks - PRE >= 0) & (peaks + POST <= len(clean))]
    if len(inside) < 2:
        return np.full((PRE + POST, 12), np.nan), np.nan
    beats = np.stack([clean[p - PRE : p + POST] for p in inside])
    template = np.median(beats, axis=0)
    core = slice(PRE - 50, PRE + 200)
    reference = template[core][:, INDEPENDENT].ravel()
    similarity = [np.corrcoef(b[core][:, INDEPENDENT].ravel(), reference)[0, 1] for b in beats]
    return template, float(np.mean(similarity))


def delineate(template: np.ndarray, rr_mean_ms: float) -> dict[str, object]:
    """Measure fiducials of the median beat using original notebook rules.

    Parameters
    ----------
    template : np.ndarray
        Median beat with the R peak at PRE.
    rr_mean_ms : float
        Mean interbeat interval in milliseconds.

    Returns
    -------
    dict[str, object]
        Fiducial locations, centered beat and RMS envelope.
    """
    slope = np.sqrt(np.sum(np.gradient(template[:, INDEPENDENT], axis=0) ** 2, axis=1))
    slope = uniform_filter1d(slope, size=5)
    quiet = slope < 0.06 * slope[PRE - 50 : PRE + 50].max()
    onset = next((i for i in range(PRE, PRE - 100, -1) if quiet[i - 8 : i].all()), PRE - 100)
    offset = next((i for i in range(PRE, PRE + 125) if quiet[i : i + 8].all()), PRE + 125)
    centered = template - np.median(template[onset - 12 : onset - 2], axis=0)
    rms = uniform_filter1d(np.sqrt(np.mean(centered[:, INDEPENDENT] ** 2, axis=1)), size=10)

    t_start = offset + int(0.08 * FS)
    t_stop = min(len(rms) - 1, onset + int(min(0.60, 0.70 * rr_mean_ms / 1000.0) * FS))
    t_stop = max(t_stop, t_start + 20)
    t_peak = t_start + int(np.argmax(rms[t_start:t_stop]))
    tail = min(len(rms) - 1, t_stop + int(0.06 * FS))
    descent = np.gradient(rms)[t_peak:tail]
    t_end = np.nan
    if len(descent) > 2 and descent.min() < 0:
        steepest = t_peak + int(np.argmin(descent))
        t_end = min(steepest + (rms[steepest : tail + 1].min() - rms[steepest]) / descent.min(), tail)

    p_start, p_stop = max(onset - int(0.30 * FS), 0), onset - int(0.04 * FS)
    p_peak = p_start + int(np.argmax(rms[p_start:p_stop]))
    p_wave = centered[p_start:p_stop, 1] - np.median(centered[p_start:p_stop, 1])
    p_index = int(np.argmax(np.abs(p_wave)))
    p_amp, p_onset = float(p_wave[p_index]), np.nan
    if abs(p_amp) >= 0.05 and p_index > int(0.02 * FS):
        search = max(p_index - int(0.10 * FS), 0)
        returned = np.flatnonzero(np.abs(p_wave[search:p_index]) < 0.2 * abs(p_amp))
        if len(returned):
            p_onset = p_start + search + int(returned[-1])
    return {
        "onset": onset,
        "offset": offset,
        "centered": centered,
        "rms": rms,
        "t_start": t_start,
        "t_stop": t_stop,
        "t_peak": t_peak,
        "t_end": t_end,
        "p_start": p_start,
        "p_stop": p_stop,
        "p_peak": p_peak,
        "p_amp": p_amp,
        "p_onset": p_onset,
    }


def spectral_features(raw: np.ndarray) -> dict[str, float]:
    """Return original relative power and spectral shape measurements.

    Parameters
    ----------
    raw : np.ndarray
        Float64 samples by canonical leads in mV.

    Returns
    -------
    dict[str, float]
        Original eight spectral measurements.
    """
    freqs, power = welch(raw[:, INDEPENDENT], fs=FS, nperseg=1024, axis=0)
    power = power.mean(axis=1)

    def band(low: float, high: float) -> float:
        return float(power[(freqs > low) & (freqs <= high)].sum())

    total, useful = band(0, 100), band(0.5, 40)
    in_band = (freqs > 0.5) & (freqs <= 40)
    share = power[in_band] / useful
    return {
        "noise_drift_frac": band(0, 0.5) / total,
        "noise_hf_frac": band(40, 100) / total,
        "noise_powerline_frac": band(48, 52) / total,
        "sp_low_rel": band(0.5, 5) / useful,
        "sp_mid_rel": band(5, 15) / useful,
        "sp_high_rel": band(15, 40) / useful,
        "sp_entropy": float(-(share * np.log(share + 1e-12)).sum() / np.log(len(share))),
        "sp_centroid_hz": float((freqs[in_band] * share).sum()),
    }


def extract_array(raw: np.ndarray) -> dict[str, object]:
    """Extract the original Avance measurements from samples by leads in mV.

    Parameters
    ----------
    raw : np.ndarray
        Finite float64 array of shape (5000, 12), 500 Hz, physical mV.

    Returns
    -------
    dict[str, object]
        Original measurements and quality status. Early failures retain available
        measurements; callers fill absent model features with missing values.

    Raises
    ------
    ValueError
        If shape, dtype or finiteness violates the array contract.
    """
    if raw.shape != (5000, 12) or raw.dtype != np.float64 or not np.isfinite(raw).all():
        raise ValueError("Expected finite float64 samples-by-leads array (5000, 12)")
    feats = {"status": "ok", "implausible": "", "flat_leads": float((np.ptp(raw, axis=0) == 0).sum())}
    if feats["flat_leads"] > 0:
        feats["status"] = "derivacion_plana"
        return feats
    clean = sosfiltfilt(SOS_CLEAN, raw, axis=0)
    feats.update(spectral_features(raw))
    for j, lead in enumerate(LEADS):
        feats[f"std_{lead}"] = float(clean[:, j].std())
        feats[f"ptp_{lead}"] = float(np.ptp(clean[:, j]))
        feats[f"kurt_{lead}"] = float(stats.kurtosis(clean[:, j]))

    peaks = detect_r_peaks(clean)
    rr = np.diff(peaks) / FS * 1000.0
    feats["n_beats"] = float(len(peaks))
    template, feats["beat_similarity"] = median_beat(clean, peaks)
    if len(rr) < 2 or np.isnan(template).any():
        feats["status"] = "sin_latidos"
        return feats
    feats["rr_mean_ms"] = float(rr.mean())
    feats["rr_sdnn_ms"] = float(rr.std(ddof=1))
    feats["rr_rmssd_ms"] = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
    feats["rr_range_rel"] = float((rr.max() - rr.min()) / rr.mean())
    feats["hr_bpm"] = float(60000.0 / rr.mean())

    m = delineate(template, rr.mean())
    onset, offset, centered, rms = m["onset"], m["offset"], m["centered"], m["rms"]
    feats["qrs_dur_ms"] = (offset - onset) / FS * 1000.0
    feats["qt_ms"] = (m["t_end"] - onset) / FS * 1000.0
    feats["pq_peak_ms"] = (onset - m["p_peak"]) / FS * 1000.0
    feats["p_rms_mv"] = float(rms[m["p_peak"]])
    feats["t_peak_rms_mv"] = float(rms[m["t_peak"]])
    feats["pr_ms"] = (onset - m["p_onset"]) / FS * 1000.0
    feats["p_amp_II"] = m["p_amp"] if np.isfinite(m["p_onset"]) else np.nan
    for j, lead in enumerate(LEADS):
        qrs = centered[onset : offset + 1, j]
        r_index = int(np.argmax(qrs))
        has_r = qrs.max() > 0
        feats[f"r_amp_{lead}"] = float(max(qrs.max(), 0.0))
        feats[f"s_depth_{lead}"] = float(max(-qrs.min(), 0.0))
        feats[f"q_depth_{lead}"] = float(max(-qrs[:r_index].min(), 0.0)) if has_r and r_index > 0 else 0.0
        feats[f"qrs_area_{lead}"] = float(qrs.sum() / FS * 1000.0)
        feats[f"st_{lead}"] = float(centered[offset + 30 : offset + 40, j].mean())
        t_wave = centered[m["t_start"] : m["t_stop"], j]
        feats[f"t_amp_{lead}"] = float(t_wave[np.argmax(np.abs(t_wave))])

    failed = [
        k for k, (low, high) in PLAUSIBLE.items() if np.isfinite(feats[k]) and not low <= feats[k] <= high
    ]
    for k in failed:
        feats[k] = np.nan
    if "pr_ms" in failed:
        feats["p_amp_II"] = np.nan
    feats["implausible"] = ",".join(failed)
    return feats
