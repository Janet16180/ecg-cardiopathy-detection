"""Signal features of Avance 2 (``notebooks/Avance2.8.ipynb``) for any 500 Hz 12-lead ECG array.

Experiment 058 recomputes the 111 variables that the Avance 4.8 SVM used, on the balanced dataset v2. The
extraction, the derived variables and the preprocessing are ported from the Avance 2 notebook cells 15, 20,
32, 46, 51, 54, 57 and 59; the only change is that ``extract`` receives an array instead of a PTB-XL path.
All transforms are fitted on the training ECGs given to ``Preprocessor.fit``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, find_peaks, sosfiltfilt, welch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, StandardScaler

FS = 500
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
INDEPENDENT = [0, 1, 6, 7, 8, 9, 10, 11]  # I, II, V1-V6
PRE, POST = int(0.35 * FS), int(0.55 * FS)
SOS_CLEAN = butter(3, [0.5, 40], btype="band", fs=FS, output="sos")
SOS_QRS = butter(2, [5, 15], btype="band", fs=FS, output="sos")
PLAUSIBLE = {"pr_ms": (60, 400), "qrs_dur_ms": (40, 220), "qt_ms": (200, 650)}
TERRITORIES = {"anterior": ["V1", "V2", "V3", "V4"], "inferior": ["II", "III", "aVF"], "lateral": ["I", "aVL", "V5", "V6"]}
CYCLIC_SOURCE = ["qrs_axis_deg", "t_axis_deg"]
COUNTS = ["n_t_inverted", "n_q_pathologic", "r_transition_lead"]
BINARY = ["low_voltage_limb", "flag_no_p_wave", "flag_medicion_fallida"]
ORDINAL = {"hr_class": ["bradicardia", "normal", "taquicardia"], "qrs_class": ["estrecho", "intermedio", "ancho"],
           "qtc_class": ["normal", "limitrofe", "prolongado"], "q_patho_class": ["0", "1", "2", "3+"]}
NOMINAL = {"axis_class": ["izquierda", "derecha", "extrema"], "r_transition_class": ["temprana", "tardia"]}
SKEW_MIN = 0.5


def detect_r_peaks(clean: np.ndarray) -> np.ndarray:
    """QRS positions from the slope energy summed over the eight independent leads (Avance 2, cell 15)."""
    qrs_band = sosfiltfilt(SOS_QRS, clean[:, INDEPENDENT], axis=0)
    slope_energy = np.sum(np.gradient(qrs_band, axis=0) ** 2, axis=1)
    envelope = np.sqrt(uniform_filter1d(slope_energy, size=int(0.12 * FS)))
    candidates, _ = find_peaks(envelope, height=0.35 * np.percentile(envelope, 98), distance=int(0.25 * FS))
    magnitude = np.sqrt(np.sum(qrs_band ** 2, axis=1))
    half = int(0.06 * FS)
    peaks = np.unique([max(c - half, 0) + int(np.argmax(magnitude[max(c - half, 0): c + half])) for c in candidates])
    edge = int(0.15 * FS)
    return peaks[(peaks >= edge) & (peaks < len(clean) - edge)]


def median_beat(clean: np.ndarray, peaks: np.ndarray) -> tuple[np.ndarray, float]:
    """Median beat of each lead and mean correlation of every beat with it (Avance 2, cell 15)."""
    inside = peaks[(peaks - PRE >= 0) & (peaks + POST <= len(clean))]
    if len(inside) < 2:
        return np.full((PRE + POST, 12), np.nan), np.nan
    beats = np.stack([clean[p - PRE: p + POST] for p in inside])
    template = np.median(beats, axis=0)
    core = slice(PRE - 50, PRE + 200)
    reference = template[core][:, INDEPENDENT].ravel()
    similarity = [np.corrcoef(b[core][:, INDEPENDENT].ravel(), reference)[0, 1] for b in beats]
    return template, float(np.mean(similarity))


def delineate(template: np.ndarray, rr_mean_ms: float) -> dict:
    """QRS onset and offset, T peak and end, P peak and onset on the median beat (Avance 2, cell 15)."""
    slope = np.sqrt(np.sum(np.gradient(template[:, INDEPENDENT], axis=0) ** 2, axis=1))
    slope = uniform_filter1d(slope, size=5)
    quiet = slope < 0.06 * slope[PRE - 50: PRE + 50].max()
    onset = next((i for i in range(PRE, PRE - 100, -1) if quiet[i - 8: i].all()), PRE - 100)
    offset = next((i for i in range(PRE, PRE + 125) if quiet[i: i + 8].all()), PRE + 125)
    centered = template - np.median(template[onset - 12: onset - 2], axis=0)
    rms = uniform_filter1d(np.sqrt(np.mean(centered[:, INDEPENDENT] ** 2, axis=1)), size=10)

    t_start = offset + int(0.08 * FS)
    t_stop = min(len(rms) - 1, onset + int(min(0.60, 0.70 * rr_mean_ms / 1000.0) * FS))
    t_stop = max(t_stop, t_start + 20)
    t_peak = t_start + int(np.argmax(rms[t_start: t_stop]))
    tail = min(len(rms) - 1, t_stop + int(0.06 * FS))
    descent = np.gradient(rms)[t_peak: tail]
    t_end = np.nan
    if len(descent) > 2 and descent.min() < 0:
        steepest = t_peak + int(np.argmin(descent))
        t_end = min(steepest + (rms[steepest: tail + 1].min() - rms[steepest]) / descent.min(), tail)

    p_start, p_stop = max(onset - int(0.30 * FS), 0), onset - int(0.04 * FS)
    p_peak = p_start + int(np.argmax(rms[p_start: p_stop]))
    p_wave = centered[p_start: p_stop, 1] - np.median(centered[p_start: p_stop, 1])
    p_index = int(np.argmax(np.abs(p_wave)))
    p_amp, p_onset = float(p_wave[p_index]), np.nan
    if abs(p_amp) >= 0.05 and p_index > int(0.02 * FS):
        search = max(p_index - int(0.10 * FS), 0)
        returned = np.flatnonzero(np.abs(p_wave[search: p_index]) < 0.2 * abs(p_amp))
        if len(returned):
            p_onset = p_start + search + int(returned[-1])
    return {"onset": onset, "offset": offset, "centered": centered, "rms": rms, "t_start": t_start, "t_stop": t_stop,
            "t_peak": t_peak, "t_end": t_end, "p_start": p_start, "p_stop": p_stop, "p_peak": p_peak,
            "p_amp": p_amp, "p_onset": p_onset}


def spectral_features(raw: np.ndarray) -> dict[str, float]:
    """Noise shares and spectral shape of the independent leads (Avance 2, cell 20)."""
    freqs, power = welch(raw[:, INDEPENDENT], fs=FS, nperseg=1024, axis=0)
    power = power.mean(axis=1)

    def band(low: float, high: float) -> float:
        return float(power[(freqs > low) & (freqs <= high)].sum())

    total, useful = band(0, 100), band(0.5, 40)
    in_band = (freqs > 0.5) & (freqs <= 40)
    share = power[in_band] / useful
    return {
        "noise_drift_frac": band(0, 0.5) / total, "noise_hf_frac": band(40, 100) / total,
        "noise_powerline_frac": band(48, 52) / total,
        "sp_low_rel": band(0.5, 5) / useful, "sp_mid_rel": band(5, 15) / useful, "sp_high_rel": band(15, 40) / useful,
        "sp_entropy": float(-(share * np.log(share + 1e-12)).sum() / np.log(len(share))),
        "sp_centroid_hz": float((freqs[in_band] * share).sum()),
    }


def extract(raw: np.ndarray) -> dict:
    """
    Primary features of one ECG (Avance 2, cell 20, reading an array instead of a file).

    Parameters
    ----------
    raw : np.ndarray
        ``(samples, 12)`` in mV at 500 Hz.

    Returns
    -------
    dict
        ``status``, ``implausible`` and every primary feature; implausible intervals become NaN.
    """
    raw = np.asarray(raw, dtype=np.float64)
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
        qrs = centered[onset: offset + 1, j]
        r_index = int(np.argmax(qrs))
        has_r = qrs.max() > 0
        feats[f"r_amp_{lead}"] = float(max(qrs.max(), 0.0))
        feats[f"s_depth_{lead}"] = float(max(-qrs.min(), 0.0))
        feats[f"q_depth_{lead}"] = float(max(-qrs[:r_index].min(), 0.0)) if has_r and r_index > 0 else 0.0
        feats[f"qrs_area_{lead}"] = float(qrs.sum() / FS * 1000.0)
        feats[f"st_{lead}"] = float(centered[offset + 30: offset + 40, j].mean())
        t_wave = centered[m["t_start"]: m["t_stop"], j]
        feats[f"t_amp_{lead}"] = float(t_wave[np.argmax(np.abs(t_wave))])

    failed = [k for k, (low, high) in PLAUSIBLE.items() if np.isfinite(feats[k]) and not low <= feats[k] <= high]
    for k in failed:
        feats[k] = np.nan
    if "pr_ms" in failed:
        feats["p_amp_II"] = np.nan
    feats["implausible"] = ",".join(failed)
    return feats


def derive(extracted: pd.DataFrame) -> pd.DataFrame:
    """
    Primary plus derived features (Avance 2, cells 21 and 32).

    Parameters
    ----------
    extracted : pd.DataFrame
        One row per ECG, the output of ``extract``.

    Returns
    -------
    pd.DataFrame
        Numeric features, without ``status``, ``implausible`` and ``flat_leads``.
    """
    measured = extracted["status"] == "ok"
    features = extracted.drop(columns=["status", "implausible", "flat_leads"]).astype(float)
    features["flag_no_p_wave"] = features.pr_ms.isna().astype(float)
    features["flag_medicion_fallida"] = (features.qrs_dur_ms.isna() | features.qt_ms.isna()).astype(float)
    rr_seconds = features.rr_mean_ms / 1000.0
    features["sokolow_lyon_mv"] = features.s_depth_V1 + features[["r_amp_V5", "r_amp_V6"]].max(axis=1)
    features["cornell_mv"] = features.r_amp_aVL + features.s_depth_V3
    features["cornell_product_mvms"] = features.cornell_mv * features.qrs_dur_ms
    features["qtc_bazett_ms"] = features.qt_ms / np.sqrt(rr_seconds)
    features["qtc_fridericia_ms"] = features.qt_ms / np.cbrt(rr_seconds)
    features["qrs_axis_deg"] = np.degrees(np.arctan2(2 * features.qrs_area_aVF, np.sqrt(3) * features.qrs_area_I))
    features["t_axis_deg"] = np.degrees(np.arctan2(2 * features.t_amp_aVF, np.sqrt(3) * features.t_amp_I))
    gap = (features.qrs_axis_deg - features.t_axis_deg).abs()
    features["qrs_t_angle_deg"] = gap.where(gap <= 180, 360 - gap)
    st = features[[f"st_{lead}" for lead in LEADS]].set_axis(LEADS, axis=1)
    for territory, leads in TERRITORIES.items():
        features[f"st_elev_{territory}_mv"] = st[leads].max(axis=1)
        features[f"st_dep_{territory}_mv"] = st[leads].min(axis=1)
    features["st_min_mv"] = st.drop(columns="aVR").min(axis=1)
    features["st_max_mv"] = st.drop(columns="aVR").max(axis=1)
    upright = ["I", "II", "V3", "V4", "V5", "V6"]
    features["n_t_inverted"] = (features[[f"t_amp_{lead}" for lead in upright]] < -0.1).sum(axis=1).where(measured)
    q_leads = ["I", "II", "aVL", "aVF", "V2", "V3", "V4", "V5", "V6"]
    q_depth = features[[f"q_depth_{lead}" for lead in q_leads]].to_numpy()
    r_height = features[[f"r_amp_{lead}" for lead in q_leads]].to_numpy()
    features["n_q_pathologic"] = pd.Series(((q_depth >= 0.25 * r_height) & (q_depth >= 0.1)).sum(axis=1),
                                           index=features.index).where(measured)
    precordial = ["V1", "V2", "V3", "V4", "V5", "V6"]
    r_share = np.stack([features[f"r_amp_{lead}"] / (features[f"r_amp_{lead}"] + features[f"s_depth_{lead}"] + 1e-9)
                        for lead in precordial], axis=1)
    features["r_frac_V1"] = r_share[:, 0]
    dominant = r_share >= 0.5
    features["r_transition_lead"] = pd.Series(np.where(dominant.any(axis=1), dominant.argmax(axis=1) + 1, 7),
                                              index=features.index).where(measured)
    features["r_prog_slope_mv"] = (-1.5 * features.r_amp_V1 - 0.5 * features.r_amp_V2 + 0.5 * features.r_amp_V3
                                   + 1.5 * features.r_amp_V4) / 5
    limb = ["I", "II", "III", "aVR", "aVL", "aVF"]
    qrs_height = pd.concat([(features[f"r_amp_{lead}"] + features[f"s_depth_{lead}"]).rename(lead) for lead in limb], axis=1)
    features["low_voltage_limb"] = (qrs_height.max(axis=1) < 0.5).astype(float).where(measured)
    return features


def features_one(raw: np.ndarray, primary_columns: list[str]) -> pd.DataFrame:
    """
    Primary plus derived features of a single ECG, laid out like a training table.

    Parameters
    ----------
    raw : np.ndarray
        ``(samples, 12)`` in mV at 500 Hz.
    primary_columns : list[str]
        Columns of the ``extract`` table the model was trained on; a key that ``extract`` did not return for
        this ECG (for example after a flat lead) becomes NaN.

    Returns
    -------
    pd.DataFrame
        One row, the output of ``derive``.
    """
    extracted = pd.DataFrame([extract(raw)]).reindex(columns=primary_columns)
    extracted["implausible"] = extracted["implausible"].fillna("")
    return derive(extracted)


class Preprocessor:
    """
    Avance 2 preprocessing fitted on training ECGs (cells 46, 51, 54, 57 and 59).

    Median imputation; clipping of continuous features to their 0.5 and 99.5 percentiles; division by the
    training deviation and Yeo-Johnson for those with skew above 0.5; categories as ordinal codes, one-hot
    columns (reference ``normal``) and binaries; axes as sine and cosine. As in Avance 2, every column is
    standardized at the end with the training mean and deviation (``StandardScaler``).

    Parameters
    ----------
    columns : list[str]
        Output columns kept, for example the 111 of ``outputs/avance2_fusion/variables_finales.csv``.
    """

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def _encoded(self, features: pd.DataFrame) -> pd.DataFrame:
        imputed = self.imputer.transform(features)
        clean = imputed.copy()
        clean[self.continuous] = imputed[self.continuous].clip(self.lower, self.upper, axis=1)
        power = clean[self.continuous] / self.scale
        power[self.to_transform] = self.power.transform(power[self.to_transform])
        categorical = pd.DataFrame(index=features.index)
        categorical["hr_class"] = np.select([imputed.hr_bpm < 60, imputed.hr_bpm >= 100], ["bradicardia", "taquicardia"], "normal")
        categorical["qrs_class"] = np.select([imputed.qrs_dur_ms < 110, imputed.qrs_dur_ms < 120], ["estrecho", "intermedio"], "ancho")
        categorical["qtc_class"] = np.select([imputed.qtc_bazett_ms < 450, imputed.qtc_bazett_ms < 480], ["normal", "limitrofe"], "prolongado")
        axis = imputed.qrs_axis_deg
        categorical["axis_class"] = np.select([(axis >= -30) & (axis <= 90), (axis < -30) & (axis >= -90), axis > 90],
                                              ["normal", "izquierda", "derecha"], "extrema")
        categorical["r_transition_class"] = np.select([imputed.r_transition_lead <= 2, imputed.r_transition_lead <= 4],
                                                      ["temprana", "normal"], "tardia")
        categorical["q_patho_class"] = imputed.n_q_pathologic.clip(upper=3).astype(int).astype(str).replace("3", "3+")
        ordinal = pd.DataFrame({name: categorical[name].map({value: code for code, value in enumerate(order)})
                                for name, order in ORDINAL.items()})
        dummies = pd.DataFrame({f"{name}_{value}": (categorical[name] == value).astype(float)
                                for name, values in NOMINAL.items() for value in values})
        binary = pd.concat([(imputed.n_t_inverted >= 1).astype(float).rename("t_inverted_any"), imputed[BINARY]], axis=1)
        cyclic = pd.DataFrame({
            "qrs_axis_sin": np.sin(np.radians(imputed.qrs_axis_deg)), "qrs_axis_cos": np.cos(np.radians(imputed.qrs_axis_deg)),
            "t_axis_sin": np.sin(np.radians(imputed.t_axis_deg)), "t_axis_cos": np.cos(np.radians(imputed.t_axis_deg)),
        })
        return pd.concat([power, ordinal, dummies, binary, cyclic], axis=1)[self.columns]

    def fit(self, features: pd.DataFrame) -> "Preprocessor":
        """Fit every transform on the training features (labels are not used)."""
        self.continuous = [c for c in features.columns if c not in CYCLIC_SOURCE + COUNTS + BINARY]
        self.imputer = SimpleImputer(strategy="median").set_output(transform="pandas").fit(features)
        imputed = self.imputer.transform(features)
        self.lower = imputed[self.continuous].quantile(0.005)
        self.upper = imputed[self.continuous].quantile(0.995)
        clean = imputed[self.continuous].clip(self.lower, self.upper, axis=1)
        skew = clean.skew()
        self.to_transform = list(skew.index[skew.abs() > SKEW_MIN])
        self.scale = clean.std().replace(0, 1)
        self.power = PowerTransformer(method="yeo-johnson", standardize=False).set_output(transform="pandas").fit(
            (clean / self.scale)[self.to_transform])
        self.scaler = StandardScaler().set_output(transform="pandas").fit(self._encoded(features))
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """The kept columns, transformed and standardized."""
        return self.scaler.transform(self._encoded(features))
