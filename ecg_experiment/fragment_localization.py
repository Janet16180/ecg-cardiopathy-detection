"""Per-section anomaly maps of xECG tokens for Experiment 041: which 250 ms of an ECG look abnormal."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from matplotlib.figure import Figure
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import nn

from .external_encoders import XECG_BATCH, XECG_DIMENSION
from .intervals import patient_groups, patient_resample
from .normal_manifold import fit_mahalanobis, mahalanobis_scores
from .waveforms import LEADS

SECTIONS = 40
SECTION_SECONDS = 0.25
PATCH_SAMPLES = 25
CLUSTERS = 32
CLUSTER_INITS = 4
BUDGET = 0.05
PREMATURE_RATIO = 0.8
BEAT_BEFORE_SECONDS = 0.10
BEAT_AFTER_SECONDS = 0.40
MINIMUM_PEAKS = 4


@dataclass
class SectionReference:
    """Normal-section reference fitted on tokens of normal ECGs."""

    position_means: np.ndarray
    centred: tuple[StandardScaler, PCA, LedoitWolf]
    uncentred: tuple[StandardScaler, PCA, LedoitWolf]
    whitening: np.ndarray
    centroids: np.ndarray


@torch.inference_mode()
def xecg_tokens(model: nn.Module, inputs: np.ndarray) -> np.ndarray:
    """
    Extract the 40 output tokens of the released xECG backbone, in microbatches of sixteen.

    Parameters
    ----------
    model : nn.Module
        Output of ``external_encoders.load_xecg_backbone`` (or ``xecg.load_xecg`` on the CPU).
    inputs : np.ndarray
        Float32 ``[records, 1000, 12]`` inputs from ``external_encoders.xecg_input``.

    Returns
    -------
    np.ndarray
        Float32 ``[records, 40, 1024]`` tokens; their mean over sections is the pooled feature.

    Raises
    ------
    ValueError
        If the output is malformed or nonfinite.
    """
    device = next(model.parameters()).device
    chunks = []
    for start in range(0, len(inputs), XECG_BATCH):
        tensor = torch.from_numpy(np.ascontiguousarray(inputs[start:start + XECG_BATCH])).to(device)
        _, tokens = model(tensor)
        chunks.append(tokens.float().cpu().numpy())
    tokens = np.concatenate(chunks).astype(np.float32, copy=False)
    if tokens.shape != (len(inputs), SECTIONS, XECG_DIMENSION) or not np.isfinite(tokens).all():
        raise ValueError(f"Invalid xECG tokens: {tokens.shape}")
    return tokens


def order_check(model: nn.Module, inputs: np.ndarray) -> dict[str, list[int]]:
    """
    Token that changes most when 1 mV is added to the first or the last 25 input samples.

    Parameters
    ----------
    model : nn.Module
        xECG backbone.
    inputs : np.ndarray
        Float32 ``[records, 1000, 12]`` inputs.

    Returns
    -------
    dict[str, list[int]]
        ``first`` and ``last``: the most changed token of each record (0 and 39 when sections are in
        time order).
    """
    base = xecg_tokens(model, inputs)
    result = {}
    for name, part in (("first", slice(0, PATCH_SAMPLES)), ("last", slice(-PATCH_SAMPLES, None))):
        shifted = inputs.copy()
        shifted[:, part] += 1.0
        change = np.linalg.norm(xecg_tokens(model, shifted) - base, axis=2)
        result[name] = change.argmax(axis=1).tolist()
    return result


def fit_section_reference(tokens: np.ndarray, seed: int) -> SectionReference:
    """
    Fit the per-position centring, the Mahalanobis models and the k-means centroids on normal tokens.

    Parameters
    ----------
    tokens : np.ndarray
        ``[records, 40, width]`` tokens of normal ECGs only.
    seed : int
        ``random_state`` of k-means.

    Returns
    -------
    SectionReference
        The fitted reference.
    """
    tokens = np.asarray(tokens, dtype=np.float64)
    position_means = tokens.mean(axis=0)
    flat = (tokens - position_means).reshape(-1, tokens.shape[2])
    centred = fit_mahalanobis(flat)
    scaler, pca, _ = centred
    whitening = np.sqrt(pca.explained_variance_)
    coordinates = pca.transform(scaler.transform(flat)) / whitening
    centroids = KMeans(n_clusters=CLUSTERS, n_init=CLUSTER_INITS, random_state=seed).fit(coordinates)
    uncentred = fit_mahalanobis(tokens.reshape(-1, tokens.shape[2]))
    return SectionReference(position_means, centred, uncentred, whitening, centroids.cluster_centers_)


def section_mahalanobis(reference: SectionReference, tokens: np.ndarray, centred: bool = True) -> np.ndarray:
    """
    Squared Mahalanobis distance of every section from the normal sections.

    Parameters
    ----------
    reference : SectionReference
        Output of ``fit_section_reference``.
    tokens : np.ndarray
        ``[records, 40, width]`` tokens.
    centred : bool
        Subtract the normal mean of each position first (map ``U``); otherwise ``U_uncentred``.

    Returns
    -------
    np.ndarray
        ``[records, 40]`` scores.
    """
    tokens = np.asarray(tokens, dtype=np.float64)
    model = reference.centred if centred else reference.uncentred
    if centred:
        tokens = tokens - reference.position_means
    return mahalanobis_scores(model, tokens.reshape(-1, tokens.shape[2])).reshape(tokens.shape[:2])


def section_kmeans(reference: SectionReference, tokens: np.ndarray) -> np.ndarray:
    """
    Squared distance of every centred, whitened section to its nearest normal cluster (map ``U_kmeans``).

    Parameters
    ----------
    reference : SectionReference
        Output of ``fit_section_reference``.
    tokens : np.ndarray
        ``[records, 40, width]`` tokens.

    Returns
    -------
    np.ndarray
        ``[records, 40]`` scores.
    """
    tokens = np.asarray(tokens, dtype=np.float64) - reference.position_means
    scaler, pca, _ = reference.centred
    coordinates = pca.transform(scaler.transform(tokens.reshape(-1, tokens.shape[2]))) / reference.whitening
    centroids = reference.centroids
    squared = ((coordinates ** 2).sum(axis=1, keepdims=True) - 2 * coordinates @ centroids.T
               + (centroids ** 2).sum(axis=1))
    return np.maximum(squared.min(axis=1), 0.0).reshape(tokens.shape[:2])


def readout_contributions(head: tuple[StandardScaler, LogisticRegression], tokens: np.ndarray) -> np.ndarray:
    """
    Per-section logit contributions of a linear readout of the mean token (map ``G``).

    Parameters
    ----------
    head : tuple[StandardScaler, LogisticRegression]
        Output of ``full_development.fit_logistic`` on pooled features.
    tokens : np.ndarray
        ``[records, 40, width]`` tokens.

    Returns
    -------
    np.ndarray
        ``[records, 40]`` contributions whose mean over sections is the readout logit.
    """
    scaler, model = head
    standardized = (np.asarray(tokens, dtype=np.float64) - scaler.mean_) / scaler.scale_
    return standardized @ model.coef_[0] + model.intercept_[0]


def gated(unsupervised: np.ndarray, contributions: np.ndarray) -> np.ndarray:
    """
    ``U`` where the readout contribution is positive, 0 elsewhere (map ``G_gated``).

    Parameters
    ----------
    unsupervised : np.ndarray
        ``[records, 40]`` map ``U``.
    contributions : np.ndarray
        ``[records, 40]`` map ``G``.

    Returns
    -------
    np.ndarray
        ``[records, 40]`` scores.
    """
    return np.where(contributions > 0, unsupervised, 0.0)


def red_threshold(normal_maps: np.ndarray, budget: float = BUDGET) -> float:
    """
    Threshold that leaves ``budget`` of the normal ECGs with at least one red section.

    Parameters
    ----------
    normal_maps : np.ndarray
        ``[records, 40]`` maps of normal ECGs.
    budget : float
        Share of normal ECGs allowed a red section.

    Returns
    -------
    float
        The ``1 - budget`` quantile of the normal ECG scores (maximum over sections).
    """
    return float(np.quantile(normal_maps.max(axis=1), 1 - budget))


def r_peaks(signal: np.ndarray, fs: int) -> np.ndarray:
    """
    R-peak samples from QRS energy summed over all leads, the rule of ``eda.signals.heart_rate``.

    Parameters
    ----------
    signal : np.ndarray
        ``[12, samples]`` waveform in mV.
    fs : int
        Sampling rate in Hz.

    Returns
    -------
    np.ndarray
        Ascending peak positions in samples.
    """
    b, a = butter(2, [8, 20], btype="band", fs=fs)
    energy = np.abs(filtfilt(b, a, np.nan_to_num(signal), axis=1)).sum(axis=0)
    peaks, _ = find_peaks(energy, distance=int(0.3 * fs), height=0.35 * np.percentile(energy, 99))
    return peaks


def premature_windows(peaks: np.ndarray, fs: int) -> list[tuple[float, float]]:
    """
    Windows in seconds around beats that come early relative to the record's median RR interval.

    Parameters
    ----------
    peaks : np.ndarray
        Output of ``r_peaks``.
    fs : int
        Sampling rate in Hz.

    Returns
    -------
    list[tuple[float, float]]
        ``(R - 0.10 s, R + 0.40 s)`` of each premature beat; empty with fewer than four peaks.
    """
    if len(peaks) < MINIMUM_PEAKS:
        return []
    rr = np.diff(peaks)
    early = np.flatnonzero(rr < PREMATURE_RATIO * np.median(rr)) + 1
    return [(peaks[i] / fs - BEAT_BEFORE_SECONDS, peaks[i] / fs + BEAT_AFTER_SECONDS) for i in early]


def section_overlap(windows: list[tuple[float, float]]) -> np.ndarray:
    """
    Sections that overlap any window.

    Parameters
    ----------
    windows : list[tuple[float, float]]
        Output of ``premature_windows``.

    Returns
    -------
    np.ndarray
        Boolean ``[40]``; section t covers ``[0.25 t, 0.25 t + 0.25)`` seconds.
    """
    starts = np.arange(SECTIONS) * SECTION_SECONDS
    overlap = np.zeros(SECTIONS, dtype=bool)
    for low, high in windows:
        overlap |= (starts < high) & (starts + SECTION_SECONDS > low)
    return overlap


def pointing(maps: np.ndarray, overlap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Whether the top section hits a target, and the hit rate of a random section.

    Parameters
    ----------
    maps : np.ndarray
        ``[records, 40]`` section scores.
    overlap : np.ndarray
        Boolean ``[records, 40]`` target sections.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``hit`` (0 or 1) and ``chance`` (share of target sections) per record.
    """
    hit = overlap[np.arange(len(maps)), maps.argmax(axis=1)].astype(np.float64)
    return hit, overlap.mean(axis=1)


def bootstrap_mean(patients: np.ndarray, values: np.ndarray, draws: int, seed: int) -> dict[str, float]:
    """
    Mean of per-record values with a whole-patient bootstrap interval.

    Parameters
    ----------
    patients : np.ndarray
        Patient ID of each record.
    values : np.ndarray
        One value per record (for paired contrasts, the per-record difference).
    draws : int
        Bootstrap draws.
    seed : int
        Seed of ``numpy.random.default_rng``.

    Returns
    -------
    dict[str, float]
        ``value``, ``ci_low`` and ``ci_high`` (2.5 and 97.5 percentiles).
    """
    groups = patient_groups(patients)
    rng = np.random.default_rng(seed)
    means = [values[patient_resample(groups, rng)].mean() for _ in range(draws)]
    low, high = np.percentile(means, [2.5, 97.5])
    return {"value": float(values.mean()), "ci_low": float(low), "ci_high": float(high)}


def plot_sections(signal: np.ndarray, fs: int, red: dict[str, np.ndarray], title: str) -> Figure:
    """
    Twelve leads with the red sections of one or more maps, one column per map.

    Parameters
    ----------
    signal : np.ndarray
        ``[12, samples]`` waveform in mV, canonical lead order.
    fs : int
        Sampling rate in Hz.
    red : dict[str, np.ndarray]
        Boolean ``[40]`` red sections, keyed by the column title.
    title : str
        Figure title.

    Returns
    -------
    Figure
        The figure; the caller saves or shows it.
    """
    figure = Figure(figsize=(7 * len(red), 11), layout="constrained")
    axes = figure.subplots(len(LEADS), len(red), sharex=True, squeeze=False)
    seconds = np.arange(signal.shape[1]) / fs
    for column, (name, sections) in enumerate(red.items()):
        axes[0, column].set_title(name)
        for row, lead in enumerate(LEADS):
            axis = axes[row, column]
            for section in np.flatnonzero(sections):
                start = section * SECTION_SECONDS
                axis.axvspan(start, start + SECTION_SECONDS, color="red", alpha=0.25, linewidth=0)
            axis.plot(seconds, signal[row], color="black", linewidth=0.6)
            axis.set_ylabel(lead, rotation=0, labelpad=14)
            axis.set_yticks([])
        axes[-1, column].set_xlabel("seconds")
    figure.suptitle(title)
    return figure
