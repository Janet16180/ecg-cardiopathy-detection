"""Per-lead anomaly maps for Experiment 042: ECG-JEPA patch tokens and beat-aligned wave pieces."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch
from matplotlib.figure import Figure
from scipy.signal import butter, filtfilt
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from torch import nn

from .external_encoders import JEPA_LEADS
from .fragment_localization import r_peaks
from .intervals import patient_groups, patient_resample
from .waveforms import LEADS

JEPA_PATCHES = 50
JEPA_PATCH_SECONDS = 0.2
JEPA_TOKEN_BATCH = 16
COMPONENTS = 64
CLUSTERS = 32
CLUSTER_INITS = 4
CHUNK_ROWS = 65536
FS = 500
WAVES = {"P": (-0.25, -0.06), "QRS": (-0.06, 0.08), "ST": (0.08, 0.20), "T": (0.20, 0.45)}
BEAT_BEFORE = 0.30
BEAT_AFTER = 0.45
BASELINE = (-0.10, -0.06)
HIGH_PASS_HZ = 0.5
MINIMUM_BEATS = 2
ANTERIOR = ("V1", "V2", "V3", "V4")
INFERIOR = ("II", "III", "aVF")


@dataclass
class TokenReference:
    """Normal-token reference: position means, standardization, PCA and Ledoit-Wolf, k-means centroids."""

    position_means: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    covariance: LedoitWolf
    centroids: np.ndarray | None


@dataclass
class UnitMap:
    """Scored units of one ECG: score, lead index (canonical order) and time span in seconds."""

    scores: np.ndarray
    leads: np.ndarray
    starts: np.ndarray
    ends: np.ndarray


@torch.inference_mode()
def jepa_tokens(encoder: nn.Module, inputs: np.ndarray, batch: int = JEPA_TOKEN_BATCH) -> np.ndarray:
    """
    Compute the 400 output tokens of the released ECG-JEPA encoder: its ``representation`` without the mean.

    Parameters
    ----------
    encoder : nn.Module
        Output of ``external_encoders.load_jepa``.
    inputs : np.ndarray
        Float32 ``[records, 8, 2500]`` inputs from ``external_encoders.jepa_input``.
    batch : int
        Records per forward pass.

    Returns
    -------
    np.ndarray
        Float32 ``[records, 400, 768]`` tokens, token ``50 * lead + patch``.

    Raises
    ------
    ValueError
        If the encoder restricts leads or the output is malformed or nonfinite.
    """
    if len(encoder.leads) != encoder.c:
        raise ValueError("Lead restriction is not supported")
    device = next(encoder.parameters()).device
    mask = encoder._cross_attention_mask().to(device)
    chunks = []
    for start in range(0, len(inputs), batch):
        x = torch.from_numpy(np.ascontiguousarray(inputs[start:start + batch])).to(device)
        x = encoder.W_P(x.reshape(len(x), -1, JEPA_PATCHES))
        x = encoder.encoder_blocks(x, encoder.pos_embed, mask)
        if encoder.norm is not None:
            x = encoder.norm(x)
        chunks.append(x.float().cpu().numpy())
    tokens = np.concatenate(chunks)
    if tokens.shape[:2] != (len(inputs), len(JEPA_LEADS) * JEPA_PATCHES) or not np.isfinite(tokens).all():
        raise ValueError(f"Invalid ECG-JEPA tokens: {tokens.shape}")
    return tokens


def jepa_order_check(encoder: nn.Module, inputs: np.ndarray,
                     cells: tuple[tuple[int, int], ...]) -> dict[str, list[int]]:
    """
    Token that changes most when 1 mV is added to one lead during one patch.

    Parameters
    ----------
    encoder : nn.Module
        ECG-JEPA encoder.
    inputs : np.ndarray
        Float32 ``[records, 8, 2500]`` inputs.
    cells : tuple[tuple[int, int], ...]
        ``(lead, patch)`` pairs to perturb.

    Returns
    -------
    dict[str, list[int]]
        For each ``"lead,patch"``, the most changed token of each record
        (``50 * lead + patch`` when in order).
    """
    base = jepa_tokens(encoder, inputs)
    samples = inputs.shape[2] // JEPA_PATCHES
    result = {}
    for lead, patch in cells:
        shifted = inputs.copy()
        shifted[:, lead, patch * samples:(patch + 1) * samples] += 1.0
        change = np.linalg.norm(jepa_tokens(encoder, shifted) - base, axis=2)
        result[f"{lead},{patch}"] = change.argmax(axis=1).tolist()
    return result


def _centred_chunks(tokens: np.ndarray, position_means: np.ndarray) -> Iterator[np.ndarray]:
    """Yield float64 rows of position-centred tokens, a few ECGs at a time."""
    records = max(1, CHUNK_ROWS // tokens.shape[1])
    for start in range(0, len(tokens), records):
        block = np.asarray(tokens[start:start + records], dtype=np.float64) - position_means
        yield block.reshape(-1, tokens.shape[2])


def fit_token_reference(tokens: np.ndarray, seed: int | None, clusters: bool = True) -> TokenReference:
    """
    Fit the position-centred Mahalanobis reference in chunks, equal to ``normal_manifold.fit_mahalanobis``.

    Parameters
    ----------
    tokens : np.ndarray
        ``[records, positions, width]`` tokens of normal ECGs (float32 is fine).
    seed : int | None
        ``random_state`` of k-means.
    clusters : bool
        Also fit the k-means centroids on the whitened coordinates.

    Returns
    -------
    TokenReference
        The fitted reference.
    """
    width = tokens.shape[2]
    position_means = np.zeros(tokens.shape[1:])
    for start in range(0, len(tokens), 256):
        position_means += np.asarray(tokens[start:start + 256], dtype=np.float64).sum(axis=0)
    position_means /= len(tokens)
    rows, total, squares = 0, np.zeros(width), np.zeros(width)
    for block in _centred_chunks(tokens, position_means):
        rows += len(block)
        total += block.sum(axis=0)
        squares += (block ** 2).sum(axis=0)
    mean = total / rows
    scale = np.sqrt(np.maximum(squares / rows - mean ** 2, 0.0))
    scale[scale == 0] = 1.0
    gram = np.zeros((width, width))
    for block in _centred_chunks(tokens, position_means):
        z = (block - mean) / scale
        gram += z.T @ z
    eigenvalues, eigenvectors = np.linalg.eigh(gram / rows)
    order = np.argsort(eigenvalues)[::-1][:COMPONENTS]
    components = eigenvectors[:, order].T
    explained_variance = eigenvalues[order] * rows / (rows - 1)
    projected = np.concatenate([((block - mean) / scale) @ components.T
                                for block in _centred_chunks(tokens, position_means)])
    covariance = LedoitWolf().fit(projected)
    centroids = None
    if clusters:
        whitened = projected / np.sqrt(explained_variance)
        fitted = KMeans(n_clusters=CLUSTERS, n_init=CLUSTER_INITS, random_state=seed).fit(whitened)
        centroids = fitted.cluster_centers_
    return TokenReference(position_means, mean, scale, components, explained_variance, covariance, centroids)


def token_scores(reference: TokenReference, tokens: np.ndarray, kmeans: bool = False) -> np.ndarray:
    """
    Squared Mahalanobis distance (or squared distance to the nearest centroid) of every token.

    Parameters
    ----------
    reference : TokenReference
        Output of ``fit_token_reference``.
    tokens : np.ndarray
        ``[records, positions, width]`` tokens.
    kmeans : bool
        Score by the nearest whitened centroid instead of the Mahalanobis distance.

    Returns
    -------
    np.ndarray
        ``[records, positions]`` scores.
    """
    scores = []
    for block in _centred_chunks(tokens, reference.position_means):
        projected = ((block - reference.mean) / reference.scale) @ reference.components.T
        if kmeans:
            whitened = projected / np.sqrt(reference.explained_variance)
            centroids = reference.centroids
            squared = ((whitened ** 2).sum(axis=1, keepdims=True) - 2 * whitened @ centroids.T
                       + (centroids ** 2).sum(axis=1))
            scores.append(np.maximum(squared.min(axis=1), 0.0))
        else:
            scores.append(reference.covariance.mahalanobis(projected))
    return np.concatenate(scores).reshape(tokens.shape[:2])


def fit_lead_references(tokens: np.ndarray, seed: int) -> list[TokenReference]:
    """
    Fit one token reference per JEPA lead on that lead's 50 patch positions.

    Parameters
    ----------
    tokens : np.ndarray
        ``[records, 400, width]`` tokens of normal ECGs.
    seed : int
        ``random_state`` of each lead's k-means.

    Returns
    -------
    list[TokenReference]
        One reference per lead, in JEPA lead order.
    """
    return [fit_token_reference(tokens[:, JEPA_PATCHES * lead:JEPA_PATCHES * (lead + 1)], seed)
            for lead in range(len(JEPA_LEADS))]


def lead_token_scores(references: list[TokenReference], tokens: np.ndarray,
                      kmeans: bool = False) -> np.ndarray:
    """
    Score every lead's tokens against that lead's reference.

    Parameters
    ----------
    references : list[TokenReference]
        Output of ``fit_lead_references``.
    tokens : np.ndarray
        ``[records, 400, width]`` tokens.
    kmeans : bool
        Score by the nearest whitened centroid instead of the Mahalanobis distance.

    Returns
    -------
    np.ndarray
        ``[records, 400]`` scores, token ``50 * lead + patch``.
    """
    width = JEPA_PATCHES
    return np.concatenate([token_scores(reference, tokens[:, width * lead:width * (lead + 1)], kmeans)
                           for lead, reference in enumerate(references)], axis=1)


def jepa_unit_map(scores: np.ndarray) -> UnitMap:
    """
    Units of one ECG's 400 JEPA token scores, with canonical lead indices and patch times.

    Parameters
    ----------
    scores : np.ndarray
        ``[400]`` scores, token ``50 * lead + patch``.

    Returns
    -------
    UnitMap
        The units.
    """
    leads = np.repeat(np.array(JEPA_LEADS), JEPA_PATCHES)
    starts = np.tile(np.arange(JEPA_PATCHES) * JEPA_PATCH_SECONDS, len(JEPA_LEADS))
    return UnitMap(np.asarray(scores, dtype=np.float64), leads, starts, starts + JEPA_PATCH_SECONDS)


def wave_lengths(fs: int = FS) -> dict[str, int]:
    """Count the samples of each wave piece after keeping every second sample."""
    return {name: len(range(round(low * fs), round(high * fs), 2)) for name, (low, high) in WAVES.items()}


def beat_pieces(signal: np.ndarray, fs: int = FS) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Cut every complete beat of one ECG into P, QRS, ST and T pieces per lead.

    Parameters
    ----------
    signal : np.ndarray
        ``[12, samples]`` waveform in mV, canonical lead order.
    fs : int
        Sampling rate in Hz.

    Returns
    -------
    tuple[np.ndarray, dict[str, np.ndarray]]
        R times in seconds of the kept beats, and for each wave a ``[beats, 12, length]`` array.
    """
    peaks = r_peaks(signal, fs)
    b, a = butter(2, HIGH_PASS_HZ, btype="high", fs=fs)
    filtered = filtfilt(b, a, np.nan_to_num(signal), axis=1)
    inside = (peaks - round(BEAT_BEFORE * fs) >= 0) & (peaks + round(BEAT_AFTER * fs) <= signal.shape[1])
    kept = peaks[inside]
    pieces = {name: np.empty((len(kept), signal.shape[0], length))
              for name, length in wave_lengths(fs).items()}
    for index, peak in enumerate(kept):
        baseline = filtered[:, peak + round(BASELINE[0] * fs):peak + round(BASELINE[1] * fs)].mean(axis=1)
        for name, (low, high) in WAVES.items():
            window = filtered[:, peak + round(low * fs):peak + round(high * fs):2]
            pieces[name][index] = window - baseline[:, None]
    return kept / fs, pieces


def median_pieces(pieces: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Median over beats of each wave piece, ``[12, length]`` per wave."""
    return {name: np.median(values, axis=0) for name, values in pieces.items()}


def fit_wave_references(pieces: dict[str, np.ndarray]) -> dict[str, list[LedoitWolf]]:
    """
    One Ledoit-Wolf reference per wave and lead.

    Parameters
    ----------
    pieces : dict[str, np.ndarray]
        ``[rows, 12, length]`` pieces of normal beats (or normal median beats) for each wave.

    Returns
    -------
    dict[str, list[LedoitWolf]]
        Twelve fitted references per wave, in canonical lead order.
    """
    return {name: [LedoitWolf().fit(values[:, lead]) for lead in range(values.shape[1])]
            for name, values in pieces.items()}


def wave_scores(references: dict[str, list[LedoitWolf]], pieces: dict[str, np.ndarray]) -> np.ndarray:
    """
    Squared Mahalanobis distance of every piece from its wave and lead reference.

    Parameters
    ----------
    references : dict[str, list[LedoitWolf]]
        Output of ``fit_wave_references``.
    pieces : dict[str, np.ndarray]
        ``[rows, 12, length]`` pieces for each wave.

    Returns
    -------
    np.ndarray
        ``[rows, 12, waves]`` scores, waves in ``WAVES`` order.
    """
    return np.stack([np.stack([references[name][lead].mahalanobis(values[:, lead])
                               for lead in range(values.shape[1])], axis=1)
                     for name, values in pieces.items()], axis=2)


def beat_unit_map(scores: np.ndarray, r_times: np.ndarray) -> UnitMap:
    """
    Units of one ECG's per-beat wave scores.

    Parameters
    ----------
    scores : np.ndarray
        ``[beats, 12, waves]`` output of ``wave_scores``.
    r_times : np.ndarray
        R time of each beat in seconds.

    Returns
    -------
    UnitMap
        One unit per beat, lead and wave.
    """
    beats, leads, waves = scores.shape
    offsets = np.array(list(WAVES.values()))
    starts = r_times[:, None, None] + offsets[None, None, :, 0]
    ends = r_times[:, None, None] + offsets[None, None, :, 1]
    lead_index = np.broadcast_to(np.arange(leads)[None, :, None], scores.shape)
    return UnitMap(scores.ravel(), lead_index.ravel(), np.broadcast_to(starts, scores.shape).ravel(),
                   np.broadcast_to(ends, scores.shape).ravel())


def lead_wave_unit_map(scores: np.ndarray) -> UnitMap:
    """Units of ``[12, waves]`` lead-and-wave scores, without a time span."""
    leads = np.broadcast_to(np.arange(scores.shape[0])[:, None], scores.shape)
    missing = np.full(scores.size, np.nan)
    return UnitMap(scores.ravel().astype(np.float64), leads.ravel(), missing, missing.copy())


def median_features(medians: dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate median pieces wave by wave, then lead by lead, into one feature vector."""
    return np.concatenate([medians[name].ravel() for name in WAVES])


def block_contributions(head: tuple, features: np.ndarray, lengths: dict[str, int]) -> np.ndarray:
    """
    Share of a linear readout's logit for each lead and wave block of the median features.

    Parameters
    ----------
    head : tuple
        ``(StandardScaler, LogisticRegression)`` fitted on ``median_features``.
    features : np.ndarray
        ``[records, features]`` median features.
    lengths : dict[str, int]
        Output of ``wave_lengths``.

    Returns
    -------
    np.ndarray
        ``[records, 12, waves]`` shares; each record's shares sum to its logit.
    """
    scaler, model = head
    terms = (np.asarray(features, dtype=np.float64) - scaler.mean_) / scaler.scale_ * model.coef_[0]
    blocks, start = [], 0
    for name in WAVES:
        width = len(LEADS) * lengths[name]
        block = terms[:, start:start + width].reshape(len(features), len(LEADS), lengths[name])
        blocks.append(block.sum(axis=2))
        start += width
    return np.stack(blocks, axis=2) + model.intercept_[0] / (len(LEADS) * len(WAVES))


def ecg_score(unit_map: UnitMap) -> float:
    """Highest unit score of one ECG."""
    return float(unit_map.scores.max())


def top_lead(unit_map: UnitMap) -> str:
    """Name of the lead of the highest unit."""
    return LEADS[int(unit_map.leads[unit_map.scores.argmax()])]


def premature_hit(unit_map: UnitMap, windows: list[tuple[float, float]]) -> tuple[float, float]:
    """
    Whether the top unit overlaps a premature-beat window, and the share of units that do.

    Parameters
    ----------
    unit_map : UnitMap
        Units with time spans.
    windows : list[tuple[float, float]]
        Output of ``fragment_localization.premature_windows``.

    Returns
    -------
    tuple[float, float]
        ``hit`` (0 or 1) and ``chance``.
    """
    overlap = np.zeros(len(unit_map.scores), dtype=bool)
    for low, high in windows:
        overlap |= (unit_map.starts < high) & (unit_map.ends > low)
    return float(overlap[unit_map.scores.argmax()]), float(overlap.mean())


def two_group_difference(patients_a: np.ndarray, values_a: np.ndarray, patients_b: np.ndarray,
                         values_b: np.ndarray, draws: int, seed: int) -> dict[str, float]:
    """
    Mean of group A minus mean of group B, resampling patients within each group.

    Parameters
    ----------
    patients_a, patients_b : np.ndarray
        Patient ID of each record in each group.
    values_a, values_b : np.ndarray
        One value per record.
    draws : int
        Bootstrap draws.
    seed : int
        Seed of ``numpy.random.default_rng``.

    Returns
    -------
    dict[str, float]
        ``value``, ``ci_low``, ``ci_high``, and the two group means.
    """
    groups_a, groups_b = patient_groups(patients_a), patient_groups(patients_b)
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(draws):
        first = values_a[patient_resample(groups_a, rng)].mean()
        differences.append(first - values_b[patient_resample(groups_b, rng)].mean())
    low, high = np.percentile(differences, [2.5, 97.5])
    return {"value": float(values_a.mean() - values_b.mean()), "ci_low": float(low), "ci_high": float(high),
            "group_a": float(values_a.mean()), "group_b": float(values_b.mean())}


def red_marks(unit_map: UnitMap, threshold: float) -> list[tuple[int, float, float]]:
    """Lead index and time span of every unit above the threshold."""
    above = np.flatnonzero(unit_map.scores > threshold)
    return [(int(unit_map.leads[i]), float(unit_map.starts[i]), float(unit_map.ends[i])) for i in above]


def plot_lead_marks(signal: np.ndarray, fs: int, marks: dict[str, list[tuple[int, float, float]]],
                    title: str) -> Figure:
    """
    Twelve leads with red marks on the lead and time of each red unit, one column per map.

    Parameters
    ----------
    signal : np.ndarray
        ``[12, samples]`` waveform in mV, canonical lead order.
    fs : int
        Sampling rate in Hz.
    marks : dict[str, list[tuple[int, float, float]]]
        Output of ``red_marks`` for each map, keyed by the column title.
    title : str
        Figure title.

    Returns
    -------
    Figure
        The figure.
    """
    figure = Figure(figsize=(7 * len(marks), 11), layout="constrained")
    axes = figure.subplots(len(LEADS), len(marks), sharex=True, squeeze=False)
    seconds = np.arange(signal.shape[1]) / fs
    for column, (name, found) in enumerate(marks.items()):
        axes[0, column].set_title(name)
        for lead, start, end in found:
            axes[lead, column].axvspan(start, end, color="red", alpha=0.3, linewidth=0)
        for row, lead in enumerate(LEADS):
            axis = axes[row, column]
            axis.plot(seconds, signal[row], color="black", linewidth=0.6)
            axis.set_ylabel(lead, rotation=0, labelpad=14)
            axis.set_yticks([])
        axes[-1, column].set_xlabel("seconds")
    figure.suptitle(title)
    return figure
