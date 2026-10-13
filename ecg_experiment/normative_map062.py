"""
Normative phase-aligned population map for Experiment 062.

Each beat is delineated with the project's waveform-only recipe (frozen ``morphology_boundaries``),
warped piecewise-linearly per lead onto a canonical grid (pre-QRS, QRS, ST, T, post-T) and kept in
mV after subtracting the PR-segment baseline. A population model fitted on clean normals (covariate
regression on heart rate, age and sex plus a bounded low-rank normal subspace) scores the record's
median beat; a within-record leave-one-out median scores each beat. Both, plus interval and RR
evidence, are put on a common scale (-log10 tail probability among normal records) and mapped back
to the original timeline. See ``docs/experiment-062-normative-map.md``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt

from ecg_experiment.morphology_boundaries import adaptive_boundaries

FS = 500
SAMPLES = 5000
LEADS = 12
PRE = 150
POST = 50
GRID_KNOTS = np.array([0, PRE, PRE + 50, PRE + 90, PRE + 180, PRE + 180 + POST], dtype=float)
BINS = int(GRID_KNOTS[-1])
SEGMENTS = ("PR", "QRS", "ST", "T", "post")
SEGMENT_OF_BIN = np.searchsorted(GRID_KNOTS, np.arange(BINS), side="right") - 1
COMPONENTS = ("persistent", "focal")
SMOOTH_BINS = 9
TAIL_STEP = 0.05
HUBER = 2.5
SUBSPACE = 6
BOX = 2.58
DOMINANT_FACTOR = 2.0
LOWPASS_HZ = 40.0
ALIGN = 4


@dataclass
class RecordFeatures:
    """
    Warped beats and summaries of one record.

    Attributes
    ----------
    times : np.ndarray
        Original-time sample position of every grid bin, (beats, BINS).
    knots : np.ndarray
        Original-time warp knots, (beats, 6).
    beats : np.ndarray
        Warped baseline-corrected beats in mV, (beats, LEADS, BINS), NaN outside the record.
    median : np.ndarray
        Median of dominant beats, (LEADS, BINS).
    dispersion : np.ndarray
        Robust beat-to-beat scale of dominant beats, (LEADS, BINS).
    dominant : np.ndarray
        Boolean mask of beats used for the median.
    focal : np.ndarray
        Beat minus leave-one-out dominant median, (beats, LEADS, BINS).
    intervals : np.ndarray
        Heart rate (bpm), QRS duration, QT and PR (seconds); PR may be NaN.
    prematurity : np.ndarray
        Preceding RR over record median RR per beat (NaN for the first beat).
    qrs_change : np.ndarray
        Absolute difference of each beat's QRS duration from the record median, in seconds.
    noise : np.ndarray
        Per beat and lead RMS of the 20-40 Hz band outside the QRS, in mV, (beats, LEADS).
    """

    times: np.ndarray
    knots: np.ndarray
    beats: np.ndarray
    median: np.ndarray
    dispersion: np.ndarray
    dominant: np.ndarray
    focal: np.ndarray
    intervals: np.ndarray
    prematurity: np.ndarray
    qrs_change: np.ndarray
    noise: np.ndarray


@dataclass
class PopulationModel:
    """
    Covariate regression and bounded low-rank normal subspace of warped median beats.

    Attributes
    ----------
    coefficients : np.ndarray
        Regression of flattened median beats on the covariate design, (design, LEADS * BINS).
    scale : np.ndarray
        Robust marginal scale of the covariate residual per dimension.
    basis : np.ndarray
        Orthonormal subspace in standardized units, (LEADS * BINS, k).
    eigenvalues : np.ndarray
        Variance of each subspace direction.
    noise : float
        Mean variance outside the subspace (probabilistic PCA prior).
    off_scale : np.ndarray
        Robust scale of the off-subspace standardized residual per dimension.
    """

    coefficients: np.ndarray
    scale: np.ndarray
    basis: np.ndarray
    eigenvalues: np.ndarray
    noise: float
    off_scale: np.ndarray


def delineate(signal: np.ndarray) -> dict[str, np.ndarray]:
    """
    Per-beat joint fiducials from the frozen waveform-only delineation.

    QRS onset and offset are the 10th and 90th percentiles of the per-lead supports; T onset and
    end the median over leads with a T support (at least three). P onset is the median over leads
    with a P support (at least three), else NaN.

    Parameters
    ----------
    signal : np.ndarray
        Raw (12, samples) waveform in mV at 500 Hz.

    Returns
    -------
    dict[str, np.ndarray]
        ``r``, ``qon``, ``qoff``, ``ton``, ``toff``, ``pon`` per beat (float, NaN where missing).
    """
    peaks, bounds = adaptive_boundaries(signal, FS)
    qon = np.floor(np.quantile(bounds[:, :, 1, 0], 0.1, axis=1)) if len(peaks) else np.zeros(0)
    qoff = np.ceil(np.quantile(bounds[:, :, 1, 1], 0.9, axis=1)) if len(peaks) else np.zeros(0)
    return {
        "r": peaks.astype(float),
        "qon": qon,
        "qoff": qoff,
        "ton": _lead_median(bounds[:, :, 2, 0]),
        "toff": _lead_median(bounds[:, :, 2, 1]),
        "pon": _lead_median(bounds[:, :, 0, 0]),
    }


def _lead_median(values: np.ndarray, minimum: int = 3) -> np.ndarray:
    valid = values >= 0
    masked = np.where(valid, values.astype(float), np.nan)
    with warnings.catch_warnings(action="ignore"):
        median = np.nanmedian(masked, axis=1) if values.shape[0] else np.zeros(0)
    return np.where(valid.sum(axis=1) >= minimum, np.round(median), np.nan)


def warp_knots(fiducials: dict[str, np.ndarray]) -> np.ndarray:
    """
    Strictly increasing original-time knots of each beat's warp.

    Missing T fiducials fall back to the record's median QRS-onset-to-T-end and QRS-offset-to-T-onset
    durations, and without any T to a Bazett-normal QT of 0.40 s times the square root of RR.

    Parameters
    ----------
    fiducials : dict[str, np.ndarray]
        ``delineate`` output.

    Returns
    -------
    np.ndarray
        (beats, 6): QRS onset - PRE, QRS onset, QRS offset, T onset, T end, T end + POST.
    """
    r, qon, qoff = fiducials["r"], fiducials["qon"], fiducials["qoff"]
    qoff = np.maximum(qoff, qon + 10)
    rr = float(np.median(np.diff(r))) / FS if len(r) > 1 else 1.0
    toff, ton = fiducials["toff"].copy(), fiducials["ton"].copy()
    found = np.isfinite(toff) & np.isfinite(ton) & (toff > qoff) & (ton > qoff) & (toff > ton)
    qt = np.median((toff - qon)[found]) if found.any() else round(0.40 * np.sqrt(rr) * FS)
    st = np.median((ton - qoff)[found]) if found.any() else 0.35 * (qt - np.median(qoff - qon))
    toff = np.where(found, toff, qon + qt)
    ton = np.where(found, ton, qoff + st)
    ton = np.clip(ton, qoff + 4, None)
    toff = np.maximum(toff, ton + 20)
    return np.stack([qon - PRE, qon, qoff, ton, toff, toff + POST], axis=1)


def bin_times(knots: np.ndarray) -> np.ndarray:
    """
    Original-time position of every grid bin for each beat.

    Parameters
    ----------
    knots : np.ndarray
        (beats, 6) warp knots.

    Returns
    -------
    np.ndarray
        (beats, BINS) fractional sample positions.
    """
    grid = np.arange(BINS, dtype=float)
    return (
        np.stack([np.interp(grid, GRID_KNOTS, row) for row in knots]) if len(knots) else np.zeros((0, BINS))
    )


def preprocess(signal: np.ndarray, qon: np.ndarray) -> np.ndarray:
    """
    Low-pass filter and subtract a PR-segment baseline interpolated between beats.

    The knot of each beat is the per-lead mean over 24 ms ending 6 ms before QRS onset; amplitude
    is never normalized.

    Parameters
    ----------
    signal : np.ndarray
        Raw (12, samples) waveform in mV.
    qon : np.ndarray
        QRS onset per beat in samples.

    Returns
    -------
    np.ndarray
        Filtered, baseline-corrected waveform in mV.
    """
    sos = butter(2, LOWPASS_HZ, btype="lowpass", fs=FS, output="sos")
    filtered = sosfiltfilt(sos, np.nan_to_num(np.asarray(signal, dtype=float)), axis=1)
    usable = qon[(qon - 15 >= 0) & (qon - 3 <= filtered.shape[1])].astype(int)
    if not len(usable):
        return filtered - np.median(filtered, axis=1, keepdims=True)
    levels = np.stack([filtered[:, q - 15 : q - 3].mean(axis=1) for q in usable], axis=1)
    centres = usable - 9.0
    samples = np.arange(filtered.shape[1])
    baseline = np.stack([np.interp(samples, centres, row) for row in levels])
    return filtered - baseline


def warp_beats(clean: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    Sample each lead at the grid positions of each beat by linear interpolation.

    Parameters
    ----------
    clean : np.ndarray
        Preprocessed (12, samples) waveform.
    times : np.ndarray
        (beats, BINS) positions from ``bin_times``.

    Returns
    -------
    np.ndarray
        (beats, 12, BINS), NaN where a position falls outside the record.
    """
    samples = np.arange(clean.shape[1])
    out = np.empty((len(times), clean.shape[0], BINS))
    for index, row in enumerate(times):
        for lead in range(clean.shape[0]):
            out[index, lead] = np.interp(row, samples, clean[lead])
        out[index, :, (row < 0) | (row > clean.shape[1] - 1)] = np.nan
    return out


def dominant_beats(beats: np.ndarray) -> np.ndarray:
    """
    Beats close to the initial median, to keep ectopic beats out of the record template.

    Parameters
    ----------
    beats : np.ndarray
        (beats, 12, BINS) warped beats.

    Returns
    -------
    np.ndarray
        Boolean mask; a beat is dominant if its RMS distance to the initial median is at most
        ``DOMINANT_FACTOR`` times the median distance. At least three beats (or all) are kept.
    """
    with warnings.catch_warnings(action="ignore"), np.errstate(all="ignore"):
        initial = np.nanmedian(beats, axis=0)
        distance = np.sqrt(np.nanmean((beats - initial) ** 2, axis=(1, 2)))
    distance = np.nan_to_num(distance, nan=np.inf)
    keep = distance <= DOMINANT_FACTOR * np.median(distance[np.isfinite(distance)])
    if keep.sum() < min(3, len(beats)):
        keep = np.zeros(len(beats), dtype=bool)
        keep[np.argsort(distance)[: min(3, len(beats))]] = True
    return keep


def leave_one_out(beats: np.ndarray, dominant: np.ndarray) -> np.ndarray:
    """
    Each beat minus the median of the other dominant beats.

    Parameters
    ----------
    beats : np.ndarray
        (beats, 12, BINS) warped beats.
    dominant : np.ndarray
        Dominant-beat mask.

    Returns
    -------
    np.ndarray
        (beats, 12, BINS) focal deviations; zeros when fewer than two dominant beats exist.
    """
    focal = np.zeros_like(beats)
    if dominant.sum() < 2:
        return focal
    with warnings.catch_warnings(action="ignore"), np.errstate(all="ignore"):
        for index in range(len(beats)):
            others = dominant.copy()
            others[index] = False
            focal[index] = beats[index] - np.nanmedian(beats[others], axis=0)
    return np.nan_to_num(focal)


def beat_noise(signal: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """
    High-frequency noise of each beat and lead, measured where the ECG itself has little 20-40 Hz content.

    Parameters
    ----------
    signal : np.ndarray
        Raw (12, samples) waveform in mV.
    knots : np.ndarray
        (beats, 6) warp knots.

    Returns
    -------
    np.ndarray
        (beats, 12) RMS of the 20-40 Hz band over the beat window minus QRS onset - 20 ms to QRS
        offset + 20 ms.
    """
    sos = butter(2, [20.0, LOWPASS_HZ], btype="bandpass", fs=FS, output="sos")
    band = sosfiltfilt(sos, np.nan_to_num(np.asarray(signal, dtype=float)), axis=1)
    out = np.zeros((len(knots), signal.shape[0]))
    samples = np.arange(signal.shape[1])
    for index, row in enumerate(knots):
        inside = (samples >= row[0]) & (samples <= row[-1])
        inside &= (samples < row[1] - 10) | (samples > row[2] + 10)
        if inside.any():
            out[index] = np.sqrt(np.mean(band[:, inside] ** 2, axis=1))
    return out


def align_knots(clean: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """
    Shift each beat's knots by whole samples so its QRS best matches the record's initial median QRS.

    One shift in [-``ALIGN``, +``ALIGN``] samples per beat, shared by all leads, minimizing the squared
    QRS-bin difference after dividing each lead by its median QRS peak-to-peak (floor 0.1 mV);
    ties prefer the smallest absolute shift. Amplitude is never fitted.

    Parameters
    ----------
    clean : np.ndarray
        Preprocessed (12, samples) waveform.
    knots : np.ndarray
        (beats, 6) warp knots.

    Returns
    -------
    np.ndarray
        Shifted knots.
    """
    qrs = SEGMENTS.index("QRS") == SEGMENT_OF_BIN
    shifts = np.array(sorted(range(-ALIGN, ALIGN + 1), key=abs))
    candidates = np.stack(
        [warp_beats(clean, bin_times(knots + shift))[:, :, qrs] for shift in shifts], axis=1
    )
    with warnings.catch_warnings(action="ignore"), np.errstate(all="ignore"):
        initial = np.nanmedian(candidates[:, 0], axis=0)
        scale = np.maximum(np.nanmedian(np.ptp(candidates[:, 0], axis=2), axis=0), 0.1)
        loss = np.nanmean(((candidates - initial) / scale[:, None]) ** 2, axis=(2, 3))
    best = shifts[np.argmin(np.nan_to_num(loss, nan=np.inf), axis=1)]
    return knots + best[:, None]


def record_features(signal: np.ndarray) -> RecordFeatures | None:
    """
    Delineate, align, warp and summarize one record.

    Beats whose QRS or baseline window is cut by the record edge are dropped (their RR still
    counts for prematurity).

    Parameters
    ----------
    signal : np.ndarray
        Raw (12, samples) waveform in mV at 500 Hz.

    Returns
    -------
    RecordFeatures or None
        None when fewer than two complete beats are found.
    """
    fiducials = delineate(signal)
    if len(fiducials["r"]) < 2:
        return None
    rr_all = np.diff(fiducials["r"]) / FS
    prematurity = np.concatenate([[np.nan], rr_all / np.median(rr_all)])
    complete = (fiducials["qon"] - 15 >= 0) & (fiducials["qoff"] + 15 <= signal.shape[1])
    if complete.sum() < 2:
        return None
    clean = preprocess(signal, fiducials["qon"])
    knots = align_knots(clean, warp_knots(fiducials)[complete])
    times = bin_times(knots)
    beats = warp_beats(clean, times)
    dominant = dominant_beats(beats)
    with warnings.catch_warnings(action="ignore"), np.errstate(all="ignore"):
        median = np.nanmedian(beats[dominant], axis=0)
        dispersion = 1.4826 * np.nanmedian(np.abs(beats[dominant] - median), axis=0)
    qrs = (knots[:, 2] - knots[:, 1]) / FS
    pr = ((fiducials["qon"] - fiducials["pon"]) / FS)[complete]
    intervals = np.array(
        [
            60.0 / np.median(rr_all),
            np.median(qrs[dominant]),
            np.median((knots[dominant, 4] - knots[dominant, 1]) / FS),
            np.nanmedian(pr[dominant]) if np.isfinite(pr[dominant]).sum() >= 2 else np.nan,
        ]
    )
    return RecordFeatures(
        times=times,
        knots=knots,
        beats=beats,
        median=np.nan_to_num(median),
        dispersion=np.nan_to_num(dispersion),
        dominant=dominant,
        focal=leave_one_out(beats, dominant),
        intervals=intervals,
        prematurity=prematurity[complete],
        qrs_change=np.abs(qrs - np.median(qrs[dominant])),
        noise=beat_noise(signal, knots),
    )


def design(heart_rate: np.ndarray, age: np.ndarray, sex: np.ndarray) -> np.ndarray:
    """
    Covariate design: intercept, heart rate and age (linear and squared), and sex.

    Parameters
    ----------
    heart_rate : np.ndarray
        Beats per minute.
    age : np.ndarray
        Years; NaN (PTB-XL ages above 89) is set to 90.
    sex : np.ndarray
        0 or 1 as in PTB-XL.

    Returns
    -------
    np.ndarray
        (records, 6) design matrix.
    """
    h = (np.asarray(heart_rate, dtype=float) - 70.0) / 15.0
    a = (np.nan_to_num(np.asarray(age, dtype=float), nan=90.0) - 50.0) / 20.0
    return np.column_stack([np.ones_like(h), h, h**2, a, a**2, np.asarray(sex, dtype=float)])


def robust_scale(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Normal-consistent median absolute deviation, floored to avoid division by zero.

    Parameters
    ----------
    values : np.ndarray
        Data.
    axis : int
        Axis to reduce.

    Returns
    -------
    np.ndarray
        1.4826 times the median absolute deviation, at least 1e-6.
    """
    centre = np.median(values, axis=axis, keepdims=True)
    return np.maximum(1.4826 * np.median(np.abs(values - centre), axis=axis), 1e-6)


def fit_population(medians: np.ndarray, covariates: np.ndarray, rank: int = SUBSPACE) -> PopulationModel:
    """
    Fit the covariate regression and the normal subspace on clean normal median beats.

    Parameters
    ----------
    medians : np.ndarray
        (records, LEADS * BINS) flattened median beats in mV.
    covariates : np.ndarray
        (records, design) from ``design``.
    rank : int
        Subspace dimension; 0 keeps only the covariate regression.

    Returns
    -------
    PopulationModel
        Fitted model.
    """
    ridge = 1e-3 * np.eye(covariates.shape[1])
    coefficients = np.linalg.solve(covariates.T @ covariates + ridge, covariates.T @ medians)
    residual = medians - covariates @ coefficients
    scale = robust_scale(residual)
    standard = residual / scale
    _, singular, vt = np.linalg.svd(standard - standard.mean(axis=0), full_matrices=False)
    variance = singular**2 / (len(standard) - 1)
    basis = vt[:rank].T
    noise = float(variance[rank:].sum() / (standard.shape[1] - rank))
    off = standard - (standard @ basis) @ basis.T
    return PopulationModel(coefficients, scale, basis, variance[:rank], noise, robust_scale(off))


def subspace_coefficients(model: PopulationModel, standard: np.ndarray) -> np.ndarray:
    """
    Robust, bounded subspace coordinates of one standardized residual.

    Iteratively reweighted (Huber) probabilistic-PCA posterior mean, then each coordinate clipped to
    ``BOX`` normal standard deviations, so that only normal-sized correlated variation (axis, global
    amplitude) is explained away.

    Parameters
    ----------
    model : PopulationModel
        Fitted model.
    standard : np.ndarray
        (LEADS * BINS,) standardized covariate residual.

    Returns
    -------
    np.ndarray
        (k,) coordinates.
    """
    rank = model.basis.shape[1]
    if rank == 0:
        return np.zeros(0)
    weights = np.ones_like(standard)
    prior = np.diag(model.noise / model.eigenvalues)
    for _ in range(5):
        weighted = model.basis * weights[:, None]
        coordinates = np.linalg.solve(model.basis.T @ weighted + prior, weighted.T @ standard)
        residual = np.abs(standard - model.basis @ coordinates) / model.off_scale
        weights = np.minimum(1.0, HUBER / np.maximum(residual, 1e-12))
    bound = BOX * np.sqrt(model.eigenvalues)
    return np.clip(coordinates, -bound, bound)


def population_residual(model: PopulationModel, median: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    """
    Part of a median beat that neither the covariates nor bounded normal variation explain.

    Parameters
    ----------
    model : PopulationModel
        Fitted model.
    median : np.ndarray
        (LEADS * BINS,) flattened median beat in mV.
    covariates : np.ndarray
        (design,) covariate row.

    Returns
    -------
    np.ndarray
        (LEADS * BINS,) residual in mV.
    """
    standard = (median - covariates @ model.coefficients) / model.scale
    coordinates = subspace_coefficients(model, standard)
    return (standard - model.basis @ coordinates) * model.scale


def smooth_abs(z: np.ndarray) -> np.ndarray:
    """
    Local root mean square over ``SMOOTH_BINS`` grid bins along the last axis.

    Parameters
    ----------
    z : np.ndarray
        Signed z-scores, last axis BINS.

    Returns
    -------
    np.ndarray
        Non-negative smoothed magnitude.
    """
    return np.sqrt(np.maximum(uniform_filter1d(z**2, SMOOTH_BINS, axis=-1, mode="nearest"), 0.0))


def tail_table(values: np.ndarray, top: float | None = None) -> np.ndarray:
    """
    Quantiles of normal-record values at tail levels 10**-u for u on a 0.05 grid.

    Parameters
    ----------
    values : np.ndarray
        Pooled normal-record values (any shape).
    top : float or None
        Largest u; defaults to log10(count / 10), capped at 4.

    Returns
    -------
    np.ndarray
        (levels, 2): u and the matching quantile, strictly increasing in both columns.
    """
    values = np.ravel(values)
    top = min(4.0, np.log10(len(values) / 10)) if top is None else top
    levels = np.arange(0, top + 1e-9, TAIL_STEP)
    quantiles = np.quantile(values, 1 - 10.0**-levels)
    quantiles = np.maximum.accumulate(quantiles + np.arange(len(levels)) * 1e-9)
    return np.column_stack([levels, quantiles])


def tail_score(values: np.ndarray, table: np.ndarray) -> np.ndarray:
    """
    Map values to -log10 normal tail probability, extrapolating one decade per last-decade spread.

    Parameters
    ----------
    values : np.ndarray
        Values on the scale of the table.
    table : np.ndarray
        ``tail_table`` output.

    Returns
    -------
    np.ndarray
        Non-negative scores, monotone in ``values``.
    """
    levels, quantiles = table[:, 0], table[:, 1]
    decade = max(int(round(1 / TAIL_STEP)), 1)
    spread = max(quantiles[-1] - quantiles[max(len(quantiles) - 1 - decade, 0)], 1e-9)
    width = levels[-1] - levels[max(len(levels) - 1 - decade, 0)]
    inside = np.interp(values, quantiles, levels)
    beyond = levels[-1] + (values - quantiles[-1]) / spread * max(width, TAIL_STEP)
    return np.where(values > quantiles[-1], beyond, inside)


def persistent_z(
    residual: np.ndarray, features: RecordFeatures, scale: np.ndarray, noise: bool
) -> np.ndarray:
    """
    Compute the population residual z-score of a record's median beat per lead and bin.

    Parameters
    ----------
    residual : np.ndarray
        (LEADS, BINS) ``population_residual`` of the record in mV.
    features : RecordFeatures
        Record summaries.
    scale : np.ndarray
        (LEADS, BINS) normal residual scale (out-of-fold).
    noise : bool
        Add the median's standard error (from beat dispersion) to the denominator.

    Returns
    -------
    np.ndarray
        (LEADS, BINS) z-scores.
    """
    error = 1.2533 * features.dispersion / np.sqrt(max(int(features.dominant.sum()), 1)) if noise else 0.0
    return residual / np.sqrt(scale**2 + error**2)


def focal_score(features: RecordFeatures, scale: np.ndarray, noise: bool) -> np.ndarray:
    """
    Smoothed within-record beat deviation magnitude per beat, lead and bin.

    With ``noise``, the denominator adds the record's beat-to-beat dispersion and the beat's own
    20-40 Hz noise, and each lead's magnitude is replaced by its geometric mean with the mean over
    leads (physiological beat changes reach several leads; electrode artifacts often do not).

    Parameters
    ----------
    features : RecordFeatures
        Record summaries.
    scale : np.ndarray
        (LEADS, BINS) normal population scale of focal deviations.
    noise : bool
        Apply the noise terms and the cross-lead coherence.

    Returns
    -------
    np.ndarray
        (beats, LEADS, BINS) non-negative scores.
    """
    if not noise:
        return smooth_abs(features.focal / scale)
    variance = scale**2 + features.dispersion**2 + features.noise[:, :, None] ** 2
    magnitude = smooth_abs(features.focal / np.sqrt(variance))
    return np.sqrt(magnitude * magnitude.mean(axis=1, keepdims=True))


def segment_tables(scores: np.ndarray, top: float | None = None) -> np.ndarray:
    """
    Tail tables per lead and segment for (records or beats, LEADS, BINS) smoothed scores.

    Parameters
    ----------
    scores : np.ndarray
        Non-negative smoothed scores of normal records.
    top : float or None
        Largest tail level.

    Returns
    -------
    np.ndarray
        (LEADS, segments, levels, 2) tables; levels share one grid (the smallest table's).
    """
    tables = [
        [tail_table(scores[:, lead, segment == SEGMENT_OF_BIN], top) for segment in range(len(SEGMENTS))]
        for lead in range(LEADS)
    ]
    length = min(len(table) for row in tables for table in row)
    return np.array([[table[:length] for table in row] for row in tables])


def calibrate_grid(scores: np.ndarray, tables: np.ndarray) -> np.ndarray:
    """
    Apply per lead and segment tail tables to smoothed scores.

    Parameters
    ----------
    scores : np.ndarray
        (..., LEADS, BINS) smoothed scores.
    tables : np.ndarray
        ``segment_tables`` output.

    Returns
    -------
    np.ndarray
        Same shape, -log10 normal tail probability.
    """
    out = np.empty_like(scores, dtype=float)
    for lead in range(LEADS):
        for segment in range(len(SEGMENTS)):
            columns = segment == SEGMENT_OF_BIN
            out[..., lead, columns] = tail_score(scores[..., lead, columns], tables[lead, segment])
    return out


def to_timeline(values: np.ndarray, knots: np.ndarray, samples: int = SAMPLES) -> np.ndarray:
    """
    Map per-beat grid scores back to the original timeline, keeping the maximum where beats overlap.

    Parameters
    ----------
    values : np.ndarray
        (beats, LEADS, BINS) scores.
    knots : np.ndarray
        (beats, 6) warp knots.
    samples : int
        Record length.

    Returns
    -------
    np.ndarray
        (LEADS, samples) float32; samples no beat covers get the minimum covered value (0 if none).
    """
    out = np.full((values.shape[1], samples), -np.inf)
    grid = np.arange(BINS, dtype=float)
    for beat, row in zip(values, knots, strict=True):
        start, end = max(int(np.ceil(row[0])), 0), min(int(np.floor(row[-1])), samples - 1)
        if end < start:
            continue
        positions = np.interp(np.arange(start, end + 1), row, GRID_KNOTS)
        for lead in range(values.shape[1]):
            out[lead, start : end + 1] = np.maximum(
                out[lead, start : end + 1], np.interp(positions, grid, beat[lead])
            )
    covered = np.isfinite(out)
    out[~covered] = out[covered].min() if covered.any() else 0.0
    return out.astype(np.float32)


INTERVALS = ("heart_rate", "qrs", "qt", "pr")
ALL_COMPONENTS = ("persistent", "focal", "interval", "beat")
VARIANTS = {
    "normative_v1": (ALL_COMPONENTS, "sum"),
    "max_combination": (ALL_COMPONENTS, "max"),
    "persistent_only": (("persistent", "interval"), "sum"),
    "focal_only": (("focal", "beat"), "sum"),
    "no_subspace": (("persistent_rank0", "focal", "interval", "beat"), "sum"),
    "no_noise_weight": (("persistent_raw", "focal_raw", "interval", "beat"), "sum"),
}
PRIMARY = "normative_v1"
PERSISTENT = (
    ("persistent", "full", True),
    ("persistent_raw", "full", False),
    ("persistent_rank0", "rank0", True),
)
FOLDS = 5
SEED = 62062


@dataclass
class Normative:
    """
    Everything fitted on clean normals that a record map needs.

    Attributes
    ----------
    models : dict[str, PopulationModel]
        ``full`` (rank ``SUBSPACE``) and ``rank0`` population models fitted on all fit normals.
    scales : dict[str, np.ndarray]
        (LEADS, BINS) normal scales: out-of-fold population residual per model, and focal deviation.
    tables : dict[str, np.ndarray]
        Tail tables per score name (grid scores per lead and segment; scalars as one table).
    interval_coefficients : np.ndarray
        Regression of QRS, QT and PR on the covariate design, (design, 3).
    interval_scales : np.ndarray
        Robust residual scale of QRS, QT and PR.
    """

    models: dict[str, PopulationModel]
    scales: dict[str, np.ndarray]
    tables: dict[str, np.ndarray]
    interval_coefficients: np.ndarray
    interval_scales: np.ndarray


def interval_values(intervals: np.ndarray, covariates: np.ndarray, normative: Normative) -> np.ndarray:
    """
    Interval deviation magnitudes: QRS (longer only), QT and PR (either direction).

    Parameters
    ----------
    intervals : np.ndarray
        (records, 4) heart rate, QRS, QT, PR.
    covariates : np.ndarray
        (records, design) design rows.
    normative : Normative
        Interval regression.

    Returns
    -------
    np.ndarray
        (records, 3) non-negative deviations; NaN PR gives 0.
    """
    z = (intervals[:, 1:] - covariates @ normative.interval_coefficients) / normative.interval_scales
    z[:, 0] = np.maximum(z[:, 0], 0.0)
    return np.nan_to_num(np.abs(z))


def beat_values(features: RecordFeatures) -> np.ndarray:
    """
    Per-beat prematurity (1 - RR ratio, floored at 0) and QRS-duration change in seconds.

    Parameters
    ----------
    features : RecordFeatures
        Record summaries.

    Returns
    -------
    np.ndarray
        (beats, 2) non-negative values.
    """
    premature = np.maximum(1.0 - np.nan_to_num(features.prematurity, nan=1.0), 0.0)
    return np.column_stack([premature, features.qrs_change])


def component_scores(
    features: RecordFeatures, covariates: np.ndarray, normative: Normative
) -> dict[str, np.ndarray]:
    """
    Calibrated component scores of one record on the grid, each broadcastable to (beats, LEADS, BINS).

    Parameters
    ----------
    features : RecordFeatures
        Record summaries.
    covariates : np.ndarray
        (design,) covariate row.
    normative : Normative
        Fitted normative quantities.

    Returns
    -------
    dict[str, np.ndarray]
        -log10 normal tail probabilities for ``persistent``, ``persistent_raw``, ``persistent_rank0``,
        ``focal``, ``focal_raw``, ``interval`` and ``beat``.
    """
    scores = {}
    residuals = {
        model: population_residual(normative.models[model], features.median.ravel(), covariates).reshape(
            LEADS, BINS
        )
        for model in ("full", "rank0")
    }
    for name, model, noise in PERSISTENT:
        z = persistent_z(residuals[model], features, normative.scales[model], noise)
        scores[name] = calibrate_grid(smooth_abs(z)[None], normative.tables[name])
    for name, noise in (("focal", True), ("focal_raw", False)):
        magnitude = focal_score(features, normative.scales["focal"], noise)
        scores[name] = calibrate_grid(magnitude, normative.tables[name])
    deviation = interval_values(features.intervals[None], covariates[None], normative)[0]
    interval = np.zeros(BINS)
    for column, segment in enumerate(("QRS", "T", "PR")):
        value = tail_score(deviation[column : column + 1], normative.tables[f"interval_{column}"])[0]
        interval[SEGMENTS.index(segment) == SEGMENT_OF_BIN] = value
    scores["interval"] = interval[None, None]
    values = beat_values(features)
    premature = tail_score(values[:, 0], normative.tables["premature"])
    widening = tail_score(values[:, 1], normative.tables["qrs_change"])
    beat = np.zeros((len(values), 1, BINS))
    beat[:, 0, SEGMENTS.index("QRS") >= SEGMENT_OF_BIN] = premature[:, None]
    qrs = SEGMENTS.index("QRS") == SEGMENT_OF_BIN
    beat[:, 0, qrs] = np.maximum(beat[:, 0, qrs], widening[:, None])
    scores["beat"] = beat
    return scores


def combine(scores: dict[str, np.ndarray], components: tuple[str, ...], mode: str, beats: int) -> np.ndarray:
    """
    Sum (Fisher-style accumulation of -log10 p) or maximum over the chosen calibrated components.

    Parameters
    ----------
    scores : dict[str, np.ndarray]
        ``component_scores`` output.
    components : tuple[str, ...]
        Names to combine.
    mode : str
        ``sum`` or ``max``.
    beats : int
        Number of beats.

    Returns
    -------
    np.ndarray
        (beats, LEADS, BINS) combined score.
    """
    out = np.zeros((beats, LEADS, BINS))
    for name in components:
        out = out + scores[name] if mode == "sum" else np.maximum(out, scores[name])
    return out


def record_maps(signal: np.ndarray, age: float, sex: float, normative: Normative) -> tuple[dict, dict]:
    """
    Build the map of every declared variant for one record on the original 500 Hz timeline.

    Parameters
    ----------
    signal : np.ndarray
        Raw (12, 5000) waveform in mV.
    age, sex : float
        Covariates (age NaN allowed).
    normative : Normative
        Fitted normative quantities.

    Returns
    -------
    tuple[dict, dict]
        Variant name to (12, 5000) float32 map, and a summary (beats, dominant beats, intervals,
        maximum of each component; ``failed`` when no map could be built).
    """
    features = record_features(signal)
    if features is None:
        flat = np.zeros((LEADS, signal.shape[1]), dtype=np.float32)
        return dict.fromkeys(VARIANTS, flat), {"failed": True, "beats": 0}
    covariates = design(features.intervals[:1], np.array([age]), np.array([sex]))[0]
    scores = component_scores(features, covariates, normative)
    beats = len(features.knots)
    maps = {
        name: to_timeline(combine(scores, components, mode, beats), features.knots, signal.shape[1])
        for name, (components, mode) in VARIANTS.items()
    }
    summary = {
        "failed": False,
        "beats": beats,
        "dominant": int(features.dominant.sum()),
        **{name: float(value) for name, value in zip(INTERVALS, features.intervals, strict=True)},
        **{f"max_{name}": float(np.max(value)) for name, value in scores.items()},
    }
    return maps, summary


def patient_folds(patients: np.ndarray, folds: int = FOLDS, seed: int = SEED) -> np.ndarray:
    """
    Assign whole patients to cross-fitting folds.

    Parameters
    ----------
    patients : np.ndarray
        Patient ID per record.
    folds : int
        Number of folds.
    seed : int
        Permutation seed.

    Returns
    -------
    np.ndarray
        Fold index per record.
    """
    unique = np.unique(patients)
    order = np.random.default_rng(seed).permutation(len(unique))
    fold_of = dict(zip(unique[order], np.arange(len(unique)) % folds, strict=True))
    return np.array([fold_of[patient] for patient in patients])


def out_of_fold_residuals(
    medians: np.ndarray, covariates: np.ndarray, folds: np.ndarray, rank: int
) -> np.ndarray:
    """
    Compute each normal record's population residual from a model fitted without its fold.

    Parameters
    ----------
    medians : np.ndarray
        (records, LEADS * BINS) flattened median beats.
    covariates : np.ndarray
        (records, design) design rows.
    folds : np.ndarray
        Fold per record.
    rank : int
        Subspace dimension.

    Returns
    -------
    np.ndarray
        (records, LEADS * BINS) residuals in mV.
    """
    residual = np.empty_like(medians)
    for fold in np.unique(folds):
        held = folds == fold
        model = fit_population(medians[~held], covariates[~held], rank)
        for index in np.flatnonzero(held):
            residual[index] = population_residual(model, medians[index], covariates[index])
    return residual


def fit_intervals(intervals: np.ndarray, covariates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Regress QRS, QT and PR on the covariate design, skipping missing PR.

    Parameters
    ----------
    intervals : np.ndarray
        (records, 4) heart rate, QRS, QT, PR.
    covariates : np.ndarray
        (records, design) design rows.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (design, 3) coefficients and (3,) robust residual scales.
    """
    coefficients, scales = [], []
    for column in range(1, 4):
        found = np.isfinite(intervals[:, column])
        beta = np.linalg.lstsq(covariates[found], intervals[found, column], rcond=None)[0]
        coefficients.append(beta)
        scales.append(float(robust_scale(intervals[found, column] - covariates[found] @ beta)))
    return np.column_stack(coefficients), np.array(scales)


def fit_normative(
    features: list[RecordFeatures], age: np.ndarray, sex: np.ndarray, patients: np.ndarray
) -> Normative:
    """
    Fit models, scales and tail tables on clean normal records.

    Population residual scales and persistent tail tables come from patient-held-out (cross-fitted)
    residuals; focal scales, interval and beat tables from the same normals in sample.

    Parameters
    ----------
    features : list[RecordFeatures]
        Features of the clean normal records.
    age, sex, patients : np.ndarray
        Covariates and patient IDs in the same order.

    Returns
    -------
    Normative
        Fitted normative quantities.
    """
    intervals = np.stack([f.intervals for f in features])
    covariates = design(intervals[:, 0], age, sex)
    medians = np.stack([f.median.ravel() for f in features])
    folds = patient_folds(patients)
    models = {
        "full": fit_population(medians, covariates, SUBSPACE),
        "rank0": fit_population(medians, covariates, 0),
    }
    residuals = {
        name: out_of_fold_residuals(medians, covariates, folds, SUBSPACE if name == "full" else 0).reshape(
            -1, LEADS, BINS
        )
        for name in models
    }
    scales = {name: robust_scale(value) for name, value in residuals.items()}
    focal = np.concatenate([f.focal for f in features]).astype(np.float32)
    scales["focal"] = robust_scale(focal)
    tables = {}
    for name, model, noise in PERSISTENT:
        z = np.stack(
            [
                persistent_z(r, f, scales[model], noise)
                for r, f in zip(residuals[model], features, strict=True)
            ]
        )
        tables[name] = segment_tables(smooth_abs(z))
    for name, noise in (("focal", True), ("focal_raw", False)):
        magnitude = np.concatenate(
            [focal_score(f, scales["focal"], noise).astype(np.float32) for f in features]
        )
        tables[name] = segment_tables(magnitude)
    interval_coefficients, interval_scales = fit_intervals(intervals, covariates)
    normative = Normative(models, scales, tables, interval_coefficients, interval_scales)
    deviation = interval_values(intervals, covariates, normative)
    for column in range(3):
        found = np.isfinite(intervals[:, column + 1])
        tables[f"interval_{column}"] = tail_table(deviation[found, column])
    values = np.concatenate([beat_values(f)[1:] for f in features])
    tables["premature"] = tail_table(values[:, 0])
    tables["qrs_change"] = tail_table(values[:, 1])
    return normative
