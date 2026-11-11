"""
MLC-03 Part A: signal-quality measures of a 12-lead ECG window.

Protocol: ``docs/mlc03-data-quality-protocol-2026-11-11.md``, Part A, items 1-6. Every function takes the
cached first 10 s of one record, float64 ``(12, 5000)`` in mV at 500 Hz, leads in ``LEADS`` order, with NaN
where WFDB read a missing sample. ``measure_window`` returns one flat dictionary per record: per-lead measures
are named ``<measure>__<lead>``, record-level measures have plain names.

Measures that need a continuous signal use ``fill_missing`` (linear interpolation over missing samples), so
that a missing sample is counted once, by ``missing_samples``, and does not also look like a step or spike.
Integrity measures (runs, edges, clipping, quantisation, identities) use the window as stored.

Some sources hold or ramp the first and last samples (PTB-XL holds roughly the last 100 ms of every lead
constant; CPSC records start with a filter transient). Steps, quiet stretches, interior constant runs and
the baseline excursion are therefore measured between 0.3 s and 9.7 s (``INTERIOR``); the edges have their
own measures (``head_run``, ``tail_run``, ``edge_step``).

Beat detection uses two detectors: the multi-lead slope-energy detector of ``measurement_extractor042``
(``detect_r_peaks``, on the 0.5-40 Hz signal of eight independent leads) and WFDB's XQRS on single leads II
and V5. The median beat and its fiducials come from the same extractor (``median_beat``, ``delineate``).
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import maximum_filter1d, median_filter, minimum_filter1d, uniform_filter1d
from scipy.signal import butter, periodogram, sosfiltfilt
from wfdb.processing import xqrs_detect

from ecg_experiment.ecg_quality import longest_constant_runs
from ecg_experiment.eda.signals import LEADS
from ecg_experiment.measurement_extractor042 import SOS_CLEAN, delineate, detect_r_peaks, median_beat

FS = 500
SAMPLES = 5000
LIMB = ("I", "II", "III", "aVR", "aVL", "aVF")
PRECORDIAL = ("V1", "V2", "V3", "V4", "V5", "V6")
LEAD_INDEX = {lead: position for position, lead in enumerate(LEADS)}
RAIL_MV = 32.7
SOS_BASELINE = butter(2, 0.5, btype="low", fs=FS, output="sos")
SOS_HIGH = butter(4, 40.0, btype="high", fs=FS, output="sos")
MAINS_HZ = (50.0, 60.0)
MAINS_HALF_WIDTH_HZ = 0.3
MAINS_BACKGROUND_HZ = (1.0, 5.0)
MAINS_HARMONICS_HZ = (50.0, 60.0, 100.0, 120.0, 150.0, 180.0, 200.0, 240.0)
HF_WINDOW = int(0.05 * FS)
BLOCK = 5
BLOCK_RATE = FS // BLOCK
STEP_MEDIANS = (int(0.2 * BLOCK_RATE) + 1, int(0.6 * BLOCK_RATE) + 1)
STEP_LAG = 3
QUIET_WINDOW = int(0.1 * FS)
EDGE = int(0.3 * FS)
INTERIOR = slice(EDGE, SAMPLES - EDGE)
DETECT_LEADS = ("II", "V5")
MIN_DETECT_AMPLITUDE_MV = 0.02
DETECT_SCALE_MV = 2.0
MATCH_TOLERANCE = int(0.075 * FS)
NEIGHBOURS = 1

logging.getLogger("wfdb").setLevel(logging.ERROR)


def fill_missing(window: np.ndarray) -> np.ndarray:
    """
    Replace missing samples by linear interpolation along each lead.

    Parameters
    ----------
    window : np.ndarray
        Float ``(leads, samples)`` with NaN for missing samples.

    Returns
    -------
    np.ndarray
        Float64 copy without NaN; a lead that is missing throughout becomes zeros.
    """
    filled = np.array(window, dtype=np.float64, copy=True)
    positions = np.arange(filled.shape[1])
    for lead in filled:
        missing = np.isnan(lead)
        if missing.all():
            lead[:] = 0.0
        elif missing.any():
            lead[missing] = np.interp(positions[missing], positions[~missing], lead[~missing])
    return filled


def longest_true_run(mask: np.ndarray) -> int:
    """
    Length of the longest run of True values in a one-dimensional mask.

    Parameters
    ----------
    mask : np.ndarray
        Boolean array.

    Returns
    -------
    int
        Run length in samples; 0 when no value is True.
    """
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    if not len(edges):
        return 0
    return int((edges[1::2] - edges[::2]).max())


def finite_ranges(window: np.ndarray) -> np.ndarray:
    """
    Range (maximum minus minimum) of the finite samples of each lead.

    Parameters
    ----------
    window : np.ndarray
        Float ``(leads, samples)``.

    Returns
    -------
    np.ndarray
        Range per lead; 0 for a lead without finite samples or without samples.
    """
    return np.array([np.ptp(lead[np.isfinite(lead)]) if np.isfinite(lead).any() else 0.0 for lead in window])


def edge_runs(window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Count the samples at the start and end of each lead that equal its first and last sample.

    Parameters
    ----------
    window : np.ndarray
        Float ``(leads, samples)``; NaN breaks a run.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Head and tail run lengths in samples, one per lead (at least 1 unless the edge sample is NaN).
    """
    head = np.array([longest_true_run(np.cumprod(lead == lead[0]).astype(bool)) for lead in window])
    tail = np.array([longest_true_run(np.cumprod(lead[::-1] == lead[-1]).astype(bool)) for lead in window])
    return head, tail


def extreme_runs(window: np.ndarray) -> np.ndarray:
    """
    Longest run of consecutive samples equal to the lead's own maximum or minimum.

    Parameters
    ----------
    window : np.ndarray
        Float ``(leads, samples)``; NaN samples are ignored for the extremes and break runs.

    Returns
    -------
    np.ndarray
        Run length in samples per lead; 0 for a lead without finite samples.
    """
    runs = []
    for lead in window:
        finite = np.isfinite(lead)
        if not finite.any():
            runs.append(0)
            continue
        high, low = lead[finite].max(), lead[finite].min()
        runs.append(max(longest_true_run(lead == high), longest_true_run(lead == low)))
    return np.array(runs)


def quantisation_step(window: np.ndarray) -> np.ndarray:
    """
    Effective amplitude step of each lead, from the distinct values inside its interquartile range.

    The step is the span of the distinct microvolt values between the 25th and 75th percentiles divided by
    their count minus one: 1.0 when every microvolt value occurs, about 4.88 for a 4.88 uV converter stored
    in whole microvolts.

    Parameters
    ----------
    window : np.ndarray
        Float ``(leads, samples)`` in mV.

    Returns
    -------
    np.ndarray
        Step in microvolts per lead; NaN when fewer than two distinct values lie in the range.
    """
    steps = []
    for lead in window:
        values = np.round(lead[np.isfinite(lead)] * 1000.0)
        if len(values) < 2:
            steps.append(np.nan)
            continue
        low, high = np.percentile(values, [25, 75])
        distinct = np.unique(values[(values >= low) & (values <= high)])
        steps.append((distinct[-1] - distinct[0]) / (len(distinct) - 1) if len(distinct) > 1 else np.nan)
    return np.array(steps, dtype=np.float64)


def duplicate_leads(window: np.ndarray) -> list[str]:
    """
    Pairs of non-constant leads that are exact copies or exact sign-flips of each other.

    Parameters
    ----------
    window : np.ndarray
        Float ``(12, samples)`` in mV, ``LEADS`` order.

    Returns
    -------
    list[str]
        ``"A=B"`` for a copy and ``"A=-B"`` for a sign-flip, in lead order.
    """
    microvolts = np.round(np.nan_to_num(window, nan=0.0) * 1000.0).astype(np.int64)
    varying = np.ptp(microvolts, axis=1) > 0
    pairs = []
    for first in range(len(LEADS)):
        for second in range(first + 1, len(LEADS)):
            if not (varying[first] and varying[second]):
                continue
            if np.array_equal(microvolts[first], microvolts[second]):
                pairs.append(f"{LEADS[first]}={LEADS[second]}")
            elif np.array_equal(microvolts[first], -microvolts[second]):
                pairs.append(f"{LEADS[first]}=-{LEADS[second]}")
    return pairs


def identity_residuals(window: np.ndarray) -> dict[str, float]:
    """
    Residuals of the four limb-lead identities, in mV.

    The identities are III = II - I, aVR = -(I + II) / 2, aVL = I - II / 2 and aVF = II - I / 2.

    Parameters
    ----------
    window : np.ndarray
        Float ``(12, samples)`` in mV, ``LEADS`` order; NaN samples are skipped.

    Returns
    -------
    dict[str, float]
        ``res_<lead>_rms`` and ``res_<lead>_max`` for III, aVR, aVL and aVF.
    """
    lead_i, lead_ii = window[LEAD_INDEX["I"]], window[LEAD_INDEX["II"]]
    expected = {"III": lead_ii - lead_i, "aVR": -(lead_i + lead_ii) / 2, "aVL": lead_i - lead_ii / 2,
                "aVF": lead_ii - lead_i / 2}
    result = {}
    for lead, value in expected.items():
        residual = window[LEAD_INDEX[lead]] - value
        result[f"res_{lead}_rms"] = float(np.sqrt(np.nanmean(residual**2)))
        result[f"res_{lead}_max"] = float(np.nanmax(np.abs(residual)))
    return result


def robust_amplitude(clean: np.ndarray) -> np.ndarray:
    """
    Range between the 1st and 99th percentiles of each lead of the 0.5-40 Hz signal.

    Parameters
    ----------
    clean : np.ndarray
        Time-major ``(samples, 12)`` filtered signal in mV.

    Returns
    -------
    np.ndarray
        Amplitude in mV per lead.
    """
    low, high = np.percentile(clean, [1, 99], axis=0)
    return high - low


def baseline_excursion(filled: np.ndarray) -> np.ndarray:
    """
    Peak-to-peak excursion of each lead's content below 0.5 Hz, between 0.3 s and 9.7 s.

    The edges are left out because some sources hold or ramp the first and last samples
    (``edge_runs``), which a low-pass filter would turn into an excursion.

    Parameters
    ----------
    filled : np.ndarray
        Float ``(12, 5000)`` in mV without NaN.

    Returns
    -------
    np.ndarray
        Excursion in mV per lead (zero-phase second-order low-pass at 0.5 Hz).
    """
    return np.ptp(sosfiltfilt(SOS_BASELINE, filled, axis=1)[:, INTERIOR], axis=1)


def mains_amplitude(filled: np.ndarray, frequency: float) -> np.ndarray:
    """
    Amplitude of a sinusoid at one mains frequency, above the neighbouring spectrum.

    A Hann-window periodogram of the 10 s (0.1 Hz bins) is summed over ``frequency`` +/- 0.3 Hz after
    subtracting the median density of the bands 1-5 Hz below and above. A sinusoid of amplitude A has power
    A**2 / 2, so the amplitude is the square root of twice the excess power.

    Parameters
    ----------
    filled : np.ndarray
        Float ``(12, samples)`` in mV without NaN.
    frequency : float
        Mains frequency in Hz.

    Returns
    -------
    np.ndarray
        Amplitude in mV per lead; 0 when the band holds no excess power.
    """
    frequencies, density = periodogram(filled, fs=FS, window="hann", detrend="linear", axis=1)
    step = frequencies[1] - frequencies[0]
    band = np.abs(frequencies - frequency) <= MAINS_HALF_WIDTH_HZ
    distance = np.abs(frequencies - frequency)
    neighbours = (distance >= MAINS_BACKGROUND_HZ[0]) & (distance <= MAINS_BACKGROUND_HZ[1])
    background = np.median(density[:, neighbours], axis=1)
    excess = np.clip(density[:, band] - background[:, None], 0.0, None).sum(axis=1) * step
    return np.sqrt(2.0 * excess)


def hf_noise(filled: np.ndarray) -> np.ndarray:
    """
    Median over time of the 50 ms moving RMS of each lead above 40 Hz, with mains lines removed.

    The signal is high-passed at 40 Hz (zero-phase fourth-order Butterworth), the Fourier bins within
    0.5 Hz of 50, 60 Hz and their harmonics up to 240 Hz are set to zero, and the moving RMS is taken over
    50 ms. The median over time sits between QRS complexes for heart rates below about 200 per minute.

    Parameters
    ----------
    filled : np.ndarray
        Float ``(12, samples)`` in mV without NaN.

    Returns
    -------
    np.ndarray
        Noise level in mV per lead.
    """
    high = sosfiltfilt(SOS_HIGH, filled, axis=1)
    spectrum = np.fft.rfft(high, axis=1)
    frequencies = np.fft.rfftfreq(high.shape[1], 1.0 / FS)
    for line in MAINS_HARMONICS_HZ:
        spectrum[:, np.abs(frequencies - line) <= 0.5] = 0.0
    cleaned = np.fft.irfft(spectrum, n=high.shape[1], axis=1)
    moving = np.sqrt(uniform_filter1d(cleaned**2, size=HF_WINDOW, axis=1))
    return np.median(moving, axis=1)


def step_size(filled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Largest abrupt change of each lead's baseline, in mV, inside the record and at its edges.

    The lead is reduced to 100 Hz block means, its baseline is estimated by median filters of 0.2 s and then
    0.6 s (which remove QRS, P and T but keep a step), and the measure is the largest change of that
    baseline over 60 ms (three blocks before to three blocks after). Changes centred between 0.3 s and
    9.7 s count as interior, the others as edge.

    Parameters
    ----------
    filled : np.ndarray
        Float ``(12, 5000)`` in mV without NaN.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Interior and edge step size in mV per lead.
    """
    blocks = filled.reshape(filled.shape[0], -1, BLOCK).mean(axis=2)
    baseline = median_filter(blocks, size=(1, STEP_MEDIANS[0]), mode="nearest")
    baseline = median_filter(baseline, size=(1, STEP_MEDIANS[1]), mode="nearest")
    change = np.abs(baseline[:, 2 * STEP_LAG :] - baseline[:, : -2 * STEP_LAG])
    centre = np.arange(change.shape[1]) + STEP_LAG
    inside = (centre >= EDGE // BLOCK) & (centre < (SAMPLES - EDGE) // BLOCK)
    return change[:, inside].max(axis=1), change[:, ~inside].max(axis=1)


def spike_size(filled: np.ndarray) -> np.ndarray:
    """
    Largest absolute second difference of each lead, in mV.

    A one-sample spike of height h gives a second difference of about 2h; a QRS complex at 500 Hz gives
    much less.

    Parameters
    ----------
    filled : np.ndarray
        Float ``(12, samples)`` in mV without NaN.

    Returns
    -------
    np.ndarray
        Largest ``|x[n+1] - 2 x[n] + x[n-1]|`` per lead.
    """
    return np.abs(np.diff(filled, n=2, axis=1)).max(axis=1)


def quiet_runs(window: np.ndarray, tolerance_mv: np.ndarray) -> np.ndarray:
    """
    Longest stretch over which each lead stays within a small band.

    A sample belongs to a quiet stretch when the 100 ms window centred on it has a range (maximum minus
    minimum) of at most the lead's tolerance. The measure is the longest run of such samples plus the
    window length, in samples.

    Parameters
    ----------
    window : np.ndarray
        Float ``(leads, samples)``; NaN samples break a run.
    tolerance_mv : np.ndarray
        Allowed range per lead, in mV.

    Returns
    -------
    np.ndarray
        Run length in samples per lead (0 when no window is quiet).
    """
    filled = np.nan_to_num(window, nan=np.inf)
    spread = maximum_filter1d(filled, QUIET_WINDOW, axis=1) - minimum_filter1d(filled, QUIET_WINDOW, axis=1)
    quiet = spread <= tolerance_mv[:, None]
    runs = np.array([longest_true_run(row) for row in quiet])
    return np.where(runs > 0, np.minimum(runs + QUIET_WINDOW - 1, window.shape[1]), 0)


def detect_single(lead: np.ndarray, clean_lead: np.ndarray) -> np.ndarray:
    """
    R peaks of one lead by WFDB's XQRS, after scaling the lead to a fixed amplitude.

    XQRS learns its thresholds from eight beats and otherwise falls back to fixed thresholds in mV, which miss
    low-amplitude leads. The lead is therefore centred on its median and scaled so that the 1-99 percentile
    range of its 0.5-40 Hz signal is ``DETECT_SCALE_MV``; on 775 train and validation smoke records this
    reduced lead-II failures (fewer than two beats) from 59 to 1.

    Parameters
    ----------
    lead : np.ndarray
        Float samples in mV without NaN.
    clean_lead : np.ndarray
        The same lead after the 0.5-40 Hz filter.

    Returns
    -------
    np.ndarray
        Sorted sample locations; empty when the filtered range is below 0.02 mV.
    """
    low, high = np.percentile(clean_lead, [1, 99])
    if high - low < MIN_DETECT_AMPLITUDE_MV:
        return np.array([], dtype=np.int64)
    scaled = (lead - np.median(lead)) * DETECT_SCALE_MV / (high - low)
    return np.sort(np.asarray(xqrs_detect(scaled, fs=FS, verbose=False), dtype=np.int64))


def match_f1(first: np.ndarray, second: np.ndarray, tolerance: int = MATCH_TOLERANCE) -> float:
    """
    Agreement of two beat lists: twice the matched beats over the total, matching within a tolerance.

    Each beat of ``first`` is matched to the nearest unmatched beat of ``second`` within ``tolerance``
    samples, in time order.

    Parameters
    ----------
    first, second : np.ndarray
        Sorted sample locations.
    tolerance : int
        Largest distance in samples for a match.

    Returns
    -------
    float
        Value in [0, 1]; NaN when both lists are empty.
    """
    if not len(first) and not len(second):
        return np.nan
    used = np.zeros(len(second), dtype=bool)
    matched = 0
    for beat in first:
        distance = np.where(used, np.inf, np.abs(second - beat))
        if len(distance) and distance.min() <= tolerance:
            used[int(np.argmin(distance))] = True
            matched += 1
    return 2.0 * matched / (len(first) + len(second))


def heart_rate(peaks: np.ndarray) -> float:
    """
    Heart rate from the median RR interval.

    Parameters
    ----------
    peaks : np.ndarray
        Sorted sample locations.

    Returns
    -------
    float
        Beats per minute; NaN with fewer than two peaks.
    """
    if len(peaks) < 2:
        return np.nan
    return float(60.0 * FS / np.median(np.diff(peaks)))


def beat_morphology(clean: np.ndarray, peaks: np.ndarray) -> dict[str, float]:
    """
    Per-lead QRS, P and T measures of the median beat, by the extractor's own rules.

    The median beat and its fiducials come from ``measurement_extractor042.median_beat`` and ``delineate``.
    Per lead: ``qrs_area`` (signed sum over QRS onset to offset, mV ms), ``qrs_ptp`` (range over the same
    span, mV), ``p_amp`` (signed extreme of the baseline-corrected P window, the extractor's rule for lead II
    applied to every lead) and ``t_amp`` (signed extreme of the T window).

    Parameters
    ----------
    clean : np.ndarray
        Time-major ``(5000, 12)`` 0.5-40 Hz signal in mV.
    peaks : np.ndarray
        R-peak locations of the multi-lead detector.

    Returns
    -------
    dict[str, float]
        ``beat_similarity``, ``qrs_onset_ms``, ``qrs_duration_ms`` and the four per-lead measures; NaN when
        fewer than two complete beats exist.
    """
    names = [f"{measure}__{lead}" for measure in ("qrs_area", "qrs_ptp", "p_amp", "t_amp") for lead in LEADS]
    result = dict.fromkeys(["beat_similarity", "qrs_duration_ms", *names], np.nan)
    rr = np.diff(peaks)
    template, similarity = median_beat(clean, peaks)
    if len(rr) < 1 or np.isnan(template).any():
        return result
    fiducials = delineate(template, float(rr.mean()) / FS * 1000.0)
    onset, offset, centered = fiducials["onset"], fiducials["offset"], fiducials["centered"]
    result["beat_similarity"] = similarity
    result["qrs_duration_ms"] = (offset - onset) / FS * 1000.0
    p_window = centered[fiducials["p_start"] : fiducials["p_stop"]]
    p_window = p_window - np.median(p_window, axis=0)
    t_window = centered[fiducials["t_start"] : fiducials["t_stop"]]
    for position, lead in enumerate(LEADS):
        qrs = centered[onset : offset + 1, position]
        result[f"qrs_area__{lead}"] = float(qrs.sum() / FS * 1000.0)
        result[f"qrs_ptp__{lead}"] = float(np.ptp(qrs))
        result[f"p_amp__{lead}"] = float(p_window[np.argmax(np.abs(p_window[:, position])), position])
        result[f"t_amp__{lead}"] = float(t_window[np.argmax(np.abs(t_window[:, position])), position])
    return result


def per_lead(name: str, values: np.ndarray) -> dict[str, float]:
    """
    Name one value per lead as ``<name>__<lead>``.

    Parameters
    ----------
    name : str
        Measure name.
    values : np.ndarray
        Twelve values in ``LEADS`` order.

    Returns
    -------
    dict[str, float]
        One entry per lead.
    """
    return {f"{name}__{lead}": float(value) for lead, value in zip(LEADS, values, strict=True)}


def beat_measures(filled: np.ndarray, clean: np.ndarray) -> dict[str, float]:
    """
    Beats by two detectors and two leads, their agreement, heart rates and the median-beat measures.

    Parameters
    ----------
    filled : np.ndarray
        Float ``(12, 5000)`` in mV without NaN.
    clean : np.ndarray
        Time-major ``(5000, 12)`` 0.5-40 Hz signal.

    Returns
    -------
    dict[str, float]
        ``n_beats_multi``, ``hr_multi``, ``n_beats_<lead>`` and ``hr_<lead>`` for II and V5 (XQRS),
        ``agree_detectors`` (multi-lead against XQRS on II), ``agree_leads`` (XQRS on II against V5),
        ``rr_max_s`` (longest multi-lead RR interval) and the outputs of ``beat_morphology``.
    """
    multi = detect_r_peaks(clean)
    single = {lead: detect_single(filled[LEAD_INDEX[lead]], clean[:, LEAD_INDEX[lead]])
              for lead in DETECT_LEADS}
    result: dict[str, float] = {"n_beats_multi": float(len(multi)), "hr_multi": heart_rate(multi)}
    for lead, peaks in single.items():
        result[f"n_beats_{lead}"] = float(len(peaks))
        result[f"hr_{lead}"] = heart_rate(peaks)
    result["agree_detectors"] = match_f1(multi, single["II"])
    result["agree_leads"] = match_f1(single["II"], single["V5"])
    result["rr_max_s"] = float(np.diff(multi).max() / FS) if len(multi) > 1 else np.nan
    result.update(beat_morphology(clean, multi))
    return result


def measure_window(window: np.ndarray) -> dict[str, float | str]:
    """
    Every Part A measure of one cached window.

    Parameters
    ----------
    window : np.ndarray
        Float ``(12, 5000)`` in mV at 500 Hz, ``LEADS`` order, NaN for missing samples.

    Returns
    -------
    dict[str, float | str]
        Record-level and per-lead measures (see the module docstring); ``duplicate_pairs`` is text.

    Raises
    ------
    ValueError
        If the window does not have the shape ``(12, 5000)``.
    """
    window = np.asarray(window, dtype=np.float64)
    if window.shape != (len(LEADS), SAMPLES):
        raise ValueError(f"expected (12, 5000), got {window.shape}")
    missing = np.isnan(window)
    filled = fill_missing(window)
    clean = sosfiltfilt(SOS_CLEAN, filled.T, axis=0)
    head, tail = edge_runs(window)
    finite_range = finite_ranges(window)
    step = quantisation_step(window)
    tolerance = 2.0 * np.nan_to_num(step, nan=1.0) / 1000.0
    pairs = duplicate_leads(window)
    result: dict[str, float | str] = {
        "missing_samples": float(missing.sum()),
        "missing_leads": float(missing.any(axis=1).sum()),
        "peak_abs_mv": float(np.nanmax(np.abs(window))) if (~missing).any() else np.nan,
        "rail_samples": float((np.abs(np.nan_to_num(window)) >= RAIL_MV).sum()),
        "duplicate_pairs": ";".join(pairs),
        "n_duplicate_pairs": float(len(pairs)),
    }
    result.update(per_lead("flat", finite_range == 0))
    result.update(per_lead("head_run", head))
    result.update(per_lead("tail_run", tail))
    result.update(per_lead("const_run", longest_constant_runs(window)))
    result.update(per_lead("const_run_interior", longest_constant_runs(window[:, INTERIOR])))
    result.update(per_lead("extreme_run", extreme_runs(window)))
    result.update(per_lead("quant_step_uv", step))
    result.update(per_lead("quiet_run", quiet_runs(window[:, INTERIOR], tolerance)))
    result.update(identity_residuals(window))
    result.update(per_lead("amp_robust", robust_amplitude(clean)))
    result.update(per_lead("baseline_ptp", baseline_excursion(filled)))
    for frequency in MAINS_HZ:
        result.update(per_lead(f"mains{int(frequency)}", mains_amplitude(filled, frequency)))
    result.update(per_lead("hf_noise", hf_noise(filled)))
    interior_step, edge_step = step_size(filled)
    result.update(per_lead("step", interior_step))
    result.update(per_lead("edge_step", edge_step))
    result.update(per_lead("spike", spike_size(filled)))
    result.update(beat_measures(filled, clean))
    return result


def measure_full_record(signal: np.ndarray) -> dict[str, float]:
    """
    Integrity measures of a whole raw recording (used for records longer than 10 s).

    Parameters
    ----------
    signal : np.ndarray
        Float ``(12, samples)`` in mV, ``LEADS`` order, NaN for missing samples.

    Returns
    -------
    dict[str, float]
        ``full_samples``, ``full_missing_samples``, ``full_flat_leads`` (constant over the whole record),
        ``full_const_run_max`` (longest constant run of any lead, samples), ``full_extreme_run_max``,
        ``full_rail_samples``, ``full_peak_abs_mv``, ``full_tail_run_min`` (samples at the end that every
        lead holds constant) and ``after10_flat_leads`` (leads constant after the first 10 s but not
        before).
    """
    signal = np.asarray(signal, dtype=np.float64)
    finite = np.isfinite(signal)
    ranges = finite_ranges(signal)
    head_range = finite_ranges(signal[:, :SAMPLES])
    rest = signal[:, SAMPLES:]
    rest_range = finite_ranges(rest)
    _, tail = edge_runs(signal)
    return {
        "full_samples": float(signal.shape[1]),
        "full_missing_samples": float((~finite).sum()),
        "full_flat_leads": float((ranges == 0).sum()),
        "full_const_run_max": float(longest_constant_runs(signal).max()),
        "full_extreme_run_max": float(extreme_runs(signal).max()),
        "full_rail_samples": float((np.abs(np.nan_to_num(signal)) >= RAIL_MV).sum()),
        "full_peak_abs_mv": float(np.nanmax(np.abs(signal))),
        "full_tail_run_min": float(tail.min()),
        "after10_flat_leads": float(((rest_range == 0) & (head_range > 0)).sum()) if rest.shape[1] else 0.0,
    }
