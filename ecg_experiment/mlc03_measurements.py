"""
Second, independent ECG measurements for the MLC-03 label audit (method B), and synthetic test records.

Method A is the frozen extractor of MLC-02 (``measurement_extractor042``): R peaks from slope energy in eight
leads, one median of all beats, and slope-threshold delineation. Method B differs at every step:

- R peaks: ``wfdb.processing.xqrs_detect`` on one lead (II, else V5, I or V2 when a lead is flat or XQRS
  finds fewer than three beats on it), moved to the nearest maximum of the eight-lead QRS energy, plus energy
  maxima far from every detection (wide ectopic complexes that XQRS misses);
- beats: each beat is correlated with a first median over the QRS core; beats with correlation of at least
  ``DOMINANT_CORRELATION`` form the dominant median beat, and premature beats that do not match it are counted
  as premature aberrant complexes;
- QRS: onset and offset where the smoothed spatial velocity of the dominant beat stays below
  ``noise + QRS_FRACTION * (peak - noise)`` for ``QUIET_MS``;
- P wave: the maximum of the atrial envelope (root mean square of I, II, V1 and V2 around the PR baseline) in
  a window before the QRS that excludes the preceding T wave; onset where the envelope falls to
  ``P_FRACTION`` of its height; P presence from the template's signal-to-noise ratio, and P consistency as the
  median correlation of each dominant beat's P segment with the template's;
- per-lead amplitudes, frontal axis from net QRS area in I and aVF, V1 morphology (number of positive peaks,
  terminal deflection), Q-wave duration and depth, ST level 60 ms after the QRS offset and signed T amplitude,
  all relative to the PR baseline.

Input is one record: ``[12, samples]`` in mV at 500 Hz, leads I, II, III, aVR, aVL, aVF, V1-V6; missing
samples may be NaN. ``measure_record`` returns a flat dictionary with a ``status``: ``ok``, ``flat`` (no
usable lead for R peaks), ``few_beats`` (fewer than three complete beats) or ``no_dominant`` (fewer than two
beats match the median). Measurements that cannot be made are NaN.

``synthetic_record`` builds a twelve-lead record from a heart vector made of raised-cosine lobes, so that
every interval, the frontal axis, ST level and T amplitude are known exactly; the limb leads obey Einthoven's
and Goldberger's relations by construction.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, detrend, find_peaks, sosfiltfilt
from wfdb import processing

FS = 500
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
INDEPENDENT = (0, 1, 6, 7, 8, 9, 10, 11)
ATRIAL = (0, 1, 6, 7)
DETECTION_LEADS = (1, 10, 0, 7)
SOS_BEAT = butter(2, [0.5, 40.0], btype="band", fs=FS, output="sos")
PRE = int(0.5 * FS)
POST = int(0.6 * FS)
DOMINANT_CORRELATION = 0.85
ABERRANT_CORRELATION = 0.80
PREMATURE_FRACTION = 0.85
GAP_FRACTION = 0.3
QRS_FRACTION = 0.04
P_FRACTION = 0.2
QUIET_MS = 10
FLAT_MV = 0.01
UPRIGHT_T = ("I", "II", "V3", "V4", "V5", "V6")
LIMB_ANGLES = {"I": 0.0, "II": 60.0, "III": 120.0, "aVR": -150.0, "aVL": -30.0, "aVF": 90.0}
LIMB_GAIN = {
    "I": 1.0,
    "II": 1.0,
    "III": 1.0,
    "aVR": np.sqrt(3) / 2,
    "aVL": np.sqrt(3) / 2,
    "aVF": np.sqrt(3) / 2,
}
PRECORDIAL_ANGLES = {"V1": 120.0, "V2": 90.0, "V3": 75.0, "V4": 60.0, "V5": 30.0, "V6": 0.0}


def samples(ms: float) -> int:
    """
    Convert a duration to a number of samples.

    Parameters
    ----------
    ms : float
        Duration in milliseconds.

    Returns
    -------
    int
        Rounded sample count at ``FS``.
    """
    return int(round(ms * FS / 1000.0))


def flat_leads(signal: np.ndarray) -> np.ndarray:
    """
    Leads without usable signal.

    Parameters
    ----------
    signal : np.ndarray
        ``[12, samples]`` in mV, NaN where missing.

    Returns
    -------
    np.ndarray
        Boolean per lead: all missing, or a standard deviation below ``FLAT_MV``.
    """
    finite = np.isfinite(signal)
    std = np.array([signal[lead, finite[lead]].std() if finite[lead].any() else 0.0 for lead in range(12)])
    return std < FLAT_MV


def bandpass(signal: np.ndarray) -> np.ndarray:
    """
    Zero-phase 0.5-40 Hz band-pass, missing samples set to zero first.

    Parameters
    ----------
    signal : np.ndarray
        ``[12, samples]`` in mV.

    Returns
    -------
    np.ndarray
        Float64 filtered signal of the same shape.
    """
    return sosfiltfilt(SOS_BEAT, np.nan_to_num(np.asarray(signal, dtype=np.float64)), axis=-1)


def spatial_velocity(clean: np.ndarray, leads: np.ndarray) -> np.ndarray:
    """
    Smoothed spatial velocity over the given leads, in mV per ms.

    Parameters
    ----------
    clean : np.ndarray
        ``[12, samples]`` filtered signal or beat.
    leads : np.ndarray
        Lead indices to use.

    Returns
    -------
    np.ndarray
        ``[samples]`` velocity smoothed over ``QUIET_MS``.
    """
    slope = np.diff(clean[leads], axis=-1, prepend=clean[leads, :1]) * FS / 1000.0
    return uniform_filter1d(np.sqrt((slope**2).sum(axis=0)), size=samples(QUIET_MS))


def xqrs_peaks(signal: np.ndarray, flat: np.ndarray) -> np.ndarray:
    """
    XQRS detections on the first usable lead of ``DETECTION_LEADS``.

    XQRS sometimes learns nothing on a lead and returns no detection; the next lead is then tried.

    Parameters
    ----------
    signal : np.ndarray
        ``[12, samples]`` raw mV.
    flat : np.ndarray
        Output of ``flat_leads``.

    Returns
    -------
    np.ndarray
        Detections of the first non-flat lead (II, V5, I, V2) that yields at least three; otherwise none.
    """
    for lead in DETECTION_LEADS:
        if flat[lead]:
            continue
        found = processing.xqrs_detect(
            sig=np.nan_to_num(signal[lead].astype(np.float64)), fs=FS, verbose=False
        )
        if len(found) >= 3:
            return np.asarray(found, dtype=int)
    return np.array([], dtype=int)


def detect_peaks(signal: np.ndarray, clean: np.ndarray, flat: np.ndarray) -> np.ndarray:
    """
    R peaks of method B.

    XQRS runs on one lead (``xqrs_peaks``). Its detections are moved to the maximum of the QRS energy (spatial
    velocity smoothed over 50 ms, a single hump per complex) within 75 ms. Energy maxima of at least
    ``GAP_FRACTION`` of the median detected beat that lie more than 250 ms from every detection are added,
    because a single-lead detector misses some wide ectopic complexes.

    Parameters
    ----------
    signal : np.ndarray
        ``[12, samples]`` raw mV.
    clean : np.ndarray
        Band-passed signal.
    flat : np.ndarray
        Output of ``flat_leads``.

    Returns
    -------
    np.ndarray
        Ascending sample positions, at least 200 ms apart.
    """
    found = xqrs_peaks(signal, flat)
    if not len(found):
        return found
    energy = uniform_filter1d(
        spatial_velocity(clean, np.array([i for i in INDEPENDENT if not flat[i]])), size=samples(50)
    )
    half = samples(75)
    moved = np.array(
        [max(p - half, 0) + int(np.argmax(energy[max(p - half, 0) : p + half])) for p in found], dtype=int
    )
    return merge_peaks(np.concatenate([moved, gap_candidates(energy, moved)]), energy)


def gap_candidates(energy: np.ndarray, detected: np.ndarray) -> np.ndarray:
    """
    QRS-energy maxima far from every detected beat.

    Parameters
    ----------
    energy : np.ndarray
        Smoothed spatial velocity.
    detected : np.ndarray
        Detected peak positions.

    Returns
    -------
    np.ndarray
        Maxima of at least ``GAP_FRACTION`` times the median detected energy, more than 250 ms from any
        detection and from the record edges' first and last 150 ms.
    """
    if not len(detected):
        return np.array([], dtype=int)
    height = GAP_FRACTION * float(np.median(energy[detected]))
    candidates, _ = find_peaks(energy, height=height, distance=samples(250))
    far = np.min(np.abs(candidates[:, None] - detected[None, :]), axis=1) > samples(250)
    inside = (candidates > samples(150)) & (candidates < len(energy) - samples(150))
    return candidates[far & inside]


def merge_peaks(peaks: np.ndarray, energy: np.ndarray) -> np.ndarray:
    """
    Keep the stronger of two peaks closer than 200 ms.

    Parameters
    ----------
    peaks : np.ndarray
        Candidate positions, any order.
    energy : np.ndarray
        Smoothed spatial velocity.

    Returns
    -------
    np.ndarray
        Ascending positions at least 200 ms apart.
    """
    kept: list[int] = []
    for peak in sorted({int(p) for p in peaks}):
        if kept and peak - kept[-1] < samples(200):
            if energy[peak] > energy[kept[-1]]:
                kept[-1] = peak
            continue
        kept.append(peak)
    return np.asarray(kept, dtype=int)


def beat_stack(clean: np.ndarray, peaks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Beats that fit entirely inside the record.

    Parameters
    ----------
    clean : np.ndarray
        Band-passed ``[12, samples]``.
    peaks : np.ndarray
        R-peak positions.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``[beats, 12, PRE + POST]`` and the index into ``peaks`` of each kept beat.
    """
    inside = np.flatnonzero((peaks - PRE >= 0) & (peaks + POST <= clean.shape[1]))
    if not len(inside):
        return np.empty((0, 12, PRE + POST)), inside
    return np.stack([clean[:, peaks[i] - PRE : peaks[i] + POST] for i in inside]), inside


def core_correlation(beats: np.ndarray, template: np.ndarray, leads: np.ndarray) -> np.ndarray:
    """
    Correlation of each beat with a template over the QRS core (60 ms before to 80 ms after the R peak).

    Parameters
    ----------
    beats : np.ndarray
        ``[beats, 12, length]``.
    template : np.ndarray
        ``[12, length]``.
    leads : np.ndarray
        Leads to concatenate.

    Returns
    -------
    np.ndarray
        One Pearson correlation per beat.
    """
    core = slice(PRE - samples(60), PRE + samples(80))
    return segment_correlation(beats[:, leads, core], template[leads, core])


def classify_beats(beats: np.ndarray, leads: np.ndarray, rr_before: np.ndarray) -> dict[str, np.ndarray]:
    """
    Dominant, premature and aberrant beats.

    Parameters
    ----------
    beats : np.ndarray
        ``[beats, 12, length]``.
    leads : np.ndarray
        Non-flat independent leads.
    rr_before : np.ndarray
        RR interval preceding each beat in ms, NaN for the first detected beat.

    Returns
    -------
    dict[str, np.ndarray]
        ``correlation`` with the dominant template, ``dominant`` and ``premature_aberrant`` masks.
    """
    first = core_correlation(beats, np.median(beats, axis=0), leads)
    dominant = first >= DOMINANT_CORRELATION
    correlation = first
    if dominant.sum() >= 2:
        correlation = core_correlation(beats, np.median(beats[dominant], axis=0), leads)
        dominant = correlation >= DOMINANT_CORRELATION
    typical_rr = np.nanmedian(rr_before[dominant]) if np.isfinite(rr_before[dominant]).any() else np.nan
    premature = rr_before < PREMATURE_FRACTION * typical_rr
    return {
        "correlation": correlation,
        "dominant": dominant,
        "premature_aberrant": premature & (correlation < ABERRANT_CORRELATION),
    }


def region(velocity: np.ndarray, start: int, low: int, high: int, threshold: float) -> tuple[int, int]:
    """
    Grow a region around ``start`` until the velocity stays below a threshold for ``QUIET_MS``.

    Parameters
    ----------
    velocity : np.ndarray
        Spatial velocity of a beat.
    start : int
        Sample inside the region.
    low, high : int
        Limits of the search.
    threshold : float
        Quiet level.

    Returns
    -------
    tuple[int, int]
        First and last sample of the region.
    """
    quiet = samples(QUIET_MS)
    onset = next((i for i in range(start, low, -1) if (velocity[i - quiet : i] < threshold).all()), low)
    offset = next((i for i in range(start, high) if (velocity[i : i + quiet] < threshold).all()), high)
    return onset, offset


def qrs_bounds(
    template: np.ndarray, leads: np.ndarray, fraction: float = QRS_FRACTION
) -> tuple[int, int, float]:
    """
    QRS onset and offset of the dominant beat.

    Parameters
    ----------
    template : np.ndarray
        ``[12, length]`` median beat with the R peak at ``PRE``.
    leads : np.ndarray
        Non-flat independent leads.
    fraction : float
        Threshold as a fraction of the velocity peak above the noise level.

    Returns
    -------
    tuple[int, int, float]
        Onset and offset samples, and the template's velocity noise level.
    """
    velocity = spatial_velocity(template, leads)
    near = slice(PRE - samples(60), PRE + samples(60))
    peak_at = near.start + int(np.argmax(velocity[near]))
    noise = float(np.percentile(velocity, 10))
    threshold = noise + fraction * (velocity[peak_at] - noise)
    onset, offset = region(velocity, peak_at, PRE - samples(150), PRE + samples(200), threshold)
    return onset, offset, noise


def baseline(template: np.ndarray, onset: int) -> np.ndarray:
    """
    PR-segment level of each lead (median from 16 to 4 ms before the QRS onset).

    Parameters
    ----------
    template : np.ndarray
        ``[12, length]``.
    onset : int
        QRS onset sample.

    Returns
    -------
    np.ndarray
        One level per lead, in mV.
    """
    return np.median(template[:, onset - samples(16) : onset - samples(4)], axis=1)


def t_end(centered: np.ndarray, offset: int, rr_ms: float) -> tuple[int, int]:
    """
    End of the T wave of the template: the steepest descending tangent of the T envelope meets zero.

    Parameters
    ----------
    centered : np.ndarray
        Baseline-corrected template ``[12, length]``.
    offset : int
        QRS offset sample.
    rr_ms : float
        Typical RR interval.

    Returns
    -------
    tuple[int, int]
        T-end sample (at most the window stop) and the T-window stop used for amplitudes.
    """
    stop = min(centered.shape[1] - 1, offset + samples(min(450.0, 0.7 * rr_ms)))
    start = offset + samples(80)
    if stop <= start + samples(40):
        return stop, stop
    envelope = uniform_filter1d(
        np.sqrt((centered[list(INDEPENDENT), start:stop] ** 2).mean(axis=0)), size=samples(10)
    )
    peak = int(np.argmax(envelope))
    if len(envelope) - peak < 3:
        return stop, stop
    slope = np.gradient(envelope[peak:])
    rising = np.flatnonzero(slope[1:] >= 0)
    slope = slope[: rising[0] + 1] if len(rising) else slope
    if len(slope) < 3 or slope.min() >= 0:
        return stop, stop
    steepest = peak + int(np.argmin(slope))
    return min(start + steepest + int(envelope[steepest] / -slope.min()), stop), stop


def p_window(onset: int, rr_ms: float, t_end_after_r: int) -> tuple[int, int]:
    """
    Search window of the P wave, after the preceding beat's T wave and at least 40 ms before the QRS.

    Parameters
    ----------
    onset : int
        QRS onset sample of the template.
    rr_ms : float
        Typical RR interval.
    t_end_after_r : int
        Samples from the R peak to the T end.

    Returns
    -------
    tuple[int, int]
        First and last sample; the window is empty when the first is not below the last.
    """
    previous_t_end = PRE - samples(rr_ms) + t_end_after_r + samples(10)
    return max(0, onset - samples(400), previous_t_end), onset - samples(40)


def p_wave(
    beats: np.ndarray,
    template: np.ndarray,
    base: np.ndarray,
    bounds: tuple[int, int],
    onset: int,
    fraction: float = P_FRACTION,
) -> dict[str, float]:
    """
    P-wave presence, consistency, amplitude and onset on the dominant beats.

    Parameters
    ----------
    beats : np.ndarray
        Dominant beats ``[beats, 12, length]``.
    template : np.ndarray
        Their median.
    base : np.ndarray
        PR-segment level per lead.
    bounds : tuple[int, int]
        Output of ``p_window``.
    onset : int
        QRS onset sample.
    fraction : float
        P onset where the envelope falls to this fraction of its height.

    Returns
    -------
    dict[str, float]
        ``p_amp_mv`` (envelope height), ``p_snr``, ``p_consistency``, ``pr_ms`` and ``p_window_ms``.
    """
    start, stop = bounds
    result = dict.fromkeys(("p_amp_mv", "p_snr", "p_consistency", "pr_ms"), np.nan)
    result["p_window_ms"] = max(stop - start, 0) * 1000.0 / FS
    if stop - start < samples(60):
        return result
    leads = list(ATRIAL)
    centered = template[leads] - base[leads, None]
    envelope = uniform_filter1d(np.sqrt((centered**2).mean(axis=0)), size=samples(20))[start:stop]
    peak = int(np.argmax(envelope))
    floor = float(np.percentile(envelope, 10))
    height = float(envelope[peak] - floor)
    residual = beats[:, leads, start:stop] - template[None, leads, start:stop]
    noise = float(residual.std()) / np.sqrt(len(beats))
    result["p_amp_mv"], result["p_snr"] = height, height / noise if noise > 0 else np.inf
    segment = slice(max(start + peak - samples(50), 0), start + peak + samples(50))
    result["p_consistency"] = float(
        np.median(segment_correlation(beats[:, leads, segment], template[leads, segment]))
    )
    below = np.flatnonzero(envelope[:peak] - floor < fraction * height)
    if len(below):
        result["pr_ms"] = (onset - (start + int(below[-1]))) * 1000.0 / FS
    return result


def segment_correlation(segments: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Correlation of each beat's multi-lead segment with a reference segment, each lead detrended first.

    A straight line is removed from every lead of every segment, so that baseline offsets and slow drifts,
    which differ between leads, do not create correlation by themselves.

    Parameters
    ----------
    segments : np.ndarray
        ``[beats, leads, samples]``.
    reference : np.ndarray
        ``[leads, samples]``.

    Returns
    -------
    np.ndarray
        One correlation per beat; 0 when a segment is constant.
    """
    flat = detrend(segments, axis=-1).reshape(len(segments), -1)
    ref = detrend(reference, axis=-1).ravel()
    denominator = np.linalg.norm(flat, axis=1) * np.linalg.norm(ref)
    return np.divide(flat @ ref, denominator, out=np.zeros(len(segments)), where=denominator > 0)


def q_wave(lead: np.ndarray) -> tuple[float, float]:
    """
    Duration and depth of an initial negative QRS deflection.

    Parameters
    ----------
    lead : np.ndarray
        Baseline-corrected QRS samples of one lead, onset to offset.

    Returns
    -------
    tuple[float, float]
        Duration in ms (0 when the first deflection beyond 0.02 mV is positive; the whole QRS for a QS
        complex) and depth in mV.
    """
    departed = np.flatnonzero(np.abs(lead) > 0.02)
    if not len(departed) or lead[departed[0]] > 0:
        return 0.0, 0.0
    above = np.flatnonzero(lead[departed[0] :] > 0)
    end = departed[0] + (int(above[0]) if len(above) else len(lead) - departed[0])
    return end * 1000.0 / FS, float(-lead[:end].min())


def v1_morphology(v1: np.ndarray) -> dict[str, float]:
    """
    Positive peaks and terminal deflection of the QRS in V1.

    Parameters
    ----------
    v1 : np.ndarray
        Baseline-corrected V1 samples from QRS onset to offset.

    Returns
    -------
    dict[str, float]
        ``v1_positive_peaks`` (peaks above 0.05 mV with 0.05 mV prominence), ``v1_terminal_mv`` (mean of the
        last 40 ms) and ``v1_rprime_mv`` (height of the last positive peak when there are at least two).
    """
    peaks, _ = find_peaks(np.concatenate([[0.0], v1, [0.0]]), height=0.05, prominence=0.05)
    terminal = float(v1[-samples(40) :].mean()) if len(v1) >= samples(40) else np.nan
    rprime = float(v1[peaks[-1] - 1]) if len(peaks) >= 2 else 0.0
    return {"v1_positive_peaks": float(len(peaks)), "v1_terminal_mv": terminal, "v1_rprime_mv": rprime}


def lead_amplitudes(
    centered: np.ndarray, onset: int, offset: int, j_plus: int, t_window: tuple[int, int]
) -> dict[str, float]:
    """
    Per-lead R, S and Q amplitudes, Q duration, net QRS area, terminal QRS level, ST levels and T amplitude.

    Parameters
    ----------
    centered : np.ndarray
        Baseline-corrected template ``[12, length]``.
    onset, offset : int
        QRS bounds.
    j_plus : int
        Sample 60 ms after the QRS offset.
    t_window : tuple[int, int]
        Start and stop of the T search.

    Returns
    -------
    dict[str, float]
        ``r_<lead>``, ``s_<lead>``, ``q_<lead>``, ``qdur_<lead>``, ``area_<lead>`` (mV ms), ``term_<lead>``
        (mean of the last 40 ms of the QRS), ``st60_<lead>`` and ``st80_<lead>`` (60 and 80 ms after the QRS
        offset) and ``t_<lead>`` for every lead.
    """
    values: dict[str, float] = {}
    for index, name in enumerate(LEADS):
        qrs = centered[index, onset : offset + 1]
        r_at = int(np.argmax(qrs))
        values[f"r_{name}"] = float(max(qrs.max(), 0.0))
        values[f"s_{name}"] = float(max(-qrs[r_at:].min(), 0.0))
        values[f"qdur_{name}"], values[f"q_{name}"] = q_wave(qrs)
        values[f"area_{name}"] = float(qrs.sum() * 1000.0 / FS)
        values[f"term_{name}"] = float(qrs[-samples(40) :].mean())
        values[f"st60_{name}"] = float(centered[index, j_plus - samples(4) : j_plus + samples(4)].mean())
        st80 = j_plus + samples(20)
        values[f"st80_{name}"] = float(centered[index, st80 - samples(4) : st80 + samples(4)].mean())
        t_wave = centered[index, t_window[0] : t_window[1]]
        values[f"t_{name}"] = float(t_wave[np.argmax(np.abs(t_wave))]) if len(t_wave) else np.nan
    return values


def rhythm(peaks: np.ndarray, kept: np.ndarray, classes: dict[str, np.ndarray]) -> dict[str, float]:
    """
    Rate and RR irregularity from method B's beats.

    Parameters
    ----------
    peaks : np.ndarray
        All R peaks.
    kept : np.ndarray
        Index into ``peaks`` of the complete beats.
    classes : dict[str, np.ndarray]
        Output of ``classify_beats`` for the complete beats.

    Returns
    -------
    dict[str, float]
        ``hr_bpm``, ``rr_cv``, ``rr_nrmssd``, ``rr_range_ms`` (all intervals) and ``rr_range_dominant_ms``
        (intervals between consecutive dominant, non-premature beats).
    """
    rr = np.diff(peaks) * 1000.0 / FS
    result = {
        "hr_bpm": 60000.0 / rr.mean(),
        "rr_cv": rr.std() / rr.mean(),
        "rr_nrmssd": float(np.sqrt(np.mean(np.diff(rr) ** 2)) / rr.mean()) if len(rr) > 1 else np.nan,
        "rr_range_ms": float(rr.max() - rr.min()),
    }
    clean = classes["dominant"] & ~classes["premature_aberrant"]
    pairs = [
        peaks[kept[i + 1]] - peaks[kept[i]]
        for i in range(len(kept) - 1)
        if kept[i + 1] == kept[i] + 1 and clean[i] and clean[i + 1]
    ]
    result["rr_range_dominant_ms"] = float(np.ptp(pairs) * 1000.0 / FS) if len(pairs) >= 2 else np.nan
    return result


def empty_result(status: str, flat: np.ndarray) -> dict[str, object]:
    """
    Build the result of a record that could not be delineated.

    Parameters
    ----------
    status : str
        Failure status.
    flat : np.ndarray
        Output of ``flat_leads``.

    Returns
    -------
    dict[str, object]
        ``status`` and ``flat_leads``; every measurement is absent.
    """
    return {"status": status, "flat_leads": float(flat.sum())}


def delineate(
    beats: np.ndarray,
    classes: dict[str, np.ndarray],
    leads: np.ndarray,
    rr_ms: float,
    fractions: tuple[float, float] = (QRS_FRACTION, P_FRACTION),
) -> dict[str, float]:
    """
    Every template measurement of the dominant beats.

    Parameters
    ----------
    beats : np.ndarray
        Complete beats ``[beats, 12, length]``.
    classes : dict[str, np.ndarray]
        Output of ``classify_beats``.
    leads : np.ndarray
        Non-flat independent leads.
    rr_ms : float
        Median RR interval of the record.
    fractions : tuple[float, float]
        QRS and P thresholds (``qrs_bounds``, ``p_wave``).

    Returns
    -------
    dict[str, float]
        QRS bounds and duration, axis, V1 morphology, P-wave measures and per-lead amplitudes.
    """
    dominant = beats[classes["dominant"]]
    template = np.median(dominant, axis=0)
    onset, offset, noise = qrs_bounds(template, leads, fractions[0])
    base = baseline(template, onset)
    centered = template - base[:, None]
    end, stop = t_end(centered, offset, rr_ms)
    values = {
        "qrs_onset_ms": (onset - PRE) * 1000.0 / FS,
        "qrs_offset_ms": (offset - PRE) * 1000.0 / FS,
        "qrs_ms": (offset - onset) * 1000.0 / FS,
        "velocity_noise": noise,
        "t_end_ms": (end - PRE) * 1000.0 / FS,
    }
    values |= lead_amplitudes(centered, onset, offset, offset + samples(60), (offset + samples(80), stop))
    values["axis_deg"] = float(np.degrees(np.arctan2(values["area_aVF"], values["area_I"])))
    values |= v1_morphology(centered[6, onset : offset + 1])
    values |= p_wave(dominant, template, base, p_window(onset, rr_ms, end - PRE), onset, fractions[1])
    return values


def measure_record(
    signal: np.ndarray, fractions: tuple[float, float] = (QRS_FRACTION, P_FRACTION)
) -> dict[str, object]:
    """
    Measure one record with method B.

    Parameters
    ----------
    signal : np.ndarray
        ``[12, samples]`` in mV at 500 Hz, NaN where missing.
    fractions : tuple[float, float]
        QRS and P thresholds; the defaults are the frozen values.

    Returns
    -------
    dict[str, object]
        ``status``, ``flat_leads``, beat counts, rhythm measures and, when a dominant beat exists, every
        ``delineate`` value.
    """
    signal = np.asarray(signal, dtype=np.float64)
    flat = flat_leads(signal)
    clean = bandpass(signal)
    peaks = detect_peaks(signal, clean, flat)
    leads = np.array([i for i in INDEPENDENT if not flat[i]])
    beats, kept = beat_stack(clean, peaks)
    if flat[list(DETECTION_LEADS)].all() or not len(leads):
        return empty_result("flat", flat)
    if len(beats) < 3:
        return empty_result("few_beats", flat)
    rr_all = np.diff(peaks) * 1000.0 / FS
    rr_before = np.array([rr_all[i - 1] if i > 0 else np.nan for i in kept])
    classes = classify_beats(beats, leads, rr_before)
    result: dict[str, object] = {
        "status": "ok",
        "flat_leads": float(flat.sum()),
        "n_beats": float(len(peaks)),
        "n_complete": float(len(beats)),
        "n_dominant": float(classes["dominant"].sum()),
        "n_premature_aberrant": float(classes["premature_aberrant"].sum()),
        "n_aberrant": float((classes["correlation"] < ABERRANT_CORRELATION).sum()),
    }
    result |= rhythm(peaks, kept, classes)
    if classes["dominant"].sum() < 2:
        result["status"] = "no_dominant"
        return result
    result |= delineate(beats, classes, leads, float(np.median(rr_all)), fractions)
    return result


def lead_vectors() -> np.ndarray:
    """
    Give the unit directions of the twelve leads in a frontal-horizontal heart-vector space.

    Returns
    -------
    np.ndarray
        ``[12, 3]``: columns are left (x), inferior (y) and anterior (z); limb leads lie in the frontal plane
        with Goldberger's gain for the augmented leads, precordial leads in the horizontal plane.
    """
    rows = []
    for name in LEADS:
        if name in LIMB_ANGLES:
            angle = np.radians(LIMB_ANGLES[name])
            rows.append([LIMB_GAIN[name] * np.cos(angle), LIMB_GAIN[name] * np.sin(angle), 0.0])
        else:
            angle = np.radians(PRECORDIAL_ANGLES[name])
            rows.append([1.5 * np.cos(angle), 0.0, 1.5 * np.sin(angle)])
    return np.asarray(rows)


def lobe(length: int, start: int, duration: int) -> np.ndarray:
    """
    Raised-cosine lobe of exact support.

    Parameters
    ----------
    length : int
        Output length in samples.
    start : int
        First sample of the lobe.
    duration : int
        Lobe length in samples.

    Returns
    -------
    np.ndarray
        Zero outside ``[start, start + duration)``, peak 1 at its middle.
    """
    shape = np.zeros(length)
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(duration) / max(duration - 1, 1))
    stop = min(start + duration, length)
    if stop > start >= 0:
        shape[start:stop] = window[: stop - start]
    return shape


def beat_components(spec: dict[str, float], ectopic: bool) -> list[tuple[float, float, float, np.ndarray]]:
    """
    Lobes of one synthetic beat: start and duration relative to the QRS onset (ms), and a heart vector.

    Parameters
    ----------
    spec : dict[str, float]
        ``qrs_ms``, ``pr_ms``, ``axis_deg``, ``rbbb`` (0 or 1), ``st_mv``, ``t_mv``, ``p_mv``.
    ectopic : bool
        A wide premature ventricular beat without P wave.

    Returns
    -------
    list[tuple[float, float, float, np.ndarray]]
        ``(start_ms, duration_ms, kind, vector)``; kind 0 is a lobe, 1 a plateau (ST level).
    """
    axis = np.radians(spec["axis_deg"])
    main = np.array([np.cos(axis), np.sin(axis), -0.5]) * 1.2
    if ectopic:
        return [
            (0.0, 150.0, 0, np.array([-0.9, -0.6, 0.8]) * 1.4),
            (230.0, 200.0, 0, np.array([0.5, 0.4, -0.3])),
        ]
    qrs = spec["qrs_ms"]
    terminal = 50.0 if spec["rbbb"] else 0.0
    body = qrs - terminal
    parts = [
        (-spec["pr_ms"], 100.0, 0, np.array([0.5, 0.85, 0.1]) * spec["p_mv"]),
        (0.0, 0.25 * body, 0, np.array([-0.15, 0.0, 0.3])),
        (0.2 * body, 0.6 * body, 0, main),
        (0.75 * body, 0.25 * body, 0, np.array([-0.2, -0.2, -0.3])),
    ]
    if spec["rbbb"]:
        parts.append((body, terminal, 0, np.array([-0.5, 0.0, 0.9]) * 0.8))
    t_vector = np.array([np.cos(axis), np.sin(axis), -0.2])
    t_vector = t_vector / np.linalg.norm(t_vector[:2]) * spec["t_mv"]
    parts.append((qrs, 100.0 + 160.0, 1, np.array([0.0, 1.0, 0.0]) * spec["st_mv"]))
    parts.append((qrs + 100.0, 200.0, 0, t_vector))
    return parts


def synthetic_beats(
    spec: dict[str, float], onsets: np.ndarray, ectopic: np.ndarray, length: int
) -> np.ndarray:
    """
    Heart vector of a sequence of synthetic beats.

    Parameters
    ----------
    spec : dict[str, float]
        As in ``beat_components``.
    onsets : np.ndarray
        QRS onset sample of each beat.
    ectopic : np.ndarray
        Boolean per beat.
    length : int
        Record length in samples.

    Returns
    -------
    np.ndarray
        ``[3, length]`` heart vector in mV.
    """
    vector = np.zeros((3, length))
    for onset, is_ectopic in zip(onsets, ectopic, strict=True):
        for start_ms, duration_ms, kind, direction in beat_components(spec, bool(is_ectopic)):
            start = onset + samples(start_ms)
            shape = lobe(length, start, samples(duration_ms))
            if kind == 1:
                shape = np.zeros(length)
                shape[max(start, 0) : min(start + samples(duration_ms), length)] = 1.0
                shape = uniform_filter1d(shape, size=samples(40))
            vector += direction[:, None] * shape[None, :]
    return vector


def synthetic_onsets(
    spec: dict[str, float], rng: np.random.Generator, length: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    QRS onsets of a synthetic rhythm and which beats are ectopic.

    Parameters
    ----------
    spec : dict[str, float]
        ``rate_bpm``, ``af`` (0 or 1), ``sinus_arrhythmia_ms`` (peak RR modulation), ``pvc_every`` (0 for
        none).
    rng : np.random.Generator
        Generator for AF intervals.
    length : int
        Record length in samples.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Onset samples and the ectopic mask.
    """
    rr = 60000.0 / spec["rate_bpm"]
    onsets, ectopic, time, beat = [], [], 300.0, 0
    while samples(time) < length:
        is_ectopic = spec["pvc_every"] > 0 and beat % spec["pvc_every"] == spec["pvc_every"] - 1
        onsets.append(samples(time))
        ectopic.append(is_ectopic)
        step = rr + spec["sinus_arrhythmia_ms"] * np.sin(2 * np.pi * time / 4000.0)
        if spec["af"]:
            step = rr * rng.uniform(0.55, 1.45)
        nxt = spec["pvc_every"] > 0 and (beat + 1) % spec["pvc_every"] == spec["pvc_every"] - 1
        time += 0.65 * step if nxt else (1.35 * step if is_ectopic else step)
        beat += 1
    return np.asarray(onsets), np.asarray(ectopic, dtype=bool)


SYNTHETIC_DEFAULTS = {
    "rate_bpm": 70.0,
    "pr_ms": 160.0,
    "qrs_ms": 90.0,
    "axis_deg": 45.0,
    "rbbb": 0.0,
    "st_mv": 0.0,
    "t_mv": 0.3,
    "p_mv": 0.15,
    "af": 0.0,
    "sinus_arrhythmia_ms": 0.0,
    "pvc_every": 0.0,
    "noise_mv": 0.0,
    "wander_mv": 0.0,
    "mains_mv": 0.0,
}


def synthetic_record(seed: int = 0, seconds: float = 10.0, **settings: float) -> tuple[np.ndarray, dict]:
    """
    Build a synthetic twelve-lead record with known measurements.

    Parameters
    ----------
    seed : int
        Seed of the noise and of AF intervals.
    seconds : float
        Duration.
    **settings : float
        Overrides of ``SYNTHETIC_DEFAULTS``: rate, PR, QRS duration (ms), frontal axis (degrees), ``rbbb``
        (terminal right-anterior lobe of 50 ms inside the QRS), ST plateau along the inferior axis (mV), T and
        P amplitudes, ``af`` (no P wave, irregular RR, 5-7 Hz atrial waves of 0.05 mV), sinus-arrhythmia RR
        modulation, a premature wide beat every ``pvc_every`` beats, white noise, 0.3 Hz wander and 50 Hz
        mains amplitude.

    Returns
    -------
    tuple[np.ndarray, dict]
        ``[12, samples]`` in mV and the truth: the settings, ``axis_net_deg`` (frontal angle of the net QRS
        vector area), ``onsets`` and ``ectopic``.

    Raises
    ------
    ValueError
        If a setting is unknown.
    """
    unknown = set(settings) - set(SYNTHETIC_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown synthetic settings {sorted(unknown)}")
    spec = {**SYNTHETIC_DEFAULTS, **settings}
    if spec["af"]:
        spec["p_mv"] = 0.0
    rng = np.random.default_rng(seed)
    length = int(seconds * FS)
    onsets, ectopic = synthetic_onsets(spec, rng, length)
    vector = synthetic_beats(spec, onsets, ectopic, length)
    time = np.arange(length) / FS
    if spec["af"]:
        atrial = sum(np.sin(2 * np.pi * f * time + rng.uniform(0, 2 * np.pi)) for f in (5.1, 6.3, 7.0)) / 3
        vector += np.array([0.3, 0.5, 0.6])[:, None] * 0.1 * atrial[None, :]
    signal = lead_vectors() @ vector
    signal += spec["wander_mv"] * np.sin(2 * np.pi * 0.3 * time)[None, :]
    signal += spec["mains_mv"] * np.sin(2 * np.pi * 50.0 * time)[None, :]
    signal += rng.normal(0.0, spec["noise_mv"], signal.shape) if spec["noise_mv"] > 0 else 0.0
    one = synthetic_beats(
        {**spec, "st_mv": 0.0}, np.array([samples(300)]), np.array([False]), samples(300 + spec["qrs_ms"]) + 1
    )
    net = one[:2, samples(300) :].sum(axis=1)
    truth = {
        **spec,
        "axis_net_deg": float(np.degrees(np.arctan2(net[1], net[0]))),
        "onsets": onsets,
        "ectopic": ectopic,
    }
    return signal, truth


def ludb_waves(record_path: str, lead: str) -> list[tuple[str, int, int, int]]:
    """
    Manually delineated waves of one LUDB lead.

    Parameters
    ----------
    record_path : str
        LUDB record path without extension.
    lead : str
        Annotation extension, the lower-case lead name (``i``, ``ii``, ..., ``v6``).

    Returns
    -------
    list[tuple[str, int, int, int]]
        ``(wave, onset, peak, offset)`` samples, wave ``p``, ``N`` or ``t``, for every complete
        ``(`` wave ``)`` triplet.
    """
    from wfdb import rdann

    annotation = rdann(record_path, lead)
    symbols, positions = annotation.symbol, annotation.sample
    return [
        (symbols[i], int(positions[i - 1]), int(positions[i]), int(positions[i + 1]))
        for i in range(1, len(symbols) - 1)
        if symbols[i] in ("p", "N", "t") and symbols[i - 1] == "(" and symbols[i + 1] == ")"
    ]


def global_waves(waves: list[list[tuple[str, int, int, int]]], kind: str) -> np.ndarray:
    """
    Global onset and offset of one wave type, joining the leads' annotations of the same beat.

    Annotations of different leads belong to the same wave when their peaks lie within 100 ms of the first
    lead's peak; the global onset is the earliest lead onset and the offset the latest lead offset.

    Parameters
    ----------
    waves : list[list[tuple[str, int, int, int]]]
        ``ludb_waves`` of each lead.
    kind : str
        ``p``, ``N`` or ``t``.

    Returns
    -------
    np.ndarray
        ``[waves, 3]``: onset, peak (median over leads) and offset samples, ordered by peak.
    """
    found = sorted(
        (peak, onset, offset) for lead in waves for wave, onset, peak, offset in lead if wave == kind
    )
    groups: list[list[tuple[int, int, int]]] = []
    for item in found:
        if groups and item[0] - groups[-1][0][0] <= samples(100):
            groups[-1].append(item)
        else:
            groups.append([item])
    return np.array(
        [
            [min(g[1] for g in group), int(np.median([g[0] for g in group])), max(g[2] for g in group)]
            for group in groups
        ],
        dtype=int,
    ).reshape(-1, 3)


def ludb_reference(record_path: str) -> dict[str, float]:
    """
    Record-level reference measurements from LUDB's manual delineation.

    Parameters
    ----------
    record_path : str
        LUDB record path without extension.

    Returns
    -------
    dict[str, float]
        Medians over annotated beats of the global QRS duration and PR interval (QRS onset minus the onset
        of the P wave that ends before it, within 400 ms), the rate from annotated QRS peaks, and counts.
    """
    waves = [ludb_waves(record_path, lead.lower()) for lead in LEADS]
    qrs, p = global_waves(waves, "N"), global_waves(waves, "p")
    pr = []
    for onset, _, _ in qrs:
        before = p[(p[:, 2] <= onset) & (onset - p[:, 0] <= samples(400))]
        if len(before):
            pr.append((onset - before[-1, 0]) * 1000.0 / FS)
    rr = np.diff(qrs[:, 1]) * 1000.0 / FS
    return {
        "qrs_ms": float(np.median(qrs[:, 2] - qrs[:, 0]) * 1000.0 / FS) if len(qrs) else np.nan,
        "pr_ms": float(np.median(pr)) if pr else np.nan,
        "hr_bpm": 60000.0 / float(rr.mean()) if len(rr) else np.nan,
        "n_qrs": float(len(qrs)),
        "n_p": float(len(p)),
        "n_pr": float(len(pr)),
    }


def synthetic_cases(count: int, seed: int) -> list[dict[str, float]]:
    """
    Random synthetic settings for measurement validation, plus fixed AF, ectopy and sinus-arrhythmia sets.

    Parameters
    ----------
    count : int
        Random cases; a further ``count // 4`` cases each of AF, a premature wide beat every fourth beat, and
        sinus arrhythmia follow.
    seed : int
        Generator seed.

    Returns
    -------
    list[dict[str, float]]
        Settings for ``synthetic_record``, each with a ``seed`` entry.
    """
    rng = np.random.default_rng(seed)
    cases = []
    for index in range(count + 3 * (count // 4)):
        rate = float(rng.uniform(40, 140))
        rbbb = float(rng.random() < 0.3)
        case = {
            "seed": index,
            "rate_bpm": rate,
            "pr_ms": float(rng.uniform(110, min(320.0, 60000.0 / rate - 250))),
            "qrs_ms": float(rng.uniform(110 if rbbb else 70, 170 if rbbb else 130)),
            "rbbb": rbbb,
            "axis_deg": float(rng.uniform(-90, 120)),
            "st_mv": float(rng.uniform(-0.2, 0.1)),
            "t_mv": float(rng.uniform(-0.4, 0.5)),
            "noise_mv": float(rng.choice([0.0, 0.01, 0.03])),
            "wander_mv": float(rng.choice([0.0, 0.1, 0.3])),
            "mains_mv": float(rng.choice([0.0, 0.02])),
        }
        kind = (index - count) // (count // 4) if index >= count else -1
        case |= [{}, {"af": 1.0}, {"pvc_every": 4.0}, {"sinus_arrhythmia_ms": float(rng.uniform(80, 200))}][
            kind + 1
        ]
        cases.append(case)
    return cases


def synthetic_levels(truth: dict) -> dict[str, float]:
    """
    Exact ST level 60 ms after the QRS and T-peak amplitude of a synthetic record in leads II and V5.

    Parameters
    ----------
    truth : dict
        Truth of ``synthetic_record``.

    Returns
    -------
    dict[str, float]
        ``st60_II``, ``st60_V5``, ``t_II`` and ``t_V5`` of the noise-free beat (ST plateau included in T).
    """
    spec = {key: value for key, value in truth.items() if key in SYNTHETIC_DEFAULTS}
    start = samples(400)
    length = start + samples(spec["qrs_ms"] + 500)
    vector = synthetic_beats(spec, np.array([start]), np.array([False]), length)
    leads = lead_vectors() @ vector
    j60 = start + samples(spec["qrs_ms"] + 60)
    t_peak = start + samples(spec["qrs_ms"] + 200)
    return {
        f"{kind}_{lead}": float(leads[LEADS.index(lead), at])
        for kind, at in (("st60", j60), ("t", t_peak))
        for lead in ("II", "V5")
    }
