"""Label-free physiological wave supports and independent boundary evaluation."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_closing, gaussian_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.signal import butter, sosfiltfilt

from ecg_experiment.fragment_localization import r_peaks

WAVE_NAMES = ("P", "QRS", "T")
FIXED_OFFSETS = np.array([[-0.25, -0.06], [-0.06, 0.08], [0.20, 0.45]])


def fixed_boundaries(peaks: np.ndarray, leads: int = 12, fs: int = 500) -> np.ndarray:
    """
    Reproduce the physiological-named fixed offsets used by Experiment 042.

    Parameters
    ----------
    peaks : np.ndarray
        Detected raw R anchor samples.
    leads : int
        Number of waveform leads.
    fs : int
        Sampling frequency.

    Returns
    -------
    np.ndarray
        Absolute sample boundaries, shape (anchors, leads, three waves, two endpoints).
    """
    offsets = np.rint(FIXED_OFFSETS * fs).astype(int)
    return np.broadcast_to(peaks[:, None, None, None] + offsets[None, None], (len(peaks), leads, 3, 2)).copy()


def qrs_support(envelope: np.ndarray, peak: int, fs: int) -> tuple[int, int]:
    """
    Find adaptive derivative-energy support around a raw R anchor.

    Parameters
    ----------
    envelope : np.ndarray
        Smoothed absolute derivative of one lead.
    peak : int
        Raw common R anchor.
    fs : int
        Sampling frequency.

    Returns
    -------
    tuple[int, int]
        QRS support boundaries in absolute samples.
    """
    low, high = max(0, peak - round(0.12 * fs)), min(len(envelope), peak + round(0.16 * fs))
    local_low, local_high = max(low, peak - round(0.06 * fs)), min(high, peak + round(0.06 * fs))
    maximum = local_low + int(envelope[local_low:local_high].argmax())
    above = envelope[low:high] > 0.12 * envelope[maximum]
    above = binary_closing(above, structure=np.ones(round(0.01 * fs) + 1))
    relative = maximum - low
    above[relative] = True
    left = np.flatnonzero(~above[:relative])
    right = np.flatnonzero(~above[relative + 1 :])
    start = low + (int(left[-1]) + 1 if len(left) else 0)
    end = maximum + 1 + (int(right[0]) if len(right) else len(above) - relative - 1)
    start, end = max(low, start - round(0.008 * fs)), min(high, end + round(0.008 * fs))
    if end - start < round(0.02 * fs):
        start, end = max(low, maximum - round(0.01 * fs)), min(high, maximum + round(0.01 * fs))
    return start, end


def wave_support(
    signal: np.ndarray, low: int, high: int, noise: float, radius: float, fs: int
) -> tuple[int, int]:
    """
    Find a polarity-independent smooth P or T support without labeled boundaries.

    Parameters
    ----------
    signal : np.ndarray
        Low-pass filtered lead samples.
    low, high : int
        Frozen search limits in absolute samples.
    noise : float
        Local PR robust noise estimate.
    radius : float
        Maximum distance from the selected amplitude peak in seconds.
    fs : int
        Sampling frequency.

    Returns
    -------
    tuple[int, int]
        Absolute support, or (-1,-1) when the slot is rejected.
    """
    low, high = max(0, low), min(len(signal), high)
    if high - low < round(0.02 * fs):
        return -1, -1
    chunk = signal[low:high]
    edge = min(round(0.01 * fs), len(chunk) // 2)
    line = np.linspace(np.median(chunk[:edge]), np.median(chunk[-edge:]), len(chunk))
    amplitude = gaussian_filter1d(np.abs(chunk - line), sigma=0.008 * fs)
    maximum = int(amplitude.argmax())
    peak_height = float(amplitude[maximum])
    if peak_height <= 3 * noise or peak_height <= np.finfo(float).eps:
        return -1, -1
    start = max(0, maximum - round(radius * fs))
    end = min(len(chunk), maximum + round(radius * fs) + 1)
    selected = np.flatnonzero(amplitude[start:end] >= max(0.15 * peak_height, 2 * noise)) + start
    if not len(selected):
        return -1, -1
    return max(low, low + int(selected[0]) - round(0.006 * fs)), min(
        high, low + int(selected[-1]) + round(0.006 * fs)
    )


def adaptive_boundaries(signal: np.ndarray, fs: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """
    Infer waveform-only physiological supports with the prospectively frozen recipe.

    Parameters
    ----------
    signal : np.ndarray
        Raw canonical leads by samples, in millivolts.
    fs : int
        Sampling frequency.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Raw R anchors and per-anchor/lead/P-QRS-T boundary pairs.
    """
    peaks = r_peaks(signal, fs)
    filtered = sosfiltfilt(butter(2, [0.5, 30], btype="bandpass", fs=fs, output="sos"), signal, axis=1)
    smooth = sosfiltfilt(butter(2, 12, btype="lowpass", fs=fs, output="sos"), filtered, axis=1)
    derivative = gaussian_filter1d(np.abs(np.gradient(filtered, axis=1)), sigma=0.006 * fs, axis=1)
    output = np.full((len(peaks), len(signal), 3, 2), -1, dtype=np.int64)
    for index, peak in enumerate(peaks):
        before = int(peaks[index - 1]) if index else -fs
        after = int(peaks[index + 1]) if index + 1 < len(peaks) else len(signal[0]) + fs
        adjacent = np.diff(peaks[max(0, index - 1) : min(len(peaks), index + 2)])
        rr = float(np.median(adjacent)) / fs if len(adjacent) else 1.0
        for lead in range(len(signal)):
            onset, offset = qrs_support(derivative[lead], int(peak), fs)
            output[index, lead, 1] = onset, offset
            baseline = smooth[lead, max(0, peak - round(0.10 * fs)) : max(0, peak - round(0.06 * fs))]
            noise = (
                1.4826 * float(np.median(np.abs(baseline - np.median(baseline)))) if len(baseline) else 0.0
            )
            output[index, lead, 0] = wave_support(
                smooth[lead],
                max(peak - round(0.35 * fs), before + round(0.2 * fs)),
                min(peak - round(0.08 * fs), onset - round(0.03 * fs)),
                noise,
                0.1,
                fs,
            )
            output[index, lead, 2] = wave_support(
                smooth[lead],
                offset + round(0.04 * fs),
                min(peak + round(0.60 * fs), after - round(0.15 * fs), peak + round(0.65 * rr * fs)),
                noise,
                0.18,
                fs,
            )
    return peaks, output


def annotation_triplets(samples: np.ndarray, symbols: list[str]) -> tuple[dict[str, np.ndarray], list[dict]]:
    """
    Parse complete WFDB onset/peak/offset triplets with explicit irregular-event audit.

    Parameters
    ----------
    samples : np.ndarray
        Sorted annotation sample positions.
    symbols : list[str]
        Corresponding WFDB annotation symbols.

    Returns
    -------
    tuple[dict[str, np.ndarray], list[dict]]
        Complete P/QRS/T onset-peak-offset arrays and irregular sequence records.
    """
    names = {"p": "P", "N": "QRS", "t": "T"}
    found: dict[str, list] = {name: [] for name in WAVE_NAMES}
    irregular = []
    index = 0
    while index < len(samples):
        group = symbols[index : index + 3]
        typed = [symbol for symbol in group if symbol in names]
        complete = len(group) == 3 and group.count("(") == 1 and group.count(")") == 1 and len(typed) == 1
        if not complete:
            irregular.append(
                {
                    "index": index,
                    "symbol": symbols[index],
                    "sample": int(samples[index]),
                    "reason": "unknown_partial_annotation",
                }
            )
            index += 1
            continue
        onset = int(samples[index + group.index("(")])
        peak = int(samples[index + group.index(typed[0])])
        offset = int(samples[index + group.index(")")])
        outside = max(onset - peak, peak - offset, 0)
        if offset <= onset or outside > 5:
            irregular.append(
                {
                    "index": index,
                    "symbol": symbols[index],
                    "sample": int(samples[index]),
                    "reason": "unknown_invalid_or_distant_peak",
                }
            )
            index += 1
            continue
        found[names[typed[0]]].append([onset, peak, offset])
        if group != ["(", typed[0], ")"]:
            irregular.append(
                {
                    "index": index,
                    "reason": "recovered_outside_peak",
                    "wave": names[typed[0]],
                    "samples": samples[index : index + 3].astype(int).tolist(),
                    "symbols": group,
                    "outside_samples": outside,
                }
            )
        index += 3
    return {name: np.array(rows, dtype=int).reshape(-1, 3) for name, rows in found.items()}, irregular


def match_peaks(peaks: np.ndarray, qrs: np.ndarray, fs: int = 500) -> dict[int, int]:
    """
    Match annotated QRS peaks to common raw anchors one to one, independently of scores.

    Parameters
    ----------
    peaks : np.ndarray
        Raw detected R anchor samples.
    qrs : np.ndarray
        Annotated QRS onset/peak/offset rows.
    fs : int
        Sampling frequency.

    Returns
    -------
    dict[int, int]
        Annotated QRS row to predicted-anchor row within the fixed 150 ms tolerance.
    """
    if not len(peaks) or not len(qrs):
        return {}
    distance = np.abs(qrs[:, 1, None] - peaks[None, :])
    # Forbidden pairs receive a large cost so valid assignment cardinality wins first.
    cost = np.where(distance <= round(0.15 * fs), distance, 1_000_000)
    truth, predicted = linear_sum_assignment(cost)
    return {
        int(a): int(b) for a, b in zip(truth, predicted, strict=True) if distance[a, b] <= round(0.15 * fs)
    }


def associate_waves(waves: np.ndarray, qrs: np.ndarray, name: str, fs: int = 500) -> dict[int, int]:
    """
    Associate true P/T waves with annotated QRS beats without reusing prediction slots.

    Parameters
    ----------
    waves : np.ndarray
        True onset/peak/offset rows of one wave class.
    qrs : np.ndarray
        True QRS onset/peak/offset rows.
    name : str
        P or T wave class.
    fs : int
        Sampling frequency.

    Returns
    -------
    dict[int, int]
        Wave row to a unique compatible annotated QRS row.
    """
    if name == "QRS":
        return {i: i for i in range(len(qrs))}
    if not len(waves) or not len(qrs):
        return {}
    difference = (
        qrs[:, 1][None, :] - waves[:, 1, None] if name == "P" else waves[:, 1, None] - qrs[:, 1][None, :]
    )
    maximum = round((0.45 if name == "P" else 0.65) * fs)
    valid = (difference > 0) & (difference <= maximum)
    cost = np.where(valid, difference, 1_000_000)
    truth, beats = linear_sum_assignment(cost)
    return {int(a): int(b) for a, b in zip(truth, beats, strict=True) if valid[a, b]}


def interval_measurements(predicted: np.ndarray, truth: np.ndarray, fs: int = 500) -> dict[str, float]:
    """
    Score a truth interval, preserving missing predictions with zero overlap.

    Parameters
    ----------
    predicted : np.ndarray
        Proposed onset and offset, (-1,-1) when absent.
    truth : np.ndarray
        True onset and offset.
    fs : int
        Sampling frequency.

    Returns
    -------
    dict[str, float]
        IoU, absolute boundary errors and 30 ms endpoint coverage.
    """
    if np.any(predicted < 0) or predicted[1] <= predicted[0]:
        return {"iou": 0.0, "onset_mae_ms": 500.0, "offset_mae_ms": 500.0, "within_30ms": 0.0, "missing": 1.0}
    intersection = max(0.0, float(min(predicted[1], truth[1]) - max(predicted[0], truth[0])))
    union = max(predicted[1], truth[1]) - min(predicted[0], truth[0])
    errors = np.abs(predicted - truth) * 1000 / fs
    return {
        "iou": intersection / union,
        "onset_mae_ms": float(errors[0]),
        "offset_mae_ms": float(errors[1]),
        "within_30ms": float(np.mean(errors <= 30)),
        "missing": 0.0,
    }
