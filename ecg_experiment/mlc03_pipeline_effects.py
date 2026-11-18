"""
MLC-03 Part A item 7: what the MLC-02 input pipelines do to real defective records.

Each pipeline is the one its candidate used (``docs/mlc02-inputs-results-2026-11-10.md``), reused by import:

- ``xecg``: missing samples set to the stored rail value (``mlc02_features.model_input``), FFT resampling of
  the whole 10 s to 100 Hz, raw mV (``external_encoders.xecg_input``); also the input of F.
- ``jepa``: the same rail value, leads I, II and V1-V6 only, FFT resampling to 250 Hz, raw mV
  (``external_encoders.jepa_input``).
- ``cpc``: the same rail value, four 2.5 s crops each resampled to 240 Hz with ``resampy``, raw mV
  (``mlc02_cpc.cpc_input``); the crops are joined in time here.
- ``scratch``: the same rail value, one polyphase pass to 100 Hz (``mlc02_scratch_network.to_model_rate``),
  then one training mean and standard deviation (``standardize``); defects are measured before the
  standardization, which is a single affine map.

For one lead of one record, ``defect_measures`` computes the same five measures on the 500 Hz window and on
every pipeline input, so their ratio says how much of a defect reaches the model.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, periodogram, sosfiltfilt

from ecg_experiment.eda.signals import LEADS
from ecg_experiment.external_encoders import JEPA_LEADS, jepa_input, xecg_input
from ecg_experiment.mlc02_cpc import cpc_input
from ecg_experiment.mlc02_features import model_input
from ecg_experiment.mlc02_scratch_network import to_model_rate
from ecg_experiment.mlc03_signal_quality import FS, fill_missing
from ecg_experiment.resample import resample_full

PIPELINES = ("xecg", "jepa", "cpc", "scratch")
RATES = {"window": FS, "xecg": 100, "jepa": 250, "cpc": 240, "scratch": 100}
EDGE_S = 0.2


def pipeline_inputs(window: np.ndarray) -> dict[str, tuple[np.ndarray, list[str]]]:
    """
    Model inputs of one cached window, as lead-major arrays in mV with their lead names.

    Parameters
    ----------
    window : np.ndarray
        Float ``(12, 5000)`` mV with NaN for missing samples.

    Returns
    -------
    dict[str, tuple[np.ndarray, list[str]]]
        Per pipeline of ``PIPELINES``: the input (``(leads, samples)`` float64) and its lead names.
    """
    stored = model_input(np.asarray(window, dtype=np.float32))
    cpc = cpc_input(stored)
    return {"xecg": (xecg_input(stored).T.astype(np.float64), list(LEADS)),
            "jepa": (jepa_input(stored).astype(np.float64), [LEADS[index] for index in JEPA_LEADS]),
            "cpc": (np.concatenate(list(cpc), axis=1).astype(np.float64), list(LEADS)),
            "scratch": (to_model_rate(stored[None])[0].astype(np.float64), list(LEADS))}


def mains_amplitude(lead: np.ndarray, rate: int, frequency: float) -> float:
    """
    Amplitude of a sinusoid at one frequency above the neighbouring spectrum.

    At or within 3 Hz of the Nyquist frequency (50 Hz at 100 Hz), the amplitude is instead the square root
    of twice the power between 47 Hz and Nyquist, since the neighbouring band is cut off there.

    Parameters
    ----------
    lead : np.ndarray
        One lead in mV.
    rate : int
        Sampling rate in Hz.
    frequency : float
        Frequency in Hz.

    Returns
    -------
    float
        Amplitude in mV; 0 for a frequency above Nyquist.
    """
    if frequency > rate / 2:
        return 0.0
    frequencies, density = periodogram(lead, fs=rate, window="hann", detrend="linear")
    step = frequencies[1] - frequencies[0]
    if frequency + 3.0 >= rate / 2:
        return float(np.sqrt(2.0 * density[frequencies >= rate / 2 - 3.0].sum() * step))
    distance = np.abs(frequencies - frequency)
    background = np.median(density[(distance >= 1.0) & (distance <= 5.0)])
    excess = np.clip(density[distance <= 0.3] - background, 0.0, None).sum() * step
    return float(np.sqrt(2.0 * excess))


def high_frequency_rms(lead: np.ndarray, rate: int) -> float:
    """
    RMS of the content above 40 Hz (0 when the rate cannot hold it).

    Parameters
    ----------
    lead : np.ndarray
        One lead in mV.
    rate : int
        Sampling rate in Hz.

    Returns
    -------
    float
        RMS in mV.
    """
    if rate / 2 <= 41.0:
        return 0.0
    high = sosfiltfilt(butter(4, 40.0, btype="high", fs=rate, output="sos"), lead)
    return float(np.sqrt(np.mean(high**2)))


def baseline_range(lead: np.ndarray, rate: int) -> float:
    """
    Peak-to-peak range of the content below 0.5 Hz.

    Parameters
    ----------
    lead : np.ndarray
        One lead in mV.
    rate : int
        Sampling rate in Hz.

    Returns
    -------
    float
        Range in mV.
    """
    return float(np.ptp(sosfiltfilt(butter(2, 0.5, btype="low", fs=rate, output="sos"), lead)))


def defect_measures(lead: np.ndarray, rate: int) -> dict[str, float]:
    """
    Compute the five measures compared between the window and the pipeline inputs.

    Parameters
    ----------
    lead : np.ndarray
        One lead in mV.
    rate : int
        Sampling rate in Hz.

    Returns
    -------
    dict[str, float]
        ``peak_abs_mv`` (largest absolute value), ``range_mv`` (1-99 percentile range), ``mains50_mv``,
        ``hf_rms_mv`` and ``baseline_mv``.
    """
    low, high = np.percentile(lead, [1, 99])
    return {"peak_abs_mv": float(np.abs(lead).max()), "range_mv": float(high - low),
            "mains50_mv": mains_amplitude(lead, rate, 50.0), "hf_rms_mv": high_frequency_rms(lead, rate),
            "baseline_mv": baseline_range(lead, rate)}


def missing_spread(window: np.ndarray, lead: str) -> dict[str, float]:
    """
    How far the rail value given to missing samples spreads in each pipeline input.

    Each pipeline is applied twice, once as it was (rail value) and once with the missing samples
    interpolated; the share of input samples of the lead that differ by more than 0.05 mV measures the
    reach of the missing samples.

    Parameters
    ----------
    window : np.ndarray
        Float ``(12, 5000)`` mV with NaN.
    lead : str
        Lead with missing samples.

    Returns
    -------
    dict[str, float]
        ``<pipeline>_share`` and ``<pipeline>_max_mv`` per pipeline, and ``window_share`` (missing share of
        the lead at 500 Hz).
    """
    as_run = pipeline_inputs(window)
    repaired = pipeline_inputs(fill_missing(window))
    result = {"window_share": float(np.isnan(window[LEADS.index(lead)]).mean())}
    for name in PIPELINES:
        signal, names = as_run[name]
        if lead not in names:
            result[f"{name}_share"], result[f"{name}_max_mv"] = np.nan, np.nan
            continue
        difference = np.abs(signal[names.index(lead)] - repaired[name][0][names.index(lead)])
        result[f"{name}_share"] = float((difference > 0.05).mean())
        result[f"{name}_max_mv"] = float(difference.max())
    return result


def edge_ringing(window: np.ndarray) -> dict[str, float]:
    """
    Ringing at the record edges from FFT resampling, against one polyphase pass at the same rate.

    FFT resampling treats the 10 s as periodic, so a difference between the first and last samples (such as
    PTB-XL's held tail) rings near both edges. The measure is the largest difference from the polyphase
    output over all leads in the first and last 0.2 s, and in the interior.

    Parameters
    ----------
    window : np.ndarray
        Float ``(12, 5000)`` mV.

    Returns
    -------
    dict[str, float]
        ``xecg_edge_mv``, ``xecg_interior_mv``, ``jepa_edge_mv`` and ``jepa_interior_mv``.
    """
    inputs = pipeline_inputs(window)
    stored = model_input(np.asarray(window, dtype=np.float32)).astype(np.float64)
    result = {}
    for name, down, up in (("xecg", 5, 1), ("jepa", 2, 1)):
        signal, names = inputs[name]
        reference = resample_full(stored[[LEADS.index(lead) for lead in names]], up, down).astype(np.float64)
        difference = np.abs(signal - reference)
        edge = int(EDGE_S * RATES[name])
        result[f"{name}_edge_mv"] = float(max(difference[:, :edge].max(), difference[:, -edge:].max()))
        result[f"{name}_interior_mv"] = float(difference[:, edge:-edge].max())
    return result
