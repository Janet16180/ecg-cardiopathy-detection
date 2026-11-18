"""
MLC-03 Part A flags: written definitions, thresholds and reasons, computed from the measures table.

Every flag is a function of the columns of ``mlc03_signal_quality.measure_window`` (and, for missing
samples, of nothing else), so the frozen code applies to any role without labels. Thresholds were set on the
train and validation records only (``docs/mlc03-signal-quality-results-2026-11-11.md`` gives the
distributions behind each reason). Leads that are constant over the whole 10 s (``flat``) are left out of
every per-lead flag except ``flat_lead``, so that one dead lead raises one flag.

``FLAGS`` holds, per flag: the group of the protocol (Part A item), the definition, the threshold, the reason
for the threshold, and the leads it can blame (used for the proposed rule "mark lead unusable").
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ecg_experiment.eda.signals import LEADS
from ecg_experiment.mlc03_signal_quality import FS, INDEPENDENT

EDGE_HOLD_MS = 150.0
RAIL_SAMPLES = 1
NEAR_FLAT_MV = 0.05
IDENTITY_MV = 0.05
REVERSAL_P_MV = 0.05
NEUTRAL_MV = 0.06
NEUTRAL_RATIO = 0.10
AMPLITUDE_REFERENCE_MV = 0.93
AMPLITUDE_FACTOR = 4.0
BASELINE_MV = 2.0
MAINS_MV = 0.05
HF_NOISE_MV = 0.05
STEP_MV = 1.0
ISOLATED_SPIKE_MV = 2.0
DROPOUT_BEATS = 3
INTERIOR_HOLD_MS = 500.0
AGREEMENT = 0.8
MIN_BEATS = 3
SPECTRAL_SPIKE_MV = 2.0
REVERSAL_P_FLOOR_MV = -0.3
REVERSAL_QRS_MS = 120.0
REVERSAL_SIMILARITY = 0.8

FLAGS: dict[str, dict[str, str]] = {
    "missing_samples": {
        "item": "1 integrity",
        "definition": "At least one sample of the first 10 s that WFDB reads as missing (stored -32768).",
        "threshold": "more than 0 samples",
        "reason": "Definitional: the value is the format-16 code for an invalid sample, not a measurement.",
    },
    "edge_hold": {
        "item": "1 integrity",
        "definition": "Every non-constant lead holds its first value (or its last value) together for at "
                      "least the threshold at the start (or end) of the 10 s.",
        "threshold": f"at least {EDGE_HOLD_MS:.0f} ms",
        "reason": "PTB-XL holds the last 100-130 ms of every record (all 19,601 records), a source "
                  "convention; in the other sources the 99th percentile of the all-lead hold is 8 ms or "
                  "less. 150 ms lies above both, so the flag marks holds beyond the PTB-XL convention.",
    },
    "adc_rail": {
        "item": "1 integrity",
        "definition": "At least one sample at the 16-bit converter limit (absolute value of at least "
                      "32.7 mV) in the first 10 s.",
        "threshold": "at least 1 sample",
        "reason": "Definitional: every source stores 1,000 units per mV in 16 bits, so +/-32.767 mV is the "
                  "storage limit and a sample there is clipped. Plateaus at a lead's own extreme below the "
                  "limit were examined and were smooth peaks under coarse quantisation, not clipping.",
    },
    "flat_lead": {
        "item": "2 lead content",
        "definition": "A lead that is constant over the first 10 s.",
        "threshold": "range 0",
        "reason": "Definitional; reproduces the MLC-01 lead flags.",
    },
    "near_flat_lead": {
        "item": "2 lead content",
        "definition": "A non-constant lead whose 1-99 percentile range of the 0.5-40 Hz signal is below the "
                      "threshold.",
        "threshold": f"below {NEAR_FLAT_MV} mV",
        "reason": "The 0.01st percentile of this range is 0.07 mV or more for every lead over the train and "
                  "validation records; a lead below 0.05 mV carries noise, not an ECG.",
    },
    "duplicate_lead": {
        "item": "2 lead content",
        "definition": "Two non-constant leads that are identical sample for sample, or identical after a "
                      "sign flip, where neither I nor II is constant (a constant I or II makes derived limb "
                      "leads coincide by the identities).",
        "threshold": "any such pair",
        "reason": "Definitional.",
    },
    "identity_violation": {
        "item": "3 lead consistency",
        "definition": "The largest residual over the 10 s of III = II - I, aVR = -(I + II)/2, "
                      "aVL = I - II/2 or aVF = II - I/2.",
        "threshold": f"above {IDENTITY_MV} mV",
        "reason": "Every source stores derived leads that satisfy the identities to rounding (largest "
                  "residual 0.0035 mV at the 99th percentile in every source); 0.05 mV is more than ten "
                  "times that.",
    },
    "reversal_la_ra": {
        "item": "3 lead consistency",
        "definition": "Median beat with lead I fully inverted (P between -0.3 and -0.05 mV, QRS area "
                      "below 0, T below 0), an upright QRS in aVR (area above 0) and in V6 (area above 0), "
                      "a QRS narrower than 120 ms and a beat similarity of at least 0.8: the pattern of "
                      "left-right arm electrode reversal with normal precordial progression.",
        "threshold": "all eight conditions",
        "reason": "Stated rule from the electrode-reversal pattern; V6 separates it from dextrocardia, P "
                  "from a right-axis QRS. The first version (the first five conditions) showed the pattern "
                  "on 7 of its 15 records; the misses were pacing spikes read as P waves (below -0.3 mV), "
                  "wide complexes (120 ms is the usual wide-QRS limit) and unstable beats.",
    },
    "reversal_ra_ll": {
        "item": "3 lead consistency",
        "definition": "Median beat with lead II fully inverted (P below -0.05 mV, QRS area below 0, T below "
                      "0) and aVR upright (QRS area above 0, P above 0.05 mV): the pattern of right arm and "
                      "left leg electrode reversal.",
        "threshold": "all five conditions",
        "reason": "Stated rule from the electrode-reversal pattern. Unreliable: 1 of the first 12 plots "
                  "showed the pattern; most were left-axis or inferior-Q morphologies. Not used for any "
                  "rule.",
    },
    "reversal_neutral": {
        "item": "3 lead consistency",
        "definition": "One of the bipolar limb leads I, II or III is nearly a flat line: its 1-99 percentile "
                      "range of the 0.5-40 Hz signal is below 0.06 mV and below 10% of the largest of the "
                      "three; the pattern of a limb electrode swapped with the right-leg (neutral) "
                      "electrode.",
        "threshold": f"below {NEUTRAL_MV} mV and below {NEUTRAL_RATIO:.0%} of the largest",
        "reason": "A physiologically small limb lead still shows a biphasic QRS. The first version (below "
                  "0.10 mV) showed a flat lead on 8 of 11 records; the 3 misses (0.064-0.084 mV) kept small "
                  "QRS deflections.",
    },
    "amplitude_outlier": {
        "item": "4 amplitude and units",
        "definition": "The median over non-constant leads of the 1-99 percentile range of the 0.5-40 Hz "
                      "signal is at least four times, or at most a quarter of, the train and validation "
                      "median (0.93 mV).",
        "threshold": f"ratio at least {AMPLITUDE_FACTOR:g} or at most 1/{AMPLITUDE_FACTOR:g}",
        "reason": "Source medians lie between 0.88 and 0.94 mV (no source differs by a constant factor); "
                  "the robust standard deviation of log10 amplitude is 0.13-0.15 within each source, so a "
                  "factor of four (0.6 in log10) is more than four robust standard deviations.",
    },
    "baseline_wander": {
        "item": "5 noise",
        "definition": "Peak-to-peak excursion of a non-constant lead below 0.5 Hz between 0.3 and 9.7 s.",
        "threshold": f"at least {BASELINE_MV} mV",
        "reason": "Twice the median lead amplitude; above the 99th percentile in CPSC 2018, CPSC-Extra and "
                  "Georgia and near it in the others.",
    },
    "mains_interference": {
        "item": "5 noise",
        "definition": "Amplitude of a 50 Hz or 60 Hz sinusoid above the neighbouring spectrum in a "
                      "non-constant lead whose largest absolute second difference is below 2 mV.",
        "threshold": f"at least {MAINS_MV} mV (0.1 mV peak to peak)",
        "reason": "0.1 mV peak to peak is one small box at standard gain, visible on paper; it lies above "
                  "the 99th percentile of PTB-XL and Georgia at 50 Hz. The first version (any lead) showed "
                  "mains on 19 of 24 plots; pacing spikes put broadband power at 50 and 60 Hz, so leads with "
                  "a second difference of 2 mV or more are left out.",
    },
    "hf_noise": {
        "item": "5 noise",
        "definition": "Median over time of the 50 ms moving RMS above 40 Hz (mains lines removed) of a "
                      "non-constant lead whose largest absolute second difference is below 2 mV.",
        "threshold": f"at least {HF_NOISE_MV} mV",
        "reason": "Medians lie at 0.007-0.013 mV and 99th percentiles at 0.028-0.066 mV; 0.05 mV RMS "
                  "between beats is visible fuzz. The first version (any lead) showed noise on 5 of the "
                  "first 12 plots: removing the mains bins of a lead with 10 mV pacing spikes spreads their "
                  "energy over the whole lead, so such leads are left out.",
    },
    "baseline_step": {
        "item": "5 noise",
        "definition": "Largest change over 60 ms of the baseline (100 Hz block means, median filters of "
                      "0.2 s "
                      "and 0.6 s) of a non-constant lead between 0.3 and 9.7 s.",
        "threshold": f"at least {STEP_MV} mV",
        "reason": "The median filters remove QRS, P and T, so no physiological baseline moves 1 mV within "
                  "60 ms; calibration sheets at 1-2, 2-4 and above 4 mV all showed artefacts.",
    },
    "isolated_spike": {
        "item": "5 noise",
        "definition": "Absolute second difference of an independent lead (I, II, V1-V6) at an instant when "
                      "at "
                      "most two independent leads have a second difference of at least 0.5 mV within two "
                      "samples.",
        "threshold": f"at least {ISOLATED_SPIKE_MV} mV",
        "reason": "Pacemaker spikes and limb-electrode artefacts appear in many leads; restricting to "
                  "isolated instants cut the paced-rhythm records among spikes of 2 mV from 649 to 47 on "
                  "train and validation. A QRS complex at 500 Hz stays below 1 mV.",
    },
    "contact_loss": {
        "item": "5 noise",
        "definition": "An independent non-constant lead shows no QRS (range below 0.05 mV within 60 ms of "
                      "the "
                      "multi-lead R peak) for consecutive beats, or a non-constant lead holds one value for "
                      "a "
                      "long stretch between 0.3 and 9.7 s.",
        "threshold": f"at least {DROPOUT_BEATS} consecutive beats, or at least {INTERIOR_HOLD_MS:.0f} ms",
        "reason": "No precordial lead has a robust amplitude below 0.13 mV at the 0.01st percentile; PTB-XL "
                  "holds exact values for up to about 350 ms in quiet stretches (99.9th percentile), so the "
                  "hold threshold is 500 ms.",
    },
    "rhythm_unmeasurable": {
        "item": "6 beats",
        "definition": "The multi-lead detector finds at least three beats, but no two of the three beat "
                      "lists (multi-lead detector, XQRS on II, XQRS on V5) that each hold at least three "
                      "beats agree.",
        "threshold": f"every such pairwise F1 below {AGREEMENT}",
        "reason": "Two independent detections that agree to F1 0.8 (matching within 75 ms) give a usable "
                  "rhythm. The first version also flagged records where the multi-lead detector found fewer "
                  "than three beats and counted lists of one beat; it showed an unreadable rhythm on 4 of "
                  "the first 12 plots, because one large artefact sets the multi-lead detector's relative "
                  "threshold and hides real beats. Those records carry an artefact flag.",
    },
}


def lead_matrix(measures: pd.DataFrame, name: str, leads: tuple[str, ...] | list[str] = LEADS) -> np.ndarray:
    """
    Per-lead measure as a ``(records, leads)`` float array.

    Parameters
    ----------
    measures : pd.DataFrame
        Measures table.
    name : str
        Measure name before ``__``.
    leads : tuple[str, ...] | list[str]
        Leads, in order.

    Returns
    -------
    np.ndarray
        Float64 copy.
    """
    return measures[[f"{name}__{lead}" for lead in leads]].to_numpy(dtype=np.float64, copy=True)


def constant_leads(measures: pd.DataFrame, leads: tuple[str, ...] | list[str] = LEADS) -> np.ndarray:
    """
    Leads constant over the 10 s.

    Parameters
    ----------
    measures : pd.DataFrame
        Measures table.
    leads : tuple[str, ...] | list[str]
        Leads, in order.

    Returns
    -------
    np.ndarray
        Boolean ``(records, leads)``.
    """
    return lead_matrix(measures, "flat", leads) > 0


def blame(mask: np.ndarray, leads: tuple[str, ...] | list[str]) -> list[str]:
    """
    Join the leads of each record whose mask is set.

    Parameters
    ----------
    mask : np.ndarray
        Boolean ``(records, leads)``.
    leads : tuple[str, ...] | list[str]
        Lead names.

    Returns
    -------
    list[str]
        Semicolon-separated lead names per record (empty when none).
    """
    names = np.asarray(leads)
    return [";".join(names[row]) for row in mask]


def edge_hold_ms(measures: pd.DataFrame) -> np.ndarray:
    """
    Longest hold at the start or end that every non-constant lead shares, in ms.

    Parameters
    ----------
    measures : pd.DataFrame
        Measures table.

    Returns
    -------
    np.ndarray
        Milliseconds per record; 0 when every lead is constant.
    """
    flat = constant_leads(measures)
    holds = []
    for name in ("head_run", "tail_run"):
        runs = np.where(flat, np.inf, lead_matrix(measures, name))
        shared = runs.min(axis=1)
        holds.append(np.where(np.isfinite(shared), shared, 0.0))
    return np.maximum(*holds) * 1000.0 / FS


def duplicate_flag(measures: pd.DataFrame) -> np.ndarray:
    """
    Exact copy or sign-flip pairs not explained by a constant lead I or II.

    Parameters
    ----------
    measures : pd.DataFrame
        Measures table.

    Returns
    -------
    np.ndarray
        Boolean per record.
    """
    limb_constant = constant_leads(measures, ["I", "II"]).any(axis=1)
    return (measures["n_duplicate_pairs"].to_numpy() > 0) & ~limb_constant


def morphology_rules(measures: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Apply the two polarity rules of electrode reversal to the median beat.

    Parameters
    ----------
    measures : pd.DataFrame
        Measures table.

    Returns
    -------
    dict[str, np.ndarray]
        ``reversal_la_ra`` and ``reversal_ra_ll`` booleans; False where the median beat is missing.
    """
    def column(name: str) -> np.ndarray:
        return measures[name].to_numpy(dtype=np.float64)

    with np.errstate(invalid="ignore"):
        la_ra = ((column("p_amp__I") < -REVERSAL_P_MV) & (column("p_amp__I") >= REVERSAL_P_FLOOR_MV)
                 & (column("qrs_area__I") < 0) & (column("t_amp__I") < 0) & (column("qrs_area__aVR") > 0)
                 & (column("qrs_area__V6") > 0) & (column("qrs_duration_ms") < REVERSAL_QRS_MS)
                 & (column("beat_similarity") >= REVERSAL_SIMILARITY))
        ra_ll = ((column("p_amp__II") < -REVERSAL_P_MV) & (column("qrs_area__II") < 0)
                 & (column("t_amp__II") < 0) & (column("qrs_area__aVR") > 0)
                 & (column("p_amp__aVR") > REVERSAL_P_MV))
    return {"reversal_la_ra": la_ra, "reversal_ra_ll": ra_ll}


def neutral_reversal(measures: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Find a nearly flat bipolar limb lead (I, II or III) beside larger ones.

    Parameters
    ----------
    measures : pd.DataFrame
        Measures table.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        Flag per record and the blamed lead.
    """
    leads = ["I", "II", "III"]
    amplitude = np.where(constant_leads(measures, leads), np.nan, lead_matrix(measures, "amp_robust", leads))
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        largest = np.nanmax(amplitude, axis=1, keepdims=True)
        mask = (amplitude < NEUTRAL_MV) & (amplitude < NEUTRAL_RATIO * largest)
    return mask.any(axis=1), blame(mask, leads)


def amplitude_ratio(measures: pd.DataFrame) -> np.ndarray:
    """
    Median over non-constant leads of the robust amplitude, divided by the train and validation median.

    Parameters
    ----------
    measures : pd.DataFrame
        Measures table.

    Returns
    -------
    np.ndarray
        Ratio per record; NaN when every lead is constant.
    """
    amplitude = np.where(constant_leads(measures), np.nan, lead_matrix(measures, "amp_robust"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(amplitude, axis=1) / AMPLITUDE_REFERENCE_MV


def best_agreement(measures: pd.DataFrame) -> np.ndarray:
    """
    Best pairwise agreement between beat lists that each hold at least ``MIN_BEATS`` beats.

    Parameters
    ----------
    measures : pd.DataFrame
        Measures table.

    Returns
    -------
    np.ndarray
        Largest F1 per record over the qualifying pairs; -1 when no pair qualifies.
    """
    counts = {name: measures[f"n_beats_{name}"].to_numpy(dtype=np.float64) >= MIN_BEATS
              for name in ("multi", "II", "V5")}
    pairs = (("agree_detectors", "multi", "II"), ("agree_leads", "II", "V5"),
             ("agree_multi_V5", "multi", "V5"))
    values = []
    for column, first, second in pairs:
        agreement = np.nan_to_num(measures[column].to_numpy(dtype=np.float64), nan=-1.0)
        values.append(np.where(counts[first] & counts[second], agreement, -1.0))
    return np.max(values, axis=0)


def lead_threshold(measures: pd.DataFrame, name: str, threshold: float, above: bool = True,
                   leads: tuple[str, ...] | list[str] = LEADS) -> tuple[np.ndarray, list[str]]:
    """
    Flag records where a non-constant lead's measure crosses a threshold.

    Parameters
    ----------
    measures : pd.DataFrame
        Measures table.
    name : str
        Per-lead measure.
    threshold : float
        Threshold, inclusive.
    above : bool
        True flags values at or above the threshold, False values below it.
    leads : tuple[str, ...] | list[str]
        Leads considered.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        Flag per record and the blamed leads.
    """
    values = lead_matrix(measures, name, leads)
    with np.errstate(invalid="ignore"):
        mask = (values >= threshold) if above else (values < threshold)
    mask &= ~constant_leads(measures, leads)
    return mask.any(axis=1), blame(mask, leads)


def spectral_threshold(measures: pd.DataFrame, name: str, threshold: float) -> tuple[np.ndarray, list[str]]:
    """
    Flag records where a spectral noise measure crosses a threshold in a lead without large spikes.

    Parameters
    ----------
    measures : pd.DataFrame
        Measures table.
    name : str
        Per-lead spectral measure (``mains50``, ``mains60`` or ``hf_noise``).
    threshold : float
        Threshold, inclusive.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        Flag per record and the blamed leads; a lead whose largest absolute second difference is at least
        ``SPECTRAL_SPIKE_MV`` is never blamed.
    """
    with np.errstate(invalid="ignore"):
        mask = lead_matrix(measures, name) >= threshold
        mask &= lead_matrix(measures, "spike") < SPECTRAL_SPIKE_MV
    mask &= ~constant_leads(measures)
    return mask.any(axis=1), blame(mask, LEADS)


def compute_flags(measures: pd.DataFrame) -> pd.DataFrame:
    """
    Every flag of ``FLAGS`` for every record, with the leads each per-lead flag blames.

    Parameters
    ----------
    measures : pd.DataFrame
        Measures table (one row per record, the columns of ``measure_window``).

    Returns
    -------
    pd.DataFrame
        ``record_key``, one boolean column per flag, ``<flag>__leads`` for flags that blame leads,
        ``any_flag`` and ``n_flags``, and the derived values ``edge_hold_ms``, ``amplitude_ratio``,
        ``best_agreement``.
    """
    flat = constant_leads(measures)
    result = pd.DataFrame({"record_key": measures["record_key"].to_numpy()})
    result["missing_samples"] = measures["missing_samples"].to_numpy() > 0
    result["edge_hold_ms"] = edge_hold_ms(measures)
    result["edge_hold"] = result["edge_hold_ms"] >= EDGE_HOLD_MS
    result["adc_rail"] = measures["rail_samples"].to_numpy() >= RAIL_SAMPLES
    result["flat_lead"] = flat.any(axis=1)
    result["flat_lead__leads"] = blame(flat, LEADS)
    result["near_flat_lead"], result["near_flat_lead__leads"] = lead_threshold(
        measures, "amp_robust", NEAR_FLAT_MV, above=False)
    result["duplicate_lead"] = duplicate_flag(measures)
    columns = [f"res_{lead}_max" for lead in ("III", "aVR", "aVL", "aVF")]
    residual = measures[columns].to_numpy(dtype=np.float64)
    result["identity_violation"] = np.nan_to_num(residual).max(axis=1) > IDENTITY_MV
    for name, values in morphology_rules(measures).items():
        result[name] = values
    result["reversal_neutral"], result["reversal_neutral__leads"] = neutral_reversal(measures)
    result["amplitude_ratio"] = amplitude_ratio(measures)
    with np.errstate(invalid="ignore"):
        result["amplitude_outlier"] = ((result["amplitude_ratio"] >= AMPLITUDE_FACTOR)
                                       | (result["amplitude_ratio"] <= 1.0 / AMPLITUDE_FACTOR))
    result["baseline_wander"], result["baseline_wander__leads"] = lead_threshold(measures, "baseline_ptp",
                                                                                 BASELINE_MV)
    mains50, leads50 = spectral_threshold(measures, "mains50", MAINS_MV)
    mains60, leads60 = spectral_threshold(measures, "mains60", MAINS_MV)
    result["mains_interference"] = mains50 | mains60
    result["mains_interference__leads"] = [";".join(dict.fromkeys(filter(None, f"{a};{b}".split(";"))))
                                           for a, b in zip(leads50, leads60, strict=True)]
    result["hf_noise"], result["hf_noise__leads"] = spectral_threshold(measures, "hf_noise", HF_NOISE_MV)
    result["baseline_step"], result["baseline_step__leads"] = lead_threshold(measures, "step", STEP_MV)
    result["isolated_spike"], result["isolated_spike__leads"] = lead_threshold(
        measures, "spike_isolated", ISOLATED_SPIKE_MV, leads=INDEPENDENT)
    dropout, dropout_leads = lead_threshold(measures, "dropout_run", DROPOUT_BEATS, leads=INDEPENDENT)
    hold, hold_leads = lead_threshold(measures, "const_run_interior", INTERIOR_HOLD_MS * FS / 1000.0)
    result["contact_loss"] = dropout | hold
    result["contact_loss__leads"] = [";".join(dict.fromkeys(filter(None, f"{a};{b}".split(";"))))
                                     for a, b in zip(dropout_leads, hold_leads, strict=True)]
    result["best_agreement"] = best_agreement(measures)
    result["rhythm_unmeasurable"] = ((measures["n_beats_multi"].to_numpy() >= MIN_BEATS)
                                     & (result["best_agreement"].to_numpy() < AGREEMENT))
    flags = list(FLAGS)
    result["n_flags"] = result[flags].sum(axis=1)
    result["any_flag"] = result["n_flags"] > 0
    return result


def flag_counts(flags: pd.DataFrame, identity: pd.DataFrame, included_only: bool = True) -> pd.DataFrame:
    """
    Count flagged records per flag, source and role.

    Parameters
    ----------
    flags : pd.DataFrame
        Output of ``compute_flags``.
    identity : pd.DataFrame
        ``record_key``, ``source``, ``role`` and ``included``.
    included_only : bool
        Count only records included in the MLC-01 build.

    Returns
    -------
    pd.DataFrame
        Long table: ``flag``, ``source``, ``role``, ``records`` (denominator) and ``flagged``.
    """
    frame = flags.merge(identity[["record_key", "source", "role", "included"]], on="record_key",
                        validate="one_to_one")
    if included_only:
        frame = frame[frame["included"]]
    rows = []
    for (source, role), group in frame.groupby(["source", "role"]):
        for flag in [*FLAGS, "any_flag"]:
            rows.append({"flag": flag, "source": source, "role": role, "records": len(group),
                         "flagged": int(group[flag].sum())})
    return pd.DataFrame(rows)


SUMMARY_LEAD_MEASURES = {"amp_robust": "min", "baseline_ptp": "max", "mains50": "max", "mains60": "max",
                         "hf_noise": "max", "step": "max", "edge_step": "max", "spike": "max",
                         "extreme_run": "max", "extreme_hits": "max", "const_run_interior": "max",
                         "quiet_run": "max", "quant_step_uv": "median", "head_run": "min", "tail_run": "min"}
SUMMARY_INDEPENDENT_MEASURES = {"spike_isolated": "max", "dropout_run": "max"}
SUMMARY_RECORD_MEASURES = ("peak_abs_mv", "n_beats_multi", "hr_multi", "agree_detectors", "agree_leads",
                           "agree_multi_V5", "beat_similarity", "qrs_duration_ms")
QUANTILES = (0.0, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0)


def record_summaries(measures: pd.DataFrame) -> pd.DataFrame:
    """
    Record-level summaries of the per-lead measures, over non-constant leads.

    Parameters
    ----------
    measures : pd.DataFrame
        Measures table.

    Returns
    -------
    pd.DataFrame
        ``record_key`` and one column per summary, named ``<reduction>_<measure>``, plus the record-level
        measures of ``SUMMARY_RECORD_MEASURES``, the largest identity residual, ``edge_hold_ms``,
        ``amplitude_ratio`` and ``best_agreement``.
    """
    reductions = {"min": np.nanmin, "max": np.nanmax, "median": np.nanmedian}
    result = pd.DataFrame({"record_key": measures["record_key"].to_numpy()})
    for group, leads in ((SUMMARY_LEAD_MEASURES, LEADS), (SUMMARY_INDEPENDENT_MEASURES, INDEPENDENT)):
        flat = constant_leads(measures, leads)
        for name, reduction in group.items():
            values = np.where(flat, np.nan, lead_matrix(measures, name, leads))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                result[f"{reduction}_{name}"] = reductions[reduction](values, axis=1)
    for name in SUMMARY_RECORD_MEASURES:
        result[name] = measures[name].to_numpy(dtype=np.float64)
    columns = [f"res_{lead}_max" for lead in ("III", "aVR", "aVL", "aVF")]
    result["max_identity_residual"] = measures[columns].to_numpy(dtype=np.float64).max(axis=1)
    result["edge_hold_ms"] = edge_hold_ms(measures)
    result["amplitude_ratio"] = amplitude_ratio(measures)
    result["best_agreement"] = best_agreement(measures)
    return result


def summary_quantiles(summaries: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame:
    """
    Quantiles of every record-level summary per source, on records included in the MLC-01 build.

    Parameters
    ----------
    summaries : pd.DataFrame
        Output of ``record_summaries``.
    identity : pd.DataFrame
        ``record_key``, ``source`` and ``included``.

    Returns
    -------
    pd.DataFrame
        Long table: ``measure``, ``source``, ``records`` and one column per quantile of ``QUANTILES``.
    """
    frame = summaries.merge(identity[["record_key", "source", "included"]], on="record_key",
                            validate="one_to_one")
    frame = frame[frame["included"]]
    rows = []
    for measure in summaries.columns.drop("record_key"):
        for source, group in frame.groupby("source"):
            values = group[measure].dropna().to_numpy()
            row = {"measure": measure, "source": source, "records": len(values)}
            row.update({f"q{quantile:g}": (float(np.quantile(values, quantile)) if len(values) else np.nan)
                        for quantile in QUANTILES})
            rows.append(row)
    return pd.DataFrame(rows)
