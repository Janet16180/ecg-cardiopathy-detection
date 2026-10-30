"""
Factorized combination maps and lead-quality suppression for Experiment 066.

A map is a sum over the 065 MIL statement heads of the head's bag probability, times its lead evidence
(MIL or textbook), times a clinical component prior painted over whole delineated components (P, PQ, QRS,
ST, T) of every beat. Rhythm heads use the MIL time-resolved evidence instead. A lead-quality classifier on
label-free per-lead features learns the PTB-XL noise annotations, and a small listwise ranking model combines
unit-level evidence. See ``docs/experiment-066-combination.md``.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.signal import butter, sosfiltfilt
from scipy.special import logsumexp

from ecg_experiment import mil065
from ecg_experiment.locbench import START, STOP
from ecg_experiment.morphology_boundaries import adaptive_boundaries
from ecg_experiment.normative_map062 import _lead_median, delineate, warp_knots
from ecg_experiment.waveforms import LEADS

FS = 500
SAMPLES = 5000
EPS = 0.05
GAMMA = 2.0
TIE = 1e-3
L2 = 0.01
QUALITY_SEED = 66066
QUALITY_FOLDS = 5
COMPONENTS = ("P", "PQ", "QRS", "ST", "T")
COMPONENT_SETS = {
    "P": ("P",),
    "PR": ("P", "PQ"),
    "QRS": ("QRS",),
    "ST": ("ST",),
    "T": ("T",),
    "beat": COMPONENTS,
}
FALLBACK_PR = 80
FALLBACK_P = 50

ALL = LEADS
LIMB = ("I", "II", "III", "aVR", "aVL", "aVF")
INFERIOR = ("II", "III", "aVF")
HIGH_LATERAL = ("I", "aVL", "V5", "V6")
ANTEROLATERAL = ("I", "aVL", "V3", "V4", "V5", "V6")
ANTEROSEPTAL = ("V1", "V2", "V3", "V4")
INFEROLATERAL = INFERIOR + HIGH_LATERAL

RHYTHM_HEADS = ("AFIB", "AFLT", "BIGU", "PAC", "PACE", "PVC", "SVARR")
# Head -> parts, each (expected leads, expected components). Written from clinical knowledge (protocol table).
PRIOR_TABLE: dict[str, tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]] = {
    "1AVB": ((ALL, ("PR",)),),
    "ABQRS": ((ALL, ("QRS",)),),
    "ALMI": ((ANTEROLATERAL, ("QRS",)),),
    "AMI": ((ANTEROSEPTAL, ("QRS",)),),
    "ANEUR": ((ANTEROSEPTAL, ("QRS", "ST")),),
    "ASMI": ((("V1", "V2", "V3"), ("QRS",)),),
    "CLBBB": ((ALL, ("QRS",)),),
    "CRBBB": ((("I", "V1", "V2", "V6"), ("QRS",)),),
    "DIG": ((("I", "II", "aVF", "V4", "V5", "V6"), ("ST", "T")),),
    "EL": ((ALL, ("ST", "T")),),
    "ILBBB": ((("I", "aVL", "V1", "V5", "V6"), ("QRS",)),),
    "ILMI": ((INFEROLATERAL, ("QRS",)),),
    "IMI": ((INFERIOR, ("QRS",)),),
    "INJAL": ((ANTEROLATERAL, ("ST",)),),
    "INJAS": ((ANTEROSEPTAL, ("ST",)),),
    "INVT": ((ALL, ("T",)),),
    "IRBBB": ((("I", "V1", "V2", "V6"), ("QRS",)),),
    "ISCAL": ((ANTEROLATERAL, ("ST", "T")),),
    "ISCAS": ((ANTEROSEPTAL, ("ST", "T")),),
    "ISCIL": ((INFEROLATERAL, ("ST", "T")),),
    "ISCIN": ((INFERIOR, ("ST", "T")),),
    "ISCLA": ((HIGH_LATERAL, ("ST", "T")),),
    "ISC_": ((ALL, ("ST", "T")),),
    "IVCD": ((ALL, ("QRS",)),),
    "LAFB": ((LIMB, ("QRS",)),),
    "LAO/LAE": ((("I", "II", "aVL", "V1"), ("P",)),),
    "LMI": ((HIGH_LATERAL, ("QRS",)),),
    "LNGQT": ((ALL, ("T",)),),
    "LOWT": ((ALL, ("T",)),),
    "LPFB": ((LIMB, ("QRS",)),),
    "LPR": ((ALL, ("PR",)),),
    "LVH": ((("I", "aVL", "V1", "V2", "V5", "V6"), ("QRS",)), (HIGH_LATERAL, ("ST", "T"))),
    "LVOLT": ((ALL, ("QRS",)),),
    "NDT": ((ALL, ("T",)),),
    "NST_": ((ALL, ("ST",)),),
    "NT_": ((ALL, ("T",)),),
    "QWAVE": ((ALL, ("QRS",)),),
    "RAO/RAE": ((("II", "III", "aVF", "V1"), ("P",)),),
    "RVH": ((("V1", "V2", "V3"), ("QRS",)),),
    "STD_": ((ALL, ("ST",)),),
    "WPW": ((ALL, ("PR", "QRS")),),
}
CABRERA = ("aVL", "I", "aVR", "II", "aVF", "III")
PRECORDIAL = ("V1", "V2", "V3", "V4", "V5", "V6")
QUALITY_FEATURES = (
    "log_hf",
    "log_hf_rel",
    "log_wander",
    "log_wander_rel",
    "log_burst",
    "log_mains_rel",
    "flat_fraction",
    "saturation_fraction",
    "neighbour_correlation",
)
UNIT_MAPS = ("factor", "textbook", "mil", "normative", "residual", "amplitude")
LOG_MAPS = ("normative", "residual", "amplitude")
UNIT_FEATURES = (
    *UNIT_MAPS,
    "q",
    *(f"component_{c}" for c in COMPONENTS),
    *(f"lead_{lead}" for lead in LEADS),
)


def delineated_components(signal: np.ndarray) -> np.ndarray:
    """
    Half-open P, PQ, QRS, ST and T intervals of every beat from the frozen 062 delineation.

    Parameters
    ----------
    signal : np.ndarray
        Raw (12, samples) waveform in mV at 500 Hz.

    Returns
    -------
    np.ndarray
        (beats, 5, 2) int64 [start, end) per component in ``COMPONENTS`` order, clipped to the record.
    """
    fiducials = delineate(signal)
    keep = (fiducials["qon"] >= 0) & (fiducials["qoff"] > fiducials["qon"])
    if not keep.any():
        return np.zeros((0, len(COMPONENTS), 2), np.int64)
    _, bounds = adaptive_boundaries(signal, FS)
    poff = _lead_median(bounds[:, :, 0, 1])
    fiducials = {name: values[keep] for name, values in fiducials.items()}
    poff = poff[keep]
    knots = warp_knots(fiducials)
    qon, qoff, ton, toff = knots[:, 1], knots[:, 2], knots[:, 3], knots[:, 4]
    pon = fiducials["pon"]
    found = np.isfinite(pon) & np.isfinite(poff) & (pon < qon)
    if found.any():
        pr_lead = np.median((qon - pon)[found])
        poff_lead = np.median((qon - np.minimum(np.maximum(poff, pon), qon))[found])
    else:
        pr_lead, poff_lead = FALLBACK_PR, FALLBACK_PR - FALLBACK_P
    pon = np.where(found, pon, qon - pr_lead)
    poff = np.where(found, poff, qon - poff_lead)
    poff = np.clip(poff, pon, qon)
    edges = np.stack([pon, poff, qon, qoff, ton, toff], axis=1)
    intervals = np.stack([edges[:, :-1], edges[:, 1:]], axis=2)
    return np.clip(np.round(intervals), 0, signal.shape[1]).astype(np.int64)


def component_masks(intervals: np.ndarray, samples: int = SAMPLES) -> np.ndarray:
    """
    Sample masks of each component over all beats.

    Parameters
    ----------
    intervals : np.ndarray
        ``delineated_components`` output.
    samples : int
        Record length.

    Returns
    -------
    np.ndarray
        Boolean (5, samples) in ``COMPONENTS`` order.
    """
    masks = np.zeros((len(COMPONENTS), samples), bool)
    for beat in intervals:
        for c, (start, end) in enumerate(beat):
            masks[c, start:end] = True
    return masks


def time_prior(masks: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    """
    One on samples covered by any named component, ``EPS`` elsewhere.

    Parameters
    ----------
    masks : np.ndarray
        ``component_masks`` output.
    names : tuple[str, ...]
        Table component names (keys of ``COMPONENT_SETS``).

    Returns
    -------
    np.ndarray
        (samples,) float64 prior.
    """
    rows = sorted({COMPONENTS.index(c) for name in names for c in COMPONENT_SETS[name]})
    return np.where(masks[rows].any(axis=0), 1.0, EPS)


def lead_indicator(leads: tuple[str, ...]) -> np.ndarray:
    """
    One on the listed leads, ``EPS`` on the others.

    Parameters
    ----------
    leads : tuple[str, ...]
        Lead names.

    Returns
    -------
    np.ndarray
        (12,) float64.
    """
    return np.where(np.isin(LEADS, leads), 1.0, EPS)


def head_components(head: str) -> tuple[str, ...]:
    """Union of a table head's components over its parts."""
    return tuple(dict.fromkeys(name for _, names in PRIOR_TABLE[head] for name in names))


def scored_steps(steps: int) -> np.ndarray:
    """Instance steps whose centre lies in the scored samples."""
    centres = mil065.step_centres(steps)
    return np.flatnonzero((centres >= START) & (centres < STOP))


def lead_means(evidence: np.ndarray) -> np.ndarray:
    """
    Mean instance evidence per head and lead over the scored steps.

    Parameters
    ----------
    evidence : np.ndarray
        (..., heads, 12, steps) evidence.

    Returns
    -------
    np.ndarray
        (..., heads, 12) float32.
    """
    return evidence[..., scored_steps(evidence.shape[-1])].mean(axis=-1).astype(np.float32)


def normalized_leads(means: np.ndarray) -> np.ndarray:
    """
    Lead evidence divided by its maximum over leads.

    Parameters
    ----------
    means : np.ndarray
        (heads, 12) lead means.

    Returns
    -------
    np.ndarray
        (heads, 12) float64 with a maximum of 1 per head (0 where a head is 0 on every lead).
    """
    top = means.max(axis=1, keepdims=True).astype(np.float64)
    return np.divide(means, top, out=np.zeros(means.shape), where=top > 0)


def rhythm_term(native: np.ndarray) -> np.ndarray:
    """
    Time-resolved rhythm evidence on the 500 Hz grid, divided by its maximum over the scored samples.

    Parameters
    ----------
    native : np.ndarray
        (12, steps) instance evidence of one head.

    Returns
    -------
    np.ndarray
        (12, 5000) float64.
    """
    full = mil065.upsample(np.asarray(native, np.float32)).astype(np.float64)
    top = full[:, START:STOP].max()
    return full / top if top > 0 else full


def factor_sum(
    heads: list[str], prob: np.ndarray, means: np.ndarray, rhythm: dict[str, np.ndarray], masks: np.ndarray
) -> np.ndarray:
    """
    Unnormalized ``factor`` map: sum over heads of probability x MIL lead evidence x component prior.

    Parameters
    ----------
    heads : list[str]
        Non-NORM head names.
    prob : np.ndarray
        (heads,) bag probabilities.
    means : np.ndarray
        (heads, 12) lead means of the evidence.
    rhythm : dict[str, np.ndarray]
        Rhythm head -> (12, steps) instance evidence.
    masks : np.ndarray
        ``component_masks`` output.

    Returns
    -------
    np.ndarray
        (12, samples) float64.
    """
    weights = normalized_leads(means)
    total = np.zeros((len(LEADS), masks.shape[1]))
    for k, head in enumerate(heads):
        if head in RHYTHM_HEADS:
            total += prob[k] * rhythm_term(rhythm[head])
        else:
            total += prob[k] * np.outer(weights[k], time_prior(masks, head_components(head)))
    return total


def textbook_region(head: str, masks: np.ndarray) -> np.ndarray:
    """
    Textbook region of one head: maximum over its parts of lead indicator x component prior.

    Rhythm heads get all leads over the whole beat.

    Parameters
    ----------
    head : str
        Head name.
    masks : np.ndarray
        ``component_masks`` output.

    Returns
    -------
    np.ndarray
        (12, samples) float64.
    """
    if head in RHYTHM_HEADS:
        return np.outer(np.ones(len(LEADS)), time_prior(masks, ("beat",)))
    parts = [np.outer(lead_indicator(leads), time_prior(masks, names)) for leads, names in PRIOR_TABLE[head]]
    return np.maximum.reduce(parts)


def textbook_sum(heads: list[str], weights: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """
    Unnormalized ``textbook`` map: sum over heads of weight x textbook region.

    Parameters
    ----------
    heads : list[str]
        Non-NORM head names.
    weights : np.ndarray
        (heads,) bag probabilities or binary true statements.
    masks : np.ndarray
        ``component_masks`` output.

    Returns
    -------
    np.ndarray
        (12, samples) float64.
    """
    total = np.zeros((len(LEADS), masks.shape[1]))
    for k, head in enumerate(heads):
        if weights[k] > 0:
            total += weights[k] * textbook_region(head, masks)
    return total


def quality_weights(q: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """Lead suppression (1 - q)^gamma."""
    return (1.0 - np.asarray(q, np.float64)) ** gamma


def finish(total: np.ndarray, amplitude: np.ndarray) -> np.ndarray:
    """
    Scale a map to a maximum of 1 over the scored samples and add the amplitude tiebreak.

    Parameters
    ----------
    total : np.ndarray
        (12, samples) non-negative map.
    amplitude : np.ndarray
        061 amplitude map of the record.

    Returns
    -------
    np.ndarray
        (12, samples) float32.
    """
    top = total[:, START:STOP].max()
    scaled = total / top if top > 0 else np.zeros_like(total)
    return (scaled + TIE * amplitude).astype(np.float32)


def _neighbours() -> list[list[int]]:
    """Neighbouring leads in the Cabrera and precordial chains."""
    index = {lead: k for k, lead in enumerate(LEADS)}
    result: list[list[int]] = [[] for _ in LEADS]
    for chain in (CABRERA, PRECORDIAL):
        for position, lead in enumerate(chain):
            near = [chain[j] for j in (position - 1, position + 1) if 0 <= j < len(chain)]
            result[index[lead]] = [index[n] for n in near]
    return result


def _flat_fraction(x: np.ndarray, run: int = 25) -> float:
    """Share of samples inside constant runs of at least ``run`` samples."""
    flat = np.concatenate([[False], np.diff(x) == 0, [False]])
    edges = np.flatnonzero(np.diff(flat.astype(np.int8)))
    lengths = edges[1::2] - edges[::2] + 1
    return float(lengths[lengths >= run].sum() / len(x))


def quality_features(signal: np.ndarray) -> np.ndarray:
    """
    Label-free per-lead quality features (protocol list), over samples 100-4899.

    Parameters
    ----------
    signal : np.ndarray
        Exact (12, 5000) signal in mV.

    Returns
    -------
    np.ndarray
        (12, 9) float64 in ``QUALITY_FEATURES`` order.
    """
    signal = np.asarray(signal, np.float64)

    def band(kind: str, cutoff: float | list[float]) -> np.ndarray:
        sos = butter(2, cutoff, btype=kind, fs=FS, output="sos")
        return sosfiltfilt(sos, signal, axis=1)[:, START:STOP]

    high = band("highpass", 40.0)
    passband = band("bandpass", [0.5, 40.0])
    wander = band("lowpass", 0.7)
    mains = band("bandpass", [48.0, 52.0])
    raw = signal[:, START:STOP]
    ptp = np.maximum(np.percentile(passband, 99, axis=1) - np.percentile(passband, 1, axis=1), 0.01)
    hf = np.sqrt(np.mean(high**2, axis=1))
    bw = np.sqrt(np.mean((wander - wander.mean(axis=1, keepdims=True)) ** 2, axis=1))
    windows = high[:, : 125 * (high.shape[1] // 125)].reshape(len(signal), -1, 125)
    window_rms = np.sqrt(np.mean(windows**2, axis=2))
    burst = window_rms.max(axis=1) / np.maximum(np.median(window_rms, axis=1), 1e-6)
    mains_rms = np.sqrt(np.mean(mains**2, axis=1))
    flat = np.array([_flat_fraction(lead) for lead in raw])
    saturation = np.mean(
        (raw == raw.max(axis=1, keepdims=True)) | (raw == raw.min(axis=1, keepdims=True)), axis=1
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.nan_to_num(np.corrcoef(passband), nan=0.0)
    neighbour = np.array([np.abs(correlation[k, near]).max() for k, near in enumerate(_neighbours())])

    def log(values: np.ndarray) -> np.ndarray:
        return np.log(np.maximum(values, 1e-6))

    return np.stack(
        [
            log(hf),
            log(hf / ptp),
            log(bw),
            log(bw / ptp),
            log(burst),
            log(mains_rms / ptp),
            flat,
            saturation,
            neighbour,
        ],
        axis=1,
    )


def patient_folds(patients: np.ndarray, folds: int = QUALITY_FOLDS, seed: int = QUALITY_SEED) -> np.ndarray:
    """
    Fold of each record: sorted unique patients shuffled with ``seed``, fold = position modulo ``folds``.

    Parameters
    ----------
    patients : np.ndarray
        Patient of each record.
    folds : int
        Number of folds.
    seed : int
        Shuffle seed.

    Returns
    -------
    np.ndarray
        (n,) fold index.
    """
    unique = np.unique(patients)
    shuffled = unique.copy()
    np.random.default_rng(seed).shuffle(shuffled)
    fold_of = {patient: position % folds for position, patient in enumerate(shuffled)}
    return np.array([fold_of[p] for p in patients])


def unit_table(intervals: np.ndarray) -> np.ndarray:
    """
    Every non-empty (beat, lead, component) unit.

    Parameters
    ----------
    intervals : np.ndarray
        ``delineated_components`` output.

    Returns
    -------
    np.ndarray
        (units, 5) int64: beat, lead, component, start, end.
    """
    rows = [
        (b, lead, c, start, end)
        for b, beat in enumerate(intervals)
        for c, (start, end) in enumerate(beat)
        if end > start
        for lead in range(len(LEADS))
    ]
    return np.array(rows, np.int64).reshape(-1, 5)


def unit_means(units: np.ndarray, field: np.ndarray) -> np.ndarray:
    """
    Mean of a (12, samples) map over each unit's lead and samples.

    Parameters
    ----------
    units : np.ndarray
        ``unit_table`` output.
    field : np.ndarray
        (12, samples) map.

    Returns
    -------
    np.ndarray
        (units,) float64.
    """
    cumulative = np.concatenate(
        [np.zeros((field.shape[0], 1)), np.cumsum(field, axis=1, dtype=np.float64)], axis=1
    )
    lead, start, end = units[:, 1], units[:, 3], units[:, 4]
    return (cumulative[lead, end] - cumulative[lead, start]) / (end - start)


def unit_design(raw: np.ndarray, q: np.ndarray, units: np.ndarray) -> np.ndarray:
    """
    Build the combiner design matrix of one record.

    Parameters
    ----------
    raw : np.ndarray
        (units, 6) unit means of ``UNIT_MAPS``.
    q : np.ndarray
        (12,) lead-quality probabilities.
    units : np.ndarray
        ``unit_table`` output.

    Returns
    -------
    np.ndarray
        (units, 24) in ``UNIT_FEATURES`` order: within-record z-scored maps (log1p for ``LOG_MAPS``), q,
        component and lead one-hots.
    """
    values = np.array(raw, np.float64)
    for name in LOG_MAPS:
        k = UNIT_MAPS.index(name)
        values[:, k] = np.log1p(np.clip(values[:, k], 0, None))
    spread = values.std(axis=0)
    z = np.divide(values - values.mean(axis=0), spread, out=np.zeros_like(values), where=spread > 0)
    component = np.eye(len(COMPONENTS))[units[:, 2]]
    lead = np.eye(len(LEADS))[units[:, 1]]
    return np.column_stack([z, q[units[:, 1]], component, lead])


def weak_labels(units: np.ndarray, statement: str, table: dict) -> np.ndarray:
    """
    Units in the statement's expected leads and components (PR = P and PQ).

    Parameters
    ----------
    units : np.ndarray
        ``unit_table`` output.
    statement : str
        ``REGION_TABLE`` statement.
    table : dict
        ``locbench_data.REGION_TABLE``.

    Returns
    -------
    np.ndarray
        (units,) boolean.
    """
    leads, names = table[statement]
    components = {COMPONENTS.index(c) for name in names for c in COMPONENT_SETS[name]}
    return np.isin(units[:, 1], [LEADS.index(lead) for lead in leads]) & np.isin(
        units[:, 2], list(components)
    )


def listwise_loss(
    w: np.ndarray, designs: list[np.ndarray], positives: list[np.ndarray], l2: float = L2
) -> tuple[float, np.ndarray]:
    """
    Mean within-record softmax loss with several positives, plus an L2 penalty, and its gradient.

    Parameters
    ----------
    w : np.ndarray
        (features,) weights.
    designs : list[np.ndarray]
        Per record (units, features) design.
    positives : list[np.ndarray]
        Per record (units,) boolean labels.
    l2 : float
        Penalty weight.

    Returns
    -------
    tuple[float, np.ndarray]
        Loss and gradient.
    """
    loss = 0.0
    grad = np.zeros_like(w)
    for x, y in zip(designs, positives, strict=True):
        s = x @ w
        total = logsumexp(s)
        soft = np.exp(s - total)
        loss += total - s[y].mean()
        grad += x.T @ soft - x[y].mean(axis=0)
    n = len(designs)
    return loss / n + l2 * float(w @ w), grad / n + 2 * l2 * w


def fit_combiner(designs: list[np.ndarray], positives: list[np.ndarray], l2: float = L2) -> np.ndarray:
    """
    Minimize ``listwise_loss`` with L-BFGS from zero.

    Parameters
    ----------
    designs, positives : list[np.ndarray]
        Training records.
    l2 : float
        Penalty weight.

    Returns
    -------
    np.ndarray
        Fitted weights.
    """
    start = np.zeros(designs[0].shape[1])
    result = minimize(listwise_loss, start, args=(designs, positives, l2), jac=True, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"Combiner did not converge: {result.message}")
    return result.x


def paint_units(scores: np.ndarray, units: np.ndarray, samples: int = SAMPLES) -> np.ndarray:
    """
    Paint unit scores on their lead and samples, maximum on overlap, record minimum elsewhere.

    Parameters
    ----------
    scores : np.ndarray
        (units,) scores.
    units : np.ndarray
        ``unit_table`` output.
    samples : int
        Record length.

    Returns
    -------
    np.ndarray
        (12, samples) float64.
    """
    field = np.full((len(LEADS), samples), -np.inf)
    for score, (_, lead, _, start, end) in zip(scores, units, strict=True):
        field[lead, start:end] = np.maximum(field[lead, start:end], score)
    field[~np.isfinite(field)] = scores.min()
    return field


def units_map(scores: np.ndarray, units: np.ndarray, amplitude: np.ndarray) -> np.ndarray:
    """
    ``units_learned`` map: painted unit scores scaled to [0, 1] in the record plus the amplitude tiebreak.

    Parameters
    ----------
    scores : np.ndarray
        (units,) combiner scores.
    units : np.ndarray
        ``unit_table`` output.
    amplitude : np.ndarray
        061 amplitude map.

    Returns
    -------
    np.ndarray
        (12, samples) float32.
    """
    if len(units) == 0:
        return (TIE * amplitude).astype(np.float32)
    field = paint_units(scores, units, amplitude.shape[1])
    low, high = field.min(), field.max()
    scaled = (field - low) / (high - low) if high > low else np.zeros_like(field)
    return (scaled + TIE * amplitude).astype(np.float32)
