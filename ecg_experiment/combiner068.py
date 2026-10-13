"""
Learned unit combiner, version 2, for Experiment 068.

The 066 combiner ranks (beat, lead, component) units of a record with a linear listwise softmax. Version 2
trains it on every ``fit`` record with a region statement, using out-of-fold 065 MIL evidence from a
patient-level cross-fit, and adds a softened MIL lead factor and diagnosis-by-component interactions to the
066 features. See ``docs/experiment-068-combiner-v2.md``.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ecg_experiment import combination066 as cb
from ecg_experiment.locbench import auroc
from ecg_experiment.waveforms import LEADS

SEED = 68068
FOLDS = 5
FOLD_SEED = cb.QUALITY_SEED
SOFT_GRID = (0.25, 0.5, 0.75)
L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1)
SOFT_MAPS = tuple(f"factor_soft{int(100 * a)}" for a in SOFT_GRID)
RAW_MAPS = (*cb.UNIT_MAPS, *SOFT_MAPS)
MIL_LEAD_MAPS = ("factor", "mil", *SOFT_MAPS)
INTERACTIONS = tuple(f"textbook_x_{c}" for c in cb.COMPONENTS)
FEATURES = (
    *RAW_MAPS,
    "q",
    *(f"component_{c}" for c in cb.COMPONENTS),
    *(f"lead_{lead}" for lead in LEADS),
    *INTERACTIONS,
)


def soft_means(means: np.ndarray, a: float) -> np.ndarray:
    """
    Softened lead factor ``(1 - a) + a L_k`` with ``L_k`` the lead evidence divided by its lead maximum.

    Parameters
    ----------
    means : np.ndarray
        (heads, 12) lead means of the MIL evidence.
    a : float
        Mixing weight in [0, 1]; 1 gives the 066 ``factor`` lead weights, 0 ignores the lead evidence.

    Returns
    -------
    np.ndarray
        (heads, 12) float64 with a maximum of 1 per head, so ``combination066.factor_sum`` keeps it.
    """
    return (1.0 - a) + a * cb.normalized_leads(means)


def raw_fields(
    names: list[str],
    prob: np.ndarray,
    means: np.ndarray,
    rhythm: dict[str, np.ndarray],
    masks: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Unnormalized ``factor``, ``textbook`` and softened-factor maps of one record.

    Parameters
    ----------
    names : list[str]
        Non-NORM head names.
    prob : np.ndarray
        (heads,) bag probabilities.
    means : np.ndarray
        (heads, 12) lead means of the evidence.
    rhythm : dict[str, np.ndarray]
        Rhythm head -> (12, steps) instance evidence.
    masks : np.ndarray
        ``combination066.component_masks`` output.

    Returns
    -------
    dict[str, np.ndarray]
        Name -> (12, samples) float64 map, each divided by its maximum over the scored samples.
    """
    fields = {
        "factor": cb.factor_sum(names, prob, means, rhythm, masks),
        "textbook": cb.textbook_sum(names, prob, masks),
    }
    for a, name in zip(SOFT_GRID, SOFT_MAPS, strict=True):
        fields[name] = cb.factor_sum(names, prob, soft_means(means, a), rhythm, masks)
    return {name: field / max(field[:, 100:4900].max(), 1e-30) for name, field in fields.items()}


def unit_design(raw: np.ndarray, q: np.ndarray, units: np.ndarray) -> np.ndarray:
    """
    Full design matrix of one record in ``FEATURES`` order.

    Parameters
    ----------
    raw : np.ndarray
        (units, 9) unit means of ``RAW_MAPS``.
    q : np.ndarray
        (12,) lead-quality probabilities.
    units : np.ndarray
        ``combination066.unit_table`` output.

    Returns
    -------
    np.ndarray
        (units, 32): within-record z-scored maps (log1p for ``combination066.LOG_MAPS``), q, component and
        lead one-hots, and the z-scored ``textbook`` feature times each component one-hot.
    """
    values = np.array(raw, np.float64)
    for name in cb.LOG_MAPS:
        k = RAW_MAPS.index(name)
        values[:, k] = np.log1p(np.clip(values[:, k], 0, None))
    spread = values.std(axis=0)
    z = np.divide(values - values.mean(axis=0), spread, out=np.zeros_like(values), where=spread > 0)
    component = np.eye(len(cb.COMPONENTS))[units[:, 2]]
    lead = np.eye(len(LEADS))[units[:, 1]]
    textbook = z[:, RAW_MAPS.index("textbook")]
    return np.column_stack([z, q[units[:, 1]], component, lead, component * textbook[:, None]])


def columns(names: tuple[str, ...]) -> np.ndarray:
    """Positions of the named features in ``FEATURES``."""
    return np.array([FEATURES.index(name) for name in names])


def feature_set(variant: str, a: float) -> tuple[str, ...]:
    """
    Return the declared features of a combiner variant.

    Parameters
    ----------
    variant : str
        ``"units_learned"`` (the 066 features), ``"units_v2"`` or ``"units_v2_nolead"``.
    a : float
        Softened-factor weight from ``SOFT_GRID`` (ignored by ``units_learned``).

    Returns
    -------
    tuple[str, ...]
        Feature names.
    """
    base = cb.UNIT_FEATURES
    if variant == "units_learned":
        return base
    soft = SOFT_MAPS[SOFT_GRID.index(a)]
    full = (*base[:6], soft, *base[6:], *INTERACTIONS)
    if variant == "units_v2":
        return full
    if variant == "units_v2_nolead":
        return tuple(name for name in full if name not in MIL_LEAD_MAPS)
    raise ValueError(f"Unknown variant {variant}")


def union_labels(units: np.ndarray, statements: list[str], table: dict) -> np.ndarray:
    """
    Weak labels of a record with one or more region statements: the union over its statements.

    Parameters
    ----------
    units : np.ndarray
        ``combination066.unit_table`` output.
    statements : list[str]
        ``REGION_TABLE`` statements of the record.
    table : dict
        ``locbench_data.REGION_TABLE``.

    Returns
    -------
    np.ndarray
        (units,) boolean.
    """
    labels = np.zeros(len(units), bool)
    for statement in statements:
        labels |= cb.weak_labels(units, statement, table)
    return labels


def stack(
    designs: list[np.ndarray], positives: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Concatenate per-record designs and labels.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``x`` (units, features), ``y`` (units,) boolean and ``starts`` (records + 1,) offsets.
    """
    starts = np.concatenate([[0], np.cumsum([len(y) for y in positives])])
    return np.concatenate(designs), np.concatenate(positives), starts


def listwise_loss(
    w: np.ndarray, x: np.ndarray, y: np.ndarray, starts: np.ndarray, l2: float
) -> tuple[float, np.ndarray]:
    """
    Vectorized 066 listwise loss: mean over records of logsumexp minus the mean positive score, plus L2.

    Parameters
    ----------
    w : np.ndarray
        (features,) weights.
    x, y, starts : np.ndarray
        ``stack`` output.
    l2 : float
        Penalty weight.

    Returns
    -------
    tuple[float, np.ndarray]
        Loss and gradient.
    """
    s = x @ w
    heads = starts[:-1]
    sizes = np.diff(starts)
    top = np.maximum.reduceat(s, heads)
    shifted = np.exp(s - np.repeat(top, sizes))
    total = np.add.reduceat(shifted, heads)
    lse = top + np.log(total)
    counts = np.add.reduceat(y.astype(np.float64), heads)
    positive_mean = np.add.reduceat(np.where(y, s, 0.0), heads) / counts
    soft = shifted / np.repeat(total, sizes)
    target = np.where(y, 1.0, 0.0) / np.repeat(counts, sizes)
    n = len(heads)
    loss = float(np.mean(lse - positive_mean)) + l2 * float(w @ w)
    grad = x.T @ (soft - target) / n + 2 * l2 * w
    return loss, grad


def fit_listwise(x: np.ndarray, y: np.ndarray, starts: np.ndarray, l2: float) -> np.ndarray:
    """
    Minimize ``listwise_loss`` with L-BFGS from zero.

    Parameters
    ----------
    x, y, starts : np.ndarray
        ``stack`` output.
    l2 : float
        Penalty weight.

    Returns
    -------
    np.ndarray
        Fitted weights.
    """
    result = minimize(
        listwise_loss, np.zeros(x.shape[1]), args=(x, y, starts, l2), jac=True, method="L-BFGS-B"
    )
    if not result.success:
        raise RuntimeError(f"Combiner did not converge: {result.message}")
    return result.x


def subset(
    x: np.ndarray, y: np.ndarray, starts: np.ndarray, records: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rows of the chosen records from a stacked design.

    Parameters
    ----------
    x, y, starts : np.ndarray
        ``stack`` output.
    records : np.ndarray
        Record positions to keep, in order.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Stacked design of the subset.
    """
    rows = np.concatenate([np.arange(starts[r], starts[r + 1]) for r in records])
    sizes = np.diff(starts)[records]
    return x[rows], y[rows], np.concatenate([[0], np.cumsum(sizes)])


def within_record_auroc(scores: np.ndarray, y: np.ndarray, starts: np.ndarray) -> float:
    """Mean over records of the AUROC of unit scores against the weak labels."""
    return float(np.mean([auroc(y[a:b], scores[a:b]) for a, b in zip(starts[:-1], starts[1:], strict=True)]))


def record_weights(starts: np.ndarray) -> np.ndarray:
    """Weight units so that every record has a total weight of 1."""
    sizes = np.diff(starts)
    return np.repeat(1.0 / sizes, sizes)
