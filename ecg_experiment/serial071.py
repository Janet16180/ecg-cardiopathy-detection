"""
Same-patient serial change maps for Experiment 071.

A patient's normal ECG and a later ECG with a new ``REGION_TABLE`` statement are each delineated and warped
with the frozen Experiment 062 recipe. The difference of their median beats per lead and grid bin is
standardized by the spread of same-patient normal-normal differences (fit partition only), smoothed and
painted onto the later ECG's beats (an interval-change variant adds QRS, QT and PR changes).
Readings compare the expected lead x wave units of the new statement with the record's other units. See
``docs/experiment-071-serial.md``.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ecg_experiment import normative_map062 as nm
from ecg_experiment.intervals import patient_groups, patient_resample
from ecg_experiment.locbench import (
    REGION_RHYTHMS,
    START,
    STOP,
    auroc,
    interval_mask,
    lead_scores,
    time_scores,
)
from ecg_experiment.locbench_data import REGION_TABLE
from ecg_experiment.waveforms import LEADS

SEED = 71071
BOOT_SEED = 61061
DRAWS = 2000
UNIT_COMPONENTS = ("P", "PQ", "QRS", "ST", "T")
TARGET_COMPONENTS = {"P": ("P",), "PR": ("P", "PQ"), "QRS": ("QRS",), "ST": ("ST",), "T": ("T",)}
FLOOR_QUANTILE = 0.10
INTERVAL_FLOOR = 0.004
MAD_SCALE = 1.4826
INTERVAL_SEGMENTS = {"qrs": "QRS", "qt": "T", "pr": "PR"}
INTERVAL_COLUMNS = {"heart_rate": 0, "qrs": 1, "qt": 2, "pr": 3}
AGE_WINDOWS = (5.0, 10.0, np.inf)
TOP_SHARE = 0.20


@dataclass
class Beats:
    """
    Frozen 062 beat summary of one record.

    Attributes
    ----------
    median : np.ndarray
        Median warped beat of the dominant beats, (12, BINS) in mV.
    knots : np.ndarray
        Warp knots of every complete beat, (beats, 6).
    intervals : np.ndarray
        Heart rate (bpm), QRS duration, QT and PR (seconds); PR may be NaN.
    """

    median: np.ndarray
    knots: np.ndarray
    intervals: np.ndarray


@dataclass
class NullModel:
    """
    Robust centre and spread of reference differences per lead and bin, and of interval differences.

    Attributes
    ----------
    centre, spread : np.ndarray
        (12, BINS) median and floored 1.4826 MAD of the median-beat differences.
    floor : float
        Spread floor in mV.
    interval_centre, interval_spread : np.ndarray
        (3,) median and 1.4826 MAD of the QRS, QT and PR differences in seconds.
    pairs : int
        Pairs used.
    """

    centre: np.ndarray
    spread: np.ndarray
    floor: float
    interval_centre: np.ndarray
    interval_spread: np.ndarray
    pairs: int


def record_flags(meta: pd.DataFrame, statements: pd.DataFrame, partition: pd.DataFrame) -> pd.DataFrame:
    """
    Pair-relevant flags of every folds 1-8 record.

    Parameters
    ----------
    meta : pd.DataFrame
        ``eda.ptbxl.load_metadata`` output (indexed by ``ecg_id``).
    statements : pd.DataFrame
        SCP statement table.
    partition : pd.DataFrame
        ``locbench_data.load_partition`` output.

    Returns
    -------
    pd.DataFrame
        ``ecg_id``, ``patient_id``, ``partition``, ``date``, ``codes`` (dict), ``region`` (tuple of
        ``REGION_TABLE`` statements at likelihood at least 50), ``rhythm_ok`` (rhythm statements only in
        SR, SARRH, SBRAD, STACH and no pacemaker note), ``normal`` (NORM at least 50, no ``REGION_TABLE``
        statement at any likelihood, ``rhythm_ok``), ``finding_free`` (no region statement at 50 or more,
        ``rhythm_ok``), ``noisy_leads``, ``age``, ``sex``.
    """
    rhythm = set(statements.index[statements["rhythm"] == 1])
    rows = meta.loc[partition["ecg_id"].to_numpy()]
    codes = rows["scp_codes"]
    rhythm_ok = codes.apply(lambda c: not any(k in rhythm and k not in REGION_RHYTHMS for k in c))
    rhythm_ok &= rows["pacemaker"].isna()
    region = codes.apply(lambda c: tuple(k for k in REGION_TABLE if c.get(k, 0) >= 50))
    any_region = codes.apply(lambda c: any(k in REGION_TABLE for k in c))
    table = pd.DataFrame(
        {
            "ecg_id": partition["ecg_id"].to_numpy().astype(int),
            "patient_id": partition["patient_id"].to_numpy().astype(int),
            "partition": partition["partition"].to_numpy(),
            "date": rows["recording_date"].to_numpy(),
            "codes": codes.to_numpy(),
            "region": region.to_numpy(),
            "rhythm_ok": rhythm_ok.to_numpy(),
            "normal": (codes.apply(lambda c: c.get("NORM", 0) >= 50) & ~any_region & rhythm_ok).to_numpy(),
            "finding_free": ((region.apply(len) == 0) & rhythm_ok).to_numpy(),
            "noisy_leads": partition["noisy_leads"].to_numpy(),
            "age": rows["age_capped"].to_numpy(),
            "sex": rows["sex"].to_numpy(),
        }
    )
    return table.sort_values(["patient_id", "date", "ecg_id"]).reset_index(drop=True)


def change_pairs(flags: pd.DataFrame, partner_rule: str, direction: str) -> pd.DataFrame:
    """
    One pair per abnormal record: the closest partner of the same patient without its new statements.

    Parameters
    ----------
    flags : pd.DataFrame
        ``record_flags`` output.
    partner_rule : str
        Boolean column a partner must satisfy (``normal`` or ``finding_free``).
    direction : str
        ``forward`` (partner recorded strictly earlier) or ``both`` (either order; ties prefer the earlier).

    Returns
    -------
    pd.DataFrame
        ``ecg_id`` (abnormal record, the map is painted on it), ``partner``, ``patient_id``, ``partition``,
        ``new`` (tuple of new statements), ``forward`` (partner earlier) and ``gap_days`` (absolute).
    """
    rows = []
    for _, group in flags.groupby("patient_id", sort=True):
        if len(group) < 2:
            continue
        for record in group.itertuples(index=False):
            if not record.region or not record.rhythm_ok:
                continue
            best = closest_partner(record, group, partner_rule, direction)
            if best is not None:
                partner, new, forward, gap = best
                rows.append(
                    {
                        "ecg_id": int(record.ecg_id),
                        "partner": int(partner.ecg_id),
                        "patient_id": int(record.patient_id),
                        "partition": record.partition,
                        "new": new,
                        "forward": forward,
                        "gap_days": gap / 86400.0,
                    }
                )
    return pd.DataFrame(
        rows, columns=["ecg_id", "partner", "patient_id", "partition", "new", "forward", "gap_days"]
    )


def closest_partner(
    record: tuple, group: pd.DataFrame, partner_rule: str, direction: str
) -> tuple[tuple, tuple[str, ...], bool, float] | None:
    """
    Find the closest eligible partner of one abnormal record among its patient's records.

    Parameters
    ----------
    record : tuple
        Abnormal record (``itertuples`` row of ``record_flags``).
    group : pd.DataFrame
        The patient's ``record_flags`` rows.
    partner_rule : str
        Boolean column a partner must satisfy.
    direction : str
        ``forward`` or ``both``.

    Returns
    -------
    tuple or None
        Partner row, new statements, whether the partner is earlier, and the gap in seconds; None if no
        partner lacks at least one of the record's statements.
    """
    best = None
    for partner in group.itertuples(index=False):
        if partner.ecg_id == record.ecg_id or not getattr(partner, partner_rule):
            continue
        forward = bool(partner.date < record.date)
        if direction == "forward" and not forward:
            continue
        new = tuple(code for code in record.region if code not in partner.codes)
        if not new:
            continue
        gap = abs((record.date - partner.date) / np.timedelta64(1, "s"))
        key = (gap, not forward, partner.ecg_id)
        if best is None or key < best[0]:
            best = (key, (partner, new, forward, gap))
    return None if best is None else best[1]


def null_pairs(flags: pd.DataFrame) -> pd.DataFrame:
    """
    Consecutive normal ECGs of the same patient, painted on the later one.

    Parameters
    ----------
    flags : pd.DataFrame
        ``record_flags`` output (sorted by patient and date).

    Returns
    -------
    pd.DataFrame
        ``ecg_id`` (later normal), ``partner`` (earlier normal), ``patient_id``, ``partition``, ``gap_days``.
    """
    normal = flags[flags["normal"]]
    rows = []
    for patient, group in normal.groupby("patient_id", sort=True):
        ids, dates, parts = (
            group["ecg_id"].to_numpy(),
            group["date"].to_numpy(),
            group["partition"].to_numpy(),
        )
        for k in range(1, len(group)):
            rows.append(
                {
                    "ecg_id": int(ids[k]),
                    "partner": int(ids[k - 1]),
                    "patient_id": int(patient),
                    "partition": parts[k],
                    "gap_days": float((dates[k] - dates[k - 1]) / np.timedelta64(1, "s")) / 86400.0,
                }
            )
    return pd.DataFrame(rows, columns=["ecg_id", "partner", "patient_id", "partition", "gap_days"])


def match_partner(target: pd.Series, pool: pd.DataFrame) -> int:
    """
    Match a normal ECG of another patient on sex, age and heart rate.

    Same sex; age within 5 years (widened to 10, then any age; ages above 89, stored as NaN, match each
    other); then the nearest heart rate, ties to the smallest ``ecg_id``.

    Parameters
    ----------
    target : pd.Series
        ``patient_id``, ``sex``, ``age``, ``heart_rate``.
    pool : pd.DataFrame
        Candidates with ``ecg_id``, ``patient_id``, ``sex``, ``age``, ``heart_rate``.

    Returns
    -------
    int
        ``ecg_id`` of the match.
    """
    candidates = pool[(pool["sex"] == target["sex"]) & (pool["patient_id"] != target["patient_id"])]
    age = candidates["age"].to_numpy(float)
    for window in AGE_WINDOWS:
        if np.isnan(target["age"]):
            keep = np.isnan(age) if np.isfinite(window) else np.ones(len(age), bool)
        else:
            keep = np.abs(age - target["age"]) <= window
            keep |= np.isnan(age) & (target["age"] + window >= 90)
        if keep.any():
            break
    chosen = candidates[keep]
    distance = np.abs(chosen["heart_rate"].to_numpy(float) - float(target["heart_rate"]))
    order = np.lexsort((chosen["ecg_id"].to_numpy(), distance))
    return int(chosen["ecg_id"].to_numpy()[order[0]])


def beat_summary(signal: np.ndarray) -> Beats | None:
    """
    Median warped beat, knots and intervals of one record with the frozen 062 recipe.

    Parameters
    ----------
    signal : np.ndarray
        Raw (12, 5000) waveform in mV.

    Returns
    -------
    Beats or None
        None when fewer than two complete beats are found.
    """
    features = nm.record_features(np.asarray(signal, np.float64))
    if features is None:
        return None
    return Beats(features.median.astype(np.float32), features.knots, features.intervals)


def interval_change(late: np.ndarray, early: np.ndarray) -> np.ndarray:
    """
    QRS, QT and PR differences (later minus earlier) in seconds; NaN where either PR is missing.

    Parameters
    ----------
    late, early : np.ndarray
        ``Beats.intervals`` of the two records.

    Returns
    -------
    np.ndarray
        (3,) differences in ``INTERVAL_SEGMENTS`` order.
    """
    columns = [INTERVAL_COLUMNS[name] for name in INTERVAL_SEGMENTS]
    return np.asarray(late, float)[columns] - np.asarray(early, float)[columns]


def robust_centre_spread(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Median and 1.4826 MAD along the first axis, ignoring NaN.

    Parameters
    ----------
    values : np.ndarray
        (pairs, ...) values.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Centre and spread with the trailing shape.
    """
    with warnings.catch_warnings(action="ignore"):
        centre = np.nanmedian(values, axis=0)
        spread = MAD_SCALE * np.nanmedian(np.abs(values - centre), axis=0)
    return centre, spread


def fit_null(differences: np.ndarray, interval_differences: np.ndarray) -> NullModel:
    """
    Fit the reference-difference model on fit-partition pairs.

    The spread of each lead and bin is floored at the ``FLOOR_QUANTILE`` quantile of all lead-bin spreads,
    so that flat, low-variance segments cannot dominate the map; interval spreads are floored at
    ``INTERVAL_FLOOR`` seconds (two samples), because record intervals are quantized to whole samples.

    Parameters
    ----------
    differences : np.ndarray
        (pairs, 12, BINS) median-beat differences in mV.
    interval_differences : np.ndarray
        (pairs, 3) ``interval_change`` rows.

    Returns
    -------
    NullModel
        Fitted model.
    """
    centre, spread = robust_centre_spread(differences)
    floor = float(np.quantile(spread, FLOOR_QUANTILE))
    interval_centre, interval_spread = robust_centre_spread(interval_differences)
    return NullModel(
        centre=centre,
        spread=np.maximum(spread, floor),
        floor=floor,
        interval_centre=interval_centre,
        interval_spread=np.maximum(interval_spread, INTERVAL_FLOOR),
        pairs=len(differences),
    )


def change_grid(late: Beats, early: Beats, model: NullModel, intervals: bool = False) -> np.ndarray:
    """
    Standardized, smoothed change magnitude per lead and grid bin, optionally plus interval-change evidence.

    Parameters
    ----------
    late, early : Beats
        Records whose difference is mapped (the map belongs to ``late``).
    model : NullModel
        Reference-difference model.
    intervals : bool
        Add the absolute standardized QRS, QT and PR changes to the QRS, T and pre-QRS bins of every lead.

    Returns
    -------
    np.ndarray
        (12, BINS) non-negative scores.
    """
    z = (late.median - early.median - model.centre) / model.spread
    score = nm.smooth_abs(z)
    if intervals:
        change = interval_change(late.intervals, early.intervals)
        terms = np.abs((change - model.interval_centre) / model.interval_spread)
        for term, segment in zip(terms, INTERVAL_SEGMENTS.values(), strict=True):
            if np.isfinite(term):
                score[:, nm.SEGMENTS.index(segment) == nm.SEGMENT_OF_BIN] += term
    return score


def paint(grid: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """
    Paint one grid map onto every beat of a record (map contract).

    Parameters
    ----------
    grid : np.ndarray
        (12, BINS) scores.
    knots : np.ndarray
        (beats, 6) warp knots of the record.

    Returns
    -------
    np.ndarray
        (12, 5000) float32; samples no beat covers get the minimum covered value.
    """
    return nm.to_timeline(np.broadcast_to(grid, (len(knots), *grid.shape)), knots)


def unit_masks(components: dict[str, np.ndarray]) -> np.ndarray:
    """
    Sample masks of the five wave units from ECGdeli components (PQ = PR minus P), samples 100-4899 only.

    Parameters
    ----------
    components : dict[str, np.ndarray]
        ``locbench.read_components`` output.

    Returns
    -------
    np.ndarray
        Boolean (5, 5000) in ``UNIT_COMPONENTS`` order.
    """
    p = interval_mask(components["P"])
    masks = np.stack(
        [
            p,
            interval_mask(components["PR"]) & ~p,
            interval_mask(components["QRS"]),
            interval_mask(components["ST"]),
            interval_mask(components["T"]),
        ]
    )
    masks[:, :START] = False
    masks[:, STOP:] = False
    return masks


def unit_scores(record_map: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """
    Mean of a map over each lead x wave unit; NaN for empty units.

    Parameters
    ----------
    record_map : np.ndarray
        (12, 5000) map.
    masks : np.ndarray
        ``unit_masks`` output.

    Returns
    -------
    np.ndarray
        (12, 5) float64.
    """
    values = np.asarray(record_map, np.float64)
    counts = masks.sum(axis=1)
    sums = values @ masks.T.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def target_units(statements: tuple[str, ...]) -> np.ndarray:
    """
    Mark the expected lead x wave units of a set of statements (union; PR = P and PQ).

    Parameters
    ----------
    statements : tuple[str, ...]
        ``REGION_TABLE`` statements.

    Returns
    -------
    np.ndarray
        Boolean (12, 5).
    """
    target = np.zeros((len(LEADS), len(UNIT_COMPONENTS)), bool)
    for code in statements:
        leads, names = REGION_TABLE[code]
        rows = np.isin(LEADS, leads)
        columns = np.isin(UNIT_COMPONENTS, [c for name in names for c in TARGET_COMPONENTS[name]])
        target |= rows[:, None] & columns[None, :]
    return target


def target_samples(
    statements: tuple[str, ...], masks: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the expected lead-sample region, leads and samples of a set of statements.

    Parameters
    ----------
    statements : tuple[str, ...]
        ``REGION_TABLE`` statements.
    masks : np.ndarray
        ``unit_masks`` output.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Boolean (12, 5000) region, (12,) leads and (5000,) samples.
    """
    target = target_units(statements)
    region = target.astype(np.int64) @ masks.astype(np.int64) > 0
    return region, target.any(axis=1), region.any(axis=0)


def unit_auroc(scores: np.ndarray, target: np.ndarray) -> float:
    """
    Within-record AUROC of unit scores, expected against other units; NaN if a class is missing.

    Parameters
    ----------
    scores : np.ndarray
        ``unit_scores`` output.
    target : np.ndarray
        ``target_units`` output.

    Returns
    -------
    float
        AUROC over the non-empty units.
    """
    keep = np.isfinite(scores)
    y = target[keep]
    if y.all() or not y.any():
        return np.nan
    return auroc(y, scores[keep])


def region_readings(
    record_map: np.ndarray, statements: tuple[str, ...], masks: np.ndarray
) -> dict[str, float]:
    """
    Read B1u and the sample-level LocBench B1, B2 and B3 for a union of expected regions.

    Parameters
    ----------
    record_map : np.ndarray
        (12, 5000) map.
    statements : tuple[str, ...]
        ``REGION_TABLE`` statements.
    masks : np.ndarray
        ``unit_masks`` output.

    Returns
    -------
    dict[str, float]
        ``B1u``, ``B1``, ``B2`` (NaN when every lead is expected) and ``B3``; NaN where a class is missing.
    """
    region, leads, samples = target_samples(statements, masks)
    window = samples[START:STOP]
    inside = region[:, START:STOP]
    result = {"B1u": unit_auroc(unit_scores(record_map, masks), target_units(statements))}
    result["B1"] = auroc(inside, record_map[:, START:STOP]) if inside.any() and not inside.all() else np.nan
    result["B2"] = auroc(leads, lead_scores(record_map)) if 0 < leads.sum() < len(LEADS) else np.nan
    result["B3"] = auroc(window, time_scores(record_map)) if window.any() and not window.all() else np.nan
    return result


def lead_auroc(
    record_map: np.ndarray,
    leads: np.ndarray,
    centre: np.ndarray | None = None,
    spread: np.ndarray | None = None,
) -> float:
    """
    Within-record AUROC of lead scores against named leads, optionally standardized per lead.

    Parameters
    ----------
    record_map : np.ndarray
        (12, 5000) map.
    leads : np.ndarray
        Boolean (12,) named leads (1-11 of them).
    centre, spread : np.ndarray or None
        Per-lead median and interquartile range of the lead score on reference records.

    Returns
    -------
    float
        AUROC.
    """
    scores = lead_scores(record_map)
    if centre is not None and spread is not None:
        scores = (scores - centre) / np.where(spread > 0, spread, 1.0)
    return auroc(leads, scores)


def null_reading(record_map: np.ndarray, masks: np.ndarray, composition: dict[tuple[str, ...], int]) -> float:
    """
    Pseudo-statement unit AUROC of a normal-normal map: the composition-weighted mean over statement sets.

    Parameters
    ----------
    record_map : np.ndarray
        (12, 5000) map of the later normal ECG.
    masks : np.ndarray
        ``unit_masks`` of that ECG.
    composition : dict[tuple[str, ...], int]
        Statement set to its count among the change pairs read.

    Returns
    -------
    float
        Weighted mean B1u over the sets where it is defined; NaN if none.
    """
    scores = unit_scores(record_map, masks)
    values, weights = [], []
    for statements, count in composition.items():
        value = unit_auroc(scores, target_units(statements))
        if np.isfinite(value):
            values.append(value)
            weights.append(count)
    return float(np.average(values, weights=weights)) if values else np.nan


def bootstrap_means(
    groups: list[tuple[np.ndarray, np.ndarray]],
    combine: Callable[[list[float]], float],
    draws: int = DRAWS,
    seed: int = BOOT_SEED,
) -> dict[str, float]:
    """
    Whole-patient bootstrap of a function of several group means, resampling the union of their patients.

    Parameters
    ----------
    groups : list[tuple[np.ndarray, np.ndarray]]
        Per group, the record values and their patient IDs; rows with NaN values are dropped first.
    combine : Callable[[list[float]], float]
        Maps the group means to the reported statistic (for example ``lambda m: m[0] - m[1]``).
    draws : int
        Bootstrap draws.
    seed : int
        Seed of ``numpy.random.default_rng``.

    Returns
    -------
    dict[str, float]
        ``value``, ``ci_low``, ``ci_high``, ``records`` (per group), ``patients`` and ``skipped_draws``.
    """
    clean = []
    for values, patients in groups:
        values = np.asarray(values, float)
        keep = np.isfinite(values)
        clean.append((values[keep], np.asarray(patients)[keep]))
    labels = np.concatenate([np.full(len(v), g) for g, (v, _) in enumerate(clean)])
    values = np.concatenate([v for v, _ in clean])
    patients = np.concatenate([p for _, p in clean])
    point = combine([float(v.mean()) for v, _ in clean])
    patient_rows = patient_groups(patients)
    rng = np.random.default_rng(seed)
    out, skipped = [], 0
    for _ in range(draws):
        rows = patient_resample(patient_rows, rng)
        means = []
        for g in range(len(clean)):
            drawn = values[rows][labels[rows] == g]
            means.append(float(drawn.mean()) if len(drawn) else np.nan)
        if any(np.isnan(means)):
            skipped += 1
            continue
        out.append(combine(means))
    low, high = np.percentile(out, [2.5, 97.5])
    return {
        "value": float(point),
        "ci_low": float(low),
        "ci_high": float(high),
        "records": [len(v) for v, _ in clean],
        "patients": int(len(np.unique(patients))),
        "skipped_draws": skipped,
    }


def paired_mean_difference(
    a: np.ndarray, b: np.ndarray, patients: np.ndarray, draws: int = DRAWS, seed: int = BOOT_SEED
) -> dict[str, float]:
    """
    Mean of a minus mean of b over records where both are defined, with a whole-patient bootstrap.

    Parameters
    ----------
    a, b : np.ndarray
        Per-record values of two maps on the same records.
    patients : np.ndarray
        Patient ID per record.
    draws : int
        Bootstrap draws.
    seed : int
        Seed.

    Returns
    -------
    dict[str, float]
        ``bootstrap_means`` output for the difference, plus ``a``, ``b`` and ``wins`` (records with a > b).
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b)
    result = bootstrap_means([(a[keep] - b[keep], np.asarray(patients)[keep])], lambda m: m[0], draws, seed)
    result.update(
        {"a": float(a[keep].mean()), "b": float(b[keep].mean()), "wins": int((a[keep] > b[keep]).sum())}
    )
    return result


def change_unit_labels(scores: np.ndarray, share: float = TOP_SHARE) -> np.ndarray:
    """
    Weak labels for the secondary model: the record's units in the top ``share`` of change scores.

    Parameters
    ----------
    scores : np.ndarray
        (units,) unit means of a change map.
    share : float
        Share of units labelled positive (at least one).

    Returns
    -------
    np.ndarray
        Boolean (units,).
    """
    k = max(1, int(round(share * len(scores))))
    order = np.lexsort((np.arange(len(scores)), -scores))
    labels = np.zeros(len(scores), bool)
    labels[order[:k]] = True
    return labels


def family_macro(
    values: np.ndarray,
    families: list[list[str]],
    patients: np.ndarray,
    minimum: int = 3,
    draws: int = DRAWS,
    seed: int = BOOT_SEED,
) -> dict[str, Any]:
    """
    Unweighted mean over families of the family mean, with a whole-patient bootstrap.

    A record counts in each of its families. Families with fewer than ``minimum`` defined records are left
    out; a draw that misses an included family is skipped and counted.

    Parameters
    ----------
    values : np.ndarray
        Per-record values (a paired difference gives the macro difference).
    families : list[list[str]]
        Families of each record.
    patients : np.ndarray
        Patient ID per record.
    minimum : int
        Fewest records for a family to enter.
    draws : int
        Bootstrap draws.
    seed : int
        Seed.

    Returns
    -------
    dict[str, Any]
        ``value``, ``ci_low``, ``ci_high``, ``families`` (included, with record counts), ``skipped_draws``.
    """
    values, patients = np.asarray(values, float), np.asarray(patients)
    rows = [(k, f) for k, names in enumerate(families) for f in names if np.isfinite(values[k])]
    counts = pd.Series([f for _, f in rows], dtype=object).value_counts()
    included = sorted(counts.index[counts >= minimum])
    if not included:
        return {"value": None, "families": {}}
    record = np.array([k for k, f in rows if f in included])
    family = np.array([included.index(f) for _, f in rows if f in included])

    def macro(positions: np.ndarray) -> float:
        drawn = np.isin(record, positions)
        multiplicity = np.bincount(positions, minlength=len(values))
        sums = np.zeros(len(included))
        sizes = np.zeros(len(included))
        np.add.at(sums, family[drawn], values[record[drawn]] * multiplicity[record[drawn]])
        np.add.at(sizes, family[drawn], multiplicity[record[drawn]])
        return float(np.mean(sums / sizes)) if (sizes > 0).all() else np.nan

    groups = patient_groups(patients)
    rng = np.random.default_rng(seed)
    out, skipped = [], 0
    for _ in range(draws):
        value = macro(patient_resample(groups, rng))
        if np.isnan(value):
            skipped += 1
        else:
            out.append(value)
    low, high = np.percentile(out, [2.5, 97.5])
    return {
        "value": macro(np.arange(len(values))),
        "ci_low": float(low),
        "ci_high": float(high),
        "families": {f: int(counts[f]) for f in included},
        "skipped_draws": skipped,
    }
