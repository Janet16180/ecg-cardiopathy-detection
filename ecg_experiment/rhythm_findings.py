"""Finding groups from the 2017 athlete criteria and their per-set statistics, for Experiment 032.

Each group is one binary target defined from PTB-XL SCP codes, Challenge SNOMED CT codes and SPH AHA codes.
A record is positive when it lists a positive code of the group, undefined (dropped from that group) when it
lists only an ambiguous code of the group, and negative otherwise. CPSC 2018 annotates only nine classes, so
its records are defined only for the groups it annotates.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .normal_manifold import metrics, patient_resamples
from .referral_budget import budget_threshold

SCHEMES = ("ptbxl", "snomed", "sph")
GROUPS: dict[str, dict[str, dict[str, frozenset[str]]]] = {
    "ventricular_ectopy": {
        "positive": {"ptbxl": frozenset({"PVC"}),
                     "snomed": frozenset({"427172004", "17338001", "164884008", "11157007", "251180001",
                                          "425856008", "164895002"}),
                     "sph": frozenset({"60"})},
        "ambiguous": {"ptbxl": frozenset({"BIGU", "TRIGU", "PRC(S)"}), "snomed": frozenset({"13640000"}),
                      "sph": frozenset()},
    },
    "preexcitation": {
        "positive": {"ptbxl": frozenset({"WPW"}), "snomed": frozenset({"74390002", "195060002"}),
                     "sph": frozenset({"108"})},
        "ambiguous": {"ptbxl": frozenset(), "snomed": frozenset({"49578007"}), "sph": frozenset({"80"})},
    },
    "af_flutter": {
        "positive": {"ptbxl": frozenset({"AFIB", "AFLT"}),
                     "snomed": frozenset({"164889003", "164890007", "195080001", "426749004"}),
                     "sph": frozenset({"50", "51"})},
        "ambiguous": {"ptbxl": frozenset(), "snomed": frozenset(), "sph": frozenset()},
    },
    "svt": {
        "positive": {"ptbxl": frozenset({"SVTAC", "PSVT"}),
                     "snomed": frozenset({"426761007", "713422000", "233897008", "251166008", "426648003"}),
                     "sph": frozenset({"54"})},
        "ambiguous": {"ptbxl": frozenset({"SVARR"}), "snomed": frozenset(), "sph": frozenset()},
    },
    "high_grade_av_block": {
        "positive": {"ptbxl": frozenset({"3AVB"}), "snomed": frozenset({"426183003", "27885002"}),
                     "sph": frozenset({"84", "87", "88"})},
        "ambiguous": {"ptbxl": frozenset({"2AVB"}),
                      "snomed": frozenset({"195042002", "233917008", "50799005"}),
                      "sph": frozenset({"85", "86"})},
    },
    "long_qt": {
        "positive": {"ptbxl": frozenset({"LNGQT"}), "snomed": frozenset({"111975006"}),
                     "sph": frozenset({"148"})},
        "ambiguous": {"ptbxl": frozenset(), "snomed": frozenset(), "sph": frozenset()},
    },
}
PARTIAL_ANNOTATION = {"cpsc_2018": frozenset({"ventricular_ectopy", "af_flutter"})}
FREQUENT_PVC_MODIFIERS = frozenset({"310", "340", "341", "342"})


def finding_label(codes: set[str], group: str, scheme: str) -> float:
    """
    Label of one record for one finding group.

    Parameters
    ----------
    codes : set[str]
        The record's codes in the scheme (SCP keys, SNOMED codes, or SPH base AHA codes).
    group : str
        Name in ``GROUPS``.
    scheme : str
        One of ``SCHEMES``.

    Returns
    -------
    float
        1.0 if a positive code is listed, NaN if only an ambiguous code is, else 0.0.
    """
    definition = GROUPS[group]
    label = 0.0
    if codes & definition["positive"][scheme]:
        label = 1.0
    elif codes & definition["ambiguous"][scheme]:
        label = np.nan
    return label


def label_table(code_sets: Iterable[set[str]], scheme: str, sources: Iterable[str] | None = None
                ) -> pd.DataFrame:
    """
    Labels of many records for every finding group.

    Parameters
    ----------
    code_sets : Iterable[set[str]]
        Codes of each record.
    scheme : str
        One of ``SCHEMES``.
    sources : Iterable[str] | None
        Source of each record; a source in ``PARTIAL_ANNOTATION`` is NaN for the groups it does not annotate.

    Returns
    -------
    pd.DataFrame
        One column per group, one row per record, with 1.0, 0.0 or NaN.
    """
    code_sets = list(code_sets)
    sources = list(sources) if sources is not None else [""] * len(code_sets)
    rows = []
    for codes, source in zip(code_sets, sources, strict=True):
        annotated = PARTIAL_ANNOTATION.get(source, frozenset(GROUPS))
        rows.append({group: finding_label(codes, group, scheme) if group in annotated else np.nan
                     for group in GROUPS})
    return pd.DataFrame(rows, columns=list(GROUPS))


def frequent_pvc(aha_code: str) -> bool:
    """
    Whether an SPH record lists PVCs as frequent, in couplets, or in a bigeminal or trigeminal pattern.

    Parameters
    ----------
    aha_code : str
        The ``AHA_Code`` field, such as ``"22;60+310"``.

    Returns
    -------
    bool
        True when a statement with base code 60 carries one of ``FREQUENT_PVC_MODIFIERS``.
    """
    statements = [statement.split("+") for statement in aha_code.split(";")]
    return any(parts[0] == "60" and FREQUENT_PVC_MODIFIERS & set(parts[1:]) for parts in statements)


def clear_normal(standard: np.ndarray, labels: pd.DataFrame) -> np.ndarray:
    """
    Mark the normal ECGs for a budget threshold: standard negatives positive in no group.

    Parameters
    ----------
    standard : np.ndarray
        The binary standard label (1, 0 or NaN) of each record.
    labels : pd.DataFrame
        Output of ``label_table`` for the same records. NaN from ``PARTIAL_ANNOTATION`` counts as not
        positive.

    Returns
    -------
    np.ndarray
        Boolean mask.
    """
    positive = (labels == 1).any(axis=1).to_numpy()
    return (np.asarray(standard, dtype=np.float64) == 0) & ~positive


def budget_sensitivity(scores: np.ndarray, y: np.ndarray, normal: np.ndarray, per_mille: int) -> float:
    """
    Share of positives above the referral-budget threshold set on the normal ECGs of the same set.

    Parameters
    ----------
    scores : np.ndarray
        Score of each record; higher means more likely positive.
    y : np.ndarray
        Binary finding label.
    normal : np.ndarray
        Boolean mask of the normal ECGs that set the threshold.
    per_mille : int
        Budget in thousandths.

    Returns
    -------
    float
        Sensitivity at the threshold (strictly above).
    """
    threshold = budget_threshold(scores[normal], per_mille)
    return float(np.mean(scores[y == 1] > threshold))


def interval(values: np.ndarray) -> dict[str, float]:
    """
    Take the 2.5th and 97.5th percentiles of bootstrap values.

    Parameters
    ----------
    values : np.ndarray
        One value per resample.

    Returns
    -------
    dict[str, float]
        ``ci_low`` and ``ci_high``.
    """
    low, high = np.percentile(values, [2.5, 97.5])
    return {"ci_low": float(low), "ci_high": float(high)}


def set_statistics(y: np.ndarray, units: np.ndarray, normal: np.ndarray, scores: dict[str, np.ndarray],
                   pairs: list[tuple[str, str]], budgets: tuple[int, ...], draws: int, seed: int
                   ) -> dict[str, object]:
    """
    AUROC, AP and budget sensitivity of several scores on one set, with paired unit bootstrap intervals.

    Every score is resampled on the same units. In each resample the budget threshold is recomputed from the
    resampled normals.

    Parameters
    ----------
    y : np.ndarray
        Binary finding label of each record.
    units : np.ndarray
        Bootstrap unit (patient or record) of each record.
    normal : np.ndarray
        Boolean mask of the normal ECGs that set the budget thresholds.
    scores : dict[str, np.ndarray]
        Scores by name.
    pairs : list[tuple[str, str]]
        Contrasts as (first, second): first minus second.
    budgets : tuple[int, ...]
        Budgets in thousandths.
    draws : int
        Bootstrap resamples.
    seed : int
        Seed of the resampling generator.

    Returns
    -------
    dict[str, object]
        ``records``, ``positives``, ``normals``, ``units``, ``invalid_draws``, and per score its ``metrics``
        (``auroc``, ``average_precision`` and ``sensitivity_at_<b>`` with intervals) and per pair the
        differences with intervals.
    """
    names = ["auroc", "average_precision", *(f"sensitivity_at_{budget}" for budget in budgets)]

    def evaluate(rows: np.ndarray) -> dict[str, np.ndarray]:
        found = {}
        for name, values in scores.items():
            ranked = metrics(y[rows], values[rows])
            sensitivities = [budget_sensitivity(values[rows], y[rows], normal[rows], budget)
                             for budget in budgets]
            found[name] = np.array([ranked["auroc"], ranked["average_precision"], *sensitivities])
        return found

    observed = evaluate(np.arange(len(y)))
    resamples, invalid = patient_resamples(units, y, draws, seed)
    drawn = [evaluate(rows) for rows in resamples]
    stacked = {name: np.stack([draw[name] for draw in drawn]) for name in scores}
    per_score = {name: {metric: {"value": float(observed[name][index]), **interval(stacked[name][:, index])}
                        for index, metric in enumerate(names)}
                 for name in scores}
    contrasts = {}
    for first, second in pairs:
        difference = stacked[first] - stacked[second]
        contrasts[f"{first}_minus_{second}"] = {
            metric: {"difference": float(observed[first][index] - observed[second][index]),
                     **interval(difference[:, index])}
            for index, metric in enumerate(names)}
    return {"records": len(y), "positives": int(y.sum()), "normals": int(normal.sum()),
            "units": len(np.unique(units)), "invalid_draws": invalid, "scores": per_score,
            "contrasts": contrasts}


def usability(auroc: dict[str, float], usable: float = 0.90, lower: float = 0.85) -> str:
    """
    Apply the pre-registered reading to one group's primary AUROC.

    Parameters
    ----------
    auroc : dict[str, float]
        ``value``, ``ci_low`` and ``ci_high``.
    usable : float
        Point AUROC required, and the bound the upper limit must stay under for ``not_usable``.
    lower : float
        Lower interval limit required.

    Returns
    -------
    str
        ``usable`` if the AUROC is at least ``usable`` and its lower limit at least ``lower``;
        ``not_usable`` if the upper limit is below ``usable``; else ``undetermined``.
    """
    reading = "undetermined"
    if auroc["value"] >= usable and auroc["ci_low"] >= lower:
        reading = "usable"
    elif auroc["ci_high"] < usable:
        reading = "not_usable"
    return reading
