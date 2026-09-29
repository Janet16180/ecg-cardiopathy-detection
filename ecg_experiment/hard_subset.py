"""Label rules, weights and decision helpers for Experiment 035, the PTB-XL hard added subset.

Experiment 022b's pooled readout labels the Challenge sources with the primary rule of
``challenge_labels`` (positive for any MI, STTC, CD or HYP code, negative for sinus rhythm alone). The rules
here move that mapping toward the PTB-XL standard label, which counts a record negative when NORM is its
only diagnostic superclass, whatever rhythm or form statements accompany it. The weights and indicators
change how the Challenge rows enter the same logistic readout.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import numpy as np

from .challenge_labels import SINUS_RHYTHM, SINUS_VARIANTS, SUPERCLASSES

SINUS_CODES = frozenset({SINUS_RHYTHM, *SINUS_VARIANTS})
SUPERCLASS_CODES = frozenset().union(*SUPERCLASSES.values())
# Challenge codes whose PTB-XL counterpart is a non-diagnostic rhythm or form statement that PTB-XL lists
# next to NORM: ventricular premature beats (PVC), atrial premature beats (PAC), low QRS voltage (LVOLT),
# left ventricular high voltage (VCLVH, HVOLT), abnormal Q waves (QWAVE) and axis deviation (the
# heart_axis column, not a statement).
PTBXL_COMPATIBLE = frozenset({"427172004", "17338001", "164884008", "284470004", "63593006", "251146004",
                              "55827005", "164917005", "39732003", "47665007"})
# Challenge ST, T and PR codes that the mapping counts as STTC or CD, while their PTB-XL counterparts are
# form statements only (STD_, STE_, INVT, TAB_ with NT_ and LOWT, LPR), which the standard label ignores.
FORM_CODES = frozenset({"429622005", "164931005", "59931005", "164934002", "164947007"})
LABEL_RULES = ("primary", "secondary", "ptbxl_negative", "form_ignored", "ptbxl_rule")
PTBXL_SINUS_VARIANTS = frozenset({"SBRAD", "STACH", "SARRH"})
PTBXL_FORM_AS_CHALLENGE = {"STD_": "STTC", "STE_": "STTC", "INVT": "STTC", "TAB_": "STTC", "NT_": "STTC",
                           "LOWT": "STTC", "LPR": "CD"}
CHALLENGE_FAMILIES = ("chapman_ningbo", "georgia", "cpsc")
# Each arm changes one thing relative to 022b's ``pooled``: the Challenge families, the Challenge label
# rule, a source indicator, row weights, or the PTB-XL rows the project label dropped.
ARM_SPECS: dict[str, dict[str, object]] = {
    "ptbxl": {"rule": None},
    "pooled": {"rule": "primary"},
    **{f"loso_{family}": {"rule": "primary",
                          "families": tuple(name for name in CHALLENGE_FAMILIES if name != family)}
       for family in CHALLENGE_FAMILIES},
    **{f"ptbxl_{family}": {"rule": "primary", "families": (family,)} for family in CHALLENGE_FAMILIES},
    "secondary_negative": {"rule": "secondary"},
    "ptbxl_negative": {"rule": "ptbxl_negative"},
    "form_ignored": {"rule": "form_ignored"},
    "ptbxl_rule": {"rule": "ptbxl_rule"},
    "source_indicator": {"rule": "primary", "indicator": True},
    "prevalence_matched": {"rule": "primary", "weights": "prevalence"},
    "ptbxl_half": {"rule": "primary", "weights": "ptbxl_half"},
    "dropped_upweighted": {"rule": "primary", "weights": "dropped_share"},
    "without_dropped": {"rule": "primary", "without_dropped": True},
}


def rule_label(codes: Iterable[str], rule: str) -> float:
    """
    Binary label of one Challenge record under one label rule.

    Parameters
    ----------
    codes : Iterable[str]
        The record's SNOMED codes.
    rule : str
        One of ``LABEL_RULES``:

        - ``primary``: the mapping's primary label (negative for sinus rhythm alone);
        - ``secondary``: negative also with the benign sinus variants;
        - ``ptbxl_negative``: negative when no superclass code is listed, a sinus code is, and every code is
          a sinus code or in ``PTBXL_COMPATIBLE``;
        - ``form_ignored``: the codes in ``FORM_CODES`` no longer count toward a superclass; negative stays
          sinus rhythm alone, so a record positive only through them becomes undefined;
        - ``ptbxl_rule``: ``form_ignored`` positives and ``ptbxl_negative`` negatives that may also list
          ``FORM_CODES``.

    Returns
    -------
    float
        1.0, 0.0, or NaN when the rule leaves the record undefined.

    Raises
    ------
    ValueError
        If ``rule`` is unknown.
    """
    if rule not in LABEL_RULES:
        raise ValueError(f"Unknown label rule: {rule}")
    listed = set(codes)
    ignored = FORM_CODES if rule in ("form_ignored", "ptbxl_rule") else frozenset()
    allowed = {"primary": {SINUS_RHYTHM}, "secondary": SINUS_CODES, "form_ignored": {SINUS_RHYTHM},
               "ptbxl_negative": SINUS_CODES | PTBXL_COMPATIBLE,
               "ptbxl_rule": SINUS_CODES | PTBXL_COMPATIBLE | FORM_CODES}[rule]
    needs_sinus = rule in ("ptbxl_negative", "ptbxl_rule")
    label = np.nan
    if listed & (SUPERCLASS_CODES - ignored):
        label = 1.0
    elif listed and listed <= allowed and (listed & SINUS_CODES or not needs_sinus):
        label = 0.0
    return label


def rule_labels(code_lists: Iterable[Iterable[str]], rule: str) -> np.ndarray:
    """
    Labels of many Challenge records under one rule.

    Parameters
    ----------
    code_lists : Iterable[Iterable[str]]
        SNOMED codes of each record.
    rule : str
        One of ``LABEL_RULES``.

    Returns
    -------
    np.ndarray
        Float labels, NaN where undefined.
    """
    return np.array([rule_label(codes, rule) for codes in code_lists], dtype=np.float64)


def ptbxl_under_challenge(codes: dict[str, float], classes: dict[str, str]) -> dict[str, float]:
    """
    Label a PTB-XL record by the Challenge mapping's rules.

    PTB-XL's NORM has no Challenge code, so it is dropped; diagnostic codes keep their superclass, the ST, T
    and PR form statements count as the Challenge's STTC and CD codes do, and every other rhythm or form
    statement is an ``other`` code.

    Parameters
    ----------
    codes : dict[str, float]
        SCP code to likelihood for one record.
    classes : dict[str, str]
        ``eda.ptbxl.diagnostic_classes`` output.

    Returns
    -------
    dict[str, float]
        ``primary`` and ``secondary`` Challenge-style labels (1, 0 or NaN).
    """
    listed = set(codes) - {"NORM"}
    positive = any(classes.get(code, "NORM") != "NORM" for code in listed) or \
        bool(listed & set(PTBXL_FORM_AS_CHALLENGE))
    primary = secondary = np.nan
    if positive:
        primary = secondary = 1.0
    elif listed == {"SR"}:
        primary = secondary = 0.0
    elif listed and listed <= {"SR", *PTBXL_SINUS_VARIANTS}:
        secondary = 0.0
    return {"primary": primary, "secondary": secondary}


def code_counts(code_lists: Iterable[Iterable[str]]) -> dict[str, int]:
    """
    Count the records listing each code, most common first.

    Parameters
    ----------
    code_lists : Iterable[Iterable[str]]
        Codes of each record; a code counts once per record.

    Returns
    -------
    dict[str, int]
        Records per code.
    """
    counts = Counter()
    for codes in code_lists:
        counts.update(set(codes))
    return dict(counts.most_common())


def prevalence_weights(families: np.ndarray, y: np.ndarray, target: float,
                       reference: str = "ptbxl") -> np.ndarray:
    """
    Weights that give every non-reference family the target prevalence and its own total weight.

    Parameters
    ----------
    families : np.ndarray
        Family of each row.
    y : np.ndarray
        Binary labels.
    target : float
        Weighted share of positives wanted in each non-reference family, in (0, 1).
    reference : str
        Family whose rows keep weight 1.

    Returns
    -------
    np.ndarray
        Weight of each row.

    Raises
    ------
    ValueError
        If a non-reference family lacks a class.
    """
    weights = np.ones(len(y), dtype=np.float64)
    for family in np.unique(families):
        if family == reference:
            continue
        rows = families == family
        positives = np.count_nonzero(y[rows] == 1)
        negatives = np.count_nonzero(rows) - positives
        if not positives or not negatives:
            raise ValueError(f"Family {family} lacks a class")
        weights[rows & (y == 1)] = target * np.count_nonzero(rows) / positives
        weights[rows & (y == 0)] = (1 - target) * np.count_nonzero(rows) / negatives
    return weights


def share_weights(selected: np.ndarray, share: float) -> np.ndarray:
    """
    Weights averaging 1 under which the selected rows carry a given share of the total weight.

    Parameters
    ----------
    selected : np.ndarray
        Boolean selection.
    share : float
        Share of the total weight wanted for the selected rows, in (0, 1).

    Returns
    -------
    np.ndarray
        Weight of each row.

    Raises
    ------
    ValueError
        If either part is empty.
    """
    selected = np.asarray(selected, dtype=bool)
    count = np.count_nonzero(selected)
    if not count or count == len(selected):
        raise ValueError("Both parts need rows")
    total = len(selected)
    return np.where(selected, share * total / count, (1 - share) * total / (total - count))


def source_indicators(families: np.ndarray, names: Iterable[str]) -> np.ndarray:
    """
    One 0/1 column per named family; rows of any other family are all zero.

    Parameters
    ----------
    families : np.ndarray
        Family of each row.
    names : Iterable[str]
        Families that get a column, in column order.

    Returns
    -------
    np.ndarray
        Float64 array of shape ``(rows, len(names))``.
    """
    return np.column_stack([(families == name).astype(np.float64) for name in names])


def patient_folds(patients: np.ndarray, folds: int, seed: int) -> np.ndarray:
    """
    Fold of each row, assigning whole patients to folds.

    The unique patients (``np.unique`` order) are permuted by ``numpy.random.default_rng(seed)`` and cut
    into ``folds`` nearly equal parts with ``np.array_split``.

    Parameters
    ----------
    patients : np.ndarray
        Patient ID of each row.
    folds : int
        Number of folds.
    seed : int
        Seed of the permutation.

    Returns
    -------
    np.ndarray
        Fold index (0 to ``folds - 1``) of each row.
    """
    unique, codes = np.unique(patients, return_inverse=True)
    order = np.random.default_rng(seed).permutation(len(unique))
    fold_of_patient = np.empty(len(unique), dtype=np.int64)
    for fold, members in enumerate(np.array_split(order, folds)):
        fold_of_patient[members] = fold
    return fold_of_patient[codes]


def recovery(arm: float, pooled: float, ptbxl: float) -> float:
    """
    Share of the pooled readout's loss against the PTB-XL-only readout that an arm recovers.

    Parameters
    ----------
    arm, pooled, ptbxl : float
        AUROC of the arm, the pooled readout and the PTB-XL-only readout on the same ECGs.

    Returns
    -------
    float
        ``(arm - pooled) / (ptbxl - pooled)``; NaN when there is no loss to recover.
    """
    loss = ptbxl - pooled
    return (arm - pooled) / loss if loss > 0 else float("nan")


def arm_verdict(hard: dict[str, float], sph: dict[str, float], hard_recovery: float,
                minimum_recovery: float = 0.5, sph_tolerance: float = 0.005) -> str:
    """
    Classify one arm by the prespecified rule on the hard subset and SPH.

    Parameters
    ----------
    hard : dict[str, float]
        Hard-subset AUROC difference, arm minus pooled, with ``difference``, ``ci_low`` and ``ci_high``.
    sph : dict[str, float]
        SPH AUROC difference, arm minus pooled, with ``difference``.
    hard_recovery : float
        Output of ``recovery`` on the hard subset.
    minimum_recovery : float
        Share of the loss an arm must recover.
    sph_tolerance : float
        Largest SPH AUROC loss allowed.

    Returns
    -------
    str
        ``fix`` (recovers enough, interval above 0, SPH kept), ``trade_off`` (recovers enough with the
        interval above 0 but loses SPH), ``suggestive`` (recovers enough, interval includes 0, SPH kept) or
        ``no_recovery``.
    """
    enough = bool(hard_recovery >= minimum_recovery)
    established = hard["ci_low"] > 0
    keeps_sph = sph["difference"] >= -sph_tolerance
    verdict = "no_recovery"
    if enough and established and keeps_sph:
        verdict = "fix"
    elif enough and established:
        verdict = "trade_off"
    elif enough and keeps_sph:
        verdict = "suggestive"
    return verdict


def arm_design(spec: dict[str, object], families: np.ndarray, labels: dict[str, np.ndarray],
               dropped: np.ndarray) -> dict[str, np.ndarray | None]:
    """
    Training rows, labels and weights of one arm over stacked PTB-XL and Challenge rows.

    Parameters
    ----------
    spec : dict[str, object]
        One value of ``ARM_SPECS``.
    families : np.ndarray
        ``ptbxl`` or the Challenge family of each row.
    labels : dict[str, np.ndarray]
        Label of each row per rule in ``LABEL_RULES`` (NaN where undefined); PTB-XL rows carry their
        standard label under every rule.
    dropped : np.ndarray
        PTB-XL rows without a project label.

    Returns
    -------
    dict[str, np.ndarray | None]
        ``selected`` rows, their integer ``y``, ``weights`` (``None`` for an unweighted fit) and
        ``indicators`` (``None`` without a source indicator), both aligned with the selected rows.

    Raises
    ------
    ValueError
        If a PTB-XL row lacks a label.
    """
    ptbxl = families == "ptbxl"
    rule = spec["rule"] or "primary"
    y_all = labels[rule]
    if np.isnan(y_all[ptbxl]).any():
        raise ValueError("Every PTB-XL row needs its standard label")
    chosen = spec.get("families", CHALLENGE_FAMILIES) if spec["rule"] else ()
    selected = (ptbxl & ~(dropped & bool(spec.get("without_dropped")))) | \
        (np.isin(families, list(chosen)) & ~np.isnan(y_all))
    y = y_all[selected].astype(np.int64)
    kind = spec.get("weights")
    weights = None
    if kind == "prevalence":
        target = float(np.mean(y_all[ptbxl]))
        weights = prevalence_weights(families[selected], y, target)
    elif kind == "ptbxl_half":
        weights = share_weights(ptbxl[selected], 0.5)
    elif kind == "dropped_share":
        weights = share_weights(dropped[selected], float(np.count_nonzero(dropped) / np.count_nonzero(ptbxl)))
    indicators = source_indicators(families[selected], CHALLENGE_FAMILIES) if spec.get("indicator") else None
    return {"selected": selected, "y": y, "weights": weights, "indicators": indicators}
