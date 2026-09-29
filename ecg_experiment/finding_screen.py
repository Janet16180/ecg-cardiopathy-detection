"""A referral screen that splits one budget between the binary readout and finding heads, for Experiment 033.

An ECG is referred when the binary readout or any finding score lies strictly above its own threshold. All
thresholds come from the same ``m`` normal ECGs. With ``k = floor(b m)``, each finding score gets a fixed
share of ``k`` as its rank, and the binary readout gets the largest rank at which at most ``k - F`` of the
normals lie above any threshold, ``F`` being the number of finding scores. Leaving one normal per extra
threshold keeps the expected false-referral rate of new normals at that of the binary readout alone.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logit

from .hybrid_score import normal_standardizer, standardize
from .local_adaptation import split_site
from .referral_budget import budget_threshold

CLIP = 1e-15


def clipped_logits(probabilities: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Logits of probabilities clipped to ``[1e-15, 1 - 1e-15]``.

    Parameters
    ----------
    probabilities : np.ndarray
        Probabilities of one head.

    Returns
    -------
    tuple[np.ndarray, int]
        Float64 logits and the number of clipped values.

    Raises
    ------
    ValueError
        If a probability is not finite or lies outside ``[0, 1]``.
    """
    values = np.asarray(probabilities, dtype=np.float64)
    if not (np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()):
        raise ValueError("Probabilities must be finite and within [0, 1]")
    clipped = np.clip(values, CLIP, 1 - CLIP)
    return logit(clipped), int((clipped != values).sum())


def finding_z(probabilities: np.ndarray, source_normal: np.ndarray) -> np.ndarray:
    """
    Standardized logit of a finding head, with the mean and SD of its source normal ECGs.

    Parameters
    ----------
    probabilities : np.ndarray
        Head probabilities of the ECGs to score.
    source_normal : np.ndarray
        Head probabilities of the source normal ECGs.

    Returns
    -------
    np.ndarray
        z-scores.
    """
    source, _ = clipped_logits(source_normal)
    values, _ = clipped_logits(probabilities)
    return standardize(values, normal_standardizer(source))


def split_thresholds(normals: np.ndarray, per_mille: int, shares: tuple[int, ...]) -> np.ndarray:
    """
    Thresholds of the binary readout and each finding score from one set of normal ECGs.

    Parameters
    ----------
    normals : np.ndarray
        ``(m, 1 + F)`` scores of the normals: the binary readout first, then the ``F`` finding scores.
    per_mille : int
        Total budget ``b`` in thousandths.
    shares : tuple[int, ...]
        Share of ``k = floor(b m)`` for each finding score, in thousandths; its rank is
        ``floor(share k / 1000)``.

    Returns
    -------
    np.ndarray
        ``1 + F`` thresholds. With no finding score, the 030 threshold of the binary readout.

    Raises
    ------
    ValueError
        If the shape and the shares disagree, the budget is out of range, or the finding ranks exceed
        ``k - F``.
    """
    scores = np.asarray(normals, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != 1 + len(shares):
        raise ValueError("Normals need one binary column and one column per finding share")
    if not shares:
        return np.array([budget_threshold(scores[:, 0], per_mille)])
    if not 0 <= per_mille < 1000:
        raise ValueError(f"Budget out of range: {per_mille}")
    limit = per_mille * len(scores) // 1000
    allowance = limit - len(shares)
    ranks = [share * limit // 1000 for share in shares]
    if sum(ranks) > allowance:
        raise ValueError("Finding ranks exceed the budget left after one normal per extra threshold")
    findings = np.array([np.sort(scores[:, 1 + column])[::-1][rank] for column, rank in enumerate(ranks)])
    above_finding = (scores[:, 1:] > findings).any(axis=1)
    binary = scores[:, 0]
    ordered = np.sort(binary)[::-1]
    rank = allowance
    while ((binary > ordered[rank]) | above_finding).sum() > allowance:
        rank -= 1
    return np.concatenate([[ordered[rank]], findings])


def referred_matrix(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Which ECGs each draw refers: above any of that draw's thresholds.

    Parameters
    ----------
    scores : np.ndarray
        ``(n, 1 + F)`` scores of the ECGs.
    thresholds : np.ndarray
        ``(draws, 1 + F)`` thresholds.

    Returns
    -------
    np.ndarray
        ``(n, draws)`` boolean referrals.
    """
    return (scores[:, None, :] > thresholds[None, :, :]).any(axis=2)


def extend_split(patients: np.ndarray, known: np.ndarray, evaluation: np.ndarray, positive: np.ndarray,
                 seed: int) -> np.ndarray:
    """
    Extend a patient split to ECGs outside it.

    An ECG outside the split joins its patient's half when that patient is in the split. The remaining
    patients are split with ``local_adaptation.split_site``, stratified by ``positive``.

    Parameters
    ----------
    patients : np.ndarray
        Patient ID of each ECG.
    known : np.ndarray
        Boolean mask of the ECGs in the existing split.
    evaluation : np.ndarray
        Evaluation mask of the existing split; read only where ``known``.
    positive : np.ndarray
        Stratum label of each ECG for the new patients.
    seed : int
        Seed of the split of the new patients.

    Returns
    -------
    np.ndarray
        Evaluation mask of every ECG.

    Raises
    ------
    ValueError
        If a patient of the existing split is in both halves.
    """
    patients = np.asarray(patients).astype(str)
    halves: dict[str, bool] = {}
    for patient, half in zip(patients[known], evaluation[known], strict=True):
        if halves.setdefault(patient, bool(half)) != bool(half):
            raise ValueError(f"Patient {patient} is in both halves")
    result = np.asarray(evaluation, dtype=bool) & known
    joined = ~known & np.isin(patients, list(halves))
    result[joined] = [halves[patient] for patient in patients[joined]]
    new = ~known & ~joined
    if new.any():
        result[new] = split_site(patients[new], np.asarray(positive, dtype=np.int64)[new], seed)
    return result
