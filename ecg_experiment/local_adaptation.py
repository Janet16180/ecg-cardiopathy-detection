"""Local recalibration and local normal references at a simulated new site, for Experiment 029.

The new site is SPH, split by patient into a local pool and a fixed evaluation half. Part A refits the
operating point of a frozen readout on a few local labeled ECGs; Part B refits the distance-from-normal
score on a few local normal ECGs. The Platt fit, the at-least-95%-sensitivity threshold and the
Mahalanobis score are the frozen ones of Experiments 027, 027b and 026.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression

from .evaluation import TARGET_SENSITIVITY
from .multisource_calibration import fit_weighted_platt, weighted_threshold
from .normal_manifold import COMPONENTS, mahalanobis_scores
from .screening_threshold import calibrate, fit_platt, screening_threshold

OPTIONS = ("threshold_only", "platt_threshold", "blend", "conservative")
CONFIDENCE = 0.90
SAMPLES_PER_COMPONENT = 4


def split_site(patients: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """
    Split a site by patient into a local pool and an evaluation half, stratified by label.

    A patient's stratum is positive when any of their ECGs is positive. Within each stratum the patients
    are shuffled with one generator (negative stratum first) and the first half, rounded down, goes to
    evaluation.

    Parameters
    ----------
    patients : np.ndarray
        Patient ID of each ECG.
    y : np.ndarray
        Binary label of each ECG.
    seed : int
        Seed of ``numpy.random.default_rng``.

    Returns
    -------
    np.ndarray
        Boolean mask of the ECGs in the evaluation half.
    """
    unique, codes = np.unique(patients, return_inverse=True)
    stratum = np.zeros(len(unique), dtype=np.int64)
    np.maximum.at(stratum, codes, np.asarray(y, dtype=np.int64))
    rng = np.random.default_rng(seed)
    evaluation = np.zeros(len(unique), dtype=bool)
    for label in (0, 1):
        members = rng.permutation(np.flatnonzero(stratum == label))
        evaluation[members[:len(members) // 2]] = True
    return evaluation[codes]


def draw_records(y: np.ndarray, size: int, rng: np.random.Generator,
                 prevalence: float | None = None) -> np.ndarray:
    """
    Draw local ECGs without replacement, at the pool's prevalence or at an assumed one.

    Parameters
    ----------
    y : np.ndarray
        Binary label of each pool ECG.
    size : int
        Number of ECGs drawn.
    rng : np.random.Generator
        Generator of this draw.
    prevalence : float | None
        ``None`` draws a simple random sample. Otherwise the number of positives is drawn from
        Binomial(size, prevalence) and that many positives and ``size`` minus that many negatives are
        drawn separately.

    Returns
    -------
    np.ndarray
        Sorted pool positions of the drawn ECGs.
    """
    if prevalence is None:
        return np.sort(rng.choice(len(y), size=size, replace=False))
    positives = int(rng.binomial(size, prevalence))
    chosen = np.concatenate([rng.choice(np.flatnonzero(y == 1), size=positives, replace=False),
                             rng.choice(np.flatnonzero(y == 0), size=size - positives, replace=False)])
    return np.sort(chosen)


def tolerance_rank(positives: int, target: float = TARGET_SENSITIVITY, confidence: float = CONFIDENCE) -> int:
    """
    Largest rank of a positive score that keeps sensitivity at the target with the given confidence.

    With the threshold at the ``r``-th lowest of ``k`` positive scores, the share of new positives below it
    is Beta(r, k - r + 1), so sensitivity is at least ``target`` with probability
    P(Binomial(k, 1 - target) >= r). The rank is the largest ``r`` for which that probability reaches
    ``confidence``.

    Parameters
    ----------
    positives : int
        Number of local positives ``k``.
    target : float
        Target sensitivity.
    confidence : float
        Required probability.

    Returns
    -------
    int
        The rank ``r``, or 0 when even the lowest positive score gives less than ``confidence``.
    """
    ranks = np.arange(1, positives + 1)
    meets = binom.sf(ranks - 1, positives, 1 - target) >= confidence - 1e-12
    return int(ranks[meets][-1]) if meets.any() else 0


def conservative_threshold(y: np.ndarray, calibrated: np.ndarray) -> tuple[float, bool]:
    """
    Distribution-free threshold that keeps 95% sensitivity with 90% confidence when enough positives exist.

    Parameters
    ----------
    y : np.ndarray
        Binary labels with at least one positive.
    calibrated : np.ndarray
        Calibrated probabilities.

    Returns
    -------
    tuple[float, bool]
        The threshold, and whether the confidence is reached. Without it, the lowest positive is used.
    """
    positive = np.sort(calibrated[np.asarray(y) == 1])
    rank = tolerance_rank(len(positive))
    return float(positive[max(rank, 1) - 1]), rank > 0


def blend_weights(source: int, local: int) -> np.ndarray:
    """
    Weights that give the source and the local calibration ECGs the same total weight.

    Parameters
    ----------
    source : int
        Number of source calibration ECGs, weighted 1 each.
    local : int
        Number of local ECGs, weighted ``source / local`` each.

    Returns
    -------
    np.ndarray
        Source weights followed by local weights.
    """
    return np.concatenate([np.ones(source), np.full(local, source / local)])


def adapt(option: str, local_logits: np.ndarray, local_y: np.ndarray, source_logits: np.ndarray,
          source_y: np.ndarray, source: tuple[LogisticRegression, float]
          ) -> tuple[LogisticRegression, float, bool, bool]:
    """
    Operating point of one adaptation option from one local draw.

    Parameters
    ----------
    option : str
        One of ``OPTIONS``.
    local_logits, local_y : np.ndarray
        Head logits and labels of the local draw.
    source_logits, source_y : np.ndarray
        Head logits and labels of the readout's source calibration ECGs.
    source : tuple[LogisticRegression, float]
        The source Platt mapping and threshold (the n = 0 operating point).

    Returns
    -------
    tuple[LogisticRegression, float, bool, bool]
        Calibrator, threshold, whether the draw fell back to the source operating point, and (for
        ``conservative``) whether the 90% confidence was reached.

    Raises
    ------
    ValueError
        If the option is unknown.
    """
    if option not in OPTIONS:
        raise ValueError(f"Unknown option: {option}")
    positives, negatives = int(local_y.sum()), int(len(local_y) - local_y.sum())
    calibrator, threshold = source
    fallback, confident = False, False
    if option == "threshold_only" and positives:
        threshold = screening_threshold(local_y, calibrate(calibrator, local_logits))
    elif option == "conservative" and positives:
        threshold, confident = conservative_threshold(local_y, calibrate(calibrator, local_logits))
    elif option == "platt_threshold" and positives and negatives:
        try:
            calibrator = fit_platt(local_logits, local_y)
        except RuntimeError:
            fallback = True
        else:
            threshold = screening_threshold(local_y, calibrate(calibrator, local_logits))
    elif option == "blend":
        logits = np.concatenate([source_logits, local_logits])
        y = np.concatenate([source_y, local_y])
        weights = blend_weights(len(source_y), len(local_y))
        try:
            calibrator = fit_weighted_platt(logits, y, weights)
        except RuntimeError:
            fallback = True
        else:
            threshold = weighted_threshold(y, calibrate(calibrator, logits), weights)
    else:
        fallback = True
    return calibrator, threshold, fallback, confident


def screening_rates(y: np.ndarray, referred: np.ndarray, prevalence: float) -> dict[str, float]:
    """
    Sensitivity, specificity and referrals per 1,000 of fixed referral decisions.

    Parameters
    ----------
    y : np.ndarray
        Binary labels with both classes.
    referred : np.ndarray
        Boolean referral decision of each ECG.
    prevalence : float
        Assumed prevalence for the second referral rate.

    Returns
    -------
    dict[str, float]
        ``sensitivity``, ``specificity``, ``referrals_per_1000`` at the observed prevalence and
        ``referrals_per_1000_assumed`` at ``prevalence``.
    """
    positive = np.asarray(y) == 1
    sensitivity = float(referred[positive].mean())
    specificity = float((~referred[~positive]).mean())
    return {"sensitivity": sensitivity, "specificity": specificity,
            "referrals_per_1000": 1000 * float(referred.mean()),
            "referrals_per_1000_assumed": 1000 * (prevalence * sensitivity
                                                  + (1 - prevalence) * (1 - specificity))}


def local_components(normals: int) -> int:
    """
    PCA components of a normal reference fitted on local normals only.

    Parameters
    ----------
    normals : int
        Number of local normal ECGs ``m``.

    Returns
    -------
    int
        ``min(64, m // 4)``, so there are at least four normals per component.
    """
    return min(COMPONENTS, normals // SAMPLES_PER_COMPONENT)


def recentered_scores(model: tuple, x: np.ndarray, local_mean: np.ndarray) -> np.ndarray:
    """
    Distance from a fixed normal reference after moving the local normal mean onto the reference mean.

    Parameters
    ----------
    model : tuple
        Output of ``normal_manifold.fit_mahalanobis`` on the non-local reference.
    x : np.ndarray
        Features to score.
    local_mean : np.ndarray
        Mean feature vector of the local normals.

    Returns
    -------
    np.ndarray
        Squared Mahalanobis distance of ``x - local_mean + reference mean``.
    """
    scaler = model[0]
    return mahalanobis_scores(model, np.asarray(x, dtype=np.float64) - local_mean + scaler.mean_)
