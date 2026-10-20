"""Exact signed contributions to frozen contextual reference-classifier margins."""

from __future__ import annotations

import hashlib
import math

import numpy as np
from scipy.stats import spearmanr

from ecg_experiment.native_reference_ntt05 import (
    NativeTokens,
    SignalRegion,
    pool_region,
    unit_vector,
)

FRACTIONS = (0.10, 0.25, 0.50)


def occupancy(tokens: NativeTokens, regions: list[SignalRegion]) -> np.ndarray:
    """Return normalized occupancy weights after frozen lead/time coverage validation."""
    rows = []
    for region in regions:
        pool_region(tokens, region)
        overlap = np.maximum(
            0,
            np.minimum(tokens.intervals[:, 1], region.stop)
            - np.maximum(tokens.intervals[:, 0], region.start),
        )
        compatible = np.array([set(leads) <= set(region.leads) for leads in tokens.token_leads])
        weights = compatible * overlap / np.diff(tokens.intervals, axis=1).ravel()
        rows.append(weights / weights.sum())
    if not rows:
        raise ValueError("At least one complete pooling region is required")
    return np.stack(rows)


def decomposition(tokens: NativeTokens, regions: list[SignalRegion], prototypes: np.ndarray) -> dict:
    """Decompose the fixed original winner-minus-runner margin through both normalizations.

    Parameters
    ----------
    tokens : NativeTokens
        Frozen contextual vectors and their original lead/time anchors.
    regions : list of SignalRegion
        Unchanged automatic QRS pooling regions.
    prototypes : np.ndarray
        Finding-by-feature unit reference prototypes in the identical embedding space.

    Returns
    -------
    dict
        Private vectors, normalized weights, signed contributions and fixed contrast.
        These contextual representation contributions are not waveform pathology labels.
    """
    prototypes = np.asarray(prototypes, dtype=float)
    if (
        prototypes.ndim != 2
        or prototypes.shape[0] < 2
        or prototypes.shape[1] != tokens.values.shape[1]
        or not np.isfinite(prototypes).all()
        or not np.allclose(np.linalg.norm(prototypes, axis=1), 1, atol=1e-12, rtol=0)
    ):
        raise ValueError("Expected at least two finite unit prototypes in the token feature space")
    weights = occupancy(tokens, regions)
    means = weights @ np.asarray(tokens.values, dtype=float)
    norms = np.linalg.norm(means, axis=1)
    if np.any(norms == 0) or not np.isfinite(norms).all():
        raise ValueError("Undefined region normalization")
    beats = means / norms[:, None]
    patient = unit_vector(beats.mean(axis=0))
    aggregate_norm = float(np.linalg.norm(beats.sum(axis=0)))
    scores = np.clip(prototypes @ patient, -1, 1)
    winner, runner = np.argsort(-scores, kind="stable")[:2]
    delta = prototypes[winner] - prototypes[runner]
    beat_contributions = beats @ delta / aggregate_norm
    token_dot = np.asarray(tokens.values, dtype=float) @ delta
    beat_tokens = weights * token_dot[None, :] / (norms[:, None] * aggregate_norm)
    return {
        "weights": weights,
        "region_norms": norms,
        "aggregate_norm": aggregate_norm,
        "beat_vectors": beats,
        "patient_vector": patient,
        "scores": scores,
        "winner": int(winner),
        "runner": int(runner),
        "delta": delta,
        "margin": float(scores[winner] - scores[runner]),
        "beat_contributions": beat_contributions,
        "beat_token_contributions": beat_tokens,
        "token_contributions": beat_tokens.sum(axis=0),
        "active_tokens": np.any(weights > 0, axis=0),
        "pooling_maximum_error": max(
            float(np.max(np.abs(vector - pool_region(tokens, region))))
            for vector, region in zip(beats, regions, strict=True)
        ),
    }


def summary(values: np.ndarray) -> dict:
    """Summarize finite observations while retaining the count of undefined values."""
    values = np.asarray(values, dtype=float)
    valid = values[np.isfinite(values)]
    result = {
        "count": int(values.size),
        "defined": int(valid.size),
        "undefined": int(values.size - valid.size),
    }
    for name, percentile in (("minimum", 0), ("p10", 10), ("median", 50), ("p90", 90), ("maximum", 100)):
        result[name] = float(np.percentile(valid, percentile)) if valid.size else None
    result["mean"] = float(valid.mean()) if valid.size else None
    return result


def mass_summary(contributions: np.ndarray) -> dict:
    """Describe signed cancellation and concentration using absolute mass, never margin shares."""
    contributions = np.asarray(contributions, dtype=float)
    if contributions.ndim != 1 or not contributions.size or not np.isfinite(contributions).all():
        raise ValueError("Expected a nonempty finite contribution vector")
    absolute = np.abs(contributions)
    mass = float(absolute.sum())
    order = np.argsort(-absolute, kind="stable")
    return {
        "count": len(contributions),
        "positive_mass": float(np.maximum(contributions, 0).sum()),
        "negative_mass": float(np.maximum(-contributions, 0).sum()),
        "absolute_mass": mass,
        "cancellation": float(1 - abs(contributions.sum()) / mass) if mass else None,
        "effective_count": float(mass**2 / np.square(absolute).sum()) if mass else None,
        "effective_fraction": float(mass**2 / np.square(absolute).sum() / len(absolute)) if mass else None,
        "top_absolute_shares": {
            str(fraction): float(absolute[order[: math.ceil(fraction * len(absolute))]].sum() / mass)
            if mass
            else 0.0
            for fraction in FRACTIONS
        },
    }


def removed_margin(beats: np.ndarray, delta: np.ndarray, removed: np.ndarray) -> float | None:
    """Remove whole cached beat vectors and renormalize while preserving the original contrast."""
    beats, delta, removed = np.asarray(beats), np.asarray(delta), np.asarray(removed)
    if (
        beats.ndim != 2
        or delta.shape != (beats.shape[1],)
        or removed.ndim != 1
        or not np.issubdtype(removed.dtype, np.integer)
        or len(np.unique(removed)) != len(removed)
        or np.any((removed < 0) | (removed >= len(beats)))
        or not np.isfinite(beats).all()
        or not np.isfinite(delta).all()
    ):
        raise ValueError("Invalid finite beat intervention or removed indices")
    keep = np.ones(len(beats), dtype=bool)
    keep[removed] = False
    vector = beats[keep].sum(axis=0)
    norm = np.linalg.norm(vector)
    return float(delta @ vector / norm) if norm else None


def rank_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    """Return average-rank Spearman correlation or null for undefined/constant comparisons."""
    first, second = np.asarray(first), np.asarray(second)
    finite = np.isfinite(first) & np.isfinite(second)
    first, second = first[finite], second[finite]
    if len(first) < 2 or np.ptp(first) == 0 or np.ptp(second) == 0:
        return None
    return float(spearmanr(first, second).statistic)


def intervention_summary(result: dict, ecg: int, seed: int) -> dict:
    """Compare selected and prespecified random cached-beat deletions, with undefined counts."""
    beats, delta, margin = result["beat_vectors"], result["delta"], result["margin"]
    single = [removed_margin(beats, delta, np.array([index])) for index in range(len(beats))]
    drops = np.array([margin - value if value is not None else np.nan for value in single])
    order = np.argsort(-result["beat_contributions"], kind="stable")
    fractions = {}
    for fraction in FRACTIONS:
        count = max(1, min(len(beats) - 1, math.ceil(fraction * len(beats))))
        selected = removed_margin(beats, delta, order[:count])
        digest = hashlib.sha256(f"ntt08:20261020:{ecg}:{seed}:{fraction:.2f}".encode()).hexdigest()
        rng = np.random.default_rng(int(digest, 16))
        removed = [rng.choice(len(beats), count, replace=False) for _ in range(100)]
        margins = [removed_margin(beats, delta, indices) for indices in removed]
        random_drops = np.array([margin - value if value is not None else np.nan for value in margins])
        random_mean = summary(random_drops)["mean"]
        selected_drop = margin - selected if selected is not None else None
        fractions[str(fraction)] = {
            "removed_count": count,
            "selected_indices": order[:count].tolist(),
            "selected_margin_drop": selected_drop,
            "random_drop": summary(random_drops),
            "selected_minus_random_mean": selected_drop - random_mean
            if selected_drop is not None and random_mean is not None
            else None,
            "random_subsets": [indices.tolist() for indices in removed],
            "random_margins": margins,
        }
    return {
        "single_beat_drops": [float(value) if np.isfinite(value) else None for value in drops],
        "single_negative_drops": int(np.sum(drops < 0)),
        "single_undefined_drops": int(np.sum(~np.isfinite(drops))),
        "contribution_drop_spearman": rank_correlation(result["beat_contributions"], drops),
        "fractions": fractions,
    }


def bank_consistency(results: list[dict]) -> dict:
    """Compare original bank pairs only when their ordered finding contrasts agree."""
    pairs = []
    for left in range(len(results)):
        for right in range(left + 1, len(results)):
            first, second = results[left], results[right]
            same = (first["winner"], first["runner"]) == (second["winner"], second["runner"])
            row = {"left": left, "right": right, "same_contrast": same}
            if same:
                first_values, second_values = first["beat_contributions"], second["beat_contributions"]
                count = max(1, math.ceil(len(first_values) * 0.25))
                first_top = {
                    int(index)
                    for index in np.argsort(-first_values, kind="stable")[:count]
                    if first_values[index] > 0
                }
                second_top = {
                    int(index)
                    for index in np.argsort(-second_values, kind="stable")[:count]
                    if second_values[index] > 0
                }
                union = first_top | second_top
                row.update(
                    signed_rank_spearman=rank_correlation(first_values, second_values),
                    top_quarter_jaccard=len(first_top & second_top) / len(union) if union else None,
                )
            pairs.append(row)
    return {"pairs": pairs, "same_contrast": sum(row["same_contrast"] for row in pairs)}
