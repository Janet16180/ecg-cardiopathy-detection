"""Conditional reference-bank sampling, clustering and label-invariant sensitivity summaries."""

from __future__ import annotations

from itertools import combinations, permutations

import numpy as np
from sklearn.metrics import adjusted_rand_score

from ecg_experiment.clustering.reference_geometry import SEED, check_budget, fit_model
from ecg_experiment.clustering02 import balanced, silhouette
from ecg_experiment.native_reference_ntt05 import unit_vector
from ecg_experiment.same_wave_ntt04 import FINDINGS


def support_draws(selection: dict, count: int = 50, seed: int = 20261020) -> list[dict]:
    """Draw distinct complete banks without using support feature values or target outcomes.

    Parameters
    ----------
    selection : dict
        Frozen predecessor reference selections, containing nine patients per finding.
    count, seed : int
        Prospective number of distinct three-shot banks and sampling seed.

    Returns
    -------
    list of dict
        Ordered local banks with class-specific reference identity triples.
    """
    pools = {}
    for finding in FINDINGS:
        rows = {
            int(row["patient_id"]): row
            for group in selection["support_draws"]
            if group["finding"] == finding
            for row in group["supports"]
        }
        pools[finding] = sorted(rows.values(), key=lambda row: (int(row["patient_id"]), int(row["ecg_id"])))
        if len(pools[finding]) != 9:
            raise ValueError("Expected nine distinct support patients per finding")
    if len({row["patient_id"] for pool in pools.values() for row in pool}) != 27:
        raise ValueError("Support classes must have distinct patients")
    if not 1 <= count <= 84**3:
        raise ValueError("Requested bank count outside finite pool")
    rng = np.random.default_rng(seed)
    result, seen = [], set()
    while len(result) < count:
        indices = tuple(tuple(sorted(rng.choice(9, 3, replace=False).tolist())) for _ in FINDINGS)
        if indices in seen:
            continue
        seen.add(indices)
        result.append(
            {
                "bank": len(result),
                "supports": {
                    finding: [pools[finding][i] for i in chosen]
                    for finding, chosen in zip(FINDINGS, indices, strict=True)
                },
            }
        )
    return result


def bank_prototypes(draw: dict, features: dict) -> np.ndarray:
    """Build patient-equal normalized prototypes in the frozen finding order."""
    return np.stack(
        [
            unit_vector(
                np.mean(
                    [
                        unit_vector(features[str(row["ecg_id"])]["native_qrs"])
                        for row in draw["supports"][finding]
                    ],
                    axis=0,
                )
            )
            for finding in FINDINGS
        ]
    )


def bank_affinities(cache: dict, prototypes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute one bank's record cosines and per-patient mean/max/std region coordinates."""
    qrs, regions = cache["qrs"], cache["region_vectors"]
    offsets = np.asarray(cache["region_offsets"])
    if prototypes.shape != (3, qrs.shape[1]) or regions.shape[1] != qrs.shape[1]:
        raise ValueError("Reference feature dimensions differ")
    if (
        len(offsets) != len(qrs) + 1
        or offsets[0] != 0
        or offsets[-1] != len(regions)
        or np.any(np.diff(offsets) < 3)
    ):
        raise ValueError("Invalid patient region boundaries")
    record = np.clip(qrs @ prototypes.T, -1, 1)
    region = np.clip(regions @ prototypes.T, -1, 1)
    summaries = np.stack(
        [
            np.concatenate(
                [region[a:b].mean(axis=0), region[a:b].max(axis=0), region[a:b].std(axis=0, ddof=0)]
            )
            for a, b in zip(offsets[:-1], offsets[1:], strict=True)
        ]
    )
    if not np.isfinite(record).all() or not np.isfinite(summaries).all():
        raise ValueError("Nonfinite bank projection")
    return record, summaries


def bank_fit(
    values: np.ndarray, fitting: np.ndarray, anchors: np.ndarray, configuration: dict, deadline: float
) -> tuple[dict, np.ndarray]:
    """Fit a fixed bank configuration, retaining finite labels even if admission fails."""
    check_budget(deadline)
    centered = values - values[fitting].mean(axis=0)
    model, ok = fit_model(
        centered[fitting],
        configuration["method"],
        configuration["covariance"],
        configuration["clusters"],
        SEED,
    )
    labels = model.predict(centered)
    if labels.shape != (len(values),) or not np.isfinite(labels).all():
        raise ValueError("Invalid cluster assignments")
    fitting_ok = balanced(labels[fitting], configuration["clusters"])
    anchor_ok = balanced(labels[anchors], configuration["clusters"])
    return {
        "converged": bool(ok),
        "fit_balanced": bool(fitting_ok),
        "anchor_balanced": bool(anchor_ok),
        "admissible": bool(ok and fitting_ok and anchor_ok),
        "iterations": int(model.n_iter_),
        "anchor_sizes": np.bincount(labels[anchors], minlength=configuration["clusters"]).tolist(),
        "silhouette": silhouette(centered[anchors], labels[anchors]),
    }, labels


def align_fitting(
    labels: np.ndarray, reference: np.ndarray, fitting: np.ndarray, clusters: int
) -> tuple[np.ndarray, list[int]]:
    """Align cluster names by maximum fitting overlap with lexicographic permutation ties."""
    if (
        labels.shape != reference.shape
        or np.any(labels < 0)
        or np.any(labels >= clusters)
        or np.any(reference < 0)
        or np.any(reference >= clusters)
    ):
        raise ValueError("Cluster labels outside alignment domain")
    table = np.zeros((clusters, clusters), dtype=np.int64)
    np.add.at(table, (labels[fitting], reference[fitting]), 1)
    choices = list(permutations(range(clusters)))
    scores = [sum(table[i, name] for i, name in enumerate(choice)) for choice in choices]
    mapping = choices[int(np.argmax(scores))]
    return np.asarray(mapping)[labels], list(mapping)


def distribution(values: np.ndarray) -> dict:
    """Describe fixed conditional observations without independence-based confidence claims."""
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Cannot summarize empty or nonfinite values")
    return {
        "minimum": float(values.min()),
        "p10": float(np.quantile(values, 0.1, method="linear")),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9, method="linear")),
        "maximum": float(values.max()),
    }


def sensitivity(
    assignments: np.ndarray,
    checks: list[dict],
    reference: np.ndarray,
    fitting: np.ndarray,
    anchors: np.ndarray,
    clusters: int,
) -> tuple[dict, dict]:
    """Summarize all banks with fitting-only alignment and label-invariant coassignment.

    Parameters
    ----------
    assignments : np.ndarray
        Banks by patients, including inadmissible fits with valid returned labels.
    checks : list of dict
        Each bank's convergence and occupancy result, in matching order.
    reference : np.ndarray
        Frozen predecessor labels for every patient.
    fitting, anchors : np.ndarray
        Nonoverlapping fitting and development-validation row indices.
    clusters : int
        Frozen number of cluster names.

    Returns
    -------
    tuple
        Aggregate sensitivity and private per-anchor values/coassignment matrix.
    """
    if len(assignments) != len(checks) or len(assignments) < 2:
        raise ValueError("Need matching checks and at least two banks")
    anchor_labels = assignments[:, anchors]
    pairwise = np.array(
        [
            adjusted_rand_score(anchor_labels[a], anchor_labels[b])
            for a, b in combinations(range(len(assignments)), 2)
        ]
    )
    aligned, mappings = [], []
    for labels in assignments:
        mapped, mapping = align_fitting(labels, reference, fitting, clusters)
        aligned.append(mapped[anchors])
        mappings.append(mapping)
    aligned = np.stack(aligned)
    agreement = np.mean(aligned == reference[anchors], axis=0)
    probabilities = np.stack([(aligned == k).mean(axis=0) for k in range(clusters)], axis=1)
    entropy = -np.sum(probabilities * np.log2(np.maximum(probabilities, np.finfo(float).tiny)), axis=1)
    coassignment = np.mean(anchor_labels[:, :, None] == anchor_labels[:, None, :], axis=0)
    result = {
        "banks": len(assignments),
        "anchor_denominator": len(anchors),
        "admissible_banks": sum(row["admissible"] for row in checks),
        "pairwise_comparisons": len(pairwise),
        "pairwise_anchor_ari": distribution(pairwise),
        "reference_anchor_ari": distribution(
            np.array([adjusted_rand_score(reference[anchors], labels) for labels in anchor_labels])
        ),
        "reference_agreement": distribution(agreement),
        "anchors_agreement_at_least_0.80": int(np.sum(agreement >= 0.8)),
        "fraction_anchors_agreement_at_least_0.80": float(np.mean(agreement >= 0.8)),
        "modal_cluster_frequency": distribution(probabilities.max(axis=1)),
        "entropy_bits": distribution(entropy),
        "pairwise_observations_not_independent": True,
    }
    return result, {
        "pairwise_ari": pairwise,
        "aligned_anchor_labels": aligned,
        "reference_agreement": agreement,
        "modal_frequency": probabilities.max(axis=1),
        "entropy_bits": entropy,
        "anchor_coassignment": coassignment,
        "alignment_maps": np.array(mappings),
    }


def primary_decision(summary: dict) -> dict:
    """Apply only the frozen regional operational robustness gate."""
    if summary["banks"] != 50 or summary["anchor_denominator"] != 411:
        raise ValueError("Primary bank or anchor denominator changed")
    conditions = {
        "at_least_45_admissible_banks": summary["admissible_banks"] >= 45,
        "pairwise_p10_ari_at_least_0.50": summary["pairwise_anchor_ari"]["p10"] >= 0.5,
        "at_least_80_percent_anchors_agree_in_80_percent_banks": summary[
            "fraction_anchors_agreement_at_least_0.80"
        ]
        >= 0.8,
    }
    return {"conditions": conditions, "primary_robustness_gate_passed": all(conditions.values())}
