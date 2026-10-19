"""Native and support-conditioned ECG clustering with fitting-only stability selection."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from ecg_experiment.clustering01 import partition_summary
from ecg_experiment.clustering02 import (
    balanced,
    fit_partition,
    fit_representation,
    silhouette,
    stability_summary,
    transform_representation,
)
from ecg_experiment.localization.jepa_native import EMBEDDING_SPACE
from ecg_experiment.native_reference_ntt05 import ReferenceInstance, reference_bank
from ecg_experiment.same_wave_ntt04 import FINDINGS, SEEDS

VIEWS = ("native_qrs", "native_global", "ntt_affinity", "ntt_regions", "physical")
COVARIANCES = ("diag", "tied", "full")
SEED = 20261018


def check_budget(deadline: float) -> None:
    """Require time remaining before admitting another fit or representation transform."""
    if time.perf_counter() >= deadline:
        raise TimeoutError("Prospective clustering and control budget exceeded")


def support_prototypes(selection: dict, features: dict, permutation_seed: int | None = None) -> np.ndarray:
    """Build three native NTT banks, optionally repartitioning individual support labels.

    Parameters
    ----------
    selection, features : dict
        Frozen NTT-06 reference identities and normalized QRS patient vectors.
    permutation_seed : int or None
        Fixed random seed for preserving three supports per finding while shuffling labels.

    Returns
    -------
    np.ndarray
        Three banks by three finding prototypes by 768 embedding coordinates.
    """
    rng = np.random.default_rng(permutation_seed)
    banks = []
    for seed in SEEDS:
        groups = [g for g in selection["support_draws"] if g["seed"] == seed]
        entries = [(g["finding"], r) for g in groups for r in g["supports"]]
        labels = np.array([label for label, _ in entries])
        if permutation_seed is not None:
            labels = rng.permutation(labels)
        instances = [
            ReferenceInstance(
                str(row["patient_id"]),
                str(label),
                np.asarray(features[str(row["ecg_id"])]["native_qrs"]),
                "frozen source reference label" if permutation_seed is None else "permuted reference label",
                EMBEDDING_SPACE,
            )
            for label, (_, row) in zip(labels, entries, strict=True)
        ]
        banks.append(reference_bank(instances, FINDINGS).prototypes)
    return np.stack(banks)


def affinity_views(cache: dict, prototypes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project patient and region vectors through the actual NTT prototype banks.

    Parameters
    ----------
    cache : dict
        Unit patient QRS vectors, unit region vectors and exact patient region offsets.
    prototypes : np.ndarray
        Bank-by-finding-by-coordinate unit prototype matrix.

    Returns
    -------
    tuple
        Mean-bank patient affinities [N,3], mean-bank region summaries [N,9],
        and individual-bank region summaries [3,N,9]. Each summary orders mean, max, std.
    """
    qrs, regions, offsets = cache["qrs"], cache["region_vectors"], cache["region_offsets"]
    if prototypes.shape != (3, 3, qrs.shape[1]) or regions.shape[1] != qrs.shape[1]:
        raise ValueError("Prototype or region feature dimensions differ")
    if len(offsets) != len(qrs) + 1 or offsets[0] != 0 or offsets[-1] != len(regions):
        raise ValueError("Malformed patient region offsets")
    if np.any(np.diff(offsets) < 3):
        raise ValueError("Every patient requires at least three complete regions")
    scores = np.clip(np.einsum("nd,bfd->bnf", qrs, prototypes), -1, 1)
    regional_scores = np.clip(np.einsum("nd,bfd->bnf", regions, prototypes), -1, 1)
    summaries = []
    for bank in regional_scores:
        summaries.append(
            np.stack(
                [
                    np.concatenate(
                        [bank[a:b].mean(axis=0), bank[a:b].max(axis=0), bank[a:b].std(axis=0, ddof=0)]
                    )
                    for a, b in zip(offsets[:-1], offsets[1:], strict=True)
                ]
            )
        )
    summaries = np.stack(summaries)
    return scores.mean(axis=0), summaries.mean(axis=0), summaries


def fit_view(values: np.ndarray, fitting: np.ndarray, view: str) -> tuple[dict, np.ndarray]:
    """Fit preprocessing on supplied identities and transform all observations without adaptation."""
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite representation")
    if view == "physical":
        frame = pd.DataFrame(values)
        nuisance = pd.DataFrame(index=frame.index)
        state = fit_representation(frame.iloc[fitting], nuisance.iloc[fitting], "raw")
        transformed = transform_representation(state, frame, nuisance)
        return {
            "dimensions": int(state["dimensions"]),
            "explained_variance": float(state["pca"].explained_variance_ratio_[: state["dimensions"]].sum()),
        }, transformed
    if view.startswith("ntt_"):
        return {"dimensions": values.shape[1], "explained_variance": 1.0}, values - values[fitting].mean(
            axis=0
        )
    pca = PCA(n_components=0.95, svd_solver="full").fit(values[fitting])
    keep = int(min(16, pca.n_components_))
    return {
        "dimensions": keep,
        "explained_variance": float(pca.explained_variance_ratio_[:keep].sum()),
    }, pca.transform(values)[:, :keep]


def representation_cache(views: dict, samples: list[np.ndarray], deadline: float) -> dict:
    """Cache independently refitted transforms once per view and fitting subsample."""
    result = {}
    for view, values in views.items():
        result[view] = []
        for fitting in samples:
            check_budget(deadline)
            state, coordinates = fit_view(values, fitting, view)
            result[view].append({**state, "coordinates": coordinates})
    return result


def fit_model(values: np.ndarray, method: str, covariance: str, clusters: int, seed: int) -> tuple[Any, bool]:
    """Conservatively reject k-means fits that reach their iteration cap."""
    model, converged = fit_partition(values, method, covariance, clusters, seed)
    if method == "kmeans":
        converged = bool(model.n_iter_ < model.max_iter)
    return model, converged


def candidate(
    view: str,
    method: str,
    covariance: str,
    clusters: int,
    cache: list,
    samples: list[np.ndarray],
    anchors: np.ndarray,
    deadline: float,
) -> tuple[dict, dict]:
    """Fit one configuration and its eight complete-pipeline subsample refits."""
    predictions, converged, fit_balanced, iteration_counts = [], [], [], []
    for draw, (state, rows) in enumerate(zip(cache, samples, strict=True)):
        check_budget(deadline)
        model, ok = fit_model(state["coordinates"][rows], method, covariance, clusters, SEED + draw)
        labels = model.predict(state["coordinates"])
        predictions.append(labels)
        converged.append(ok)
        iteration_counts.append({"iterations": int(model.n_iter_), "limit": int(model.max_iter)})
        fit_balanced.append(balanced(labels[rows], clusters))
    reference = predictions[0][anchors]
    refits = np.stack([labels[anchors] for labels in predictions[1:]])
    stability = stability_summary(reference, refits)
    separation = silhouette(cache[0]["coordinates"][anchors], reference)
    conditions = {
        "all_converged": all(converged),
        "all_fits_five_percent_occupancy": all(fit_balanced),
        "all_anchors_five_percent_occupancy": all(
            balanced(labels[anchors], clusters) for labels in predictions
        ),
        "silhouette_at_least_0.05": separation is not None and separation >= 0.05,
        "median_stability_at_least_0.80": stability["median"] >= 0.8,
        "minimum_stability_at_least_0.60": stability["minimum"] >= 0.6,
    }
    key = f"{view}:{method}:{covariance}:{clusters}"
    row = {
        "key": key,
        "view": view,
        "method": method,
        "covariance": covariance,
        "clusters": clusters,
        "eligible": all(conditions.values()),
        "conditions": conditions,
        "converged": converged,
        "iteration_counts": iteration_counts,
        "fitting_occupancy_passed": fit_balanced,
        "silhouette": separation,
        "stability": stability,
        "anchor_sizes": {
            str(k): int(n) for k, n in zip(*np.unique(reference, return_counts=True), strict=True)
        },
    }
    return row, {"all": predictions[0], "anchor": reference, "refits": refits}


def selection_key(row: dict) -> tuple:
    """Rank fixed unsupervised criteria without annotations or nuisance metrics."""
    complexity = 0 if row["method"] == "kmeans" else 1 + COVARIANCES.index(row["covariance"])
    return (
        row["eligible"],
        row["stability"]["median"],
        row["stability"]["minimum"],
        row["silhouette"] if row["silhouette"] is not None else -2,
        -row["clusters"],
        -complexity,
    )


def select(candidates: list[dict]) -> dict:
    """Select one eligible winner, or an explicitly ineligible fallback, per frozen view."""
    return {
        view: max([row for row in candidates if row["view"] == view], key=selection_key) for view in VIEWS
    }


def single_reference_fit(
    values: np.ndarray, fitting: np.ndarray, anchors: np.ndarray, configuration: dict, deadline: float
) -> tuple[dict, np.ndarray]:
    """Refit one selected NTT configuration for a new support bank or label permutation."""
    check_budget(deadline)
    state, transformed = fit_view(values, fitting, "ntt_regions")
    model, ok = fit_model(
        transformed[fitting],
        configuration["method"],
        configuration["covariance"],
        configuration["clusters"],
        SEED,
    )
    all_labels = model.predict(transformed)
    labels = all_labels[anchors]
    return {
        "converged": ok,
        "iterations": int(model.n_iter_),
        "iteration_limit": int(model.max_iter),
        "fit_balanced": balanced(all_labels[fitting], configuration["clusters"]),
        "anchor_balanced": balanced(labels, configuration["clusters"]),
        "silhouette": silhouette(transformed[anchors], labels),
        "dimensions": int(state["dimensions"]),
    }, labels


def primary_decision(
    selected: dict, assignments: dict, bank_labels: list[np.ndarray], bank_checks: list[dict]
) -> dict:
    """Freeze the cross-view and individual-bank persistence decision without target labels."""
    if len(bank_labels) != 3 or len(bank_checks) != 3:
        raise ValueError("Exactly three individual reference banks are required")
    cross = {}
    for i, first in enumerate(VIEWS):
        for second in VIEWS[i + 1 :]:
            cross[f"{first}:{second}"] = float(
                adjusted_rand_score(
                    assignments[selected[first]["key"]]["anchor"],
                    assignments[selected[second]["key"]]["anchor"],
                )
            )
    bank_ari = {
        f"{i}:{j}": float(adjusted_rand_score(bank_labels[i], bank_labels[j]))
        for i in range(3)
        for j in range(i + 1, 3)
    }
    conditions = {
        "native_qrs_eligible": selected["native_qrs"]["eligible"],
        "ntt_regions_eligible": selected["ntt_regions"]["eligible"],
        "native_ntt_ari_at_least_0.50": cross["native_qrs:ntt_regions"] >= 0.5,
        "all_bank_fits_converged_balanced": all(
            c["converged"] and c["fit_balanced"] and c["anchor_balanced"] for c in bank_checks
        ),
        "all_bank_pairwise_ari_at_least_0.50": min(bank_ari.values()) >= 0.5,
    }
    return {
        "primary_persistence_gate_passed": all(conditions.values()),
        "conditions": conditions,
        "cross_view_anchor_ari": cross,
        "individual_bank_pairwise_ari": bank_ari,
    }


def density_views(cache: dict, deadline: float) -> tuple[list[dict], dict]:
    """Describe transductive density groups without admitting them into primary selection."""
    rows, partitions = [], {}
    for view, states in cache.items():
        values = states[0]["coordinates"]
        for size in (40, 80):
            check_budget(deadline)
            labels = HDBSCAN(min_cluster_size=size, min_samples=10, n_jobs=2).fit_predict(values)
            key = f"{view}:hdbscan:{size}"
            summary = partition_summary(values, labels)
            summary["descriptive_coverage_heuristic_passed"] = summary.pop("eligible")
            rows.append(
                {
                    "key": key,
                    "view": view,
                    "min_cluster_size": size,
                    "transductive_includes_validation_anchors": True,
                    **summary,
                }
            )
            partitions[key] = labels
    return rows, partitions
