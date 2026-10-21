"""Fit-only preprocessing and stability-first morphology mixture comparison."""

from __future__ import annotations

import ast
import csv
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from ecg_experiment.clustering01 import CATEGORICAL, NUMERIC, SEED

VIEWS = ("raw", "residual", "shape")
COVARIANCES = ("diag", "tied", "full")
QUALITY = ("baseline_drift", "static_noise", "burst_noise", "electrodes_problems", "extra_beats", "pacemaker")


def anchor_patient(patient: int) -> bool:
    """Return the frozen internal development-anchor allocation for one patient."""
    key = f"cluster02_anchor:{SEED}:{int(patient)}"
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % 5 == 0


def permitted_metadata(path: Path, ecg_ids: pd.Series, diagnoses: bool = False) -> pd.DataFrame:
    """Interpret metadata only after membership in an already verified training cohort."""
    allowed = set(ecg_ids)
    rows = []
    with path.open() as stream:
        for row in csv.DictReader(stream):
            ecg = int(row["ecg_id"])
            if ecg not in allowed:
                continue
            values = {
                "ecg_id": ecg,
                "device": row["device"] or "missing",
                "site": row["site"] or "missing",
                "quality": int(any(row[name] not in ("", "False", "0") for name in QUALITY)),
            }
            if diagnoses:
                values["scp_codes"] = ast.literal_eval(row["scp_codes"])
            rows.append(values)
    return pd.DataFrame(rows).set_index("ecg_id").loc[ecg_ids].reset_index()


def shape_features(features: pd.DataFrame, nuisance: pd.DataFrame, view: str) -> pd.DataFrame:
    """Apply the fixed per-record amplitude control without fitting any cohort statistic."""
    result = features.copy()
    if view == "shape":
        columns = [name for name in features if name.startswith(("r_amp_", "s_amp_", "st_", "t_amp_"))]
        result[columns] = result[columns].div(nuisance.signal_std.clip(lower=1e-6), axis=0)
    return result


def fit_representation(features: pd.DataFrame, nuisance: pd.DataFrame, view: str) -> dict[str, Any]:
    """
    Fit transformations exclusively on the supplied fitting-patient rows.

    Parameters
    ----------
    features, nuisance : pd.DataFrame
        Aligned morphology and fixed nuisance measurements, with no anchor rows.
    view : str
        Raw, nuisance residualized, or amplitude-normalized shape.

    Returns
    -------
    dict
        Fitted transformers and immutable clipping/PCA dimensionality choices.
    """
    source = shape_features(features, nuisance, view)
    if source.isna().all().any():
        raise ValueError("An entire feature is missing in a fitting sample")
    state: dict[str, Any] = {"view": view, "imputer": SimpleImputer(strategy="median")}
    values = state["imputer"].fit_transform(source)
    state["clip"] = np.quantile(values, [0.01, 0.99], axis=0)
    state["scaler"] = RobustScaler().fit(np.clip(values, *state["clip"]))
    values = state["scaler"].transform(np.clip(values, *state["clip"]))
    if view == "residual":
        state["nuisance_imputer"] = SimpleImputer(strategy="median", keep_empty_features=True)
        numeric = state["nuisance_imputer"].fit_transform(nuisance[list(NUMERIC)])
        state["nuisance_scaler"] = RobustScaler().fit(numeric)
        numeric = state["nuisance_scaler"].transform(numeric)
        state["encoder"] = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        categorical = state["encoder"].fit_transform(
            nuisance[list(CATEGORICAL)].fillna("missing").astype(str)
        )
        design = np.column_stack([numeric, categorical])
        state["ridge"] = Ridge(alpha=10).fit(design, values)
        values = values - state["ridge"].predict(design)
        state["residual_scaler"] = RobustScaler().fit(values)
        values = state["residual_scaler"].transform(values)
    state["pca"] = PCA(n_components=0.95, svd_solver="full").fit(values)
    state["dimensions"] = min(16, state["pca"].n_components_)
    return state


def transform_representation(
    state: dict[str, Any], features: pd.DataFrame, nuisance: pd.DataFrame
) -> np.ndarray:
    """Apply fixed fitting-only transformations to any aligned observations, without adaptation."""
    values = state["imputer"].transform(shape_features(features, nuisance, state["view"]))
    values = state["scaler"].transform(np.clip(values, *state["clip"]))
    if state["view"] == "residual":
        numeric = state["nuisance_imputer"].transform(nuisance[list(NUMERIC)])
        numeric = state["nuisance_scaler"].transform(numeric)
        categorical = state["encoder"].transform(nuisance[list(CATEGORICAL)].fillna("missing").astype(str))
        values = values - state["ridge"].predict(np.column_stack([numeric, categorical]))
        values = state["residual_scaler"].transform(values)
    return state["pca"].transform(values)[:, : state["dimensions"]]


def fit_partition(
    values: np.ndarray, method: str, covariance: str, clusters: int, seed: int
) -> tuple[Any, bool]:
    """Fit one fixed clustering model without access to patient annotations."""
    if method == "kmeans":
        model = KMeans(n_clusters=clusters, n_init=10, random_state=seed).fit(values)
        return model, True
    model = GaussianMixture(
        n_components=clusters,
        covariance_type=covariance,
        n_init=3,
        reg_covar=1e-3,
        max_iter=400,
        tol=1e-3,
        random_state=seed,
    ).fit(values)
    return model, bool(model.converged_)


def balanced(labels: np.ndarray, clusters: int) -> bool:
    """Require every requested cluster and at least five percent of the population in each."""
    _, counts = np.unique(labels, return_counts=True)
    return len(counts) == clusters and float(counts.min() / len(labels)) >= 0.05


def silhouette(values: np.ndarray, labels: np.ndarray) -> float | None:
    """Return anchor silhouette when at least two nonempty groups exist."""
    if not 1 < len(np.unique(labels)) < len(labels):
        return None
    return float(silhouette_score(values, labels))


def stability_summary(reference: np.ndarray, refits: np.ndarray) -> dict[str, Any]:
    """Summarize refit prediction agreement on identical anchor patients."""
    values = [float(adjusted_rand_score(reference, new)) for new in refits]
    return {
        "values": values,
        "minimum": min(values),
        "median": float(np.median(values)),
        "maximum": max(values),
    }


def eligibility(row: dict[str, Any]) -> dict[str, bool]:
    """Apply the frozen geometric and complete-pipeline stability gates."""
    return {
        "all_converged": all(row["converged"]),
        "reference_fit_balanced": row["reference_fit_balanced"],
        "reference_anchor_balanced": row["reference_anchor_balanced"],
        "all_refits_anchor_balanced": all(row["refits_anchor_balanced"]),
        "silhouette_at_least_0.05": row["silhouette"] is not None and row["silhouette"] >= 0.05,
        "median_stability_at_least_0.80": row["stability"]["median"] >= 0.80,
        "minimum_stability_at_least_0.60": row["stability"]["minimum"] >= 0.60,
    }


def rank_key(row: dict[str, Any]) -> tuple:
    """Order eligible models first, then stability, separation, parsimony and fixed covariance order."""
    covariance = COVARIANCES.index(row["covariance"]) if row["covariance"] in COVARIANCES else 0
    return (
        row["eligible"],
        row["stability"]["median"],
        row["stability"]["minimum"],
        row["silhouette"] if row["silhouette"] is not None else -2,
        -row["clusters"],
        -covariance,
    )


def select_candidates(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Select a baseline and a mixture per view using no labels or nuisance associations."""
    result = {}
    for view in VIEWS:
        for method in ("kmeans", "gmm"):
            subset = [row for row in candidates if row["view"] == view and row["method"] == method]
            if subset:
                result[f"{view}:{method}"] = max(subset, key=rank_key)
    return result


def coassociation(partitions: list[np.ndarray]) -> np.ndarray:
    """Average label-permutation-invariant pairwise coassignment across representations."""
    return np.mean([labels[:, None] == labels[None, :] for labels in partitions], axis=0)


def consensus_partition(partitions: list[np.ndarray], clusters: int) -> np.ndarray:
    """Cluster one minus the fixed view-averaged coassignment matrix by average linkage."""
    distance = 1.0 - coassociation(partitions)
    return AgglomerativeClustering(n_clusters=clusters, metric="precomputed", linkage="average").fit_predict(
        distance
    )


def consensus_candidates(
    selected: dict[str, dict[str, Any]], assignments: dict[str, dict[str, np.ndarray]], raw_anchor: np.ndarray
) -> tuple[list[dict[str, Any]], dict]:
    """Build consensus only when every prespecified mixture view qualifies independently."""
    winners = [selected.get(f"{view}:gmm") for view in VIEWS]
    if not all(row is not None and row["eligible"] for row in winners):
        return [], {}
    states = [assignments[row["key"]] for row in winners]
    rows, labels = [], {}
    for clusters in (2, 4, 8):
        reference = consensus_partition([state["anchor"] for state in states], clusters)
        refits = np.stack(
            [consensus_partition([state["refits"][draw] for state in states], clusters) for draw in range(8)]
        )
        key = f"consensus:{clusters}"
        row = {
            "key": key,
            "view": "consensus",
            "method": "consensus",
            "covariance": "none",
            "clusters": clusters,
            "converged": [True] * 9,
            "reference_fit_balanced": True,
            "reference_anchor_balanced": balanced(reference, clusters),
            "refits_anchor_balanced": [balanced(new, clusters) for new in refits],
            "silhouette": silhouette(raw_anchor, reference),
            "stability": stability_summary(reference, refits),
        }
        row["conditions"] = eligibility(row)
        row["eligible"] = all(row["conditions"].values())
        rows.append(row)
        labels[key] = {"anchor": reference, "refits": refits}
    return rows, labels


def robustness_decision(
    selected: dict[str, dict[str, Any]],
    assignments: dict[str, dict[str, np.ndarray]],
    consensus: dict[str, Any] | None,
) -> dict[str, Any]:
    """Freeze the label-free persistence decision before annotation interpretation."""
    winners = [selected.get(f"{view}:gmm") for view in VIEWS]
    comparisons = {}
    for first, second in ((0, 1), (0, 2), (1, 2)):
        if winners[first] is not None and winners[second] is not None:
            comparisons[f"{VIEWS[first]}:{VIEWS[second]}"] = float(
                adjusted_rand_score(
                    assignments[winners[first]["key"]]["anchor"],
                    assignments[winners[second]["key"]]["anchor"],
                )
            )
    conditions = {
        "all_three_mixture_views_eligible": all(row is not None and row["eligible"] for row in winners),
        "every_cross_view_ari_at_least_0.50": len(comparisons) == 3 and min(comparisons.values()) >= 0.5,
        "consensus_eligible": consensus is not None and consensus["eligible"],
    }
    mixture, baseline = selected.get("residual:gmm"), selected.get("residual:kmeans")
    improvement = (
        mixture["stability"]["minimum"] - baseline["stability"]["minimum"]
        if mixture is not None and baseline is not None
        else None
    )
    return {
        "robust_across_views": all(conditions.values()),
        "conditions": conditions,
        "cross_view_anchor_ari": comparisons,
        "residual_minimum_ari_gain": improvement,
        "residual_stability_improvement": mixture is not None
        and baseline is not None
        and mixture["eligible"]
        and baseline["eligible"]
        and improvement >= 0.1,
    }
