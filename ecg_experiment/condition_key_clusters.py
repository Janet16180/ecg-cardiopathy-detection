"""Training-fitted descriptive cluster summaries without exporting individual coordinates."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score

from .condition_keys import GROUPS, SEED


def cluster_table(labels: np.ndarray, flags: np.ndarray) -> list[dict[str, Any]]:
    """
    Summarize cluster sizes and overlapping annotation prevalences.

    Parameters
    ----------
    labels : np.ndarray
        Integer cluster assignments.
    flags : np.ndarray
        Binary group memberships.

    Returns
    -------
    list[dict[str, Any]]
        Per-cluster counts and memberships; group totals may overlap.
    """
    rows = []
    for cluster in range(5):
        subset = flags[labels == cluster]
        rows.append({"cluster": cluster, "records": len(subset),
                     "normal": int((~subset.any(axis=1)).sum()),
                     **{name: int(subset[:, c].sum()) for c, name in enumerate(GROUPS)}})
    return rows


def cluster_analysis(
    train: np.ndarray, query: np.ndarray, flags: np.ndarray, qflags: np.ndarray, families: np.ndarray
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """
    Fit training-only PCA and k-means and export aggregate grids and cluster tables.

    Parameters
    ----------
    train, query : np.ndarray
        Standardized unit whole-record embeddings.
    flags, qflags : np.ndarray
        Training and query memberships.
    families : np.ndarray
        Training source-family identities.

    Returns
    -------
    tuple[dict[str, Any], dict[str, np.ndarray]]
        Publishable aggregate tables/grids and private per-record projections/assignments.
    """
    pca = PCA(n_components=16, svd_solver="randomized", random_state=SEED)
    projected = pca.fit_transform(train)
    qprojected = pca.transform(query)
    model = KMeans(n_clusters=5, n_init=10, random_state=SEED).fit(projected)
    assigned, qassigned = model.labels_, model.predict(qprojected)
    bitmask = flags @ (2 ** np.arange(4))
    edges = [np.linspace(*np.quantile(projected[:, c], [0.005, 0.995]), 51) for c in range(2)]
    grids = {}
    masks = {"all": np.ones(len(query), bool), "normal": ~qflags.any(axis=1),
             **{name: qflags[:, c] == 1 for c, name in enumerate(GROUPS)}}
    for name, mask in masks.items():
        grid, _, _ = np.histogram2d(qprojected[mask, 0], qprojected[mask, 1], bins=edges)
        grids[name] = {"counts": grid.astype(int).tolist(), "records": int(mask.sum()),
                       "outside_grid": int(mask.sum() - grid.sum())}
    source_counts = [{"cluster": c, **{str(f): int(((families == f) & (assigned == c)).sum())
                                      for f in np.unique(families)}} for c in range(5)]
    annotation_ami = float(adjusted_mutual_info_score(bitmask, assigned))
    aggregate = {"pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                 "training_ami": {"source_family": float(adjusted_mutual_info_score(families, assigned)),
                                  "annotation_combination": annotation_ami},
                 "training": cluster_table(assigned, flags), "sph": cluster_table(qassigned, qflags),
                 "training_source_counts": source_counts, "grid_edges": [e.tolist() for e in edges],
                 "sph_pca_grids": grids, "clusters": 5, "fit_on": "training_only",
                 "inertia": float(model.inertia_)}
    private = {"training_pca": projected, "query_pca": qprojected,
               "training_cluster": assigned, "query_cluster": qassigned,
               "pca_components": pca.components_, "pca_mean": pca.mean_,
               "cluster_centers": model.cluster_centers_}
    return aggregate, private
