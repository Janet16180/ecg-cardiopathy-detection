"""Compare frozen CPC clusters and geometry with MIMIC machine summaries."""

from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import fisher_exact
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from . import ROOT
from .cpc_pool import Pool
from .files import sha256_file, sha256_json, write_json_atomic
from .mimic_cpc_clustering import CACHE, MODEL, PCA_DIMENSIONS, SEED, extract, select_one_per_patient
from .mimic_machine_disagreement import (
    MACHINE, OFFICIAL_SHA256, CLUSTERS, SCORES, join_machine_rows, read_local_tables,
)

OUTPUT = ROOT / "outputs/mimic_cpc_cluster_label_profile"
PROTOCOL = ROOT / "docs/mimic-cpc-cluster-label-profile.md"
PERMUTATIONS = 1000
NEIGHBORS = 10


def group_summary(items: list[dict[str, object]]) -> dict[str, object]:
    """Count exact cart summaries and fixed CPC flags for one patient per row."""
    statuses = Counter(str(item["machine_status"]) for item in items)
    summarized = sum(statuses[status] for status in ("abnormal", "normal", "borderline"))
    abnormal = [item for item in items if item["machine_status"] == "abnormal"]
    normal = [item for item in items if item["machine_status"] == "normal"]
    return {
        "patients_and_ecgs": len(items),
        "machine_status_counts": dict(sorted(statuses.items())),
        "machine_abnormal_fraction_all": len(abnormal) / len(items),
        "machine_abnormal_fraction_explicit_summary": len(abnormal) / summarized,
        "explicit_summary_count": summarized,
        "cpc_flagged_all": sum(bool(item["cpc_flagged"]) for item in items),
        "cpc_flagged_machine_abnormal": sum(bool(item["cpc_flagged"]) for item in abnormal),
        "cpc_flagged_machine_normal": sum(bool(item["cpc_flagged"]) for item in normal),
    }


def proximity(reduced: np.ndarray, items: list[dict[str, object]]) -> dict[str, object]:
    """Measure abnormal/normal machine-label homophily without fitting labels."""
    eligible = np.array([
        index for index, item in enumerate(items)
        if item["machine_status"] in {"abnormal", "normal"}
    ], dtype=np.int64)
    labels = np.array([items[index]["machine_status"] == "abnormal" for index in eligible])
    carts = np.array([str(items[index]["cart_id"]) for index in eligible])
    clusters = np.array([str(items[index]["cluster"]) for index in eligible])
    neighbor_model = NearestNeighbors(n_neighbors=NEIGHBORS, metric="euclidean")
    neighbor_indices = neighbor_model.fit(reduced[eligible]).kneighbors(return_distance=False)
    # kneighbors(X=None) excludes each query itself.
    if neighbor_indices.shape != (len(eligible), NEIGHBORS):
        raise RuntimeError("Unexpected nearest-neighbor shape")

    def statistic(candidate: np.ndarray) -> float:
        abnormal_neighbor_fraction = candidate[neighbor_indices].mean(axis=1)
        return float(abnormal_neighbor_fraction[candidate].mean()
                     - abnormal_neighbor_fraction[~candidate].mean())

    observed = statistic(labels)
    neighbor_fraction = labels[neighbor_indices].mean(axis=1)
    by_cluster = {}
    for cluster in ("-1", "0", "1"):
        in_cluster = clusters == cluster
        abnormal = in_cluster & labels
        normal = in_cluster & ~labels
        by_cluster[cluster] = {
            "abnormal_queries": int(abnormal.sum()),
            "normal_queries": int(normal.sum()),
            "mean_abnormal_neighbor_fraction_for_abnormal_queries":
                float(neighbor_fraction[abnormal].mean()),
            "mean_abnormal_neighbor_fraction_for_normal_queries":
                float(neighbor_fraction[normal].mean()),
        }
    rng = np.random.default_rng(SEED)
    cart_groups = defaultdict(list)
    for index, cart in enumerate(carts):
        cart_groups[cart].append(index)
    cart_groups = [np.asarray(indices) for indices in cart_groups.values()]
    stratified = np.empty(PERMUTATIONS)
    unstratified = np.empty(PERMUTATIONS)
    for trial in range(PERMUTATIONS):
        permuted = labels.copy()
        for indices in cart_groups:
            permuted[indices] = rng.permutation(permuted[indices])
        stratified[trial] = statistic(permuted)
        unstratified[trial] = statistic(rng.permutation(labels))
    return {
        "method": "10 nearest other patients among exact abnormal/normal summaries; Euclidean PCA16 unit vectors",
        "eligible_ecgs_and_patients": len(eligible),
        "machine_abnormal": int(labels.sum()),
        "machine_normal": int((~labels).sum()),
        "distinct_carts": len(cart_groups),
        "mean_abnormal_neighbor_fraction_for_abnormal_queries": float(neighbor_fraction[labels].mean()),
        "mean_abnormal_neighbor_fraction_for_normal_queries": float(neighbor_fraction[~labels].mean()),
        "query_groups_with_global_neighbors": by_cluster,
        "same_cart_neighbor_fraction": float((carts[neighbor_indices] == carts[:, None]).mean()),
        "observed_difference": observed,
        "permutations": PERMUTATIONS,
        "within_cart_null_mean": float(stratified.mean()),
        "within_cart_null_95pct_interval": [float(x) for x in np.quantile(stratified, [0.025, 0.975])],
        "within_cart_one_sided_p": (1 + int(np.count_nonzero(stratified >= observed)))
        / (PERMUTATIONS + 1),
        "global_null_mean": float(unstratified.mean()),
        "global_null_95pct_interval": [float(x) for x in np.quantile(unstratified, [0.025, 0.975])],
        "global_one_sided_p": (1 + int(np.count_nonzero(unstratified >= observed)))
        / (PERMUTATIONS + 1),
    }


def run() -> dict[str, object]:
    """Reproduce frozen features, then compare them with independent cart text."""
    if sha256_file(MACHINE) != OFFICIAL_SHA256:
        raise RuntimeError("Official machine table hash changed")
    cluster_report = json.loads((CLUSTERS.parent / "report.json").read_text())
    if sha256_file(CLUSTERS) != cluster_report["identified_clusters_sha256"]:
        raise RuntimeError("Frozen cluster assignments changed")
    if sha256_file(MODEL) != cluster_report["encoder_sha256"]:
        raise RuntimeError("Frozen CPC encoder changed")
    pool = Pool(CACHE)
    rows = select_one_per_patient(pool.rows)
    if sha256_json([row["ecg_id"] for row in rows]) != cluster_report["selected_ids_sha256"]:
        raise RuntimeError("Patient sample changed")
    scores, clusters = read_local_tables()
    joined = join_machine_rows(scores, clusters)
    items = [joined[row["ecg_id"]] for row in rows]
    groups = {str(group): group_summary([
        item for item in items if item["cluster"] == str(group)
    ]) for group in (-1, 0, 1)}
    cluster_zero = groups["0"]["machine_status_counts"]
    others = Counter(groups["-1"]["machine_status_counts"])
    others.update(groups["1"]["machine_status_counts"])
    odds, p_value = fisher_exact([
        [cluster_zero.get("abnormal", 0), sum(cluster_zero.get(s, 0) for s in ("normal", "borderline"))],
        [others["abnormal"], others["normal"] + others["borderline"]],
    ])
    features = extract(pool, rows)
    reduced = normalize(PCA(n_components=PCA_DIMENSIONS, svd_solver="randomized",
                            random_state=SEED).fit_transform(features))
    report: dict[str, object] = {
        "interpretation": "Exploratory association with ECG-machine summaries, not adjudicated cardiopathy",
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "one_ecg_per_patient": len(rows),
        "group_summary": groups,
        "cluster_zero_vs_other_explicit_summaries": {
            "odds_ratio": float(odds),
            "two_sided_fisher_p_exploratory": float(p_value),
        },
        "proximity": proximity(reduced, items),
        "machine_sha256": OFFICIAL_SHA256,
        "encoder_sha256": sha256_file(MODEL),
        "cache_rows_sha256": sha256_file(CACHE / "rows.csv"),
        "scores_sha256": sha256_file(SCORES),
        "clusters_sha256": sha256_file(CLUSTERS),
        "protocol_sha256": sha256_file(PROTOCOL),
        "source_sha256": sha256_file(ROOT / "ecg_experiment/mimic_cpc_cluster_label_profile.py"),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUTPUT / "report.json", report)
    return report
