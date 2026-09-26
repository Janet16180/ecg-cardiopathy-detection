"""Cluster CPC-negative MIMIC ECGs, then inspect independent machine text."""

from __future__ import annotations

import csv
import gzip
import os
from collections import Counter

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from . import ROOT
from .cpc_pool import Pool
from .files import sha256_file, sha256_json, write_json_atomic
from .mimic_cpc_clustering import CACHE, MODEL, SEED, extract
from .mimic_machine_disagreement import (
    MACHINE,
    OFFICIAL_SHA256,
    SCORES,
    join_machine_rows,
    read_local_tables,
)

OUTPUT = ROOT / "outputs/mimic_cpc_negative_clustering"
MIN_CLUSTER_SIZE = 50
MIN_SAMPLES = 10
PCA_DIMENSIONS = 16


def summarize(
    rows: list[dict[str, str]], groups: np.ndarray, machine: dict[str, dict[str, object]]
) -> list[dict[str, object]]:
    """Describe machine summaries and patient concentration after clustering."""
    result = []
    for group_id in sorted({int(group) for group in groups}):
        selected = [row for row, group in zip(rows, groups, strict=True) if group == group_id]
        patients = Counter(row["patient_id"] for row in selected)
        statuses = Counter(str(machine[row["ecg_id"]]["machine_status"])
                           if row["ecg_id"] in machine else "missing" for row in selected)
        abnormal_patients = Counter(
            row["patient_id"] for row in selected
            if row["ecg_id"] in machine and machine[row["ecg_id"]]["machine_status"] == "abnormal"
        )
        carts = Counter(str(machine[row["ecg_id"]]["cart_id"])
                        for row in selected if row["ecg_id"] in machine)
        covered = sum(statuses[status] for status in ("abnormal", "normal", "borderline"))
        result.append({
            "cluster": group_id,
            "ecgs": len(selected),
            "patients": len(patients),
            "largest_patient_fraction": max(patients.values()) / len(selected),
            "machine_status_counts": dict(statuses),
            "machine_abnormal_patients": len(abnormal_patients),
            "largest_abnormal_patient_fraction": (
                max(abnormal_patients.values()) / statuses["abnormal"]
                if statuses["abnormal"] else None
            ),
            "distinct_carts": len(carts),
            "largest_cart_fraction": max(carts.values()) / len(selected) if carts else None,
            "machine_abnormal_fraction_of_ecgs": statuses["abnormal"] / len(selected),
            "machine_abnormal_fraction_with_explicit_summary":
                statuses["abnormal"] / covered if covered else None,
        })
    return result


def run() -> dict[str, object]:
    """Cluster fixed CPC negatives without using machine labels in fitting."""
    if sha256_file(MACHINE) != OFFICIAL_SHA256:
        raise RuntimeError("Machine source identity differs from the official receipt")
    all_scores, _ = read_local_tables()
    negatives = {ecg_id: row for ecg_id, row in all_scores.items() if row["flagged"] == "0"}
    if len(negatives) != 4782:
        raise RuntimeError(f"Unexpected CPC-negative cohort size: {len(negatives)}")
    pool = Pool(CACHE)
    rows = [row for row in pool.rows if row["ecg_id"] in negatives]
    if len(rows) != len(negatives):
        raise RuntimeError("CPC-negative ECG missing from frozen waveform cache")
    features = extract(pool, rows)
    pca = PCA(n_components=PCA_DIMENSIONS, svd_solver="randomized", random_state=SEED)
    reduced = normalize(pca.fit_transform(features))
    clusterer = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES,
                        cluster_selection_method="eom", metric="euclidean", n_jobs=1,
                        copy=True)
    groups = clusterer.fit_predict(reduced)
    machine = join_machine_rows(negatives, {})
    summary = summarize(rows, groups, machine)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / "negative_ecg_clusters.csv.gz"
    temporary = target.with_name(target.name + ".tmp")
    with gzip.open(temporary, "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("ecg_id", "patient_id", "cluster", "membership",
                         "cpc_score", "machine_status"))
        writer.writerows((row["ecg_id"], row["patient_id"], int(group),
                          f"{float(membership):.8f}",
                          negatives[row["ecg_id"]]["cpc_probability_ptb_calibrated"],
                          machine[row["ecg_id"]]["machine_status"] if row["ecg_id"] in machine
                          else "missing")
                         for row, group, membership in zip(
                             rows, groups, clusterer.probabilities_, strict=True))
    os.replace(temporary, target)
    report: dict[str, object] = {
        "method": "Frozen CPC-negative cohort, SSL CPC -> PCA16 -> HDBSCAN",
        "definition": "Machine abnormalities are exploratory proxy annotations, not confirmed misses",
        "seed": SEED,
        "ecgs": len(rows),
        "patients": len({row["patient_id"] for row in rows}),
        "selected_ids_sha256": sha256_json([row["ecg_id"] for row in rows]),
        "pca_explained_variance_fraction": float(pca.explained_variance_ratio_.sum()),
        "min_cluster_size": MIN_CLUSTER_SIZE,
        "min_samples": MIN_SAMPLES,
        "cluster_count_excluding_noise": len(summary) - int(any(item["cluster"] == -1 for item in summary)),
        "clusters": summary,
        "encoder_sha256": sha256_file(MODEL),
        "scores_sha256": sha256_file(SCORES),
        "machine_sha256": OFFICIAL_SHA256,
        "source_sha256": sha256_file(ROOT / "ecg_experiment/mimic_negative_clustering.py"),
        "identified_clusters": str(target.relative_to(ROOT)),
        "identified_clusters_sha256": sha256_file(target),
    }
    write_json_atomic(OUTPUT / "report.json", report)
    return report
