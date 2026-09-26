"""Read-only diagnosis geometry check for frozen CPC features."""

from __future__ import annotations

import ast
import csv
import json
from collections import Counter, defaultdict

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from . import ROOT
from .cpc_pool import Pool
from .files import sha256_file, sha256_json, write_json_atomic
from .mimic_cpc_cluster_label_profile import proximity
from .mimic_cpc_clustering import CACHE, MODEL, extract, select_one_per_patient
from .mimic_machine_disagreement import (
    CLUSTERS,
    MACHINE,
    OFFICIAL_SHA256,
    join_machine_rows,
    read_local_tables,
)

OUTPUT = ROOT / "outputs/ptb_mimic_cpc_diagnosis_geometry"
PROTOCOL = ROOT / "docs/ptb-mimic-cpc-diagnosis-geometry.md"
PTB = ROOT / "data/raw/ptb-xl/1.0.3"
PTB_MANIFEST = ROOT / "data/processed/ptbxl/seed42_fraction1/all_train_ssl.csv"
CLASSES = ("MI", "CD", "STTC", "HYP")
SEED = 42
PERMUTATIONS = 1000
NEIGHBORS = 10


def select_ptb(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Select one ECG per PTB training patient using a fixed seed."""
    by_patient: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["source"] == "ptbxl" and row["split"] == "train":
            by_patient[row["patient_id"]].append(row)
    rng = np.random.default_rng(SEED)
    return [sorted(group, key=lambda row: int(row["ecg_id"]))[rng.integers(len(group))]
            for _, group in sorted(by_patient.items())]


def diagnosis_classes(rows: list[dict[str, str]]) -> list[set[str]]:
    """Map released PTB SCP statements to diagnostic superclasses."""
    with (PTB / "scp_statements.csv").open(newline="") as handle:
        classes = {row[""]: row["diagnostic_class"] for row in csv.DictReader(handle)
                   if row["diagnostic"] == "1.0" and row["diagnostic_class"]}
    by_id = {row["ecg_id"]: row for row in rows}
    found: dict[str, set[str]] = {}
    with (PTB / "ptbxl_database.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            cached = by_id.get(row["ecg_id"])
            if cached is None:
                continue
            if cached["patient_id"] != row["patient_id"]:
                raise RuntimeError("PTB metadata patient mismatch")
            codes = ast.literal_eval(row["scp_codes"])
            found[row["ecg_id"]] = {
                classes[code] for code, confidence in codes.items()
                if code in classes and float(confidence) >= 50
            }
    if len(found) != len(rows):
        raise RuntimeError("Selected PTB ECG missing source annotation")
    return [found[row["ecg_id"]] for row in rows]


def ptb_neighbor_report(vectors: np.ndarray, labels: list[set[str]]) -> dict[str, object]:
    """Compute diagnosis-specific ten-neighbor enrichment and permutation checks."""
    neighbors = NearestNeighbors(n_neighbors=NEIGHBORS, metric="euclidean")
    indices = neighbors.fit(vectors).kneighbors(return_distance=False)
    rng = np.random.default_rng(SEED)
    report = {}
    for name in CLASSES:
        positive = np.asarray([name in row for row in labels])
        observed = float(positive[indices][positive].mean())
        null = np.empty(PERMUTATIONS)
        for draw in range(PERMUTATIONS):
            shuffled = rng.permutation(positive)
            null[draw] = float(shuffled[indices][shuffled].mean())
        report[name] = {
            "positive_patients": int(positive.sum()),
            "prevalence": float(positive.mean()),
            "mean_same_diagnosis_fraction_in_10_neighbors": observed,
            "enrichment_vs_prevalence": observed / positive.mean(),
            "permutation_null_95pct_interval": [float(x) for x in np.quantile(null, [0.025, 0.975])],
            "one_sided_permutation_p_exploratory":
                (1 + int(np.count_nonzero(null >= observed))) / (PERMUTATIONS + 1),
        }
    return report


def ptb_cluster_report(vectors: np.ndarray, labels: list[set[str]]) -> dict[str, object]:
    """Summarize unlabeled HDBSCAN groups in a seeded PTB subset."""
    sampled_indices = np.sort(np.random.default_rng(SEED).choice(len(labels), size=4000,
                                                                    replace=False))
    vectors = vectors[sampled_indices]
    labels = [labels[index] for index in sampled_indices]
    groups = HDBSCAN(min_cluster_size=80, min_samples=20,
                     cluster_selection_method="eom", metric="euclidean",
                     n_jobs=1, copy=True).fit_predict(vectors)
    summary = {}
    for group in sorted(set(groups)):
        group_labels = [labels[index] for index in np.flatnonzero(groups == group)]
        summary[str(int(group))] = {
            "patients": len(group_labels),
            "diagnosis_counts": {name: sum(name in row for row in group_labels)
                                 for name in ("NORM", *CLASSES)},
            "norm_only": sum(row == {"NORM"} for row in group_labels),
            "no_qualifying_diagnostic_class": sum(not row for row in group_labels),
        }
    return {"method": "Seeded 4000 PTB patients, PCA16 unit vectors, HDBSCAN size80/min_samples20/eom",
            "selected_indices_sha256": sha256_json(sampled_indices.tolist()), "groups": summary}


def run() -> dict[str, object]:
    """Run the frozen PTB diagnostic and MIMIC transfer geometry analyses."""
    if sha256_file(MACHINE) != OFFICIAL_SHA256:
        raise RuntimeError("MIMIC machine source changed")
    prior = json.loads((CLUSTERS.parent / "report.json").read_text())
    if sha256_file(CLUSTERS) != prior["identified_clusters_sha256"]:
        raise RuntimeError("Frozen MIMIC cluster IDs changed")
    if sha256_file(MODEL) != prior["encoder_sha256"]:
        raise RuntimeError("Frozen CPC encoder changed")
    pool = Pool(CACHE)
    ptb_rows = select_ptb(pool.rows)
    with PTB_MANIFEST.open(newline="") as handle:
        manifest = {row["ecg_id"]: row for row in csv.DictReader(handle)}
    if len(manifest) != 17418 or any(
        row["ecg_id"] not in manifest
        or row["patient_id"] != manifest[row["ecg_id"]]["patient_id"]
        for row in ptb_rows
    ):
        raise RuntimeError("PTB patient sample differs from training manifest")
    labels = diagnosis_classes(ptb_rows)
    print(f"Extracting PTB features for {len(ptb_rows)} patients", flush=True)
    ptb_features = extract(pool, ptb_rows)
    pca = PCA(n_components=16, svd_solver="randomized", random_state=SEED)
    ptb_vectors = normalize(pca.fit_transform(ptb_features))
    print("Computing PTB diagnosis neighborhoods", flush=True)
    ptb_neighbors = ptb_neighbor_report(ptb_vectors, labels)
    print("Clustering seeded PTB subset", flush=True)
    ptb_clusters = ptb_cluster_report(ptb_vectors, labels)

    mimic_rows = select_one_per_patient(pool.rows)
    if sha256_json([row["ecg_id"] for row in mimic_rows]) != prior["selected_ids_sha256"]:
        raise RuntimeError("MIMIC patient sample changed")
    scores, clusters = read_local_tables()
    joined = join_machine_rows(scores, clusters)
    mimic_items = [joined[row["ecg_id"]] for row in mimic_rows]
    print(f"Extracting MIMIC features for {len(mimic_rows)} patients", flush=True)
    mimic_vectors = normalize(pca.transform(extract(pool, mimic_rows)))
    combinations = Counter(tuple(sorted(row)) for row in labels)
    report: dict[str, object] = {
        "interpretation": (
            "Exploratory ECG-annotation geometry; no clinical diagnosis "
            "or external subtype validation"
        ),
        "ptb_training_patients_and_ecgs": len(ptb_rows),
        "ptb_sample_ids_sha256": sha256_json([row["ecg_id"] for row in ptb_rows]),
        "ptb_label_combinations": {";".join(key) or "none": count
                                   for key, count in sorted(combinations.items())},
        "ptb_diagnosis_neighbor_report": ptb_neighbors,
        "ptb_unsupervised_clusters": ptb_clusters,
        "mimic_patient_ecgs": len(mimic_rows),
        "mimic_ptb_fitted_pca_machine_summary_proximity": proximity(mimic_vectors, mimic_items),
        "pca_fit": "PTB train only, 16D randomized PCA seed42, unit vectors after transform",
        "ptb_pca_variance_fraction": float(pca.explained_variance_ratio_.sum()),
        "command": (
            "UV_CACHE_DIR=/tmp/ecg_uv_cache uv run --no-sync python -m "
            "scripts.reports.profile_ptb_mimic_cpc_diagnoses"
        ),
        "source_hashes": {
            "protocol": sha256_file(PROTOCOL),
            "analysis": sha256_file(ROOT / "ecg_experiment/ptb_mimic_cpc_diagnosis_geometry.py"),
            "encoder": sha256_file(MODEL),
            "cache_rows": sha256_file(CACHE / "rows.csv"),
            "ptb_manifest": sha256_file(PTB_MANIFEST),
            "ptb_database": sha256_file(PTB / "ptbxl_database.csv"),
            "ptb_statements": sha256_file(PTB / "scp_statements.csv"),
            "mimic_clusters": sha256_file(CLUSTERS),
            "mimic_machine": OFFICIAL_SHA256,
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUTPUT / "report.json", report)
    return report
