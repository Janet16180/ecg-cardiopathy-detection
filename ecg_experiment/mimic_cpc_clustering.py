"""Patient-balanced HDBSCAN exploration of self-supervised CPC features."""

from __future__ import annotations

import csv
import gzip
import json
import os
import time
from collections import Counter, defaultdict

import numpy as np
import torch
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize

from . import ROOT
from .cpc import CPCEncoder
from .cpc_pool import Pool, loader
from .files import sha256_file, sha256_json, write_json_atomic

CACHE = ROOT / "data/processed/cpc_pool_40k"
MODEL = ROOT / "outputs/experiment004_cpc_40k/cpc_ssl/encoder.pt"
NORMALIZATION = ROOT / "outputs/experiment004_cpc_40k/normalization.json"
FLAGS = ROOT / "outputs/mimic_cpc_flag_audit/identified_scores.csv.gz"
OUTPUT = ROOT / "outputs/mimic_cpc_clustering"
SEED = 42
PCA_DIMENSIONS = 16
MIN_CLUSTER_SIZE = 80
MIN_SAMPLES = 20
BATCH_SIZE = 128


def select_one_per_patient(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Choose one MIMIC recording per patient with a fixed seed."""
    by_patient: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["source"] == "mimic":
            by_patient[row["patient_id"]].append(row)
    rng = np.random.default_rng(SEED)
    return [sorted(group, key=lambda row: row["ecg_id"])[rng.integers(len(group))]
            for _, group in sorted(by_patient.items())]


def read_flags() -> dict[str, tuple[float, bool]]:
    """Read the prior fixed-threshold CPC audit by ECG ID."""
    with gzip.open(FLAGS, "rt", newline="") as handle:
        return {row["ecg_id"]: (float(row["cpc_probability_ptb_calibrated"]),
                                row["flagged"] == "1") for row in csv.DictReader(handle)}


def extract(pool: Pool, rows: list[dict[str, str]]) -> np.ndarray:
    """Extract SSL CPC pooled contexts without saving waveform or feature arrays."""
    normalization = json.loads(NORMALIZATION.read_text())
    if pool.metadata["signals_sha256"] != normalization["source"]["signals_sha256"]:
        raise RuntimeError("CPC signal cache differs from SSL model input")
    mean = np.asarray(normalization["mean"], dtype=np.float32)
    std = np.asarray(normalization["std"], dtype=np.float32)
    checkpoint = torch.load(MODEL, map_location="cpu", weights_only=False)
    encoder = CPCEncoder()
    encoder.load_state_dict(checkpoint["encoder"], strict=True)
    encoder.eval()
    features = []
    with torch.inference_mode():
        for signals, _, _ in loader(pool, rows, mean, std, BATCH_SIZE, False, None, "cpu"):
            _, contexts = encoder(signals)
            features.append(encoder.pooled(contexts).numpy())
    result = np.concatenate(features)
    if result.shape != (len(rows), 512) or not np.isfinite(result).all():
        raise RuntimeError("Invalid CPC feature matrix")
    return result


def run() -> dict[str, object]:
    """Cluster one ECG per MIMIC patient and summarize post hoc CPC flags."""
    start = time.monotonic()
    pool = Pool(CACHE)
    rows = select_one_per_patient(pool.rows)
    if len(rows) != 7892:
        raise RuntimeError(f"Unexpected patient count: {len(rows)}")
    flags_by_id = read_flags()
    if any(row["ecg_id"] not in flags_by_id for row in rows):
        raise RuntimeError("Selected ECG missing from prior CPC flag audit")
    features = extract(pool, rows)
    pca = PCA(n_components=PCA_DIMENSIONS, svd_solver="randomized", random_state=SEED)
    reduced = normalize(pca.fit_transform(features))
    if not np.isfinite(reduced).all():
        raise RuntimeError("Invalid PCA features")
    clusterer = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES,
                        cluster_selection_method="eom", metric="euclidean", n_jobs=1,
                        copy=True)
    groups = clusterer.fit_predict(reduced)
    membership = clusterer.probabilities_
    sensitivity = []
    for dimensions, minimum_size in ((16, 120), (32, 80)):
        variant_features = reduced if dimensions == 16 else normalize(PCA(
            n_components=dimensions, svd_solver="randomized", random_state=SEED
        ).fit_transform(features))
        variant = HDBSCAN(min_cluster_size=minimum_size, min_samples=MIN_SAMPLES,
                          cluster_selection_method="eom", metric="euclidean", n_jobs=1,
                          copy=True)
        variant_groups = variant.fit_predict(variant_features)
        sensitivity.append({
            "pca_dimensions": dimensions,
            "min_cluster_size": minimum_size,
            "cluster_count_excluding_noise": len(set(variant_groups)) - int(-1 in variant_groups),
            "noise_count": int(np.count_nonzero(variant_groups == -1)),
            "adjusted_rand_index_including_noise_vs_primary": float(
                adjusted_rand_score(groups, variant_groups)
            ),
        })
    scores = np.asarray([flags_by_id[row["ecg_id"]][0] for row in rows])
    flags = np.asarray([flags_by_id[row["ecg_id"]][1] for row in rows])
    counts = Counter(int(value) for value in groups)
    summary = []
    for group_id, count in sorted(counts.items()):
        selected = groups == group_id
        summary.append({"cluster": group_id, "ecgs_and_patients": count,
                        "cpc_flagged": int(flags[selected].sum()),
                        "cpc_flagged_fraction": float(flags[selected].mean()),
                        "median_cpc_score": float(np.median(scores[selected])),
                        "median_membership": float(np.median(membership[selected]))})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / "patient_sample_clusters.csv.gz"
    temporary = target.with_name(target.name + ".tmp")
    with gzip.open(temporary, "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("ecg_id", "patient_id", "cluster", "membership",
                         "cpc_score_ptb_calibrated", "cpc_flagged"))
        writer.writerows((row["ecg_id"], row["patient_id"], int(group),
                          f"{float(probability):.8f}", f"{float(score):.8f}", int(flag))
                         for row, group, probability, score, flag in zip(
                             rows, groups, membership, scores, flags, strict=True))
    os.replace(temporary, target)
    report: dict[str, object] = {
        "method": "SSL CPC 512D -> PCA16 -> unit vectors -> sklearn HDBSCAN",
        "seed": SEED,
        "selected_ecgs_and_patients": len(rows),
        "selected_ids_sha256": sha256_json([row["ecg_id"] for row in rows]),
        "pca_explained_variance_fraction": float(pca.explained_variance_ratio_.sum()),
        "min_cluster_size": MIN_CLUSTER_SIZE,
        "min_samples": MIN_SAMPLES,
        "cluster_count_excluding_noise": len(counts) - int(-1 in counts),
        "noise_count": counts.get(-1, 0),
        "overall_cpc_flagged": int(flags.sum()),
        "overall_cpc_flagged_fraction": float(flags.mean()),
        "clusters": summary,
        "post_hoc_parameter_sensitivity": sensitivity,
        "elapsed_seconds": time.monotonic() - start,
        "encoder_sha256": sha256_file(MODEL),
        "cache_rows_sha256": sha256_file(CACHE / "rows.csv"),
        "prior_flags_sha256": sha256_file(FLAGS),
        "identified_clusters": str(target.relative_to(ROOT)),
        "identified_clusters_sha256": sha256_file(target),
    }
    write_json_atomic(OUTPUT / "report.json", report)
    return report
