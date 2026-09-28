"""Experiment 024: frozen embedding geometry audit and diagnosis prototypes on PTB-XL development patients."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, normalize
from threadpoolctl import threadpool_limits

from ecg_experiment.eda.ptbxl import (
    diagnostic_classes,
    load_metadata,
    load_statements,
    read_signal,
    superclasses,
)
from ecg_experiment.eda.signals import LEADS
from ecg_experiment.embedding_geometry import (
    NEIGHBORS,
    SEED,
    age_decade,
    ami_summary,
    audit_candidates,
    heart_rate_band,
    label_disagreement,
    medoids,
    method_scores,
    neighbor_indices,
    patient_subset,
    ridge_r2,
    same_device_summary,
    unit_space,
)
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.full_development import cohorts, fit_logistic, patient_bootstrap, predict, ptb_table
from scripts.experiments import run_label_efficiency025 as label_efficiency

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/experiment024_embedding_geometry_v1"
PRIOR = ROOT / "outputs/experiment020_full_development_v2"
PTB_RAW = ROOT / "data/raw/ptb-xl/1.0.3"
HEART_RATE = ROOT / "outputs/eda/features/ptbxl_500hz.parquet"
PRIOR_AUROC = 0.8886420735557192
PROBABILITY_TOLERANCE = 1e-5
BOOTSTRAP_DRAWS = 2000
THREADS = 4
DEVICE_MINIMUM = 50
SUBCLASS_MINIMUM = {"train": 100, "development": 20}
KMEANS_K = (2, 4, 8, 16)
HDBSCAN_PATIENTS = 4000
GROUPS = ("NORM", "MI", "STTC", "CD", "HYP")
SUPERCLASS_TASKS = ("MI", "STTC", "CD", "HYP")
AUDIT_SHOWN = 20
BINARY_CONTRASTS = (("prototype", "probe"), ("prototype_device_balanced", "probe"), ("knn", "probe"),
                    ("knn_other_device", "probe"), ("multi_prototype_2", "probe"),
                    ("multi_prototype_4", "probe"), ("multi_prototype_8", "probe"),
                    ("knn_other_device", "knn"), ("prototype_device_balanced", "prototype"))
ENCODER_CONTRASTS = (("prototype", "probe"), ("knn_other_device", "knn"),
                     ("prototype_device_balanced", "prototype"))
SOURCES = (
    "ecg_experiment/embedding_geometry.py", "ecg_experiment/full_development.py",
    "ecg_experiment/eda/ptbxl.py", "ecg_experiment/eda/signals.py", "ecg_experiment/files.py",
    "scripts/experiments/run_embedding_geometry024.py", "scripts/experiments/run_label_efficiency025.py",
    "pyproject.toml", "uv.lock",
    "docs/experiment-024-embedding-geometry.md",
)


def identity() -> dict[str, Any]:
    """Hash every input and source; the released caches must match their extraction receipts."""
    caches = label_efficiency.identity()
    inputs = {**caches["inputs"], "heart_rate": sha256_file(HEART_RATE),
              "ptbxl_sha256sums": sha256_file(PTB_RAW / "SHA256SUMS.txt")}
    if inputs["cpc_features"] != json.loads((PRIOR / "result.json").read_text())["features_sha256"]:
        raise ValueError("The Experiment 020 CPC features changed")
    return {"inputs": inputs, "sources": {name: sha256_file(ROOT / name) for name in SOURCES},
            "verified_against_receipts": caches["verified_against_receipts"]}


def record_details() -> pd.DataFrame:
    """Superclass combination, subclasses, SCP codes, report, human validation and heart rate per record."""
    meta = load_metadata()
    statements = load_statements()
    classes = diagnostic_classes(statements)
    diagnostic = statements[statements["diagnostic"] == 1]
    subclass_of = diagnostic["diagnostic_subclass"].to_dict()
    rate = pd.read_parquet(HEART_RATE, columns=["record_id", "heart_rate"]).groupby("record_id")["heart_rate"]
    columns = {
        "combination": meta["scp_codes"].apply(lambda codes: superclasses(codes, classes)),
        "subclasses": meta["scp_codes"].apply(
            lambda codes: sorted({subclass_of[code] for code in codes if code in subclass_of} - {"NORM"})),
        "scp_codes": meta["scp_codes"].apply(json.dumps),
        "report": meta["report"],
        "validated_by_human": meta["validated_by_human"].astype(bool),
        "heart_rate": rate.first().reindex(meta.index.astype(str)),
    }
    return pd.DataFrame({name: values.to_numpy() for name, values in columns.items()},
                        index="ptbxl:" + meta.index.astype(str))


def integrity(train: pd.DataFrame, dev: pd.DataFrame, x_train: np.ndarray,
              x_dev: np.ndarray) -> tuple[StandardScaler, np.ndarray, dict[str, float]]:
    """Refit Experiment 020's ``cpc_standard`` head and reproduce its saved probabilities and AUROC."""
    with np.load(PRIOR / "development_predictions.npz", allow_pickle=True) as saved:
        if not np.array_equal(saved["record_ids"], dev.index.to_numpy()):
            raise ValueError("Experiment 020 development order changed")
        prior = saved["cpc_standard"]
    mask = train["standard"].notna().to_numpy()
    head = fit_logistic(x_train[mask], train.loc[mask, "standard"].to_numpy(dtype=np.int64))
    probabilities = predict(head, x_dev)
    difference = float(np.abs(probabilities - prior).max())
    labeled = dev["standard"].notna().to_numpy()
    auroc = float(roc_auc_score(dev.loc[labeled, "standard"].astype(int), probabilities[labeled]))
    if difference > PROBABILITY_TOLERANCE or abs(auroc - PRIOR_AUROC) > 1e-6:
        raise ValueError(f"cpc_standard does not reproduce Experiment 020: {difference}, {auroc}")
    checks = {"training_rows": int(mask.sum()), "max_abs_difference": difference, "auroc": auroc,
              "development_rows": int(labeled.sum())}
    return head[0], probabilities, checks


def evaluate_task(u_train: np.ndarray, y_train: np.ndarray, devices_train: np.ndarray, u_query: np.ndarray,
                  query: pd.DataFrame, y_query: np.ndarray, probe: np.ndarray,
                  pairs: tuple[tuple[str, str], ...]) -> dict[str, Any]:
    """AUROC of the probe and every distance-based method, and paired bootstrap contrasts."""
    scores = {"probe": probe, **method_scores(u_train, y_train, devices_train, u_query,
                                               query["device"].to_numpy())}
    patients = query["patient_id"].to_numpy()
    bootstrap = {f"{first}_minus_{second}": patient_bootstrap(
        patients, y_query, scores[first], scores[second], draws=BOOTSTRAP_DRAWS, seed=SEED)
        for first, second in pairs}
    return {"training_rows": len(y_train), "training_positives": int(y_train.sum()),
            "query_rows": len(y_query), "query_positives": int(y_query.sum()),
            "query_patients": int(query["patient_id"].nunique()),
            "auroc": {name: float(roc_auc_score(y_query, values)) for name, values in scores.items()},
            "bootstrap": bootstrap}


def task_membership(frame: pd.DataFrame, name: str, level: str) -> np.ndarray:
    """Rows positive for a superclass or subclass task."""
    if level == "superclass":
        return frame["combination"].str.split("+").apply(lambda found: name in found).to_numpy()
    return frame["subclasses"].apply(lambda found: name in found).to_numpy()


def eligible_subclasses(train: pd.DataFrame, dev: pd.DataFrame) -> list[str]:
    """List subclasses with enough positive labeled training and development ECGs."""
    counts = {side: pd.Series([code for found in frame["subclasses"] for code in found]).value_counts()
              for side, frame in (("train", train), ("development", dev))}
    names = [name for name in counts["train"].index
             if counts["train"][name] >= SUBCLASS_MINIMUM["train"]
             and counts["development"].get(name, 0) >= SUBCLASS_MINIMUM["development"]]
    return sorted(names)


def one_vs_norm(train: pd.DataFrame, dev: pd.DataFrame, x_train: np.ndarray, x_dev: np.ndarray,
                u_train: np.ndarray, u_dev: np.ndarray, names: list[str], level: str) -> dict[str, Any]:
    """Probe and distance methods for each task: task present versus NORM-only."""
    results = {}
    for name in names:
        positive_train, positive_dev = task_membership(train, name, level), task_membership(dev, name, level)
        rows_train = positive_train | (train["standard"] == 0).to_numpy()
        rows_dev = positive_dev | (dev["standard"] == 0).to_numpy()
        y_train, y_dev = positive_train[rows_train].astype(int), positive_dev[rows_dev].astype(int)
        probe = predict(fit_logistic(x_train[rows_train], y_train), x_dev[rows_dev])
        devices = train.loc[rows_train, "device"].to_numpy()
        results[name] = evaluate_task(u_train[rows_train], y_train, devices, u_dev[rows_dev], dev[rows_dev],
                                      y_dev, probe, (("prototype", "probe"),))
        print(json.dumps({"stage": f"{level}:{name}", "auroc": results[name]["auroc"]}), flush=True)
    return results


def nuisance(x_train: np.ndarray, x_query: np.ndarray, train: pd.DataFrame, query: pd.DataFrame,
             devices: list[str]) -> dict[str, Any]:
    """Device and sex AUROC with the fixed probe, and age and heart-rate R² with ridge regression."""
    device_auroc = {}
    for device in devices:
        head = fit_logistic(x_train, (train["device"] == device).to_numpy(dtype=np.int64))
        device_auroc[device] = float(roc_auc_score(query["device"] == device, predict(head, x_query)))
    sex = predict(fit_logistic(x_train, train["male"].to_numpy(dtype=np.int64)), x_query)
    known_train, known_query = train["age"].notna().to_numpy(), query["age"].notna().to_numpy()
    return {
        "device_auroc": device_auroc,
        "sex_auroc": float(roc_auc_score(query["male"].to_numpy(dtype=np.int64), sex)),
        "age_r2": ridge_r2(x_train[known_train], train.loc[known_train, "age"].to_numpy(),
                           x_query[known_query], query.loc[known_query, "age"].to_numpy()),
        "heart_rate_r2": ridge_r2(x_train, train["heart_rate"].to_numpy(), x_query,
                                  query["heart_rate"].to_numpy()),
        "rows": {"train": len(train), "query": len(query), "age_train": int(known_train.sum()),
                 "age_query": int(known_query.sum())},
    }


def eligible_devices(frame: pd.DataFrame) -> list[str]:
    """Devices with at least 50 query ECGs."""
    counts = frame["device"].value_counts()
    return sorted(counts[counts >= DEVICE_MINIMUM].index)


def factors(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Categorical factors compared with each unsupervised partition."""
    return {"superclass_combination": frame["combination"].to_numpy(), "device": frame["device"].to_numpy(),
            "sex": frame["male"].astype(int).astype(str).to_numpy(),
            "age_decade": age_decade(frame["age"].to_numpy(dtype=np.float64)),
            "heart_rate_band": heart_rate_band(frame["heart_rate"].to_numpy(dtype=np.float64))}


def cluster_profile(partition: np.ndarray, frame: pd.DataFrame) -> dict[str, Any]:
    """Size, NORM-only share and most common device of each cluster."""
    profile = {}
    for label in np.unique(partition):
        rows = frame[partition == label]
        device = rows["device"].value_counts()
        profile[str(int(label))] = {"records": len(rows),
                                    "norm_only": float((rows["combination"] == "NORM").mean()),
                                    "top_device": device.index[0],
                                    "top_device_share": float(device.iloc[0] / len(rows))}
    return profile


def structure(u_train: np.ndarray, train: pd.DataFrame) -> dict[str, Any]:
    """AMI of k-means and HDBSCAN partitions with diagnosis, device and demographics; no label is fitted."""
    all_factors = factors(train)
    results = {}
    for k in KMEANS_K:
        partition = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(u_train)
        results[f"kmeans_{k}"] = {"records": len(train), "ami": ami_summary(partition, all_factors),
                                  "clusters": cluster_profile(partition, train)}
    pca = PCA(n_components=16, svd_solver="randomized", random_state=SEED).fit(u_train)
    subset = patient_subset(train["patient_id"].to_numpy(), HDBSCAN_PATIENTS)
    vectors = normalize(pca.transform(u_train[subset]))
    partition = HDBSCAN(min_cluster_size=80, min_samples=20, cluster_selection_method="eom",
                        n_jobs=1, copy=True).fit_predict(vectors)
    sampled = train.iloc[subset]
    results["hdbscan_pca16"] = {"records": len(subset), "noise": int((partition == -1).sum()),
                                "pca_variance_fraction": float(pca.explained_variance_ratio_.sum()),
                                "ami": ami_summary(partition, factors(sampled)),
                                "clusters": cluster_profile(partition, sampled)}
    return results


def official_hashes() -> dict[str, str]:
    """Official PTB-XL SHA-256 digests keyed by release-relative path."""
    lines = (PTB_RAW / "SHA256SUMS.txt").read_text().splitlines()
    return {path: digest for digest, path in (line.split(maxsplit=1) for line in lines)}


def plot_ecg(row: pd.Series, group: str, path: Path) -> None:
    """Plot a raw 10 s, 12-lead PTB-XL record on a 0.2 s by 0.5 mV grid, each lead median-centered."""
    signal, fs = read_signal(row["filename_hr"])
    centered = signal - np.median(signal, axis=0)
    limit = max(1.0, 1.05 * float(np.nanmax(np.abs(centered))))
    seconds = np.arange(len(signal)) / fs
    figure, axes = plt.subplots(12, 1, figsize=(11, 14), sharex=True, sharey=True)
    for lead, axis in enumerate(axes):
        axis.plot(seconds, centered[:, lead], color="#0b0b0b", linewidth=0.7)
        axis.set_xticks(np.arange(0, 10.01, 1.0))
        axis.set_xticks(np.arange(0, 10.01, 0.2), minor=True)
        axis.set_yticks([])
        axis.set_yticks(np.arange(-np.floor(limit * 2) / 2, limit, 0.5), minor=True)
        axis.set_ylim(-limit, limit)
        axis.grid(which="both", color="#efc9c9", linewidth=0.4)
        axis.set_ylabel(LEADS[lead], rotation=0, ha="right", va="center")
        axis.tick_params(which="both", left=False)
    axes[-1].set_xlim(0, 10)
    axes[-1].set_xlabel("Time (s); grid 0.2 s by 0.5 mV")
    figure.suptitle(f"{group} reference ECG: PTB-XL ecg_id {row['ecg_id']} ({row['device']}, "
                    f"{fs} Hz, raw, median-centered)", fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(path, dpi=110)
    plt.close(figure)


def reference_ecgs(u_train: np.ndarray, train: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Five most prototypical training ECGs per group and a plot of the first, from verified raw files."""
    official = official_hashes()
    rows, figures = [], {}
    (OUTPUT / "figures").mkdir(parents=True, exist_ok=True)
    for group in GROUPS:
        positions, similarity = medoids(u_train, (train["combination"] == group).to_numpy())
        for rank, (position, value) in enumerate(zip(positions, similarity, strict=True), start=1):
            row = train.iloc[position]
            rows.append({"group": group, "rank": rank, "record_id": row.name, "ecg_id": int(row["ecg_id"]),
                         "patient_id": row["patient_id"], "similarity": float(value), "device": row["device"],
                         "standard": int(row["standard"]), "scp_codes": row["scp_codes"],
                         "validated_by_human": bool(row["validated_by_human"]), "report": row["report"]})
        top = train.iloc[positions[0]]
        for suffix in (".hea", ".dat"):
            name = f"{top['filename_hr']}{suffix}"
            if sha256_file(PTB_RAW / name) != official[name]:
                raise ValueError(f"PTB-XL raw file differs from the official manifest: {name}")
        path = OUTPUT / "figures" / f"reference_{group.lower()}_{int(top['ecg_id'])}.png"
        plot_ecg(top, group, path)
        figures[group] = path.name
    return pd.DataFrame(rows), figures


def encoder_comparison(train: pd.DataFrame, dev: pd.DataFrame
                       ) -> tuple[dict[str, Any], pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Analyses 1 and 2 for each cached encoder on the 025 pool and the original development ECGs."""
    pool, evaluation, counts = label_efficiency.select_rows(train, dev)
    features = label_efficiency.load_features(train, dev, pool, evaluation)
    y_pool = pool["standard"].to_numpy(dtype=np.int64)
    y_eval = evaluation["standard"].to_numpy(dtype=np.int64)
    devices = eligible_devices(evaluation)
    results = {"counts": counts, "devices": devices}
    for name, (x_pool, x_eval) in features.items():
        scaler = StandardScaler().fit(np.asarray(x_pool, dtype=np.float64))
        u_pool, u_eval = unit_space(scaler, x_pool), unit_space(scaler, x_eval)
        probe = predict(fit_logistic(x_pool, y_pool), x_eval)
        task = evaluate_task(u_pool, y_pool, pool["device"].to_numpy(), u_eval, evaluation, y_eval, probe,
                             ENCODER_CONTRASTS)
        neighbors = neighbor_indices(u_pool, u_eval)
        results[name] = {**task, "width": int(x_pool.shape[1]),
                         "nuisance": nuisance(x_pool, x_eval, pool, evaluation, devices),
                         "neighborhood": same_device_summary(neighbors, pool["device"].to_numpy(),
                                                             evaluation["device"].to_numpy())}
        print(json.dumps({"stage": f"encoder:{name}", "auroc": task["auroc"]}), flush=True)
    return results, pool, features


def label_audit(pool: pd.DataFrame, u_cpc: np.ndarray,
                u_jepa: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Training ECGs whose 25 neighbors mostly carry the other standard label in both spaces."""
    y = pool["standard"].to_numpy(dtype=np.int64)
    own = np.arange(len(pool))
    cpc = label_disagreement(neighbor_indices(u_cpc, u_cpc, NEIGHBORS, own, own), y)
    jepa = label_disagreement(neighbor_indices(u_jepa, u_jepa, NEIGHBORS, own, own), y)
    mask = audit_candidates(cpc, jepa)
    frame = pool.assign(cpc_disagreement=cpc, jepa_disagreement=jepa,
                        mean_disagreement=(cpc + jepa) / 2)[mask]
    frame = frame.sort_values(["mean_disagreement", "ecg_id"], ascending=[False, True])
    columns = ["ecg_id", "patient_id", "standard", "combination", "scp_codes", "validated_by_human", "device",
               "cpc_disagreement", "jepa_disagreement", "mean_disagreement", "report"]
    summary = {
        "pool_records": len(pool), "candidates": int(mask.sum()),
        "candidates_by_label": {str(k): int(v) for k, v in frame["standard"].value_counts().items()},
        "candidates_validated_by_human": float(frame["validated_by_human"].mean()),
        "pool_validated_by_human": float(pool["validated_by_human"].mean()),
        "cpc_at_least_0_8": int((cpc >= 0.8).sum()), "jepa_at_least_0_8": int((jepa >= 0.8).sum()),
        "mean_disagreement": {"cpc": float(cpc.mean()), "jepa": float(jepa.mean())},
        "correlation": float(np.corrcoef(cpc, jepa)[0, 1]),
    }
    return frame[columns], summary


def plot_summary(comparison: dict[str, Any], path: Path) -> None:
    """Plot development AUROC by method, one panel per encoder, with the probe as a reference line."""
    methods = ["probe", "prototype", "prototype_device_balanced", "multi_prototype_2", "multi_prototype_4",
               "multi_prototype_8", "knn", "knn_other_device"]
    titles = {"cpc": "CPC (ours)", "jepa": "ECG-JEPA", "xecg": "xECG", "released_cpc": "Released ECG-CPC"}
    figure, axes = plt.subplots(1, 4, figsize=(13, 4.2), sharey=True, sharex=True)
    for axis, name in zip(axes, titles, strict=True):
        values = [comparison[name]["auroc"][method] for method in methods]
        axis.axvline(values[0], color="#52514e", linewidth=1, linestyle="--")
        axis.scatter(values, range(len(methods)), s=40, color="#2a78d6", zorder=3)
        for position, value in enumerate(values):
            axis.annotate(f"{value:.3f}", (value, position), xytext=(6, 0), textcoords="offset points",
                          va="center", fontsize=8, color="#0b0b0b")
        axis.set_title(titles[name], fontsize=10)
        axis.grid(axis="x", color="#e5e4e0", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(range(len(methods)), methods)
    axes[0].invert_yaxis()
    for axis in axes:
        axis.set_xlabel("Development AUROC")
    figure.suptitle("Experiment 024: probe (dashed) versus distance-based scores, 1,306 original "
                    "development ECGs, standard label", fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def cpc_analyses(train: pd.DataFrame, dev: pd.DataFrame, x_train: np.ndarray, x_dev: np.ndarray,
                 scaler: StandardScaler, probabilities: np.ndarray) -> dict[str, Any]:
    """Analyses 1 and 2 on the CPC space of Experiment 020."""
    labeled_train = train["standard"].notna().to_numpy()
    labeled_dev = dev["standard"].notna().to_numpy()
    train_l, dev_l = train[labeled_train], dev[labeled_dev]
    u_train, u_dev = unit_space(scaler, x_train[labeled_train]), unit_space(scaler, x_dev[labeled_dev])
    y_train, y_dev = train_l["standard"].to_numpy(dtype=np.int64), dev_l["standard"].to_numpy(dtype=np.int64)
    binary = evaluate_task(u_train, y_train, train_l["device"].to_numpy(), u_dev, dev_l, y_dev,
                           probabilities[labeled_dev], BINARY_CONTRASTS)
    print(json.dumps({"stage": "binary", "auroc": binary["auroc"]}), flush=True)
    subclasses = eligible_subclasses(train_l, dev_l)
    neighbors = neighbor_indices(u_train, u_dev)
    return {
        "binary": binary,
        "superclasses": one_vs_norm(train_l, dev_l, x_train[labeled_train], x_dev[labeled_dev], u_train,
                                    u_dev, list(SUPERCLASS_TASKS), "superclass"),
        "subclass_names": subclasses,
        "subclasses": one_vs_norm(train_l, dev_l, x_train[labeled_train], x_dev[labeled_dev], u_train, u_dev,
                                  subclasses, "subclass"),
        "nuisance": nuisance(x_train, x_dev, train, dev, eligible_devices(dev)),
        "neighborhood": same_device_summary(neighbors, train_l["device"].to_numpy(),
                                            dev_l["device"].to_numpy()),
    }


def main() -> None:
    """Run the frozen development-only embedding geometry audit once."""
    if (OUTPUT / "result.json").exists():
        raise FileExistsError("Experiment 024 v1 has already run")
    start = time.monotonic()
    timings = {}
    run_identity = identity()
    groups = cohorts(ptb_table())
    details = record_details()
    train, dev = groups["train"].join(details), groups["development"].join(details)
    with np.load(PRIOR / "features.npz") as saved:
        x_train, x_dev = saved["train"], saved["development"]
    if x_train.shape != (len(train), 512) or x_dev.shape != (len(dev), 512):
        raise ValueError("The Experiment 020 CPC features do not match the cohort")
    with threadpool_limits(limits=THREADS):
        scaler, probabilities, checks = integrity(train, dev, x_train, x_dev)
        print(json.dumps({"stage": "integrity", **checks}), flush=True)
        cpc = cpc_analyses(train, dev, x_train, x_dev, scaler, probabilities)
        timings["cpc_analyses"] = time.monotonic() - start
        labeled = train["standard"].notna().to_numpy()
        clusters = structure(unit_space(scaler, x_train), train)
        timings["structure"] = time.monotonic() - start
        medoid_table, figures = reference_ecgs(unit_space(scaler, x_train[labeled]), train[labeled])
        comparison, pool, features = encoder_comparison(train, dev)
        timings["encoder_comparison"] = time.monotonic() - start
        jepa_scaler = StandardScaler().fit(np.asarray(features["jepa"][0], dtype=np.float64))
        audit, audit_summary = label_audit(pool, unit_space(scaler, features["cpc"][0]),
                                           unit_space(jepa_scaler, features["jepa"][0]))
    medoid_table.to_csv(OUTPUT / "medoids.csv", index=False)
    audit.to_csv(OUTPUT / "label_audit.csv", index=False)
    plot_summary(comparison, OUTPUT / "figures" / "auroc_by_method_and_encoder.png")
    figures["summary"] = "auroc_by_method_and_encoder.png"
    result = {
        "status": "complete_development_only", "identity": run_identity, "integrity": checks,
        "counts": {"training_rows": len(train), "training_labeled": int(labeled.sum()),
                   "development_rows": len(dev), "development_labeled": int(dev["standard"].notna().sum()),
                   "development_patients": int(dev["patient_id"].nunique())},
        "cpc": cpc, "structure": clusters, "encoder_comparison": comparison, "label_audit": audit_summary,
        "label_audit_shown": AUDIT_SHOWN, "figures": figures,
        "figure_sha256": {name: sha256_file(OUTPUT / "figures" / file) for name, file in figures.items()},
        "medoids_sha256": sha256_file(OUTPUT / "medoids.csv"),
        "label_audit_sha256": sha256_file(OUTPUT / "label_audit.csv"),
        "stage_seconds": timings, "total_seconds": time.monotonic() - start,
        "calibration_test_evaluated": False,
    }
    write_json_atomic(OUTPUT / "result.json", result)
    print(json.dumps({"stage": "complete", "total_seconds": result["total_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
