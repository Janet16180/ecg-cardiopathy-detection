"""Experiment 024b: embedding geometry of PTB-XL plus the Challenge hospitals, with 024 as the reference."""

from __future__ import annotations

import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score, homogeneity_score
from sklearn.preprocessing import StandardScaler, normalize
from threadpoolctl import threadpool_limits

from ecg_experiment.challenge_features import RAW_ROOTS, canonical_window, read_verified
from ecg_experiment.downloads import parse_checksums
from ecg_experiment.eda.ptbxl import read_signal
from ecg_experiment.eda.signals import LEADS
from ecg_experiment.embedding_geometry import (
    SEED as KMEANS_SEED,
)
from ecg_experiment.embedding_geometry import (
    ami_summary,
    device_balanced_scores,
    medoids,
    prototype_scores,
    unit_space,
)
from ecg_experiment.external_readout import prior_cpc_features, prior_identity
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.full_development import cohorts, fit_logistic, predict, ptb_table
from ecg_experiment.multisource_geometry import (
    bootstrap_partition,
    bootstrap_source_probe,
    challenge_combination,
    chunked_neighbors,
    equal_family_draw,
    explained_entropy,
    neighbor_shares,
    one_vs_rest_probabilities,
    paired_interval,
)
from ecg_experiment.multisource_manifold import equal_family_subsample
from ecg_experiment.multisource_readout import fit_readout
from ecg_experiment.normal_manifold import bootstrap_metrics, contrast, metrics, patient_resamples
from scripts.experiments import run_embedding_geometry024 as geometry024
from scripts.experiments import run_label_efficiency025 as label_efficiency
from scripts.experiments.run_calibrated_threshold027 import prior022_identity, sph_rows
from scripts.experiments.run_multisource_calibration027b import feature_identity
from scripts.experiments.run_multisource_manifold026b import challenge_features, interval_side, split_table
from scripts.experiments.run_multisource_readout022b import sph_features
from scripts.experiments.run_sph_external022 import open_caches, ptb_encoder_features

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/experiment024b_multisource_geometry_v1"
FIGURES = ROOT / "docs/figures/experiment-024b"
PRIOR022 = ROOT / "outputs/experiment022_sph_external_v3"
PRIOR022B = ROOT / "outputs/experiment022b_multisource_readout_v1"
PRIOR024 = ROOT / "outputs/experiment024_embedding_geometry_v1"
SPLITS = ROOT / "data/processed/challenge_splits_v1"
PTB_RAW = ROOT / "data/raw/ptb-xl/1.0.3"
ENCODERS = ("xecg", "jepa", "cpc")
PRIMARY_ENCODER = "xecg"
PRIMARY_K = 8
KMEANS_K = (2, 4, 8, 16)
SEED = 33033
BOOTSTRAP_DRAWS = 2000
THREADS = 4
FAMILIES = ["chapman_ningbo", "cpsc", "georgia", "ptbxl"]
SOURCE_NAMES = ["chapman_shaoxing", "cpsc_2018", "cpsc_2018_extra", "georgia", "ningbo", "ptbxl"]
GROUPS = ("NORM", "MI", "STTC", "CD", "HYP")
HDBSCAN_PER_FAMILY = 1000
MEDOID_COUNT = 50
SMALL_AMI = 0.02
AMI_TOLERANCE = 1e-6
REPRODUCTION_TOLERANCE = 1e-9
EXPECTED_POOLED = {"ptbxl": (17083, 9840), "chapman_ningbo": (13002, 9602), "georgia": (5136, 4113),
                   "cpsc": (4356, 3805)}
EXPECTED_DRAW = {"ptbxl": (4356, 2498), "chapman_ningbo": (4356, 3181), "georgia": (4356, 3485),
                 "cpsc": (4356, 3805)}
EXPECTED_HELDOUT = {"ptbxl": (1572, 884), "chapman_ningbo": (4432, 3254), "georgia": (1718, 1372),
                    "cpsc": (1462, 1279)}
EXPECTED_SPH = (21008, 7190)
SPH_METHODS = {"ptbxl": ("probe", "prototype", "knn"),
               "pooled": ("probe", "prototype", "prototype_family_balanced", "knn")}
SOURCES = (
    "ecg_experiment/multisource_geometry.py", "ecg_experiment/embedding_geometry.py",
    "ecg_experiment/multisource_manifold.py", "ecg_experiment/multisource_readout.py",
    "ecg_experiment/external_readout.py", "ecg_experiment/full_development.py",
    "ecg_experiment/normal_manifold.py", "ecg_experiment/challenge_features.py",
    "ecg_experiment/eda/ptbxl.py", "ecg_experiment/files.py",
    "scripts/experiments/run_multisource_geometry024b.py", "scripts/experiments/run_embedding_geometry024.py",
    "scripts/experiments/run_label_efficiency025.py", "scripts/experiments/run_multisource_readout022b.py",
    "scripts/experiments/run_multisource_manifold026b.py",
    "scripts/experiments/run_calibrated_threshold027.py",
    "scripts/experiments/run_multisource_calibration027b.py", "scripts/experiments/run_sph_external022.py",
    "pyproject.toml", "uv.lock", "docs/experiment-024b-multisource-geometry.md",
)


def identity() -> tuple[dict[str, Any], Any, dict[str, Any]]:
    """
    Hash every input and require the reused artifacts to match their receipts.

    Returns
    -------
    tuple[dict[str, Any], Any, dict[str, Any]]
        The identity record, 022's opened JEPA and xECG caches, and 022's result.

    Raises
    ------
    ValueError
        If an artifact differs from the hash its producer recorded.
    """
    result022b = json.loads((PRIOR022B / "result.json").read_text())
    for name, digest in result022b["outputs_sha256"].items():
        if sha256_file(PRIOR022B / name) != digest:
            raise ValueError(f"Experiment 022b {name} differs from its receipt")
    result024 = json.loads((PRIOR024 / "result.json").read_text())
    if sha256_file(PRIOR024 / "medoids.csv") != result024["medoids_sha256"]:
        raise ValueError("Experiment 024 medoids.csv differs from its receipt")
    prior_result = json.loads((PRIOR022 / "result.json").read_text())
    caches, cache_hashes = open_caches()
    if cache_hashes != prior_result["identity"]["feature_caches"]:
        raise ValueError("Feature caches differ from Experiment 022")
    feature_metadata, feature_hashes = feature_identity()
    record = {
        "experiment022": prior022_identity(), "experiment020": prior_identity(),
        "experiment022_features_sha256": sha256_file(PRIOR022 / "features.npz"),
        "experiment022b": {name: sha256_file(PRIOR022B / name)
                           for name in ("result.json", "predictions.npz", "training_rows.csv")},
        "experiment024": {name: sha256_file(PRIOR024 / name) for name in ("result.json", "medoids.csv")},
        "experiment025_caches": label_efficiency.identity(),
        "feature_caches": cache_hashes, "challenge_features": feature_hashes,
        "challenge_features_identity": feature_metadata["identity"],
        "challenge_split": {name: sha256_file(SPLITS / name) for name in ("rows.csv", "metadata.json")},
        "ptbxl_sha256sums": sha256_file(PTB_RAW / "SHA256SUMS.txt"),
        "sources": {name: sha256_file(ROOT / name) for name in SOURCES},
    }
    return record, caches, prior_result


def reproduce_024(train: pd.DataFrame, dev: pd.DataFrame) -> dict[str, Any]:
    """
    R1-R3: 024's CPC k-means AMI, encoder prototype-versus-probe comparison and CPC medoids.

    Parameters
    ----------
    train, dev : pd.DataFrame
        ``full_development.cohorts`` rows joined with 024's ``record_details``.

    Returns
    -------
    dict[str, Any]
        The reproduced values and their largest differences from 024.

    Raises
    ------
    ValueError
        If a value differs from 024 by more than its tolerance.
    """
    prior = json.loads((PRIOR024 / "result.json").read_text())
    x_train, x_dev = prior_cpc_features()["train"], prior_cpc_features()["development"]
    scaler, _, integrity = geometry024.integrity(train, dev, x_train, x_dev)
    u_all = unit_space(scaler, x_train)
    structure, worst_ami = {}, 0.0
    for k in KMEANS_K:
        partition = KMeans(n_clusters=k, n_init=10, random_state=KMEANS_SEED).fit_predict(u_all)
        ami = ami_summary(partition, {"superclass_combination": train["combination"].to_numpy(),
                                      "device": train["device"].to_numpy()})
        saved = prior["structure"][f"kmeans_{k}"]["ami"]
        worst_ami = max(worst_ami, *(abs(value - saved[name]) for name, value in ami.items()))
        structure[f"kmeans_{k}"] = ami
    print(json.dumps({"stage": "reproduce:structure", "worst": worst_ami}), flush=True)

    labeled = train["standard"].notna().to_numpy()
    u_labeled, frame = unit_space(scaler, x_train[labeled]), train[labeled]
    saved_medoids = pd.read_csv(PRIOR024 / "medoids.csv")
    medoid_match, worst_similarity = True, 0.0
    for group in GROUPS:
        positions, similarity = medoids(u_labeled, (frame["combination"] == group).to_numpy())
        expected = saved_medoids[saved_medoids["group"] == group].sort_values("rank")
        medoid_match &= np.array_equal(frame["ecg_id"].to_numpy()[positions], expected["ecg_id"].to_numpy())
        worst_similarity = max(worst_similarity, float(np.abs(similarity - expected["similarity"]).max()))

    pool, evaluation, _ = label_efficiency.select_rows(train, dev)
    features = label_efficiency.load_features(train, dev, pool, evaluation)
    y_pool = pool["standard"].to_numpy(dtype=np.int64)
    y_eval = evaluation["standard"].to_numpy(dtype=np.int64)
    comparison, worst_auroc, interval_match = {}, 0.0, True
    for name in ENCODERS:
        x_pool, x_eval = features[name]
        pool_scaler = StandardScaler().fit(np.asarray(x_pool, dtype=np.float64))
        probe = predict(fit_logistic(x_pool, y_pool), x_eval)
        task = geometry024.evaluate_task(unit_space(pool_scaler, x_pool), y_pool, pool["device"].to_numpy(),
                                         unit_space(pool_scaler, x_eval), evaluation, y_eval, probe,
                                         (("prototype", "probe"),))
        saved = prior["encoder_comparison"][name]
        worst_auroc = max(worst_auroc, *(abs(value - saved["auroc"][method])
                                         for method, value in task["auroc"].items()))
        interval = task["bootstrap"]["prototype_minus_probe"]
        interval_match &= interval == saved["bootstrap"]["prototype_minus_probe"]
        comparison[name] = {"auroc": task["auroc"], "prototype_minus_probe": interval}
        print(json.dumps({"stage": f"reproduce:encoder:{name}", "worst_auroc": worst_auroc}), flush=True)

    checks = {"integrity": integrity, "structure_worst_ami_difference": worst_ami,
              "medoids_identical": bool(medoid_match),
              "medoid_worst_similarity_difference": worst_similarity,
              "encoder_worst_auroc_difference": worst_auroc,
              "encoder_intervals_identical": bool(interval_match)}
    if worst_ami > AMI_TOLERANCE or not medoid_match or worst_similarity > REPRODUCTION_TOLERANCE or \
            worst_auroc > REPRODUCTION_TOLERANCE or not interval_match:
        raise ValueError(f"Experiment 024 is not reproduced: {checks}")
    return {"checks": checks, "structure": structure, "encoder_comparison": comparison}


def challenge_rows() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """
    Challenge training rows kept by 022b's quality flags, and the calibration rows, with features.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]
        Kept training rows, calibration rows, and per encoder the features of kept training rows followed by
        the calibration rows.

    Raises
    ------
    ValueError
        If the training rows differ from 022b's saved training rows.
    """
    rows, x = challenge_features(split_table())
    in_train = (rows["split"] == "train").to_numpy()
    in_calibration = (rows["split"] == "calibration").to_numpy()
    train = rows[in_train].reset_index(drop=True)
    saved = pd.read_csv(PRIOR022B / "training_rows.csv", dtype={"record": str},
                        usecols=["source", "record", "kept"])
    if not (np.array_equal(saved["source"], train["source"])
            and np.array_equal(saved["record"], train["record"])):
        raise ValueError("Challenge training rows differ from Experiment 022b")
    kept = saved["kept"].to_numpy(dtype=bool)
    features = {name: np.concatenate([values[in_train][kept], values[in_calibration]])
                for name, values in x.items()}
    return train[kept].reset_index(drop=True), rows[in_calibration].reset_index(drop=True), features


def assemble(caches: Any
             ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, np.ndarray]], pd.DataFrame]:
    """
    Pooled training rows (022b's order), held-out rows and their features.

    Parameters
    ----------
    caches : Any
        Output of 022's ``open_caches``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, np.ndarray]], pd.DataFrame]
        Pooled rows and held-out rows (``record_id``, ``family``, ``source``, ``y``, ``combination`` and
        plotting columns), features per encoder keyed by ``pool`` and ``heldout``, and the PTB-XL training
        rows with a standard label.
    """
    groups = cohorts(ptb_table())
    details = geometry024.record_details()
    ptb_train, ptb_dev = groups["train"].join(details), groups["development"].join(details)
    standard, labeled = ptb_train["standard"].notna().to_numpy(), ptb_dev["standard"].notna().to_numpy()
    with np.load(PRIOR022 / "ptb_features.npz") as saved:
        extracted = {name: saved[name] for name in saved.files}
    encoded = {**ptb_encoder_features(groups, caches, extracted), "cpc": prior_cpc_features()}
    challenge_train, challenge_calibration, challenge_x = challenge_rows()

    def ptb_frame(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"record_id": frame.index.to_numpy(dtype=str), "family": "ptbxl",
                             "source": "ptbxl",
                             "y": frame["standard"].to_numpy(dtype=np.int64),
                             "combination": frame["combination"].to_numpy(dtype=str),
                             "patient_id": frame["patient_id"].to_numpy(dtype=str),
                             "device": frame["device"].to_numpy(dtype=str),
                             "filename_hr": frame["filename_hr"].to_numpy(dtype=str),
                             "path": "", "window_start": 0})

    def challenge_frame(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"record_id": (frame["source"] + ":" + frame["record"]).to_numpy(dtype=str),
                             "family": frame["family"].to_numpy(dtype=str),
                             "source": frame["source"].to_numpy(dtype=str),
                             "y": frame["primary"].to_numpy(dtype=np.int64),
                             "combination": challenge_combination(frame),
                             "patient_id": (frame["source"] + ":" + frame["record"]).to_numpy(dtype=str),
                             "device": "", "filename_hr": "", "path": frame["path"].to_numpy(dtype=str),
                             "window_start": frame["window_start"].to_numpy(dtype=np.int64)})

    pool = pd.concat([ptb_frame(ptb_train[standard]), challenge_frame(challenge_train)], ignore_index=True)
    heldout = pd.concat([ptb_frame(ptb_dev[labeled]), challenge_frame(challenge_calibration)],
                        ignore_index=True)
    split = len(challenge_train)
    features = {name: {"pool": np.concatenate([encoded[name]["train"][standard], challenge_x[name][:split]]),
                       "heldout": np.concatenate([encoded[name]["development"][labeled],
                                                  challenge_x[name][split:]])}
                for name in ENCODERS}
    return pool, heldout, features, ptb_train[standard]


def family_counts(frame: pd.DataFrame) -> dict[str, tuple[int, int]]:
    """Count records and positives per family."""
    return {family: (len(part), int(part["y"].sum())) for family, part in frame.groupby("family")}


def check_counts(pool: pd.DataFrame, draw: np.ndarray, heldout: pd.DataFrame,
                 sph: pd.DataFrame) -> dict[str, Any]:
    """
    Require the pooled, balanced-draw, held-out and SPH counts of the protocol.

    Parameters
    ----------
    pool : pd.DataFrame
        Pooled training rows.
    draw : np.ndarray
        Positions of the balanced draw in ``pool``.
    heldout : pd.DataFrame
        Held-out rows.
    sph : pd.DataFrame
        SPH rows with ``primary``.

    Returns
    -------
    dict[str, Any]
        The counts found.

    Raises
    ------
    ValueError
        If a count differs.
    """
    found = {"pooled": family_counts(pool), "draw": family_counts(pool.iloc[draw]),
             "heldout": family_counts(heldout), "sph": (len(sph), int(sph["primary"].sum()))}
    expected = {"pooled": EXPECTED_POOLED, "draw": EXPECTED_DRAW, "heldout": EXPECTED_HELDOUT,
                "sph": EXPECTED_SPH}
    if found != expected:
        raise ValueError(f"Counts differ from the protocol: {found}")
    found["draw_sources"] = pool.iloc[draw]["source"].value_counts().to_dict()
    return found


def sph_geometry(x: np.ndarray, pool: dict[str, np.ndarray], sph: dict[str, np.ndarray],
                 saved: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """
    Analysis 2: probe, prototype and kNN scores at SPH for the ``ptbxl`` and ``pooled`` training sets.

    Parameters
    ----------
    x : np.ndarray
        Pooled training features.
    pool : dict[str, np.ndarray]
        ``y`` and ``family`` of the pooled rows.
    sph : dict[str, np.ndarray]
        SPH ``x``, ``y`` and ``patients``.
    saved : dict[str, np.ndarray]
        022b's SPH probabilities per training set.

    Returns
    -------
    tuple[dict[str, Any], dict[str, np.ndarray]]
        Metrics, contrasts and reproduction differences; and the pooled readout-space unit vectors of the
        pooled rows, for the medoids.

    Raises
    ------
    ValueError
        If a refitted readout differs from 022b's saved SPH probabilities.
    """
    selections = {"ptbxl": pool["family"] == "ptbxl", "pooled": np.ones(len(x), dtype=bool)}
    scores, differences, spaces = {}, {}, {}
    for arm, selected in selections.items():
        y = pool["y"][selected]
        head = fit_readout(x[selected], y)
        scores[f"{arm}:probe"] = predict(head, sph["x"])
        differences[arm] = float(np.abs(scores[f"{arm}:probe"] - saved[arm]).max())
        if differences[arm] > REPRODUCTION_TOLERANCE:
            raise ValueError(f"The {arm} readout does not reproduce Experiment 022b: {differences[arm]}")
        u_train, u_sph = unit_space(head[0], x[selected]), unit_space(head[0], sph["x"])
        scores[f"{arm}:prototype"] = prototype_scores(u_train, y, u_sph)
        scores[f"{arm}:knn"] = y[chunked_neighbors(u_train, u_sph)].mean(axis=1)
        if arm == "pooled":
            scores["pooled:prototype_family_balanced"] = device_balanced_scores(u_train, y, pool["family"],
                                                                                u_sph)
        spaces[arm] = u_train
    observed = {name: metrics(sph["y"], values) for name, values in scores.items()}
    resamples, invalid = patient_resamples(sph["patients"], sph["y"], BOOTSTRAP_DRAWS, SEED)
    resampled = bootstrap_metrics(sph["y"], scores, resamples)
    pairs = [("ptbxl:prototype", "ptbxl:probe"), ("pooled:prototype", "pooled:probe"),
             ("ptbxl:knn", "ptbxl:probe"), ("pooled:knn", "pooled:probe"),
             ("pooled:prototype_family_balanced", "pooled:prototype"), ("pooled:probe", "ptbxl:probe"),
             ("pooled:prototype", "ptbxl:prototype")]
    contrasts = {f"{first}_minus_{second}": contrast(observed, resampled, first, second)
                 for first, second in pairs}
    change = {}
    for metric in ("auroc", "average_precision"):
        gap = {arm: (observed[f"{arm}:prototype"][metric] - observed[f"{arm}:probe"][metric],
                     resampled[f"{arm}:prototype"][metric] - resampled[f"{arm}:probe"][metric])
               for arm in selections}
        change[metric] = paired_interval(gap["pooled"][0] - gap["ptbxl"][0], gap["pooled"][1],
                                         gap["ptbxl"][1])
    contrasts["gap_change_pooled_minus_ptbxl"] = change
    return ({"metrics": observed, "contrasts": contrasts, "invalid_draws": invalid,
             "reproduction_022b": differences}, spaces)


def partition_summary(partition: np.ndarray, factors: dict[str, np.ndarray], y: np.ndarray,
                      families: np.ndarray, bootstrap: bool) -> dict[str, Any]:
    """
    AMI, entropy explained, fixed-label and fixed-family AMI and cluster make-up of one partition.

    Parameters
    ----------
    partition : np.ndarray
        Cluster label of each draw row.
    factors : dict[str, np.ndarray]
        ``family``, ``source``, ``standard`` and ``combination`` of the same rows.
    y : np.ndarray
        Standard label.
    families : np.ndarray
        Family of each row.
    bootstrap : bool
        Whether to add record-bootstrap intervals.

    Returns
    -------
    dict[str, Any]
        The partition's summaries.
    """
    ami = ami_summary(partition, factors)
    entropy = explained_entropy(partition, factors)
    within_label = {str(label): float(adjusted_mutual_info_score(families[y == label], partition[y == label]))
                    for label in (0, 1)}
    within_family = {family: float(adjusted_mutual_info_score(y[families == family],
                                                              partition[families == family]))
                     for family in FAMILIES}
    clusters = {}
    for label in np.unique(partition):
        members = partition == label
        shares = pd.Series(families[members]).value_counts(normalize=True)
        clusters[str(int(label))] = {"records": int(members.sum()), "positive": float(y[members].mean()),
                                     "top_family": shares.index[0], "top_family_share": float(shares.iloc[0])}
    summary = {"ami": ami, "entropy_explained": entropy, "family_ami_within_label": within_label,
               "standard_ami_within_family": within_family, "clusters": clusters}
    if bootstrap:
        drawn = bootstrap_partition(partition, factors, BOOTSTRAP_DRAWS, SEED)
        kept = {name: factors[name] for name in ("family", "standard")}
        drawn_entropy = bootstrap_partition(partition, kept, BOOTSTRAP_DRAWS, SEED, homogeneity_score)
        summary["ami_intervals"] = {name: [float(v) for v in np.percentile(values, [2.5, 97.5])]
                                    for name, values in drawn.items()}
        summary["contrasts"] = {
            "family_minus_standard": paired_interval(ami["family"] - ami["standard"], drawn["family"],
                                                     drawn["standard"]),
            "family_minus_combination": paired_interval(ami["family"] - ami["combination"], drawn["family"],
                                                        drawn["combination"]),
            "entropy_family_minus_standard": paired_interval(
                entropy["family"] - entropy["standard"], drawn_entropy["family"], drawn_entropy["standard"]),
        }
    return summary


def structure(u_draw: np.ndarray, frame: pd.DataFrame, subset: np.ndarray) -> dict[str, Any]:
    """
    Analysis 1: k-means and HDBSCAN partitions of the balanced draw against hospital and diagnosis.

    Parameters
    ----------
    u_draw : np.ndarray
        Draw-space unit vectors of the balanced draw.
    frame : pd.DataFrame
        The draw rows, with ``family``, ``source``, ``y`` and ``combination``.
    subset : np.ndarray
        Positions within the draw of the HDBSCAN subset.

    Returns
    -------
    dict[str, Any]
        Summary per partition.
    """
    y, families = frame["y"].to_numpy(), frame["family"].to_numpy(dtype=str)
    factors = {"family": families, "source": frame["source"].to_numpy(dtype=str),
               "standard": y.astype(str), "combination": frame["combination"].to_numpy(dtype=str)}
    results = {}
    for k in KMEANS_K:
        partition = KMeans(n_clusters=k, n_init=10, random_state=KMEANS_SEED).fit_predict(u_draw)
        results[f"kmeans_{k}"] = partition_summary(partition, factors, y, families, bootstrap=True)
    pca = PCA(n_components=16, svd_solver="randomized", random_state=SEED).fit(u_draw)
    partition = HDBSCAN(min_cluster_size=80, min_samples=20, cluster_selection_method="eom", n_jobs=1,
                        copy=True).fit_predict(normalize(pca.transform(u_draw[subset])))
    results["hdbscan_pca16"] = {
        **partition_summary(partition, {name: values[subset] for name, values in factors.items()}, y[subset],
                            families[subset], bootstrap=False),
        "records": len(subset), "noise": int((partition == -1).sum()),
        "pca_variance_fraction": float(pca.explained_variance_ratio_.sum())}
    return results


def ptbxl_structure(x: np.ndarray, frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    Within-PTB-XL k-means AMI with device, standard label and superclass combination.

    Parameters
    ----------
    x : np.ndarray
        Features of the PTB-XL training rows with a standard label.
    frame : pd.DataFrame
        The same rows, with ``device``, ``y`` and ``combination``.

    Returns
    -------
    dict[str, dict[str, float]]
        AMI per factor for each k.
    """
    u = unit_space(StandardScaler().fit(np.asarray(x, dtype=np.float64)), x)
    factors = {"device": frame["device"].to_numpy(dtype=str), "standard": frame["y"].to_numpy().astype(str),
               "combination": frame["combination"].to_numpy(dtype=str)}
    return {f"kmeans_{k}": ami_summary(
        KMeans(n_clusters=k, n_init=10, random_state=KMEANS_SEED).fit_predict(u), factors) for k in KMEANS_K}


def source_probe(x_draw: np.ndarray, draw: pd.DataFrame, x_heldout: np.ndarray,
                 heldout: pd.DataFrame) -> dict[str, Any]:
    """
    Analysis 3: how well the fixed logistic readout identifies the family and the source.

    Parameters
    ----------
    x_draw, x_heldout : np.ndarray
        Raw features of the balanced draw and the held-out rows.
    draw, heldout : pd.DataFrame
        The same rows with ``family`` and ``source``.

    Returns
    -------
    dict[str, Any]
        Metrics with record-bootstrap intervals, by family and by source.
    """
    results = {}
    for column, names in (("family", FAMILIES), ("source", SOURCE_NAMES)):
        probabilities = one_vs_rest_probabilities(x_draw, draw[column].to_numpy(dtype=str), x_heldout, names)
        results[column] = bootstrap_source_probe(heldout[column].to_numpy(dtype=str), probabilities, names,
                                                 BOOTSTRAP_DRAWS, SEED)
    return results


def neighbor_checks(u_draw: np.ndarray, draw: pd.DataFrame, u_heldout: np.ndarray, heldout: pd.DataFrame,
                    u_sph: np.ndarray) -> dict[str, Any]:
    """
    Same-family neighbor shares, ``knn`` against ``knn_other_family``, and SPH's neighbor families.

    Parameters
    ----------
    u_draw, u_heldout, u_sph : np.ndarray
        Draw-space unit vectors.
    draw, heldout : pd.DataFrame
        Draw and held-out rows with ``family`` and ``y``.

    Returns
    -------
    dict[str, Any]
        Per held-out family: neighbor shares by family, and ``knn`` and ``knn_other_family`` metrics and
        contrast; and SPH's neighbor shares by family.
    """
    families, y = draw["family"].to_numpy(dtype=str), draw["y"].to_numpy()
    query_families = heldout["family"].to_numpy(dtype=str)
    neighbors = chunked_neighbors(u_draw, u_heldout)
    other = chunked_neighbors(u_draw, u_heldout, families, query_families)
    scores = {"knn": y[neighbors].mean(axis=1), "knn_other_family": y[other].mean(axis=1)}
    results = {}
    for family in FAMILIES:
        rows = query_families == family
        labels = heldout.loc[rows, "y"].to_numpy()
        chosen = {name: values[rows] for name, values in scores.items()}
        observed = {name: metrics(labels, values) for name, values in chosen.items()}
        units = np.arange(rows.sum()).astype(str)
        resamples, invalid = patient_resamples(units, labels, BOOTSTRAP_DRAWS, SEED)
        resampled = bootstrap_metrics(labels, chosen, resamples)
        results[family] = {
            "queries": int(rows.sum()),
            "neighbor_shares": neighbor_shares(neighbors[rows], families, FAMILIES),
            "metrics": observed, "invalid_draws": invalid,
            "knn_other_family_minus_knn": contrast(observed, resampled, "knn_other_family", "knn")}
    sph_neighbors = chunked_neighbors(u_draw, u_sph)
    return {"heldout": results, "sph_neighbor_shares": neighbor_shares(sph_neighbors, families, FAMILIES)}


def medoid_families(u_pool: np.ndarray, pool: pd.DataFrame) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """
    Analysis 4: families of the pooled members closest to each group prototype.

    Parameters
    ----------
    u_pool : np.ndarray
        Pooled readout-space unit vectors of the pooled rows.
    pool : pd.DataFrame
        Pooled rows with ``family`` and ``combination``.

    Returns
    -------
    tuple[dict[str, Any], dict[str, np.ndarray]]
        Per group: family counts of the top 5 and top 50 and the group's family shares; and the positions and
        similarities of the top 50.
    """
    families = pool["family"].to_numpy(dtype=str)
    summary, ranked = {}, {}
    for group in GROUPS:
        mask = (pool["combination"] == group).to_numpy()
        positions, similarity = medoids(u_pool, mask, MEDOID_COUNT)
        ranked[group] = np.column_stack([positions, similarity])
        summary[group] = {
            "members": int(mask.sum()),
            "top5": pd.Series(families[positions[:5]]).value_counts().to_dict(),
            "top50": pd.Series(families[positions]).value_counts().to_dict(),
            "group_shares": pd.Series(families[mask]).value_counts(normalize=True).to_dict(),
            "top5_one_family": bool(len(set(families[positions[:5]])) == 1)}
    return summary, ranked


def evaluate_encoder(name: str, x_pool: np.ndarray, x_heldout: np.ndarray, x_sph: np.ndarray,
                     pool: pd.DataFrame, heldout: pd.DataFrame, sph: dict[str, np.ndarray], draw: np.ndarray,
                     subset: np.ndarray, saved: dict[str, np.ndarray]) -> dict[str, Any]:
    """
    Every analysis of the protocol for one encoder, with one BLAS thread.

    Parameters
    ----------
    name : str
        Encoder name.
    x_pool, x_heldout, x_sph : np.ndarray
        Features of the pooled, held-out and SPH rows.
    pool, heldout : pd.DataFrame
        Pooled and held-out rows.
    sph : dict[str, np.ndarray]
        SPH ``y`` and ``patients``.
    draw : np.ndarray
        Positions of the balanced draw in ``pool``.
    subset : np.ndarray
        Positions of the HDBSCAN subset within the draw.
    saved : dict[str, np.ndarray]
        022b's SPH probabilities of the ``ptbxl`` and ``pooled`` readouts.

    Returns
    -------
    dict[str, Any]
        Results and the medoid rankings.
    """
    started = time.monotonic()
    with threadpool_limits(limits=1):
        pooled = {"y": pool["y"].to_numpy(), "family": pool["family"].to_numpy(dtype=str)}
        sph_result, spaces = sph_geometry(x_pool, pooled, {**sph, "x": x_sph}, saved)
        print(json.dumps({"stage": f"{name}:sph", "auroc": {key: round(value["auroc"], 4) for key, value in
                                                            sph_result["metrics"].items()}}), flush=True)
        medoid_summary, ranked = medoid_families(spaces["pooled"], pool)
        frame = pool.iloc[draw].reset_index(drop=True)
        scaler = StandardScaler().fit(np.asarray(x_pool[draw], dtype=np.float64))
        u_draw = unit_space(scaler, x_pool[draw])
        clusters = structure(u_draw, frame, subset)
        print(json.dumps({"stage": f"{name}:structure",
                          "ami": {key: value["ami"] for key, value in clusters.items()}}), flush=True)
        ptbxl = (pool["family"] == "ptbxl").to_numpy()
        within_ptbxl = ptbxl_structure(x_pool[ptbxl], pool[ptbxl])
        probe = source_probe(x_pool[draw], frame, x_heldout, heldout)
        neighbors = neighbor_checks(u_draw, frame, unit_space(scaler, x_heldout), heldout,
                                    unit_space(scaler, x_sph))
    result = {"sph": sph_result, "medoids": medoid_summary, "structure": clusters,
              "ptbxl_structure": within_ptbxl, "source_probe": probe, "neighbors": neighbors,
              "seconds": time.monotonic() - started}
    print(json.dumps({"stage": f"{name}:done", "seconds": result["seconds"]}), flush=True)
    return {"result": result, "ranked": ranked}


def medoid_table(ranked: dict[str, dict[str, np.ndarray]], pool: pd.DataFrame) -> pd.DataFrame:
    """Top-50 medoid rows per encoder and group with record IDs and families."""
    rows = []
    for name, groups in ranked.items():
        for group, values in groups.items():
            for rank, (position, similarity) in enumerate(values, start=1):
                row = pool.iloc[int(position)]
                rows.append({"encoder": name, "group": group, "rank": rank, "record_id": row["record_id"],
                             "family": row["family"], "source": row["source"], "y": int(row["y"]),
                             "combination": row["combination"], "similarity": float(similarity)})
    return pd.DataFrame(rows)


def raw_window(row: pd.Series) -> tuple[np.ndarray, str]:
    """
    Lead-major 10 s window of a pooled record from its verified raw file, and a description.

    Parameters
    ----------
    row : pd.Series
        A pooled row.

    Returns
    -------
    tuple[np.ndarray, str]
        ``(12, samples)`` window in mV and its title text.

    Raises
    ------
    ValueError
        If a PTB-XL raw file differs from the official manifest.
    """
    if row["family"] == "ptbxl":
        official = geometry024.official_hashes()
        for suffix in (".hea", ".dat"):
            file_name = f"{row['filename_hr']}{suffix}"
            if sha256_file(PTB_RAW / file_name) != official[file_name]:
                raise ValueError(f"PTB-XL raw file differs from the official manifest: {file_name}")
        signal, _ = read_signal(row["filename_hr"])
        return signal.T, f"PTB-XL {row['record_id'].split(':')[1]} ({row['device']})"
    root = RAW_ROOTS[row["source"]]
    checksums = parse_checksums((root / "SHA256SUMS.txt").read_text())
    signal, _, names = read_verified(root, row["path"], checksums)
    description = f"{row['source']} {row['record_id'].split(':')[1]}"
    return canonical_window(signal, names, int(row["window_start"])), description


def plot_reference(row: pd.Series, group: str, path: Path) -> None:
    """Plot a raw 10 s, 12-lead window on a 0.2 s by 0.5 mV grid, each lead median-centred, as 024 did."""
    window, description = raw_window(row)
    centered = window - np.median(window, axis=1, keepdims=True)
    limit = max(1.0, 1.05 * float(np.nanmax(np.abs(centered))))
    seconds = np.arange(window.shape[1]) / 500
    figure, axes = plt.subplots(12, 1, figsize=(11, 14), sharex=True, sharey=True)
    for lead, axis in enumerate(axes):
        axis.plot(seconds, centered[lead], color="#0b0b0b", linewidth=0.7)
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
    figure.suptitle(f"{group} pooled reference ECG (xECG): {description}, 500 Hz, raw, median-centred",
                    fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(path, dpi=110)
    plt.close(figure)


def plot_ami(results: dict[str, Any], path: Path) -> None:
    """Plot k-means AMI by factor and k, one panel per encoder, with the within-PTB-XL device AMI dashed."""
    series = (("family", "Source family", "#2a78d6"), ("source", "Source (6)", "#eb6834"),
              ("standard", "Standard label", "#1baf7a"), ("combination", "Superclass combination", "#eda100"))
    titles = {"xecg": "xECG", "jepa": "ECG-JEPA", "cpc": "CPC (ours)"}
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
    for axis, name in zip(axes, ENCODERS, strict=True):
        clusters = results[name]["structure"]
        for key, label, color in series:
            values = [clusters[f"kmeans_{k}"]["ami"][key] for k in KMEANS_K]
            axis.plot(range(len(KMEANS_K)), values, color=color, linewidth=2, marker="o", markersize=6,
                      label=label)
            if key in ("family", "standard"):
                axis.annotate(f"{values[-1]:.2f}", (len(KMEANS_K) - 1, values[-1]), xytext=(6, 0),
                              textcoords="offset points", va="center", fontsize=8, color="#0b0b0b")
        device = [results[name]["ptbxl_structure"][f"kmeans_{k}"]["device"] for k in KMEANS_K]
        axis.plot(range(len(KMEANS_K)), device, color="#898781", linewidth=1.5, linestyle="--",
                  label="Device, within PTB-XL")
        axis.set_xticks(range(len(KMEANS_K)), [f"k = {k}" for k in KMEANS_K])
        axis.set_title(titles[name], fontsize=10)
        axis.grid(axis="y", color="#e5e4e0", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("AMI with the k-means partition")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9)
    figure.suptitle("Experiment 024b: k-means clusters of 17,424 pooled training ECGs "
                    "(4,356 per source family)", fontsize=11)
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    figure.savefig(path, dpi=130)
    plt.close(figure)


def plot_sph(results: dict[str, Any], path: Path) -> None:
    """Plot SPH AUROC by method and training set, one panel per encoder."""
    methods = [("ptbxl", method) for method in SPH_METHODS["ptbxl"]] + \
        [("pooled", method) for method in SPH_METHODS["pooled"]]
    colors = {"ptbxl": "#2a78d6", "pooled": "#eb6834"}
    titles = {"xecg": "xECG", "jepa": "ECG-JEPA", "cpc": "CPC (ours)"}
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True, sharex=True)
    for axis, name in zip(axes, ENCODERS, strict=True):
        observed = results[name]["sph"]["metrics"]
        for position, (arm, method) in enumerate(methods):
            value = observed[f"{arm}:{method}"]["auroc"]
            label = {"ptbxl": "PTB-XL labels", "pooled": "Pooled labels"}[arm] if method == "probe" else None
            axis.scatter(value, position, s=40, color=colors[arm], zorder=3, label=label)
            axis.annotate(f"{value:.3f}", (value, position), xytext=(6, 0), textcoords="offset points",
                          va="center", fontsize=8, color="#0b0b0b")
        axis.set_title(titles[name], fontsize=10)
        axis.grid(axis="x", color="#e5e4e0", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlabel("SPH AUROC")
    axes[0].set_yticks(range(len(methods)), [f"{arm}: {method}" for arm, method in methods])
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    figure.suptitle("Experiment 024b: probe versus prototype and kNN at SPH (21,008 ECGs), standard label",
                    fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def gap_reading(gap: float) -> str:
    """024's reading of a prototype-minus-probe AUROC gap."""
    reading = "intermediate"
    if gap >= -0.02:
        reading = "usable_as_is"
    elif gap < -0.05:
        reading = "diagnosis_does_not_dominate_distances"
    return reading


def reading(results: dict[str, Any]) -> dict[str, Any]:
    """
    Apply the prespecified decision rule and describe the robustness and secondary contrasts.

    Parameters
    ----------
    results : dict[str, Any]
        Per-encoder results.

    Returns
    -------
    dict[str, Any]
        Primary contrast, side and decision; the robustness signs; and the prototype readings.
    """
    primary_partition = results[PRIMARY_ENCODER]["structure"][f"kmeans_{PRIMARY_K}"]
    primary = primary_partition["contrasts"]["family_minus_standard"]
    side = interval_side(primary)
    decision = {"above 0": "clusters_follow_hospital", "below 0": "clusters_follow_diagnosis"}.get(
        side, "not_distinguished")
    robustness = {name: {f"kmeans_{k}": {contrast_name: interval_side(values) for contrast_name, values in
                                         output["structure"][f"kmeans_{k}"]["contrasts"].items()}
                         for k in KMEANS_K} for name, output in results.items()}
    within_label = {name: {f"kmeans_{k}": all(value > output["structure"][f"kmeans_{k}"]["ami"]["standard"]
                                              for value in output["structure"][f"kmeans_{k}"]
                                              ["family_ami_within_label"].values())
                           for k in KMEANS_K} for name, output in results.items()}
    prototypes = {}
    for name, output in results.items():
        contrasts = output["sph"]["contrasts"]
        pooled_gap = contrasts["pooled:prototype_minus_pooled:probe"]["auroc"]
        change = contrasts["gap_change_pooled_minus_ptbxl"]["auroc"]
        prototypes[name] = {"pooled_gap": pooled_gap, "pooled_gap_side": interval_side(pooled_gap),
                            "pooled_gap_reading": gap_reading(pooled_gap["difference"]),
                            "ptbxl_gap": contrasts["ptbxl:prototype_minus_ptbxl:probe"]["auroc"],
                            "gap_change": change,
                            "gap_change_reading": {"above 0": "pooling_narrows",
                                                   "below 0": "pooling_widens"}.get(interval_side(change),
                                                                                    "not_distinguished")}
    return {"primary_contrast": primary, "primary_side": side, "decision": decision,
            "primary_small": abs(primary["difference"]) < SMALL_AMI, "robustness_sides": robustness,
            "family_ami_within_each_label_above_standard_ami": within_label, "prototypes": prototypes}


def main() -> None:
    """Run the frozen multi-source embedding geometry study once."""
    if (OUTPUT / "result.json").exists():
        raise FileExistsError("Experiment 024b v1 has already run")
    started = time.monotonic()
    run_identity, caches, prior022 = identity()
    groups = cohorts(ptb_table())
    details = geometry024.record_details()
    with threadpool_limits(limits=THREADS):
        reference = reproduce_024(groups["train"].join(details), groups["development"].join(details))
    print(json.dumps({"stage": "reproduce", **reference["checks"]}), flush=True)

    pool, heldout, features, _ = assemble(caches)
    sph, _, sph_hashes = sph_rows(prior022)
    sph_x = sph_features(sph, prior022)
    draw = equal_family_subsample(pool["family"].to_numpy(dtype=str), SEED)
    subset = equal_family_draw(pool.iloc[draw]["family"].to_numpy(dtype=str), HDBSCAN_PER_FAMILY, SEED)
    counts = check_counts(pool, draw, heldout, sph)
    print(json.dumps({"stage": "counts", **{key: str(value) for key, value in counts.items()}}), flush=True)
    with np.load(PRIOR022B / "predictions.npz") as saved:
        saved_sph = {name: {arm: saved[f"sph_{name}_{arm}"] for arm in ("ptbxl", "pooled")}
                     for name in ENCODERS}
        if not np.array_equal(saved["sph_ecg_ids"], sph["ecg_id"].to_numpy(dtype=str)):
            raise ValueError("022b SPH order differs")
    sph_inputs = {"y": sph["primary"].to_numpy(dtype=np.int64),
                  "patients": sph["patient_id"].to_numpy(dtype=str)}

    with ProcessPoolExecutor(max_workers=len(ENCODERS)) as executor:
        futures = {name: executor.submit(evaluate_encoder, name, features[name]["pool"],
                                         features[name]["heldout"], sph_x[name], pool, heldout, sph_inputs,
                                         draw, subset, saved_sph[name])
                   for name in ENCODERS}
        outputs = {name: future.result() for name, future in futures.items()}
    results = {name: output["result"] for name, output in outputs.items()}

    (OUTPUT / "figures").mkdir(parents=True, exist_ok=True)
    table = medoid_table({name: output["ranked"] for name, output in outputs.items()}, pool)
    table.to_csv(OUTPUT / "medoids.csv", index=False)
    figures = {"ami": "ami_by_factor_and_k.png", "sph": "sph_auroc_by_method.png"}
    plot_ami(results, OUTPUT / "figures" / figures["ami"])
    plot_sph(results, OUTPUT / "figures" / figures["sph"])
    for group in GROUPS:
        chosen = (table["encoder"] == PRIMARY_ENCODER) & (table["group"] == group) & (table["rank"] == 1)
        top = table[chosen].iloc[0]
        row = pool[pool["record_id"] == top["record_id"]].iloc[0]
        file_name = f"reference_{group.lower()}_{row['record_id'].replace(':', '_')}.png"
        plot_reference(row, group, OUTPUT / "figures" / file_name)
        figures[f"reference_{group}"] = file_name
    FIGURES.mkdir(parents=True, exist_ok=True)
    for file_name in figures.values():
        shutil.copyfile(OUTPUT / "figures" / file_name, FIGURES / file_name)

    result = {
        "status": "complete", "identity": {**run_identity, "sph_manifest": sph_hashes}, "counts": counts,
        "seed": SEED, "bootstrap_draws": BOOTSTRAP_DRAWS, "primary_encoder": PRIMARY_ENCODER,
        "primary_k": PRIMARY_K, "reference_024": reference, "encoders": results, "reading": reading(results),
        "figures": figures,
        "figure_sha256": {name: sha256_file(OUTPUT / "figures" / file) for name, file in figures.items()},
        "medoids_sha256": sha256_file(OUTPUT / "medoids.csv"),
        "challenge_test_read": False, "ptbxl_calibration_read": False, "ptbxl_test_read": False,
        "runtime_seconds": time.monotonic() - started,
    }
    write_json_atomic(OUTPUT / "result.json", result)
    print(json.dumps({"stage": "complete", "runtime_seconds": result["runtime_seconds"],
                      "decision": result["reading"]["decision"],
                      "primary": result["reading"]["primary_contrast"]}), flush=True)


if __name__ == "__main__":
    main()
