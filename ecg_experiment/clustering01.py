"""Reserved-patient-safe physical ECG morphology clustering."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.cluster import HDBSCAN, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

from ecg_experiment.eda.signals import LEADS
from ecg_experiment.intervals import patient_groups, patient_resample

SEED = 20261018
METHODS = ("kmeans", "gmm", "spectral", "hdbscan")
CATEGORICAL = ("device", "site")
NUMERIC = ("hr_bpm", "signal_std", "baseline_fraction", "beat_consistency", "constant_leads", "quality")
FEATURES = ["qrs_ms", "qt_ms", "rr_cv", "axis_sin", "axis_cos"] + [
    f"{kind}_{lead}" for kind in ("r_amp", "s_amp", "st", "t_amp") for lead in LEADS
]


def digest(value: str) -> str:
    """Return SHA-256 of a UTF-8 string."""
    return hashlib.sha256(value.encode()).hexdigest()


def train_patient(patient: int) -> bool:
    """Check the canonical PR67 training bucket for one integer patient ID."""
    return int(digest(f"ssl_synthetic_partition_v1:{SEED}:{int(patient)}"), 16) % 10000 < 8000


def select_cohort(identities: pd.DataFrame, limit: int = 2000) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Quarantine reserved patients before selecting one ECG per training patient.

    Parameters
    ----------
    identities : pd.DataFrame
        Identifier, patient, fold and filename columns only.
    limit : int
        Maximum deterministic hash-selected patient count.

    Returns
    -------
    tuple
        Selected identifiers and aggregate integrity counts.
    """
    table = identities.copy()
    table["patient_id"] = table.patient_id.astype(int)
    forbidden = set(table.loc[table.strat_fold.isin([9, 10]), "patient_id"])
    safe = table.loc[table.strat_fold.between(1, 8)]
    counts = {"fold1_8_records": len(safe), "fold1_8_patients": safe.patient_id.nunique()}
    if counts != {"fold1_8_records": 17418, "fold1_8_patients": 15023}:
        raise ValueError(f"NTT-01 identity-count mismatch: {counts}")
    allowed = safe.loc[~safe.patient_id.isin(forbidden) & safe.patient_id.map(train_patient)]
    counts.update(train_records=len(allowed), train_patients=allowed.patient_id.nunique())
    chosen = allowed.sort_values("ecg_id").drop_duplicates("patient_id").copy()
    chosen["sample_key"] = chosen.patient_id.map(lambda value: digest(f"cluster01:{SEED}:{value}"))
    chosen = chosen.sort_values("sample_key").head(limit).drop(columns="sample_key").reset_index(drop=True)
    counts["selected_patients"] = len(chosen)
    return chosen, counts


def morphology(measures: pd.DataFrame) -> pd.DataFrame:
    """Build physical morphology features without threshold-based diagnostic criteria."""
    frame = measures.copy()
    axis = np.arctan2(frame.r_amp_aVF - frame.s_amp_aVF, frame.r_amp_I - frame.s_amp_I)
    frame["axis_sin"], frame["axis_cos"] = np.sin(axis), np.cos(axis)
    frame["rr_cv"] = frame.rr_std_ms / frame.rr_mean_ms
    return frame.reindex(columns=FEATURES).replace([np.inf, -np.inf], np.nan)


def representation(
    features: pd.DataFrame, nuisance: pd.DataFrame, residual: bool
) -> tuple[np.ndarray, float]:
    """
    Fit the entire unsupervised transform on the supplied patient subset.

    Parameters
    ----------
    features, nuisance : pd.DataFrame
        Aligned morphology measurements and acquisition/quality variables.
    residual : bool
        Whether to remove ridge-predicted nuisance variation before PCA.

    Returns
    -------
    tuple
        PCA coordinates and explained variance fraction.
    """
    if features.isna().all().any():
        raise ValueError("At least one morphology feature is entirely missing")
    values = SimpleImputer(strategy="median").fit_transform(features)
    values = np.clip(values, *np.quantile(values, [0.01, 0.99], axis=0))
    values = RobustScaler().fit_transform(values)
    if residual:
        numeric = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(
            nuisance[list(NUMERIC)].replace([np.inf, -np.inf], np.nan)
        )
        numeric = RobustScaler().fit_transform(numeric)
        categories = pd.get_dummies(nuisance[list(CATEGORICAL)].fillna("missing").astype(str), dtype=float)
        design = np.column_stack([numeric, categories])
        model = Ridge(alpha=10).fit(design, values)
        values = values - model.predict(design)
        values = RobustScaler().fit_transform(values)
    pca = PCA(n_components=0.95, svd_solver="full").fit(values)
    keep = min(16, pca.n_components_)
    return pca.transform(values)[:, :keep], float(pca.explained_variance_ratio_[:keep].sum())


def partition(values: np.ndarray, method: str, parameter: int, seed: int = SEED) -> np.ndarray:
    """Fit a prespecified clustering algorithm without access to diagnostic labels."""
    if method == "kmeans":
        return KMeans(n_clusters=parameter, n_init=10, random_state=seed).fit_predict(values)
    if method == "gmm":
        model = GaussianMixture(
            n_components=parameter,
            covariance_type="diag",
            n_init=3,
            max_iter=300,
            reg_covar=1e-4,
            random_state=seed,
        )
        labels = model.fit_predict(values)
        if not model.converged_:
            raise ValueError("Gaussian mixture failed to converge")
        return labels
    if method == "spectral":
        return SpectralClustering(
            n_clusters=parameter,
            affinity="nearest_neighbors",
            n_neighbors=15,
            assign_labels="kmeans",
            n_init=10,
            random_state=seed,
            n_jobs=2,
        ).fit_predict(values)
    if method == "hdbscan":
        return HDBSCAN(min_cluster_size=parameter, min_samples=10, n_jobs=2).fit_predict(values)
    raise ValueError(f"Unknown method: {method}")


def partition_summary(values: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Describe cluster coverage and selectability using only unlabeled coordinates."""
    assigned = labels >= 0
    names, counts = np.unique(labels[assigned], return_counts=True)
    silhouette = None
    if 1 < len(names) < assigned.sum():
        silhouette = float(
            silhouette_score(
                values[assigned], labels[assigned], sample_size=min(1000, assigned.sum()), random_state=SEED
            )
        )
    eligible = (
        len(names) >= 2 and assigned.mean() >= 0.7 and counts.min(initial=len(labels)) >= 0.02 * len(labels)
    )
    return {
        "clusters": len(names),
        "assigned_fraction": float(assigned.mean()),
        "noise": int((~assigned).sum()),
        "sizes": dict(zip(map(str, names), map(int, counts), strict=True)),
        "silhouette": silhouette,
        "eligible": bool(eligible),
    }


def choose(values: np.ndarray, method: str) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    """Select a configuration by the frozen unsupervised rule, retaining every candidate."""
    candidates, partitions = [], []
    for parameter in [40, 80] if method == "hdbscan" else [2, 4, 8]:
        labels = partition(values, method, parameter)
        candidates.append({"parameter": parameter, **partition_summary(values, labels)})
        partitions.append(labels)
    scores = [
        (row["eligible"], row["silhouette"] if row["silhouette"] is not None else -2) for row in candidates
    ]
    best = max(range(len(candidates)), key=lambda index: scores[index])
    return partitions[best], candidates[best].copy(), candidates


def stability(
    features: pd.DataFrame,
    nuisance: pd.DataFrame,
    residual: bool,
    method: str,
    parameter: int,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Refit complete pipelines on five fixed patient subsamples and compare assignments."""
    rows = []
    for draw in range(5):
        chosen = np.sort(
            np.random.default_rng(SEED + draw).choice(len(features), int(0.8 * len(features)), False)
        )
        values, _ = representation(features.iloc[chosen], nuisance.iloc[chosen], residual)
        new = partition(values, method, parameter, SEED + draw)
        assigned = (labels[chosen] >= 0) & (new >= 0)
        valid = assigned.sum() >= 2 and len(np.unique(new[assigned])) >= 2
        ari = float(adjusted_rand_score(labels[chosen][assigned], new[assigned])) if valid else None
        rows.append(
            {
                "draw": draw,
                "ari_assigned": ari,
                "ari_all": float(adjusted_rand_score(labels[chosen], new)),
                "jointly_assigned_fraction": float(assigned.mean()),
            }
        )
    valid_ari = [row["ari_assigned"] for row in rows if row["ari_assigned"] is not None]
    return {
        "draws": rows,
        "valid_draws": len(valid_ari),
        "median": float(np.median(valid_ari)) if valid_ari else None,
        "minimum": float(np.min(valid_ari)) if valid_ari else None,
        "maximum": float(np.max(valid_ari)) if valid_ari else None,
    }


def quantile_factor(values: pd.Series) -> np.ndarray:
    """Return descriptive quartile codes with missing measurements as an explicit level."""
    return pd.qcut(values, 4, labels=False, duplicates="drop").fillna(-1).astype(str).to_numpy()


def associations(labels: np.ndarray, flags: pd.DataFrame, nuisance: pd.DataFrame) -> dict[str, float]:
    """Compute descriptive AMI with coexisting annotations and prespecified nuisance factors."""
    factors = {
        key: nuisance[key].fillna("missing").astype(str).to_numpy() for key in (*CATEGORICAL, "quality")
    }
    for key in ("hr_bpm", "signal_std", "baseline_fraction", "beat_consistency"):
        factors[key] = quantile_factor(nuisance[key])
    factors["superclass_combination"] = flags.astype(int).astype(str).agg("".join, axis=1).to_numpy()
    return {key: float(adjusted_mutual_info_score(values, labels)) for key, values in factors.items()}


def enrichments(labels: np.ndarray, flags: pd.DataFrame, key: str) -> list[dict[str, Any]]:
    """Describe multilabel cluster membership and one-sided Fisher enrichment tests."""
    rows = []
    for cluster in np.unique(labels[labels >= 0]):
        member = labels == cluster
        for name in flags:
            present = flags[name].to_numpy(bool)
            inside, outside = int((present & member).sum()), int((present & ~member).sum())
            _, p_value = fisher_exact(
                [[inside, int(member.sum()) - inside], [outside, int((~member).sum()) - outside]],
                alternative="greater",
            )
            prevalence = float(present[member].mean())
            rows.append(
                {
                    "key": key,
                    "cluster": int(cluster),
                    "label": name,
                    "n": int(member.sum()),
                    "positive": inside,
                    "prevalence": prevalence,
                    "lift": prevalence / present.mean() if present.any() else 0.0,
                    "p": float(p_value),
                    "multi_abnormal_fraction": float(
                        (flags.drop(columns="NORM").sum(axis=1)[member] >= 2).mean()
                    ),
                }
            )
    return rows


def correct_enrichments(rows: list[dict[str, Any]]) -> None:
    """Attach Benjamini-Hochberg q values for one family across every reported test."""
    order = np.argsort([row["p"] for row in rows])
    values = np.array([rows[index]["p"] for index in order]) * len(rows) / np.arange(1, len(rows) + 1)
    corrected = np.minimum.accumulate(values[::-1])[::-1].clip(0, 1)
    for index, value in zip(order, corrected, strict=True):
        rows[index]["q"] = float(value)


def ami_difference(
    patients: np.ndarray, combinations: np.ndarray, first: np.ndarray, second: np.ndarray
) -> dict[str, Any]:
    """Bootstrap a conditional AMI contrast using shared whole-patient resampling helpers."""
    groups, rng = patient_groups(patients), np.random.default_rng(SEED)
    values = []
    for _ in range(200):
        rows = patient_resample(groups, rng)
        values.append(
            adjusted_mutual_info_score(combinations[rows], first[rows])
            - adjusted_mutual_info_score(combinations[rows], second[rows])
        )
    return {
        "difference": float(
            adjusted_mutual_info_score(combinations, first) - adjusted_mutual_info_score(combinations, second)
        ),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "draws": 200,
        "conditional_on_fitted_clusters": True,
    }


def decision(results: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the predeclared comparison and exploratory morphology discovery gate."""
    baseline = results["residual:kmeans"]
    options = [
        key
        for key in results
        if key.startswith("residual:") and not key.endswith("kmeans") and results[key]["eligible"]
    ]
    if not options:
        return {"promising": False, "alternative": None, "reason": "No eligible residualized alternative"}
    key = max(options, key=lambda name: results[name]["silhouette"])
    alternative = results[key]
    enriched = {
        row["label"]
        for row in rows
        if row["key"] == key and row["label"] != "NORM" and row["q"] < 0.05 and row["lift"] >= 1.5
    }
    stable = alternative["stability"]
    ami = alternative["ami"]
    checks = {
        "baseline_eligible": baseline["eligible"],
        "five_valid_stability_draws": stable["valid_draws"] == 5,
        "stability_median_at_least_0.70": stable["median"] is not None and stable["median"] >= 0.7,
        "stability_minimum_at_least_0.50": stable["minimum"] is not None and stable["minimum"] >= 0.5,
        "silhouette_gain_at_least_0.02": alternative["silhouette"] - baseline["silhouette"] >= 0.02,
        "two_enriched_abnormal_classes": len(enriched) >= 2,
        "annotation_ami_exceeds_device": ami["superclass_combination"] > ami["device"],
        "annotation_ami_exceeds_rate": ami["superclass_combination"] > ami["hr_bpm"],
    }
    return {
        "promising": all(checks.values()),
        "alternative": key,
        "conditions": checks,
        "enriched_abnormal_classes": sorted(enriched),
    }
