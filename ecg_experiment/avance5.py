"""Executed development comparisons for the team's fifth academic delivery."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import ROOT
from .age_cohort_inputs import canonical_integral_id
from .files import sha256_file, write_json_atomic
from .intervals import paired_auroc_difference
from .paths import to_stored
from .pr74_screening_metrics import budget_threshold, point_metrics

OUT = ROOT / "outputs/avance5_final"
REPORT = ROOT / "reports/avance5_final"
PRIOR = ROOT / "outputs/screen01_matched_transfer_20261027/cuda_runtime_execution"
CORRECTIVE = ROOT / "outputs/pr74_corrective_evaluation_20261028"
TRAIN = ROOT / "outputs/pr75_hgb_resource_successor_20261028/qualify"
COHORT = ROOT / "outputs/cohort_age_label_priority_20261028"
SEED = 42
N_JOBS = 2
SVM_CONFIG = {"C": 2.61, "gamma": 0.0063, "cache_size": 700}
MLP_CONFIG = {
    "hidden_layer_sizes": (64,), "alpha": 0.12969816897222822,
    "batch_size": 64, "learning_rate_init": 0.0003153674921831635,
    "max_iter": 250, "early_stopping": False, "random_state": SEED,
}
NAMES = {
    "logistic": "Logística · medidas", "svm": "SVM · medidas", "mlp": "MLP · medidas",
    "rf": "Random forest", "hgb": "Histogram boosting", "voting": "Voto SVM + MLP",
    "stacking": "Stacking SVM + MLP", "xecg": "xECG + logística",
    "blend": "Mezcla boosting + xECG",
}


def json_read(path: Path) -> dict[str, Any]:
    """Read a UTF-8 aggregate or receipt without deserializing executable objects."""
    return json.loads(path.read_text(encoding="utf-8"))


def checked_array(path: Path, expected: str) -> np.ndarray:
    """Read a numeric array only after its immutable recorded digest matches."""
    if sha256_file(path) != expected:
        raise ValueError(f"Changed input: {to_stored(path)}")
    return np.load(path, allow_pickle=False, mmap_mode="r")


def training_data() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Bind the exact all-age 25k manifest and its two previously extracted feature tables."""
    job = json_read(ROOT / "outputs/pr75_hgb_resource_successor_20261028/evaluate/all_age_xecg.job.json")
    hashes = job["input_sha256"]
    names = ("training_rows.csv", "training_measurements.csv", "training_vectors.npy")
    paths = [TRAIN / name for name in names]
    for path in paths:
        if sha256_file(path) != hashes[to_stored(path)]:
            raise ValueError(f"Training binding changed: {path.name}")
    rows = pd.read_csv(paths[0], dtype=str, keep_default_na=False)
    cohort = pd.read_csv(COHORT / "all_age.csv", dtype=str, keep_default_na=False)
    if not rows.equals(cohort) or len(rows) != 25000 or not rows["split"].eq("train").all():
        raise ValueError("Expected the exact 25,000 admitted training records")
    raw = pd.read_csv(paths[1]).replace([np.inf, -np.inf], np.nan)
    vectors = np.load(paths[2], allow_pickle=False)
    if raw.shape != (25000, 130) or vectors.shape != (25000, 1024) or not np.isfinite(vectors).all():
        raise ValueError("Training feature schema changed")
    return rows, raw, vectors


def numeric_npz(path: Path) -> dict[str, np.ndarray]:
    """Read only numeric members, avoiding archived object identifiers entirely."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files if key not in {"patient_id", "ecg_id"}}


def site_inputs(site: str, columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Read existing PTB or SPH development roles without reading any protected test signal."""
    frame = pd.read_csv(PRIOR / f"qualify/{site}_rows.csv", dtype={"ecg_id": str, "patient_id": str})
    frame["target"] = frame["target"].astype(int)
    if site == "ptb":
        admitted = frame.strat_fold.ge(8).to_numpy()
        vectors = np.load(PRIOR / "qualify/ptb_xecg.npy", allow_pickle=False, mmap_mode="r")[admitted]
        frame = frame.loc[admitted].reset_index(drop=True)
        frame["evaluation"] = frame.strat_fold.eq(9)
        features = pd.read_csv(ROOT / "outputs/avance2_fusion/caracteristicas_primarias_ee39c8f6f7.csv",
                               usecols=["ecg_id", *columns], dtype={"ecg_id": str}).set_index("ecg_id")
    else:
        frame["evaluation"] = frame.role.eq("evaluation")
        vectors = np.load(PRIOR / "extract/sph_xecg.npy", allow_pickle=False, mmap_mode="r")
        path = ROOT / "outputs/meeting_20261027_recovered/experiment042_measurement_transfer_v1"
        features = pd.read_csv(path / "preparation/measurements.csv", usecols=["ecg_id", *columns],
                               dtype={"ecg_id": str}).set_index("ecg_id")
    raw = features.loc[frame.ecg_id, columns].astype(float).replace([np.inf, -np.inf], np.nan)
    raw = raw.reset_index(drop=True)
    if vectors.shape != (len(frame), 1024):
        raise ValueError("Development identities and vectors differ")
    if site == "ptb" and not set(frame.strat_fold) <= {8, 9}:
        raise ValueError("Unexpected evaluation fold")
    return frame, raw, np.asarray(vectors)


def cv_splits(rows: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Keep known patients together and use individual records where identity is unavailable."""
    known = rows.patient_id.ne("")
    groups = np.where(known, rows.source + ":patient:" + rows.patient_id, rows.record_id)
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    splits = list(splitter.split(np.zeros(len(rows)), rows.target.astype(int), groups))
    for fit, validation in splits:
        if set(groups[fit]) & set(groups[validation]):
            raise ValueError("Patient crosses an internal split")
    return splits


def measure_model() -> Any:
    """Return a training-fitted imputer/scaler and logistic model."""
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         StandardScaler(), LogisticRegression(C=0.01, max_iter=3000, tol=1e-6))


def candidates() -> dict[str, Any]:
    """Adapt the previous SVM/MLP recipes and add two homogeneous ensemble families."""
    def scaled(model: Any) -> Any:
        """Fit preprocessing exclusively inside a training fold."""
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                             StandardScaler(), model)
    return {
        "logistic": measure_model(),
        "svm": scaled(SVC(**SVM_CONFIG)),
        "mlp": scaled(MLPClassifier(**MLP_CONFIG)),
        "rf": make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                            RandomForestClassifier(n_estimators=160, min_samples_leaf=3,
                                                   n_jobs=1, random_state=SEED)),
        "hgb": HistGradientBoostingClassifier(max_iter=160, early_stopping=False, random_state=SEED),
        "xecg": make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000, tol=1e-6)),
    }


def score(model: Any, features: np.ndarray | pd.DataFrame) -> np.ndarray:
    """Use probability outputs or a fixed sigmoid of the SVM margin as an uncalibrated score."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    return expit(model.decision_function(features))


def tune_models(models: dict[str, Any], raw: pd.DataFrame, vectors: np.ndarray,
                y: np.ndarray, cv: list) -> tuple[dict[str, Any], pd.DataFrame]:
    """Optimize three relevant recipes on training groups only, retaining costs and all candidates."""
    grids = {
        "rf": {"randomforestclassifier__max_features": ["sqrt", 0.5]},
        "hgb": {"max_leaf_nodes": [7, 15, 31], "learning_rate": [0.08]},
        "xecg": {"logisticregression__C": [0.001, 0.01, 0.1]},
    }
    records = []
    for name, grid in grids.items():
        started = time.perf_counter()
        search = GridSearchCV(models[name], grid, scoring="roc_auc", cv=cv, n_jobs=N_JOBS,
                              refit=False, error_score="raise")
        search.fit(vectors if name == "xecg" else raw, y)
        elapsed = time.perf_counter() - started
        models[name] = clone(models[name]).set_params(**search.best_params_)
        for params, mean, sd in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"],
                                    search.cv_results_["std_test_score"], strict=True):
            records.append({"modelo": NAMES[name], "parametros": json.dumps(params), "AUROC_VC": mean,
                            "desv_VC": sd, "busqueda_s": elapsed})
        print(f"Ajuste {NAMES[name]}: AUROC VC {search.best_score_:.4f}, {elapsed:.1f} s", flush=True)
    return models, pd.DataFrame(records)


def fit_candidates(rows: pd.DataFrame, raw: pd.DataFrame, vectors: np.ndarray) -> dict[str, Any]:
    """Fit individual controls and truly OOF heterogeneous ensembles before any development score."""
    y = rows.target.to_numpy(dtype=int)
    cv = cv_splits(rows)
    models, search = tune_models(candidates(), raw, vectors, y, cv)
    oof, fit_seconds, oof_seconds = {}, {}, {}
    for name, model in models.items():
        features = vectors if name == "xecg" else raw
        started = time.perf_counter()
        method = "predict_proba" if hasattr(model, "predict_proba") else "decision_function"
        pred = cross_val_predict(model, features, y, cv=cv, method=method, n_jobs=N_JOBS)
        oof[name] = pred[:, 1] if method == "predict_proba" else expit(pred)
        oof_seconds[name] = time.perf_counter() - started
        started = time.perf_counter()
        model.fit(features, y)
        fit_seconds[name] = time.perf_counter() - started
        print(f"Modelo {NAMES[name]}: OOF {roc_auc_score(y, oof[name]):.4f}; "
              f"ajuste final {fit_seconds[name]:.1f} s", flush=True)
    oof["voting"] = (oof["svm"] + oof["mlp"]) / 2
    oof["blend"] = (oof["hgb"] + oof["xecg"]) / 2
    meta = np.column_stack([oof["svm"], oof["mlp"]])
    meta_model = LogisticRegression(C=0.1, max_iter=1000)
    oof["stacking"] = cross_val_predict(meta_model, meta, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    started = time.perf_counter()
    meta_model.fit(meta, y)
    meta_seconds = time.perf_counter() - started
    models["stacking"] = meta_model
    for name, parts in {"voting": ("svm", "mlp"), "stacking": ("svm", "mlp"),
                        "blend": ("hgb", "xecg")}.items():
        fit_seconds[name] = sum(fit_seconds[part] for part in parts)
        oof_seconds[name] = sum(oof_seconds[part] for part in parts)
    fit_seconds["stacking"] += oof_seconds["stacking"] + meta_seconds
    eligible = [name for name in oof if name != "stacking"]
    selected = max(eligible, key=lambda name: roc_auc_score(y, oof[name]))
    # Cross-fitting the meta head alone is insufficient for nested stacking model selection.
    # Its external development prediction remains honest; use the other OOF recipes for selection.
    return {"models": models, "oof": oof, "fit_seconds": fit_seconds, "oof_seconds": oof_seconds,
            "search": search, "selected": selected, "training_y": y}


def predict_candidates(fitted: dict[str, Any], raw: pd.DataFrame,
                       vectors: np.ndarray) -> dict[str, np.ndarray]:
    """Score every recipe with shared fitted base learners and no validation refitting."""
    models = fitted["models"]
    predictions = {name: score(model, vectors if name == "xecg" else raw)
                   for name, model in models.items() if name != "stacking"}
    predictions["voting"] = (predictions["svm"] + predictions["mlp"]) / 2
    predictions["blend"] = (predictions["hgb"] + predictions["xecg"]) / 2
    predictions["stacking"] = score(models["stacking"],
                                     np.column_stack([predictions["svm"], predictions["mlp"]]))
    return predictions


def independence(rows: pd.DataFrame, site: str, frame: pd.DataFrame) -> None:
    """Reject observed overlap in records or available PTB patient identities before scoring."""
    prefix = "ptbxl:" if site == "ptb" else "sph:"
    if set(rows.record_id) & set(prefix + frame.ecg_id):
        raise ValueError("Training/evaluation record overlap")
    if site == "ptb":
        known = rows.loc[rows.source.eq("ptbxl") & rows.patient_id.ne(""), "patient_id"]
        known_ids = set(known.str.removeprefix("ptbxl:").map(canonical_integral_id))
        evaluation_ids = set(frame.patient_id.str.removeprefix("ptbxl:").map(canonical_integral_id))
        if known_ids & evaluation_ids:
            raise ValueError("Training/evaluation patient overlap")


def run_comparison() -> dict[str, Any]:
    """Execute training, normal-only calibration and reused-hospital development evaluation."""
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    rows, raw, vectors = training_data()
    sites = {site: site_inputs(site, list(raw.columns)) for site in ("ptb", "sph")}
    for site, (frame, _, _) in sites.items():
        independence(rows, site, frame)
    fitted = fit_candidates(rows, raw, vectors)
    records, predictions = [], {}
    for site, (frame, measures, embedded) in sites.items():
        held = frame.evaluation.to_numpy()
        y = frame.target.to_numpy()
        predictions[site] = predict_candidates(fitted, measures, embedded)
        for name, scores in predictions[site].items():
            threshold = budget_threshold(y[~held], scores[~held])["threshold"]
            records.append({"id": name, "modelo": NAMES[name], "sitio": site,
                            "AUROC_VC": roc_auc_score(fitted["training_y"], fitted["oof"][name]),
                            "ajuste_final_s": fitted["fit_seconds"][name],
                            "OOF_s": fitted["oof_seconds"][name], "umbral": threshold,
                            **point_metrics(y[held], scores[held], threshold)})
        np.savez_compressed(OUT / f"{site}_predictions.npz", **predictions[site], target=y, evaluation=held)
    table = pd.DataFrame(records).sort_values(["sitio", "auroc"], ascending=[True, False])
    table.to_csv(REPORT / "comparativa.csv", index=False)
    fitted["search"].to_csv(REPORT / "hiperparametros.csv", index=False)
    for name, model in fitted["models"].items():
        joblib.dump(model, OUT / f"{name}.joblib")
    result = {"selected": fitted["selected"], "selected_label": NAMES[fitted["selected"]],
              "rows": len(rows), "features": len(raw.columns), "records": table.to_dict("records"),
              "scope": "reused_development", "closed_test_scored": False,
              "selection": "training OOF AUROC, excluding non-nested stacking diagnostic",
              "sources": rows.source.value_counts().to_dict(),
              "source_sha256": {to_stored(Path(__file__)): sha256_file(Path(__file__))},
              "input_sha256": {to_stored(TRAIN / name): sha256_file(TRAIN / name) for name in
                               ("training_rows.csv", "training_measurements.csv", "training_vectors.npy")}}
    write_json_atomic(REPORT / "summary.json", result)
    return {"summary": result, "table": table, "fitted": fitted, "sites": sites,
            "predictions": predictions, "rows": rows}


def pretrained_comparison() -> pd.DataFrame:
    """Refit three frozen released representations on identical PTB training and evaluation roles."""
    table = pd.read_csv(ROOT / "outputs/avance2_fusion/ptbxl_caracteristicas_seleccionadas.csv",
                        usecols=["ecg_id", "patient_id", "strat_fold", "target"])
    table = table.loc[table.strat_fold.le(9)].reset_index(drop=True)
    train, calibration, held = (table.strat_fold.le(7), table.strat_fold.eq(8), table.strat_fold.eq(9))
    records = []
    for name, label in (("ecg_jepa", "ECG-JEPA"), ("ecg_cpc", "ECG-CPC S4"), ("xecg", "xECG")):
        base = ROOT / "outputs/avance4_modelos/features" / name
        ids = np.load(base / "ecg_ids.npy", allow_pickle=False).astype(int)
        position = pd.Index(ids).get_indexer(table.ecg_id)
        if np.any(position < 0):
            raise ValueError("A pretrained feature is absent")
        features = np.load(base / "features.npy", allow_pickle=False, mmap_mode="r")[position]
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000, tol=1e-6))
        started = time.perf_counter()
        model.fit(features[train], table.target[train])
        elapsed = time.perf_counter() - started
        scores = score(model, features)
        threshold = budget_threshold(table.target[calibration].to_numpy(), scores[calibration])["threshold"]
        metadata = json_read(base / "metadata.json")
        records.append({"modelo": label, "entrenamiento_ECG": int(train.sum()),
                        "ajuste_cabeza_s": elapsed, "dimension": features.shape[1],
                        "extraccion_historica_s": metadata.get("seconds", metadata.get("elapsed_seconds")),
                        **point_metrics(table.target[held].to_numpy(), scores[held], threshold)})
    result = pd.DataFrame(records).sort_values("auroc", ascending=False)
    result.to_csv(REPORT / "modelos_preentrenados.csv", index=False)
    return result


def replay_existing() -> dict[str, Any]:
    """Recompute the published 25k blend's exact point metrics from numeric archived predictions."""
    aggregate_path = ROOT / "reports/pr74-corrective-evaluation-2026-10-28/aggregate.json"
    aggregate = json_read(aggregate_path)
    arrays = numeric_npz(CORRECTIVE / "evaluate/sph_predictions.npz")
    model = next(row for row in aggregate["models"] if row["id"] == "all_age_blend")
    point = next(row for row in model["evaluations"] if row["pool"] == "sph")
    held = arrays["evaluation"].astype(bool)
    observed = point_metrics(arrays["target"][held], arrays["all_age_blend"][held], point["threshold"])
    for key, value in observed.items():
        if abs(value - point["metrics"][key]) > 1e-12:
            raise ValueError(f"Historical replay failed: {key}")
    return {"verified": True, "metrics": observed, "aggregate": aggregate,
            "aggregate_sha256": sha256_file(aggregate_path)}


def age_comparison(aggregate: dict[str, Any]) -> pd.DataFrame:
    """Show saved strict and broad cohort comparisons with their original fitted-model scope."""
    records = []
    for model in aggregate["models"]:
        if not model["id"].endswith("_blend"):
            continue
        point = next(row for row in model["evaluations"] if row["pool"] == "sph")
        records.append({"cohort": model["cohort"], **aggregate["training"].get(model["cohort"], {}),
                        **point["metrics"]})
    return pd.DataFrame(records)


def plot_final(run: dict[str, Any], site: str = "sph") -> list[Path]:
    """Save four distinct diagnostics of the prospectively selected recipe."""
    selected = run["summary"]["selected"]
    frame = run["sites"][site][0]
    held = frame.evaluation.to_numpy()
    y = frame.target.to_numpy()[held]
    scores = run["predictions"][site][selected][held]
    record = run["table"].query("sitio == @site and id == @selected").iloc[0]
    paths = []
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in ("xecg", "hgb", "blend", selected):
        if name in [line.get_gid() for line in ax.lines]:
            continue
        fpr, tpr, _ = roc_curve(y, run["predictions"][site][name][held])
        line, = ax.plot(fpr, tpr, label=NAMES[name])
        line.set_gid(name)
    ax.plot([0, 1], [0, 1], ":", color="0.6")
    ax.set(xlabel="Falsos positivos / normales de referencia", ylabel="Sensibilidad", title="ROC · SPH")
    ax.legend(fontsize=8)
    paths.append(save_figure(fig, "01_roc"))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    precision, recall, _ = precision_recall_curve(y, scores)
    ax.plot(recall, precision, label=f"AP = {average_precision_score(y, scores):.3f}")
    ax.axhline(y.mean(), linestyle=":", color="0.5", label=f"Prevalencia del proxy = {y.mean():.1%}")
    ax.set(xlabel="Sensibilidad", ylabel="Precisión de las alertas", title="Precisión–recall · SPH")
    ax.legend()
    paths.append(save_figure(fig, "02_precision_recall"))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    matrix = np.array([[record.tn, record.fp], [record.fn, record.tp]], dtype=int)
    ConfusionMatrixDisplay(matrix, display_labels=["Proxy normal", "Proxy anormal"]).plot(
        ax=ax, colorbar=False, values_format="d", cmap="Blues")
    ax.set(xlabel="Predicción del modelo", ylabel="Referencia",
           title="Umbral fijado con normales de calibración")
    paths.append(save_figure(fig, "03_matriz_confusion"))
    sweep = threshold_budgets(run, site)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sweep.normal_budget * 100, sweep.sensitivity * 100, "o-", label="Sensibilidad")
    ax.plot(sweep.normal_budget * 100, (1 - sweep.specificity) * 100, "o-",
            label="Falsos positivos observados")
    ax.set(xlabel="Presupuesto de normales derivados en calibración (%)", ylabel="Porcentaje en evaluación",
           title="Más alertas permiten recuperar más positivos")
    ax.legend()
    paths.append(save_figure(fig, "04_umbrales"))
    return paths


def save_figure(figure: Any, stem: str) -> Path:
    """Save a standalone publication-friendly figure and preserve the displayed notebook figure."""
    path = REPORT / f"{stem}.png"
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    return path


def threshold_budgets(run: dict[str, Any], site: str = "sph") -> pd.DataFrame:
    """Evaluate prespecified normal-referral budgets, always choosing cutoffs on calibration only."""
    frame = run["sites"][site][0]
    y = frame.target.to_numpy()
    held = frame.evaluation.to_numpy()
    scores = run["predictions"][site][run["summary"]["selected"]]
    normals = np.sort(scores[~held & (y == 0)])
    rows = []
    for budget in (0.01, 0.025, 0.05, 0.10, 0.20):
        allowed = int(np.floor(len(normals) * budget))
        threshold = float(normals[-allowed - 1])
        rows.append({"normal_budget": budget, "threshold": threshold,
                     "calibration_normals": len(normals),
                     "flagged_calibration_normals": int((normals > threshold).sum()),
                     **point_metrics(y[held], scores[held], threshold)})
    return pd.DataFrame(rows)


def conditional_projection(sensitivity: float, specificity: float, people: int = 10000) -> pd.DataFrame:
    """Compute scenarios assuming both transportable prevalence and unchanged operating characteristics."""
    references = [("Drezner et al. (2016), NCAA", 13 / 5258),
                  ("Williams et al. (2019), secundaria", 16 / 3620)]
    rows = []
    for reference, prevalence in references:
        positives = people * prevalence
        tp = positives * sensitivity
        fp = (people - positives) * (1 - specificity)
        rows.append({"referencia": reference, "personas": people, "prevalencia_supuesta": prevalence,
                     "positivos_hipoteticos": positives, "detectados_hipoteticos": tp,
                     "omitidos_hipoteticos": positives - tp, "falsas_alertas_hipoteticas": fp,
                     "revisiones_hipoteticas": tp + fp, "VPP_hipotetico": tp / (tp + fp)})
    return pd.DataFrame(rows)


def comparison_interval(run: dict[str, Any]) -> dict[str, Any]:
    """Report paired patient uncertainty for the fixed selected-versus-xECG development contrast."""
    frame = run["sites"]["sph"][0]
    held = frame.evaluation.to_numpy()
    selected = run["summary"]["selected"]
    return paired_auroc_difference(
        frame.patient_id.to_numpy()[held], frame.target.to_numpy()[held],
        run["predictions"]["sph"][selected][held], run["predictions"]["sph"]["xecg"][held],
        draws=2000, seed=20261029)
