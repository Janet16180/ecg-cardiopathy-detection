"""CLF-01: external SPH comparison of RBF SVM, MLP and logistic regression against the frozen HGB."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import ROOT
from .files import sha256_file, write_json_atomic, write_npz_atomic
from .full_development import fit_logistic, predict
from .heads01_heads import SEED, fit_hgb, paired_sensitivity
from .heads01_study import BASELINE, RAW, finish, predecessor, site_point
from .intervals import paired_auroc_difference
from .pr74_raw_study import evaluation_features, feature_matrix
from .pr74_screening_io import read_json, relative, stored
from .pr74_screening_study import assert_evaluation_independence

OUTPUT = Path("outputs/clf01_measurement_classifiers_20261028")
PROTOCOL = Path("docs/clf01-measurement-classifiers-protocol-2026-10-28.md")
REPORT = Path("reports/clf01-measurement-classifiers-2026-10-28/aggregate.json")
FROZEN = ("heads01_heads", "heads01_study", "heads01_features", "pr74_raw_study", "pr74_screening_io",
          "pr74_screening_metrics", "pr74_screening_study", "pr74_fresh_baseline", "cpu_screening041",
          "full_development", "intervals", "files", "paths")
CHALLENGERS = ("svm", "mlp", "lr")
LEVEL = 1 - 0.05 / len(CHALLENGERS)


def binding() -> dict[str, Any]:
    """Require committed CLF-01 source and protocol and capture every scientific source hash."""
    own = [ROOT / "ecg_experiment/clf01_study.py", ROOT / "scripts/experiments/run_clf01.py",
           ROOT / "tests/test_clf01.py", ROOT / PROTOCOL]
    for path in own:
        blob = subprocess.check_output(["git", "show", "HEAD:" + stored(path.relative_to(ROOT))], cwd=ROOT)
        if blob != path.read_bytes():
            raise ValueError(f"Scientific source must be committed before execution: {path.name}")
    sources = own + [ROOT / f"ecg_experiment/{name}.py" for name in FROZEN]
    return {"source_sha256": {stored(path.relative_to(ROOT)): sha256_file(path) for path in sources},
            "protocol": {"path": str(PROTOCOL), "sha256": sha256_file(ROOT / PROTOCOL)},
            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                    text=True).strip(),
            "scope": "reused_development", "threads": 2}


def challenger(name: str) -> Pipeline:
    """
    Build the fixed training-only imputer, scaler and Avance 4.8 classifier.

    Parameters
    ----------
    name : str
        ``"svm"``, ``"mlp"`` or ``"lr"``.

    Returns
    -------
    Pipeline
        Unfitted pipeline.
    """
    models = {
        "svm": SVC(kernel="rbf", C=3, gamma=0.0056, probability=True, random_state=42),
        "mlp": MLPClassifier(hidden_layer_sizes=(128,), activation="relu", alpha=3.6929e-05,
                             learning_rate_init=3.6825e-04, batch_size=64, early_stopping=True,
                             max_iter=500, random_state=42),
        "lr": LogisticRegression(C=0.1, max_iter=5000),
    }
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                         models[name])


def standalone(model: Pipeline, x: np.ndarray) -> np.ndarray:
    """Standalone score: SVM decision function as in Avance 4.8, otherwise the positive probability."""
    if isinstance(model[-1], SVC):
        return model.decision_function(x)
    return model.predict_proba(x)[:, 1]


def fit_all(reference: dict[str, Any], raw: pd.DataFrame, y: np.ndarray) -> tuple[dict[str, Any], dict]:
    """Fit the three challengers on the training cohort only, next to the verified reference heads."""
    fitted, seconds = dict(reference), {}
    for name in CHALLENGERS:
        started = time.perf_counter()
        fitted[name] = challenger(name).fit(raw.to_numpy(), y)
        seconds[name] = time.perf_counter() - started
        print(f"Fitted {name} in {seconds[name]:.1f} s", flush=True)
    return fitted, seconds


def score_all(fitted: dict[str, Any], raw: pd.DataFrame, mean: np.ndarray) -> dict[str, np.ndarray]:
    """Score every model alone and each challenger in place of HGB in the fixed 0.5/0.5 blend."""
    xecg = predict(fitted["xecg"], mean)
    hgb = fitted["hgb"].predict_proba(raw)[:, 1]
    scores = {"hgb": hgb, "xecg": xecg, "hgb_blend": 0.5 * hgb + 0.5 * xecg}
    values = raw.to_numpy()
    for name in CHALLENGERS:
        model = fitted[name]
        scores[name] = standalone(model, values)
        scores[f"{name}_blend"] = 0.5 * model.predict_proba(values)[:, 1] + 0.5 * xecg
    return scores


def integrity(site: str, scores: dict[str, np.ndarray], input_root: Path, frame: pd.DataFrame) -> dict:
    """Require the refitted references to equal the saved PR 74 scores and saved SPH points."""
    saved = np.load(input_root / RAW / f"evaluate/{site}_predictions.npz")
    if not np.array_equal(saved["ecg_id"], frame["ecg_id"].to_numpy(dtype=str)):
        raise ValueError("Saved predictions and evaluation rows are misaligned")
    result = {}
    for ours, theirs in (("hgb", "all_age_hgb"), ("xecg", "all_age_xecg"), ("hgb_blend", "all_age_blend")):
        worst = float(np.max(np.abs(scores[ours] - saved[theirs])))
        if worst > 1e-12:
            raise ValueError(f"Refitted {theirs} at {site} differs by {worst}")
        result[f"{theirs}/{site}"] = worst
    if site == "sph":
        published = read_json(input_root / RAW / "evaluate/screening_evaluation.json")
        points = {(m["id"], e["pool"]): e for m in published["models"] for e in m["evaluations"]}
        y, held = frame["target"].to_numpy(dtype=int), frame["evaluation"].to_numpy()
        for ours, theirs in (("hgb", "all_age_hgb"), ("hgb_blend", "all_age_blend")):
            if site_point(y, held, scores[ours], site) != points[(theirs, site)]:
                raise ValueError(f"Saved {theirs} SPH point does not reproduce")
        result["saved_sph_points_reproduced"] = ["all_age_hgb", "all_age_blend"]
    return result


def contrast(points: dict, scores: dict, frame: pd.DataFrame, first: str, second: str) -> dict[str, Any]:
    """Paired sensitivity (98.33% and 95%), specificity change and AUROC difference with the fixed rule."""
    held = frame["evaluation"].to_numpy()
    y = frame.loc[held, "target"].to_numpy(dtype=int)
    patients = frame.loc[held, "patient_id"].to_numpy(dtype=str)
    a, b = scores[first][held], scores[second][held]
    sensitivity = paired_sensitivity(patients, y, a > points[first]["threshold"],
                                     b > points[second]["threshold"], (0.95, LEVEL), 2000, SEED)
    low, high = sensitivity["intervals"][str(LEVEL)]
    verdict = ("challenger_better" if low > 0 else "reference_better" if high < 0
               else "no_difference_established")
    return {"first": first, "second": second, "sensitivity": sensitivity, "decision": verdict,
            "specificity_change": points[first]["metrics"]["specificity"]
            - points[second]["metrics"]["specificity"],
            "auroc": paired_auroc_difference(patients, y, a, b, 2000, SEED)}


def run(input_root: Path, output: Path) -> dict[str, Any]:
    """Integrity check, then all fits, then SPH and PTB scores, contrasts and the aggregate report."""
    receipt = binding()
    started = time.perf_counter()
    pool, arms, hashes = predecessor(input_root)
    cohort = arms["all_age"]
    raw, vectors = feature_matrix(input_root, RAW, BASELINE, pool)
    positions = pd.Index(pool["record_id"]).get_indexer(cohort["record_id"])
    y = cohort["target"].to_numpy(dtype=int)
    reference = {"hgb": fit_hgb(raw.iloc[positions], y), "xecg": fit_logistic(vectors[positions], y)}
    checks, sites = {}, {}
    for site in ("sph", "ptb"):
        frame, mean, site_raw = evaluation_features(input_root, BASELINE, site)
        assert_evaluation_independence({"all_age": cohort}, site, frame)
        xecg, hgb = predict(reference["xecg"], mean), reference["hgb"].predict_proba(site_raw)[:, 1]
        checks.update(integrity(site, {"hgb": hgb, "xecg": xecg, "hgb_blend": 0.5 * hgb + 0.5 * xecg},
                                input_root, frame))
        sites[site] = (frame, mean, site_raw)
    print("Integrity check passed", flush=True)
    fitted, seconds = fit_all(reference, raw.iloc[positions], y)
    directory = input_root / relative(output) / "run"
    directory.mkdir(parents=True, exist_ok=False)
    models, decision = {}, {}
    for site, (frame, mean, site_raw) in sites.items():
        scores = score_all(fitted, site_raw, mean)
        target, held = frame["target"].to_numpy(dtype=int), frame["evaluation"].to_numpy()
        points = {name: site_point(target, held, values, site) for name, values in scores.items()}
        for name, point in points.items():
            models.setdefault(name, []).append(point)
        if site == "sph":
            decision = {"alone": {name: contrast(points, scores, frame, name, "hgb") for name in CHALLENGERS},
                        "blend": {name: contrast(points, scores, frame, f"{name}_blend", "hgb_blend")
                                  for name in CHALLENGERS}}
        write_npz_atomic(directory / f"{site}_predictions.npz", **scores,
                         ecg_id=frame["ecg_id"].to_numpy(dtype=str), target=target, evaluation=held)
    joblib.dump(fitted, directory / "models.joblib")
    result = {"schema_version": 1, "scope": "reused_development", "interval_level": LEVEL,
              "training": {"cohort": "all_age", "records": len(cohort), "features": raw.shape[1]},
              "integrity": checks, "fit_seconds": seconds, "decision": decision,
              "models": [{"id": name, "evaluations": points} for name, points in models.items()],
              "mlp_iterations": int(fitted["mlp"][-1].n_iter_),
              "svm_support_vectors": int(fitted["svm"][-1].support_vectors_.shape[0])}
    write_json_atomic(directory / "evaluation.json", result)
    receipt = finish(input_root, output, "run", receipt, hashes,
                     {"fit_seconds": seconds, "wall_seconds": time.perf_counter() - started})
    result["receipt_sha256"] = sha256_file(directory / "completion.json")
    write_json_atomic(ROOT / REPORT, result)
    return {"report": str(REPORT), "decision": {k: {n: v["decision"] for n, v in d.items()}
                                                for k, d in decision.items()}}
