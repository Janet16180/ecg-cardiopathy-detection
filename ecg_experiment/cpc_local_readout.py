"""Frozen compact-CPC equal-width local-readout comparison."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import resource
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler


def sha256(path: Path) -> str:
    """Return the SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one auditable JSON receipt."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_csv(path: Path, key: str) -> dict[str, dict[str, str]]:
    """Read a metadata CSV and reject duplicate or blank identities."""
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    identities = [row[key] for row in rows]
    if any(not value for value in identities) or len(identities) != len(set(identities)):
        raise ValueError(f"Duplicate or blank {key} in {path}")
    return dict(zip(identities, rows, strict=True))


def _record_id(row: dict[str, str]) -> str:
    """Express an old PTB row in the clean manifest's namespace."""
    if row["source"] != "ptbxl":
        raise ValueError("Unexpected cache source")
    return f"ptbxl:{row['ecg_id']}"


def load_rows(root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:  # noqa: C901
    """Validate identities and load only allowlisted train/development rows."""
    cache = root / "outputs/experiment009_cpc_prediction_mismatch/features"
    clean = root / "outputs/data_quality/clean_rerun_preflight_v1"
    with (cache / "rows.csv").open(newline="") as handle:
        old_rows = list(csv.DictReader(handle))
    old_ids = [_record_id(row) for row in old_rows]
    if len(old_ids) != len(set(old_ids)):
        raise ValueError("Duplicate cache IDs")
    old = dict(zip(old_ids, enumerate(old_rows), strict=True))
    full = read_csv(clean / "labels_fraction1.csv", "record_id")
    limited = read_csv(clean / "labels_fraction0.1.csv", "record_id")
    held = read_csv(clean / "heldout_references.csv", "record_id")
    dev = {key: row for key, row in held.items() if row["split"] == "development"}
    if len(full) != 15359 or len(limited) != 1518 or len(dev) != 1306:
        raise ValueError("Unexpected clean cohort counts")
    if not set(limited) <= set(full) or set(full) & set(held):
        raise ValueError("Label nesting or partition separation failed")
    if len({row["patient_id"] for row in dev.values()}) != 1173:
        raise ValueError("Unexpected development patient count")
    train_patients = {row["patient_id"] for row in full.values()}
    held_patients = {row["patient_id"] for row in held.values()}
    if train_patients & held_patients:
        raise ValueError("Train/held-out patient overlap")
    for key, row in limited.items():
        if row != full[key]:
            raise ValueError("Limited/full label disagreement")
    for cohort, split in ((full, "train"), (dev, "validation")):
        for key, row in cohort.items():
            if key not in old:
                raise ValueError(f"Missing cache ID {key}")
            historical = old[key][1]
            if historical["split"] != split:
                raise ValueError(f"Partition disagreement {key}")
            if f"ptbxl:{historical['patient_id']}" != row["patient_id"]:
                raise ValueError(f"Patient disagreement {key}")
            if cohort is dev and key not in held:
                raise ValueError("Development identity missing")
            if row["target"] not in ("0", "1"):
                raise ValueError("Nonbinary target")
    for cohort in (full, limited, dev):
        if len({row["patient_id"] for row in cohort.values()}) == 0:
            raise ValueError("Empty patient set")
    ordered = {
        name: sorted(cohort) for name, cohort in (("full", full), ("limited", limited), ("development", dev))
    }
    features = np.load(cache / "features.npy", mmap_mode="r", allow_pickle=False)
    if features.shape != (19126, 3, 512) or features.dtype != np.dtype("float32"):
        raise ValueError("Unexpected feature shape/dtype")
    selected = {}
    for name, ids in ordered.items():
        indices = np.asarray([old[key][0] for key in ids], dtype=np.intp)
        selected[name] = features[indices, :2, :].copy()
        if not np.isfinite(selected[name]).all():
            raise ValueError(f"Nonfinite selected features: {name}")
    metadata = {
        "counts": {name: len(ids) for name, ids in ordered.items()},
        "development_patients": len({row["patient_id"] for row in dev.values()}),
        "train_patients": len(train_patients),
        "development_patient_ids": [dev[key]["patient_id"] for key in ordered["development"]],
        "targets": {
            name: [
                int((full if name == "full" else limited if name == "limited" else dev)[key]["target"])
                for key in ids
            ]
            for name, ids in ordered.items()
        },
        "selected_id_sha256": {
            name: hashlib.sha256("\n".join(ids).encode()).hexdigest() for name, ids in ordered.items()
        },
        "class_counts": {
            name: dict(
                Counter(
                    (full if name == "full" else limited if name == "limited" else dev)[key]["target"]
                    for key in ids
                )
            )
            for name, ids in ordered.items()
        },
    }
    return selected, metadata


def arm_matrix(features: np.ndarray, arm: str) -> np.ndarray:
    """Build the prespecified 512-coordinate arm in float64."""
    if arm == "A":
        return np.asarray(features[:, 0, :], dtype=np.float64)
    if arm == "B":
        return np.concatenate((features[:, 0, :256], features[:, 1, 256:]), axis=1).astype(np.float64)
    raise ValueError("Unknown arm")


def fit_head(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Fit one train-only scaler and fixed convex logistic head."""
    scaler = StandardScaler().fit(x)
    transformed = scaler.transform(x)
    model = LogisticRegression(
        C=0.01,
        penalty="l2",
        fit_intercept=True,
        solver="lbfgs",
        max_iter=5000,
        tol=1e-8,
        class_weight=None,
        random_state=42,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(transformed, y)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise RuntimeError("Logistic fit did not converge")
    if int(model.n_iter_[0]) >= 5000:
        raise RuntimeError("Logistic fit reached iteration limit")
    if x.shape[1] != 512 or model.coef_.size + model.intercept_.size != 513:
        raise ValueError("Unmatched classifier width")
    return {
        "mean": scaler.mean_,
        "scale": scaler.scale_,
        "coef": model.coef_[0],
        "intercept": model.intercept_[0],
        "n_iter": int(model.n_iter_[0]),
        "model": model,
        "scaler": scaler,
    }


def replay_logits(x: np.ndarray, head: dict[str, Any]) -> np.ndarray:
    """Recompute logits from saved float64 parameters."""
    return ((x - head["mean"]) / head["scale"]) @ head["coef"] + head["intercept"]


def bootstrap_draws(patient_ids: list[str], draws: int = 2000) -> list[np.ndarray]:
    """Sample development patients, retaining every ECG per sampled patient."""
    unique = sorted(set(patient_ids))
    indices = {patient: np.flatnonzero(np.asarray(patient_ids) == patient) for patient in unique}
    rng = np.random.default_rng(250925)
    return [
        np.concatenate([indices[unique[i]] for i in rng.integers(len(unique), size=len(unique))])
        for _ in range(draws)
    ]


def synthetic_bootstrap_seconds(patient_ids: list[str]) -> float:
    """Time paired draw generation and an outcome-free synthetic AUROC loop."""
    start = time.perf_counter()
    draws = bootstrap_draws(patient_ids, 2000)
    synthetic_y = np.arange(len(patient_ids)) % 2
    synthetic_a = np.linspace(0.0, 1.0, len(patient_ids))
    synthetic_b = synthetic_a[::-1]
    for indices in draws:
        if len(np.unique(synthetic_y[indices])) == 2:
            roc_auc_score(synthetic_y[indices], synthetic_a[indices])
            roc_auc_score(synthetic_y[indices], synthetic_b[indices])
    return time.perf_counter() - start


def metrics(y: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    """Evaluate fixed uncalibrated development predictions."""
    probabilities = expit(logits)
    return {
        "auroc": float(roc_auc_score(y, probabilities)),
        "ap": float(average_precision_score(y, probabilities)),
        "log_loss": float(log_loss(y, probabilities)),
    }


def run(root: Path, manifest_path: Path, expected_sha256: str) -> None:  # noqa: C901
    """Verify immutable inputs, gate cost, then score development only."""
    output = root / "outputs/cpc_local_readout_v1"
    output.mkdir(exist_ok=True)
    start = time.perf_counter()
    receipt: dict[str, Any] = {
        "status": "started",
        "command": os.sys.argv,
        "manifest_sha256": expected_sha256,
        "gpu_used": False,
    }
    try:
        if sha256(manifest_path) != expected_sha256:
            raise ValueError("Manifest SHA-256 mismatch")
        manifest = json.loads(manifest_path.read_text())
        for relative, expected in manifest["input_sha256"].items():
            if sha256(root / relative) != expected:
                raise ValueError(f"Input SHA-256 mismatch: {relative}")
        for relative, expected in manifest["source_sha256"].items():
            if sha256(root / relative) != expected:
                raise ValueError(f"Source SHA-256 mismatch: {relative}")
        receipt["hash_seconds"] = time.perf_counter() - start
        selected, metadata = load_rows(root)
        receipt["load_seconds"] = time.perf_counter() - start - receipt["hash_seconds"]
        receipt["metadata"] = metadata
        receipt["software"] = {
            "python": os.sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        }
        for package, expected in manifest["software"].items():
            if receipt["software"][package] != expected:
                raise ValueError(f"Software version mismatch: {package}")
        y_full = np.asarray(metadata["targets"]["full"], dtype=np.int8)
        heads = {}
        full_fit_start = time.perf_counter()
        for arm in ("A", "B"):
            x_full = arm_matrix(selected["full"], arm)
            heads[f"full_{arm}"] = fit_head(x_full, y_full)
            del x_full
        receipt["full_fit_seconds"] = time.perf_counter() - full_fit_start
        receipt["synthetic_bootstrap_seconds"] = synthetic_bootstrap_seconds(
            metadata["development_patient_ids"]
        )
        elapsed = time.perf_counter() - start
        remaining = 2 * receipt["full_fit_seconds"] * 1518 / 15359
        remaining += 2 * receipt["synthetic_bootstrap_seconds"] + 120 + 30
        projection = elapsed + 1.5 * remaining
        receipt["cost_gate"] = {
            "elapsed_seconds": elapsed,
            "remaining_seconds": remaining,
            "projected_total_seconds": projection,
            "ceiling_seconds": 7200,
            "passed": projection <= 7200 and elapsed <= 7200,
        }
        receipt["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write_json(output / "cost_gate.json", receipt)
        if not receipt["cost_gate"]["passed"]:
            raise RuntimeError("Cost gate failed before development outcomes")
        y_limited = np.asarray(metadata["targets"]["limited"], dtype=np.int8)
        for arm in ("A", "B"):
            heads[f"limited_{arm}"] = fit_head(arm_matrix(selected["limited"], arm), y_limited)
        predictions = {}
        for budget in ("limited", "full"):
            for arm in ("A", "B"):
                head = heads[f"{budget}_{arm}"]
                x_dev = arm_matrix(selected["development"], arm)
                direct = head["model"].decision_function(head["scaler"].transform(x_dev))
                replay = replay_logits(x_dev, head)
                error = float(np.max(np.abs(direct - replay)))
                if not np.allclose(direct, replay, rtol=1e-12, atol=1e-10):
                    raise RuntimeError(f"Saved parameter replay failed: {budget}_{arm}")
                predictions[f"{budget}_{arm}"] = replay
                head["replay_max_abs_error"] = error
        parameter_hashes = {}
        for name, head in heads.items():
            path = output / f"head_{name}.npz"
            np.savez(
                path,
                mean=head["mean"],
                scale=head["scale"],
                coef=head["coef"],
                intercept=np.asarray(head["intercept"]),
                n_iter=np.asarray(head["n_iter"]),
            )
            loaded = np.load(path, allow_pickle=False)
            replay = (
                (arm_matrix(selected["development"], name[-1]) - loaded["mean"]) / loaded["scale"]
            ) @ loaded["coef"] + loaded["intercept"]
            if not np.allclose(replay, predictions[name], rtol=1e-12, atol=1e-10):
                raise RuntimeError(f"Serialized parameter replay failed: {name}")
            parameter_hashes[path.name] = sha256(path)
        y_dev = np.asarray(metadata["targets"]["development"], dtype=np.int8)
        draws = bootstrap_draws(metadata["development_patient_ids"])
        valid = [indices for indices in draws if len(np.unique(y_dev[indices])) == 2]
        results = {}
        for budget in ("limited", "full"):
            a = predictions[f"{budget}_A"]
            b = predictions[f"{budget}_B"]
            differences = np.asarray(
                [
                    roc_auc_score(y_dev[idx], expit(b[idx])) - roc_auc_score(y_dev[idx], expit(a[idx]))
                    for idx in valid
                ]
            )
            a_metrics, b_metrics = metrics(y_dev, a), metrics(y_dev, b)
            results[budget] = {
                "A": a_metrics,
                "B": b_metrics,
                "delta_auroc_B_minus_A": b_metrics["auroc"] - a_metrics["auroc"],
                "paired_patient_interval_95": np.quantile(differences, [0.025, 0.975]).tolist(),
            }
        elapsed = time.perf_counter() - start
        if elapsed > 7200:
            raise RuntimeError("Observed cumulative time exceeded ceiling")
        report = {
            "status": "complete_development_only",
            "results": results,
            "bootstrap": {
                "draws": len(draws),
                "invalid_single_class_draws": len(draws) - len(valid),
                "seed": 250925,
            },
            "point_screen_passed": (
                results["limited"]["delta_auroc_B_minus_A"] >= 0.002
                and results["full"]["delta_auroc_B_minus_A"] >= -0.002
            ),
            "parameters_sha256": parameter_hashes,
            "replay_max_abs_errors": {key: head["replay_max_abs_error"] for key, head in heads.items()},
            "elapsed_seconds": elapsed,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "manifest_sha256": expected_sha256,
            "calibration_test_evaluated": False,
        }
        write_json(output / "report.json", report)
        receipt["status"] = "complete_development_only"
        receipt["elapsed_seconds"] = elapsed
        receipt["report_sha256"] = sha256(output / "report.json")
        write_json(output / "completion.json", receipt)
    except Exception as error:
        receipt["status"] = "failed"
        receipt["failure"] = repr(error)
        receipt["elapsed_seconds"] = time.perf_counter() - start
        write_json(output / "failure.json", receipt)
        raise
