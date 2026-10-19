"""Run one frozen exact linear-logit SHAP background-sensitivity baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file
from ecg_experiment.paths import to_stored
from ecg_experiment.shap01 import (
    BACKGROUNDS,
    auc_interval,
    fit_fixed_model,
    group_matrix,
    linear_shap,
    read_labels,
    reconstruction,
    sensitivity,
    training_role,
    verified_cohort,
)

matplotlib.use("Agg")

SOURCES = (
    "docs/shap-01-linear-background-sensitivity.md",
    "ecg_experiment/shap01.py",
    "scripts/reports/shap01.py",
    "tests/test_shap01.py",
    "ecg_experiment/clustering01.py",
    "ecg_experiment/eda/ptbxl.py",
    "ecg_experiment/eda/signals.py",
    "ecg_experiment/intervals.py",
    "ecg_experiment/paths.py",
    "ecg_experiment/files.py",
    "ecg_experiment/__init__.py",
    "pyproject.toml",
    "uv.lock",
)


def write_json(path: Path, value: Any) -> None:
    """Write strict JSON with visible formatting for local receipts."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_hashes() -> dict[str, str]:
    """Require every executed source to match committed bytes before fitting."""
    hashes = {}
    for name in SOURCES:
        committed = subprocess.run(["git", "show", f"HEAD:{name}"], cwd=ROOT, check=True, capture_output=True)
        digest = sha256_file(ROOT / name)
        if hashlib.sha256(committed.stdout).hexdigest() != digest:
            raise ValueError(f"Uncommitted source: {name}")
        hashes[name] = digest
    return hashes


def plots(
    groups: list[str], grouped: dict, bases: dict, example: int, logits: np.ndarray, output: Path
) -> None:
    """Write aggregate and fixed-exemplar group bars without patient IDs or waveforms."""
    import matplotlib.pyplot as plt

    positions = np.arange(len(groups))
    for name, individual in (("aggregate_groups", False), ("fixed_example", True)):
        figure, axis = plt.subplots(figsize=(11, 7), constrained_layout=True)
        for offset, background in enumerate(BACKGROUNDS):
            values = grouped[background][example] if individual else np.abs(grouped[background]).mean(axis=0)
            axis.barh(positions + (offset - 0.5) * 0.34, values, height=0.34, label=background)
        axis.set_yticks(positions, groups)
        axis.set_xlabel(
            "Signed contribution (log-odds)" if individual else "Mean absolute contribution (log-odds)"
        )
        axis.legend()
        axis.axvline(0, color="black", linewidth=0.7)
        title = "Internal evaluation: exact grouped linear-logit attributions"
        if individual:
            title = (
                f"Fixed first evaluation example: model logit {logits[example]:.3f}\n"
                f"All-training base {bases[BACKGROUNDS[0]]:.3f}; "
                f"strict-normal base {bases[BACKGROUNDS[1]]:.3f}"
            )
        axis.set_title(title)
        figure.savefig(output / f"{name}.png", dpi=160)
        figure.savefig(output / f"{name}.svg")
        plt.close(figure)


def attribution_artifacts(
    standardized: np.ndarray,
    coefficient: np.ndarray,
    intercept: float,
    backgrounds: dict,
    evaluation: np.ndarray,
    identities: pd.DataFrame,
    feature_names: list[str],
    logits: np.ndarray,
    probabilities: np.ndarray,
    output: Path,
) -> tuple[dict, dict, dict, pd.DataFrame]:
    """Write exact feature/group values and fail on violated additive identities."""
    groups, matrix = group_matrix(feature_names)
    bases, feature_values, grouped, errors = {}, {}, {}, {}
    explanation = identities.iloc[evaluation][["ecg_id", "patient_id"]].reset_index(drop=True).copy()
    explanation["logit"], explanation["probability"] = logits, probabilities
    for name, rows in backgrounds.items():
        base, phi = linear_shap(standardized[evaluation], coefficient, intercept, standardized[rows])
        bases[name], feature_values[name], grouped[name] = base, phi, phi @ matrix
        feature_total, grouped_total = phi.sum(axis=1), grouped[name].sum(axis=1)
        rebuilt, positive, other = reconstruction(base, grouped[name])
        errors[f"{name}_feature_additivity"] = float(np.max(np.abs(base + feature_total - logits)))
        errors[f"{name}_group_additivity"] = float(np.max(np.abs(base + grouped_total - logits)))
        errors[f"{name}_group_conservation"] = float(np.max(np.abs(feature_total - grouped_total)))
        errors[f"{name}_positive_reconstruction"] = float(np.max(np.abs(rebuilt - logits)))
        errors[f"{name}_probability_reconstruction"] = float(np.max(np.abs(expit(rebuilt) - probabilities)))
        feature_frame = pd.DataFrame(phi, columns=feature_names)
        group_frame = pd.DataFrame(grouped[name], columns=groups)
        for frame in (feature_frame, group_frame):
            frame.insert(0, "patient_id", explanation.patient_id)
            frame.insert(0, "ecg_id", explanation.ecg_id)
        feature_frame.to_csv(output / f"features_{name}_local.csv", index=False)
        group_frame.to_csv(output / f"groups_{name}_local.csv", index=False)
        explanation[f"{name}_base"] = base
        explanation[f"{name}_top5_positive_sum"] = positive
        explanation[f"{name}_remaining_sum"] = other
        explanation[f"{name}_reconstructed_logit"] = rebuilt
    first, second = BACKGROUNDS
    expected = coefficient * (
        standardized[backgrounds[first]].mean(axis=0) - standardized[backgrounds[second]].mean(axis=0)
    )
    errors["patient_independent_feature_background_shift"] = float(
        np.max(np.abs(feature_values[second] - feature_values[first] - expected))
    )
    errors["patient_independent_group_background_shift"] = float(
        np.max(np.abs(grouped[second] - grouped[first] - expected @ matrix))
    )
    errors["base_cancels_total_background_shift"] = float(abs(bases[second] - bases[first] + expected.sum()))
    if any(error > 1e-10 for error in errors.values()):
        raise ValueError("Exact linear explanation identity failed")
    changes, patient_changes = sensitivity(grouped[first], grouped[second], groups)
    explanation = pd.concat([explanation, patient_changes], axis=1)
    explanation.to_csv(output / "explanations_local.csv", index=False)
    group_summary = {
        name: {
            "base_logit": bases[name],
            "mean_absolute": dict(zip(groups, np.abs(values).mean(axis=0).tolist(), strict=True)),
            "rank_order": [groups[i] for i in np.argsort(-np.abs(values).mean(axis=0), kind="stable")],
        }
        for name, values in grouped.items()
    }
    example = int(np.argmin(explanation.patient_id.to_numpy()))
    plots(groups, grouped, bases, example, logits, output)
    return changes, {"errors": errors, "all_at_most_1e_minus10": True}, group_summary, explanation


def run(output: Path) -> dict[str, Any]:
    """Fit one model, compare two train-only backgrounds and bind all fixed artifacts."""
    started = time.perf_counter()
    sources = source_hashes()
    predecessor = ROOT / "outputs/cluster01_physical_morphology"
    metadata = ROOT / "data/raw/ptb-xl/1.0.3/ptbxl_database.csv"
    features, identities, permitted, integrity = verified_cohort(predecessor, metadata)
    target, strict = read_labels(identities, permitted, metadata)
    scaler, model, train, evaluation, backgrounds = fit_fixed_model(
        features.to_numpy(),
        target,
        identities.patient_id.to_numpy(),
        strict,
    )
    mean_error = float(np.max(np.abs(scaler.mean_ - features.to_numpy()[train].mean(axis=0))))
    if int(scaler.n_samples_seen_) != len(train) or mean_error > 1e-10:
        raise ValueError("Scaler statistics are not exclusively internal-training statistics")
    if set(identities.iloc[train].patient_id) & set(identities.iloc[evaluation].patient_id):
        raise ValueError("Internal training/evaluation patient overlap")
    if any(not set(rows) <= set(train) for rows in backgrounds.values()):
        raise ValueError("Evaluation row entered an explanation background")
    standardized = scaler.transform(features.to_numpy())
    probabilities = model.predict_proba(standardized[evaluation])[:, 1]
    logits = model.decision_function(standardized[evaluation])
    changes, numerical, groups, explanations = attribution_artifacts(
        standardized,
        model.coef_[0],
        float(model.intercept_[0]),
        backgrounds,
        evaluation,
        identities,
        list(features),
        logits,
        probabilities,
        output,
    )
    numerical["scaler_training_mean_max_error"] = mean_error
    numerical["scaler_training_sample_count_exact"] = True
    numerical["backgrounds_and_patients_train_only"] = True
    if not np.array_equal(probabilities, model.predict_proba(standardized[evaluation])[:, 1]):
        raise ValueError("Explanation calculation changed model predictions")
    identities = identities.copy()
    identities["target"], identities["strict_project_label"] = target, strict
    identities["internal_train_role"] = identities.patient_id.map(training_role)
    identities["used_for_model_training"] = identities.index.isin(train)
    identities["used_for_evaluation"] = identities.index.isin(evaluation)
    identities["used_for_normal_background"] = identities.index.isin(backgrounds[BACKGROUNDS[1]])
    identities.to_csv(output / "cohort_local.csv", index=False)
    model_state = {
        "feature_order": list(features),
        "coefficient": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "standard_scaler_mean": scaler.mean_.tolist(),
        "standard_scaler_scale": scaler.scale_.tolist(),
        "standard_scaler_variance": scaler.var_.tolist(),
        "scaler_training_rows": int(scaler.n_samples_seen_),
        "model_iterations": model.n_iter_.tolist(),
        "model_parameters": model.get_params(),
        "penalty_semantics": "L2: l1_ratio=0, C=1, lbfgs",
        "background_mean_standardized": {
            name: standardized[rows].mean(axis=0).tolist() for name, rows in backgrounds.items()
        },
    }
    write_json(output / "model_local.json", model_state)
    group_names, membership = group_matrix(list(features))
    first, second = BACKGROUNDS
    shift = model.coef_[0] * (
        standardized[backgrounds[first]].mean(axis=0) - standardized[backgrounds[second]].mean(axis=0)
    )
    evaluated = identities.iloc[evaluation]
    context = auc_interval(evaluated.patient_id.to_numpy(), target[evaluation].astype(int), probabilities)
    artifact_hashes = {
        to_stored(path): sha256_file(path)
        for path in output.iterdir()
        if path.is_file() and path.name != "status.json"
    }
    elapsed = time.perf_counter() - started
    if elapsed > 300:
        raise TimeoutError("Five-minute SHAP-01 CPU budget exceeded")
    return {
        "study": "SHAP-01",
        "status": "complete_linear_background_sensitivity",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_sha256": sources,
        "predecessor_integrity": integrity,
        "artifact_sha256": artifact_hashes,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "counts": {
            "cohort": len(features),
            "defined_target": int(np.isfinite(target).sum()),
            "undefined_target": int((~np.isfinite(target)).sum()),
            "internal_training": len(train),
            "internal_evaluation": len(evaluation),
            "training_positive": int(target[train].sum()),
            "evaluation_positive": int(target[evaluation].sum()),
            "backgrounds": {name: len(rows) for name, rows in backgrounds.items()},
        },
        "background_sensitivity": changes,
        "numerical_checks": numerical,
        "group_summary": groups,
        "constant_group_background_shift": dict(zip(group_names, (shift @ membership).tolist(), strict=True)),
        "secondary_evaluation_auroc": context,
        "predictions_identical_across_backgrounds": True,
        "evaluation_in_backgrounds": False,
        "waveforms_read": False,
        "gpu_used": False,
        "new_independent_clinical_test": False,
        "model_fits": 1,
        "fixed_example": {
            "patient_id": int(explanations.patient_id.min()),
            "selection": "lowest internal evaluation patient ID",
        },
        "versions": {
            name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib")
        },
        "seconds": elapsed,
    }


def main() -> None:
    """Run into a fresh directory, preserving failure and avoiding mutable-status hashing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/shap01_linear_background")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    status = {
        "status": "running",
        "command": ["python", "-m", "scripts.reports.shap01", "--output", to_stored(args.output)],
        "thread_environment": {
            key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }
    write_json(args.output / "status.json", status)
    try:
        with threadpool_limits(limits=2):
            result = run(args.output)
        result.update(command=status["command"], thread_environment=status["thread_environment"])
        write_json(args.output / "result.json", result)
    except Exception as error:
        status.update(status="failed", error_type=type(error).__name__)
        write_json(args.output / "status.json", status)
        raise
    status.update(status="complete", result_sha256=sha256_file(args.output / "result.json"))
    write_json(args.output / "status.json", status)
    print(
        json.dumps(
            {
                "counts": result["counts"],
                "sensitivity": result["background_sensitivity"],
                "auroc": result["secondary_evaluation_auroc"],
                "seconds": result["seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
