"""Run the fixed training-only stability selection and covariance comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from ecg_experiment import ROOT
from ecg_experiment.clustering01 import (
    SEED,
    ami_difference,
    associations,
    correct_enrichments,
    enrichments,
    partition,
    partition_summary,
    representation,
)
from ecg_experiment.clustering02 import (
    COVARIANCES,
    VIEWS,
    anchor_patient,
    balanced,
    consensus_candidates,
    eligibility,
    fit_partition,
    fit_representation,
    permitted_metadata,
    rank_key,
    robustness_decision,
    select_candidates,
    silhouette,
    stability_summary,
    transform_representation,
)
from ecg_experiment.eda.ptbxl import diagnostic_classes, superclass_flags
from ecg_experiment.paths import to_stored

PREVIOUS = ROOT / "outputs/cluster01_physical_morphology"
EXPECTED = "74f0c82918f158a61368d308acdab5d5e98fe759852c966de76867e5c99bc8e7"
SOURCES = (
    "docs/cluster-02-stability-selection.md",
    "ecg_experiment/clustering02.py",
    "scripts/experiments/run_clustering02.py",
    "tests/test_clustering02.py",
    "ecg_experiment/clustering01.py",
    "ecg_experiment/eda/signals.py",
    "ecg_experiment/eda/ptbxl.py",
    "ecg_experiment/intervals.py",
    "ecg_experiment/paths.py",
    "pyproject.toml",
    "uv.lock",
)


def sha256(path: Path) -> str:
    """Hash exact local artifact bytes without reading any waveform input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: Any) -> None:
    """Write strict JSON preserving failures rather than replacing nonfinite numbers."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def frozen_sources() -> dict[str, str]:
    """Require committed exact source bytes before any new clustering analysis."""
    hashes = {}
    for name in SOURCES:
        committed = subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=ROOT)
        hashes[name] = sha256(ROOT / name)
        if hashlib.sha256(committed).hexdigest() != hashes[name]:
            raise ValueError(f"Source is not committed: {name}")
    return hashes


def verified_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    """Verify the exact preceding features and reservation identities without waveform reads."""
    if sha256(PREVIOUS / "result.json") != EXPECTED:
        raise ValueError("CLUSTER-01 result changed")
    old = json.loads((PREVIOUS / "result.json").read_text())
    names = (
        "morphology_features.csv",
        "analyzed_training_patients.csv",
        "physical_measurements.csv",
        "assignments.csv",
        "reservation_manifest.json",
        "reservation_policy.json",
    )
    hashes = {to_stored(PREVIOUS / "result.json"): EXPECTED}
    for name in names:
        path = PREVIOUS / name
        hashes[to_stored(path)] = sha256(path)
        if hashes[to_stored(path)] != old["outputs_sha256"][to_stored(path)]:
            raise ValueError(f"Predecessor artifact changed: {name}")
    for name, expected in old["input_sha256"].items():
        hashes[name] = sha256(ROOT / name)
        if hashes[name] != expected:
            raise ValueError(f"Predecessor metadata changed: {name}")
    identities = pd.read_csv(PREVIOUS / "analyzed_training_patients.csv")
    manifest = json.loads((PREVIOUS / "reservation_manifest.json").read_text())
    verify_identities(identities, manifest)
    features = pd.read_csv(PREVIOUS / "morphology_features.csv")
    if features.shape != (1999, 53) or not np.isfinite(features.to_numpy()).all():
        raise ValueError("Unexpected predecessor feature matrix")
    measures = pd.read_csv(PREVIOUS / "physical_measurements.csv").set_index("ecg_id")
    measures = measures.loc[identities.ecg_id].reset_index(drop=True)
    metadata = permitted_metadata(ROOT / "data/raw/ptb-xl/1.0.3/ptbxl_database.csv", identities.ecg_id)
    nuisance = pd.concat([measures, metadata[["device", "site", "quality"]]], axis=1)
    return identities, features, nuisance, old, hashes


def verify_identities(identities: pd.DataFrame, manifest: dict) -> None:
    """Check exact patient ownership, closed-fold exclusion and uniqueness before data interpretation."""
    allowed = {int(row["ecg_id"]): row for row in manifest["records"] if row["role"] == "train"}
    for row in identities.itertuples():
        if row.ecg_id not in allowed or int(allowed[row.ecg_id]["patient_id"]) != row.patient_id:
            raise ValueError("Patient is outside the verified PR67 training partition")
        if row.strat_fold not in range(1, 9):
            raise ValueError("Forbidden source fold")
    if len(identities) != 1999 or not identities.patient_id.is_unique:
        raise ValueError("Predecessor analyzed cohort differs")
    if not identities.ecg_id.is_unique:
        raise ValueError("Repeated ECG in predecessor cohort")


def reproduce(features: pd.DataFrame, nuisance: pd.DataFrame, old: dict) -> dict:
    """Reproduce preceding k-means partitions exactly on the saved training-only measurements."""
    original = pd.read_csv(PREVIOUS / "assignments.csv")
    checks = {}
    for view in ("raw", "residual"):
        key = f"{view}:kmeans"
        values, _ = representation(features, nuisance, view == "residual")
        labels = partition(values, "kmeans", old["results"][key]["parameter"])
        summary = partition_summary(values, labels)
        ari = float(adjusted_rand_score(original[key], labels))
        error = abs(summary["silhouette"] - old["results"][key]["silhouette"])
        if ari != 1 or error > 1e-10:
            raise ValueError(f"Predecessor reproduction failed: {key}, ARI={ari}, error={error}")
        checks[key] = {"assignment_ari": ari, "silhouette_absolute_error": error}
    return checks


def cached_views(features: pd.DataFrame, nuisance: pd.DataFrame, fitting: np.ndarray) -> tuple[dict, list]:
    """Precompute each fitting-only transform for the fixed reference and eight patient subsamples."""
    samples = [fitting]
    for draw in range(8):
        rng = np.random.default_rng(SEED + 1 + draw)
        samples.append(np.sort(rng.choice(fitting, int(0.8 * len(fitting)), replace=False)))
    views = {}
    for view in VIEWS:
        views[view] = []
        for rows in samples:
            state = fit_representation(features.iloc[rows], nuisance.iloc[rows], view)
            values = transform_representation(state, features, nuisance)
            views[view].append(
                {
                    "coordinates": values,
                    "dimensions": state["dimensions"],
                    "explained_variance": float(
                        state["pca"].explained_variance_ratio_[: state["dimensions"]].sum()
                    ),
                }
            )
    return views, samples


def evaluate_candidate(
    view: str,
    method: str,
    covariance: str,
    clusters: int,
    cache: list,
    samples: list,
    anchors: np.ndarray,
    deadline: float,
) -> tuple[dict, dict]:
    """Fit one reference model and eight full-pipeline refits, stopping between model fits."""
    started = time.perf_counter()
    predictions, convergence, fit_labels = [], [], None
    for index, (state, rows) in enumerate(zip(cache, samples, strict=True)):
        if time.perf_counter() >= deadline:
            raise TimeoutError("Ten-minute budget reached before admitting another model fit")
        values = state["coordinates"]
        model, converged = fit_partition(
            values[rows], method, covariance, clusters, SEED if index == 0 else SEED + index
        )
        predictions.append(model.predict(values))
        convergence.append(converged)
        if index == 0:
            fit_labels = predictions[0][rows]
    reference, refits = predictions[0][anchors], np.stack([labels[anchors] for labels in predictions[1:]])
    key = f"{view}:{method}:{covariance}:{clusters}"
    row = {
        "key": key,
        "view": view,
        "method": method,
        "covariance": covariance,
        "clusters": clusters,
        "converged": convergence,
        "reference_fit_balanced": balanced(fit_labels, clusters),
        "reference_anchor_balanced": balanced(reference, clusters),
        "refits_anchor_balanced": [balanced(labels, clusters) for labels in refits],
        "silhouette": silhouette(cache[0]["coordinates"][anchors], reference),
        "stability": stability_summary(reference, refits),
        "anchor_sizes": {
            str(k): int(v) for k, v in zip(*np.unique(reference, return_counts=True), strict=True)
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    row["conditions"] = eligibility(row)
    row["eligible"] = all(row["conditions"].values())
    return row, {"all": predictions[0], "anchor": reference, "refits": refits}


def model_matrix(
    views: dict, samples: list, anchors: np.ndarray, output: Path, deadline: float
) -> tuple[list, dict, bool]:
    """Execute all thirty-six frozen configurations without inspecting diagnostic labels."""
    candidates, assignments = [], {}
    for view in VIEWS:
        for method, covariances in (("kmeans", ("none",)), ("gmm", COVARIANCES)):
            for covariance in covariances:
                for clusters in (2, 4, 8):
                    try:
                        row, labels = evaluate_candidate(
                            view, method, covariance, clusters, views[view], samples, anchors, deadline
                        )
                    except TimeoutError:
                        dump(
                            output / "budget_stop.json",
                            {"incomplete_candidate": f"{view}:{method}:{covariance}:{clusters}"},
                        )
                        return candidates, assignments, False
                    candidates.append(row)
                    assignments[row["key"]] = labels
                    dump(output / "unsupervised_progress.json", candidates)
                    np.savez_compressed(
                        output / "assignments_local.npz",
                        **{
                            f"{key}|{name}": value
                            for key, arrays in assignments.items()
                            for name, value in arrays.items()
                        },
                    )
                    print(
                        f"{row['key']}: eligible={row['eligible']} median={row['stability']['median']:.3f} "
                        f"min={row['stability']['minimum']:.3f} seconds={row['elapsed_seconds']:.2f}",
                        flush=True,
                    )
    return candidates, assignments, True


def interpret(
    selected: dict,
    consensus: dict | None,
    assignments: dict,
    identities: pd.DataFrame,
    nuisance: pd.DataFrame,
    anchors: np.ndarray,
    output: Path,
) -> dict:
    """Interpret labels only after unsupervised selection and robustness decisions are saved."""
    metadata = permitted_metadata(ROOT / "data/raw/ptb-xl/1.0.3/ptbxl_database.csv", identities.ecg_id, True)
    classes = diagnostic_classes(pd.read_csv(ROOT / "data/raw/ptb-xl/1.0.3/scp_statements.csv", index_col=0))
    flags = superclass_flags(metadata, classes).drop(columns="abnormal")
    anchor_flags, anchor_nuisance = (
        flags.iloc[anchors].reset_index(drop=True),
        nuisance.iloc[anchors].reset_index(drop=True),
    )
    partitions = dict(selected)
    if consensus is not None and consensus["eligible"]:
        partitions["consensus"] = consensus
    rows, summaries = [], {}
    for key, row in partitions.items():
        labels = assignments[row["key"]]
        summaries[key] = {
            "eligible": row["eligible"],
            "anchor_ami": associations(labels["anchor"], anchor_flags, anchor_nuisance),
        }
        if "all" in labels:
            summaries[key]["full_cohort_ami_descriptive"] = associations(labels["all"], flags, nuisance)
        rows.extend(enrichments(labels["anchor"], anchor_flags, key))
    correct_enrichments(rows)
    pd.DataFrame(rows).to_csv(output / "anchor_enrichment.csv", index=False)
    combinations = anchor_flags.astype(int).astype(str).agg("".join, axis=1).to_numpy()
    contrast = ami_difference(
        identities.patient_id.to_numpy()[anchors],
        combinations,
        assignments[selected["residual:gmm"]["key"]]["anchor"],
        assignments[selected["residual:kmeans"]["key"]]["anchor"],
    )
    return {
        "selected_associations": summaries,
        "anchor_annotations": anchor_flags.sum().astype(int).to_dict(),
        "anchor_multi_abnormal": int((anchor_flags.drop(columns="NORM").sum(axis=1) >= 2).sum()),
        "bh_family_tests": len(rows),
        "residual_annotation_ami_contrast": contrast,
    }


def plot_candidates(candidates: list, output: Path) -> None:
    """Plot all prespecified covariance and representation choices as aggregate evidence."""
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True, constrained_layout=True)
    for axis, view in zip(axes, VIEWS, strict=True):
        rows = [row for row in candidates if row["view"] == view]
        names = [f"{row['method']} {row['covariance']} k={row['clusters']}" for row in rows]
        positions = np.arange(len(rows))
        axis.barh(positions, [row["stability"]["median"] for row in rows], color="lightsteelblue")
        axis.scatter(
            [row["stability"]["minimum"] for row in rows],
            positions,
            color="black",
            s=18,
            label="Minimum across eight refits",
        )
        axis.set_yticks(positions, names)
        axis.axvline(0.6, color="gray", linestyle="--")
        axis.axvline(0.8, color="steelblue", linestyle="--")
        axis.set(xlim=(-0.1, 1.02), title=view, xlabel="Anchor assignment ARI")
    axes[-1].legend(loc="lower right", fontsize=8)
    figure.savefig(output / "candidate_stability.png", dpi=160)
    plt.close(figure)


def candidate_table(candidates: list, output: Path) -> None:
    """Store every candidate's fixed selection quantities and failed conditions as a flat table."""
    rows = []
    for row in candidates:
        rows.append(
            {
                "key": row["key"],
                "eligible": row["eligible"],
                "silhouette": row["silhouette"],
                "minimum_anchor_ari": row["stability"]["minimum"],
                "median_anchor_ari": row["stability"]["median"],
                "maximum_anchor_ari": row["stability"]["maximum"],
                "failed_conditions": ";".join(k for k, passed in row["conditions"].items() if not passed),
                "elapsed_seconds": row["elapsed_seconds"],
            }
        )
    pd.DataFrame(rows).to_csv(output / "candidate_table.csv", index=False)


def run(output: Path) -> dict:
    """Execute one source-frozen comparison with a strict admission deadline for model fits."""
    started = time.perf_counter()
    source_hashes = frozen_sources()
    identities, features, nuisance, old, input_hashes = verified_inputs()
    reproduction = reproduce(features, nuisance, old)
    anchor_mask = identities.patient_id.map(anchor_patient).to_numpy()
    anchors, fitting = np.flatnonzero(anchor_mask), np.flatnonzero(~anchor_mask)
    identities.assign(role=np.where(anchor_mask, "anchor_selection", "fit")).to_csv(
        output / "cohort_local.csv", index=False
    )
    views, samples = cached_views(features, nuisance, fitting)
    sample_hashes = [
        hashlib.sha256("\n".join(map(str, identities.patient_id.iloc[rows])).encode()).hexdigest()
        for rows in samples
    ]
    dump(output / "subsample_patient_hashes.json", sample_hashes)
    dump(
        output / "provenance.json",
        {
            "source_sha256": source_hashes,
            "input_sha256": input_hashes,
            "predecessor_reproduction": reproduction,
            "subsample_patient_sha256": sample_hashes,
            "reference_seed": SEED,
            "refit_seeds": list(range(SEED + 1, SEED + 9)),
            "fit_patient_counts": [len(rows) for rows in samples],
            "clusters": [2, 4, 8],
            "views": list(VIEWS),
            "mixture_covariances": list(COVARIANCES),
            "gmm": {"n_init": 3, "reg_covar": 1e-3, "max_iter": 400, "tol": 1e-3},
            "kmeans": {"n_init": 10},
            "ridge_alpha": 10,
            "pca_variance": 0.95,
            "pca_component_cap": 16,
            "pca_whiten": False,
        },
    )
    candidates, assignments, complete = model_matrix(views, samples, anchors, output, started + 600)
    result = {
        "study": "CLUSTER-02",
        "status": "complete" if complete else "partial_budget_stop",
        "source_sha256": source_hashes,
        "input_sha256": input_hashes,
        "predecessor_reproduction": reproduction,
        "counts": {"all": len(identities), "fitting": len(fitting), "anchor_selection": len(anchors)},
        "subsample_patient_sha256": sample_hashes,
        "candidates": candidates,
        "pca": {
            view: [{k: v for k, v in row.items() if k != "coordinates"} for row in states]
            for view, states in views.items()
        },
        "waveforms_read": 0,
        "gpu_used": False,
        "threads": 2,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
    }
    if complete:
        selected = select_candidates(candidates)
        consensus_rows, consensus_labels = consensus_candidates(
            selected, assignments, views["raw"][0]["coordinates"][anchors]
        )
        assignments.update(consensus_labels)
        consensus = max(consensus_rows, key=rank_key) if consensus_rows else None
        decision = robustness_decision(selected, assignments, consensus)
        dump(
            output / "unsupervised_decision.json",
            {
                "selected": selected,
                "consensus": consensus,
                "consensus_candidates": consensus_rows,
                "decision": decision,
            },
        )
        result.update(
            selected=selected, consensus=consensus, consensus_candidates=consensus_rows, decision=decision
        )
        result["interpretation"] = interpret(
            selected, consensus, assignments, identities, nuisance, anchors, output
        )
        np.savez_compressed(
            output / "assignments_local.npz",
            **{
                f"{key}|{name}": value
                for key, arrays in assignments.items()
                for name, value in arrays.items()
            },
        )
    plot_candidates(candidates, output)
    candidate_table(candidates, output)
    result["elapsed_seconds"] = time.perf_counter() - started
    result["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    result["immutable_output_sha256"] = {
        to_stored(path): sha256(path)
        for path in output.iterdir()
        if path.is_file() and path.name != "status.json"
    }
    return result


def main() -> None:
    """Create an isolated output directory and bind final result bytes to completion status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/cluster02_stability_selection")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    status = args.output / "status.json"
    dump(status, {"status": "running"})
    try:
        with threadpool_limits(limits=2):
            result = run(args.output)
        dump(args.output / "result.json", result)
        dump(status, {"status": result["status"], "result_sha256": sha256(args.output / "result.json")})
    except Exception as error:
        dump(status, {"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
