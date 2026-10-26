"""Run the prospectively frozen physical morphology clustering investigation."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import wfdb
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from ecg_experiment import ROOT
from ecg_experiment.clustering01 import (
    METHODS,
    ami_difference,
    associations,
    choose,
    correct_enrichments,
    decision,
    enrichments,
    morphology,
    representation,
    select_cohort,
    stability,
)
from ecg_experiment.eda.avance2_features import ecg_measurements
from ecg_experiment.eda.ptbxl import diagnostic_classes, superclass_flags
from ecg_experiment.eda.signals import canonical_order
from ecg_experiment.paths import to_stored

DATA = ROOT / "data/raw/ptb-xl/1.0.3"
QUALITY = ("baseline_drift", "static_noise", "burst_noise", "electrodes_problems", "extra_beats", "pacemaker")
SOURCES = (
    "docs/cluster-01-physical-morphology.md",
    "ecg_experiment/clustering01.py",
    "scripts/experiments/run_clustering01.py",
    "ecg_experiment/eda/avance2_features.py",
    "ecg_experiment/eda/signals.py",
    "ecg_experiment/eda/ptbxl.py",
    "ecg_experiment/intervals.py",
    "ecg_experiment/paths.py",
    "pyproject.toml",
    "uv.lock",
)


def sha256(path: Path) -> str:
    """Hash exact file bytes for the execution receipt."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_metadata(selected: pd.DataFrame, include_diagnoses: bool = False) -> pd.DataFrame:
    """Interpret diagnosis and nuisance metadata only for prospectively selected training records."""
    allowed = set(selected.ecg_id)
    rows = []
    with (DATA / "ptbxl_database.csv").open() as stream:
        for row in csv.DictReader(stream):
            ecg_id = int(row["ecg_id"])
            if ecg_id not in allowed:
                continue
            rows.append(
                {
                    "ecg_id": ecg_id,
                    "device": row["device"] or "missing",
                    "site": row["site"] or "missing",
                    "scp_codes": ast.literal_eval(row["scp_codes"]) if include_diagnoses else {},
                    "quality": int(any(row[field] not in ("", "False", "0") for field in QUALITY)),
                }
            )
    return pd.DataFrame(rows).set_index("ecg_id").loc[selected.ecg_id].reset_index()


def measure(selected: pd.DataFrame, output: Path) -> tuple[pd.DataFrame, list[dict[str, Any]], str]:
    """Extract original 500 Hz physical measurements and hash every selected input waveform."""
    rows, failures = [], []
    input_hash = hashlib.sha256()
    for position, row in enumerate(selected.itertuples()):
        path = DATA / row.filename_hr
        for extension in (".hea", ".dat"):
            target = path.with_suffix(extension)
            input_hash.update(to_stored(target).encode())
            input_hash.update(target.read_bytes())
        signal, fields = wfdb.rdsamp(str(path))
        if fields["fs"] != 500 or set(fields["units"]) != {"mV"}:
            raise ValueError("Unexpected source sampling rate or physical units")
        signal = canonical_order(signal, fields["sig_name"])
        try:
            measurements = ecg_measurements(signal, 500)
        except (ValueError, IndexError) as error:
            failures.append({"ecg_id": int(row.ecg_id), "error": str(error)})
            measurements = {}
        rows.append({"ecg_id": int(row.ecg_id), **measurements})
        if position % 200 == 0:
            print(f"Measured {position + 1}/{len(selected)} training ECGs", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "physical_measurements.csv", index=False)
    return frame, failures, input_hash.hexdigest()


def fit_all(features: pd.DataFrame, nuisance: pd.DataFrame, output: Path) -> tuple[dict[str, Any], dict]:
    """Fit and select all frozen methods without accessing any diagnosis annotations."""
    results, assignments = {}, {}
    for view in ("raw", "residual"):
        residual = view == "residual"
        values, explained = representation(features, nuisance, residual)
        for method in METHODS:
            key = f"{view}:{method}"
            started = time.perf_counter()
            labels, summary, candidates = choose(values, method)
            summary["stability"] = stability(
                features, nuisance, residual, method, summary["parameter"], labels
            )
            summary.update(
                candidates=candidates,
                pca_dimensions=values.shape[1],
                explained_variance=explained,
                elapsed_seconds=time.perf_counter() - started,
            )
            results[key], assignments[key] = summary, labels
            print(f"Finished {key}: {summary['elapsed_seconds']:.1f}s", flush=True)
            (output / "unsupervised_progress.json").write_text(json.dumps(results, indent=2))
    return results, assignments


def plots(results: dict[str, Any], rows: list[dict[str, Any]], output: Path) -> None:
    """Write aggregate stability, nuisance-association and multilabel prevalence figures."""
    names = list(results)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].barh(names, [results[key]["silhouette"] or 0 for key in names])
    axes[0].set_xlabel("Silhouette (assigned observations)")
    axes[1].barh(names, [results[key]["stability"]["median"] or 0 for key in names])
    axes[1].axvline(0.7, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Median 80% subsample ARI (jointly assigned)")
    figure.savefig(output / "cluster_stability.png", dpi=160)
    plt.close(figure)
    factors = list(results[names[0]]["ami"])
    matrix = np.array([[results[name]["ami"][factor] for factor in factors] for name in names])
    figure, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    artist = axis.imshow(matrix, aspect="auto", vmin=0, cmap="viridis")
    axis.set_xticks(range(len(factors)), factors, rotation=40, ha="right")
    axis.set_yticks(range(len(names)), names)
    figure.colorbar(artist, ax=axis, label="Adjusted mutual information")
    figure.savefig(output / "cluster_associations.png", dpi=160)
    plt.close(figure)
    selected = pd.DataFrame(rows)
    selected = selected[selected.key.eq("residual:kmeans")]
    table = selected.pivot(index="cluster", columns="label", values="prevalence")
    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    artist = axis.imshow(table, vmin=0, vmax=1, aspect="auto", cmap="Blues")
    axis.set_xticks(range(len(table.columns)), table.columns)
    axis.set_yticks(range(len(table)), [f"Cluster {value}" for value in table.index])
    figure.colorbar(artist, ax=axis, label="Annotation prevalence (labels may coexist)")
    figure.savefig(output / "cluster_multilabel_prevalence.png", dpi=160)
    plt.close(figure)


def run(output: Path) -> dict[str, Any]:
    """Execute the fixed cohort, morphology, unsupervised selection and posthoc interpretation."""
    started = time.perf_counter()
    source_hashes = {name: sha256(ROOT / name) for name in SOURCES}
    identities = pd.read_csv(
        DATA / "ptbxl_database.csv", usecols=["ecg_id", "patient_id", "strat_fold", "filename_hr"]
    )
    selected, counts = select_cohort(identities)
    manifest_path = Path("/tmp/ecg-shared-research-partition.json")
    policy_path = Path("/tmp/ecg-shared-research-policy.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest["metadata_sha256"] != sha256(DATA / "ptbxl_database.csv"):
        raise ValueError("Shared reservation manifest metadata hash mismatch")
    permitted = {int(row["ecg_id"]) for row in manifest["records"] if row["role"] == "train"}
    if not set(selected.ecg_id) <= permitted or counts["train_records"] != len(permitted):
        raise ValueError("Independent selection disagrees with shared PR67 reservation manifest")
    (output / "reservation_manifest.json").write_bytes(manifest_path.read_bytes())
    (output / "reservation_policy.json").write_bytes(policy_path.read_bytes())
    selected.to_csv(output / "selected_training_patients.csv", index=False)
    measurements, failures, waveform_hash = measure(selected, output)
    features = morphology(measurements)
    keep = features.isna().mean(axis=1).le(0.2)
    selected, measurements, features = [
        frame.loc[keep].reset_index(drop=True) for frame in (selected, measurements, features)
    ]
    selected.to_csv(output / "analyzed_training_patients.csv", index=False)
    metadata = selected_metadata(selected)
    nuisance = pd.concat(
        [measurements.drop(columns="ecg_id"), metadata[["device", "site", "quality"]]], axis=1
    )
    features.to_csv(output / "morphology_features.csv", index=False)
    results, assignments = fit_all(features, nuisance, output)
    pd.DataFrame(assignments).to_csv(output / "assignments.csv", index=False)
    metadata = selected_metadata(selected, include_diagnoses=True)
    flags = superclass_flags(
        metadata, diagnostic_classes(pd.read_csv(DATA / "scp_statements.csv", index_col=0))
    )
    flags = flags.drop(columns="abnormal")
    rows = []
    for key, labels in assignments.items():
        results[key]["ami"] = associations(labels, flags, nuisance)
        rows.extend(enrichments(labels, flags, key))
    correct_enrichments(rows)
    pd.DataFrame(rows).to_csv(output / "enrichment.csv", index=False)
    outcome = decision(results, rows)
    if outcome["alternative"] is not None:
        combinations = flags.astype(int).astype(str).agg("".join, axis=1).to_numpy()
        outcome["conditional_ami_contrast"] = ami_difference(
            selected.patient_id.to_numpy(),
            combinations,
            assignments[outcome["alternative"]],
            assignments["residual:kmeans"],
        )
    changes = {
        method: float(adjusted_rand_score(assignments[f"raw:{method}"], assignments[f"residual:{method}"]))
        for method in METHODS
    }
    plots(results, rows, output)
    outputs = {to_stored(path): sha256(path) for path in output.iterdir() if path.is_file()}
    return {
        "study": "CLUSTER-01",
        "counts": {**counts, "analyzed": len(selected), "excluded": int((~keep).sum())},
        "measurement_failures": failures,
        "source_sha256": source_hashes,
        "input_sha256": {
            to_stored(DATA / name): sha256(DATA / name)
            for name in ("ptbxl_database.csv", "scp_statements.csv")
        },
        "waveform_combined_sha256": waveform_hash,
        "outputs_sha256": outputs,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "threads": 2,
        },
        "feature_names": list(features),
        "missing_feature_fraction": float(features.isna().to_numpy().mean()),
        "annotations": {name: int(flags[name].sum()) for name in flags},
        "multi_abnormal_patients": int((flags.drop(columns="NORM").sum(axis=1) >= 2).sum()),
        "results": results,
        "raw_residual_ari": changes,
        "decision": outcome,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    """Create a fresh output directory and preserve execution status even on failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/cluster01_physical_morphology")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    status = args.output / "status.json"
    status.write_text(json.dumps({"status": "running"}))
    try:
        with threadpool_limits(limits=2):
            result = run(args.output)
        (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        status.write_text(json.dumps({"status": "completed"}))
    except Exception as error:
        status.write_text(json.dumps({"status": "failed", "type": type(error).__name__, "error": str(error)}))
        raise


if __name__ == "__main__":
    main()
