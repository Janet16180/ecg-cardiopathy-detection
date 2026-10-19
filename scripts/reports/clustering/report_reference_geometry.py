"""Export aggregate-only native and No Time to Train clustering results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.paths import to_stored

FIELDS = (
    "study",
    "status",
    "source_sha256",
    "git_commit",
    "runtime",
    "counts",
    "predecessor_reproduction",
    "candidates",
    "selected",
    "decision",
    "individual_bank_checks",
    "support_label_permutations",
    "density_descriptive",
    "interpretation",
    "pca",
    "patient_sample_hashes",
    "selection_frozen_before_target_annotation_parse",
    "unsupervised_selection_sha256",
    "native_extraction_receipt_sha256",
    "embedding_space",
    "seconds",
    "waveform_reads",
    "new_encoder_inference",
    "gpu_used",
)


def main() -> None:
    """Write privacy-allowlisted receipts and standalone aggregate figures from completed outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result", type=Path, default=ROOT / "outputs/cluster03_native_reference/result.json"
    )
    parser.add_argument("--cache", type=Path, default=ROOT / "outputs/cluster03_native_cache_v2/result.json")
    parser.add_argument(
        "--profile", type=Path, default=ROOT / "outputs/cluster03_native_profile_v2/result.json"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "reports/clustering/cluster03-native-reference")
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    cache = json.loads(args.cache.read_text())
    profile = json.loads(args.profile.read_text())
    for path in (args.result, args.cache, args.profile):
        status = json.loads(path.with_name("status.json").read_text())
        if status.get("status") != "completed" or status.get("result_sha256") != sha256_file(path):
            raise ValueError("Completed result binding differs")
    args.output.mkdir(parents=True, exist_ok=True)
    aggregate = {name: result[name] for name in FIELDS}
    aggregate["local_result"] = {"path": to_stored(args.result), "sha256": sha256_file(args.result)}
    aggregate["extraction"] = {
        name: cache[name]
        for name in (
            "candidate_patients",
            "accepted_patients",
            "excluded_patients",
            "retention_fraction",
            "retention_gate_passed",
            "load_seconds",
            "extraction_forward_count",
            "profile_forwards_reused",
            "seconds",
            "peak_rss_bytes",
        )
    }
    times = np.asarray(cache["record_seconds"])
    aggregate["extraction"]["record_seconds_summary"] = {
        "mean": float(times.mean()),
        "maximum": float(times.max()),
        "median": float(np.median(times)),
    }
    aggregate["profile"] = {
        name: profile[name]
        for name in (
            "profile_gate_passed",
            "gates",
            "load_seconds",
            "synthetic_forward_count",
            "real_profile_forward_count",
            "projected_extraction_seconds",
            "seconds",
            "peak_rss_bytes",
        )
    }
    write_json_atomic(args.output / "aggregate.json", aggregate)

    import matplotlib.pyplot as plt

    views = list(result["selected"])
    selected = result["selected"]
    x = np.arange(len(views))
    figure, axis = plt.subplots(figsize=(10, 4), constrained_layout=True)
    axis.bar(x, [selected[v]["stability"]["median"] for v in views], label="Median refit ARI")
    axis.scatter(
        x, [selected[v]["stability"]["minimum"] for v in views], color="black", label="Minimum refit ARI"
    )
    axis.set(
        xticks=x,
        xticklabels=views,
        ylim=(-0.05, 1.05),
        ylabel="Validation-anchor assignment ARI",
        title="Prespecified selected clusters; ineligible fallbacks retain failed status",
    )
    axis.tick_params(axis="x", rotation=15)
    axis.legend()
    figure.savefig(args.output / "selected_stability.png", dpi=150)
    plt.close(figure)

    factors = (
        "superclass_combination",
        "device",
        "site",
        "quality",
        "hr_bpm",
        "signal_std",
        "baseline_fraction",
        "beat_consistency",
    )
    matrix = np.array(
        [[result["interpretation"]["selected"][v]["anchor_associations"][f] for f in factors] for v in views]
    )
    figure, axis = plt.subplots(figsize=(12, 4), constrained_layout=True)
    im = axis.imshow(matrix, aspect="auto", vmin=0, vmax=max(0.2, float(matrix.max())), cmap="Blues")
    axis.set(
        yticks=np.arange(len(views)),
        yticklabels=views,
        xticks=np.arange(len(factors)),
        xticklabels=factors,
        title="Descriptive associations after selection; reference views use labeled supports",
    )
    axis.tick_params(axis="x", rotation=25)
    for i in range(len(views)):
        for j in range(len(factors)):
            axis.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)
    figure.colorbar(im, ax=axis, label="Adjusted mutual information")
    figure.savefig(args.output / "annotation_and_quality_associations.png", dpi=150)
    plt.close(figure)

    rows = [r for r in result["interpretation"]["anchor_enrichments"] if r["key"] == "ntt_regions"]
    clusters = sorted({r["cluster"] for r in rows})
    labels = ("NORM", "MI", "STTC", "CD", "HYP")
    matrix = np.array(
        [
            [next(r["prevalence"] for r in rows if r["cluster"] == c and r["label"] == f) for f in labels]
            for c in clusters
        ]
    )
    figure, axis = plt.subplots(figsize=(8, max(3, len(clusters) * 0.6)), constrained_layout=True)
    im = axis.imshow(matrix, vmin=0, vmax=1, aspect="auto", cmap="Blues")
    axis.set(
        xticks=np.arange(5),
        xticklabels=labels,
        yticks=np.arange(len(clusters)),
        yticklabels=[f"Cluster {c}" for c in clusters],
        title="NTT regional-affinity groups: overlapping ECG annotations",
    )
    for i in range(len(clusters)):
        for j in range(5):
            axis.text(j, i, f"{matrix[i, j]:.0%}", ha="center", va="center")
    figure.colorbar(im, ax=axis, label="Within-cluster annotation prevalence")
    figure.savefig(args.output / "ntt_cluster_annotations.png", dpi=150)
    plt.close(figure)
    print(json.dumps({"aggregate": to_stored(args.output), "local_result_sha256": sha256_file(args.result)}))


if __name__ == "__main__":
    main()
