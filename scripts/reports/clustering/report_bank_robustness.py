"""Export explicitly allowlisted CLUSTER-04 aggregates and standalone figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file, write_json_atomic


def aggregate(result: dict, digest: str) -> dict:
    """Retain study-level results without identities, signal hashes or per-patient artifacts."""
    return {
        "study": result["study"],
        "status": result["status"],
        "result_sha256": digest,
        "source_commit": result["git_commit"],
        "counts": result["counts"],
        "predecessor_reproduction": result["predecessor_reproduction"],
        "summaries": result["summaries"],
        "decision": result["decision"],
        "classification": result["classification"],
        "annotation_ami": result["annotation_ami"],
        "seconds": result["seconds"],
        "geometric_decision_frozen_before_annotation_parse": result[
            "geometric_decision_frozen_before_annotation_parse"
        ],
        "qualifications": [
            "Conditional finite development support pool; not an independent clinical test.",
            "Pairwise bank comparisons are correlated, without independent-sample confidence intervals.",
            "The secondary five-bank ensembles cannot rescue the primary or CLUSTER-03 decision.",
            "Physical classifier bank uncertainty was not resampled.",
            "Cluster persistence does not establish disease categories, localization or etiology.",
        ],
    }


def figure_distributions(result: dict, arrays: dict, output: Path) -> None:
    """Plot bank variation without exposing patient identities or individual ECGs."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for prefix, label, color in (
        ("regional", "Single bank, regional", "#1565a7"),
        ("record", "Single bank, record", "#a05a15"),
        ("ensemble", "Five-bank mean, regional", "#21784c"),
    ):
        values = np.sort(arrays[prefix + "_pairwise_ari"])
        axes[0].plot(values, np.arange(1, len(values) + 1) / len(values), label=label, color=color)
        values = arrays[prefix + "_reference_agreement"]
        thresholds = np.linspace(0, 1, 501)
        persistence = np.mean(values[:, None] >= thresholds[None, :], axis=0)
        axes[1].plot(thresholds, persistence, label=label, color=color)
    axes[0].set(
        xlabel="Pairwise anchor ARI",
        ylabel="Empirical cumulative fraction",
        title="Conditional bank concordance",
        xlim=(0, 1),
        ylim=(0, 1.01),
    )
    axes[1].set(
        xlabel="Required fraction of banks matching original assignment",
        ylabel="Fraction of 411 validation anchors",
        title="Fitting-aligned reference persistence",
        xlim=(0, 1),
        ylim=(0, 1.01),
    )
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].axvline(0.8, color="0.5", linestyle=":", linewidth=1)
    fig.suptitle("CLUSTER-04: reference-bank sensitivity on reused development patients")
    fig.savefig(output / "bank_stability.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    values = np.array(result["classification"]["bank_accuracy_values"])
    ax.hist(values, bins=np.arange(0.825, 1.026, 1 / 60), color="#1565a7", edgecolor="white")
    ax.axvline(
        result["classification"]["physical_original_three_bank_mean_balanced_accuracy"],
        color="#a05a15",
        linestyle="--",
        label="Original physical three-bank average",
    )
    ax.set(
        xlabel="Balanced accuracy on the same 60 explored targets",
        ylabel="Number of sampled banks",
        title="Secondary classification sensitivity; no new clinical test",
        xlim=(0.82, 1.01),
    )
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(output / "classification_sensitivity.png", dpi=180)
    plt.close(fig)


def main() -> None:
    """Verify local receipt bindings before rendering safe aggregate artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/cluster04_bank_robustness"))
    parser.add_argument("--output", type=Path, default=Path("reports/clustering/cluster04-bank-robustness"))
    args = parser.parse_args()
    source = args.input if args.input.is_absolute() else ROOT / args.input
    output = args.output if args.output.is_absolute() else ROOT / args.output
    path = source / "result.json"
    digest = sha256_file(path)
    status = json.loads((source / "status.json").read_text())
    if status != {"status": "completed", "result_sha256": digest}:
        raise ValueError("Study receipt lacks completion binding")
    result = json.loads(path.read_text())
    for name, expected in result["artifact_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError("Study artifact changed")
    with np.load(source / "assignments_local.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output / "aggregate.json", aggregate(result, digest))
    figure_distributions(result, arrays, output)


if __name__ == "__main__":
    main()
