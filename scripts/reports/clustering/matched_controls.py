"""Export only aggregate-safe CLUSTER-05 results and a standalone concordance figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import adjusted_rand_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.paths import from_stored


def posthoc_scale_diagnostic(result: dict, result_path: Path) -> dict:
    """Compare declared GMM configurations after scoring using existing saved assignments."""
    rows = []
    transforms = json.loads(result_path.with_name("transforms_local.json").read_text())
    configs = (("ntt_region9", "tied", 2), ("ntt_region9", "diag", 2), ("ntt_record3", "full", 3))
    with np.load(result_path.with_name("assignments_local.npz"), allow_pickle=False) as saved:
        for base, covariance, clusters in configs:
            raw = f"{base}_raw:gmm:{covariance}:{clusters}"
            scaled = f"{base}_rms:gmm:{covariance}:{clusters}"
            rows.append(
                {
                    "base": base,
                    "covariance": covariance,
                    "clusters": clusters,
                    "raw_average_fitting_coordinate_variance": transforms[base + "_raw"][0]["fitting_rms"]
                    ** 2,
                    "raw_reg_covar_to_average_variance": 1e-3
                    / transforms[base + "_raw"][0]["fitting_rms"] ** 2,
                    "rms_reg_covar_to_average_variance": 1e-3,
                    "anchor_ari": float(
                        adjusted_rand_score(saved[raw + "|anchor"], saved[scaled + "|anchor"])
                    ),
                    "all_ari": float(adjusted_rand_score(saved[raw + "|all"], saved[scaled + "|all"])),
                    "refit_anchor_ari": [
                        float(adjusted_rand_score(a, b))
                        for a, b in zip(saved[raw + "|refits"], saved[scaled + "|refits"], strict=True)
                    ],
                    "raw_eligible": next(
                        row["eligible"] for row in result["candidates"] if row["key"] == raw
                    ),
                    "rms_eligible": next(
                        row["eligible"] for row in result["candidates"] if row["key"] == scaled
                    ),
                }
            )
    return {
        "study": "CLUSTER-05",
        "posthoc_descriptive_saved_assignment_comparison": True,
        "new_fits": 0,
        "result_sha256": sha256_file(result_path),
        "comparisons": rows,
    }


def export(result_path: Path, destination: Path) -> dict:
    """Export aggregate metrics without identities, sample indices, coordinates or inputs."""
    result = json.loads(result_path.read_text())
    if result["status"] != "completed":
        raise ValueError("A completed study is required")
    status = json.loads(result_path.with_name("status.json").read_text())
    if status != {"status": "completed", "result_sha256": sha256_file(result_path)}:
        raise ValueError("Completion binding differs")
    for name, digest in result["artifact_sha256"].items():
        if sha256_file(from_stored(name)) != digest:
            raise ValueError(f"Bound local artifact changed: {name}")
    selection = result_path.with_name("unsupervised_selection.json")
    if sha256_file(selection) != result["unsupervised_selection_sha256"]:
        raise ValueError("Frozen selection changed")
    transforms = json.loads(result_path.with_name("transforms_local.json").read_text())
    metrics = (
        "dimensions",
        "fitting_rms",
        "divisor",
        "patient_pca_explained_variance",
        "equal_patient_fitting_region_captured_variance",
    )
    aggregate = {
        "study": "CLUSTER-05",
        "result_sha256": sha256_file(result_path),
        "report_exporter_sha256": sha256_file(Path(__file__)),
        "execution_commit": result["git_commit"],
        "source_sha256": result["source_sha256"],
        **{
            key: result[key]
            for key in (
                "counts",
                "predecessor_replay",
                "candidates",
                "selected",
                "decision",
                "kmeans_scalar_invariance",
                "individual_bank_checks",
                "individual_bank_pairwise_ari",
                "support_label_controls",
                "interpretation",
                "seconds",
                "waveform_reads",
                "new_encoder_inference",
                "gpu_used",
                "selection_frozen_before_target_annotation_parse",
                "unsupervised_selection_sha256",
            )
        },
        "preprocessing_summary": {
            view: [{key: state[key] for key in metrics if key in state} for state in states]
            for view, states in transforms.items()
        },
    }
    destination.mkdir(parents=True, exist_ok=True)
    write_json_atomic(destination / "aggregate.json", aggregate)
    write_json_atomic(
        destination / "posthoc_scale_diagnostic.json", posthoc_scale_diagnostic(result, result_path)
    )
    pairs = result["decision"]["all_selected_pairwise_anchor_ari"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    names = ("Regional9", "Patient3", "Patient9")
    for offset, scale, color in ((-0.18, "raw", "#69798b"), (0.18, "rms", "#216e9e")):
        keys = (
            f"native_region9_{scale}:ntt_region9_{scale}",
            f"native_patient3_{scale}:ntt_record3_{scale}",
            f"ntt_region9_{scale}:native_patient9_{scale}",
        )
        axes[0].bar(np.arange(3) + offset, [pairs[key] for key in keys], width=0.34, label=scale, color=color)
    axes[0].axhline(0.5, color="#ad5a30", linestyle="--", label="Primary ARI threshold")
    axes[0].set(
        xticks=np.arange(3),
        xticklabels=names,
        ylabel="Selected anchor ARI",
        ylim=(-0.05, 1.05),
        title="Matched native versus NTT geometry",
    )
    axes[0].legend(fontsize=8)
    controls = result["support_label_controls"]
    axes[1].scatter(
        np.arange(1, 21), [row["anchor_ari_to_true_support_selected"] for row in controls], color="#216e9e"
    )
    axes[1].set(
        xlabel="Fixed support-label shuffle",
        ylabel="Anchor ARI to true-support partition",
        ylim=(-0.05, 1.05),
        title="Conditional support-grouping sensitivity",
    )
    fig.savefig(destination / "matched_concordance.png", dpi=180)
    plt.close(fig)
    return aggregate


def main() -> None:
    """Write small aggregate artifacts from the fixed local completed receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result", type=Path, default=ROOT / "outputs/cluster05_matched_controls/result.json"
    )
    parser.add_argument(
        "--destination", type=Path, default=ROOT / "reports/clustering/cluster05-matched-controls"
    )
    args = parser.parse_args()
    export(args.result, args.destination)


if __name__ == "__main__":
    main()
