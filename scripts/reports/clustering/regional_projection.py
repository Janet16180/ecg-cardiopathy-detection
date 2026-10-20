"""Export aggregate-only CLUSTER-07 regional projection findings."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file, write_json_atomic


def export(output: Path, destination: Path) -> None:
    """Write safe aggregates and a standalone scientific figure.

    Parameters
    ----------
    output : pathlib.Path
        Completed local execution directory.
    destination : pathlib.Path
        Tracked aggregate-report directory.

    Returns
    -------
    None
        Aggregate JSON and PNG are written to destination.
    """
    path = output / "result.json"
    result = json.loads(path.read_text())
    status = json.loads((output / "status.json").read_text())
    digest = sha256_file(path)
    if result["status"] != "completed" or status.get("result_sha256") != digest:
        raise ValueError("Completed result binding differs")
    safe_keys = (
        "study",
        "status",
        "source_sha256",
        "git_commit",
        "runtime",
        "counts",
        "predecessor_replay",
        "candidates",
        "selected",
        "decision",
        "interpretation",
        "projection_diagnostics",
        "selection_frozen_before_target_annotation_parse",
        "unsupervised_selection_sha256",
        "seconds",
        "waveform_reads",
        "new_encoder_inference",
        "gpu_used",
    )
    aggregate = {key: result[key] for key in safe_keys}
    aggregate["local_result_sha256"] = digest
    aggregate["report_exporter_sha256"] = sha256_file(Path(__file__))
    aggregate["private_identities_coordinates_bases_and_individual_hashes_omitted"] = True
    destination.mkdir(parents=True, exist_ok=True)
    write_json_atomic(destination / "aggregate.json", aggregate)
    diagnostics = result["projection_diagnostics"][0]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    pairs = result["decision"]["all_selected_pairwise_anchor_ari"]
    values = [
        pairs["regional_covariance9_rms:ntt_region9_rms"],
        pairs["native_region9_rms:ntt_region9_rms"],
        pairs["regional_covariance9_rms:native_region9_rms"],
    ]
    axes[0].bar(
        ["Regional PCA / NTT", "Patient PCA / NTT", "Regional / patient PCA"],
        values,
        color=["#276984", "#7a8c92", "#a6b9bd"],
    )
    axes[0].axhline(0.5, color="#8e3a31", linestyle="--", label="Primary threshold")
    axes[0].set(
        ylabel="Validation-anchor adjusted Rand index",
        ylim=(-0.05, 1),
        title="Selected RMS regional partitions",
    )
    axes[0].tick_params(axis="x", labelrotation=15)
    axes[0].legend(frameon=False)
    names = ["Region variance", "Patient-vector variance"]
    regional = [
        diagnostics["regional_basis_region_variance_retained"],
        diagnostics["regional_basis_patient_variance_retained"],
    ]
    patient = [
        diagnostics["patient_basis_region_variance_retained"],
        diagnostics["patient_basis_patient_variance_retained"],
    ]
    axes[1].bar([-0.18, 0.82], regional, width=0.36, label="Regional PCA3", color="#276984")
    axes[1].bar([0.18, 1.18], patient, width=0.36, label="Patient PCA3", color="#a6b9bd")
    axes[1].set(
        xticks=[0, 1],
        xticklabels=names,
        ylabel="Fitting variance fraction retained",
        ylim=(0, 1),
        title="Reference fitting sample",
    )
    axes[1].legend(frameon=False)
    figure.savefig(destination / "regional_projection.png", dpi=180)
    plt.close(figure)


def main() -> None:
    """Export the fixed completed study into the tracked report directory.

    Returns
    -------
    None
        Fixed aggregate outputs are written locally.
    """
    export(
        ROOT / "outputs/cluster07_regional_projection",
        ROOT / "reports/clustering/cluster07-regional-projection",
    )


if __name__ == "__main__":
    main()
