"""Export allowlisted NTT-09 patient-first aggregates without individual attribution maps."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file, write_json_atomic


def export(result_path: Path, output: Path) -> None:
    """Write aggregate scalar summaries and a figure built only from those summaries.

    Parameters
    ----------
    result_path : Path
        Source-bound completed local result.
    output : Path
        Tracked aggregate-only report directory.

    Returns
    -------
    None
    """
    result = json.loads(result_path.read_text())
    if result["status"] != "completed":
        raise ValueError("Cannot export incomplete study")
    output.mkdir(parents=True, exist_ok=True)
    aggregate = {
        key: result[key]
        for key in (
            "study",
            "status",
            "git_commit",
            "counts",
            "runtime",
            "seconds",
            "decision",
            "aggregate",
            "support_exposure",
            "maximum_contribution_sum_error",
            "full_pool_winner_runner_ties",
            "waveform_reads",
            "new_encoder_inferences",
            "new_clinical_annotations",
            "clinical_localization_evaluated",
            "closed_tests_read",
        )
    }
    aggregate["predecessor_reproduction"] = {
        key: result["predecessor_reproduction"][key]
        for key in (
            "evaluations",
            "patients",
            "maximum_array_error",
            "maximum_summary_error",
            "original_aggregate_reproduced",
        )
    }
    aggregate["result_sha256"] = sha256_file(result_path)
    write_json_atomic(output / "aggregate.json", aggregate)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for axis, metric, title in zip(
        axes,
        ("positive_quarter_jaccard", "signed_rank"),
        ("Positive-quarter beat-set agreement", "Signed beat-rank agreement"),
        strict=True,
    ):
        rows = [
            aggregate["aggregate"][family]["pair_metrics"][metric]["patient_mean"]
            for family in ("single", "ensemble")
        ]
        medians = np.array([row["median"] if row["median"] is not None else np.nan for row in rows])
        lower = np.array([row["p10"] if row["p10"] is not None else np.nan for row in rows])
        upper = np.array([row["p90"] if row["p90"] is not None else np.nan for row in rows])
        axis.errorbar(
            [0, 1],
            medians,
            yerr=np.stack([medians - lower, upper - medians]),
            fmt="o",
            capsize=6,
            markersize=7,
        )
        axis.set_xticks([0, 1], ["50 single banks", "10 five-bank ensembles"])
        axis.set_xlim(-0.4, 1.4)
        axis.set_ylim(-1 if metric == "signed_rank" else 0, 1.05)
        axis.set_title(title)
        axis.set_ylabel("Patient-mean pair agreement")
    fig.suptitle("Conditional fixed-contrast sensitivity: median and 10th–90th patient percentiles")
    fig.tight_layout()
    fig.savefig(output / "contribution_stability.png", dpi=180)
    plt.close(fig)


def main() -> None:
    """Parse the local input and aggregate output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result", type=Path, default=ROOT / "outputs/ntt09_reference_contribution_stability/result.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports/localization/ntt09-reference-contribution-stability"
    )
    args = parser.parse_args()
    export(args.result, args.output)


if __name__ == "__main__":
    main()
