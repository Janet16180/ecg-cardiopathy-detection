"""Export explicit aggregate-only matched-union contribution controls."""

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
    """Write aggregate fields and a figure without identities, weights or individual maps.

    Parameters
    ----------
    result_path : Path
        Completed source-bound local result.
    output : Path
        Aggregate report directory.

    Returns
    -------
    None
    """
    result = json.loads(result_path.read_text())
    if result["status"] != "completed":
        raise ValueError("Cannot export an incomplete study")
    output.mkdir(parents=True, exist_ok=True)
    aggregate = {
        key: result[key]
        for key in (
            "study",
            "status",
            "git_commit",
            "runtime",
            "seconds",
            "counts",
            "decision",
            "aggregate",
            "prototype_diagnostics",
            "source_exposure",
            "unique_support_people_matched",
            "equal_computation_budget",
            "maximum_contribution_sum_error",
            "maximum_analytic_A_contribution_error",
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
            "patients",
            "decompositions",
            "bank_pairs",
            "pair_metrics",
            "maximum_array_error",
            "maximum_complete_summary_error",
            "all_null_masks_exact",
            "complete_primary_reproduced",
        )
    }
    aggregate["result_sha256"] = sha256_file(result_path)
    write_json_atomic(output / "aggregate.json", aggregate)
    within = [
        result["aggregate"]["methods"][method]["pair_metrics"]["positive_quarter_jaccard"]["patient_mean"]
        for method in ("A", "B", "C")
    ]
    between = [
        result["aggregate"]["matching_block_comparisons"][key]["positive_quarter_jaccard"]["patient_mean"]
        for key in ("A:B", "A:C", "B:C")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, rows, labels, title in zip(
        axes,
        (within, between),
        (("A: bank normalized", "B: uniform union", "C: occurrence weighted"), ("A/B", "A/C", "B/C")),
        ("Within-method stability across blocks", "Same-block agreement between methods"),
        strict=True,
    ):
        center = np.array([row["median"] if row["median"] is not None else np.nan for row in rows])
        lower = np.array([row["p10"] if row["p10"] is not None else np.nan for row in rows])
        upper = np.array([row["p90"] if row["p90"] is not None else np.nan for row in rows])
        axis.errorbar(range(3), center, yerr=np.stack([center - lower, upper - center]), fmt="o", capsize=6)
        axis.set_xticks(range(3), labels, rotation=12)
        axis.set_xlim(-0.4, 2.4)
        axis.set_ylim(0, 1.05)
        axis.set_title(title)
        axis.set_ylabel("Patient-mean positive-quarter Jaccard")
    fig.suptitle("Matched support unions: patient medians and 10th–90th percentiles")
    fig.tight_layout()
    fig.savefig(output / "matched_union_stability.png", dpi=180)
    plt.close(fig)


def main() -> None:
    """Parse local input and aggregate output destinations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=ROOT / "outputs/ntt10_matched_union/result.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/localization/ntt10-matched-union")
    args = parser.parse_args()
    export(args.result, args.output)


if __name__ == "__main__":
    main()
