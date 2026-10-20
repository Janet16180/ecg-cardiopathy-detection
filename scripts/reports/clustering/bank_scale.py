"""Export explicitly allowlisted aggregate CLUSTER-06 summaries and a static figure."""

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
    """Export fixed sensitivity summaries without identities, vectors or individual assignments.

    Parameters
    ----------
    result_path : Path
        Completed source-bound local result.
    output : Path
        Tracked aggregate report directory.

    Returns
    -------
    None
    """
    result = json.loads(result_path.read_text())
    if result["status"] != "completed":
        raise ValueError("Cannot export incomplete study")
    output.mkdir(parents=True, exist_ok=True)
    rows = [row for checks in result["equivalence_checks"].values() for row in checks]
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
            "predecessor_replay",
            "summaries",
            "fixed_floor_raw_rms_ari",
            "equivalence_passed",
            "waveform_reads",
            "new_encoder_inference",
            "gpu_used",
            "target_annotations_parsed",
        )
    }
    aggregate["result_sha256"] = sha256_file(result_path)
    aggregate["numerical_equivalence"] = {
        "comparisons": len(rows),
        "passed": sum(row["passed"] for row in rows),
        "minimum_all_ari": min(row["all_ari"] for row in rows),
        "minimum_anchor_ari": min(row["anchor_ari"] for row in rows),
        "maximum_errors": {
            key: max(row["maximum_errors"][key] for row in rows) for key in rows[0]["maximum_errors"]
        },
        "failed_conditions": {
            key: sum(not row["conditions"][key] for row in rows) for key in rows[0]["conditions"]
        },
    }
    write_json_atomic(output / "aggregate.json", aggregate)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for axis, configuration in zip(axes, ("regional_tied2", "regional_diag2", "record_full3"), strict=True):
        for offset, kind, label in (
            (-0.17, "single", "50 single banks"),
            (0.17, "ensemble", "10 five-bank means"),
        ):
            values = [
                result["summaries"][f"{configuration}:{kind}:{scale}"]["pairwise_anchor_ari"]["p10"]
                for scale in ("raw", "rms", "equivalent")
            ]
            axis.bar(np.arange(3) + offset, values, width=0.34, label=label)
        axis.set_xticks(range(3), ("Raw", "RMS, fixed floor", "RMS, scaled floor"), rotation=20)
        axis.set_title(configuration.replace("_", " "))
        axis.set_ylim(0, 1)
    axes[0].set_ylabel("10th percentile pairwise anchor ARI")
    axes[0].legend(fontsize=8)
    fig.suptitle("Conditional support-bank stability at fixed GMM configurations")
    fig.tight_layout()
    fig.savefig(output / "bank_scale_stability.png", dpi=180)
    plt.close(fig)


def main() -> None:
    """Parse explicit local input and aggregate output destinations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=ROOT / "outputs/cluster06_bank_scale/result.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/clustering/cluster06-bank-scale")
    args = parser.parse_args()
    export(args.result, args.output)


if __name__ == "__main__":
    main()
