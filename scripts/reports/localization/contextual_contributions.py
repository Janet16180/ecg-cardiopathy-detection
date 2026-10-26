"""Export aggregate-only contextual-contribution results and a standalone research figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ecg_experiment.files import sha256_file, write_json_atomic


def export(result_path: Path, output: Path) -> dict:
    """Copy an explicit aggregate allowlist without patient arrays, identities or input hashes."""
    result = json.loads(result_path.read_text())
    if result["status"] != "completed":
        raise ValueError("Only a completed result can enter the aggregate report")
    names = (
        "study",
        "status",
        "git_commit",
        "primary_engineering_gate_passed",
        "gates",
        "maximum_errors",
        "predecessor_reproduction",
        "aggregate",
        "deletion_fractions",
        "random_subsets_per_fraction",
        "context_scope",
        "waveform_reads",
        "new_encoder_inferences",
        "new_clinical_annotations",
        "clinical_localization_evaluated",
        "closed_tests_read",
        "seconds",
    )
    public = {name: result[name] for name in names}
    public["result_sha256"] = sha256_file(result_path)
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output / "aggregate.json", public)
    plot(public, output / "contextual_contributions.png")
    return public


def plot(public: dict, path: Path) -> None:
    """Plot patient-first aggregate descriptors without individual signal or attribution data."""
    metrics = public["aggregate"]["patient_first_metrics"]
    consistency = public["aggregate"]["bank_consistency"]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    x = np.arange(2)
    fraction = [metrics[f"{level}_effective_fraction"] for level in ("beat", "token")]
    medians = np.array([row["median"] for row in fraction])
    errors = np.array(
        [[row["median"] - row["p10"] for row in fraction], [row["p90"] - row["median"] for row in fraction]]
    )
    axes[0].bar(x, medians, color=["tab:blue", "tab:orange"], yerr=errors, capsize=4)
    axes[0].set(
        xticks=x,
        xticklabels=["Beats", "Active tokens"],
        ylim=(0, 1.05),
        ylabel="Effective fraction: median, p10–p90",
        title="Contribution mass is distributed",
    )
    x = np.arange(3)
    fractions = ("0.1", "0.25", "0.5")
    selected = [metrics[f"deletion_selected_drop_{fraction}"]["mean"] for fraction in fractions]
    random = [metrics[f"deletion_random_mean_drop_{fraction}"]["mean"] for fraction in fractions]
    axes[1].bar(x - 0.18, selected, width=0.36, label="Top signed contributions")
    axes[1].bar(x + 0.18, random, width=0.36, label="Random subsets")
    axes[1].set(
        xticks=x,
        xticklabels=["10%", "25%", "50%"],
        xlabel="Fraction of cached beats removed (ceiling)",
        ylabel="Mean fixed-contrast margin drop",
        title="Representation deletion sensitivity",
    )
    axes[1].legend(fontsize=8)
    rows = [consistency["metrics"][name] for name in ("top_quarter_jaccard", "signed_rank_spearman")]
    medians = np.array([row["median"] for row in rows])
    errors = np.array(
        [[row["median"] - row["p10"] for row in rows], [row["p90"] - row["median"] for row in rows]]
    )
    axes[2].bar(np.arange(2), medians, yerr=errors, capsize=4, color="tab:green")
    axes[2].set(
        xticks=np.arange(2),
        xticklabels=["Top-quarter\nJaccard", "Signed-rank\nSpearman"],
        ylim=(-0.2, 1.05),
        ylabel="Patient-first median, p10–p90",
        title=f"Same-contrast banks: {consistency['same_ordered_contrast_pairs']}/180 pairs",
    )
    figure.suptitle("Contextual model contributions: clinical localization not evaluated", fontsize=13)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    """Write safe aggregate tables and a figure from the completed private receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result", type=Path, default=Path("outputs/ntt08_contextual_contributions/result.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/localization/ntt08-contextual-contributions")
    )
    args = parser.parse_args()
    export(args.result, args.output)


if __name__ == "__main__":
    main()
