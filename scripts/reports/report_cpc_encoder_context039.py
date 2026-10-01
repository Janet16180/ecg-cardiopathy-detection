"""Produce the CPC encoder/context report from completed audited local outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ecg_experiment import ROOT
from ecg_experiment.cpc_encoder_variants039 import architecture_spec
from ecg_experiment.encoder_context_analysis039 import (
    CONTEXTS,
    ENCODERS,
    OUTPUT,
    SEEDS,
    TIERS,
    aggregate,
    cell_path,
)
from ecg_experiment.files import sha256_file


def _score_table(result: dict[str, Any], budget: str) -> list[str]:
    """Render executed per-seed and mean scores for one label budget."""
    lines = ["| Tier | Encoder | Context | 39042 | 39043 | 39044 | Mean AUROC ± seed SD | Mean AP |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for tier in TIERS:
        for encoder in ENCODERS:
            for context in CONTEXTS:
                score = result["scores"][f"{tier}_{encoder}_{budget}_{context}"]
                seeds = " | ".join(f"{value:.5f}" for value in score["auroc"])
                lines.append(f"| {tier}k | {encoder} | {context} | {seeds} | "
                             f"{score['mean_auroc']:.5f} ± {score['sd_auroc']:.5f} | "
                             f"{score['mean_average_precision']:.5f} |")
    return lines


def _contrast_table(result: dict[str, Any]) -> list[str]:
    """Render every prespecified primary contrast with its simultaneous band."""
    lines = ["| Primary comparison | Mean ΔAUROC | Patient 95% CI | Simultaneous 95% band | Promising |",
             "| --- | ---: | --- | --- | --- |"]
    for name, contrast in result["inference"]["contrasts"].items():
        if contrast["primary"]:
            lines.append(f"| {name} | {contrast['difference']:+.5f} | "
                         f"[{contrast['ci_low']:+.5f}, {contrast['ci_high']:+.5f}] | "
                         f"[{contrast['simultaneous_low']:+.5f}, {contrast['simultaneous_high']:+.5f}] | "
                         f"{'yes' if contrast['promising'] else 'no'} |")
    return lines


def _resource_table(root: Path) -> list[str]:
    """Render actual training cost and admitted full-path profile projections."""
    lines = ["| Tier | Encoder | Seed | GRU seconds | xLSTM seconds | Projected cell seconds | GPU peak GB |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for tier in TIERS:
        for seed in SEEDS:
            for encoder in ENCODERS:
                path = cell_path(root, tier, encoder, seed)
                training = json.loads((path / "training.json").read_text())
                profile = json.loads((path / "profile.json").read_text())
                peak = max(item["peak_gpu_memory_bytes"] for item in profile["arms"].values()) / 1e9
                lines.append(f"| {tier}k | {encoder} | {seed} | "
                             f"{training['arms']['gru']['elapsed_seconds']:.2f} | "
                             f"{training['arms']['xlstm']['elapsed_seconds']:.2f} | "
                             f"{profile['projected_total_seconds']:.2f} | {peak:.2f} |")
    return lines


def _plot(root: Path, result: dict[str, Any]) -> None:
    """Save a standalone figure showing the six limited-label architecture packages."""
    destination = root / "docs/figures/experiment-039"
    destination.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, tier in zip(axes, TIERS, strict=True):
        for offset, context in enumerate(CONTEXTS):
            scores = [result["scores"][f"{tier}_{encoder}_limited_{context}"] for encoder in ENCODERS]
            positions = np.arange(len(ENCODERS)) + (offset - 0.5) * 0.18
            axis.errorbar(positions, [item["mean_auroc"] for item in scores],
                          yerr=[item["sd_auroc"] for item in scores], marker="o",
                          linestyle="none", capsize=3, label=context)
            for position, score in zip(positions, scores, strict=True):
                axis.scatter([position] * len(SEEDS), score["auroc"], s=12, alpha=0.45)
        axis.set_xticks(np.arange(len(ENCODERS)), ENCODERS)
        axis.set_title(f"{tier}k cohort; 1,518 labels")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Development AUROC; mean ± training-seed SD")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(destination / "encoder_context.png", dpi=180)
    plt.close(figure)


def report(root: Path = ROOT) -> Path:
    """Write a report only from the complete original factorial and local diagnostic receipts.

    Parameters
    ----------
    root : Path
        Repository root containing the complete audited study.

    Returns
    -------
    Path
        Aggregate results report path.
    """
    result = aggregate(root)
    inference = result["inference"]
    ledger = json.loads((root / OUTPUT / "day_ledger.json").read_text())
    promising = [name for name, value in inference["contrasts"].items() if value["promising"]]
    outcome = (f"The frozen rule identified {len(promising)} promising primary development comparisons."
               if promising else "No original encoder comparison met the frozen promising rule.")
    lines = ["# Experiment 039: CPC encoder/context results", "",
             "All 36 fresh fits completed their fixed exposure budgets and passed artifact audits.", outcome,
             "The complete original factorial is reported below; all outcomes are development-only.", "",
             "## Limited-label results (primary)", "", *_score_table(result, "limited"), "",
             "![AUROC by architecture and seed](figures/experiment-039/encoder_context.png)", "",
             "## Primary paired encoder comparisons", "", *_contrast_table(result), "",
             f"Intervals use {inference['draws']:,} shared whole-patient draws, seed {inference['seed']}; "
             f"{inference['skipped_draws']} single-class draws were skipped. The approximate simultaneous "
             f"band covers eight primary contrasts with radius {inference['simultaneous_radius']:.5f}.",
             "Each draw averages per-seed AUROC differences, without forming a probability ensemble. Seed",
             "spread is reported separately; patient intervals condition on these three trained seeds.", "",
             "## Full-label results (secondary)", "", *_score_table(result, "full"), "",
             "## Secondary context and cohort contrasts", "",
             "| Comparison | Mean ΔAUROC | Paired patient 95% interval |", "| --- | ---: | --- |"]
    for name, value in inference["contrasts"].items():
        if not value["primary"]:
            lines.append(f"| {name} | {value['difference']:+.5f} | "
                         f"[{value['ci_low']:+.5f}, {value['ci_high']:+.5f}] |")
    lines += ["", "## Failure diagnostics and corrections", ""]
    diagnostics = sorted((root / OUTPUT / "diagnostics").glob("*/diagnostic.json"))
    if diagnostics:
        for path in diagnostics:
            receipt = json.loads(path.read_text())
            lines.append(f"- {receipt['encoder']}/{receipt['context']}: {receipt['conclusion']} "
                         f"([local receipt](../{path.relative_to(root).as_posix()})).")
    elif result["severe_failures"]:
        raise ValueError("Severe underperformance requires its one diagnostic before the final report")
    else:
        lines.append("No original fit crossed the severe-development-underperformance trigger. Training-only")
        lines.append("feature-collapse checks appear in the separate diagnostic-gate receipt.")
    lines += ["", "## Execution and integrity", "",
              "Both cohort tiers are v4 manifests verified byte-identical to v3; the existing immutable 038",
              "waveform caches are reused. Historical 019 and both 038 scores were replayed exactly before",
              "new scores. Fresh CNN arms are the matched controls. No waveform or score from calibration,",
              "Challenge test, EchoNext test, SPH or the final frozen test was used.", "",
              "Every fit used 250,000 exposures / 1,954 updates, unchanged ordinary CPC horizons, float32,",
              "AdamW 1e-4/0.01 and clipping at 1. The readout used fixed train-only scaling/logistic C=0.01,",
              "1,518 or 15,359 clean labels and 1,306 development ECGs from 1,173 patients. Native GRU",
              "and the existing 038 mLSTM implementation were reused. Checkpoint recovery passed in every",
              "real full-path profile; saved heads replayed their probabilities exactly in every audit.", "",
              "| Encoder | GRU parameters | xLSTM parameters | Raw receptive field |",
              "| --- | ---: | ---: | ---: |"]
    for encoder in ENCODERS:
        gru = architecture_spec(encoder, "gru")
        xlstm = architecture_spec(encoder, "xlstm")
        lines.append(f"| {encoder} | {gru['total_parameters']:,} | {xlstm['total_parameters']:,} | "
                     f"{gru['receptive_field_samples']} samples |")
    lines += ["", "## Resources", "", *_resource_table(root), "",
              f"Charged new executable work before report generation: {ledger['total_seconds']:.2f} seconds "
              "against the 28,800-second day ceiling; stage failures are included. Historical cache",
              "construction is disclosed in the per-cell profiles and is not charged again.", "",
              "## Limits and follow-ups", "",
              "These are compact concrete architecture packages with different parameter counts, one fixed",
              "optimization recipe and three seeds. The patch arm is a local projection, not a Transformer.",
              "The bounded multiscale support preserves CPC's future-target separation; this study does not",
              "test unrestricted long-window encoders. Development patients were repeatedly inspected in",
              "earlier experiments. No result establishes clinical benefit, diagnosis or a screening",
              "operating point. Cohort scaling also changes source mix and repetitions per record.", "",
              "All executed aggregate evidence, source identities, checkpoints, predictions, heads and stage",
              f"ledgers remain under `{OUTPUT}/`. Aggregate receipt SHA-256: "
              f"`{sha256_file(root / OUTPUT / 'aggregate.json')}`. Follow-ups enter the ranked backlog; the",
              "candidate screening pipeline and closed final test require their own decisions.", ""]
    path = root / "docs/experiment-039-cpc-encoder-context-results.md"
    path.write_text("\n".join(lines))
    _plot(root, result)
    return path


def main() -> None:
    """Generate the report from immutable local results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(report())


if __name__ == "__main__":
    main()
