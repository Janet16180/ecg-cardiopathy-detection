"""Plot saved visual-reference choices without rerunning models or changing results."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from ecg_experiment import ROOT
from ecg_experiment.visual_reference_ntt03 import physical_lead


def main() -> None:
    """Save the fixed first target of each class with its cached reference scores."""
    output = ROOT / "outputs/ntt03_visual_reference"
    result = json.loads((output / "result.json").read_text())
    selected = json.loads((output / "selection_local.json").read_text())
    scores = {row["qid"]: row for row in result["candidate_results"] if row["cropped"]}
    fig, axes = plt.subplots(len(selected), 2, figsize=(13, 3 * len(selected)), constrained_layout=True)
    for index, group in enumerate(selected):
        target = group["targets"][0]
        cached = scores[target["qid"]]
        trace = physical_lead(target)
        axis, bars = axes[index]
        axis.plot(np.arange(len(trace)) / 100, trace, color="black", linewidth=0.8)
        for option, interval in enumerate(target["intervals"]):
            color = "tab:blue" if option == target["answer"] else "gray"
            axis.axvspan(*interval, color=color, alpha=0.2)
            axis.text(np.mean(interval), axis.get_ylim()[1], str(option + 1), ha="center", va="top")
        axis.set(
            xlim=(0, 10),
            xlabel="Seconds",
            ylabel="mV",
            title=group["diagnosis"].replace("_", " ") + f" · supplied lead {target['lead']}",
        )
        positions = np.arange(len(target["intervals"]))
        bars.bar(positions - 0.18, cached["real_scores"], width=0.36, label="Marked references")
        bars.bar(positions + 0.18, cached["wrong_scores"], width=0.36, label="Wrong-window references")
        bars.set(
            xticks=positions,
            xticklabels=[str(i + 1) for i in positions],
            xlabel="Supplied candidate window",
            ylabel="Cosine similarity",
            ylim=(0, 1),
            title=f"Nominated answer: {target['answer'] + 1}; selected: {cached['real_choice'] + 1}",
        )
        bars.legend(fontsize=8)
    fig.suptitle(
        "Frozen visual matching on supplied windows; blue = nominated silver evidence\n"
        "Known finding/lead, three reference patients; no autonomous diagnostic claim",
        fontsize=12,
    )
    fig.savefig(output / "reference_matching_examples.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
