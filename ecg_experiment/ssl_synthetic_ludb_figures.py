"""Render fixed LUDB timing qualification figures without selecting cases by outcomes."""

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ecg_experiment import ssl_synthetic_ludb_qualification as core


def point_figure(destination: Path, statistics: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Show all twelve fixed lead comparisons and all-record annotation coverage."""
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    points = statistics["per_lead"]
    positions = np.arange(12)
    axes[0].plot(positions, [points[lead]["AP"] for lead in core.LEADS], "o-", label="Original activity AP")
    axes[0].plot(positions, [points[lead]["mean_shift_AP"] for lead in core.LEADS], "o-",
                 label="Mean of six fixed shifted APs")
    axes[0].set(xticks=positions, xticklabels=core.LEADS, ylim=(0, 1), ylabel="Mean subject AP")
    axes[0].legend()
    coverage = [[row["leads"][index]["coverage_fraction"] for row in rows] for index in range(12)]
    axes[1].boxplot(coverage, tick_labels=core.LEADS, showfliers=True)
    axes[1].set(ylim=(0, 1), ylabel="Admitted fraction of5000samples", xlabel="Canonical lead")
    figure.suptitle("Public LUDB QRS timing qualification; wave masks are not disease masks")
    figure.savefig(destination / "lead_AP_and_coverage.png", dpi=120)
    plt.close(figure)


def overlay_figure(destination: Path, cases: dict[int, tuple[dict[str, Any], np.ndarray]]) -> None:
    """Render only prospectively fixed parser IDs5,10,15,20 and leadsII,V2 after primary work."""
    figure, axes = plt.subplots(4, 2, figsize=(16, 12), constrained_layout=True)
    colors = {"P": "tab:blue", "QRS": "tab:red", "T": "tab:green"}
    for index, identifier in enumerate([5, 10, 15, 20]):
        row, signal = cases[identifier]
        for column, lead in enumerate(["ii", "v2"]):
            lead_index = core.LEADS.index(lead)
            metadata, values = row["leads"][lead_index], signal[lead_index]
            start, end = metadata["start"], metadata["end_exclusive"]
            times = np.arange(start, end) / 500
            axis = axes[index, column]
            axis.plot(times, values[start:end], color="black", linewidth=.7)
            for wave in metadata["waves"]:
                axis.axvspan(wave["onset"] / 500, (wave["offset"] + 1) / 500,
                             color=colors[wave["wave"]], alpha=.15)
            twin = axis.twinx()
            twin.plot(times, core.activity_proxy(values)[start:end], color="tab:orange", linewidth=.7)
            twin.set(ylim=(0, 1), ylabel="Fixed proxy")
            axis.set(title=f"Public record{identifier}, lead{lead.upper()} (parser role, no AP)",
                     xlabel="Seconds", ylabel="Released physical signal")
    figure.suptitle("Fixed cases: P blue, QRS red, T green; annotation envelope only")
    figure.savefig(destination / "fixed_wave_proxy_overlays.png", dpi=120)
    plt.close(figure)
