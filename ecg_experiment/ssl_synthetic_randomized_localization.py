"""Learn introduced numerical support on generated backgrounds with randomized edit geometry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score
from torch import nn

from ecg_experiment import ssl_synthetic_generated_quality as quality
from ecg_experiment.intervals import patient_groups, patient_resample

MODES = ("sham", "nuisance", "focal", "repeated")


def edit_parameters(index: int) -> list[dict[str, Any]]:
    """Draw the frozen focal then repeated parameters from a background-specific RNG.

    Parameters
    ----------
    index : int
        Fixed generated-background index, from zero through 127.

    Returns
    -------
    list
        Focal and repeated lead, sign, amplitude, width and center declarations.
    """
    rng = np.random.default_rng(2026103400 + index)
    parameters = []
    for mode in ("focal", "repeated"):
        lead = int(rng.integers(0, 8))
        sign = float(rng.choice([-1.0, 1.0]))
        amplitude = float(rng.uniform(0.4, 0.8))
        width = int(rng.choice([61, 81, 101]))
        if mode == "focal":
            centers = [int(rng.integers(500, 4501))]
            spacing = None
        else:
            start = int(rng.integers(500, 1000))
            spacing = int(rng.integers(400, 601))
            centers = list(range(start, 4501, spacing))
        parameters.append(
            {
                "mode": mode,
                "independent_lead": lead,
                "sign": sign,
                "amplitude": amplitude,
                "width": width,
                "centers": centers,
                "spacing": spacing,
            }
        )
    return parameters


def intervention_checks(
    variants: np.ndarray, strict: np.ndarray, support: np.ndarray, parameters: list[dict[str, Any]]
) -> dict[str, Any]:
    """Verify exact final numerical truth, affected-lead hulls and retained intervention energy."""
    checks = {}
    for position, parameter in enumerate(parameters):
        difference = variants[position + 2].astype(np.float64) - variants[1].astype(np.float64)
        basis = np.zeros((8, 1), dtype=np.float32)
        basis[parameter["independent_lead"]] = 1
        affected = np.flatnonzero(quality.derive_leads(basis)[:, 0] != 0)
        hull = np.zeros((12, 5000), dtype=bool)
        half = parameter["width"] // 2
        for center in parameter["centers"]:
            hull[affected, center - half : center + half + 1] = True
        energy = difference**2
        mask = support[position + 2]
        checks[parameter["mode"]] = {
            "strict_mask_exact": bool(np.array_equal(strict[position], difference != 0)),
            "threshold_mask_exact": bool(np.array_equal(mask, np.abs(difference) > 0.01)),
            "nonempty": bool(mask.any() and strict[position].any()),
            "threshold_subset_strict": bool(np.all(~mask | strict[position])),
            "outside_hull_cells": int(np.count_nonzero(strict[position] & ~hull)),
            "affected_canonical_leads": affected.tolist(),
            "support_cells": int(mask.sum()),
            "strict_cells": int(strict[position].sum()),
            "retained_energy_fraction": float(energy[mask].sum() / energy.sum()),
            "below_threshold_energy_fraction": float(energy[~mask].sum() / energy.sum()),
            "below_threshold_maximum_residual": float(np.abs(difference[~mask]).max()),
        }
    return checks


def make_case(raw: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build new randomized fixtures without consuming the earlier fixed-position variants.

    Parameters
    ----------
    raw : np.ndarray
        Native eight-lead float32 generator output.
    index : int
        Fixed background index determining the prospective random draws.

    Returns
    -------
    tuple
        Four observed variants, two strict masks, four target masks and metadata.
    """
    standardized, transformation = quality.standardize(raw)
    phase = np.random.default_rng(2026103100 + index).uniform(0, 2 * np.pi)
    nuisance = (0.1 * np.sin(2 * np.pi * 0.33 * (np.arange(5000) / 500) + phase)).astype(np.float32)
    baseline = (standardized + nuisance).astype(np.float32)
    parameters = edit_parameters(index)
    variants = [quality.derive_leads(standardized), quality.derive_leads(baseline)]
    for parameter in parameters:
        independent = baseline.copy()
        bump = (parameter["sign"] * parameter["amplitude"] * np.hanning(parameter["width"])).astype(
            np.float32
        )
        half = parameter["width"] // 2
        for center in parameter["centers"]:
            independent[parameter["independent_lead"], center - half : center + half + 1] += bump
        variants.append(quality.derive_leads(independent))
    observed = np.stack(variants)
    difference = observed[2:].astype(np.float64) - observed[1].astype(np.float64)
    strict = difference != 0
    support = np.zeros((4, 12, 5000), dtype=bool)
    support[2:] = np.abs(difference) > 0.01
    row = {
        "index": index,
        "seed": 2026103000 + index,
        "split": "train" if index < 80 else "evaluation",
        "standardization": transformation,
        "edits": parameters,
        "checks": intervention_checks(observed, strict, support, parameters),
    }
    return observed, strict, support, row


def build_model() -> nn.Sequential:
    """Create the fixed observed-only fully convolutional localization network."""
    return nn.Sequential(
        nn.Conv1d(12, 32, 25, padding=12),
        nn.ReLU(),
        nn.Conv1d(32, 32, 25, dilation=4, padding=48),
        nn.ReLU(),
        nn.Conv1d(32, 12, 1),
    )


def training_schedule() -> np.ndarray:
    """Return all 400 matched batches of two complete four-variant backgrounds."""
    rng = np.random.default_rng(2026103502)
    pairs = np.concatenate([rng.permutation(80).reshape(40, 2) for _ in range(10)])
    return (pairs[:, :, None] * 4 + np.arange(4)[None, None, :]).reshape(400, 8)


def training_targets(support: np.ndarray, arm: str) -> np.ndarray:
    """Keep the supervised labels or assign the fixed next-background positive-mask control."""
    if support.shape != (80, 4, 12, 5000) or support.dtype != np.bool_:
        raise ValueError("Training targets require exactly the 80 training backgrounds")
    if arm == "supervised":
        return support.reshape(320, 12, 5000)
    if arm != "permuted":
        raise ValueError("Unknown prospective training arm")
    targets = support.copy()
    targets[:, 2:] = support[(np.arange(80) + 1) % 80, 2:]
    return targets.reshape(320, 12, 5000)


def positive_weight(support: np.ndarray) -> float:
    """Fit the frozen capped positive class weight using training masks alone."""
    if support.shape != (80, 4, 12, 5000):
        raise ValueError("Class weighting may read training masks only")
    positives = np.count_nonzero(support)
    if not positives:
        raise ValueError("Training supports contain no positive cells")
    return float(min(20, (support.size - positives) / positives))


def score_maps(probabilities: np.ndarray, observed: np.ndarray, support: np.ndarray) -> list[dict[str, Any]]:
    """Score both positive modes against their masks and matched nuisance controls.

    Parameters
    ----------
    probabilities : np.ndarray
        Saved 48 by 4 by 12 by 5000 probability maps from one fitted arm.
    observed : np.ndarray
        Corresponding observed evaluation variants.
    support : np.ndarray
        Separate scoring targets, absent from model inputs.

    Returns
    -------
    list
        Every evaluation seed's AP, paired response, selectivity and baseline scores.
    """
    rows = []
    for index in range(48):
        row = {"index": index + 80, "seed": 2026103080 + index}
        for position, mode in [(2, "focal"), (3, "repeated")]:
            mask = support[index, position].ravel()
            edited_ap = float(average_precision_score(mask, probabilities[index, position].ravel()))
            nuisance_ap = float(average_precision_score(mask, probabilities[index, 1].ravel()))
            amplitude_ap = float(average_precision_score(mask, np.abs(observed[index, position]).ravel()))
            delta = probabilities[index, position].astype(np.float64) - probabilities[index, 1].astype(
                np.float64
            )
            inside = float(delta.ravel()[mask].mean())
            outside = float(delta.ravel()[~mask].mean())
            row[mode] = {
                "edited_ap": edited_ap,
                "nuisance_ap": nuisance_ap,
                "response_ap": edited_ap - nuisance_ap,
                "delta_inside": inside,
                "delta_outside": outside,
                "selectivity": inside - outside,
                "amplitude_ap": amplitude_ap,
                "prevalence": float(mask.mean()),
            }
        row["mean_positive_ap"] = (row["focal"]["edited_ap"] + row["repeated"]["edited_ap"]) / 2
        rows.append(row)
    return rows


def paired_interval(values: np.ndarray) -> dict[str, Any]:
    """Bootstrap all fixed evaluation generator seeds with the shared grouping helpers."""
    groups = patient_groups(np.arange(48))
    rng = np.random.default_rng(2026103503)
    draws = [float(values[patient_resample(groups, rng)].mean()) for _ in range(2000)]
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "mean": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "draws": 2000,
        "seed": 2026103503,
        "skipped_draws": 0,
        "unit": "generator_seed",
    }


def summarize_metrics(supervised: list[dict[str, Any]], permuted: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute frozen paired contrasts and all twelve prospective scientific gates.

    Parameters
    ----------
    supervised, permuted : list
        Matched 48-seed rows from the two fixed fitted arms.

    Returns
    -------
    dict
        All paired intervals, twelve gate values and their conjunction.
    """
    contrasts, gates = {}, {}
    for baseline in ["absolute_amplitude", "permuted"]:
        mode_values = []
        for mode in ["focal", "repeated"]:
            control = np.array(
                [
                    other[mode]["edited_ap"] if baseline == "permuted" else row[mode]["amplitude_ap"]
                    for row, other in zip(supervised, permuted, strict=True)
                ]
            )
            values = np.array([row[mode]["edited_ap"] for row in supervised]) - control
            mode_values.append(values)
            key = f"supervised_minus_{baseline}_{mode}"
            contrasts[key] = paired_interval(values)
            gates[key + "_guard"] = contrasts[key]["ci_low"] > -0.01
        key = f"supervised_minus_{baseline}_pooled"
        contrasts[key] = paired_interval(np.mean(mode_values, axis=0))
        gates[key + "_mean"] = contrasts[key]["mean"] >= 0.05
        gates[key + "_lower"] = contrasts[key]["ci_low"] > 0
    for mode in ["focal", "repeated"]:
        for metric in ["response_ap", "selectivity"]:
            key = f"supervised_{mode}_{metric}"
            contrasts[key] = paired_interval(np.array([row[mode][metric] for row in supervised]))
            gates[key + "_lower"] = contrasts[key]["ci_low"] > 0
    return {"contrasts": contrasts, "scientific_gates": gates, "scientific_passed": all(gates.values())}


def control_scores(probabilities: np.ndarray) -> dict[str, Any]:
    """Describe held-out sham/nuisance predictions using the fixed threshold only."""
    result = {}
    for position, mode in [(0, "sham"), (1, "nuisance")]:
        values = probabilities[:, position].reshape(48, -1).astype(np.float64)
        result[mode] = {
            "mean_probability": float(values.mean()),
            "maximum_probability": float(values.max()),
            "record_maximum_gt_0p5_fraction": float(np.mean(values.max(axis=1) > 0.5)),
            "record_mean_probabilities": values.mean(axis=1).tolist(),
            "record_maximum_probabilities": values.max(axis=1).tolist(),
        }
    return result


def add_spatial_prior(rows: list[dict[str, Any]], support: np.ndarray, prior: np.ndarray) -> None:
    """Add the fixed training-derived spatial baseline to each held-out positive case."""
    for index, row in enumerate(rows):
        for position, mode in [(2, "focal"), (3, "repeated")]:
            row[mode]["spatial_prior_ap"] = float(
                average_precision_score(support[index, position].ravel(), prior.ravel())
            )


def make_figures(
    destination: Path,
    observed: np.ndarray,
    support: np.ndarray,
    maps: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> list[str]:
    """Render the first eight fixed evaluation seeds, loss curves and control distributions.

    Parameters
    ----------
    destination : Path
        Local study directory containing the saved losses.
    observed, support : np.ndarray
        Evaluation waveforms and target masks.
    maps : dict
        Both arms' evaluation probability maps.
    rows : list
        All 128 background metadata records.
    metrics : dict
        Fixed paired contrasts and intervals.

    Returns
    -------
    list
        Three predetermined local PNG artifact names.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = []
    times = np.arange(5000) / 500
    for position, mode in [(2, "focal"), (3, "repeated")]:
        fig, axes = plt.subplots(8, 4, figsize=(20, 20))
        for index in range(8):
            independent = rows[index + 80]["edits"][position - 2]["independent_lead"]
            lead = independent if independent < 2 else independent + 4
            axes[index, 0].plot(times, observed[index, 1, lead], linewidth=0.5, label="Nuisance")
            axes[index, 0].plot(times, observed[index, position, lead], linewidth=0.5, label="Edited")
            axes[index, 0].plot(
                times,
                observed[index, position, lead] - observed[index, 1, lead],
                linewidth=0.5,
                label="Difference",
            )
            axes[index, 0].set_ylabel(f"Seed {2026103080 + index}; robust scale units")
            axes[index, 1].imshow(
                support[index, position],
                aspect="auto",
                interpolation="nearest",
                extent=(0, 10, 12, 0),
                vmin=0,
                vmax=1,
            )
            for column, arm in [(2, "supervised"), (3, "permuted")]:
                axes[index, column].imshow(
                    maps[arm][index, position],
                    aspect="auto",
                    interpolation="nearest",
                    extent=(0, 10, 12, 0),
                    vmin=0,
                    vmax=1,
                )
        for column, title in enumerate(
            [
                mode + " observed and numerical difference",
                "Introduced support",
                "Supervised probability",
                "Permuted-target probability",
            ]
        ):
            axes[0, column].set_title(title)
            axes[-1, column].set_xlabel("Seconds")
        axes[0, 0].legend(fontsize=6)
        fig.tight_layout()
        name = f"first_eight_{mode}.png"
        fig.savefig(destination / name, dpi=110)
        plt.close(fig)
        names.append(name)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for arm in ["supervised", "permuted"]:
        losses = np.load(destination / arm / "losses.npy", allow_pickle=False)
        axes[0].plot(np.arange(1, 401), losses, label=arm, linewidth=0.7)
        for position, mode in [(0, "sham"), (1, "nuisance")]:
            maxima = maps[arm][:, position].reshape(48, -1).max(axis=1)
            axes[2].hist(maxima, bins=np.linspace(0, 1, 21), histtype="step", label=f"{arm}/{mode}")
    contrast_names = list(metrics["contrasts"])
    values = [metrics["contrasts"][key] for key in contrast_names]
    axes[1].errorbar(
        np.arange(len(values)),
        [v["mean"] for v in values],
        yerr=np.array([[v["mean"] - v["ci_low"], v["ci_high"] - v["mean"]] for v in values]).T,
        fmt="o",
    )
    axes[1].set_xticks(np.arange(len(values)), contrast_names, rotation=90, fontsize=5)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[0].set(xlabel="Optimizer update", ylabel="Weighted BCE loss")
    axes[2].set(xlabel="Control record maximum probability", ylabel="Count")
    axes[0].legend()
    axes[2].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(destination / "learning_diagnostics.png", dpi=120)
    plt.close(fig)
    return names + ["learning_diagnostics.png"]
