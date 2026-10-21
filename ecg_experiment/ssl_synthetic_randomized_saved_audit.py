"""Independent reconstruction and metric arithmetic for randomized synthetic localization."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from ecg_experiment.intervals import patient_groups, patient_resample
from ecg_experiment.ssl_synthetic_pretrained_saved_audit import require
from ecg_experiment.ssl_synthetic_quality_saved_audit import average_precision, canonical

MODES = ("sham", "nuisance", "focal", "repeated")


def parameters_for_index(index: int) -> list[dict[str, Any]]:
    """Independently regenerate prospective randomized edit parameters.

    Parameters
    ----------
    index : int
        Frozen generator-background position.

    Returns
    -------
    list[dict[str, Any]]
        Ordered focal and repeated intervention parameters.
    """
    generator = np.random.default_rng(2026103400 + index)
    result = []
    for mode in ("focal", "repeated"):
        lead = int(generator.integers(0, 8))
        sign = int(generator.choice([-1.0, 1.0]))
        amplitude = float(generator.uniform(0.4, 0.8))
        width = int(generator.choice([61, 81, 101]))
        spacing = None
        if mode == "focal":
            centers = [int(generator.integers(500, 4501))]
        else:
            start = int(generator.integers(500, 1000))
            spacing = int(generator.integers(400, 601))
            centers = list(range(start, 4501, spacing))
        result.append(
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
    return result


def reconstruct_case(
    raw: np.ndarray, index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Rebuild final observed float32 values and truth without production helpers.

    Parameters
    ----------
    raw : np.ndarray
        Source native eight-lead generator output.
    index : int
        Background index fixing all prospective RNG states.

    Returns
    -------
    tuple
        Four variants, positive strict masks, four thresholded masks and metadata.
    """
    require(raw.dtype == np.float32 and raw.shape == (8, 5000), "parent native geometry")
    native = raw.astype(np.float64)
    center = np.median(native, axis=1)
    low, high = np.quantile(native, [0.05, 0.95], axis=1, method="linear")
    scale = high - low
    require(bool(np.all(scale > 1e-6)), "parent robust scale")
    values = ((native - center[:, None]) / scale[:, None]).astype(np.float32)
    phase = np.random.default_rng(2026103100 + index).uniform(0, 2 * np.pi)
    times = np.arange(5000) / 500
    nuisance = (0.1 * np.sin(2 * np.pi * 0.33 * times + phase)).astype(np.float32)
    baseline = values + nuisance
    edits = parameters_for_index(index)
    variants = [canonical(values), canonical(baseline)]
    for edit in edits:
        changed = baseline.copy()
        width = edit["width"]
        offset = (np.hanning(width) * (edit["sign"] * edit["amplitude"])).astype(np.float32)
        for location in edit["centers"]:
            changed[edit["independent_lead"], location - width // 2 : location + width // 2 + 1] += offset
        variants.append(canonical(changed))
    observed = np.stack(variants)
    difference = observed[2:].astype(np.float64) - observed[1].astype(np.float64)
    strict = difference != 0
    support = np.zeros((4, 12, 5000), dtype=bool)
    support[2:] = np.abs(difference) > 0.01
    row = {
        "index": index,
        "seed": 2026103000 + index,
        "split": "train" if index < 80 else "evaluation",
        "standardization": {"median": center.tolist(), "q95_minus_q05": scale.tolist()},
        "edits": edits,
        "checks": intervention_values(observed, strict, support, edits),
    }
    return observed, strict, support, row


def intervention_values(
    observed: np.ndarray, strict: np.ndarray, support: np.ndarray, edits: list[dict[str, Any]]
) -> dict[str, Any]:
    """Independently verify all affected leads, windows and support energy.

    Parameters
    ----------
    observed, strict, support : np.ndarray
        Four final signals and exact/thresholded numerical masks.
    edits : list[dict[str, Any]]
        Prospective intervention declarations.

    Returns
    -------
    dict[str, Any]
        Complete per-mode integrity measurements.
    """
    result = {}
    for mode_index, edit in enumerate(edits):
        lead = edit["independent_lead"]
        affected = [0, 2, 3, 4, 5] if lead == 0 else [1, 2, 3, 4, 5] if lead == 1 else [lead + 4]
        delta = observed[mode_index + 2].astype(np.float64) - observed[1].astype(np.float64)
        actual_strict, actual_support = delta != 0, np.abs(delta) > 0.01
        require(np.array_equal(actual_strict, strict[mode_index]), "strict support identity")
        require(np.array_equal(actual_support, support[mode_index + 2]), "threshold support identity")
        hull = np.zeros((12, 5000), dtype=bool)
        half = edit["width"] // 2
        for location in edit["centers"]:
            for affected_lead in affected:
                hull[affected_lead, location - half : location + half + 1] = True
        outside = int(np.count_nonzero(actual_strict & ~hull))
        require(outside == 0 and bool(actual_support.any()), "nonempty support confined to hull")
        squared = delta**2
        retained = float(squared[actual_support].sum() / squared.sum())
        require(retained >= 0.99, "intervention retained energy")
        result[edit["mode"]] = {
            "strict_mask_exact": True,
            "threshold_mask_exact": True,
            "nonempty": True,
            "threshold_subset_strict": not bool(np.any(actual_support & ~actual_strict)),
            "outside_hull_cells": outside,
            "affected_canonical_leads": affected,
            "support_cells": int(actual_support.sum()),
            "strict_cells": int(actual_strict.sum()),
            "retained_energy_fraction": retained,
            "below_threshold_energy_fraction": float(squared[~actual_support].sum() / squared.sum()),
            "below_threshold_maximum_residual": float(np.abs(delta[~actual_support]).max()),
        }
    return result


def expected_schedule() -> np.ndarray:
    """Reproduce all matched background permutations and complete-variant batches.

    Returns
    -------
    np.ndarray
        Shape (400, 8) integer flattened training-case indices.
    """
    generator = np.random.default_rng(2026103502)
    result = []
    for _ in range(10):
        shuffled = generator.permutation(80)
        for position in range(0, 80, 2):
            pair = shuffled[position : position + 2]
            result.append([int(index * 4 + mode) for index in pair for mode in range(4)])
    return np.array(result, dtype=np.int64)


def score_rows(
    probability: np.ndarray,
    observed: np.ndarray,
    support: np.ndarray,
    spatial_prior: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Independently score full-grid AP, matched-baseline response and selectivity.

    Parameters
    ----------
    probability, observed, support : np.ndarray
        Held-out seed predictions, signals and masks in four-variant order.
    spatial_prior : np.ndarray or None
        Training-only mean positive support map, when declared.

    Returns
    -------
    list[dict[str, Any]]
        Per-seed measurements.
    """
    require(probability.shape == (48, 4, 12, 5000) and probability.dtype == np.float32, "map geometry")
    require(
        bool(np.isfinite(probability).all()) and bool(((probability >= 0) & (probability <= 1)).all()),
        "finite probability range",
    )
    result = []
    for index in range(48):
        row = {"index": index + 80, "seed": 2026103080 + index}
        for position, mode in ((2, "focal"), (3, "repeated")):
            mask = support[index, position]
            edited = average_precision(mask, probability[index, position])
            baseline = average_precision(mask, probability[index, 1])
            delta = probability[index, position].astype(np.float64) - probability[index, 1].astype(np.float64)
            inside, outside = float(delta[mask].mean()), float(delta[~mask].mean())
            row[mode] = {
                "edited_ap": edited,
                "nuisance_ap": baseline,
                "response_ap": edited - baseline,
                "amplitude_ap": average_precision(mask, np.abs(observed[index, position])),
                "prevalence": float(mask.mean()),
                "delta_inside": inside,
                "delta_outside": outside,
                "selectivity": inside - outside,
            }
            if spatial_prior is not None:
                row[mode]["spatial_prior_ap"] = average_precision(mask, spatial_prior)
        row["mean_positive_ap"] = (row["focal"]["edited_ap"] + row["repeated"]["edited_ap"]) / 2
        result.append(row)
    return result


def paired_interval(values: np.ndarray) -> dict[str, Any]:
    """Compute a fixed whole-seed paired percentile interval.

    Parameters
    ----------
    values : np.ndarray
        One paired value per 48 evaluation generator seeds.

    Returns
    -------
    dict[str, Any]
        Frozen seed-bootstrap interval metadata.
    """
    require(values.shape == (48,), "48 whole-seed paired values")
    groups = patient_groups(np.arange(48))
    rng = np.random.default_rng(2026103503)
    draws = [float(values[patient_resample(groups, rng)].mean()) for _ in range(2000)]
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "mean": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "draws": 2000,
        "seed": 2026103503,
        "skipped_draws": 0,
        "unit": "generator_seed",
    }


def initial_state() -> dict[str, torch.Tensor]:
    """Regenerate the one declared CPU initialization without a model forward.

    Returns
    -------
    dict[str, torch.Tensor]
        Six exact initial parameter tensors.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026103501)
        model = nn.Sequential(
            nn.Conv1d(12, 32, 25, padding=12),
            nn.ReLU(),
            nn.Conv1d(32, 32, 25, dilation=4, padding=48),
            nn.ReLU(),
            nn.Conv1d(32, 12, 1),
        )
    return model.state_dict()


def summarize_rows(supervised: list[dict[str, Any]], permuted: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute all paired scientific contrasts and the twelve-gate conjunction.

    Parameters
    ----------
    supervised, permuted : list[dict[str, Any]]
        Independently reconstructed per-seed metric rows.

    Returns
    -------
    dict[str, Any]
        Contrast intervals, gate values and complete prospective decision.
    """
    contrasts, gates = {}, {}
    for comparator in ("absolute_amplitude", "permuted"):
        gains = []
        for mode in ("focal", "repeated"):
            direct = np.array([row[mode]["edited_ap"] for row in supervised])
            other = (
                np.array([row[mode]["edited_ap"] for row in permuted])
                if comparator == "permuted"
                else np.array([row[mode]["amplitude_ap"] for row in supervised])
            )
            difference = direct - other
            gains.append(difference)
            key = f"supervised_minus_{comparator}_{mode}"
            contrasts[key] = paired_interval(difference)
            gates[key + "_guard"] = contrasts[key]["ci_low"] > -0.01
        key = f"supervised_minus_{comparator}_pooled"
        contrasts[key] = paired_interval((gains[0] + gains[1]) / 2)
        gates[key + "_mean"] = contrasts[key]["mean"] >= 0.05
        gates[key + "_lower"] = contrasts[key]["ci_low"] > 0
    for mode in ("focal", "repeated"):
        for measure in ("response_ap", "selectivity"):
            key = f"supervised_{mode}_{measure}"
            contrasts[key] = paired_interval(np.array([row[mode][measure] for row in supervised]))
            gates[key + "_lower"] = contrasts[key]["ci_low"] > 0
    return {"contrasts": contrasts, "scientific_gates": gates, "scientific_passed": all(gates.values())}


def control_rows(probability: np.ndarray) -> dict[str, Any]:
    """Recompute descriptive sham/nuisance levels and strict fixed-cutoff flags.

    Parameters
    ----------
    probability : np.ndarray
        Full held-out probability bank.

    Returns
    -------
    dict[str, Any]
        Complete control summary including all record means/maxima.
    """
    result = {}
    for position, mode in ((0, "sham"), (1, "nuisance")):
        values = probability[:, position].reshape(48, -1).astype(np.float64)
        record_mean, record_max = values.mean(axis=1), values.max(axis=1)
        result[mode] = {
            "mean_probability": float(values.mean()),
            "maximum_probability": float(values.max()),
            "record_maximum_gt_0p5_fraction": float(np.mean(record_max > 0.5)),
            "record_mean_probabilities": record_mean.tolist(),
            "record_maximum_probabilities": record_max.tolist(),
        }
    return result


def verify_optimizer(state: dict[str, Any], model_state: dict[str, torch.Tensor], step: int) -> None:
    """Validate complete saved Adam shape, moment and step declarations without updates.

    Parameters
    ----------
    state : dict[str, Any]
        Weights-only CPU-loaded optimizer state.
    model_state : dict[str, torch.Tensor]
        Named finite model parameter tensors in registration order.
    step : int
        Prospective update count.
    """
    require(set(state) == {"state", "param_groups"} and len(state["param_groups"]) == 1, "one Adam group")
    group = state["param_groups"][0]
    required = {
        "lr": 0.001,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0,
        "amsgrad": False,
        "foreach": False,
        "fused": False,
        "maximize": False,
        "capturable": False,
        "differentiable": False,
    }
    require(all(group[key] == value for key, value in required.items()), "frozen Adam options")
    ids = group["params"]
    require(
        len(ids) == 6 and len(set(ids)) == 6 and set(state["state"]) == set(ids), "six optimizer parameters"
    )
    for identifier, parameter in zip(ids, model_state.values(), strict=True):
        value = state["state"][identifier]
        require(set(value) == {"step", "exp_avg", "exp_avg_sq"}, "complete standard Adam state")
        count = value["step"]
        require(
            isinstance(count, torch.Tensor) and count.numel() == 1 and count.item() == step,
            "exact optimizer step count",
        )
        for name in ("exp_avg", "exp_avg_sq"):
            tensor = value[name]
            require(tensor.dtype == torch.float32 and tensor.shape == parameter.shape, "Adam moment geometry")
            require(bool(torch.isfinite(tensor).all()), "finite Adam moments")
        require(bool((value["exp_avg_sq"] >= 0).all()), "nonnegative second moment")


def exact_tree(left: Any, right: Any) -> bool:
    """Compare saved model/optimizer objects, requiring native tensor byte equality.

    Parameters
    ----------
    left, right : Any
        Strict CPU-loaded saved state trees.

    Returns
    -------
    bool
        Complete tree identity.
    """
    if isinstance(left, torch.Tensor):
        return (
            isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and left.contiguous().numpy().tobytes() == right.contiguous().numpy().tobytes()
        )
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(exact_tree(value, right[key]) for key, value in left.items())
        )
    if isinstance(left, (tuple, list)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(exact_tree(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right
