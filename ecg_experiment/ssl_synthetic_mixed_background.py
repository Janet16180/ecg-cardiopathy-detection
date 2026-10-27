"""Matched-budget mixed-background localizer training and fixed transfer comparisons."""

from typing import Any

import numpy as np

from ecg_experiment import ssl_synthetic_background_editor_transfer as transfer
from ecg_experiment import ssl_synthetic_randomized_localization as original
from ecg_experiment.intervals import patient_groups, patient_resample

TRAINING_ARMS = ("generated", "mixed")
METRICS = (
    "edited_ap",
    "nuisance_site_ap",
    "response_ap",
    "amplitude_ap",
    "amplitude_gain",
    "donor_site_ap",
    "donor_gain",
    "selectivity",
)


def real_case(physical: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply the original training editor after exactly one native-lead robust transformation.

    Parameters
    ----------
    physical : np.ndarray
        Finite canonical12-lead PTB-XL physical samples with shape(12,5000).
    index : int
        Replaced original training geometry slot40 through79.

    Returns
    -------
    tuple
        Four original observed variants, strict masks, threshold masks and frozen transformation metadata.
    """
    if physical.shape != (12, 5000) or not np.isfinite(physical).all() or index not in range(40, 80):
        raise ValueError("Require the fixed real training geometry and finite canonical physical samples")
    native = physical[list(transfer.NATIVE)].astype(np.float32)
    return original.make_case(native, index)


def indexed_rows(rows: list[dict[str, Any]]) -> dict[tuple, dict[str, Any]]:
    """Require all paired training-arm, fit, evaluation-source and geometry identities exactly once.

    Parameters
    ----------
    rows : list
        Complete768 fixed model/background rows with a separate training_arm identity.

    Returns
    -------
    dict
        Complete unique identity-to-row mapping with no cohort or fit selection.
    """
    indexed = {(r["training_arm"], r["source"], r["slot"], r["fit_id"]): r for r in rows}
    expected = {
        (arm, source, slot, fit)
        for arm in TRAINING_ARMS
        for source in transfer.SOURCES
        for slot in range(64)
        for fit in transfer.FITS
    }
    if len(rows) != len(expected) or set(indexed) != expected or any(r["arm"] != "B" for r in rows):
        raise ValueError("Every fixed paired-training/evaluation/fit row is required exactly once")
    return indexed


def positive_vectors(indexed: dict[tuple, dict[str, Any]]) -> dict[str, np.ndarray]:
    """Average modes within record and then fixed fits, retaining every source/family/mode endpoint.

    Parameters
    ----------
    indexed : dict
        Complete fixed row mapping.

    Returns
    -------
    dict
        Ordered64-slot conditional endpoint vectors, including each mode and the equal-mode pool.
    """
    vectors = {}
    for arm in TRAINING_ARMS:
        for source in transfer.SOURCES:
            for family in ("hann", "biphasic"):
                for mode in ("focal", "repeated", "pooled"):
                    names = [
                        f"{family}_{m}" for m in (("focal", "repeated") if mode == "pooled" else (mode,))
                    ]
                    for metric in METRICS:
                        vectors[f"{arm}__{source}__{family}__{mode}__{metric}"] = np.array(
                            [
                                np.mean(
                                    [
                                        np.mean(
                                            [
                                                indexed[arm, source, slot, fit]["positives"][name][metric]
                                                for name in names
                                            ]
                                        )
                                        for fit in transfer.FITS
                                    ]
                                )
                                for slot in range(64)
                            ]
                        )
    return vectors


def control_vectors(indexed: dict[tuple, dict[str, Any]]) -> dict[str, np.ndarray]:
    """Retain matched sham/nuisance magnitudes without calling them clinical alarm rates.

    Parameters
    ----------
    indexed : dict
        Complete fixed row mapping.

    Returns
    -------
    dict
        Per-record means across all three fixed fits for each declared control endpoint.
    """
    return {
        f"{arm}__{source}__{control}__{metric}": np.array(
            [
                np.mean(
                    [indexed[arm, source, slot, fit]["controls"][control][metric] for fit in transfer.FITS]
                )
                for slot in range(64)
            ]
        )
        for arm in TRAINING_ARMS
        for source in transfer.SOURCES
        for control in ("sham", "nuisance")
        for metric in ("mean", "q95", "maximum", "top600_mean", "maximum_gt_0p5")
    }


def per_fit_points(indexed: dict[tuple, dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve every fitted-state and edit-mode point without winner selection.

    Parameters
    ----------
    indexed : dict
        Complete paired training-arm row mapping.

    Returns
    -------
    list
        All48 training-arm/evaluation-source/fit/positive-mode means over64 geometry slots.
    """
    return [
        {
            "training_arm": arm,
            "source": source,
            "fit_id": fit,
            "variant": variant,
            "points": {
                metric: float(
                    np.mean(
                        [indexed[arm, source, slot, fit]["positives"][variant][metric] for slot in range(64)]
                    )
                )
                for metric in METRICS
            },
        }
        for arm in TRAINING_ARMS
        for source in transfer.SOURCES
        for fit in transfer.FITS
        for variant in transfer.POSITIVES
    ]


def decision(statistics: dict[str, Any]) -> dict[str, bool]:
    """Apply only the fourteen prospectively declared advancement and preservation checks.

    Parameters
    ----------
    statistics : dict
        All fixed paired endpoint means and shared-slot conditional intervals.

    Returns
    -------
    dict
        Fourteen individual Boolean checks; no exploratory endpoint can replace one.
    """
    primary = statistics["mixed_minus_generated__real__biphasic__pooled__edited_ap"]
    gates = {
        "primary_AP_gain_at_least_0p01": primary["mean"] >= 0.01,
        "primary_AP_gain_lower_positive": primary["ci_low"] > 0,
    }
    for metric in ("response_ap", "amplitude_gain", "donor_gain"):
        row = statistics[f"mixed__real__biphasic__pooled__{metric}"]
        gates[f"mixed_{metric}_point_positive"] = row["mean"] > 0
        gates[f"mixed_{metric}_lower_positive"] = row["ci_low"] > 0
    for mode in ("focal", "repeated"):
        gates[f"real_biphasic_{mode}_noninferiority"] = (
            statistics[f"mixed_minus_generated__real__biphasic__{mode}__edited_ap"]["ci_low"] > -0.01
        )
    for family in ("hann", "biphasic"):
        gates[f"generated_{family}_noninferiority"] = (
            statistics[f"mixed_minus_generated__generated__{family}__pooled__edited_ap"]["ci_low"] > -0.01
        )
    for control in ("sham", "nuisance"):
        gates[f"real_{control}_top600_preservation"] = (
            statistics[f"mixed_minus_generated__real__{control}__top600_mean"]["ci_high"] < 0.02
        )
    return gates


def summarize(rows: list[dict[str, Any]], config: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    """Use one shared whole-slot bootstrap while retaining all three paired fits without resampling.

    Parameters
    ----------
    rows : list
        All768 fixed fitted-state/evaluation-background endpoint rows.
    config : dict
        Frozen2000-draw bootstrap settings with seed2026106701.

    Returns
    -------
    tuple
        Complete output-derived statistics/gates and all ordered bootstrap vectors.
    """
    indexed = indexed_rows(rows)
    vectors = positive_vectors(indexed)
    vectors.update(control_vectors(indexed))
    for key in list(vectors):
        if key.startswith("mixed__"):
            suffix = key[len("mixed__") :]
            vectors["mixed_minus_generated__" + suffix] = vectors[key] - vectors["generated__" + suffix]
    names = list(vectors)
    matrix = np.stack([vectors[name] for name in names], axis=1)
    if not np.isfinite(matrix).all():
        raise ValueError("All fixed conditional endpoints must be finite")
    rng = np.random.default_rng(config["bootstrap"]["seed"])
    groups = patient_groups(np.arange(64))
    draws = np.stack(
        [matrix[patient_resample(groups, rng)].mean(axis=0) for _ in range(config["bootstrap"]["draws"])]
    )
    statistics = {
        name: {
            "mean": float(matrix[:, pos].mean()),
            "ci_low": float(np.quantile(draws[:, pos], 0.025, method="linear")),
            "ci_high": float(np.quantile(draws[:, pos], 0.975, method="linear")),
        }
        for pos, name in enumerate(names)
    }
    gates = decision(statistics)
    return {
        "endpoints": statistics,
        "endpoint_order": names,
        "per_fit_mode": per_fit_points(indexed),
        "primary": statistics["mixed_minus_generated__real__biphasic__pooled__edited_ap"],
        "scientific_gates": gates,
        "scientific_passed": all(gates.values()),
        "clinical_claim": False,
        "bootstrap": {
            "draws": len(draws),
            "seed": config["bootstrap"]["seed"],
            "fits_resampled": False,
            "unit": "shared64geometry_slots_fixed_three_paired_fits",
            "skipped_draws": 0,
        },
    }, draws
