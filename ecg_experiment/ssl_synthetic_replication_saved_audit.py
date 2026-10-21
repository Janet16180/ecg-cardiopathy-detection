"""Independent saved-output arithmetic for matched localization fit replication."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from ecg_experiment.ssl_synthetic_pretrained_saved_audit import require
from ecg_experiment.ssl_synthetic_quality_saved_audit import average_precision


def initialization(seed: int) -> dict[str, torch.Tensor]:
    """Regenerate one declared CPU initialization without any model forward.

    Parameters
    ----------
    seed : int
        Frozen fit initialization seed.

    Returns
    -------
    dict[str, torch.Tensor]
        Six native parameter tensors in registration order.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = nn.Sequential(
            nn.Conv1d(12, 32, 25, padding=12),
            nn.ReLU(),
            nn.Conv1d(32, 32, 25, dilation=4, padding=48),
            nn.ReLU(),
            nn.Conv1d(32, 12, 1),
        )
    return model.state_dict()


def schedule_for_seed(seed: int) -> np.ndarray:
    """Independently reproduce ten complete TRAIN permutations expanded to matched batches.

    Parameters
    ----------
    seed : int
        Frozen fit schedule seed.

    Returns
    -------
    np.ndarray
        Ordered shape (400,8) integer training-case schedule.
    """
    rng = np.random.default_rng(seed)
    batches = []
    for _ in range(10):
        order = rng.permutation(80)
        for offset in range(0, 80, 2):
            batches.append(
                [int(4 * index + mode) for index in order[offset : offset + 2] for mode in range(4)]
            )
    return np.array(batches, dtype=np.int64)


def panel_metrics(
    probability: np.ndarray, observed: np.ndarray, support: np.ndarray, prior: np.ndarray, indices: list[int]
) -> list[dict[str, Any]]:
    """Recompute all full-grid case metrics for a declared spent or fresh panel.

    Parameters
    ----------
    probability, observed, support : np.ndarray
        Saved probability maps, observed signals and exact thresholded target masks.
    prior : np.ndarray
        Unchanged training-only spatial prior.
    indices : list[int]
        Frozen editor/background indices identifying the panel.

    Returns
    -------
    list[dict[str, Any]]
        Complete independent per-seed AP, response, selectivity and prior metrics.
    """
    expected = (len(indices), 4, 12, 5000)
    require(probability.shape == expected and probability.dtype == np.float32, "panel probability geometry")
    require(
        observed.shape == support.shape == expected
        and observed.dtype == np.float32
        and support.dtype == bool,
        "panel observed/target geometry",
    )
    require(
        bool(np.isfinite(probability).all()) and bool(((probability >= 0) & (probability <= 1)).all()),
        "finite native panel probability range",
    )
    rows = []
    for offset, index in enumerate(indices):
        row = {"index": index, "seed": 2026103000 + index}
        for position, mode in ((2, "focal"), (3, "repeated")):
            target = support[offset, position]
            require(bool(target.any()), "nonempty panel positive target")
            direct = average_precision(target, probability[offset, position])
            nuisance = average_precision(target, probability[offset, 1])
            delta = probability[offset, position].astype(np.float64) - probability[offset, 1].astype(
                np.float64
            )
            inside = float(delta[target].mean())
            outside = float(delta[~target].mean())
            row[mode] = {
                "edited_ap": direct,
                "nuisance_ap": nuisance,
                "response_ap": direct - nuisance,
                "delta_inside": inside,
                "delta_outside": outside,
                "selectivity": inside - outside,
                "amplitude_ap": average_precision(target, np.abs(observed[offset, position])),
                "prevalence": float(target.mean()),
                "spatial_prior_ap": average_precision(target, prior),
            }
        row["mean_positive_ap"] = (row["focal"]["edited_ap"] + row["repeated"]["edited_ap"]) / 2
        rows.append(row)
    return rows


def panel_controls(probability: np.ndarray) -> dict[str, Any]:
    """Recompute descriptive strict-cutoff control summaries for any frozen panel size.

    Parameters
    ----------
    probability : np.ndarray
        Four variants per declared evaluation seed.

    Returns
    -------
    dict[str, Any]
        Complete means, maxima and record-level descriptive flags.
    """
    result = {}
    for position, mode in ((0, "sham"), (1, "nuisance")):
        values = probability[:, position].reshape(len(probability), -1).astype(np.float64)
        record_max = values.max(axis=1)
        result[mode] = {
            "mean_probability": float(values.mean()),
            "maximum_probability": float(values.max()),
            "record_maximum_gt_0p5_fraction": float(np.mean(record_max > 0.5)),
            "record_mean_probabilities": values.mean(axis=1).tolist(),
            "record_maximum_probabilities": record_max.tolist(),
        }
    return result


def contrast_matrices(fits: list[dict[str, list[dict[str, Any]]]]) -> dict[str, np.ndarray]:
    """Build paired fit-by-background arrays without production score helpers.

    Parameters
    ----------
    fits : list
        Matched arms with equally ordered background rows.

    Returns
    -------
    dict
        Ten full-grid endpoint contrast matrices.
    """
    require(bool(fits), "nonempty fit set")
    count = len(fits[0]["supervised"])
    require(all(len(fit[arm]) == count for fit in fits for arm in ("supervised", "permuted")), "paired rows")
    result = {}
    for comparator in ("absolute_amplitude", "permuted"):
        for mode in ("focal", "repeated"):
            supervised = np.array([[row[mode]["edited_ap"] for row in fit["supervised"]] for fit in fits])
            other = np.array(
                [
                    [
                        row[mode]["edited_ap" if comparator == "permuted" else "amplitude_ap"]
                        for row in fit["permuted" if comparator == "permuted" else "supervised"]
                    ]
                    for fit in fits
                ]
            )
            result[f"supervised_minus_{comparator}_{mode}"] = supervised - other
        result[f"supervised_minus_{comparator}_pooled"] = (
            result[f"supervised_minus_{comparator}_focal"] + result[f"supervised_minus_{comparator}_repeated"]
        ) / 2
    for mode in ("focal", "repeated"):
        for measure in ("response_ap", "selectivity"):
            result[f"supervised_{mode}_{measure}"] = np.array(
                [[row[mode][measure] for row in fit["supervised"]] for fit in fits], dtype=np.float64
            )
    return result


def resampled_summaries(vectors: dict[str, np.ndarray], crossed: bool) -> dict[str, Any]:
    """Independently apply prospective shared-column crossed or conditional draws.

    Parameters
    ----------
    vectors : dict
        Fit-by-background arrays for all paired contrasts.
    crossed : bool
        Sample three fit rows before the single common background draw.

    Returns
    -------
    dict
        Complete interval metadata with shared draws across contrasts.
    """
    from ecg_experiment.intervals import patient_groups, patient_resample

    shape = (3, 64) if crossed else (1, 64)
    require(
        all(value.shape == shape and np.isfinite(value).all() for value in vectors.values()),
        "interval geometry",
    )
    seed = 2026104501 if crossed else 2026104502
    rng = np.random.default_rng(seed)
    groups = patient_groups(np.arange(64))
    samples = {key: np.empty(2000, dtype=np.float64) for key in vectors}
    for draw in range(2000):
        rows = rng.integers(0, 3, 3) if crossed else np.array([0])
        columns = patient_resample(groups, rng)
        for key, values in vectors.items():
            samples[key][draw] = values[rows[:, None], columns[None, :]].mean()
    result = {}
    for key, values in vectors.items():
        low, high = np.quantile(samples[key], [0.025, 0.975], method="linear")
        result[key] = {
            "mean": float(values.mean()),
            "ci_low": float(low),
            "ci_high": float(high),
            "draws": 2000,
            "seed": seed,
            "skipped_draws": 0,
            "unit": "fit_and_generator_seed" if crossed else "generator_seed_conditional_on_fit",
            "fit_count": shape[0],
            "background_count": 64,
        }
    return result


def replication_summary(fits: list[dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    """Independently check the eighteen-gate conjunction and all fit ranges.

    Parameters
    ----------
    fits : list
        Exactly the three new matched pairs on the same sixty-four fresh backgrounds.

    Returns
    -------
    dict
        Crossed and conditional summaries and the prospective decision.
    """
    require(len(fits) == 3, "three new primary fits")
    vectors = contrast_matrices(fits)
    summaries = resampled_summaries(vectors, True)
    gates = {}
    for comparator in ("absolute_amplitude", "permuted"):
        stem = f"supervised_minus_{comparator}"
        for mode in ("focal", "repeated"):
            gates[f"{stem}_{mode}_guard"] = summaries[f"{stem}_{mode}"]["ci_low"] > -0.01
        key = f"{stem}_pooled"
        gates[key + "_mean"] = summaries[key]["mean"] >= 0.05
        gates[key + "_lower"] = summaries[key]["ci_low"] > 0
        for index in range(3):
            gates[f"fit_{index + 1}_{key}_mean_positive"] = float(vectors[key][index].mean()) > 0
    for mode in ("focal", "repeated"):
        for measure in ("response_ap", "selectivity"):
            key = f"supervised_{mode}_{measure}"
            gates[key + "_lower"] = summaries[key]["ci_low"] > 0
    require(len(gates) == 18, "all eighteen scientific gates")
    ranges = {}
    for key, values in vectors.items():
        means = values.mean(axis=1)
        ranges[key] = {
            "fit_means": means.tolist(),
            "minimum": float(means.min()),
            "maximum": float(means.max()),
        }
    return {
        "contrasts": summaries,
        "per_fit_contrasts": [
            resampled_summaries({key: values[row : row + 1] for key, values in vectors.items()}, False)
            for row in range(3)
        ],
        "fit_mean_ranges": ranges,
        "scientific_gates": gates,
        "scientific_passed": all(gates.values()),
        "primary_fit_count": 3,
        "primary_background_count": 64,
        "uncertainty_scope": "crossed three-fit/background resampling; no patient or clinical uncertainty",
    }


def hotspot_rows(probability: np.ndarray, support: np.ndarray, indices: list[int]) -> list[dict[str, Any]]:
    """Calculate secondary top-six-hundred overlap measures with independent tie sorting.

    Parameters
    ----------
    probability, support : np.ndarray
        Full-grid maps and separate target masks.
    indices : list
        Background identities in saved row order.

    Returns
    -------
    list
        Per-seed persistence and positive support measurements.
    """
    from ecg_experiment.ssl_synthetic_control_diagnosis_audit import top_selection

    result = []
    for offset, index in enumerate(indices):
        masks = [top_selection(probability[offset, mode])[0] for mode in range(4)]
        row: dict[str, Any] = {
            "index": index,
            "seed": 2026103000 + index,
            "sham_nuisance_persistence": float(np.count_nonzero(masks[0] & masks[1]) / 600),
        }
        for position, mode in ((2, "focal"), (3, "repeated")):
            target = support[offset, position]
            intersection = np.count_nonzero(masks[position] & target)
            precision = float(intersection / 600)
            nuisance_precision = float(np.count_nonzero(masks[1] & target) / 600)
            row[mode] = {
                "nuisance_hotspot_persistence": float(np.count_nonzero(masks[position] & masks[1]) / 600),
                "support_precision": precision,
                "support_recall": float(intersection / np.count_nonzero(target)),
                "matched_nuisance_support_precision": nuisance_precision,
                "precision_response": precision - nuisance_precision,
            }
        result.append(row)
    return result
