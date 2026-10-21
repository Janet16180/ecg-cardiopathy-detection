"""Describe fixed cross-lead activity synchrony without a physiological-quality threshold."""

from typing import Any

import numpy as np

from ecg_experiment import ssl_synthetic_generated_quality as quality
from ecg_experiment.intervals import patient_groups, patient_resample

INDEPENDENT_INDICES = (0, 1, 6, 7, 8, 9, 10, 11)
VARIABLES = ("native_jaccard", "shifted_jaccard", "difference")


def pair_jaccards(masks: np.ndarray) -> list[float]:
    """Retain all28 unordered-pair values in the frozen mean-Jaccard loop order.

    Parameters
    ----------
    masks : np.ndarray
        Eight Boolean full-record activity masks.

    Returns
    -------
    list[float]
        Every pair's intersection-over-union; empty unions contribute zero.
    """
    pairs = []
    for first in range(8):
        for second in range(first):
            union = np.count_nonzero(masks[first] | masks[second])
            pairs.append(float(np.count_nonzero(masks[first] & masks[second]) / union) if union else 0.)
    return pairs


def prepare(raw: np.ndarray, index: int, cohort: str) -> tuple[dict[str, Any], np.ndarray]:
    """Apply identical frozen representation and coherence code to one admitted record.

    Parameters
    ----------
    raw : np.ndarray
        Exactly8x5000 finite native or physical-cast float32 samples.
    index : int
        Zero-based generator-bank position or LUDB numeric ID minus one.
    cohort : str
        Either generated or ludb; selects a prospectively fixed control seed.

    Returns
    -------
    tuple
        Complete row, including allpair values and transformations, and Boolean activity masks.
    """
    if cohort not in ["generated", "ludb"] or not 0 <= index < (128 if cohort == "generated" else 200):
        raise ValueError("Invalid fixed cohort identity")
    standardized, parameters = quality.standardize(raw)
    masks = quality.activity_masks(standardized)
    seed = (2026103200 if cohort == "generated" else 2026106300) + index
    coherence = quality.coherence(standardized, seed - 2026103200)
    shifted = np.stack([np.roll(mask, shift) for mask, shift in zip(masks, coherence["shifts"], strict=True)])
    native_pairs, shifted_pairs = pair_jaccards(masks), pair_jaccards(shifted)
    if (float(np.mean(native_pairs)) != coherence["native_jaccard"]
            or float(np.mean(shifted_pairs)) != coherence["shifted_jaccard"]):
        raise ValueError("Saved pair arithmetic must match the frozen coherence reduction exactly")
    tolerance = 8 * np.finfo(np.float64).eps
    values = np.asarray([*native_pairs, *shifted_pairs, coherence["native_jaccard"],
                         coherence["shifted_jaccard"]])
    if (not np.isfinite(values).all() or not ((values >= -tolerance) & (values <= 1 + tolerance)).all()
            or not np.isfinite(coherence["difference"]) or abs(coherence["difference"]) > 1 + tolerance):
        raise ValueError("Coherence values exceed frozen finite eight-epsilon admission")
    row = {"cohort": cohort, "index": index, "record_id": index + 1 if cohort == "ludb" else None,
           "generator_seed": 2026103000 + index if cohort == "generated" else None,
           "control_seed": seed, "standardization": parameters, "coherence": coherence,
           "native_pair_jaccards": native_pairs, "shifted_pair_jaccards": shifted_pairs,
           "activity_occupancy": masks.mean(axis=1).tolist(),
           "empty_activity_masks": int(np.count_nonzero(~masks.any(axis=1)))}
    return row, masks


def distributions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe every prespecified variable and fixed-bin distribution without hypothesis tests."""
    result = {}
    for name in VARIABLES:
        values = np.asarray([row["coherence"][name] for row in rows])
        quantiles = np.percentile(values, [0, 25, 50, 75, 100], method="linear")
        edges = np.linspace(-1, 1, 21) if name == "difference" else np.linspace(0, 1, 21)
        counts, _ = np.histogram(values, bins=edges)
        result[name] = {"mean": float(values.mean()), "std_ddof1": float(values.std(ddof=1)),
                        **dict(zip(["min", "q25", "median", "q75", "max"], quantiles.tolist(), strict=True)),
                        "histogram_edges": edges.tolist(), "histogram_counts": counts.tolist(),
                        "below_first_edge": int(np.count_nonzero(values < edges[0])),
                        "above_last_edge": int(np.count_nonzero(values > edges[-1]))}
    return {"records": len(rows), "variables": result,
            "mean_activity_occupancy": (
                np.asarray([row["activity_occupancy"] for row in rows]).mean(0).tolist()),
            "empty_activity_masks": sum(row["empty_activity_masks"] for row in rows)}


def summarize(generated: list[dict[str, Any]],
              real: list[dict[str, Any]]) -> tuple[dict[str, Any], np.ndarray]:
    """Bootstrap the descriptive real-minus-generated contrast independently by whole records.

    Parameters
    ----------
    generated, real : list[dict]
        Complete sorted128-generator and200-LUDB cohorts with no omissions.

    Returns
    -------
    tuple
        Every cohort distribution and descriptive interval, plus2000x3 cohort means and contrasts.
    """
    for rows, cohort, count in [(generated, "generated", 128), (real, "ludb", 200)]:
        if ([row["index"] for row in rows] != list(range(count))
                or any(row["cohort"] != cohort for row in rows)):
            raise ValueError("Require complete sorted fixed cohort indices")
    first = np.asarray([row["coherence"]["difference"] for row in real])
    second = np.asarray([row["coherence"]["difference"] for row in generated])
    real_groups, generated_groups = patient_groups(np.arange(200)), patient_groups(np.arange(128))
    rng, bootstrap = np.random.default_rng(2026106401), np.empty((2000, 3), dtype=np.float64)
    for index in range(2000):
        real_rows = patient_resample(real_groups, rng)
        generated_rows = patient_resample(generated_groups, rng)
        first_mean, second_mean = float(first[real_rows].mean()), float(second[generated_rows].mean())
        bootstrap[index] = first_mean, second_mean, first_mean - second_mean
    low, high = np.percentile(bootstrap[:, 2], [2.5, 97.5], method="linear")
    return {"generated": distributions(generated), "ludb": distributions(real),
            "primary_descriptive": {"contrast": "real_minus_generated_mean_native_minus_shifted_jaccard",
                                    "mean": float(first.mean() - second.mean()),
                                    "ci_low": float(low), "ci_high": float(high), "draws": 2000,
                                    "seed": 2026106401, "skipped_draws": 0,
                                    "unit": "independent_whole_LUDB_record_and_generator_seed"},
            "clinical_quality_claim": False}, bootstrap
