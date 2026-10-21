"""Independent activity-mask and cohort arithmetic for the conditional LUDB coherence bridge."""

from typing import Any

import numpy as np

from ecg_experiment.intervals import patient_groups, patient_resample

VARIABLES = ("native_jaccard", "shifted_jaccard", "difference")


def standardize(raw: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Independently reproduce the admitted float32 eight-lead robust representation."""
    values = np.asarray(raw)
    if values.shape != (8, 5000) or values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError("Expected finite float32 native eight-lead geometry")
    physical = values.astype(np.float64)
    medians = np.median(physical, axis=1)
    low, high = np.quantile(physical, [.05, .95], axis=1, method="linear")
    ranges = high - low
    if np.any(ranges <= 1e-6) or not np.isfinite(ranges).all():
        raise ValueError("At least one lead fails frozen robust-range floor")
    normalized = ((physical - medians[:, None]) / ranges[:, None]).astype(np.float32)
    return normalized, {"median": medians.tolist(), "q95_minus_q05": ranges.tolist()}


def activity_masks(standardized: np.ndarray) -> np.ndarray:
    """Form strict upper-decile full-record derivative-energy masks without normalization."""
    values = np.asarray(standardized, dtype=np.float64)
    if values.shape != (8, 5000) or not np.isfinite(values).all():
        raise ValueError("Invalid standardized bridge geometry")
    differences = np.zeros_like(values)
    differences[:, 1:] = values[:, 1:] - values[:, :-1]
    envelope = np.asarray([np.convolve(row * row, np.full(20, .05), mode="same")
                           for row in differences])
    thresholds = np.quantile(envelope, .9, axis=1, method="linear")
    return envelope > thresholds[:, None]


def pair_jaccards(masks: np.ndarray) -> list[float]:
    """Compute every lower-triangle pair by vectorized set cardinalities, retaining frozen order."""
    values = np.asarray(masks)
    if values.shape != (8, 5000) or values.dtype != bool:
        raise ValueError("Expected eight Boolean full-record masks")
    first, second = np.tril_indices(8, k=-1)
    intersections = np.count_nonzero(values[first] & values[second], axis=1)
    unions = np.count_nonzero(values[first] | values[second], axis=1)
    scores = np.zeros(28, dtype=np.float64)
    np.divide(intersections, unions, out=scores, where=unions > 0)
    return scores.tolist()


def prepare(raw: np.ndarray, index: int, cohort: str) -> tuple[dict[str, Any], np.ndarray]:
    """Independently reconstruct one original or public-real bridge row and every activity cell."""
    if cohort not in ("generated", "ludb") or not 0 <= index < (128 if cohort == "generated" else 200):
        raise ValueError("Invalid fixed bridge cohort identity")
    standardized, transformation = standardize(raw)
    masks = activity_masks(standardized)
    seed = (2026103200 if cohort == "generated" else 2026106300) + index
    offsets = np.random.default_rng(seed).integers(1, 5000, size=8)
    controls = np.asarray([np.roll(mask, offset) for mask, offset in zip(masks, offsets, strict=True)])
    native, shifted = pair_jaccards(masks), pair_jaccards(controls)
    native_mean, shifted_mean = float(np.mean(native)), float(np.mean(shifted))
    coherence = {"native_jaccard": native_mean, "shifted_jaccard": shifted_mean,
                 "difference": native_mean - shifted_mean, "shifts": offsets.tolist()}
    row = {"cohort": cohort, "index": index, "record_id": index + 1 if cohort == "ludb" else None,
           "generator_seed": 2026103000 + index if cohort == "generated" else None,
           "control_seed": seed, "standardization": transformation, "coherence": coherence,
           "native_pair_jaccards": native, "shifted_pair_jaccards": shifted,
           "activity_occupancy": masks.mean(axis=1).tolist(),
           "empty_activity_masks": int(np.count_nonzero(~masks.any(axis=1)))}
    epsilon = 8 * np.finfo(np.float64).eps
    if not all(np.isfinite(value) and -epsilon <= value <= 1 + epsilon for value in [*native, *shifted]):
        raise ValueError("Jaccard outside frozen roundoff range")
    return row, masks


def distributions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Independently reconstruct every prespecified cohort statistic and fixed-bin count."""
    result = {}
    for name in VARIABLES:
        values = np.array([row["coherence"][name] for row in rows], dtype=np.float64)
        quantiles = np.percentile(values, [0, 25, 50, 75, 100], method="linear")
        edges = np.linspace(-1, 1, 21) if name == "difference" else np.linspace(0, 1, 21)
        counts = np.histogram(values, bins=edges)[0]
        result[name] = {"mean": float(values.mean()), "std_ddof1": float(values.std(ddof=1)),
                        **dict(zip(("min", "q25", "median", "q75", "max"),
                                   quantiles.tolist(), strict=True)),
                        "histogram_edges": edges.tolist(), "histogram_counts": counts.tolist(),
                        "below_first_edge": int(np.count_nonzero(values < edges[0])),
                        "above_last_edge": int(np.count_nonzero(values > edges[-1]))}
    return {"records": len(rows), "variables": result,
            "mean_activity_occupancy": np.asarray([row["activity_occupancy"] for row in rows]
                                                ).mean(0).tolist(),
            "empty_activity_masks": sum(row["empty_activity_masks"] for row in rows)}


def summarize(generated: list[dict[str, Any]],
              real: list[dict[str, Any]]) -> tuple[dict[str, Any], np.ndarray]:
    """Reconstruct the fixed independent two-cohort bootstrap with real draws consumed first."""
    for rows, cohort, count in ((generated, "generated", 128), (real, "ludb", 200)):
        if ([row["index"] for row in rows] != list(range(count))
                or any(row["cohort"] != cohort for row in rows)):
            raise ValueError("Require complete sorted cohorts without omissions")
    real_values = np.asarray([row["coherence"]["difference"] for row in real])
    generated_values = np.asarray([row["coherence"]["difference"] for row in generated])
    real_groups = patient_groups(np.arange(200))
    generated_groups = patient_groups(np.arange(128))
    rng = np.random.default_rng(2026106401)
    draws = np.empty((2000, 3), dtype=np.float64)
    for draw in range(2000):
        real_rows = patient_resample(real_groups, rng)
        generated_rows = patient_resample(generated_groups, rng)
        real_mean = float(real_values[real_rows].mean())
        generated_mean = float(generated_values[generated_rows].mean())
        draws[draw] = real_mean, generated_mean, real_mean - generated_mean
    low, high = np.percentile(draws[:, 2], [2.5, 97.5], method="linear")
    result = {"generated": distributions(generated), "ludb": distributions(real),
              "primary_descriptive": {"contrast": "real_minus_generated_mean_native_minus_shifted_jaccard",
                                      "mean": float(real_values.mean() - generated_values.mean()),
                                      "ci_low": float(low), "ci_high": float(high), "draws": 2000,
                                      "seed": 2026106401, "skipped_draws": 0,
                                      "unit": "independent_whole_LUDB_record_and_generator_seed"},
              "clinical_quality_claim": False}
    return result, draws


def old_primary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reconstruct the old128-seed paired interval exactly before the new cohort comparison."""
    values = np.asarray([row["coherence"]["difference"] for row in rows])
    groups = patient_groups(np.arange(128))
    rng = np.random.default_rng(2026103301)
    draws = [float(values[patient_resample(groups, rng)].mean()) for _ in range(2000)]
    low, high = np.percentile(draws, [2.5, 97.5], method="linear")
    return {"mean_difference": float(values.mean()), "ci_low": float(low), "ci_high": float(high),
            "draws": 2000, "seed": 2026103301, "skipped_draws": 0, "unit": "generator_seed"}
