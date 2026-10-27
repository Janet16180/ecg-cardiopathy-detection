"""Independent saved-map lead-presence and true-lead-conditional temporal ranking."""

from typing import Any

import numpy as np

from ecg_experiment.intervals import patient_groups, patient_resample
from ecg_experiment.ssl_synthetic_pretrained_saved_audit import require
from ecg_experiment.ssl_synthetic_quality_saved_audit import average_precision

ARMS = ("B", "P", "K", "PK")
ENDPOINTS = ("active_lead_temporal_AP", "lead_presence_AP")


def case_endpoints(probability: np.ndarray, support: np.ndarray) -> dict[str, Any]:
    """Independently separate true-lead temporal ranking from twelve-lead selection.

    Parameters
    ----------
    probability, support : np.ndarray
        Saved native positive-variant probabilities and exact thresholded support.

    Returns
    -------
    dict
        Every active-lead AP, their equal mean and full twelve-lead presence AP.
    """
    require(probability.shape == support.shape == (12, 5000), "lead/time grid geometry")
    require(probability.dtype == np.float32 and support.dtype == bool, "lead/time native dtypes")
    require(
        np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all(),
        "finite native probability range",
    )
    active = np.flatnonzero(np.any(support, axis=1))
    require(0 < len(active) < 12, "both lead-presence classes")
    values = []
    for lead in active:
        target = support[lead]
        require(0 < np.count_nonzero(target) < 5000, "both within-lead temporal classes")
        values.append(average_precision(target, probability[lead]))
    return {
        "active_leads": active.tolist(),
        "active_lead_AP": values,
        "active_lead_temporal_AP": float(np.mean(values)),
        "lead_presence_AP": average_precision(np.any(support, axis=1), np.max(probability, axis=1)),
    }


def endpoint_rows(
    probability: np.ndarray, support: np.ndarray, fit_id: int, arm: str
) -> list[dict[str, Any]]:
    """Compute the complete fixed-order positive-mode rows from one saved model bank.

    Parameters
    ----------
    probability, support : np.ndarray
        Sixty-four backgrounds with four variants in native geometry.
    fit_id, arm : identifiers
        Declared fit and objective, used only for receipt identities.

    Returns
    -------
    list
        One hundred twenty-eight rows, background-major and focal then repeated.
    """
    require(probability.shape == support.shape == (64, 4, 12, 5000), "full diagnostic panel geometry")
    require(fit_id in (1, 2, 3) and arm in ARMS, "fixed fit/arm identity")
    return [
        {
            "fit_id": fit_id,
            "arm": arm,
            "index": 2000 + offset,
            "seed": 2026105000 + offset,
            "mode": mode,
            **case_endpoints(probability[offset, position], support[offset, position]),
        }
        for offset in range(64)
        for mode, position in (("focal", 2), ("repeated", 3))
    ]


def contrast_matrices(rows: list[dict[str, Any]]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Collect equal-lead case values into the six fixed paired contrasts and all fit means.

    Parameters
    ----------
    rows : list
        All1536 independently reconstructed endpoint records.

    Returns
    -------
    tuple
        Six three-fit by sixty-four-background matrices and every arm/fit mean.
    """
    require(len(rows) == 1536, "complete twelve-model two-mode row count")
    catalog = {(row["fit_id"], row["arm"], row["index"], row["mode"]): row for row in rows}
    require(len(catalog) == len(rows), "no duplicate diagnostic identity")
    arrays, means = {}, {}
    for fit in (1, 2, 3):
        means[str(fit)] = {}
        for arm in ARMS:
            means[str(fit)][arm] = {}
            for mode in ("focal", "repeated"):
                values = [catalog[(fit, arm, index, mode)] for index in range(2000, 2064)]
                require(
                    all(row["seed"] == 2026105000 + offset for offset, row in enumerate(values)),
                    "ordered diagnostic seeds",
                )
                means[str(fit)][arm][mode] = {}
                for endpoint in ENDPOINTS:
                    array = np.array([row[endpoint] for row in values], dtype=np.float64)
                    require(np.isfinite(array).all(), "finite endpoint values")
                    arrays[(fit, arm, mode, endpoint)] = array
                    means[str(fit)][arm][mode][endpoint] = float(array.mean())
            means[str(fit)][arm]["pooled"] = {}
            for endpoint in ENDPOINTS:
                arrays[(fit, arm, "pooled", endpoint)] = (
                    arrays[(fit, arm, "focal", endpoint)] + arrays[(fit, arm, "repeated", endpoint)]
                ) / 2
                means[str(fit)][arm]["pooled"][endpoint] = float(
                    arrays[(fit, arm, "pooled", endpoint)].mean()
                )
    contrasts = {}
    for endpoint in ENDPOINTS:
        for mode in ("focal", "repeated", "pooled"):
            contrasts[f"PK_minus_B__{endpoint}__{mode}"] = np.array(
                [
                    arrays[(fit, "PK", mode, endpoint)] - arrays[(fit, "B", mode, endpoint)]
                    for fit in (1, 2, 3)
                ]
            )
    return contrasts, means


def crossed_intervals(vectors: dict[str, np.ndarray]) -> dict[str, Any]:
    """Use one common fit-then-background draw for all six descriptive endpoints.

    Parameters
    ----------
    vectors : dict
        Exactly six paired arrays of shape(3,64).

    Returns
    -------
    dict
        Shared-draw percentile summaries with no outcome-based success label.
    """
    require(
        len(vectors) == 6
        and all(value.shape == (3, 64) and np.isfinite(value).all() for value in vectors.values()),
        "six crossed endpoint geometries",
    )
    rng = np.random.default_rng(2026105601)
    groups = patient_groups(np.arange(64))
    samples = {key: np.empty(2000) for key in vectors}
    for draw in range(2000):
        fit_rows = rng.integers(0, 3, 3)
        background_columns = patient_resample(groups, rng)
        for key, values in vectors.items():
            samples[key][draw] = values[fit_rows[:, None], background_columns[None, :]].mean()
    result = {}
    for key, values in vectors.items():
        low, high = np.percentile(samples[key], [2.5, 97.5])
        result[key] = {
            "mean": float(values.mean()),
            "ci_low": float(low),
            "ci_high": float(high),
            "draws": 2000,
            "seed": 2026105601,
            "skipped_draws": 0,
            "fit_count": 3,
            "background_count": 64,
        }
    return result


def diagnostic_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize every fit and both ranking views without adding a scientific decision.

    Parameters
    ----------
    rows : list
        All declared fit/arm/background/positive-mode endpoint rows.

    Returns
    -------
    dict
        Six intervals, all arm/fit points and paired-fit ranges.
    """
    vectors, means = contrast_matrices(rows)
    ranges = {}
    for key, values in vectors.items():
        fit_means = values.mean(axis=1)
        ranges[key] = {
            "fit_means": fit_means.tolist(),
            "minimum": float(fit_means.min()),
            "maximum": float(fit_means.max()),
        }
    return {
        "descriptive_contrasts": crossed_intervals(vectors),
        "arm_fit_means": means,
        "fit_contrast_ranges": ranges,
    }
