"""Recover frozen saved-row aggregation while retaining float64 AP roundoff unchanged."""

from typing import Any

import numpy as np

from ecg_experiment.intervals import patient_groups, patient_resample
from ecg_experiment.ssl_synthetic_lead_time_diagnostic import ARMS, ENDPOINTS, MODES

AP_TOLERANCE = 8 * np.finfo(np.float64).eps


def validate_rows(rows: list[dict[str, Any]]) -> None:
    """Admit only the complete frozen row grid and bounded machine-roundoff AP values.

    Parameters
    ----------
    rows : list
        Original1536 saved rows, without rounding, clamping, dropping or recomputing an endpoint.
    """
    expected = [
        (fit, arm, index, mode)
        for fit in [1, 2, 3]
        for arm in ARMS
        for index in range(2000, 2064)
        for mode in MODES
    ]
    actual = [(row["fit_id"], row["arm"], row["index"], row["mode"]) for row in rows]
    if actual != expected:
        raise ValueError("Recovery requires the complete original fixed row ordering")
    for row in rows:
        if row["seed"] != 2026103000 + row["index"]:
            raise ValueError("Recovery generator seed identity differs")
        leads = row["active_leads"]
        values = np.array(row["active_lead_AP"], dtype=np.float64)
        if (
            not 0 < len(leads) < 12
            or leads != sorted(set(leads))
            or any(not isinstance(lead, int) or not 0 <= lead < 12 for lead in leads)
            or values.shape != (len(leads),)
        ):
            raise ValueError("Recovery active lead identities or per-lead AP geometry differs")
        endpoints = np.array([row[name] for name in ENDPOINTS], dtype=np.float64)
        for admitted in [values, endpoints]:
            if (
                not np.isfinite(admitted).all()
                or not ((admitted >= -AP_TOLERANCE) & (admitted <= 1 + AP_TOLERANCE)).all()
            ):
                raise ValueError("Saved AP exceeds the prospective eight-epsilon admission envelope")
        if float(values.mean()) != row["active_lead_temporal_AP"]:
            raise ValueError("Saved equal-active-lead temporal AP mean differs")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate unchanged raw rows using the original six contrasts and crossed draws.

    Parameters
    ----------
    rows : list
        Complete saved-row grid admitted under the prospective eight-epsilon envelope.

    Returns
    -------
    dict
        The same six descriptive intervals, all arm/fit means and paired-fit ranges as version1.
    """
    validate_rows(rows)
    catalog = {(row["fit_id"], row["arm"], row["index"], row["mode"]): row for row in rows}
    arrays, means = {}, {str(fit): {arm: {} for arm in ARMS} for fit in [1, 2, 3]}
    for arm in ARMS:
        for endpoint in ENDPOINTS:
            for mode in MODES:
                arrays[(arm, endpoint, mode)] = np.array(
                    [
                        [catalog[(fit, arm, index, mode)][endpoint] for index in range(2000, 2064)]
                        for fit in [1, 2, 3]
                    ],
                    dtype=np.float64,
                )
            arrays[(arm, endpoint, "pooled")] = (
                arrays[(arm, endpoint, "focal")] + arrays[(arm, endpoint, "repeated")]
            ) / 2
        for fit in [1, 2, 3]:
            means[str(fit)][arm] = {
                mode: {
                    endpoint: float(arrays[(arm, endpoint, mode)][fit - 1].mean()) for endpoint in ENDPOINTS
                }
                for mode in [*MODES, "pooled"]
            }
    vectors = {
        f"PK_minus_B__{endpoint}__{mode}": arrays[("PK", endpoint, mode)] - arrays[("B", endpoint, mode)]
        for endpoint in ENDPOINTS
        for mode in [*MODES, "pooled"]
    }
    rng = np.random.default_rng(2026105601)
    groups = patient_groups(np.arange(64))
    samples = {key: np.empty(2000, dtype=np.float64) for key in vectors}
    for draw in range(2000):
        fits = rng.integers(0, 3, 3)
        backgrounds = patient_resample(groups, rng)
        for key, value in vectors.items():
            samples[key][draw] = value[fits[:, None], backgrounds[None, :]].mean()
    contrasts = {
        key: {
            "mean": float(value.mean()),
            "ci_low": float(np.quantile(samples[key], 0.025, method="linear")),
            "ci_high": float(np.quantile(samples[key], 0.975, method="linear")),
            "draws": 2000,
            "seed": 2026105601,
            "skipped_draws": 0,
            "fit_count": 3,
            "background_count": 64,
        }
        for key, value in vectors.items()
    }
    ranges = {
        key: {
            "fit_means": value.mean(axis=1).tolist(),
            "minimum": float(value.mean(axis=1).min()),
            "maximum": float(value.mean(axis=1).max()),
        }
        for key, value in vectors.items()
    }
    return {"descriptive_contrasts": contrasts, "arm_fit_means": means, "fit_contrast_ranges": ranges}
