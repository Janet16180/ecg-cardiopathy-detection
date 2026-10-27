"""Frozen-model input representation and prospective public-landmark diagnostics."""

from typing import Any

import numpy as np

from ecg_experiment import ssl_synthetic_generated_quality as quality

ARMS = ("B", "P", "K", "PK")
FITS = (1, 2, 3)
NATIVE_INDICES = (0, 1, 6, 7, 8, 9, 10, 11)
NATIVE_LEADS = ("i", "ii", "v1", "v2", "v3", "v4", "v5", "v6")


def prepare_input(physical: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the unchanged generated-training representation to the eight physical native leads.

    Parameters
    ----------
    physical : np.ndarray
        Finite canonical twelve-lead physical samples with shape (12, 5000).

    Returns
    -------
    tuple[np.ndarray, dict[str, Any]]
        Float32 observed-only model input and the original robust transformation declarations.
    """
    if physical.shape != (12, 5000) or not np.isfinite(physical).all():
        raise ValueError("Expected finite canonical twelve-lead5000-sample physical input")
    native = physical[list(NATIVE_INDICES)].astype(np.float32)
    standardized, transformation = quality.standardize(native)
    return quality.derive_leads(standardized), transformation


def association(probability: np.ndarray, energy: np.ndarray) -> dict[str, Any]:
    """Describe tied-rank association and strict-q90 overlap without replacing undefined correlation.

    Parameters
    ----------
    probability : np.ndarray
        Frozen probability values with the prospective full-record or cropped geometry.
    energy : np.ndarray
        Fixed squared-derivative activity values computed from the exact standardized native input.

    Returns
    -------
    dict[str, Any]
        Declared transformed values, identities or complete descriptive results.
    """
    from ecg_experiment.ssl_synthetic_ludb_peak_timing import midranks

    first, second = midranks(probability, probability), midranks(energy, energy)
    first -= first.mean()
    second -= second.mean()
    constant_map, constant_energy = bool(np.ptp(probability) == 0), bool(np.ptp(energy) == 0)
    reason = (
        "constant_map_and_energy"
        if constant_map and constant_energy
        else "constant_map"
        if constant_map
        else "constant_energy"
        if constant_energy
        else None
    )
    correlation = (
        None
        if reason
        else float(np.dot(first, second) / np.sqrt(np.dot(first, first) * np.dot(second, second)))
    )
    masks = [values > np.quantile(values, 0.9, method="linear") for values in [probability, energy]]
    union = int(np.count_nonzero(masks[0] | masks[1]))
    return {
        "spearman": correlation,
        "spearman_undefined_reason": reason,
        "jaccard": float(np.count_nonzero(masks[0] & masks[1]) / union) if union else 0.0,
        "empty_jaccard_union": union == 0,
    }


def score_lead(probability: np.ndarray, energy: np.ndarray, structural: dict[str, Any]) -> dict[str, Any]:
    """Score only explicit retained N timestamps and fixed within-span controls on a native lead.

    Parameters
    ----------
    probability : np.ndarray
        Frozen probability values with the prospective full-record or cropped geometry.
    energy : np.ndarray
        Fixed squared-derivative activity values computed from the exact standardized native input.
    structural : dict[str, Any]
        Saved original lead span and explicit N timestamps, including duplicate multiplicity.

    Returns
    -------
    dict[str, Any]
        Declared transformed values, identities or complete descriptive results.
    """
    from ecg_experiment.ssl_synthetic_ludb_peak_timing import SHIFTS, midranks

    start, end = structural["start"], structural["end_exclusive"]
    values, activity = probability[start:end].astype(np.float64), energy[start:end]
    if (
        values.shape != activity.shape
        or not len(values)
        or not np.isfinite(values).all()
        or not np.isfinite(activity).all()
        or np.any(values < 0)
        or np.any(values > 1)
    ):
        raise ValueError("Require finite probability and activity vectors within the retained span")
    positions = np.asarray(structural["N_samples"], dtype=np.int64) - start
    shifts = [int(np.floor(len(values) * fraction)) for fraction in SHIFTS]
    if (
        not len(positions)
        or positions.min() < 0
        or positions.max() >= len(values)
        or len(set(shifts)) != 6
        or not all(0 < shift < len(values) for shift in shifts)
    ):
        raise ValueError("Landmark multiplicity and fixed shift geometry must remain valid")
    ranks = midranks(values, values)
    native = ranks[positions]
    controls = [np.roll(ranks, shift)[positions] for shift in shifts]
    mean_controls = [float(value.mean()) for value in controls]
    associations = [
        association(values, activity),
        *[association(np.roll(values, shift), activity) for shift in shifts],
    ]
    defined = [row["spearman"] for row in associations[1:] if row["spearman"] is not None]
    return {
        "lead": structural["lead"],
        "start": start,
        "end_exclusive": end,
        "N_samples": structural["N_samples"],
        "shift_samples": shifts,
        "N_midrank_values": native.tolist(),
        "N_mean_midrank": float(native.mean()),
        "shift_N_midrank_values": [value.tolist() for value in controls],
        "shift_N_mean_midrank": mean_controls,
        "mean_shift_N_midrank": float(np.mean(mean_controls)),
        "gain": float(native.mean()) - float(np.mean(mean_controls)),
        "constant_map": bool(np.ptp(values) == 0),
        "probability_mean": float(values.mean()),
        "probability_q95": float(np.quantile(values, 0.95, method="linear")),
        "probability_max": float(values.max()),
        "energy_association": associations[0],
        "shift_energy_association": associations[1:],
        "mean_shift_energy_spearman": float(np.mean(defined)) if defined else None,
        "defined_shift_spearman_count": len(defined),
        "mean_shift_energy_jaccard": float(np.mean([row["jaccard"] for row in associations[1:]])),
    }


def score_record(
    probabilities: np.ndarray, energy: np.ndarray, structure: dict[str, Any], fit: int, arm: str
) -> dict[str, Any]:
    """Retain all eight native-lead rows and their fixed equal-weight model-record reduction.

    Parameters
    ----------
    probabilities : np.ndarray
        Complete float32 twelve-output probability maps.
    energy : np.ndarray
        Fixed squared-derivative activity values computed from the exact standardized native input.
    structure : dict[str, Any]
        Saved role and annotation rows; original annotation streams are not reopened.
    fit : int
        Prospectively fixed fit ID from one through three.
    arm : str
        One of the four fixed B, P, K and PK objective identities.

    Returns
    -------
    dict[str, Any]
        Declared transformed values, identities or complete descriptive results.
    """
    if probabilities.shape != (12, 5000) or energy.shape != (8, 5000):
        raise ValueError("Full twelve-output maps and eight-input energy arrays are required")
    leads = [
        score_lead(probabilities[index], energy[position], structure["leads"][index])
        for position, index in enumerate(NATIVE_INDICES)
    ]
    controls = np.asarray([row["shift_N_mean_midrank"] for row in leads]).mean(0)
    native = float(np.mean([row["N_mean_midrank"] for row in leads]))
    return {
        "record_id": structure["record_id"],
        "fit_id": fit,
        "arm": arm,
        "leads": leads,
        "N_mean_midrank": native,
        "shift_N_mean_midrank": controls.tolist(),
        "mean_shift_N_midrank": float(controls.mean()),
        "gain": native - float(controls.mean()),
    }


def defined_summary(values: list[float | None]) -> dict[str, Any]:
    """Preserve explicit denominators when summarizing potentially undefined correlations.

    Parameters
    ----------
    values : list[float | None]
        Values whose undefined entries remain explicit rather than imputed.

    Returns
    -------
    dict[str, Any]
        Declared transformed values, identities or complete descriptive results.
    """
    defined = [value for value in values if value is not None]
    return {
        "mean": float(np.mean(defined)) if defined else None,
        "defined_count": len(defined),
        "undefined_count": len(values) - len(defined),
        "total_count": len(values),
    }


def aggregate(rows: list[dict[str, Any]], config: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    """Use shared whole-record draws after equal native-lead and fixed-fit reductions.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Complete saved numerical rows with every required record, model and lead.
    config : dict[str, Any]
        Committed prospective scientific configuration and hard resource limits.

    Returns
    -------
    tuple[dict[str, Any], np.ndarray]
        Declared transformed values, identities or complete descriptive results.
    """
    from ecg_experiment.intervals import patient_groups, patient_resample

    identifiers = [identifier for identifier in range(1, 201) if identifier % 5]
    indexed = {(row["record_id"], row["fit_id"], row["arm"]): row for row in rows}
    if (
        len(indexed) != 1920
        or len(rows) != 1920
        or any(
            (identifier, fit, arm) not in indexed
            for identifier in identifiers
            for fit in FITS
            for arm in ARMS
        )
    ):
        raise ValueError("Require every160 record bythree fixed fits byfour arms exactly once")
    gains = np.asarray(
        [
            [[indexed[identifier, fit, arm]["gain"] for arm in ARMS] for fit in FITS]
            for identifier in identifiers
        ]
    ).mean(1)
    rng = np.random.default_rng(config["bootstrap"]["seed"])
    groups = patient_groups(np.arange(160))
    draws = np.asarray(
        [gains[patient_resample(groups, rng)].mean(0) for _ in range(config["bootstrap"]["draws"])]
    )
    contrasts = np.column_stack([draws, draws[:, 3] - draws[:, 0]])
    points = np.r_[gains.mean(0), (gains[:, 3] - gains[:, 0]).mean()]
    intervals = np.percentile(contrasts, [2.5, 97.5], axis=0, method="linear")
    estimates = {
        name: {
            "mean": float(points[index]),
            "ci_low": float(intervals[0, index]),
            "ci_high": float(intervals[1, index]),
        }
        for index, name in enumerate([*ARMS, "PK_minus_B"])
    }
    arm_summaries = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        leads = [lead for row in selected for lead in row["leads"]]
        arm_summaries[arm] = {
            "gain": estimates[arm],
            "N_mean_midrank": float(np.mean([row["N_mean_midrank"] for row in selected])),
            "shift_N_mean_midrank": np.asarray([row["shift_N_mean_midrank"] for row in selected])
            .mean(0)
            .tolist(),
            "energy_spearman": defined_summary([row["energy_association"]["spearman"] for row in leads]),
            "shift_energy_spearman": defined_summary([row["mean_shift_energy_spearman"] for row in leads]),
            "energy_jaccard": float(np.mean([row["energy_association"]["jaccard"] for row in leads])),
            "shift_energy_jaccard": float(np.mean([row["mean_shift_energy_jaccard"] for row in leads])),
            "constant_map_leads": sum(row["constant_map"] for row in leads),
            "empty_jaccard_unions": sum(row["energy_association"]["empty_jaccard_union"] for row in leads),
            "probability_mean": float(np.mean([row["probability_mean"] for row in leads])),
            "probability_q95_mean": float(np.mean([row["probability_q95"] for row in leads])),
            "probability_max": float(max(row["probability_max"] for row in leads)),
        }
    by_fit_lead = {
        str(fit): {
            arm: {
                lead: float(
                    np.mean(
                        [
                            indexed[identifier, fit, arm]["leads"][position]["gain"]
                            for identifier in identifiers
                        ]
                    )
                )
                for position, lead in enumerate(NATIVE_LEADS)
            }
            for arm in ARMS
        }
        for fit in FITS
    }
    return {
        "primary_descriptive": estimates["B"],
        "per_arm": arm_summaries,
        "PK_minus_B": estimates["PK_minus_B"],
        "per_fit_arm_lead_gain": by_fit_lead,
        "bootstrap_seed": config["bootstrap"]["seed"],
        "bootstrap_draws": len(draws),
        "skipped_draws": 0,
        "bootstrap_columns": [*ARMS, "PK_minus_B"],
        "records": 160,
        "model_record_rows": 1920,
        "native_lead_rows": 15360,
        "fixed_fit_count": 3,
    }, contrasts
