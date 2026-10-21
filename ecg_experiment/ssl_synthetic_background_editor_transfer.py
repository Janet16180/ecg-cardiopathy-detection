"""Fixed numerical editors and frozen-model background-transfer diagnostics."""

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score

from ecg_experiment import ssl_synthetic_generated_quality as quality
from ecg_experiment import ssl_synthetic_randomized_localization as original
from ecg_experiment.intervals import patient_groups, patient_resample

SOURCES = ("generated", "real")
VARIANTS = ("sham", "nuisance", "hann_focal", "hann_repeated", "biphasic_focal", "biphasic_repeated")
POSITIVES = VARIANTS[2:]
FITS = (1, 2, 3)
ARMS = ("B", "P", "K", "PK")
NATIVE = (0, 1, 6, 7, 8, 9, 10, 11)


def kernel(width: int, family: str) -> np.ndarray:
    """Construct the original Hann or nominal-energy-matched held-out biphasic shape.

    Parameters
    ----------
    width : int
        Frozen odd window width,61,81 or101 samples.
    family : str
        Either hann or biphasic; no parameters are fitted.

    Returns
    -------
    np.ndarray
        Float64 waveform before multiplication by the fixed sign and amplitude.
    """
    if width not in (61, 81, 101) or family not in ("hann", "biphasic"):
        raise ValueError("Unknown prospective kernel geometry")
    hann = np.hanning(width)
    if family == "hann":
        return hann
    wave = hann * np.sin(2 * np.pi * np.arange(width) / (width - 1))
    return wave * np.sqrt(np.sum(hann**2) / np.sum(wave**2))


def make_case(
    standardized: np.ndarray, index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply six matched variants without refitting or rescaling after an intervention.

    Parameters
    ----------
    standardized : np.ndarray
        Finite float32 eight-native-lead input with shape(8,5000).
    index : int
        Shared geometry identity2000 through2063, unrelated to real patient identity.

    Returns
    -------
    tuple
        Six canonical inputs, four strict masks, four threshold masks and full geometry metadata.
    """
    if (
        standardized.shape != (8, 5000)
        or standardized.dtype != np.float32
        or not np.isfinite(standardized).all()
    ):
        raise ValueError("Require finite standardized float32 native inputs")
    if index not in range(2000, 2064):
        raise ValueError("Geometry index outside fixed shared64 slots")
    phase = np.random.default_rng(2026103100 + index).uniform(0, 2 * np.pi)
    nuisance = (0.1 * np.sin(2 * np.pi * 0.33 * (np.arange(5000) / 500) + phase)).astype(np.float32)
    baseline = (standardized + nuisance).astype(np.float32)
    parameters = original.edit_parameters(index)
    variants = [quality.derive_leads(standardized), quality.derive_leads(baseline)]
    declarations = []
    for family in ("hann", "biphasic"):
        for parameter in parameters:
            independent = baseline.copy()
            shape = kernel(parameter["width"], family)
            bump = (parameter["sign"] * parameter["amplitude"] * shape).astype(np.float32)
            half = parameter["width"] // 2
            for center in parameter["centers"]:
                independent[parameter["independent_lead"], center - half : center + half + 1] += bump
            variants.append(quality.derive_leads(independent))
            declarations.append(
                {
                    **parameter,
                    "family": family,
                    "kernel_peak_abs": float(np.max(np.abs(bump))),
                    "kernel_mean": float(bump.astype(np.float64).mean()),
                    "kernel_squared_l2": float(np.sum(bump.astype(np.float64) ** 2)),
                }
            )
    observed = np.stack(variants)
    delta = observed[2:].astype(np.float64) - observed[1].astype(np.float64)
    strict, support = delta != 0, np.abs(delta) > 0.01
    checks = []
    for position, declaration in enumerate(declarations):
        mask, exact, energy = support[position], strict[position], delta[position] ** 2
        if not mask.any() or mask.all() or not np.all(~mask | exact):
            raise ValueError("An intervention lacks valid exact final-input support")
        basis = np.zeros((8, 1), dtype=np.float32)
        basis[declaration["independent_lead"]] = 1
        affected = np.flatnonzero(quality.derive_leads(basis)[:, 0] != 0)
        hull = np.zeros((12, 5000), dtype=bool)
        half = declaration["width"] // 2
        for center in declaration["centers"]:
            hull[affected, center - half : center + half + 1] = True
        if np.any(exact & ~hull):
            raise ValueError("Final numerical difference escapes its declared lead/time hull")
        checks.append(
            {
                "variant": POSITIVES[position],
                "support_cells": int(mask.sum()),
                "strict_cells": int(exact.sum()),
                "prevalence": float(mask.mean()),
                "actual_delta_squared_l2": float(energy.sum()),
                "actual_delta_peak_abs": float(np.abs(delta[position]).max()),
                "retained_energy_fraction": float(energy[mask].sum() / energy.sum()),
                "below_threshold_maximum_residual": float(np.abs(delta[position][~mask]).max()),
                "affected_canonical_leads": affected.tolist(),
                "outside_hull_cells": 0,
            }
        )
    return (
        observed,
        strict,
        support,
        {"geometry_index": index, "nuisance_phase": float(phase), "edits": declarations, "checks": checks},
    )


def score_record(
    probabilities: np.ndarray,
    observed: np.ndarray,
    support: np.ndarray,
    source: str,
    slot: int,
    fit: int,
    arm: str,
    donor_support: np.ndarray,
) -> dict[str, Any]:
    """Measure actual numerical support and matched controls on the complete twelve-output grid.

    Parameters
    ----------
    probabilities, observed : np.ndarray
        Six finite float32 canonical variants with shape(6,12,5000).
    support : np.ndarray
        Four Boolean final-input target masks in positive-variant order.
    source : str
        Real or generated background cohort.
    slot : int
        Shared geometry slot0 through63.
    fit : int
        Fixed fit1,2 or3, never selected or resampled.
    arm : str
        Fixed B,P,K orPK model.

    donor_support : np.ndarray
        Same-source next-slot actual support, with no overlap filtering or rerandomization.

    Returns
    -------
    dict
        Four complete positive endpoint rows and sham/nuisance probability summaries.
    """
    if source not in SOURCES or slot not in range(64) or fit not in FITS or arm not in ARMS:
        raise ValueError("Unknown fixed source/slot/model identity")
    if (
        probabilities.shape != (6, 12, 5000)
        or observed.shape != probabilities.shape
        or probabilities.dtype != np.float32
        or observed.dtype != np.float32
        or support.shape != (4, 12, 5000)
        or support.dtype != np.bool_
        or donor_support.shape != support.shape
        or donor_support.dtype != np.bool_
        or not np.isfinite(probabilities).all()
        or not np.isfinite(observed).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
    ):
        raise ValueError("Invalid finite probability/input/target geometry")
    row: dict[str, Any] = {
        "source": source,
        "slot": slot,
        "geometry_index": 2000 + slot,
        "fit_id": fit,
        "arm": arm,
        "positives": {},
        "controls": {},
    }
    for position, name in enumerate(POSITIVES):
        mask = support[position].ravel()
        expected = np.abs(observed[position + 2].astype(np.float64) - observed[1].astype(np.float64)) > 0.01
        if not np.array_equal(expected, support[position]) or not mask.any() or mask.all():
            raise ValueError("Saved final-input support identity differs")
        edited = float(average_precision_score(mask, probabilities[position + 2].ravel()))
        control = float(average_precision_score(mask, probabilities[1].ravel()))
        donor = float(
            average_precision_score(donor_support[position].ravel(), probabilities[position + 2].ravel())
        )
        amplitude = float(average_precision_score(mask, np.abs(observed[position + 2]).ravel()))
        difference = probabilities[position + 2].astype(np.float64) - probabilities[1].astype(np.float64)
        inside, outside = float(difference.ravel()[mask].mean()), float(difference.ravel()[~mask].mean())
        row["positives"][name] = {
            "edited_ap": edited,
            "nuisance_site_ap": control,
            "response_ap": edited - control,
            "amplitude_ap": amplitude,
            "amplitude_gain": edited - amplitude,
            "donor_site_ap": donor,
            "donor_gain": edited - donor,
            "donor_prevalence": float(donor_support[position].mean()),
            "prevalence": float(mask.mean()),
            "delta_inside": inside,
            "delta_outside": outside,
            "selectivity": inside - outside,
        }
    for position, name in enumerate(VARIANTS[:2]):
        values = probabilities[position].ravel().astype(np.float64)
        row["controls"][name] = {
            "mean": float(values.mean()),
            "q95": float(np.quantile(values, 0.95, method="linear")),
            "maximum": float(values.max()),
            "top600_mean": float(np.sort(values, kind="stable")[-600:].mean()),
            "maximum_gt_0p5": bool(values.max() > 0.5),
        }
    return row


def positive_vectors(indexed: dict) -> dict[str, np.ndarray]:
    """Build fixed slot vectors without resampling model fits.

    Parameters
    ----------
    indexed : dict
        Complete fixed model/source/slot metric rows.
    vectors : dict, optional
        Existing fixed positive endpoint vectors.

    Returns
    -------
    dict
        Fixed64-slot vectors in deterministic endpoint order.
    """
    vectors = {}
    for source in SOURCES:
        for arm in ARMS:
            for family in ("hann", "biphasic"):
                for metric in (
                    "edited_ap",
                    "nuisance_site_ap",
                    "response_ap",
                    "amplitude_ap",
                    "amplitude_gain",
                    "donor_site_ap",
                    "donor_gain",
                    "selectivity",
                ):
                    vectors[f"{source}__{arm}__{family}__{metric}"] = np.array(
                        [
                            np.mean(
                                [
                                    indexed[source, slot, fit, arm]["positives"][f"{family}_{mode}"][metric]
                                    for fit in FITS
                                    for mode in ("focal", "repeated")
                                ]
                            )
                            for slot in range(64)
                        ]
                    )
    return vectors


def factorial_vectors(vectors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Build fixed slot vectors without resampling model fits.

    Parameters
    ----------
    indexed : dict
        Complete fixed model/source/slot metric rows.
    vectors : dict, optional
        Existing fixed positive endpoint vectors.

    Returns
    -------
    dict
        Fixed64-slot vectors in deterministic endpoint order.
    """
    for arm in ARMS:
        for metric in (
            "edited_ap",
            "nuisance_site_ap",
            "response_ap",
            "amplitude_ap",
            "amplitude_gain",
            "donor_site_ap",
            "donor_gain",
            "selectivity",
        ):
            real = vectors[f"real__{arm}__biphasic__{metric}"] - vectors[f"real__{arm}__hann__{metric}"]
            generated = (
                vectors[f"generated__{arm}__biphasic__{metric}"]
                - vectors[f"generated__{arm}__hann__{metric}"]
            )
            vectors[f"interaction__{arm}__{metric}"] = real - generated
            for source in SOURCES:
                for family in ("hann", "biphasic"):
                    vectors[f"{source}__PK_minus_B__{family}__{metric}"] = (
                        vectors[f"{source}__PK__{family}__{metric}"]
                        - vectors[f"{source}__B__{family}__{metric}"]
                    )
    for arm in ARMS:
        for metric in ("edited_ap", "response_ap", "amplitude_gain", "donor_gain", "selectivity"):
            for source in SOURCES:
                vectors[f"{source}__{arm}__hann_minus_biphasic__{metric}"] = (
                    vectors[f"{source}__{arm}__hann__{metric}"]
                    - vectors[f"{source}__{arm}__biphasic__{metric}"]
                )
            for family in ("hann", "biphasic"):
                vectors[f"real_minus_generated__{arm}__{family}__{metric}"] = (
                    vectors[f"real__{arm}__{family}__{metric}"]
                    - vectors[f"generated__{arm}__{family}__{metric}"]
                )
    return vectors


def control_vectors(indexed: dict) -> dict[str, np.ndarray]:
    """Build fixed slot vectors without resampling model fits.

    Parameters
    ----------
    indexed : dict
        Complete fixed model/source/slot metric rows.
    vectors : dict, optional
        Existing fixed positive endpoint vectors.

    Returns
    -------
    dict
        Fixed64-slot vectors in deterministic endpoint order.
    """
    vectors = {}
    for arm in ARMS:
        for control in VARIANTS[:2]:
            for source in SOURCES:
                for metric in ("mean", "q95", "maximum", "top600_mean", "maximum_gt_0p5"):
                    vectors[f"{source}__{arm}__{control}__{metric}"] = np.array(
                        [
                            np.mean(
                                [indexed[source, slot, fit, arm]["controls"][control][metric] for fit in FITS]
                            )
                            for slot in range(64)
                        ]
                    )
    return vectors


def summarize(rows: list[dict[str, Any]], config: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    """Use shared whole-slot draws with fixed fits and six conjunctive engineering checks.

    Parameters
    ----------
    rows : list
        All1536 fixed model/source/background endpoint dictionaries.
    config : dict
        Frozen bootstrap settings and descriptive scientific contract.

    Returns
    -------
    tuple
        All-arm/source/family summaries, explicit primary verdict and all shared bootstrap values.
    """
    indexed = {(r["source"], r["slot"], r["fit_id"], r["arm"]): r for r in rows}
    expected = {(s, n, f, a) for s in SOURCES for n in range(64) for f in FITS for a in ARMS}
    if len(rows) != len(expected) or set(indexed) != expected:
        raise ValueError("Every fixed source/background/model row is required exactly once")
    vectors = factorial_vectors(positive_vectors(indexed))
    vectors.update(control_vectors(indexed))
    names = list(vectors)
    matrix = np.stack([vectors[name] for name in names], axis=1)
    if not np.isfinite(matrix).all():
        raise ValueError("All conditional slot endpoints must be finite")
    rng = np.random.default_rng(config["bootstrap"]["seed"])
    groups = patient_groups(np.arange(64))
    draws = np.stack(
        [matrix[patient_resample(groups, rng)].mean(axis=0) for _ in range(config["bootstrap"]["draws"])]
    )
    statistics = {
        name: {
            "mean": float(matrix[:, position].mean()),
            "ci_low": float(np.quantile(draws[:, position], 0.025, method="linear")),
            "ci_high": float(np.quantile(draws[:, position], 0.975, method="linear")),
        }
        for position, name in enumerate(names)
    }
    primary = {
        metric: statistics[f"real__B__biphasic__{metric}"]
        for metric in ("response_ap", "amplitude_gain", "donor_gain")
    }
    gates = {
        f"{metric}__{field}_positive": primary[metric][field] > 0
        for metric in primary
        for field in ("mean", "ci_low")
    }
    return {
        "endpoint_order": names,
        "endpoints": statistics,
        "primary": primary,
        "scientific_gates": gates,
        "scientific_passed": all(gates.values()),
        "bootstrap": {
            "draws": len(draws),
            "seed": config["bootstrap"]["seed"],
            "unit": "64_shared_geometry_slots_fixed_three_fits",
            "skipped_draws": 0,
        },
        "clinical_claim": False,
    }, draws
