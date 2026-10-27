"""Independent final-input reconstruction and fixed-fit background/editor comparison arithmetic."""

from typing import Any

import numpy as np

from ecg_experiment.intervals import patient_groups, patient_resample
from ecg_experiment.ssl_synthetic_quality_saved_audit import average_precision, canonical
from ecg_experiment.ssl_synthetic_randomized_saved_audit import parameters_for_index

SOURCES = ("generated", "real")
ARMS = ("B", "P", "K", "PK")
FITS = (1, 2, 3)
FAMILIES = ("hann", "biphasic")
MODES = ("focal", "repeated")
POSITIVES = tuple(f"{family}_{mode}" for family in FAMILIES for mode in MODES)
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


def kernel(width: int, family: str) -> np.ndarray:
    """Reconstruct a float64 nominal-energy-matched kernel without production editor code.

    Parameters
    ----------
    width : int
        One of the three fixed odd widths.
    family : str
        Original Hann or held-out bipolar family.

    Returns
    -------
    np.ndarray
        Signed amplitude is applied separately before the final float32 cast.
    """
    if width not in (61, 81, 101) or family not in FAMILIES:
        raise ValueError("Unknown frozen kernel geometry")
    window = np.hanning(width)
    if family == "hann":
        return window
    phase = 2 * np.pi * np.arange(width) / (width - 1)
    bipolar = window * np.sin(phase)
    return bipolar * np.sqrt(np.sum(window * window) / np.sum(bipolar * bipolar))


def make_case(native: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Reconstruct six float32 canonical variants and actual final-difference masks independently.

    Parameters
    ----------
    native : np.ndarray
        Already standardized finite float32 native8×5000, never standardized again.
    index : int
        Shared operator geometry index2000..2063.

    Returns
    -------
    tuple
        Complete inputs, strict masks, threshold masks and measured operator metadata.
    """
    if native.shape != (8, 5000) or native.dtype != np.float32 or not np.isfinite(native).all():
        raise ValueError("Require unchanged standardized native geometry")
    if index not in range(2000, 2064):
        raise ValueError("Shared geometry index outside prospective roster")
    phase = np.random.default_rng(2026103100 + index).uniform(0, 2 * np.pi)
    nuisance = (0.1 * np.sin(2 * np.pi * 0.33 * (np.arange(5000) / 500) + phase)).astype(np.float32)
    baseline = native + nuisance
    output = [canonical(native), canonical(baseline)]
    declarations = []
    for family in FAMILIES:
        for parameter in parameters_for_index(index):
            bump = (parameter["sign"] * parameter["amplitude"] * kernel(parameter["width"], family)).astype(
                np.float32
            )
            changed = baseline.copy()
            half = parameter["width"] // 2
            for center in parameter["centers"]:
                changed[parameter["independent_lead"], center - half : center + half + 1] += bump
            output.append(canonical(changed))
            declarations.append(
                {
                    **parameter,
                    "family": family,
                    "kernel_peak_abs": float(np.max(np.abs(bump))),
                    "kernel_mean": float(bump.astype(np.float64).mean()),
                    "kernel_squared_l2": float(np.sum(bump.astype(np.float64) ** 2)),
                }
            )
    observed = np.stack(output)
    differences = observed[2:].astype(np.float64) - observed[1].astype(np.float64)
    strict, support = differences != 0, np.abs(differences) > 0.01
    checks = []
    for position, declaration in enumerate(declarations):
        mask, exact, difference = support[position], strict[position], differences[position]
        energy = difference * difference
        lead = declaration["independent_lead"]
        affected = [0, 2, 3, 4, 5] if lead == 0 else [1, 2, 3, 4, 5] if lead == 1 else [lead + 4]
        hull = np.zeros((12, 5000), dtype=bool)
        half = declaration["width"] // 2
        for center in declaration["centers"]:
            hull[affected, center - half : center + half + 1] = True
        if not mask.any() or mask.all() or np.any(mask & ~exact) or np.any(exact & ~hull):
            raise ValueError("Final-difference masks violate the numerical intervention contract")
        checks.append(
            {
                "variant": POSITIVES[position],
                "support_cells": int(mask.sum()),
                "strict_cells": int(exact.sum()),
                "prevalence": float(mask.mean()),
                "actual_delta_squared_l2": float(energy.sum()),
                "actual_delta_peak_abs": float(np.abs(difference).max()),
                "retained_energy_fraction": float(energy[mask].sum() / energy.sum()),
                "below_threshold_maximum_residual": float(np.abs(difference[~mask]).max()),
                "affected_canonical_leads": affected,
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
    donor: np.ndarray,
) -> dict[str, Any]:
    """Independently score all targets and same-mask nuisance, amplitude and permuted-donor controls.

    Parameters
    ----------
    probabilities, observed : np.ndarray
        Six full float32 canonical maps and model inputs.
    support, donor : np.ndarray
        Four actual final masks and next-slot masks in the same source.
    source, arm : str
        Fixed source and checkpoint arm.
    slot, fit : int
        Fixed shared operator slot and model fit.

    Returns
    -------
    dict
        Complete model-background endpoints and uncalibrated control summaries.
    """
    if source not in SOURCES or arm not in ARMS or fit not in FITS or slot not in range(64):
        raise ValueError("Unknown prospective source/slot/model identity")
    if (
        probabilities.shape != (6, 12, 5000)
        or observed.shape != probabilities.shape
        or probabilities.dtype != np.float32
        or observed.dtype != np.float32
        or support.shape != (4, 12, 5000)
        or donor.shape != support.shape
        or support.dtype != bool
        or donor.dtype != bool
        or not np.isfinite(probabilities).all()
        or not np.isfinite(observed).all()
        or probabilities.min() < 0
        or probabilities.max() > 1
    ):
        raise ValueError("Invalid probability/input/final-mask geometry")
    positives = {}
    for position, name in enumerate(POSITIVES):
        target = support[position]
        delta = observed[position + 2].astype(np.float64) - observed[1].astype(np.float64)
        if not np.array_equal(target, np.abs(delta) > 0.01) or not target.any() or target.all():
            raise ValueError("Target is not the complete final-input support")
        if not donor[position].any() or donor[position].all():
            raise ValueError("Invalid nonself donor mask")
        edited = average_precision(target, probabilities[position + 2])
        nuisance = average_precision(target, probabilities[1])
        amplitude = average_precision(target, np.abs(observed[position + 2]))
        donor_ap = average_precision(donor[position], probabilities[position + 2])
        change = probabilities[position + 2].astype(np.float64) - probabilities[1].astype(np.float64)
        inside, outside = float(change[target].mean()), float(change[~target].mean())
        positives[name] = {
            "edited_ap": edited,
            "nuisance_site_ap": nuisance,
            "response_ap": edited - nuisance,
            "amplitude_ap": amplitude,
            "amplitude_gain": edited - amplitude,
            "donor_site_ap": donor_ap,
            "donor_gain": edited - donor_ap,
            "donor_prevalence": float(donor[position].mean()),
            "prevalence": float(target.mean()),
            "delta_inside": inside,
            "delta_outside": outside,
            "selectivity": inside - outside,
        }
    controls = {}
    for position, name in enumerate(("sham", "nuisance")):
        scores = probabilities[position].astype(np.float64).ravel()
        controls[name] = {
            "mean": float(scores.mean()),
            "q95": float(np.quantile(scores, 0.95, method="linear")),
            "maximum": float(scores.max()),
            "top600_mean": float(np.sort(scores, kind="stable")[-600:].mean()),
            "maximum_gt_0p5": bool(scores.max() > 0.5),
        }
    return {
        "source": source,
        "slot": slot,
        "geometry_index": 2000 + slot,
        "fit_id": fit,
        "arm": arm,
        "positives": positives,
        "controls": controls,
    }


def endpoint_vectors(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Reduce modes/fixed fits within every source/geometry slot before any resampling.

    Parameters
    ----------
    rows : list
        Every source×64slots×3fits×4arms exactly once.

    Returns
    -------
    dict
        Declared ordered endpoint names and64 equal-weight slot values.
    """
    indexed = {(row["source"], row["slot"], row["fit_id"], row["arm"]): row for row in rows}
    expected = {
        (source, slot, fit, arm) for source in SOURCES for slot in range(64) for fit in FITS for arm in ARMS
    }
    if len(rows) != 1536 or set(indexed) != expected:
        raise ValueError("Require all1536 source/slot/model rows exactly once")
    vectors = {}
    for source in SOURCES:
        for arm in ARMS:
            for family in FAMILIES:
                values = np.asarray(
                    [
                        [
                            [
                                [
                                    indexed[source, slot, fit, arm]["positives"][f"{family}_{mode}"][metric]
                                    for metric in METRICS
                                ]
                                for mode in MODES
                            ]
                            for fit in FITS
                        ]
                        for slot in range(64)
                    ]
                )
                collapsed = values.mean(axis=(1, 2))
                for position, metric in enumerate(METRICS):
                    vectors[f"{source}__{arm}__{family}__{metric}"] = collapsed[:, position]
    return _contrasts(vectors, indexed)


def _contrasts(vectors: dict[str, np.ndarray], indexed: dict[Any, Any]) -> dict[str, np.ndarray]:
    """Name all prospective source/family/objective contrasts and separate control vectors."""
    for arm in ARMS:
        for metric in METRICS:
            real = vectors[f"real__{arm}__biphasic__{metric}"] - vectors[f"real__{arm}__hann__{metric}"]
            generated = (
                vectors[f"generated__{arm}__biphasic__{metric}"]
                - vectors[f"generated__{arm}__hann__{metric}"]
            )
            vectors[f"interaction__{arm}__{metric}"] = real - generated
            for source in SOURCES:
                for family in FAMILIES:
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
            for family in FAMILIES:
                vectors[f"real_minus_generated__{arm}__{family}__{metric}"] = (
                    vectors[f"real__{arm}__{family}__{metric}"]
                    - vectors[f"generated__{arm}__{family}__{metric}"]
                )
        _control_vectors(vectors, indexed, arm)
    return vectors


def _control_vectors(vectors: dict[str, np.ndarray], indexed: dict[Any, Any], arm: str) -> None:
    """Retain every sham/nuisance raw-probability endpoint without rank null interpretations."""
    for control in ("sham", "nuisance"):
        for source in SOURCES:
            for metric in ("mean", "q95", "maximum", "top600_mean", "maximum_gt_0p5"):
                vectors[f"{source}__{arm}__{control}__{metric}"] = np.asarray(
                    [
                        np.mean(
                            [indexed[source, slot, fit, arm]["controls"][control][metric] for fit in FITS]
                        )
                        for slot in range(64)
                    ]
                )


def summarize(rows: list[dict[str, Any]], config: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    """Reconstruct all shared slot-bootstrap columns and the six fixed numerical-transfer gates.

    Parameters
    ----------
    rows : list
        Complete independently reconstructed endpoint rows.
    config : dict
        Frozen shared-slot RNG and draw count.

    Returns
    -------
    tuple
        Complete estimates/gates and all ordered2000×endpoint draws.
    """
    vectors = endpoint_vectors(rows)
    names = list(vectors)
    matrix = np.stack([vectors[name] for name in names], axis=1)
    if not np.isfinite(matrix).all():
        raise ValueError("All retained prospective slot endpoints must be finite")
    rng, groups = np.random.default_rng(config["bootstrap"]["seed"]), patient_groups(np.arange(64))
    draws = np.asarray(
        [matrix[patient_resample(groups, rng)].mean(axis=0) for _ in range(config["bootstrap"]["draws"])]
    )
    limits = np.quantile(draws, [0.025, 0.975], axis=0, method="linear")
    estimates = {
        name: {
            "mean": float(matrix[:, position].mean()),
            "ci_low": float(limits[0, position]),
            "ci_high": float(limits[1, position]),
        }
        for position, name in enumerate(names)
    }
    primary = {
        metric: estimates[f"real__B__biphasic__{metric}"]
        for metric in ("response_ap", "amplitude_gain", "donor_gain")
    }
    gates = {
        f"{metric}__{field}_positive": row[field] > 0
        for metric, row in primary.items()
        for field in ("mean", "ci_low")
    }
    return {
        "endpoint_order": names,
        "endpoints": estimates,
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
