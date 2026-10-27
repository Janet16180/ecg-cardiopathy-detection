"""Independent PTB membership and paired-background training comparison arithmetic."""

import csv
import hashlib
import io
from decimal import Decimal, InvalidOperation
from itertools import product
from pathlib import PurePosixPath
from typing import Any

import numpy as np

from ecg_experiment import ssl_synthetic_background_editor_transfer_audit as transfer
from ecg_experiment.intervals import patient_groups, patient_resample

TRAINING_ARMS = ("generated", "mixed")
METRICS = transfer.METRICS
CONTROLS = ("mean", "q95", "maximum", "top600_mean", "maximum_gt_0p5")


def identifier(value: str) -> str:
    """Normalize a positive integral metadata identity without truncating decimals.

    Parameters
    ----------
    value : str
        Original identity or fold text.

    Returns
    -------
    str
        Positive canonical integer text.
    """
    try:
        number = Decimal(value)
    except InvalidOperation as error:
        raise ValueError("Identity must be a positive finite integer") from error
    if not number.is_finite() or number <= 0 or number != int(number):
        raise ValueError("Identity must be a positive finite integer")
    return str(int(number))


def cohort_candidates(payload: bytes) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Reconstruct source-local TRAIN eligibility using only four PTB metadata fields.

    Parameters
    ----------
    payload : bytes
        Source CSV bytes; diagnoses and demographic fields are never interpreted.

    Returns
    -------
    tuple
        One lowest-ECG record per eligible patient in the declared digest order,
        and aggregate source-local exclusion counts. No waveform is opened.
    """
    fields = ("ecg_id", "patient_id", "strat_fold", "filename_hr")
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    if reader.fieldnames is None or any(reader.fieldnames.count(k) != 1 for k in fields):
        raise ValueError("Require each permitted metadata field exactly once")
    rows = []
    seen = set()
    for raw in reader:
        ecg, patient, fold = (identifier(raw[k]) for k in fields[:3])
        path = PurePosixPath(raw["filename_hr"])
        if (
            ecg in seen
            or int(fold) not in range(1, 11)
            or len(path.parts) != 3
            or path.parts[0] != "records500"
            or path.parts[1] in (".", "..")
            or path.name != f"{int(ecg):05d}_hr"
            or "\\" in str(path)
        ):
            raise ValueError("Invalid metadata identity, fold or waveform path")
        seen.add(ecg)
        rows.append(
            {"ecg_id": ecg, "patient_id": patient, "strat_fold": int(fold), "filename_hr": str(path)}
        )
    if not rows:
        raise ValueError("Empty metadata")
    quarantined = {r["patient_id"] for r in rows if r["strat_fold"] >= 8}
    patients = {r["patient_id"] for r in rows}
    original_train = {
        p
        for p in patients
        if int(hashlib.sha256(f"ssl_synthetic_partition_v1:20261018:{p}".encode()).hexdigest(), 16)
        % 10000
        < 8000
    }
    eligible = [r for r in rows if r["patient_id"] in original_train - quarantined]
    first = {}
    for row in sorted(eligible, key=lambda r: int(r["ecg_id"])):
        first.setdefault(row["patient_id"], row)
    ordered = sorted(
        first.values(),
        key=lambda r: (
            hashlib.sha256(f"ssl_mixed_background_v1:train:{r['patient_id']}".encode()).hexdigest(),
            int(r["patient_id"]),
        ),
    )
    return ordered, {
        "source_records": len(rows),
        "source_patients": len(patients),
        "fold8_9_10_quarantined_patients": len(quarantined),
        "eligible_records": len(eligible),
        "eligible_patients": len(ordered),
    }


def source_vectors(
    indexed: dict[tuple, dict[str, Any]], arm: str, source: str
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Reduce one complete evaluation source before shared-slot resampling.

    Parameters
    ----------
    indexed : dict
        Complete unique model-background row mapping.
    arm, source : str
        Fixed training treatment and evaluation source.

    Returns
    -------
    tuple
        Named positive endpoint vectors and twelve per-fit/mode summaries.
    """
    vectors = {}
    per_fit = []
    cube = np.asarray(
        [
            [
                [
                    [indexed[arm, source, slot, fit]["positives"][p][m] for m in METRICS]
                    for p in transfer.POSITIVES
                ]
                for fit in transfer.FITS
            ]
            for slot in range(64)
        ],
        dtype=np.float64,
    )
    for family_index, family in enumerate(transfer.FAMILIES):
        for mode_index, mode in enumerate((*transfer.MODES, "pooled")):
            positions = (
                [2 * family_index, 2 * family_index + 1]
                if mode == "pooled"
                else [2 * family_index + mode_index]
            )
            reduced = cube[:, :, positions, :].mean(axis=2).mean(axis=1)
            for metric_index, metric in enumerate(METRICS):
                vectors[f"{arm}__{source}__{family}__{mode}__{metric}"] = reduced[:, metric_index]
    for fit_index, fit in enumerate(transfer.FITS):
        for position, variant in enumerate(transfer.POSITIVES):
            points = cube[:, fit_index, position, :].mean(axis=0)
            per_fit.append(
                {
                    "training_arm": arm,
                    "source": source,
                    "fit_id": fit,
                    "variant": variant,
                    "points": dict(zip(METRICS, map(float, points), strict=True)),
                }
            )
    return vectors, per_fit


def endpoint_vectors(rows: list[dict[str, Any]]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Reduce every fixed fit and mode while retaining the shared 64-slot pairing.

    Parameters
    ----------
    rows : list
        Complete two-training-arm by two-source by 64-slot by three-fit rows.

    Returns
    -------
    tuple
        Named endpoint vectors and all 48 fit/mode point summaries.
    """
    indexed = {(r["training_arm"], r["source"], r["slot"], r["fit_id"]): r for r in rows}
    expected = {
        (arm, source, slot, fit)
        for arm in TRAINING_ARMS
        for source in transfer.SOURCES
        for slot in range(64)
        for fit in transfer.FITS
    }
    if len(rows) != 768 or set(indexed) != expected or any(r["arm"] != "B" for r in rows):
        raise ValueError("Require all768 paired training/evaluation identities exactly once")
    vectors = {}
    per_fit = []
    for arm, source in product(TRAINING_ARMS, transfer.SOURCES):
        cell, fits = source_vectors(indexed, arm, source)
        vectors.update(cell)
        per_fit.extend(fits)
    for arm in TRAINING_ARMS:
        for source in transfer.SOURCES:
            for control in ("sham", "nuisance"):
                for metric in CONTROLS:
                    vectors[f"{arm}__{source}__{control}__{metric}"] = np.asarray(
                        [
                            [
                                indexed[arm, source, slot, fit]["controls"][control][metric]
                                for fit in transfer.FITS
                            ]
                            for slot in range(64)
                        ],
                        dtype=np.float64,
                    ).mean(axis=1)
    for name in list(vectors):
        if name.startswith("mixed__"):
            suffix = name.removeprefix("mixed__")
            vectors["mixed_minus_generated__" + suffix] = vectors[name] - vectors["generated__" + suffix]
    if not all(np.isfinite(values).all() for values in vectors.values()):
        raise ValueError("Nonfinite paired endpoint")
    return vectors, per_fit


def decision(endpoints: dict[str, dict[str, float]]) -> dict[str, bool]:
    """Reconstruct the fourteen fixed advancement and preservation comparisons.

    Parameters
    ----------
    endpoints : dict
        Complete named means and lower/upper conditional confidence limits.

    Returns
    -------
    dict
        Every prespecified check, with its strict or inclusive boundary retained.
    """
    primary = endpoints["mixed_minus_generated__real__biphasic__pooled__edited_ap"]
    gates = {
        "primary_AP_gain_at_least_0p01": primary["mean"] >= 0.01,
        "primary_AP_gain_lower_positive": primary["ci_low"] > 0,
    }
    for metric in ("response_ap", "amplitude_gain", "donor_gain"):
        cell = endpoints[f"mixed__real__biphasic__pooled__{metric}"]
        gates[f"mixed_{metric}_point_positive"] = cell["mean"] > 0
        gates[f"mixed_{metric}_lower_positive"] = cell["ci_low"] > 0
    for mode in transfer.MODES:
        key = f"mixed_minus_generated__real__biphasic__{mode}__edited_ap"
        gates[f"real_biphasic_{mode}_noninferiority"] = endpoints[key]["ci_low"] > -0.01
    for family in transfer.FAMILIES:
        key = f"mixed_minus_generated__generated__{family}__pooled__edited_ap"
        gates[f"generated_{family}_noninferiority"] = endpoints[key]["ci_low"] > -0.01
    for control in ("sham", "nuisance"):
        key = f"mixed_minus_generated__real__{control}__top600_mean"
        gates[f"real_{control}_top600_preservation"] = endpoints[key]["ci_high"] < 0.02
    return gates


def summarize(rows: list[dict[str, Any]], config: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    """Independently resample shared evaluation slots with the fitted states held fixed.

    Parameters
    ----------
    rows : list
        Complete unchanged model-background metric rows.
    config : dict
        Frozen bootstrap configuration; no model or editor selection is performed.

    Returns
    -------
    tuple
        Complete endpoint/gate report and all 2000 named-column bootstrap draws.
    """
    if config["bootstrap"]["seed"] != 2026106701 or config["bootstrap"]["draws"] != 2000:
        raise ValueError("Bootstrap differs from prospective paired-slot protocol")
    vectors, fits = endpoint_vectors(rows)
    names = list(vectors)
    matrix = np.column_stack(list(vectors.values()))
    groups = patient_groups(np.arange(64))
    rng = np.random.default_rng(2026106701)
    draws = np.stack([matrix[patient_resample(groups, rng)].mean(axis=0) for _ in range(2000)])
    bounds = np.quantile(draws, [0.025, 0.975], axis=0, method="linear")
    endpoints = {
        name: {
            "mean": float(matrix[:, i].mean()),
            "ci_low": float(bounds[0, i]),
            "ci_high": float(bounds[1, i]),
        }
        for i, name in enumerate(names)
    }
    gates = decision(endpoints)
    return {
        "endpoints": endpoints,
        "endpoint_order": names,
        "per_fit_mode": fits,
        "primary": endpoints["mixed_minus_generated__real__biphasic__pooled__edited_ap"],
        "scientific_gates": gates,
        "scientific_passed": all(gates.values()),
        "clinical_claim": False,
        "bootstrap": {
            "draws": 2000,
            "seed": 2026106701,
            "fits_resampled": False,
            "unit": "shared64geometry_slots_fixed_three_paired_fits",
            "skipped_draws": 0,
        },
    }, draws
