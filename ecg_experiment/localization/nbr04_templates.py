"""
NBR-04 leave-region-out personalization and the disease-direction-free subspace.

``masked_subspace_coefficients`` is 062's ``subspace_coefficients`` with the Huber weights multiplied by a
keep mask, so the bins of the replaced region never inform the coordinates; with every bin kept it returns
062's coordinates exactly.
"""

from __future__ import annotations

from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from .. import normative_map062 as nm
from .. import replacement070 as rp
from ..pr74_screening_features import read_training_signal
from . import nbr01_counterpart as nc
from .nbr02_templates import SEX_CODE

COMPONENT_BINS = {0: (0, 150), 1: (150, 200), 2: (200, 330)}
SUPERCLASSES = ("MI", "STTC", "CD", "HYP")
DEGENERATE = 1e-3
_STATE: dict[str, Any] = {}


def region_keep(item: nc.Variant) -> np.ndarray:
    """
    Canonical bins that may inform the coordinates when one region is replaced.

    Parameters
    ----------
    item : nc.Variant
        A normal-counterpart variant.

    Returns
    -------
    np.ndarray
        Flattened (12 * BINS,) boolean keep mask: the region's component bins (P = PR segment, QRS, ST-T =
        ST and T segments of 062's grid) in its leads are dropped; the full replacement drops every bin.
    """
    keep = np.ones((nc.LEADS, nm.BINS), bool)
    if item.name.endswith("full"):
        keep[:] = False
    for lead in item.leads:
        for part in item.components:
            start, stop = COMPONENT_BINS[part]
            keep[lead, start:stop] = False
    return keep.ravel()


def masked_subspace_coefficients(
    model: nm.PopulationModel, standard: np.ndarray, keep: np.ndarray
) -> np.ndarray:
    """
    062's robust bounded subspace coordinates with dropped bins given zero weight.

    Parameters
    ----------
    model : nm.PopulationModel
        Fitted model.
    standard : np.ndarray
        (LEADS * BINS,) standardized covariate residual.
    keep : np.ndarray
        (LEADS * BINS,) boolean mask of bins that may inform the coordinates.

    Returns
    -------
    np.ndarray
        (k,) coordinates, clipped at ``BOX`` standard deviations; zeros when no bin is kept.
    """
    rank = model.basis.shape[1]
    if rank == 0:
        return np.zeros(0)
    mask = keep.astype(float)
    weights = np.ones_like(standard) * mask
    prior = np.diag(model.noise / model.eigenvalues)
    for _ in range(5):
        weighted = model.basis * weights[:, None]
        coordinates = np.linalg.solve(model.basis.T @ weighted + prior, weighted.T @ standard)
        residual = np.abs(standard - model.basis @ coordinates) / model.off_scale
        weights = np.minimum(1.0, nm.HUBER / np.maximum(residual, 1e-12)) * mask
    bound = nm.BOX * np.sqrt(model.eigenvalues)
    return np.clip(coordinates, -bound, bound)


def region_differences(
    raw: np.ndarray, age: float, sex: float, model: nm.PopulationModel, items: list[nc.Variant]
) -> dict[str, Any] | None:
    """
    One counterpart-minus-record difference per region, each from coordinates fitted without that region.

    Parameters
    ----------
    raw : np.ndarray
        (12, 5000) signal in mV.
    age, sex : float
        Covariates, sex coded 0 male and 1 female.
    model : nm.PopulationModel
        Rank-6 model.
    items : list[nc.Variant]
        The 52 normal-counterpart variants.

    Returns
    -------
    dict[str, Any] or None
        ``differences`` (52, 12, 5000) float32, ``coordinates`` (52, k), ``clipped`` (52,) counts,
        ``movement`` (52,) mean absolute grid difference from the rank-0 template and ``knots``.
    """
    raw = np.asarray(raw, np.float64)
    features = nm.record_features(raw)
    if features is None:
        return None
    clean = nm.preprocess(raw, nm.delineate(raw)["qon"])
    covariates = nm.design(features.intervals[:1], np.array([age]), np.array([sex]))[0]
    mean = covariates @ model.coefficients
    standard = (features.median.ravel() - mean) / model.scale
    bound = nm.BOX * np.sqrt(model.eigenvalues)
    differences, coordinates, clipped, movement = [], [], [], []
    for item in items:
        values = masked_subspace_coefficients(model, standard, region_keep(item))
        shift = model.scale * (model.basis @ values)
        template = (mean + shift).reshape(nc.LEADS, nm.BINS)
        differences.append(
            (rp.template_timeline(template, features.knots, raw.shape[1]) - clean).astype(np.float32)
        )
        coordinates.append(values)
        clipped.append(int(np.sum(np.isclose(np.abs(values), bound, rtol=0.0, atol=1e-12))))
        movement.append(float(np.mean(np.abs(shift))))
    return {
        "differences": np.stack(differences),
        "coordinates": np.stack(coordinates),
        "clipped": np.array(clipped),
        "movement": np.array(movement),
        "knots": features.knots,
    }


def region_signals(
    raw: np.ndarray, components: np.ndarray, differences: np.ndarray, items: list[nc.Variant]
) -> tuple[np.ndarray, np.ndarray]:
    """
    ``nbr01_counterpart.variant_signals`` with one difference per variant.

    Parameters
    ----------
    raw : np.ndarray
        (12, 5000) signal in mV.
    components : np.ndarray
        (3, 5000) component masks of the record (NBR-01's plan).
    differences : np.ndarray
        (variants, 12, 5000) float32 differences.
    items : list[nc.Variant]
        Variants, all with source ``"counterpart"``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (variants, 12, 5000) signals and (variants,) energies.
    """
    raw = np.asarray(raw, np.float64)
    weights = rp.taper(torch.from_numpy(nc.variant_masks(components, items))).numpy()
    signals = np.repeat(raw[None], len(items), axis=0)
    energy = np.zeros(len(items))
    for position in range(len(items)):
        change = weights[position] * differences[position].astype(np.float64)
        signals[position] = raw + change
        energy[position] = float(np.sum(change**2))
    return signals, energy


def _features_job(position: int) -> nm.RecordFeatures | None:
    row = _STATE["rows"].iloc[position]
    with threadpool_limits(limits=1):
        return nm.record_features(read_training_signal(_STATE["input_root"], row))


def cohort_standardized(
    input_root: Path, rows: pd.DataFrame, model: nm.PopulationModel, workers: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Standardized covariate residuals of cohort rows with known age and sex that 062 can delineate.

    Parameters
    ----------
    input_root : Path
        Checkout holding the raw sources.
    rows : pd.DataFrame
        All-age cohort rows.
    model : nm.PopulationModel
        Model whose coefficients and scale standardize the medians.
    workers : int
        Forked single-thread processes.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Positions of the used rows in ``rows`` and their (used, LEADS * BINS) standardized residuals.
    """
    known = rows["age"].ne("") & rows["age_missing"].eq("False") & rows["sex_normalized"].isin(list(SEX_CODE))
    candidates = np.flatnonzero(known.to_numpy())
    _STATE.update(rows=rows.iloc[candidates].reset_index(drop=True), input_root=input_root)
    with get_context("fork").Pool(workers) as pool:
        features = pool.map(_features_job, range(len(candidates)), chunksize=8)
    used = [position for position, value in enumerate(features) if value is not None]
    age = rows["age"].iloc[candidates[used]].astype(float).to_numpy()
    sex = rows["sex_normalized"].iloc[candidates[used]].map(SEX_CODE).to_numpy()
    intervals = np.stack([features[position].intervals for position in used])
    covariates = nm.design(intervals[:, 0], age, sex)
    medians = np.stack([features[position].median.ravel() for position in used])
    return candidates[used], (medians - covariates @ model.coefficients) / model.scale


def disease_directions(standard: np.ndarray, flags: pd.DataFrame, normal: np.ndarray) -> dict[str, Any]:
    """
    Mean standardized residual of each superclass minus that of the normals, and an orthonormal set of them.

    Parameters
    ----------
    standard : np.ndarray
        (records, LEADS * BINS) standardized residuals.
    flags : pd.DataFrame
        0/1 superclass columns aligned with ``standard``.
    normal : np.ndarray
        Boolean mask of the normal rows.

    Returns
    -------
    dict[str, Any]
        ``raw`` (4, D) mean differences, ``orthonormal`` (D, 4) basis of their span, counts and norms.
    """
    reference = standard[normal].mean(axis=0)
    raw = np.stack([standard[flags[name].to_numpy() == 1].mean(axis=0) - reference for name in SUPERCLASSES])
    orthonormal, _ = np.linalg.qr(raw.T)
    return {
        "raw": raw,
        "orthonormal": orthonormal,
        "counts": {name: int(flags[name].sum()) for name in SUPERCLASSES},
        "normals": int(normal.sum()),
        "norms": {name: float(np.linalg.norm(value)) for name, value in zip(SUPERCLASSES, raw, strict=True)},
    }


def remove_directions(
    model: nm.PopulationModel, directions: np.ndarray, normals: np.ndarray
) -> tuple[nm.PopulationModel, dict[str, Any]]:
    """
    Project disease directions out of a model's basis, re-orthonormalize, and refit its variances on normals.

    Parameters
    ----------
    model : nm.PopulationModel
        Rank-k model (C6).
    directions : np.ndarray
        (D, m) orthonormal disease directions.
    normals : np.ndarray
        (n, D) standardized residuals of the fit normals.

    Returns
    -------
    tuple[nm.PopulationModel, dict[str, Any]]
        The new model (same coefficients and scale; basis orthogonal to the directions, rotated to its
        principal axes on the normals; eigenvalues, noise and off-subspace scale refitted as
        ``fit_population`` does) and diagnostics. Basis directions whose projected singular value falls
        below ``DEGENERATE`` are dropped.
    """
    projected = model.basis - directions @ (directions.T @ model.basis)
    left, singular, _ = np.linalg.svd(projected, full_matrices=False)
    span = left[:, singular > DEGENERATE]
    centred = normals - normals.mean(axis=0)
    coordinates = centred @ span
    covariance = coordinates.T @ coordinates / (len(normals) - 1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    basis = span @ vectors[:, order]
    eigenvalues = values[order]
    total = float(np.sum(centred**2) / (len(normals) - 1))
    noise = (total - float(eigenvalues.sum())) / (normals.shape[1] - basis.shape[1])
    off = normals - (normals @ basis) @ basis.T
    fitted = nm.PopulationModel(
        model.coefficients, model.scale, basis, eigenvalues, noise, nm.robust_scale(off)
    )
    inside = [
        float(np.linalg.norm(model.basis.T @ directions[:, column])) for column in range(directions.shape[1])
    ]
    return fitted, {
        "projected_singular_values": singular.tolist(),
        "rank": int(basis.shape[1]),
        "eigenvalues": eigenvalues.tolist(),
        "noise": noise,
        "original_eigenvalues": model.eigenvalues.tolist(),
        "original_noise": model.noise,
        "direction_norm_inside_original_span": inside,
    }
