"""Prospective alignment and normal phase calibration for Experiment 054."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from . import ROOT
from .ann_heads import ragged_unit_maps
from .files import sha256_file
from .fragment_localization import bootstrap_mean
from .lead_wave_maps import UnitMap
from .localization_inputs052 import analyse, load_baseline
from .raw_residual052 import (
    AFTER,
    BEFORE,
    FS,
    baseline_support,
    fixed_window_map,
    support_mask,
)

SEED = 54054
PHASE_EDGES = np.array([-0.25, -0.06, 0.08, 0.20, 0.45])


def aligned_field(signal: np.ndarray, peaks: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Align QRS morphology without fitting beat amplitude, then score original-time residuals.

    Parameters
    ----------
    signal : np.ndarray
        Twelve leads by samples in mV at 500 Hz.
    peaks : np.ndarray
        Fixed original R sample positions, used only for alignment.

    Returns
    -------
    tuple
        Original-time squared normalized residual field, and templates/shifts/scales for audit.
    """
    kept = peaks[(peaks >= BEFORE) & (peaks + AFTER <= signal.shape[1])]
    if len(kept) < 3:
        raise ValueError("Fewer than three complete beats")
    positions = kept[:, None] + np.arange(-135, 235)[None]
    valid = (positions >= 0) & (positions < signal.shape[1])
    beats = signal[:, np.clip(positions, 0, signal.shape[1] - 1)].transpose(1, 0, 2).copy()
    beats = np.where(valid[:, None], beats, np.nan)
    beats -= np.median(beats[:, :, 85:105], axis=2, keepdims=True)
    scale = np.maximum(np.median(np.ptp(beats[:, :, 105:175], axis=2), axis=0), 0.1)
    initial = np.nanmedian(beats[:, :, 5:365], axis=0)
    shifts_to_try = np.array([0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5])
    losses = np.stack(
        [
            np.mean(
                ((beats[:, :, 105 + shift : 175 + shift] - initial[None, :, 100:170]) / scale[None, :, None])
                ** 2,
                axis=(1, 2),
            )
            for shift in shifts_to_try
        ],
        axis=1,
    )
    shifts = shifts_to_try[np.argmin(losses, axis=1)]
    aligned = np.stack([beat[:, 5 + shift : 365 + shift] for beat, shift in zip(beats, shifts, strict=True)])
    field = np.zeros_like(signal, dtype=np.float64)
    predictions = []
    for index, (peak, shift) in enumerate(zip(kept, shifts, strict=True)):
        reference = np.nanmedian(np.delete(aligned, index, axis=0), axis=0)
        prediction = reference[:, np.arange(350) + 5 - shift]
        if not np.isfinite(prediction).all():
            raise ValueError("Nonfinite leave-one-out prediction")
        energy = ((beats[index, :, 10:360] - prediction) / scale[:, None]) ** 2
        left = max(peak - BEFORE, 0 if index == 0 else (kept[index - 1] + peak) // 2)
        right = min(
            peak + AFTER, signal.shape[1] if index == len(kept) - 1 else (peak + kept[index + 1]) // 2
        )
        field[:, left:right] = energy[:, left - peak + BEFORE : right - peak + BEFORE]
        predictions.append(prediction)
    return field, {
        "peaks": kept,
        "shifts": shifts,
        "scales": scale,
        "template": np.nanmedian(aligned, axis=0),
        "predictions": np.stack(predictions),
        "original_beats": beats[:, :, 10:360],
    }


def phase_indices(unit_map: UnitMap, peaks: np.ndarray, samples: int) -> np.ndarray:
    """Assign frozen original-R phase strata to the common candidate centers.

    Parameters
    ----------
    unit_map : UnitMap
        Fixed-duration window candidates.
    peaks : np.ndarray
        Original R positions; complete anchors only are used.
    samples : int
        Original recording length.

    Returns
    -------
    np.ndarray
        Phase index 0 through 3 for each candidate.
    """
    kept = peaks[(peaks >= BEFORE) & (peaks + AFTER <= samples)] / FS
    centers = (unit_map.starts + unit_map.ends) / 2
    anchors = kept[np.abs(centers[:, None] - kept[None]).argmin(axis=1)]
    relative = centers - anchors
    phases = np.searchsorted(PHASE_EDGES[1:-1], relative, side="right")
    if np.any(relative < -0.25 - 1e-8) or np.any(relative > 0.45 + 1e-8):
        raise ValueError("Candidate center outside complete-beat phase strata")
    return phases


def record_phase_maxima(unit_map: UnitMap, phases: np.ndarray) -> np.ndarray:
    """Compute all 48 lead/phase record maxima for a calibration normal.

    Parameters
    ----------
    unit_map : UnitMap
        Uncalibrated aligned residual windows.
    phases : np.ndarray
        Frozen phase indices for each window.

    Returns
    -------
    np.ndarray
        Twelve-by-four record maxima; every channel must exist.
    """
    maxima = np.empty((12, 4))
    for lead in range(12):
        for phase in range(4):
            channel = unit_map.scores[(unit_map.leads == lead) & (phases == phase)]
            if not len(channel):
                raise ValueError("Missing normal lead/phase channel")
            maxima[lead, phase] = np.max(channel)
    return maxima


def calibrated_map(unit_map: UnitMap, phases: np.ndarray, references: np.ndarray) -> UnitMap:
    """Calibrate windows against conservative normal record-max survival probabilities.

    Parameters
    ----------
    unit_map : UnitMap
        Original-time aligned residual energies.
    phases : np.ndarray
        Frozen phase indices.
    references : np.ndarray
        Sorted calibration record maxima, shape [patients, 12, 4], same N in every channel.

    Returns
    -------
    UnitMap
        Negative log smoothed conservative tails on identical supports.
    """
    count = len(references)
    scores = np.empty_like(unit_map.scores)
    for lead in range(12):
        for phase in range(4):
            selected = (unit_map.leads == lead) & (phases == phase)
            below = np.searchsorted(references[:, lead, phase], unit_map.scores[selected], side="left")
            scores[selected] = -np.log((count - below + 1) / (count + 1))
    return UnitMap(scores, unit_map.leads, unit_map.starts, unit_map.ends)


def shared_mask(signal: np.ndarray, peaks: np.ndarray, baseline: UnitMap | None = None) -> np.ndarray:
    """Return the same eligible sample support used in Experiment 052.

    Parameters
    ----------
    signal : np.ndarray
        Original recording.
    peaks : np.ndarray
        Original R positions.
    baseline : UnitMap or None
        Saved U_B map for development. Training normals use the identical historical edge rule.

    Returns
    -------
    np.ndarray
        Common sample eligibility.
    """
    mask = support_mask(signal.shape[1], peaks)
    if baseline is not None:
        return mask & baseline_support(baseline, signal.shape[1])
    historical = np.zeros(signal.shape[1], dtype=bool)
    for peak in peaks[(peaks >= 150) & (peaks + AFTER <= signal.shape[1])]:
        historical[peak - BEFORE : peak + AFTER] = True
    return mask & historical


def make_maps(
    signal: np.ndarray, peaks: np.ndarray, references: np.ndarray, baseline: UnitMap | None = None
) -> tuple[dict[str, UnitMap], dict[str, np.ndarray]]:
    """Build aligned ablation and the single calibrated primary on equal supports.

    Parameters
    ----------
    signal : np.ndarray
        Original ECG.
    peaks : np.ndarray
        Fixed original peaks.
    references : np.ndarray
        Frozen sorted normal record-max references.
    baseline : UnitMap or None
        Historical development comparator.

    Returns
    -------
    tuple
        Named maps and shift/template audit arrays.
    """
    field, audit = aligned_field(signal, peaks)
    raw = fixed_window_map(field, shared_mask(signal, peaks, baseline))
    phases = phase_indices(raw, peaks, signal.shape[1])
    return {"aligned_only": raw, "aligned_phase": calibrated_map(raw, phases, references)}, audit


def primary_summary(table: pd.DataFrame, names: tuple[str, ...], seed: int) -> dict[str, Any]:
    """Summarize the common-support primary location metrics without display tie breaking.

    Parameters
    ----------
    table : pd.DataFrame
        Per-record metrics with method names and patient IDs.
    names : tuple[str, ...]
        Methods to summarize.
    seed : int
        Whole-patient bootstrap seed.

    Returns
    -------
    dict
        Per-method metrics and the paired primary gain over U_B_fixed.
    """
    pvc = table[table["hit"].notna()]
    result = {}
    for name in names:
        group = pvc[pvc["map"] == name]
        result[name] = {
            "records": len(group),
            "hit": float(group["hit"].mean()),
            "chance": float(group["chance"].mean()),
            "excess": bootstrap_mean(
                group["patient_id"].to_numpy(), (group["hit"] - group["chance"]).to_numpy(), 2000, seed
            ),
            "median_distance_seconds": float(group["distance_seconds"].median()),
            "mean_overlap_fraction": float(group["overlap_fraction"].mean()),
            "tie_records": int((group["tie_count"] > 1).sum()),
            "mean_tie_count": float(group["tie_count"].mean()),
        }
        for key in ("legacy_hit", "legacy_chance", "ownership_hit", "ownership_chance"):
            if key in group:
                result[name][key] = float(group[key].mean())
    primary = pvc[pvc["map"] == names[0]].set_index("ecg_id")
    baseline = pvc[pvc["map"] == "U_B_fixed"].set_index("ecg_id").loc[primary.index]
    gain = (primary["hit"] - primary["chance"]) - (baseline["hit"] - baseline["chance"])
    paired = bootstrap_mean(primary["patient_id"].to_numpy(), gain.to_numpy(), 2000, seed)
    result["paired_gain"] = paired
    result["passes_real_rule"] = paired["value"] >= 0.10 and paired["ci_low"] > 0
    return result


def synthetic_summary(table: pd.DataFrame, seed: int) -> dict[str, Any]:
    """Compute frozen exact joint localization and paired change summaries.

    Parameters
    ----------
    table : pd.DataFrame
        Edited and unchanged paired per-record values.
    seed : int
        Patient bootstrap seed.

    Returns
    -------
    dict
        Known-support engineering metrics, with unchanged decision thresholds.
    """
    result = {}
    for kind in ("ST", "QRS", "ST_time_shift", "ST_lead_permutation"):
        group = table[table["kind"] == kind]
        joint = bootstrap_mean(group["patient_id"].to_numpy(), group["joint_hit"].to_numpy(), 2000, seed)
        change = bootstrap_mean(
            group["patient_id"].to_numpy(), (group["score"] - group["control_score"]).to_numpy(), 2000, seed
        )
        result[kind] = {
            "joint_localization": joint,
            "score_change": change,
            "median_distance_seconds": float(group["distance_seconds"].median()),
            "mean_overlap_fraction": float(group["overlap_fraction"].mean()),
            "passes": joint["value"] >= 0.80 and change["ci_low"] > 0,
        }
    return result


def reproduce052() -> tuple[
    dict[str, Any], list[UnitMap], dict[int, list[tuple[float, float]]], dict[str, Any]
]:
    """Verify frozen 052 receipts and reproduce its complete primary/descriptive/synthetic metrics.

    Returns
    -------
    tuple
        Historical rows, U_B maps, targets and completed integrity checks.
    """
    prior_path = ROOT / "outputs/experiment052_raw_residual_v1"
    prior = json.loads((prior_path / "result.json").read_text())
    for group in ("sources", "inputs"):
        for name, expected in prior["identity"][group].items():
            if sha256_file(ROOT / name) != expected:
                raise ValueError(f"052 pinned {group} mismatch: {name}")
    for name, expected in prior["outputs_sha256"].items():
        if sha256_file(prior_path / name) != expected:
            raise ValueError(f"052 output mismatch: {name}")
    rows, baseline, windows, integrity = load_baseline()
    table = pd.read_csv(prior_path / "development_locations.csv", float_precision="round_trip")
    primary = primary_summary(table, ("raw_residual", "U_B_fixed"), 52052)
    primary["excluded"] = []
    if primary != prior["development"]["primary"]:
        raise ValueError("052 complete primary summary differs")
    with np.load(prior_path / "development_maps.npz") as saved:
        arrays = {name: saved[name] for name in saved.files}
    fixed = ragged_unit_maps(arrays, "U_B_fixed")
    raw = ragged_unit_maps(arrays, "raw_residual")
    descriptive = analyse(rows, {"U_B": fixed, "raw_residual": raw}, windows, False)
    descriptive["maps"]["U_B_fixed"] = descriptive["maps"].pop("U_B")
    descriptive["thresholds"]["U_B_fixed"] = descriptive["thresholds"].pop("U_B")
    if json.loads(json.dumps(descriptive)) != prior["development"]["descriptive_042_rule"]:
        raise ValueError("052 complete descriptive metrics differ")
    synthetic = pd.read_csv(prior_path / "synthetic_locations.csv", float_precision="round_trip")
    found = synthetic_summary(synthetic, 52052)
    nuisance = synthetic[synthetic["kind"] == "nuisance"]
    threshold = float(np.quantile(nuisance["control_score"], 0.95))
    found["nuisance"] = {
        "in_sample_threshold": threshold,
        "unchanged_flag_share": float((nuisance["control_score"] > threshold).mean()),
        "nuisance_flag_share": float((nuisance["score"] > threshold).mean()),
        "score_change": bootstrap_mean(
            nuisance["patient_id"].to_numpy(),
            (nuisance["score"] - nuisance["control_score"]).to_numpy(),
            2000,
            52052,
        ),
    }
    selected = synthetic[synthetic["kind"] == "ST"]["ecg_id"].tolist()
    development_patients = set(rows["scored"]["patient_id"])
    found.update(
        {
            "records": len(selected),
            "ecg_ids": selected,
            "patients_disjoint": not bool(set(synthetic["patient_id"]) & development_patients),
            "skipped_fewer_than_five_complete_beats": [],
        }
    )
    if found != prior["synthetic"]:
        raise ValueError("052 synthetic summaries differ")
    return (
        rows,
        baseline,
        windows,
        {
            "U_B": integrity,
            "052_primary_equal": True,
            "052_descriptive_equal": True,
            "052_synthetic_equal": True,
            "052_result_sha256": sha256_file(prior_path / "result.json"),
        },
    )
