"""Exact synthetic-support metrics and predecessor integrity for Experiment 058."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import ROOT
from .files import sha256_file
from .fragment_localization import bootstrap_mean
from .incart_localization056 import METHODS, benchmark_summary
from .lead_wave_maps import UnitMap
from .paths import from_stored
from .st_episode057 import mean_interval
from .synthetic_generator058 import FS, SEED


def localization_metrics(unit_map: UnitMap | None, mask: np.ndarray, energy: np.ndarray) -> dict[str, Any]:
    """Evaluate a fixed-area mark against independently generated observable changes.

    Parameters
    ----------
    unit_map : UnitMap or None
        Frozen-method candidate scores on the original time axis; None means inference failed.
    mask : np.ndarray
        Evaluation-only exact lead/sample target mask.
    energy : np.ndarray
        Evaluation-only squared noise-free paired difference.

    Returns
    -------
    dict
        Tie-averaged hit, chance, IoU, changed energy fraction and distance; no exclusion.
    """
    empty = {
        "hit": 0.0,
        "chance": 0.0,
        "iou": 0.0,
        "energy_capture": 0.0,
        "distance_seconds": 12.0,
        "ties": 0,
        "score": 0.0,
        "display_start": 0.0,
        "display_lead": -1,
        "candidates": 0,
    }
    if unit_map is None or not len(unit_map.scores):
        return empty
    tied = np.flatnonzero(np.isclose(unit_map.scores, unit_map.scores.max(), rtol=1e-10, atol=1e-10))
    centers = np.rint((unit_map.starts + unit_map.ends) * FS / 2).astype(int)
    hit = mask[unit_map.leads.astype(int), centers]
    results = []
    target_area = int(mask.sum())
    total_energy = float(energy.sum())
    for index in tied:
        lead = int(unit_map.leads[index])
        left, right = round(unit_map.starts[index] * FS), round(unit_map.ends[index] * FS)
        overlap = int(mask[lead, left:right].sum())
        support = np.flatnonzero(mask[lead])
        distance = np.min(np.abs(support - centers[index])) / FS if len(support) else 12.0
        results.append(
            (
                overlap / (target_area + right - left - overlap) if target_area else 0,
                float(energy[lead, left:right].sum()) / total_energy if total_energy else 0,
                distance,
            )
        )
    means = np.mean(results, axis=0)
    display = int(tied[len(tied) // 2])
    return {
        "hit": float(hit[tied].mean()),
        "chance": float(hit.mean()),
        "iou": float(means[0]),
        "energy_capture": float(means[1]),
        "distance_seconds": float(means[2]),
        "ties": len(tied),
        "score": float(unit_map.scores.max()),
        "display_start": float(unit_map.starts[display]),
        "display_lead": int(unit_map.leads[display]),
        "candidates": len(unit_map.scores),
    }


def grouped_summary(table: pd.DataFrame) -> dict[str, Any]:
    """Keep every generator, distribution and intervention visible in paired summaries.

    Parameters
    ----------
    table : pd.DataFrame
        All test case/method rows, including inference failures and uninformative targets.

    Returns
    -------
    dict
        Per-axis intervals and frozen provisional robustness gates, never clinical promotion.
    """
    result = {}
    for (family, cohort), group in table.groupby(["family", "cohort"]):
        summary = {"kinds": {}}
        for kind, rows in group.groupby("kind"):
            methods = {}
            for method, values in rows.groupby("map"):
                methods[method] = {
                    name: bootstrap_mean(values.subject.to_numpy(), values[name].to_numpy(), 2000, SEED)
                    for name in ("hit", "chance", "target_occupancy", "iou", "energy_capture", "flag")
                }
                methods[method]["uninformative"] = int((~values.informative).sum())
                methods[method]["inference_failures"] = int(values.inference_failure.sum())
            first = rows[rows["map"] == METHODS[0]].set_index("subject")
            second = rows[rows["map"] == METHODS[1]].set_index("subject").loc[first.index]
            summary["kinds"][kind] = {
                "methods": methods,
                "paired_hit_gain": bootstrap_mean(
                    first.index.to_numpy(), (first.hit - second.hit).to_numpy(), 2000, SEED
                ),
            }
        transient = group[group.kind.isin(["qrs", "st", "t"])].pivot_table(
            index="subject", columns="map", values="hit"
        )
        gain = bootstrap_mean(
            transient.index.to_numpy(), (transient[METHODS[0]] - transient[METHODS[1]]).to_numpy(), 2000, SEED
        )
        gain.update({"subjects": len(transient), "draws": 2000, "seed": SEED})
        summary["transient_paired_gain"] = gain
        sham = group[group.kind == "sham"].set_index(["subject", "map"])
        guards = {}
        for kind in ("drift", "gain", "noise"):
            values = group[(group.kind == kind) & (group["map"] == METHODS[0])].set_index("subject")
            control = sham.xs(METHODS[0], level="map").loc[values.index]
            guards[kind] = bootstrap_mean(
                values.index.to_numpy(), (values.flag - control.flag).to_numpy(), 2000, SEED
            )
        summary["nuisance_increases"] = guards
        summary["provisional_transient_robustness_pass"] = (
            gain["value"] >= 0.1
            and gain["ci_low"] > 0
            and all(
                summary["kinds"][kind]["methods"][METHODS[0]]["hit"]["value"] >= 0.8
                for kind in ("qrs", "st", "t")
            )
            and all(value["ci_high"] <= 0.05 for value in guards.values())
        )
        result[f"{family}/{cohort}"] = summary
    return result


def _hash_receipt(directory: Path, groups: tuple[str, ...]) -> dict[str, Any]:
    """Verify a predecessor's pinned source, input and output hashes.

    Parameters
    ----------
    directory : Path
        Completed predecessor output directory.
    groups : tuple of str
        Identity hash groups to verify.

    Returns
    -------
    dict
        Parsed receipt, after every requested hash is checked.
    """
    receipt = json.loads((directory / "result.json").read_text())
    for group in groups:
        for path, expected in receipt["identity"][group].items():
            if sha256_file(from_stored(path)) != expected:
                raise ValueError(f"Predecessor {group} hash mismatch: {path}")
    for path, expected in receipt["outputs_sha256"].items():
        if sha256_file(directory / path) != expected:
            raise ValueError(f"Predecessor output hash mismatch: {path}")
    return receipt


def reproduce_predecessors() -> dict[str, Any]:
    """Reproduce live 056 endpoints and 057 primary/stress metrics before new scores.

    Returns
    -------
    dict
        Exact reproduction checks and predecessor receipt identities.
    """
    base = ROOT / "outputs/experiment056_incart_localization_v1"
    prior = _hash_receipt(base, ("sources", "inputs"))
    patients = pd.read_csv(base / "patient_metrics.csv", float_precision="round_trip")
    found = benchmark_summary(patients)
    nuisance = pd.read_csv(base / "nuisance.csv", float_precision="round_trip")
    thresholds = prior["thresholds"]
    nuisance_found = {}
    for method in METHODS:
        values = nuisance[nuisance["map"] == method]
        ordinary = float((values.score > thresholds[method]).mean())
        changed = float((values.nuisance_score > thresholds[method]).mean())
        rank = values.groupby("record")[["score", "nuisance_score"]].corr(method="spearman")
        nuisance_found[method] = {
            "unchanged_flag_share": ordinary,
            "nuisance_flag_share": changed,
            "increase": changed - ordinary,
            "mean_score_change": float((values.nuisance_score - values.score).mean()),
            "mean_record_spearman": float(rank.xs("score", level=1)["nuisance_score"].mean()),
        }
    found["gates"]["nuisance"] = nuisance_found[METHODS[0]]["increase"] <= 0.05
    normal = pd.read_csv(base / "normal_beat_scores.csv", float_precision="round_trip")
    if {method: float(normal[method].quantile(0.95)) for method in METHODS} != thresholds:
        raise ValueError("056 threshold reproduction differs")
    if found != prior["benchmark"] or nuisance_found != prior["nuisance"]:
        raise ValueError("056 exact metric reproduction differs")
    expert = pd.read_csv(base / "reference_beats.csv")
    detected = pd.read_csv(base / "detected_beats.csv")
    if (
        len(expert) != prior["coverage"]["reference_beats"]
        or len(detected) != prior["coverage"]["scored_detections"]
    ):
        raise ValueError("056 coverage reproduction differs")
    if expert.symbol.value_counts().to_dict() != prior["coverage"]["reference_types"]:
        raise ValueError("056 reference types differ")
    if detected.symbol.value_counts().to_dict() != prior["coverage"]["matched_types"]:
        raise ValueError("056 matched types differ")
    return {
        "056_exact_benchmark": True,
        "056_exact_nuisance": True,
        "056_exact_thresholds": True,
        "056_exact_coverage": True,
        "056_receipt_sha256": sha256_file(base / "result.json"),
        **reproduce057(),
    }


def reproduce057() -> dict[str, Any]:
    """Hash-check and reproduce the frozen ST episode primary and stress comparisons.

    Returns
    -------
    dict
        Exact primary and stress checks, plus receipt identity.
    """
    other = ROOT / "outputs/experiment057_st_episode_v1"
    second = _hash_receipt(other, ("sources", "data"))
    rows = json.loads((other / "per_record.json").read_text())
    eligible = [row for row in rows if row["eligible_st"]]
    ids = np.asarray([row["patient"] for row in eligible])
    gain = mean_interval(np.asarray([row["st"]["hit"] - row["whole"]["hit"] for row in eligible]), ids)
    if gain != second["paired_gain"]:
        raise ValueError("057 primary gain differs")
    for method in ("st", "whole"):
        found = mean_interval(np.asarray([row[method]["hit"] for row in eligible]), ids)
        if any(found[key] != second["primary"][method][key] for key in found):
            raise ValueError("057 primary hit differs")
    axis = [row for row in rows if row["eligible_axis"]]
    stress = mean_interval(
        np.asarray([row["st"]["axis_hit"] - row["whole"]["axis_hit"] for row in axis]),
        np.asarray([row["patient"] for row in axis]),
    )
    if stress != second["axis_paired_hit_increase"]:
        raise ValueError("057 stress reproduction differs")
    return {
        "057_exact_primary": True,
        "057_exact_stress": True,
        "057_receipt_sha256": sha256_file(other / "result.json"),
    }
