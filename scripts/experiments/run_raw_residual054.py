"""Execute the frozen CPU alignment and normal record-max phase experiment."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ecg_experiment import ROOT
from ecg_experiment.external_encoders import read_ptb_float64
from ecg_experiment.files import sha256_file, write_json_atomic, write_npz_atomic
from ecg_experiment.fragment_localization import bootstrap_mean, r_peaks
from ecg_experiment.lead_wave_maps import UnitMap
from ecg_experiment.paths import to_stored
from ecg_experiment.provenance import git_head
from ecg_experiment.raw_residual052 import (
    FS,
    baseline_field,
    display_top,
    fixed_window_map,
    location_metrics,
    perturb,
    review_figure,
    tied_top,
)
from ecg_experiment.raw_residual054 import (
    SEED,
    aligned_field,
    make_maps,
    phase_indices,
    primary_summary,
    record_phase_maxima,
    reproduce052,
    shared_mask,
    synthetic_summary,
)

OUTPUT = ROOT / "outputs/experiment054_aligned_phase_residual_v1"
PROTOCOL = "docs/experiment-054-aligned-phase-residual.md"
COHORTS = "docs/experiment-054-cohorts.json"
LOG = logging.getLogger("experiment054")
METHODS = ("aligned_phase", "aligned_only")
EXAMPLES = (47, 219, 8, 184, 287, 30, 69)


def _save_maps(path: Path, keys: list[str], maps: dict[str, list[UnitMap]]) -> None:
    arrays = {"record_keys": np.asarray(keys)}
    for name, values in maps.items():
        arrays[f"{name}_offsets"] = np.concatenate([[0], np.cumsum([len(m.scores) for m in values])])
        for field in ("scores", "leads", "starts", "ends"):
            arrays[f"{name}_{field}"] = np.concatenate([getattr(m, field) for m in values])
    write_npz_atomic(path, **arrays)


def _manifest(rows: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    manifest = json.loads((ROOT / COHORTS).read_text())
    if (
        sha256_file(ROOT / "outputs/experiment052_raw_residual_v1/result.json")
        != manifest["052_result_sha256"]
    ):
        raise ValueError("Frozen 052 cohort exclusion receipt changed")
    if sha256_file(ROOT / "data/raw/ptb-xl/1.0.3/ptbxl_database.csv") != manifest["metadata_sha256"]:
        raise ValueError("Frozen cohort metadata changed")
    groups = {
        name: rows["fit"].set_index("ecg_id").loc[ids].reset_index()
        for name, ids in manifest["groups"].items()
    }
    patient_sets = [set(group["patient_id"]) for group in groups.values()]
    for index, patients in enumerate(patient_sets):
        if len(patients) != len(list(groups.values())[index]):
            raise ValueError("Duplicate cohort patient")
        others = set().union(*patient_sets[:index], *patient_sets[index + 1 :])
        if patients & (others | set(rows["scored"]["patient_id"])):
            raise ValueError("Cohort patient leakage")
    excluded = set(
        rows["fit"].loc[rows["fit"]["ecg_id"].isin(manifest["excluded_052_ecg_ids"]), "patient_id"]
    )
    if any(patients & excluded for patients in patient_sets):
        raise ValueError("052 synthetic patient reused")
    return groups, manifest


def _fit_reference(frame: pd.DataFrame, output: Path) -> np.ndarray:
    maxima, shifts = [], []
    for position, row in frame.iterrows():
        signal = read_ptb_float64(row["filename_hr"])
        peaks = r_peaks(signal, FS)
        field, audit = aligned_field(signal, peaks)
        unit_map = fixed_window_map(field, shared_mask(signal, peaks))
        maxima.append(record_phase_maxima(unit_map, phase_indices(unit_map, peaks, signal.shape[1])))
        shifts.extend(
            {"ecg_id": int(row["ecg_id"]), "peak": int(peak), "shift": int(shift)}
            for peak, shift in zip(audit["peaks"], audit["shifts"], strict=True)
        )
        if position % 100 == 0:
            LOG.info("reference %s/1000", position)
    references = np.sort(np.stack(maxima), axis=0)
    if references.shape != (1000, 12, 4) or not np.isfinite(references).all():
        raise ValueError("Incomplete normal record-max reference")
    write_npz_atomic(
        output / "normal_reference.npz",
        references=references,
        ecg_ids=frame["ecg_id"].to_numpy(),
        record_maxima=np.stack(maxima),
    )
    pd.DataFrame(shifts).to_csv(output / "calibration_shifts.csv", index=False)
    return references


def _location(unit_map: UnitMap, targets: list[tuple[float, float]], peaks: np.ndarray) -> dict[str, Any]:
    supports = [(low + 0.04, low + 0.18) for low, _ in targets]
    found = location_metrics(unit_map, supports)
    top = tied_top(unit_map)
    centers = (unit_map.starts + unit_map.ends) / 2
    anchor = np.array([low + 0.10 for low, _ in targets])
    found["distance_seconds"] = float(np.abs(centers[top, None] - anchor[None]).min(axis=1).mean())
    overlap = np.zeros(len(centers), dtype=bool)
    for low, high in targets:
        overlap |= (unit_map.starts < high) & (unit_map.ends > low)
    owners = peaks[np.abs(centers[:, None] * FS - peaks[None]).argmin(axis=1)] / FS
    owned = np.isclose(owners[:, None], anchor[None], atol=1 / FS).any(axis=1)
    found.update(
        {
            "legacy_hit": float(overlap[top].mean()),
            "legacy_chance": float(overlap.mean()),
            "ownership_hit": float(owned[top].mean()),
            "ownership_chance": float(owned.mean()),
        }
    )
    return found


def _overlay(output: Path, ecg_id: int, audit: dict[str, np.ndarray]) -> None:
    write_npz_atomic(output / "templates" / f"ECG_{ecg_id}.npz", **audit)
    from matplotlib.figure import Figure

    figure = Figure(figsize=(14, 11), layout="constrained")
    axes = figure.subplots(12, 1, sharex=True)
    seconds = np.arange(-125, 225) / FS
    for lead, axis in enumerate(axes):
        for beat in audit["original_beats"]:
            axis.plot(seconds, beat[lead], color="grey", alpha=0.4, linewidth=0.5)
        axis.plot(seconds, audit["template"][lead, 5:355], color="black", linewidth=0.9)
        axis.set_yticks([])
    figure.suptitle(f"ECG {ecg_id}: original baseline-subtracted beats and aligned median template")
    figure.savefig(output / "figures" / f"template_{ecg_id}.png", dpi=120)


def _development(
    rows: dict[str, Any],
    baseline: list[UnitMap],
    windows: dict[int, list[tuple[float, float]]],
    references: np.ndarray,
    output: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    maps = {name: [] for name in (*METHODS, "U_B_fixed")}
    records, shifts, keys = [], [], []
    for position, (_index, row) in enumerate(rows["scored"].iterrows()):
        signal = read_ptb_float64(row["filename_hr"])
        peaks = r_peaks(signal, FS)
        current, audit = make_maps(signal, peaks, references, baseline[position])
        current["U_B_fixed"] = fixed_window_map(
            baseline_field(baseline[position], signal.shape[1]),
            shared_mask(signal, peaks, baseline[position]),
        )
        ecg_id = int(row["ecg_id"])
        keys.append(str(ecg_id))
        shifts.extend(
            {"ecg_id": ecg_id, "peak": int(peak), "shift": int(shift)}
            for peak, shift in zip(audit["peaks"], audit["shifts"], strict=True)
        )
        for name, unit_map in current.items():
            maps[name].append(unit_map)
            top = display_top(unit_map)
            found = {
                "ecg_id": ecg_id,
                "patient_id": row["patient_id"],
                "map": name,
                "score": float(unit_map.scores.max()),
                "start": float(unit_map.starts[top]),
                "end": float(unit_map.ends[top]),
                "lead": int(unit_map.leads[top]),
                "saturated_windows": int(np.isclose(unit_map.scores, np.log(1001), atol=1e-10).sum())
                if name == "aligned_phase"
                else 0,
            }
            if ecg_id in windows:
                found.update(_location(unit_map, windows[ecg_id], peaks))
            records.append(found)
        if ecg_id in EXAMPLES:
            review_figure(signal, current, f"ECG {ecg_id}: prospective aligned residual maps").savefig(
                output / "figures" / f"ECG_{ecg_id}.png", dpi=120
            )
            _overlay(output, ecg_id, audit)
        if position % 100 == 0:
            LOG.info("development %s/1604", position)
    table = pd.DataFrame(records)
    table.to_csv(output / "development_locations.csv", index=False)
    pd.DataFrame(shifts).to_csv(output / "development_shifts.csv", index=False)
    _save_maps(output / "development_maps.npz", keys, maps)
    summary = primary_summary(table, (*METHODS, "U_B_fixed"), SEED)
    summary["excluded"] = []
    summary["saturated_records"] = int(
        (table[table["map"] == "aligned_phase"]["saturated_windows"] > 0).sum()
    )
    _real_review(rows, table, maps, keys, windows, output)
    return summary, table


def _real_review(
    rows: dict[str, Any],
    table: pd.DataFrame,
    maps: dict[str, list[UnitMap]],
    keys: list[str],
    windows: dict[int, list[tuple[float, float]]],
    output: Path,
) -> None:
    scored = table[table["hit"].notna()]
    first = scored[scored["map"] == "aligned_phase"].set_index("ecg_id")
    second = scored[scored["map"] == "U_B_fixed"].set_index("ecg_id").loc[first.index]
    gain = first["hit"] - second["hit"]
    for kind, selected in (("win", gain > 0), ("loss", gain < 0)):
        ids = sorted(first.index[selected].tolist())
        if not ids:
            continue
        ecg_id = ids[0]
        position = keys.index(str(ecg_id))
        row = rows["scored"][rows["scored"]["ecg_id"] == ecg_id].iloc[0]
        supports = [(low + 0.04, low + 0.18) for low, _ in windows[ecg_id]]
        review_figure(
            read_ptb_float64(row["filename_hr"]),
            {name: value[position] for name, value in maps.items()},
            f"ECG {ecg_id}: deterministic real {kind}",
            supports,
        ).savefig(output / "figures" / f"real_{kind}_{ecg_id}.png", dpi=120)


def _controls(frame: pd.DataFrame, references: np.ndarray, output: Path) -> dict[str, float]:
    records = []
    for _, row in frame.iterrows():
        signal = read_ptb_float64(row["filename_hr"])
        current, _ = make_maps(signal, r_peaks(signal, FS), references)
        records.extend(
            {"ecg_id": int(row["ecg_id"]), "map": name, "score": float(unit_map.scores.max())}
            for name, unit_map in current.items()
        )
    table = pd.DataFrame(records)
    table.to_csv(output / "threshold_controls.csv", index=False)
    return {name: float(table[table["map"] == name]["score"].quantile(0.95)) for name in METHODS}


def _synthetic_trials(
    signal: np.ndarray, peaks: np.ndarray, rng: np.random.Generator
) -> list[tuple[str, np.ndarray, list[tuple[float, float]], list[int]]]:
    kept = peaks[(peaks >= 125) & (peaks + 225 <= signal.shape[1])]
    if len(kept) < 5:
        raise ValueError("Frozen synthetic ECG has fewer than five complete beats")
    target = int(rng.choice(kept[1:-1]))
    st_leads = [int(rng.integers(0, 12))]
    qrs_leads = sorted(rng.choice(12, 3, replace=False).tolist())
    trials = []
    for kind, leads in (("ST", st_leads), ("QRS", qrs_leads)):
        edited, supports = perturb(signal, peaks, target, leads, kind)
        trials.append((kind, edited, supports, leads))
    shifted = int(kept[1:-1][np.flatnonzero(kept[1:-1] != target)[0]])
    edited, supports = perturb(signal, peaks, shifted, st_leads, "ST")
    trials.append(("ST_time_shift", edited, supports, st_leads))
    permutation = np.roll(np.arange(12), 1)
    leads = [int(np.flatnonzero(permutation == st_leads[0])[0])]
    edited, supports = perturb(signal[permutation], peaks, target, leads, "ST")
    trials.append(("ST_lead_permutation", edited, supports, leads))
    return trials


def _synthetics(
    frame: pd.DataFrame, references: np.ndarray, thresholds: dict[str, float], output: Path
) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    records, keys, figures, excluded = [], [], set(), []
    maps = {name: [] for name in METHODS}
    for _, row in frame.iterrows():
        signal = read_ptb_float64(row["filename_hr"])
        peaks = r_peaks(signal, FS)
        kept = peaks[(peaks >= 125) & (peaks + 225 <= signal.shape[1])]
        if len(kept) < 5:
            excluded.append(int(row["ecg_id"]))
            continue
        original, _ = make_maps(signal, peaks, references)
        ecg_id = int(row["ecg_id"])
        for kind, edited, supports, leads in _synthetic_trials(signal, peaks, rng):
            trial_references = (
                references[:, np.roll(np.arange(12), 1), :] if kind == "ST_lead_permutation" else references
            )
            current, _ = make_maps(edited, peaks, trial_references)
            keys.append(f"{ecg_id}:{kind}")
            for name, unit_map in current.items():
                maps[name].append(unit_map)
                found = location_metrics(unit_map, supports, leads)
                records.append(
                    {
                        "ecg_id": ecg_id,
                        "patient_id": row["patient_id"],
                        "kind": kind,
                        "map": name,
                        "control_score": float(original[name].scores.max()),
                        "support_start": supports[0][0],
                        "support_end": supports[0][1],
                        "changed_leads": json.dumps(leads),
                        **found,
                    }
                )
            _synthetic_review(edited, current, supports, leads, ecg_id, kind, figures, output)
        seconds = np.arange(signal.shape[1]) / FS
        nuisance, _ = make_maps(
            signal + 0.15 + 0.05 * np.sin(2 * np.pi * 0.2 * seconds)[None], peaks, references
        )
        keys.append(f"{ecg_id}:nuisance")
        for name, unit_map in nuisance.items():
            maps[name].append(unit_map)
            records.append(
                {
                    "ecg_id": ecg_id,
                    "patient_id": row["patient_id"],
                    "kind": "nuisance",
                    "map": name,
                    "control_score": float(original[name].scores.max()),
                    "score": float(unit_map.scores.max()),
                }
            )
    table = pd.DataFrame(records)
    table.to_csv(output / "synthetic_locations.csv", index=False)
    _save_maps(output / "synthetic_maps.npz", keys, maps)
    result = {}
    for name in METHODS:
        selected = table[table["map"] == name]
        result[name] = synthetic_summary(selected, SEED)
        result[name]["records"] = int(selected["ecg_id"].nunique())
        result[name]["excluded_fewer_than_five_complete_beats"] = excluded
        group = selected[selected["kind"] == "nuisance"]
        difference = group["score"] - group["control_score"]
        unchanged = float((group["control_score"] > thresholds[name]).mean())
        flagged = float((group["score"] > thresholds[name]).mean())
        result[name]["nuisance"] = {
            "threshold": thresholds[name],
            "unchanged_flag_share": unchanged,
            "nuisance_flag_share": flagged,
            "increase": flagged - unchanged,
            "score_change": bootstrap_mean(group["patient_id"].to_numpy(), difference.to_numpy(), 2000, SEED),
        }
    return result


def _synthetic_review(
    signal: np.ndarray,
    maps: dict[str, UnitMap],
    supports: list[tuple[float, float]],
    leads: list[int],
    ecg_id: int,
    kind: str,
    figures: set[str],
    output: Path,
) -> None:
    if kind not in ("ST", "QRS"):
        return
    primary = maps["aligned_phase"]
    status = "joint_success" if location_metrics(primary, supports, leads)["joint_hit"] == 1 else "failure"
    key = f"{kind}_{status}"
    if key in figures:
        return
    review_figure(
        signal, maps, f"Synthetic {kind}, ECG {ecg_id}: {status}; green = exact support", supports
    ).savefig(output / "figures" / f"synthetic_{key}_{ecg_id}.png", dpi=120)
    figures.add(key)


def _identity(stems: list[str]) -> dict[str, Any]:
    sources = (
        PROTOCOL,
        COHORTS,
        "ecg_experiment/raw_residual054.py",
        "scripts/experiments/run_raw_residual054.py",
        "ecg_experiment/raw_residual052.py",
        "ecg_experiment/localization_inputs052.py",
        "ecg_experiment/fragment_localization.py",
        "ecg_experiment/intervals.py",
        "ecg_experiment/paths.py",
        "ecg_experiment/lead_wave_maps.py",
        "ecg_experiment/external_encoders.py",
        "pyproject.toml",
        "uv.lock",
    )
    base = ROOT / "data/raw/ptb-xl/1.0.3"
    waveforms = {
        to_stored(base / f"{stem}.{suffix}"): sha256_file(base / f"{stem}.{suffix}")
        for stem in sorted(set(stems))
        for suffix in ("hea", "dat")
    }
    return {
        "sources": {name: sha256_file(ROOT / name) for name in sources},
        "waveforms": waveforms,
        "git_head": git_head(ROOT),
        "seed": SEED,
        "draws": 2000,
        "threads": 2,
    }


def main() -> None:
    """Run the prospective recipe only after all predecessor integrity checks pass.

    Returns
    -------
    None
        Writes one new local output receipt and review package.
    """
    partial = OUTPUT.with_name(OUTPUT.name + ".partial")
    if OUTPUT.exists() or partial.exists():
        raise FileExistsError(OUTPUT)
    partial.mkdir()
    (partial / "figures").mkdir()
    (partial / "templates").mkdir()
    logging.basicConfig(
        level=logging.INFO, handlers=[logging.FileHandler(partial / "run.log"), logging.StreamHandler()]
    )
    began = time.perf_counter()
    with threadpool_limits(limits=2):
        LOG.info("reproducing full 052 and U_B before new scores")
        rows, baseline, windows, integrity = reproduce052()
        groups, manifest = _manifest(rows)
        identity = _identity(
            rows["scored"]["filename_hr"].tolist()
            + [stem for group in groups.values() for stem in group["filename_hr"]]
        )
        LOG.info("integrity and patient separation passed; fitting frozen record-max reference")
        references = _fit_reference(groups["cdf_calibration"], partial)
        thresholds = _controls(groups["threshold_controls"], references, partial)
        development, _ = _development(rows, baseline, windows, references, partial)
        synthetic = _synthetics(groups["synthetic"], references, thresholds, partial)
        primary_synthetic = synthetic["aligned_phase"]
        promote = (
            development["passes_real_rule"]
            and primary_synthetic["ST"]["passes"]
            and primary_synthetic["QRS"]["passes"]
        )
        result = {
            "identity": identity,
            "integrity": integrity,
            "cohorts": manifest,
            "development": development,
            "synthetic": synthetic,
            "promote_for_review": bool(promote),
            "elapsed_seconds": time.perf_counter() - began,
            "excluded": [],
        }
        result["outputs_sha256"] = {
            path.name: sha256_file(path)
            for path in partial.iterdir()
            if path.is_file() and path.name != "run.log"
        }
        write_json_atomic(partial / "result.json", result)
    partial.rename(OUTPUT)
    LOG.info("complete %.1fs; promote=%s", result["elapsed_seconds"], promote)


if __name__ == "__main__":
    main()
