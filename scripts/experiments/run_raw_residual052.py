"""Run the frozen, CPU-only raw morphology localization experiment."""

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
from ecg_experiment.localization_inputs052 import analyse, load_baseline
from ecg_experiment.paths import to_stored
from ecg_experiment.provenance import git_head
from ecg_experiment.raw_residual052 import (
    FS,
    baseline_field,
    baseline_support,
    display_top,
    fixed_window_map,
    location_metrics,
    perturb,
    residual_field,
    review_figure,
    support_mask,
    tied_top,
)

OUTPUT = ROOT / "outputs/experiment052_raw_residual_v1"
PROTOCOL = "docs/experiment-052-raw-residual.md"
SEED = 52052
DRAWS = 2000
EXAMPLES = (47, 219, 8, 184, 287, 30, 69)
LOG = logging.getLogger("experiment052")


def _interval(patients: np.ndarray, values: np.ndarray) -> dict[str, float]:
    return bootstrap_mean(patients, values, DRAWS, SEED)


def _identity() -> dict[str, Any]:
    sources = (
        PROTOCOL,
        "ecg_experiment/raw_residual052.py",
        "ecg_experiment/localization_inputs052.py",
        "scripts/experiments/run_raw_residual052.py",
        "ecg_experiment/fragment_localization.py",
        "ecg_experiment/intervals.py",
        "ecg_experiment/lead_wave_maps.py",
        "ecg_experiment/paths.py",
        "ecg_experiment/full_development.py",
        "ecg_experiment/eda/ptbxl.py",
        "ecg_experiment/external_encoders.py",
        "ecg_experiment/ann_heads.py",
        "pyproject.toml",
        "uv.lock",
    )
    inputs = (
        "outputs/experiment042_lead_wave_maps_v1/unit_scores.npz",
        "outputs/experiment042_lead_wave_maps_v1/result.json",
        "outputs/experiment041_fragment_localization_v1/result.json",
        "outputs/experiment041_fragment_localization_v1/section_scores.npz",
        "data/raw/ptb-xl/1.0.3/ptbxl_database.csv",
        "data/raw/ptb-xl/1.0.3/scp_statements.csv",
        "outputs/data_quality/clean_rerun_preflight_v1/heldout_references.csv",
        "outputs/data_quality/clean_rerun_preflight_v1/labels_fraction1.csv",
        "outputs/data_quality/clean_rerun_preflight_v1/labels_fraction0.1.csv",
        "data/processed/training_union_500hz_v1/train_manifest.csv",
        "data/processed/pretrained/ecg-jepa-full-public/ecg_ids.npy",
        "outputs/experiment016_xecg_probe_finetune/features/ecg_ids.npy",
        "outputs/experiment004_cpc_40k/released_features/ecg_ids.npy",
    )
    return {
        "sources": {name: sha256_file(ROOT / name) for name in sources},
        "inputs": {name: sha256_file(ROOT / name) for name in inputs},
        "git_head": git_head(ROOT),
        "seed": SEED,
        "draws": DRAWS,
        "threads": 2,
    }


def _waveform_hashes(stems: list[str]) -> dict[str, str]:
    base = ROOT / "data/raw/ptb-xl/1.0.3"
    return {
        to_stored(base / f"{stem}.{suffix}"): sha256_file(base / f"{stem}.{suffix}")
        for stem in stems
        for suffix in ("hea", "dat")
    }


def _serialize_maps(path: Path, ids: list[int], maps: dict[str, list[UnitMap]]) -> None:
    arrays = {"ecg_ids": np.asarray(ids, dtype=np.int64)}
    for name, found in maps.items():
        arrays[f"{name}_offsets"] = np.concatenate([[0], np.cumsum([len(m.scores) for m in found])])
        for field in ("scores", "leads", "starts", "ends"):
            arrays[f"{name}_{field}"] = np.concatenate([getattr(m, field) for m in found])
    write_npz_atomic(path, **arrays)


def _primary_summary(
    rows: dict[str, Any],
    windows: dict[int, list[tuple[float, float]]],
    output: Path,
    maps: dict[str, list[UnitMap]],
    kept_ids: list[int],
    table: pd.DataFrame,
    excluded: list[int],
) -> dict[str, Any]:
    pvc = table[table["hit"].notna()]
    summary = {}
    for name in maps:
        group = pvc[pvc["map"] == name]
        excess = group["hit"] - group["chance"]
        summary[name] = {
            "records": len(group),
            "hit": float(group["hit"].mean()),
            "chance": float(group["chance"].mean()),
            "excess": _interval(group["patient_id"].to_numpy(), excess.to_numpy()),
            "median_distance_seconds": float(group["distance_seconds"].median()),
            "mean_overlap_fraction": float(group["overlap_fraction"].mean()),
            "tie_records": int((group["tie_count"] > 1).sum()),
            "mean_tie_count": float(group["tie_count"].mean()),
        }
        for key in ("legacy_hit", "legacy_chance", "ownership_hit", "ownership_chance"):
            summary[name][key] = float(group[key].mean())
    first = pvc[pvc["map"] == "raw_residual"].set_index("ecg_id")
    second = pvc[pvc["map"] == "U_B_fixed"].set_index("ecg_id").loc[first.index]
    gain = (first["hit"] - first["chance"]) - (second["hit"] - second["chance"])
    paired = _interval(first["patient_id"].to_numpy(), gain.to_numpy())
    summary.update(
        {
            "paired_gain": paired,
            "excluded": excluded,
            "passes_real_rule": paired["value"] >= 0.10 and paired["ci_low"] > 0,
        }
    )
    for kind, mask in (("real_win", gain > 0), ("real_loss", gain < 0)):
        ids = sorted(first.index[mask].tolist())
        if not ids:
            continue
        ecg_id = ids[0]
        row = rows["scored"][rows["scored"]["ecg_id"] == ecg_id].iloc[0]
        position = kept_ids.index(ecg_id)
        supports = [(low + 0.04, low + 0.18) for low, _ in windows[ecg_id]]
        review_figure(
            read_ptb_float64(row["filename_hr"]),
            {name: found[position] for name, found in maps.items()},
            f"ECG {ecg_id}: deterministic {kind}; green = automatic target",
            supports,
        ).savefig(output / "figures" / f"{kind}_{ecg_id}.png", dpi=120)
    return summary


def _development(
    rows: dict[str, Any], baseline: list[UnitMap], windows: dict[int, list[tuple[float, float]]], output: Path
) -> tuple[dict[str, Any], dict[str, list[UnitMap]], pd.DataFrame]:
    maps: dict[str, list[UnitMap]] = {"raw_residual": [], "U_B_fixed": []}
    records, excluded, kept_ids = [], [], []
    for position, (_index, row) in enumerate(rows["scored"].iterrows()):
        signal = read_ptb_float64(row["filename_hr"])
        peaks = r_peaks(signal, FS)
        kept = peaks[(peaks >= 125) & (peaks + 225 <= signal.shape[1])]
        if len(kept) < 3:
            excluded.append(int(row["ecg_id"]))
            continue
        mask = support_mask(signal.shape[1], peaks) & baseline_support(baseline[position], signal.shape[1])
        current = {
            "raw_residual": fixed_window_map(residual_field(signal, peaks), mask),
            "U_B_fixed": fixed_window_map(baseline_field(baseline[position], signal.shape[1]), mask),
        }
        if not np.array_equal(current["raw_residual"].starts, current["U_B_fixed"].starts):
            raise ValueError("Unequal candidate support")
        ecg_id = int(row["ecg_id"])
        kept_ids.append(ecg_id)
        for name, unit_map in current.items():
            maps[name].append(unit_map)
            top = display_top(unit_map)
            found = {
                "ecg_id": ecg_id,
                "patient_id": row["patient_id"],
                "map": name,
                "top_start": float(unit_map.starts[top]),
                "top_end": float(unit_map.ends[top]),
                "top_lead": int(unit_map.leads[top]),
                "score": float(unit_map.scores.max()),
            }
            if ecg_id in windows:
                supports = [(left + 0.04, left + 0.18) for left, _ in windows[ecg_id]]
                found.update(location_metrics(unit_map, supports))
                tops = tied_top(unit_map)
                centers = (unit_map.starts + unit_map.ends) / 2
                targets = np.array([left + 0.10 for left, _ in windows[ecg_id]])
                distance = np.abs(centers[tops, None] - targets[None]).min(axis=1).mean()
                overlap = np.zeros(len(centers), dtype=bool)
                for left, right in windows[ecg_id]:
                    overlap |= (unit_map.starts < right) & (unit_map.ends > left)
                owners = peaks[np.abs(centers[:, None] * FS - peaks[None]).argmin(axis=1)] / FS
                owns_target = np.isclose(owners[:, None], targets[None], atol=1 / FS).any(axis=1)
                found.update(
                    {
                        "legacy_hit": float(overlap[tops].mean()),
                        "legacy_chance": float(overlap.mean()),
                        "ownership_hit": float(owns_target[tops].mean()),
                        "ownership_chance": float(owns_target.mean()),
                        "distance_seconds": float(distance),
                    }
                )
            records.append(found)
        if ecg_id in EXAMPLES:
            figure = review_figure(signal, current, f"ECG {ecg_id}: exploratory morphology location")
            figure.savefig(output / "figures" / f"ECG_{ecg_id}.png", dpi=120)
        if position % 100 == 0:
            LOG.info("development %s/%s", position, len(rows["scored"]))
    table = pd.DataFrame(records)
    table.to_csv(output / "development_locations.csv", index=False)
    summary = _primary_summary(rows, windows, output, maps, kept_ids, table, excluded)
    _serialize_maps(output / "development_maps.npz", kept_ids, maps)
    paired_rows = {
        key: (
            value[value["ecg_id"].isin(kept_ids)]
            if isinstance(value, pd.DataFrame) and "ecg_id" in value
            else value
        )
        for key, value in rows.items()
    }
    descriptive = analyse(
        paired_rows,
        {"U_B": maps["U_B_fixed"], "raw_residual": maps["raw_residual"]},
        {i: found for i, found in windows.items() if i in kept_ids},
        False,
    )
    descriptive["maps"]["U_B_fixed"] = descriptive["maps"].pop("U_B")
    descriptive["thresholds"]["U_B_fixed"] = descriptive["thresholds"].pop("U_B")
    return {"primary": summary, "descriptive_042_rule": descriptive}, maps, table


def _synthetics(rows: dict[str, Any], output: Path) -> tuple[dict[str, Any], list[str]]:
    patients = set(rows["scored"]["patient_id"])
    normals = rows["fit"].sort_values("ecg_id").drop_duplicates("patient_id")
    normals = normals[~normals["patient_id"].isin(patients)]
    rng = np.random.default_rng(SEED)
    records, stems, figures, selected, skipped = [], [], set(), [], []
    for _, row in normals.iterrows():
        signal = read_ptb_float64(row["filename_hr"])
        peaks = r_peaks(signal, FS)
        kept = peaks[(peaks >= 125) & (peaks + 225 <= signal.shape[1])]
        if len(kept) < 5:
            skipped.append(int(row["ecg_id"]))
            continue
        peak = int(rng.choice(kept[1:-1]))
        st_leads = [int(rng.integers(0, 12))]
        qrs_leads = sorted(rng.choice(12, 3, replace=False).tolist())
        sample_mask = support_mask(signal.shape[1], peaks)
        control = fixed_window_map(residual_field(signal, peaks), sample_mask)
        control_score = float(control.scores.max())
        ecg_id = int(row["ecg_id"])
        selected.append(ecg_id)
        stems.append(str(row["filename_hr"]))
        for kind, leads in (("ST", st_leads), ("QRS", qrs_leads)):
            edited, supports = perturb(signal, peaks, peak, leads, kind)
            unit_map = fixed_window_map(residual_field(edited, peaks), sample_mask)
            found = location_metrics(unit_map, supports, leads)
            records.append(
                {
                    "ecg_id": ecg_id,
                    "patient_id": row["patient_id"],
                    "kind": kind,
                    "control_score": control_score,
                    "support_start": supports[0][0],
                    "support_end": supports[0][1],
                    "changed_leads": json.dumps(leads),
                    **found,
                }
            )
            status = "success" if found["joint_hit"] == 1 else "failure"
            figure_key = f"{kind}_{status}"
            if figure_key not in figures:
                review_figure(
                    edited,
                    {"raw_residual": unit_map},
                    f"Synthetic {kind}, ECG {ecg_id}: {status}; green = exact edit",
                    supports,
                ).savefig(output / "figures" / f"synthetic_{figure_key}_{ecg_id}.png", dpi=120)
                figures.add(figure_key)
        shifted_peak = int(kept[1:-1][np.flatnonzero(kept[1:-1] != peak)[0]])
        shifted, shift_support = perturb(signal, peaks, shifted_peak, st_leads, "ST")
        shifted_map = fixed_window_map(residual_field(shifted, peaks), sample_mask)
        records.append(
            {
                "ecg_id": ecg_id,
                "patient_id": row["patient_id"],
                "kind": "ST_time_shift",
                "control_score": control_score,
                **location_metrics(shifted_map, shift_support, st_leads),
            }
        )
        permutation = np.roll(np.arange(12), 1)
        permuted_leads = [int(np.flatnonzero(permutation == st_leads[0])[0])]
        permuted, permutation_support = perturb(signal[permutation], peaks, peak, permuted_leads, "ST")
        permuted_map = fixed_window_map(residual_field(permuted, peaks), sample_mask)
        records.append(
            {
                "ecg_id": ecg_id,
                "patient_id": row["patient_id"],
                "kind": "ST_lead_permutation",
                "control_score": control_score,
                **location_metrics(permuted_map, permutation_support, permuted_leads),
            }
        )
        seconds = np.arange(signal.shape[1]) / FS
        nuisance = signal + 0.15 + 0.05 * np.sin(2 * np.pi * 0.2 * seconds)[None]
        nuisance_score = float(fixed_window_map(residual_field(nuisance, peaks), sample_mask).scores.max())
        records.append(
            {
                "ecg_id": ecg_id,
                "patient_id": row["patient_id"],
                "kind": "nuisance",
                "control_score": control_score,
                "score": nuisance_score,
            }
        )
        if len(selected) >= 200:
            break
    table = pd.DataFrame(records)
    table.to_csv(output / "synthetic_locations.csv", index=False)
    summary = {
        "records": len(selected),
        "ecg_ids": selected,
        "patients_disjoint": True,
        "skipped_fewer_than_five_complete_beats": skipped,
    }
    for kind in ("ST", "QRS", "ST_time_shift", "ST_lead_permutation"):
        group = table[table["kind"] == kind]
        changes = group["score"] - group["control_score"]
        joint = _interval(group["patient_id"].to_numpy(), group["joint_hit"].to_numpy())
        score_change = _interval(group["patient_id"].to_numpy(), changes.to_numpy())
        summary[kind] = {
            "joint_localization": joint,
            "score_change": score_change,
            "median_distance_seconds": float(group["distance_seconds"].median()),
            "mean_overlap_fraction": float(group["overlap_fraction"].mean()),
            "passes": joint["value"] >= 0.80 and score_change["ci_low"] > 0,
        }
    nuisance = table[table["kind"] == "nuisance"]
    threshold = float(np.quantile(nuisance["control_score"], 0.95))
    summary["nuisance"] = {
        "in_sample_threshold": threshold,
        "unchanged_flag_share": float((nuisance["control_score"] > threshold).mean()),
        "nuisance_flag_share": float((nuisance["score"] > threshold).mean()),
        "score_change": _interval(
            nuisance["patient_id"].to_numpy(), (nuisance["score"] - nuisance["control_score"]).to_numpy()
        ),
    }
    return summary, stems


def main() -> None:
    """Execute the frozen protocol and write a provenance-backed local result.

    Returns
    -------
    None
        Creates the output directory only after successful integrity checks and analysis.
    """
    partial = OUTPUT.with_name(OUTPUT.name + ".partial")
    if OUTPUT.exists() or partial.exists():
        raise FileExistsError(OUTPUT)
    partial.mkdir()
    (partial / "figures").mkdir()
    logging.basicConfig(
        level=logging.INFO, handlers=[logging.FileHandler(partial / "run.log"), logging.StreamHandler()]
    )
    began = time.perf_counter()
    with threadpool_limits(limits=2):
        identity = _identity()
        LOG.info("reproducing full historical U_B before any new scores")
        rows, baseline, windows, integrity = load_baseline()
        LOG.info("full U_B metrics reproduced exactly")
        development, _, _ = _development(rows, baseline, windows, partial)
        synthetic, synthetic_stems = _synthetics(rows, partial)
        identity["waveforms"] = _waveform_hashes(
            sorted(set(rows["scored"]["filename_hr"].tolist() + synthetic_stems))
        )
        passes = (
            development["primary"]["passes_real_rule"]
            and synthetic["ST"]["passes"]
            and synthetic["QRS"]["passes"]
        )
        result = {
            "identity": identity,
            "integrity": integrity,
            "development": development,
            "synthetic": synthetic,
            "promote_for_review": bool(passes),
            "elapsed_seconds": time.perf_counter() - began,
        }
        result["outputs_sha256"] = {
            path.name: sha256_file(path)
            for path in partial.iterdir()
            if path.is_file() and path.name != "run.log"
        }
        write_json_atomic(partial / "result.json", result)
    partial.rename(OUTPUT)
    LOG.info("complete %s in %.1fs; promote=%s", to_stored(OUTPUT), result["elapsed_seconds"], passes)


if __name__ == "__main__":
    main()
