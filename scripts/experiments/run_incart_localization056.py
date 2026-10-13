"""Run the prespecified public expert beat-identity benchmark on CPU."""

from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wfdb
from threadpoolctl import threadpool_limits

from ecg_experiment import ROOT
from ecg_experiment.external_encoders import read_ptb_float64
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.fragment_localization import r_peaks
from ecg_experiment.incart_localization056 import (
    BEAT_SYMBOLS,
    METHODS,
    benchmark_summary,
    match_beats,
    patient_metrics,
    read_record,
    reproduce054,
    score_signal,
)
from ecg_experiment.lead_wave_maps import (
    WAVES,
    UnitMap,
    beat_pieces,
    beat_unit_map,
    fit_wave_references,
    wave_lengths,
    wave_scores,
)
from ecg_experiment.paths import to_stored
from ecg_experiment.provenance import git_head
from ecg_experiment.raw_residual052 import FS, review_figure

OUTPUT = ROOT / "outputs/experiment056_incart_localization_v1"
RAW = ROOT / "data/raw/incart/1.0.0"
PROTOCOL = "docs/experiment-056-incart-beat-localization.md"
MANIFEST = "docs/experiment-056-incart-records.json"
LOG = logging.getLogger("experiment056")


def _check_reference(rows: dict[str, Any], baseline: list, references: dict) -> tuple[float, float]:
    max_absolute, max_relative = 0.0, 0.0
    for position, (_index, row) in enumerate(rows["scored"].iterrows()):
        times, pieces = beat_pieces(read_ptb_float64(row["filename_hr"]), FS)
        rebuilt = beat_unit_map(wave_scores(references, pieces), times)
        saved = baseline[position]
        for field in ("leads", "starts", "ends"):
            if not np.array_equal(getattr(rebuilt, field), getattr(saved, field)):
                raise ValueError("Rebuilt U_B geometry differs")
        difference = np.abs(rebuilt.scores - saved.scores)
        max_absolute = max(max_absolute, float(difference.max()))
        max_relative = max(max_relative, float((difference / np.maximum(np.abs(saved.scores), 1e-12)).max()))
    if max_absolute > 1e-8 or max_relative > 1e-10:
        raise ValueError(f"Rebuilt U_B scores differ: {max_absolute}, {max_relative}")
    LOG.info("U_B reconstruction complete: max abs %.3g, relative %.3g", max_absolute, max_relative)
    return max_absolute, max_relative


def _fit_reference(rows: dict[str, Any], baseline: list, output: Path) -> tuple[dict, dict[str, float], dict]:
    expected = json.loads((ROOT / "outputs/experiment042_lead_wave_maps_v1/result.json").read_text())[
        "arm_b"
    ]["fit_beats"]
    paths = {name: output / f"temporary_{name}_pieces.npy" for name in WAVES}
    arrays = {
        name: np.lib.format.open_memmap(paths[name], mode="w+", dtype=np.float64, shape=(expected, 12, size))
        for name, size in wave_lengths(FS).items()
    }
    offset = 0
    for position, (_index, row) in enumerate(rows["fit"].iterrows()):
        times, pieces = beat_pieces(read_ptb_float64(row["filename_hr"]), FS)
        if len(times) < 2:
            continue
        for name in WAVES:
            arrays[name][offset : offset + len(times)] = pieces[name]
        offset += len(times)
        if position % 500 == 0:
            LOG.info("reference extraction %s/5872", position)
    if offset != expected:
        raise ValueError("Frozen normal reference beat count changed")
    references = fit_wave_references(arrays)
    del arrays
    for path in paths.values():
        path.unlink()
    with (output / "normal_reference.pkl").open("wb") as target:
        pickle.dump(references, target)
    max_absolute, max_relative = _check_reference(rows, baseline, references)
    normal_scores = []
    for position, (_index, row) in enumerate(rows["fit"].iterrows()):
        signal = read_ptb_float64(row["filename_hr"])
        values, _ = score_signal(signal, references)
        values["ecg_id"] = int(row["ecg_id"])
        normal_scores.append(values[["ecg_id", "r_sample", *METHODS]])
        if position % 500 == 0:
            LOG.info("normal beat thresholds %s/5872", position)
    normal = pd.concat(normal_scores, ignore_index=True)
    normal.to_csv(output / "normal_beat_scores.csv", index=False)
    thresholds = {name: float(normal[name].quantile(0.95)) for name in METHODS}
    return (
        references,
        thresholds,
        {
            "fit_ecgs": len(rows["fit"]),
            "fit_beats": expected,
            "calibration_scored_beats": len(normal),
            "max_score_abs_difference": max_absolute,
            "max_score_relative_difference": max_relative,
        },
    )


def _record_scores(
    signal: np.ndarray, references: dict, record: str, patient: int
) -> tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame]:
    frames, excluded = [], []
    nuisance_rows = []
    for core, start in enumerate(range(0, signal.shape[1], 5000)):
        stop = min(start + 5000, signal.shape[1])
        left, right = max(0, start - 250), min(signal.shape[1], stop + 250)
        context = signal[:, left:right]
        peaks = r_peaks(context, FS)
        if np.count_nonzero((peaks >= 125) & (peaks + 225 <= context.shape[1])) < 3:
            excluded.append(
                {
                    "record": record,
                    "patient": patient,
                    "core": core,
                    "reason": "fewer_than_three_complete_beats",
                }
            )
            continue
        values, _ = score_signal(context, references, peaks)
        if core == 0:
            seconds = np.arange(context.shape[1]) / FS
            changed = context + 0.15 + 0.05 * np.sin(2 * np.pi * 0.2 * seconds)[None]
            nuisance, _ = score_signal(changed, references, peaks)
            if not np.array_equal(values["r_sample"], nuisance["r_sample"]):
                raise ValueError("Nuisance altered inference anchors")
            for name in METHODS:
                nuisance_rows.extend(
                    {
                        "record": record,
                        "patient": patient,
                        "r_sample": int(peak),
                        "map": name,
                        "score": float(score),
                        "nuisance_score": float(changed_score),
                    }
                    for peak, score, changed_score in zip(
                        values["r_sample"], values[name], nuisance[name], strict=True
                    )
                )
        values["r_sample"] += left
        for name in METHODS:
            values[f"{name}_start"] += left / FS
        values = values[(values["r_sample"] >= start) & (values["r_sample"] < stop)].copy()
        values["core"] = core
        values["record"] = record
        values["patient"] = patient
        frames.append(values)
    return pd.concat(frames, ignore_index=True), excluded, pd.DataFrame(nuisance_rows)


def _annotate(detected: pd.DataFrame, record: str, patient: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    annotation = wfdb.rdann(str(RAW / record), "atr")
    symbols = np.asarray(annotation.symbol)
    beat = np.isin(symbols, list(BEAT_SYMBOLS))
    samples = annotation.sample[beat]
    symbols = symbols[beat]
    reference_times = samples / 257
    match = match_beats(detected["r_sample"].to_numpy() / FS, reference_times)
    detected = detected.copy()
    detected["reference_index"] = match
    detected["symbol"] = [symbols[index] if index >= 0 else "unmatched" for index in match]
    detected["match_distance_seconds"] = [
        abs(time - reference_times[index]) if index >= 0 else np.nan
        for time, index in zip(detected["r_sample"] / FS, match, strict=True)
    ]
    matched = np.zeros(len(samples), dtype=bool)
    matched[match[match >= 0]] = True
    reference = pd.DataFrame(
        {
            "record": record,
            "patient": patient,
            "reference_index": np.arange(len(samples)),
            "original_sample_257": samples,
            "time_seconds": reference_times,
            "core": np.floor(reference_times / 10).astype(int),
            "symbol": symbols,
            "matched": matched,
        }
    )
    counts = {
        "record": record,
        "patient": patient,
        "reference_beats": len(samples),
        "nonbeat_annotations": int((~beat).sum()),
        "scored_detections": len(detected),
        "matched": int(matched.sum()),
        "unmatched_reference": int((~matched).sum()),
        "unmatched_detections": int((match < 0).sum()),
    }
    return detected, reference, counts


def _chunk_metrics(detected: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    records = []
    for core, truth in reference.groupby("core"):
        if not {"N", "V"} <= set(truth["symbol"]):
            continue
        candidates = detected[detected["core"] == core]
        for name in METHODS:
            if not len(candidates):
                hit, chance = 0.0, 0.0
            else:
                score = candidates[name].to_numpy()
                top = np.isclose(score, score.max(), rtol=1e-10, atol=1e-10)
                ventricular = candidates["symbol"].to_numpy() == "V"
                hit, chance = float(ventricular[top].mean()), float(ventricular.mean())
            records.append(
                {
                    "record": str(truth.iloc[0]["record"]),
                    "patient": int(truth.iloc[0]["patient"]),
                    "core": int(core),
                    "map": name,
                    "hit": hit,
                    "chance": chance,
                    "excess": hit - chance,
                    "candidates": len(candidates),
                    "reference_v": int((truth["symbol"] == "V").sum()),
                }
            )
    return pd.DataFrame(records)


def _review(
    signal: np.ndarray,
    record: str,
    detected: pd.DataFrame,
    chunks: pd.DataFrame,
    references: dict,
    output: Path,
    chosen: set[str],
) -> None:
    if not len(chunks):
        return
    first = chunks[chunks["map"] == METHODS[0]].set_index("core")
    second = chunks[chunks["map"] == METHODS[1]].set_index("core").loc[first.index]
    difference = first["hit"] - second["hit"]
    for kind, mask in (("win", difference > 0), ("loss", difference < 0)):
        cores = first.index[mask].tolist()
        if not cores or kind in chosen:
            continue
        core = int(cores[0])
        start, stop = core * 5000, (core + 1) * 5000
        left, right = max(0, start - 250), min(signal.shape[1], stop + 250)
        context = signal[:, left:right]
        _, maps = score_signal(context, references)
        maps = _restrict_maps(maps, context, start - left, stop - left)
        ventricular = detected[(detected["core"] == core) & (detected["symbol"] == "V")]["r_sample"]
        supports = [((peak - left) / FS - 0.10, (peak - left) / FS + 0.10) for peak in ventricular]
        review_figure(
            context, maps, f"INCART {record} core {core}: {kind}; green = matched expert V", supports
        ).savefig(output / "figures" / f"{kind}_{record}_{core}.png", dpi=120)
        chosen.add(kind)


def _restrict_maps(maps: dict, signal: np.ndarray, start: int, stop: int) -> dict:
    peaks = r_peaks(signal, FS)
    complete = peaks[(peaks >= 125) & (peaks + 225 <= signal.shape[1])]
    restricted = {}
    for name, unit_map in maps.items():
        centers = (unit_map.starts + unit_map.ends) / 2 * FS
        owner = complete[np.abs(centers[:, None] - complete[None]).argmin(axis=1)]
        selected = (owner >= start) & (owner < stop)
        restricted[name] = UnitMap(
            unit_map.scores[selected],
            unit_map.leads[selected],
            unit_map.starts[selected],
            unit_map.ends[selected],
        )
    return restricted


def _review_errors(
    signal: np.ndarray,
    record: str,
    detected: pd.DataFrame,
    truth: pd.DataFrame,
    references: dict,
    thresholds: dict,
    output: Path,
    chosen: set[str],
) -> None:
    normal = detected[(detected["symbol"] == "N") & (detected[METHODS[0]] > thresholds[METHODS[0]])]
    if len(normal) and "normal_false_alarm" not in chosen:
        found = normal.iloc[0]
        left, right = (
            max(0, int(found["r_sample"]) - 2500),
            min(signal.shape[1], int(found["r_sample"]) + 2500),
        )
        marks = {}
        for name in METHODS:
            start = float(found[f"{name}_start"]) - left / FS
            marks[name] = UnitMap(
                np.array([found[name]]),
                np.array([int(found[f"{name}_lead"])]),
                np.array([start]),
                np.array([start + 0.14]),
            )
        review_figure(
            signal[:, left:right], marks, f"INCART {record}: first expert-N beat above candidate threshold"
        ).savefig(output / "figures" / f"normal_false_alarm_{record}.png", dpi=120)
        chosen.add("normal_false_alarm")
    missed = truth[(truth["symbol"] == "V") & ~truth["matched"]]
    if not len(missed) or "unmatched_expert_v" in chosen:
        return
    found = missed.iloc[0]
    center = round(float(found["time_seconds"]) * FS)
    left, right = max(0, center - 2500), min(signal.shape[1], center + 2500)
    context = signal[:, left:right]
    peaks = r_peaks(context, FS)
    if np.count_nonzero((peaks >= 125) & (peaks + 225 <= context.shape[1])) < 3:
        return
    _, maps = score_signal(context, references)
    target = float(found["time_seconds"]) - left / FS
    review_figure(
        context,
        maps,
        f"INCART {record}: unmatched expert V; annotation timing is approximate",
        [(target - 0.15, target + 0.15)],
    ).savefig(output / "figures" / f"unmatched_expert_v_{record}.png", dpi=120)
    chosen.add("unmatched_expert_v")


def _nuisance_summary(table: pd.DataFrame, thresholds: dict[str, float]) -> dict:
    result = {}
    for name in METHODS:
        selected = table[table["map"] == name]
        ordinary = float((selected["score"] > thresholds[name]).mean())
        changed = float((selected["nuisance_score"] > thresholds[name]).mean())
        rank = selected.groupby("record")[["score", "nuisance_score"]].corr(method="spearman")
        result[name] = {
            "unchanged_flag_share": ordinary,
            "nuisance_flag_share": changed,
            "increase": changed - ordinary,
            "mean_score_change": float((selected["nuisance_score"] - selected["score"]).mean()),
            "mean_record_spearman": float(rank.xs("score", level=1)["nuisance_score"].mean()),
        }
    return result


def _identity(rows: dict[str, Any]) -> dict:
    manifest = json.loads((ROOT / MANIFEST).read_text())
    sources = (
        PROTOCOL,
        MANIFEST,
        "ecg_experiment/incart_localization056.py",
        "scripts/experiments/run_incart_localization056.py",
        "ecg_experiment/raw_residual054.py",
        "ecg_experiment/raw_residual052.py",
        "ecg_experiment/localization_inputs052.py",
        "ecg_experiment/lead_wave_maps.py",
        "ecg_experiment/fragment_localization.py",
        "ecg_experiment/intervals.py",
        "ecg_experiment/paths.py",
        "ecg_experiment/resample.py",
        "ecg_experiment/external_encoders.py",
        "pyproject.toml",
        "uv.lock",
    )
    inputs = {
        to_stored(RAW / f"{row['record']}.{suffix}"): sha256_file(RAW / f"{row['record']}.{suffix}")
        for row in manifest["records"]
        for suffix in ("hea", "dat", "atr")
    }
    if any(
        inputs[to_stored(RAW / f"{row['record']}.hea")] != row["header_sha256"] for row in manifest["records"]
    ):
        raise ValueError("Audited INCART header changed")
    challenge = ROOT / manifest["challenge_overlap_audit"]["manifest"]
    if sha256_file(challenge) != manifest["challenge_overlap_audit"]["sha256"]:
        raise ValueError("Closed-source overlap audit changed")
    ptb = ROOT / "data/raw/ptb-xl/1.0.3"
    inputs.update(
        {
            to_stored(ptb / f"{stem}.{suffix}"): sha256_file(ptb / f"{stem}.{suffix}")
            for stem in rows["fit"]["filename_hr"]
            for suffix in ("hea", "dat")
        }
    )
    return {
        "sources": {name: sha256_file(ROOT / name) for name in sources},
        "inputs": inputs,
        "git_head": git_head(ROOT),
        "seed": 56056,
        "threads": 2,
        "download_source": "https://physionet-open.s3.amazonaws.com/incartdb/1.0.0/",
        "closed_source_overlap_audit": manifest["challenge_overlap_audit"],
    }


def main() -> None:
    """Execute the frozen independent benchmark after all predecessor integrity checks.

    Returns
    -------
    None
        Writes one local receipt-backed benchmark and review package.
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
        LOG.info("reproducing 054/052/full U_B before any benchmark scores")
        rows, baseline, integrity = reproduce054()
        identity = _identity(rows)
        references, thresholds, reconstruction = _fit_reference(rows, baseline, partial)
        manifest = json.loads((ROOT / MANIFEST).read_text())
        all_detected, all_reference, all_chunks, all_nuisance, coverage, excluded, chosen = (
            [],
            [],
            [],
            [],
            [],
            [],
            set(),
        )
        for entry in manifest["records"]:
            record = entry["record"]
            signal, patient = read_record(RAW / record)
            if patient != entry["patient"]:
                raise ValueError("Patient identity changed")
            LOG.info("scoring %s patient %s before reading its expert annotations", record, patient)
            detected, skipped, nuisance = _record_scores(signal, references, record, patient)
            detected, truth, counts = _annotate(detected, record, patient)
            chunks = _chunk_metrics(detected, truth)
            _review(signal, record, detected, chunks, references, partial, chosen)
            _review_errors(signal, record, detected, truth, references, thresholds, partial, chosen)
            all_detected.append(detected)
            all_reference.append(truth)
            all_chunks.append(chunks)
            all_nuisance.append(nuisance)
            coverage.append(counts)
            excluded.extend(skipped)
        beats = pd.concat(all_detected, ignore_index=True)
        truth = pd.concat(all_reference, ignore_index=True)
        chunks = pd.concat(all_chunks, ignore_index=True)
        nuisance_table = pd.concat(all_nuisance, ignore_index=True)
        patients = patient_metrics(beats, chunks, truth, thresholds)
        for name, table in (
            ("detected_beats", beats),
            ("reference_beats", truth),
            ("mixed_cores", chunks),
            ("patient_metrics", patients),
            ("nuisance", nuisance_table),
            ("coverage", pd.DataFrame(coverage)),
        ):
            table.to_csv(partial / f"{name}.csv", index=False)
        summary = benchmark_summary(patients)
        nuisance = _nuisance_summary(nuisance_table, thresholds)
        summary["gates"]["nuisance"] = nuisance[METHODS[0]]["increase"] <= 0.05
        result = {
            "identity": identity,
            "integrity": integrity,
            "U_B_reconstruction": reconstruction,
            "thresholds": thresholds,
            "benchmark": summary,
            "nuisance": nuisance,
            "coverage": {
                "records": len(coverage),
                "patients": int(truth["patient"].nunique()),
                "reference_beats": len(truth),
                "scored_detections": len(beats),
                "reference_types": truth["symbol"].value_counts().to_dict(),
                "matched_types": beats["symbol"].value_counts().to_dict(),
                "excluded_cores": excluded,
            },
            "promote_for_review": bool(all(summary["gates"].values())),
            "elapsed_seconds": time.perf_counter() - began,
        }
        result["outputs_sha256"] = {
            path.name: sha256_file(path)
            for path in partial.iterdir()
            if path.is_file() and path.name != "run.log"
        }
        write_json_atomic(partial / "result.json", result)
    partial.rename(OUTPUT)
    LOG.info("complete %.1fs; promote=%s", result["elapsed_seconds"], result["promote_for_review"])


if __name__ == "__main__":
    main()
