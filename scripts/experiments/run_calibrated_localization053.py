"""Execute frozen Experiment 053 with patient-disjoint training-normal calibration."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from ecg_experiment import ROOT
from ecg_experiment.calibrated_localization import (
    calibrated_scores,
    chi_square_scores,
    equal_width_pieces,
    mean_interval,
    region_difference,
    tied_lead_membership,
    tied_premature_hit,
)
from ecg_experiment.eda.ptbxl import load_metadata, load_statements
from ecg_experiment.external_encoders import read_ptb_float64
from ecg_experiment.files import sha256_file, write_json_atomic, write_npz_atomic
from ecg_experiment.fragment_localization import premature_windows, r_peaks
from ecg_experiment.full_development import cohorts, ptb_table
from ecg_experiment.intervals import paired_auroc_difference
from ecg_experiment.lead_wave_maps import (
    ANTERIOR,
    INFERIOR,
    WAVES,
    UnitMap,
    beat_pieces,
    beat_unit_map,
    ecg_score,
    fit_wave_references,
    lead_wave_unit_map,
    plot_lead_marks,
    premature_hit,
    red_marks,
    top_lead,
    wave_lengths,
    wave_scores,
)
from ecg_experiment.paths import to_stored
from ecg_experiment.provenance import git_head
from ecg_experiment.waveforms import LEADS

OUTPUT = ROOT / "outputs/experiment053_calibrated_units_v1"
PRIOR = ROOT / "outputs/experiment042_lead_wave_maps_v1"
PROTOCOL = ROOT / "docs/experiment-053-calibrated-units.md"
CACHE_PATHS = (
    ROOT / "data/processed/pretrained/ecg-jepa-full-public/ecg_ids.npy",
    ROOT / "outputs/experiment016_xecg_probe_finetune/features/ecg_ids.npy",
    ROOT / "outputs/experiment004_cpc_40k/released_features/ecg_ids.npy",
)
SEED = 53053
LOG = logging.getLogger(__name__)


def selected_rows() -> dict[str, pd.DataFrame]:
    """
    Select unchanged 042 training-normal and development rows from metadata.

    Returns
    -------
    dict[str, pd.DataFrame]
        Frozen predecessor cohorts and location proxy sets.
    """
    groups = cohorts(ptb_table())
    common = set.intersection(*(set(np.load(path).astype(int)) for path in CACHE_PATHS))
    fit = groups["train"].query("standard == 0")
    fit = fit[fit["ecg_id"].isin(common)]
    development = groups["development"]
    evaluation = development[
        development["original"] & development["standard"].notna() & development["ecg_id"].isin(common)
    ]
    statements, metadata = load_statements(), load_metadata()
    diagnostic = statements[statements["diagnostic"] == 1]
    subclass = diagnostic["diagnostic_subclass"].to_dict()
    superclass = diagnostic["diagnostic_class"].to_dict()
    codes = metadata["scp_codes"].apply(set)
    subclasses = codes.map(lambda found: {subclass[c] for c in found if c in subclass})
    superclasses = codes.map(lambda found: {superclass[c] for c in found if c in superclass})
    anterior = set(
        subclasses.index[subclasses.map(lambda found: "AMI" in found and not found & {"IMI", "LMI", "PMI"})]
    )
    inferior = set(
        subclasses.index[subclasses.map(lambda found: "IMI" in found and not found & {"AMI", "LMI", "PMI"})]
    )
    benign = {
        i
        for i, found in codes.items()
        if superclasses[i] == {"NORM"}
        and found <= {"NORM", "SR", "SBRAD", "SARRH"}
        and found & {"SBRAD", "SARRH"}
    }
    pvc = set(codes.index[codes.map(lambda found: "PVC" in found)])
    rows = {
        "fit": fit,
        "development": development,
        "evaluation": evaluation,
        "anterior": development[development["ecg_id"].isin(anterior)],
        "inferior": development[development["ecg_id"].isin(inferior)],
        "benign": development[development["ecg_id"].isin(benign)],
        "pvc": development[development["ecg_id"].isin(pvc)],
    }
    expected = {
        "fit": 5872,
        "development": 1604,
        "evaluation": 1306,
        "anterior": 146,
        "inferior": 149,
        "benign": 52,
        "pvc": 84,
    }
    if {name: len(frame) for name, frame in rows.items()} != expected:
        raise ValueError("Metadata cohorts differ from frozen predecessor")
    return rows


def partition(fit: pd.DataFrame, development: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Split normal training patients before covariance and tail fitting.

    Parameters
    ----------
    fit : pd.DataFrame
        Normal training ECG metadata.
    development : pd.DataFrame
        Development metadata, used only to assert disjoint patients.

    Returns
    -------
    dict[str, pd.DataFrame]
        Reference, channel calibration and independent normal control records.
    """
    patients = np.unique(fit["patient_id"])
    patients = np.random.default_rng(SEED).permutation(patients)
    first, second = int(len(patients) * 0.7), int(len(patients) * 0.2)
    parts = dict(
        zip(
            ("reference", "calibration", "control"),
            (patients[:first], patients[first : first + second], patients[first + second :]),
            strict=True,
        )
    )
    if set(patients) & set(development["patient_id"]):
        raise ValueError("Training and development patients overlap")
    return {name: fit[fit["patient_id"].isin(ids)] for name, ids in parts.items()}


def read_beats(stem: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Read local PTB-XL raw waveforms and apply frozen 042 beat extraction.

    Parameters
    ----------
    stem : str
        Repository-local PTB-XL waveform stem.

    Returns
    -------
    tuple[np.ndarray, dict[str, np.ndarray]]
        Complete R times and fixed-offset raw pieces.
    """
    return beat_pieces(read_ptb_float64(stem), 500)


def concatenate(beats: dict[int, tuple], frame: pd.DataFrame, equal: bool = False) -> dict[str, np.ndarray]:
    """
    Concatenate fit pieces with an optional common-width coordinate control.

    Parameters
    ----------
    beats : dict[int, tuple]
        Extracted local beat pieces.
    frame : pd.DataFrame
        Training records to concatenate.
    equal : bool
        Use the common 35-point block control.

    Returns
    -------
    dict[str, np.ndarray]
        Concatenated channel pieces.
    """
    found = [equal_width_pieces(beats[i][1]) if equal else beats[i][1] for i in frame["ecg_id"]]
    return {name: np.concatenate([piece[name] for piece in found]) for name in WAVES}


def reconstruct(beats: dict[int, tuple], rows: dict[str, pd.DataFrame]) -> tuple[list[UnitMap], dict]:
    """
    Reconstruct every cached U_B unit before computing candidate scores.

    Parameters
    ----------
    beats : dict[int, tuple]
        Extracted local beat pieces.
    rows : dict[str, pd.DataFrame]
        Frozen predecessor rows and normal control.

    Returns
    -------
    tuple[list[UnitMap], dict]
        Exact baseline maps and integrity measurements.
    """
    reference = fit_wave_references(concatenate(beats, rows["fit"]))
    maps = [
        beat_unit_map(wave_scores(reference, beats[i][1]), beats[i][0]) for i in rows["development"]["ecg_id"]
    ]
    prior = json.loads((PRIOR / "result.json").read_text())
    if sha256_file(PRIOR / "unit_scores.npz") != prior["outputs_sha256"]["unit_scores.npz"]:
        raise ValueError("Predecessor score archive hash differs from receipt")
    with np.load(PRIOR / "unit_scores.npz") as saved:
        if not np.array_equal(saved["ecg_ids"], rows["development"]["ecg_id"].to_numpy()):
            raise ValueError("Predecessor scored order changed")
        scores = np.concatenate([unit.scores for unit in maps])
        differences = np.abs(scores - saved["U_B_scores"])
        relative = float(np.max(differences / np.maximum(1, np.abs(saved["U_B_scores"]))))
        if relative >= 1e-8:
            raise ValueError(f"Predecessor unit reproduction failed: {relative}")
        offsets = saved["U_B_offsets"]
        for index, unit in enumerate(maps):
            old = saved["U_B_scores"][offsets[index] : offsets[index + 1]]
            if unit.leads[unit.scores.argmax()] != saved["U_B_leads"][offsets[index] + old.argmax()]:
                raise ValueError("Exact legacy top lead differs")
        for field in ("leads", "starts", "ends"):
            if not np.array_equal(
                np.concatenate([getattr(unit, field) for unit in maps]), saved[f"U_B_{field}"]
            ):
                raise ValueError(f"Predecessor unit support differs: {field}")
    control = [
        ecg_score(beat_unit_map(wave_scores(reference, beats[i][1]), beats[i][0]))
        for i in rows["control"]["ecg_id"]
    ]
    return maps, {
        "max_relative_unit_difference": relative,
        "max_absolute_unit_difference": float(differences.max()),
        "control_threshold_in_sample": float(np.quantile(control, 0.95)),
    }


def candidate_maps(
    beats: dict[int, tuple], rows: dict[str, pd.DataFrame], equal: bool
) -> tuple[dict[str, list[UnitMap]], dict[str, float], dict]:
    """
    Fit candidate maps using reference and calibration training patients only.

    Parameters
    ----------
    beats : dict[int, tuple]
        Extracted local beat pieces.
    rows : dict[str, pd.DataFrame]
        Disjoint reference, calibration, control and development cohorts.
    equal : bool
        Evaluate common coordinate counts.

    Returns
    -------
    tuple[dict[str, list[UnitMap]], dict[str, float], dict]
        Candidate maps, independent thresholds and fit diagnostics.
    """
    pieces = concatenate(beats, rows["reference"], equal)
    references = fit_wave_references(pieces)
    calibration = np.sort(wave_scores(references, concatenate(beats, rows["calibration"], equal)), axis=0)
    prefix = "equal35" if equal else "calibrated"
    names = [f"{prefix}_focal", f"{prefix}_persistent"]
    if not equal:
        names.extend(["chi_square_persistent", "raw_distance_persistent"])
    output = {name: [] for name in names}
    maxima = {name: [] for name in names}
    for group in ("control", "development"):
        for ecg_id in rows[group]["ecg_id"]:
            times, raw = beats[ecg_id]
            raw = equal_width_pieces(raw) if equal else raw
            distances = wave_scores(references, raw)
            transformed = calibrated_scores(calibration, distances, presorted=True)
            units = {
                names[0]: beat_unit_map(transformed, times),
                names[1]: lead_wave_unit_map(np.median(transformed, axis=0)),
            }
            if not equal:
                dimensions = np.array(list(wave_lengths().values()))
                units[names[2]] = lead_wave_unit_map(
                    np.median(chi_square_scores(distances, dimensions), axis=0)
                )
                units[names[3]] = lead_wave_unit_map(np.median(distances, axis=0))
            for name, unit in units.items():
                if group == "control":
                    maxima[name].append(ecg_score(unit))
                else:
                    output[name].append(unit)
    thresholds = {name: float(np.quantile(values, 0.95)) for name, values in maxima.items()}
    floor = np.log(len(calibration) + 1)
    saturation = {
        name: float(np.mean([np.mean(unit.scores >= floor - 1e-12) for unit in units]))
        for name, units in output.items()
        if name.startswith(prefix)
    }
    control_block_winners = {name: [0, 0, 0, 0] for name in names}
    for ecg_id in rows["control"]["ecg_id"]:
        distances = wave_scores(
            references, equal_width_pieces(beats[ecg_id][1]) if equal else beats[ecg_id][1]
        )
        calibrated = calibrated_scores(calibration, distances, presorted=True)
        channel_scores = np.median(calibrated, axis=0)
        selected = np.abs(channel_scores - channel_scores.max()) <= 1e-12
        weights = selected.sum(axis=0) / selected.sum()
        for block in range(4):
            control_block_winners[names[1]][block] += float(weights[block]) / len(rows["control"])
    return (
        output,
        thresholds,
        {
            "calibration_beats": len(calibration),
            "development_unit_saturation": saturation,
            "control_persistent_top_block_shares": control_block_winners[names[1]],
            "reference_beats": len(next(iter(pieces.values()))),
            "control_any_mark": {
                name: float(np.mean(np.array(values) > thresholds[name])) for name, values in maxima.items()
            },
        },
    )


def region_metrics(
    rows: dict[str, pd.DataFrame], top: dict[int, UnitMap], baseline: dict[int, UnitMap]
) -> dict:
    """
    Report both location contrasts and the paired symmetric primary endpoint.

    Parameters
    ----------
    rows : dict[str, pd.DataFrame]
        Frozen location proxy cohorts.
    top, baseline : dict[int, UnitMap]
        Candidate and baseline units by ECG identifier.

    Returns
    -------
    dict
        Regional and paired symmetric contrast intervals.
    """
    result = {}
    region_hits = {}
    for name, leads in (("anterior", ANTERIOR), ("inferior", INFERIOR)):
        region_hits[name] = {
            group: np.array(
                [
                    tied_lead_membership(
                        top[i].scores, top[i].leads, np.array([LEADS.index(lead) for lead in leads])
                    )
                    for i in rows[group]["ecg_id"]
                ]
            )
            for group in ("anterior", "inferior")
        }
        first, second = ("anterior", "inferior") if name == "anterior" else ("inferior", "anterior")
        result[name] = region_difference(
            rows[first]["patient_id"].to_numpy(),
            region_hits[name][first],
            rows[second]["patient_id"].to_numpy(),
            region_hits[name][second],
        )
    values = {
        group: (region_hits["anterior"][group] - region_hits["inferior"][group]) / 2
        for group in ("anterior", "inferior")
    }
    previous = {
        group: np.array(
            [
                (
                    tied_lead_membership(
                        baseline[i].scores,
                        baseline[i].leads,
                        np.array([LEADS.index(lead) for lead in ANTERIOR]),
                    )
                    - tied_lead_membership(
                        baseline[i].scores,
                        baseline[i].leads,
                        np.array([LEADS.index(lead) for lead in INFERIOR]),
                    )
                )
                / 2
                for i in rows[group]["ecg_id"]
            ]
        )
        for group in values
    }
    a, b = rows["anterior"]["patient_id"].to_numpy(), rows["inferior"]["patient_id"].to_numpy()
    result["symmetric"] = region_difference(a, values["anterior"], b, values["inferior"])
    result["symmetric_gain"] = region_difference(
        a, values["anterior"] - previous["anterior"], b, values["inferior"] - previous["inferior"]
    )
    return result


def analyse(
    rows: dict[str, pd.DataFrame],
    maps: dict[str, list[UnitMap]],
    thresholds: dict[str, float],
    windows: dict[int, list[tuple[float, float]]],
) -> dict:
    """
    Compute all fixed metrics without fitting on development normal rows.

    Parameters
    ----------
    rows : dict[str, pd.DataFrame]
        Frozen metadata cohorts.
    maps : dict[str, list[UnitMap]]
        Development maps in frozen order.
    thresholds : dict[str, float]
        Fixed candidate control and baseline continuity thresholds.
    windows : dict[int, list[tuple[float, float]]]
        Frozen automatic premature-beat proxy windows.

    Returns
    -------
    dict
        Executed primary, secondary and guardrail measurements.
    """
    ids = rows["development"]["ecg_id"].to_numpy()
    evaluation = rows["evaluation"]
    y = evaluation["standard"].to_numpy(int)
    eval_ids = evaluation["ecg_id"].to_numpy()
    patients = evaluation["patient_id"].to_numpy()
    baseline = dict(zip(ids, maps["U_B"], strict=True))
    base_scores = dict(zip(ids, map(ecg_score, maps["U_B"]), strict=True))
    base = np.array([base_scores[i] for i in eval_ids])
    result = {}
    pvc_ids = np.array(list(windows))
    pvc_patients = rows["development"].set_index("ecg_id").loc[pvc_ids, "patient_id"].to_numpy()
    base_units = dict(zip(ids, maps["U_B"], strict=True))
    base_pvc = np.array(
        [
            tied_premature_hit(base_units[i].scores, base_units[i].starts, base_units[i].ends, windows[i])
            for i in pvc_ids
        ]
    )
    benign_ids = rows["benign"]["ecg_id"].to_numpy()
    baseline_benign = np.array([ecg_score(base_units[i]) > thresholds["U_B"] for i in benign_ids], float)
    for name, units in maps.items():
        by_id = dict(zip(ids, units, strict=True))
        top = by_id
        scores = np.array([ecg_score(by_id[i]) for i in eval_ids])
        normal_ids = eval_ids[y == 0]
        groups = {"normal": normal_ids, "positive": eval_ids[y == 1], "benign": benign_ids}
        any_mark = {
            group: float(np.mean([ecg_score(by_id[i]) > thresholds[name] for i in chosen]))
            for group, chosen in groups.items()
        }
        result[name] = {
            "auroc": float(roc_auc_score(y, scores)),
            "threshold": thresholds[name],
            "top_lead_tie_rate": float(
                np.mean(
                    [
                        len(np.unique(unit.leads[np.abs(unit.scores - unit.scores.max()) <= 1e-12])) > 1
                        for unit in units
                    ]
                )
            ),
            "any_mark": any_mark,
            "lead": region_metrics(rows, top, baseline),
            "auroc_gain": paired_auroc_difference(patients, y, scores, base, seed=SEED),
            "marked_unit_fraction": {
                group: float(np.mean([np.mean(by_id[i].scores > thresholds[name]) for i in chosen]))
                for group, chosen in groups.items()
            },
            "benign_gain": mean_interval(
                rows["benign"]["patient_id"].to_numpy(),
                np.array([ecg_score(by_id[i]) > thresholds[name] for i in benign_ids], float)
                - baseline_benign,
            ),
        }
        if name.endswith("focal") or name == "U_B":
            pairs = np.array(
                [
                    tied_premature_hit(by_id[i].scores, by_id[i].starts, by_id[i].ends, windows[i])
                    for i in pvc_ids
                ]
            )
            result[name]["premature"] = {
                "records": len(pvc_ids),
                "hit": float(pairs[:, 0].mean()),
                "chance": float(pairs[:, 1].mean()),
                "excess": mean_interval(pvc_patients, pairs[:, 0] - pairs[:, 1]),
                "excess_gain": mean_interval(
                    pvc_patients, pairs[:, 0] - pairs[:, 1] - base_pvc[:, 0] + base_pvc[:, 1]
                ),
            }
    gain = result["calibrated_persistent"]["lead"]["symmetric_gain"]
    candidate = result["calibrated_persistent"]
    result["decision"] = {
        "primary_pass": bool(
            gain["estimate"] >= 0.10
            and gain["ci_low"] > 0
            and candidate["lead"]["anterior"]["estimate"] > 0
            and candidate["lead"]["inferior"]["estimate"] > 0
        ),
        "benign_guardrail": candidate["benign_gain"]["estimate"] <= 0.05,
        "focal_guardrail": result["calibrated_focal"]["premature"]["excess_gain"]["ci_low"] > -0.05,
    }
    result["decision"]["worth_reviewing"] = all(result["decision"].values())
    return result


def save_artifacts(rows: dict, maps: dict, thresholds: dict, beats: dict) -> dict:
    """
    Save all unit scores and deterministic local waveform review examples.

    Parameters
    ----------
    rows, maps, thresholds, beats : dict
        Metadata, maps, fixed thresholds and extracted beats.

    Returns
    -------
    dict
        Fixed example names and identifiers.
    """
    arrays = {"ecg_ids": rows["development"]["ecg_id"].to_numpy()}
    for name, units in maps.items():
        arrays[f"{name}_offsets"] = np.r_[0, np.cumsum([len(unit.scores) for unit in units])]
        for field in ("scores", "leads", "starts", "ends"):
            arrays[f"{name}_{field}"] = np.concatenate([getattr(unit, field) for unit in units])
    write_npz_atomic(OUTPUT / "unit_scores.npz", **arrays)
    examples = json.loads((PRIOR / "result.json").read_text())["examples"]
    frame = rows["development"].set_index("ecg_id")
    positions = dict(zip(frame.index, range(len(frame)), strict=True))
    folder = OUTPUT / "figures"
    folder.mkdir(exist_ok=True)
    for name, ecg_id in examples.items():
        selected = ("U_B", "calibrated_focal", "calibrated_persistent")
        marks = {key: red_marks(maps[key][positions[ecg_id]], thresholds[key]) for key in selected[:2]}
        persistent = maps["calibrated_persistent"][positions[ecg_id]]
        times = beats[ecg_id][0]
        offsets = np.array(list(WAVES.values()))
        indices = np.flatnonzero(persistent.scores > thresholds["calibrated_persistent"])
        marks["calibrated_persistent (repeated block support)"] = [
            (
                int(persistent.leads[index]),
                float(time + offsets[index % 4, 0]),
                float(time + offsets[index % 4, 1]),
            )
            for index in indices
            for time in times
        ]
        figure = plot_lead_marks(
            read_ptb_float64(frame.loc[ecg_id, "filename_hr"]),
            500,
            marks,
            f"{name}: ECG {ecg_id}; fixed-offset units",
        )
        figure.savefig(folder / f"{name}_{ecg_id}.png", dpi=110)
        figure.clear()
    return examples


def verify_legacy_baseline(rows: dict, baseline: list[UnitMap], prior: dict, windows: dict) -> None:
    """
    Verify frozen predecessor summaries before any candidate score.

    Parameters
    ----------
    rows : dict
        Predecessor metadata cohorts.
    baseline : list[UnitMap]
        Reconstructed predecessor unit maps.
    prior : dict
        Original predecessor receipt.
    windows : dict
        Frozen premature-beat proxy windows.

    Returns
    -------
    None
        Raises if any predecessor measurement differs.
    """
    by_id = dict(zip(rows["development"]["ecg_id"], baseline, strict=True))
    frame = rows["evaluation"]
    score = np.array([ecg_score(by_id[i]) for i in frame["ecg_id"]])
    old_metrics = prior["maps"]["U_B"]
    if roc_auc_score(frame["standard"].to_numpy(int), score) != old_metrics["auroc"]:
        raise ValueError("Pre-score baseline AUROC reproduction failed")
    pairs = np.array([premature_hit(by_id[i], windows[i]) for i in windows])
    for observed, expected in (
        (pairs[:, 0].mean(), old_metrics["hit_rate"]),
        (pairs[:, 1].mean(), old_metrics["chance_rate"]),
        ((pairs[:, 0] - pairs[:, 1]).mean(), old_metrics["hit_minus_chance"]["value"]),
    ):
        if abs(float(observed) - expected) > 1e-15:
            raise ValueError("Pre-score baseline premature reproduction failed")
    for region, leads, first, second in (
        ("anterior", ANTERIOR, "anterior", "inferior"),
        ("inferior", INFERIOR, "inferior", "anterior"),
    ):
        a = np.mean([top_lead(by_id[i]) in leads for i in rows[first]["ecg_id"]])
        b = np.mean([top_lead(by_id[i]) in leads for i in rows[second]["ecg_id"]])
        if abs(a - b - old_metrics["lead_contrast"][region]["value"]) > 1e-15:
            raise ValueError("Pre-score baseline lead reproduction failed")


def run() -> None:
    """
    Run the frozen experiment on local raw ECGs and write an auditable receipt.

    Returns
    -------
    None
        Writes the complete local experiment artifacts.
    """
    if OUTPUT.exists():
        raise FileExistsError(f"Preserve existing experiment artifacts: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(OUTPUT / "run.log")],
    )
    start = time.perf_counter()
    rows = selected_rows()
    if set(rows["anterior"]["patient_id"]) & set(rows["inferior"]["patient_id"]):
        raise ValueError("Infarct groups have overlapping patients")
    rows.update(partition(rows["fit"], rows["development"]))
    LOG.info(
        "Selected rows and patient-disjoint normal partition: %s",
        {name: (len(frame), frame["patient_id"].nunique()) for name, frame in rows.items()},
    )
    everyone = pd.concat([rows["fit"], rows["development"]])
    with ThreadPoolExecutor(max_workers=2) as executor:
        found = list(executor.map(read_beats, everyone["filename_hr"]))
    beats = dict(zip(everyone["ecg_id"], found, strict=True))
    if any(len(times) < 2 for times, _ in beats.values()):
        raise ValueError("Unexpected unusable ECG differs from predecessor")
    LOG.info("Read %s raw ECGs in %.1fs", len(beats), time.perf_counter() - start)
    prior = json.loads((PRIOR / "result.json").read_text())
    windows = {}
    for ecg_id, stem in zip(rows["pvc"]["ecg_id"], rows["pvc"]["filename_hr"], strict=True):
        peaks = r_peaks(read_ptb_float64(stem), 500)
        selected = premature_windows(peaks, 500)
        if len(peaks) >= 4 and selected:
            windows[int(ecg_id)] = selected
    if len(windows) != 73:
        raise ValueError("Eligible premature-beat proxy count differs from 042")
    with threadpool_limits(limits=2):
        baseline, integrity = reconstruct(beats, rows)
        verify_legacy_baseline(rows, baseline, prior, windows)
        LOG.info("Reconstructed exact U_B: %s", integrity)
        candidate, thresholds, fit_receipt = candidate_maps(beats, rows, False)
        LOG.info("Fitted fixed primary and chi-square control")
        equal, equal_thresholds, equal_receipt = candidate_maps(beats, rows, True)
        maps = {"U_B": baseline, **candidate, **equal}
        thresholds.update(equal_thresholds)
        prior = json.loads((PRIOR / "result.json").read_text())
        thresholds["U_B"] = prior["thresholds"]["U_B"]
        LOG.info("Scoring fixed endpoints for %s maps, PVC eligible %s", len(maps), len(windows))
        metrics = analyse(rows, maps, thresholds, windows)
    predecessor = prior["maps"]["U_B"]
    legacy = {}
    by_id = dict(zip(rows["development"]["ecg_id"], baseline, strict=True))
    for name, leads, first, second in (
        ("anterior", ANTERIOR, "anterior", "inferior"),
        ("inferior", INFERIOR, "inferior", "anterior"),
    ):
        a = np.array([float(top_lead(by_id[i]) in leads) for i in rows[first]["ecg_id"]])
        b = np.array([float(top_lead(by_id[i]) in leads) for i in rows[second]["ecg_id"]])
        legacy[name] = region_difference(
            rows[first]["patient_id"].to_numpy(), a, rows[second]["patient_id"].to_numpy(), b, seed=42042
        )
        if abs(legacy[name]["estimate"] - predecessor["lead_contrast"][name]["value"]) > 1e-15:
            raise ValueError("Legacy lead contrast reproduction failed")
    integrity["legacy_lead_contrasts"] = legacy
    integrity["premature_chance_absolute_difference"] = abs(
        metrics["U_B"]["premature"]["chance"] - predecessor["chance_rate"]
    )
    integrity["premature_excess_absolute_difference"] = abs(
        metrics["U_B"]["premature"]["excess"]["estimate"] - predecessor["hit_minus_chance"]["value"]
    )
    integrity.update(
        {
            "auroc_absolute_difference": abs(metrics["U_B"]["auroc"] - predecessor["auroc"]),
            "premature_hit_absolute_difference": abs(
                metrics["U_B"]["premature"]["hit"] - predecessor["hit_rate"]
            ),
        }
    )
    if any(
        integrity[name] > 1e-15
        for name in (
            "auroc_absolute_difference",
            "premature_hit_absolute_difference",
            "premature_chance_absolute_difference",
            "premature_excess_absolute_difference",
        )
    ):
        raise ValueError("Predecessor summary reproduction failed")
    examples = save_artifacts(rows, maps, thresholds, beats)
    write_json_atomic(
        OUTPUT / "partition.json",
        {
            name: {
                "ecg_ids": frame["ecg_id"].astype(int).tolist(),
                "patient_ids": frame["patient_id"].astype(int).tolist(),
            }
            for name, frame in rows.items()
        },
    )
    sources = [
        PROTOCOL,
        Path(__file__),
        ROOT / "ecg_experiment/calibrated_localization.py",
        ROOT / "ecg_experiment/lead_wave_maps.py",
        ROOT / "ecg_experiment/fragment_localization.py",
        ROOT / "ecg_experiment/intervals.py",
        ROOT / "ecg_experiment/full_development.py",
        ROOT / "ecg_experiment/external_encoders.py",
        ROOT / "ecg_experiment/eda/ptbxl.py",
    ]
    inputs = [
        PRIOR / "result.json",
        PRIOR / "unit_scores.npz",
        *CACHE_PATHS,
        ROOT / "data/raw/ptb-xl/1.0.3/ptbxl_database.csv",
        ROOT / "data/raw/ptb-xl/1.0.3/scp_statements.csv",
    ]
    waveform_identity = {}
    for stem in everyone["filename_hr"]:
        for extension in (".hea", ".dat"):
            path = ROOT / "data/raw/ptb-xl/1.0.3" / f"{stem}{extension}"
            waveform_identity[to_stored(path)] = sha256_file(path)
    write_json_atomic(OUTPUT / "waveform_hashes.json", waveform_identity)
    result = {
        "experiment": 53,
        "git_head": git_head(ROOT),
        "seed": SEED,
        "seconds": time.perf_counter() - start,
        "counts": {
            name: {"records": len(frame), "patients": int(frame["patient_id"].nunique())}
            for name, frame in rows.items()
        },
        "integrity": integrity,
        "fit": fit_receipt,
        "equal_width_fit": equal_receipt,
        "metrics": metrics,
        "examples": examples,
        "source_hashes": {to_stored(path): sha256_file(path) for path in sources},
        "input_hashes": {to_stored(path): sha256_file(path) for path in inputs},
        "output_hashes": {
            to_stored(OUTPUT / name): sha256_file(OUTPUT / name)
            for name in ("partition.json", "unit_scores.npz", "waveform_hashes.json")
        },
    }
    write_json_atomic(OUTPUT / "result.json", result)
    LOG.info("Completed in %.1fs: %s", result["seconds"], metrics["decision"])


if __name__ == "__main__":
    run()
