"""Experiment 055: independently annotated evaluation of label-free ECG wave supports."""

from __future__ import annotations

import importlib.metadata
import json
import logging
import time
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from threadpoolctl import threadpool_limits

from ecg_experiment import ROOT
from ecg_experiment.calibrated_localization import mean_interval
from ecg_experiment.files import sha256_file, write_json_atomic, write_npz_atomic
from ecg_experiment.localization_predecessor_audit import reproduce_042
from ecg_experiment.ludb_boundary_validation import (
    audit_files,
    evaluate_record,
    read_ludb_record,
    record_summary,
)
from ecg_experiment.morphology_boundaries import WAVE_NAMES
from ecg_experiment.paths import to_stored
from ecg_experiment.provenance import git_head

DATA = ROOT / "data/raw/ludb/1.0.1"
OUTPUT = ROOT / "outputs/experiment055_ludb_boundaries_v1"
PROTOCOL = ROOT / "docs/experiment-055-ludb-wave-boundaries.md"
SOURCES = (
    "ecg_experiment/morphology_boundaries.py",
    "ecg_experiment/ludb_boundary_validation.py",
    "ecg_experiment/localization_predecessor_audit.py",
    "ecg_experiment/calibrated_localization.py",
    "ecg_experiment/fragment_localization.py",
    "ecg_experiment/lead_wave_maps.py",
    "ecg_experiment/intervals.py",
    "ecg_experiment/full_development.py",
    "ecg_experiment/external_encoders.py",
    "ecg_experiment/eda/ptbxl.py",
    "ecg_experiment/waveforms.py",
    "ecg_experiment/files.py",
    "ecg_experiment/paths.py",
    "scripts/experiments/run_ludb_boundaries055.py",
    "pyproject.toml",
    "uv.lock",
    "docs/experiment-055-ludb-wave-boundaries.md",
)
SEED = 55055
LOG = logging.getLogger(__name__)


def aggregate(summaries: dict[int, dict]) -> dict:
    """
    Compute the prospectively fixed subject-level primary and per-wave guardrails.

    Parameters
    ----------
    summaries : dict[int, dict]
        Per-record wave/class summaries, including every failed truth slot.

    Returns
    -------
    dict
        Paired primary confidence interval, wave diagnostics and frozen decision.
    """
    ids = np.array(list(summaries))
    fixed = np.array([summaries[i]["primary"]["fixed"] for i in ids])
    adaptive = np.array([summaries[i]["primary"]["adaptive"] for i in ids])
    result = {
        "records": len(ids),
        "subjects": len(ids),
        "fixed_primary_iou": float(fixed.mean()),
        "adaptive_primary_iou": float(adaptive.mean()),
        "primary_gain": mean_interval(ids, adaptive - fixed, seed=SEED),
        "waves": {},
    }
    metrics = ("iou", "onset_mae_ms", "offset_mae_ms", "within_30ms", "missing")
    for wave in WAVE_NAMES:
        chosen = [i for i in ids if wave in summaries[i]["classes"]]
        measurements = {
            method: {
                metric: np.array([summaries[i]["classes"][wave][method][metric] for i in chosen])
                for metric in metrics
            }
            for method in ("fixed", "adaptive")
        }
        result["waves"][wave] = {
            "records_with_truth": len(chosen),
            "eligible_truth_waves": sum(summaries[i]["classes"][wave]["truth_waves"] for i in chosen),
            "fixed": {metric: float(values.mean()) for metric, values in measurements["fixed"].items()},
            "adaptive": {metric: float(values.mean()) for metric, values in measurements["adaptive"].items()},
            "iou_gain": mean_interval(
                np.array(chosen), measurements["adaptive"]["iou"] - measurements["fixed"]["iou"], seed=SEED
            ),
        }
    gain = result["primary_gain"]
    result["decision"] = {
        "practical_gain": gain["estimate"] >= 0.10,
        "supported_gain": gain["ci_low"] > 0,
        "all_wave_gains_nonnegative": all(
            wave["iou_gain"]["estimate"] >= 0 for wave in result["waves"].values()
        ),
    }
    result["decision"]["worth_reviewing"] = all(result["decision"].values())
    return result


def fixed_figures() -> None:
    """
    Render all prospectively chosen records, including poor boundary predictions.

    Returns
    -------
    None
        Writes only local waveform review figures.
    """
    directory = OUTPUT / "figures"
    directory.mkdir()
    for record_id in (1, 50, 100, 150, 200):
        signal, annotations, _ = read_ludb_record(DATA, record_id)
        with np.load(OUTPUT / "records" / f"{record_id}_predictions.npz") as saved:
            predictions = {name: saved[name] for name in ("fixed", "adaptive")}
        figure = Figure(figsize=(16, 8), layout="constrained")
        axes = figure.subplots(2, 3, sharex=True)
        for row, (lead_index, lead) in enumerate(((1, "ii"), (6, "v1"))):
            for column, wave in enumerate(WAVE_NAMES):
                axis = axes[row, column]
                axis.plot(np.arange(5000) / 500, signal[lead_index], color="black", linewidth=0.5)
                for start, _, end in annotations[f"{lead}_{wave}"]:
                    axis.axvspan(start / 500, end / 500, color="green", alpha=0.15)
                for method, color, style in (("fixed", "blue", ":"), ("adaptive", "red", "--")):
                    for start, end in predictions[method][:, lead_index, column]:
                        if start >= 0:
                            axis.axvline(start / 500, color=color, linestyle=style, alpha=0.7, linewidth=0.6)
                            axis.axvline(end / 500, color=color, linestyle=style, alpha=0.7, linewidth=0.6)
                axis.set_title(f"Lead {lead.upper()}, {wave}")
                axis.set_xlabel("seconds")
                axis.set_ylabel("mV")
        figure.suptitle(f"LUDB {record_id}: truth green, fixed blue dotted, morphology red dashed")
        figure.savefig(directory / f"record_{record_id}.png", dpi=110)
        figure.clear()


def execute_records() -> tuple[dict, dict, dict]:
    """
    Evaluate exactly 200 subjects without annotation-guided tuning or exclusions.

    Returns
    -------
    tuple[dict, dict, dict]
        Per-record summaries, complete format/scaling audits and matching coverage.
    """
    directory = OUTPUT / "records"
    directory.mkdir()
    summaries, audits, coverage = {}, {}, {}
    for record_id in range(1, 201):
        signal, annotations, audit = read_ludb_record(DATA, record_id)
        measurements, diagnostics, predictions = evaluate_record(signal, annotations)
        summary = record_summary(measurements)
        summaries[record_id], audits[record_id], coverage[record_id] = summary, audit, diagnostics
        write_json_atomic(
            directory / f"{record_id}.json",
            {
                "record_id": record_id,
                "summary": summary,
                "coverage": diagnostics,
                "wave_measurements": measurements,
            },
        )
        write_npz_atomic(directory / f"{record_id}_predictions.npz", **predictions)
        if record_id % 25 == 0:
            LOG.info("Scored %s/200 subjects with frozen morphology recipe", record_id)
    return summaries, audits, coverage


def coverage_summary(coverage: dict) -> dict:
    """
    Account for every annotation and unannotated prediction slot by wave class.

    Parameters
    ----------
    coverage : dict
        Per-record, per-lead matching audit.

    Returns
    -------
    dict
        Aggregate matching, missingness and unannotated-slot counts.
    """
    all_leads = [lead for record in coverage.values() for lead in record.values()]
    result = {
        "qrs_truth_total": sum(lead["qrs_truth_total"] for lead in all_leads),
        "qrs_anchor_matches": sum(lead["qrs_anchor_matches"] for lead in all_leads),
        "waves": {},
    }
    for name in WAVE_NAMES:
        values = [lead["waves"][name] for lead in all_leads]
        result["waves"][name] = {
            key: sum(row[key] for row in values)
            for key in ("eligible_truth", "edge_truth_excluded", "truth_without_matched_slot")
        }
        for key in ("unannotated_interior_prediction_slots", "rejected_interior_slots"):
            result["waves"][name][key] = {
                method: sum(row[key][method] for row in values) for method in ("fixed", "adaptive")
            }
    return result


def duration_summary() -> dict:
    """
    Describe true and proposed wave durations without excluding failures from the primary.

    Returns
    -------
    dict
        Conditional valid interval duration quantiles, with explicit contributing counts.
    """
    durations = {wave: {method: [] for method in ("truth", "fixed", "adaptive")} for wave in WAVE_NAMES}
    for identifier in range(1, 201):
        record = json.loads((OUTPUT / "records" / f"{identifier}.json").read_text())
        for row in record["wave_measurements"]:
            durations[row["wave"]]["truth"].append((row["truth_offset"] - row["truth_onset"]) * 2)
            for method in ("fixed", "adaptive"):
                interval = row[method]
                if not interval["missing"]:
                    durations[row["wave"]][method].append(
                        (interval["predicted_offset"] - interval["predicted_onset"]) * 2
                    )
    return {
        wave: {
            method: {
                "count": len(values),
                "median": float(np.median(values)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
            }
            for method, values in methods.items()
            if values
        }
        for wave, methods in durations.items()
    }


def run() -> None:
    """
    Run the frozen public-data boundary experiment and preserve complete provenance.

    Returns
    -------
    None
        Writes a unique local output directory and complete result receipt.
    """
    if OUTPUT.exists():
        raise FileExistsError(f"Preserve existing experiment output: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(OUTPUT / "run.log")],
    )
    began = time.perf_counter()
    identity = {
        "source_commit": git_head(ROOT),
        "protocol_commits": ["5069b24", "e0f4c5e"],
        "sources": {name: sha256_file(ROOT / name) for name in SOURCES},
        "versions": {
            name: importlib.metadata.version(name) for name in ("numpy", "scipy", "wfdb", "scikit-learn")
        },
        "input_data": to_stored(DATA),
        "data_source_url": "https://physionet.org/content/ludb/1.0.1/",
        "archive_sha256": sha256_file(ROOT / "data/raw/ludb/ludb-1.0.1.zip"),
        "evaluation_subject_ids": list(range(1, 201)),
        "fitting_subject_ids": [],
        "seed": SEED,
        "bootstrap_draws": 2000,
    }
    write_json_atomic(OUTPUT / "launch.json", identity)
    data_audit = audit_files(DATA)
    record_names = (DATA / "RECORDS").read_text().splitlines()
    if sorted(int(Path(name).name) for name in record_names) != list(range(1, 201)):
        raise ValueError("Official record list differs from 200 frozen subjects")
    LOG.info("Verified public LUDB official checksums: %s", data_audit)
    with threadpool_limits(limits=2):
        integrity = reproduce_042(ROOT)
        LOG.info("Exactly reproduced predecessor U_B metrics before LUDB scoring")
        summaries, audits, coverage = execute_records()
        metrics = aggregate(summaries)
    write_json_atomic(OUTPUT / "data_audit.json", audits)
    write_json_atomic(OUTPUT / "record_summaries.json", summaries)
    fixed_figures()
    counts = {
        name: sum(
            audit["annotation_triplet_counts"][f"{lead.casefold()}_{name}"]
            for audit in audits.values()
            for lead in audit["lead_names"]
        )
        for name in WAVE_NAMES
    }
    typed_counts = {
        name: sum(audit["raw_typed_peak_counts"][name] for audit in audits.values()) for name in WAVE_NAMES
    }
    unknown_events = sum(
        event["reason"].startswith("unknown_")
        for audit in audits.values()
        for events in audit["irregular_annotation_events"].values()
        for event in events
    )
    recovered = sum(
        event["reason"] == "recovered_outside_peak"
        for audit in audits.values()
        for events in audit["irregular_annotation_events"].values()
        for event in events
    )
    input_hashes = {to_stored(path): sha256_file(path) for path in DATA.rglob("*") if path.is_file()}
    outputs = {to_stored(path): sha256_file(path) for path in OUTPUT.rglob("*") if path.is_file()}
    result = {
        "experiment": 55,
        "status": "completed",
        "identity": identity,
        "data_audit": data_audit,
        "annotation_counts": counts,
        "raw_typed_peak_counts": typed_counts,
        "published_wave_counts": {"P": 16797, "QRS": 21966, "T": 19666},
        "unknown_annotation_events": int(unknown_events),
        "recovered_annotation_groups": int(recovered),
        "predecessor_integrity": integrity,
        "metrics": metrics,
        "coverage": coverage_summary(coverage),
        "duration_distributions_ms": duration_summary(),
        "seconds": time.perf_counter() - began,
        "input_hashes": input_hashes,
        "output_hashes": outputs,
        "waveform_diagnoses_used": False,
        "annotation_tuning": False,
        "closed_project_test_access": False,
    }
    write_json_atomic(OUTPUT / "result.json", result)
    LOG.info("Completed in %.1fs: %s", result["seconds"], metrics["decision"])


if __name__ == "__main__":
    run()
