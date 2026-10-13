"""
Descriptive, undeclared readings of the Experiment 066 outputs on ECGAD development; writes descriptives.json.

Three readings, none a candidate: per-statement mean A2 of each map; a 2 x 2 decomposition of ``factor``
against ``textbook`` (lead source MIL or table, rhythm route MIL time evidence or whole beat); and the share
of marked lead-samples in each delineated component.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ecg_experiment import ROOT
from ecg_experiment import combination066 as cb
from ecg_experiment import mil065 as mil
from ecg_experiment.eda.ptbxl import load_metadata, load_statements
from ecg_experiment.files import write_json_atomic
from ecg_experiment.locbench import START, STOP, ecgad_record_metrics, load_benchmark
from ecg_experiment.locbench_baselines061 import amplitude_map, exact_signal
from ecg_experiment.locbench_cache import open_cache

OUT = ROOT / "outputs/experiment066_combination_v1"
WORK = Path(os.environ.get("ECG066_WORK", "/tmp/ecg066-work")) / "full"
MAPS = ("qrs_prior", "textbook_true", "textbook", "factor", "factor_q", "units_learned")
MIN_RECORDS = 5


def tables_folder(name: str) -> Path:
    """Tables folder of a 066 map or of the copied bar."""
    return OUT / ("ref/qrs_prior" if name == "qrs_prior" else f"locbench/{name}/tables")


def decomposition_sum(
    heads: list[str],
    prob: np.ndarray,
    means: np.ndarray,
    rhythm: dict[str, np.ndarray],
    masks: np.ndarray,
    mil_leads: bool,
    mil_rhythm: bool,
) -> np.ndarray:
    """
    Sum over heads with a chosen lead source and rhythm route.

    Parameters
    ----------
    heads : list[str]
        Non-NORM heads.
    prob : np.ndarray
        Bag probabilities.
    means : np.ndarray
        (heads, 12) MIL lead means.
    rhythm : dict[str, np.ndarray]
        Rhythm head evidence (12, steps).
    masks : np.ndarray
        Component masks.
    mil_leads : bool
        MIL lead evidence (``factor``) or the table's lead indicator (``textbook``) for table heads.
    mil_rhythm : bool
        MIL time-resolved evidence or whole-beat painting for rhythm heads.

    Returns
    -------
    np.ndarray
        (12, samples) unnormalized map.
    """
    weights = cb.normalized_leads(means)
    total = np.zeros((12, masks.shape[1]))
    for k, head in enumerate(heads):
        if head in cb.RHYTHM_HEADS:
            if mil_rhythm:
                total += prob[k] * cb.rhythm_term(rhythm[head])
            else:
                lead = weights[k] if mil_leads else np.ones(12)
                total += prob[k] * np.outer(lead, cb.time_prior(masks, ("beat",)))
        elif mil_leads:
            total += prob[k] * np.outer(weights[k], cb.time_prior(masks, cb.head_components(head)))
        else:
            total += prob[k] * cb.textbook_region(head, masks)
    return total


def main() -> None:
    """Compute the three descriptive readings and write them."""
    bench = load_benchmark()
    ecgad = bench.ecgad.reset_index(drop=True)
    a2 = pd.DataFrame({name: pd.read_csv(tables_folder(name) / "ecgad.csv")["A2"] for name in MAPS})
    a2["ecg_id"] = ecgad["ecg_id"].to_numpy()
    heads = json.loads((ROOT / "outputs/experiment065_mil_v1/result.json").read_text())["heads"]
    abnormal = [k for k, h in enumerate(heads) if h != mil.REFERENCE_HEAD]
    names = [heads[k] for k in abnormal]
    y, mask = mil.statement_labels(load_metadata(), load_statements(), ecgad["ecg_id"].to_numpy(), heads)
    positive = (y * mask).astype(bool)
    per_statement = {}
    for k, head in enumerate(heads):
        rows = positive[:, k] & a2["qrs_prior"].notna().to_numpy()
        if rows.sum() >= MIN_RECORDS:
            per_statement[head] = {"records": int(rows.sum())} | {
                name: float(a2.loc[rows, name].mean()) for name in MAPS
            }

    signals, index = open_cache()
    row = dict(zip(index.ecg_id.astype(int), range(len(index)), strict=True))
    ids = np.load(WORK / "mil_ids.npy")
    mil_row = dict(zip(ids.astype(int).tolist(), range(len(ids)), strict=True))
    bags = np.load(WORK / "mil_bags.npy", mmap_mode="r")
    lead_means = np.load(WORK / "mil_lead_means.npy", mmap_mode="r")
    rhythm_all = np.load(WORK / "mil_rhythm.npy", mmap_mode="r")
    combos = {
        "mil_leads_mil_rhythm": (True, True),
        "mil_leads_beat_rhythm": (True, False),
        "table_leads_mil_rhythm": (False, True),
        "table_leads_beat_rhythm": (False, False),
    }
    values = {name: [] for name in combos}
    component_marks = np.zeros(len(cb.COMPONENTS) + 1)
    component_samples = np.zeros(len(cb.COMPONENTS) + 1)
    marked_leads = []
    for position, ecg_id in enumerate(ecgad["ecg_id"].astype(int)):
        stored = signals[row[ecg_id]]
        intervals = cb.delineated_components(np.asarray(stored, np.float64))
        masks = cb.component_masks(intervals)
        amplitude = amplitude_map(exact_signal(stored))
        m = mil_row[ecg_id]
        prob = 1.0 / (1.0 + np.exp(-np.asarray(bags[m], np.float64)[abnormal]))
        means = np.asarray(lead_means[m])[abnormal]
        rhythm = {h: np.asarray(rhythm_all[m, j], np.float32) for j, h in enumerate(cb.RHYTHM_HEADS)}
        labels = bench.ecgad_labels[position]
        for name, (mil_leads, mil_rhythm) in combos.items():
            total = decomposition_sum(names, prob, means, rhythm, masks, mil_leads, mil_rhythm)
            values[name].append(ecgad_record_metrics(cb.finish(total, amplitude), labels)["A2"])
        if ecgad["marked"].iloc[position]:
            first = np.argmax(np.vstack([masks, np.ones((1, masks.shape[1]), bool)]), axis=0)[START:STOP]
            marks = labels[:, START:STOP]
            marked_leads.append(int(marks.any(axis=1).sum()))
            for c in range(len(cb.COMPONENTS) + 1):
                component_marks[c] += marks[:, first == c].sum()
                component_samples[c] += 12 * (first == c).sum()
    decomposition = {}
    for name, series in values.items():
        series = np.array(series)
        decomposition[name] = {
            "A2_mean": float(np.nanmean(series)),
            "records": int((~np.isnan(series)).sum()),
            "wins_vs_qrs_prior": int(np.nansum(series > a2["qrs_prior"].to_numpy())),
        }
    rebuilt = np.array(values["mil_leads_mil_rhythm"])
    decomposition["factor_rebuild_max_abs_difference"] = float(np.nanmax(np.abs(rebuilt - a2["factor"])))
    labels_order = [*cb.COMPONENTS, "none"]
    result = {
        "note": "Undeclared descriptive readings on ECGAD development; none is a candidate.",
        "per_statement_mean_A2": per_statement,
        "decomposition_A2": decomposition,
        "wins_vs_qrs_prior": {
            name: int((a2[name] > a2["qrs_prior"]).sum()) for name in MAPS if name != "qrs_prior"
        },
        "mean_marked_leads_per_marked_record": float(np.mean(marked_leads)),
        "marked_records": len(marked_leads),
        "marked_share_by_component": dict(
            zip(labels_order, (component_marks / component_marks.sum()).round(4).tolist(), strict=True)
        ),
        "sample_share_by_component": dict(
            zip(labels_order, (component_samples / component_samples.sum()).round(4).tolist(), strict=True)
        ),
    }
    write_json_atomic(OUT / "descriptives.json", result)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
