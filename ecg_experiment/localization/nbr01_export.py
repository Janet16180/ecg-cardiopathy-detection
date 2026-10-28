"""NBR-01 examples file for the app page: 16 SPH ECGs drawn within groups, never by result."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..files import write_text_atomic
from ..pr74_fresh_baseline import read_signal
from . import nbr01_analysis as na
from . import nbr01_counterpart as nc
from . import nbr01_features as nf
from . import nbr01_selection as ns
from . import nbr01_study as study

QUOTAS = {"inferior": 3, "anterior": 3, "lateral": 2, "rbbb": 2, "lbbb": 2, "stt": 2, "normal": 2}
MISSING_LATERAL = ("inferior", "anterior")
SEED = 20261028
FS = 250
SAMPLES = 2500
EXPORT_MODELS = {"all_age": "all_age_blend", "ptb_baseline": "ptb_baseline_blend"}
SCORE_KEYS = ("original", "full", "lead", "component", "grid", "self_swap_grid", "self_swap_lead",
              "self_swap_full", "hgb_grid", "xecg_grid", "hgb_original", "xecg_original")
AGGREGATE = Path("reports/nbr01-normal-beat-replacement-2026-10-28/results.json")
SUBSETS = ("abnormal", "normal", "inferior", "anterior", "lbbb", "rbbb", "stt")
SUMMARY_NAMES = ("territory", "territory_minus_energy", "swap_territory", "wave", "wave_minus_energy",
                 "qrs_bbb", "qrs_bbb_minus_energy", "stt_stt", "stt_stt_minus_energy", "bbb_lead",
                 "bbb_lead_minus_energy", "top1_hit", "top1_hit_minus_energy", "top3_hit",
                 "top3_hit_minus_energy")
SUMMARY_SUBSET_NAMES = ("flagged", "original", "full_score", "full_drop", "swap_full_drop", "grid_abs",
                        "swap_grid_abs", "lead_abs", "swap_lead_abs", "grid_flip", "lead_flip",
                        "component_flip", "full_flip", "swap_grid_flip", "swap_lead_flip", "swap_full_flip")


def quotas(available: dict[str, int]) -> dict[str, int]:
    """
    Return the fixed example counts per group.

    The two lateral slots move to inferior and anterior when SPH has no lateral ECG.

    Parameters
    ----------
    available : dict[str, int]
        Prepared records per group.

    Returns
    -------
    dict[str, int]
        Examples per group.
    """
    result = dict(QUOTAS)
    if available.get("lateral", 0) == 0:
        for group in MISSING_LATERAL:
            result[group] += 1
        result["lateral"] = 0
    return result


def draw_examples(groups: np.ndarray, ecg_ids: np.ndarray, seed: int = SEED) -> list[int]:
    """
    Positions of the examples: within each group, a seeded draw without replacement among sorted IDs.

    Parameters
    ----------
    groups, ecg_ids : np.ndarray
        Group and ECG ID of each analysed record.
    seed : int
        Generator seed.

    Returns
    -------
    list[int]
        Record positions in ``QUOTAS`` group order.
    """
    rng = np.random.default_rng(seed)
    counts = quotas({group: int(np.sum(groups == group)) for group in QUOTAS})
    chosen = []
    for group, count in counts.items():
        members = np.flatnonzero(groups == group)
        members = members[np.argsort(ecg_ids[members], kind="stable")]
        if count and len(members):
            chosen += members[rng.choice(len(members), min(count, len(members)), replace=False)].tolist()
    return chosen


def at_250(signal: np.ndarray) -> list[list[float]]:
    """Average sample pairs of a (12, 5000) 500 Hz signal to 250 Hz and round to microvolts."""
    reduced = 0.5 * (signal[:, 0::2] + signal[:, 1::2])
    return np.round(reduced, 3).tolist()


def runs(mask: np.ndarray) -> list[list[int]]:
    """Half-open 250 Hz sample ranges of the contiguous runs of a 500 Hz boolean mask."""
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    pairs = zip(edges[0::2], edges[1::2], strict=True)
    return [[int(start) // 2, (int(end) + 1) // 2] for start, end in pairs]


def model_scores(scores: np.ndarray, model: int) -> dict[str, Any]:
    """
    Page scores of one record under one model.

    Parameters
    ----------
    scores : np.ndarray
        (variants, models, kinds) scores of the record.
    model : int
        Model position.

    Returns
    -------
    dict[str, Any]
        Blend scores after each replacement, the component grids and the unreplaced component scores.
    """
    where = na.positions()
    hgb, xecg, blend = (scores[:, model, k] for k in range(len(nf.KINDS)))

    def rounded(values: np.ndarray) -> Any:
        return np.round(values, 6).tolist()

    return {"original": rounded(blend[0]), "full": rounded(blend[where["full"]]),
            "lead": rounded(blend[where["lead"]]), "component": rounded(blend[where["component"]]),
            "grid": rounded(blend[where["grid"]]), "self_swap_grid": rounded(blend[where["swap_grid"]]),
            "self_swap_lead": rounded(blend[where["swap_lead"]]),
            "self_swap_full": rounded(blend[where["swap_full"]]),
            "hgb_grid": rounded(hgb[where["grid"]]), "xecg_grid": rounded(xecg[where["grid"]]),
            "hgb_original": rounded(hgb[0]), "xecg_original": rounded(xecg[0])}


def validate(data: dict[str, Any]) -> None:
    """
    Check an examples object against the page schema.

    Parameters
    ----------
    data : dict[str, Any]
        Parsed examples file.

    Raises
    ------
    ValueError
        If any field is missing or has the wrong shape.
    """
    def need(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    need(data.get("schema_version") == 1 and data.get("fs") == FS, "schema_version or fs")
    need(list(data.get("leads", [])) == list(ns.LEAD_NAMES), "leads")
    need(list(data.get("components", [])) == list(nc.COMPONENTS), "components")
    need(set(data.get("models", {})) == set(EXPORT_MODELS.values()), "models")
    need(all(isinstance(value.get("threshold"), float) for value in data["models"].values()), "thresholds")
    for example in data.get("examples", []):
        need(isinstance(example.get("id"), str) and example.get("group") in ns.GROUPS, "id or group")
        need(isinstance(example.get("statements"), list), "statements")
        need(example.get("sex") in ("M", "F"), "sex")
        need(isinstance(example.get("age"), int), "age")
        for key in ("signal", "counterpart"):
            array = np.asarray(example.get(key), dtype=float)
            need(array.shape == (len(ns.LEAD_NAMES), SAMPLES) and np.isfinite(array).all(), key)
        need(set(example.get("components", {})) == set(nc.COMPONENTS), "component ranges")
        for ranges in example["components"].values():
            need(all(len(pair) == 2 and 0 <= pair[0] < pair[1] <= SAMPLES for pair in ranges), "range")
        need(set(example.get("scores", {})) == set(EXPORT_MODELS.values()), "score models")
        for scores in example["scores"].values():
            need(set(SCORE_KEYS) <= set(scores), "score keys")
            need(np.shape(scores["lead"]) == (12,), "lead")
            need(np.shape(scores["component"]) == (3,), "component")
            need(np.shape(scores["self_swap_lead"]) == (12,), "self_swap_lead")
            for key in ("grid", "self_swap_grid", "hgb_grid", "xecg_grid"):
                need(np.shape(scores[key]) == (12, 3), key)
            need(all(isinstance(scores[key], float) for key in ("original", "full", "self_swap_full",
                                                                 "hgb_original", "xecg_original")), "scalars")
        need(np.shape(example.get("energy_grid")) == (12, 3), "energy_grid")
    summary = data.get("summary", {})
    need({"statistics", "n", "decisions", "primary_endpoint"} <= set(summary), "summary")
    statistics = summary["statistics"].values()
    need(all({"point", "low", "high"} <= set(value) for value in statistics), "statistics")
    need(all(f"{model}/territory" in summary["statistics"] for model in EXPORT_MODELS.values()), "territory")


def summary_block(statistics: dict[str, dict[str, float]], counts: pd.DataFrame,
                  integrity: dict[str, Any], primary: str) -> dict[str, Any]:
    """
    Collect the study-level numbers for the page, keyed exactly as in the aggregate ``results.json``.

    Parameters
    ----------
    statistics : dict[str, dict[str, float]]
        ``statistics`` of the analyze stage.
    counts : pd.DataFrame
        Records and flagged records per group, indexed by group.
    integrity : dict[str, Any]
        Score-stage integrity record.
    primary : str
        Primary endpoint recorded by the prepare stage.

    Returns
    -------
    dict[str, Any]
        Selected statistics with intervals, group sizes and the protocol's decision labels.
    """
    keys = [key for key in statistics if key.startswith("energy/")]
    for model in EXPORT_MODELS.values():
        keys += [f"{model}/{name}" for name in SUMMARY_NAMES]
        keys += [f"{model}/{subset}/{name}" for subset in SUBSETS for name in SUMMARY_SUBSET_NAMES]
    selected = {key: {name: statistics[key][name] for name in ("point", "low", "high")} for key in keys}

    def above(key: str) -> bool:
        return statistics[key]["low"] > 0

    decisions = {
        "primary": "territory localization shown" if above("all_age_blend/territory") else "not shown",
        "qualifier": ("beyond replaced energy" if above("all_age_blend/territory_minus_energy")
                      else "not shown beyond replaced energy"),
        "operator_check": "passed" if above("all_age_blend/abnormal/full_drop") else "flagged",
        "integrity_registered": "passed" if integrity["registered_check_passed"] else "failed",
        "integrity_amendment_1": ("not needed" if integrity["amendment_1"] is None
                                  else "passed" if integrity["amendment_1"]["passed"] else "failed"),
        "secondary_model": "ptb_baseline_blend, descriptive, no decision"}
    n = {group: {name: int(value) for name, value in row.items()} for group, row in counts.iterrows()}
    return {"source": str(AGGREGATE), "primary_endpoint": primary,
            "primary_statistic": "all_age_blend/territory",
            "intervals": "95% whole-patient bootstrap, 2,000 draws, seed 20261028", "statistics": selected,
            "n": n, "decisions": decisions}


def export(input_root: Path) -> dict[str, Any]:
    """Write ``examples.json`` from the scored records and validate it."""
    receipt = study.binding()
    study.require(input_root, "score")
    study.require(input_root, "analyze")
    data = np.load(input_root / study.OUTPUT / "score/scores.npz")
    rows = study.prepared_rows(input_root).set_index("ecg_id").loc[data["ecg_id"]].reset_index()
    limits = nf.thresholds(input_root)
    where = na.positions()
    examples = []
    for position in draw_examples(data["group"], data["ecg_id"]):
        row = rows.iloc[position]
        plan = nf.load_plan(study.WORK / "full/plans" / f"{row['ecg_id']}.npz")
        raw = read_signal(input_root, row)
        examples.append({
            "id": str(row["ecg_id"]), "group": str(row["group"]),
            "statements": ns.statement_text(row["aha_codes"], input_root),
            "age": int(round(float(row["age"]))), "sex": "M" if row["male"] == "1.0" else "F",
            "signal": at_250(raw), "counterpart": at_250(raw + plan["difference"].astype(np.float64)),
            "components": {name: runs(plan["components"][part]) for part, name in enumerate(nc.COMPONENTS)},
            "scores": {EXPORT_MODELS[model]: model_scores(data["scores"][position], m)
                       for m, model in enumerate(nf.MODELS)},
            "energy_grid": np.round(data["energy"][position][where["grid"]], 6).tolist()})
    value = {"schema_version": 1, "fs": FS, "leads": list(ns.LEAD_NAMES), "components": list(nc.COMPONENTS),
             "models": {EXPORT_MODELS[model]: {"threshold": limits[f"{model}_blend"]} for model in nf.MODELS},
             "examples": examples}
    output = input_root / study.OUTPUT
    value["summary"] = summary_block(
        json.loads((output / "analyze/results.json").read_text())["statistics"],
        pd.read_csv(output / "analyze/counts.csv", index_col="group"),
        json.loads((output / "score/integrity.json").read_text()),
        study.require(input_root, "prepare")["primary_endpoint"])
    validate(value)
    path = input_root / study.OUTPUT / "examples.json"
    write_text_atomic(path, json.dumps(value, separators=(",", ":")))
    directory = study.stage_dir(input_root, "export")
    groups = pd.Series([example["group"] for example in examples]).value_counts().to_dict()
    details = {"examples": len(examples), "groups": groups, "path": str(study.OUTPUT / "examples.json"),
               "bytes": path.stat().st_size}
    (directory / "export.json").write_text(json.dumps(details, indent=2))
    return study.complete(input_root, "export", receipt, details)
