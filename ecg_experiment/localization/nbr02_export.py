"""NBR-02 page examples: NBR-01's 16 ECGs with the C6 counterpart, scores and an all-arm summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..files import read_json, write_text_atomic
from ..pr74_fresh_baseline import read_signal
from . import nbr01_analysis as na
from . import nbr01_export as ne
from . import nbr01_features as nf
from . import nbr01_study as study01
from . import nbr02_analysis as na2
from . import nbr02_study as study
from .nbr02_templates import ARMS, NEW_ARMS

ARM = "C6"
NBR01_EXAMPLES = study01.OUTPUT / "examples.json"


def arm_keys(statistics: dict[str, Any], arm: str) -> list[str]:
    """Return the arm's prefixed keys of the NBR-01 page summary, plus its selectivities."""
    names = [f"{model}/{name}" for model in ne.EXPORT_MODELS.values() for name in ne.SUMMARY_NAMES]
    names += [f"{model}/{subset}/{name}" for model in ne.EXPORT_MODELS.values() for subset in ne.SUBSETS
              for name in ne.SUMMARY_SUBSET_NAMES]
    names += [f"{score}/selectivity" for score in na2.SCORES]
    keys = [f"{arm}/{name}" for name in names]
    return keys + [key for key in statistics if key.startswith(f"{arm}/energy/")]


def summary_block(results: dict[str, Any], counts: pd.DataFrame, fitsets: dict[str, Any]) -> dict[str, Any]:
    """
    Collect every arm's page statistics, the arm contrasts, template diagnostics and the decision.

    Parameters
    ----------
    results : dict[str, Any]
        NBR-02 ``analyze/results.json``.
    counts : pd.DataFrame
        NBR-01 records and flagged records per group.
    fitsets : dict[str, Any]
        Details of the fitsets stage.

    Returns
    -------
    dict[str, Any]
        Summary for the page, statistics keyed as in the aggregate.
    """
    statistics = results["statistics"]
    keys = [key for arm in ARMS for key in arm_keys(statistics, arm)]
    contrasts = {f"{first}-{second}" for first, second in na2.CONTRASTS}
    keys += [key for key in statistics
             if key.split("/")[0] in contrasts or key.startswith(("template/", "clip/"))]
    selected = {key: {name: statistics[key][name] for name in ("point", "low", "high")} for key in keys}
    decision = results["decision"]
    decisions = {name: {key: part[key] for key in ("statistic", "point", "low", "high", "bound", "passed")}
                 for name, part in decision.items() if isinstance(part, dict)}
    decisions["C6_replaces_P0"] = decision["C6_replaces_P0"]
    n = {group: {name: int(value) for name, value in row.items()} for group, row in counts.iterrows()}
    return {"source": str(study.OUTPUT / "analyze/results.json"), "arms": list(ARMS),
            "primary_statistic": "C6-P0/all_age_blend/selectivity",
            "intervals": "95% whole-patient bootstrap, 2,000 draws, seed 20261028", "statistics": selected,
            "n": n, "decisions": decisions, "fit_sets": fitsets["fit_sets"]}


def validate(data: dict[str, Any]) -> None:
    """
    Check the NBR-02 examples: NBR-01's schema on the C6 view plus the NBR-02 additions.

    Parameters
    ----------
    data : dict[str, Any]
        Parsed examples file.

    Raises
    ------
    ValueError
        If a field is missing or malformed.
    """
    summary = data.get("summary", {})
    statistics = summary.get("statistics", {})
    view = {**data, "summary": {**summary, "primary_endpoint": "decision", "statistics": {
        key[len(ARM) + 1:]: value for key, value in statistics.items() if key.startswith(f"{ARM}/")}}}
    ne.validate(view)
    if data.get("arm") != ARM or not isinstance(data.get("fit_set"), dict):
        raise ValueError("arm or fit_set")
    if not all(f"{arm}/all_age_blend/territory" in statistics for arm in ARMS):
        raise ValueError("arm statistics")
    parts = summary.get("decisions", {})
    if not {"A_selectivity", "B_territory_noninferior", "C_operator_check", "C6_replaces_P0"} <= set(parts):
        raise ValueError("decisions")


def export(input_root: Path) -> dict[str, Any]:
    """Write ``examples.json`` for the C6 arm and validate it."""
    receipt = study.binding()
    analyzed = study.require(input_root, "analyze")
    fitsets = study.require(input_root, "fitsets")
    base = dict(np.load(input_root / study.NBR01_SCORES))
    new = dict(np.load(input_root / study.OUTPUT / "score/scores.npz"))
    position_of = {ecg_id: position for position, ecg_id in enumerate(base["ecg_id"])}
    arm = NEW_ARMS.index(ARM)
    scores, energy = na2.arm_arrays(base["scores"], base["energy"], new["scores"][:, arm],
                                    new["energy"][:, arm])
    rows = study.sample_rows(input_root).set_index("ecg_id")
    where = na.positions()
    previous = read_json(input_root / NBR01_EXAMPLES)
    examples = []
    for example in previous["examples"]:
        position = position_of[example["id"]]
        with np.load(study.WORK / "full/plans" / f"{example['id']}.npz") as prepared:
            difference = prepared[f"{ARM}_difference"].astype(np.float64)
        raw = read_signal(input_root, rows.loc[example["id"]])
        examples.append({**example, "counterpart": ne.at_250(raw + difference),
                         "scores": {ne.EXPORT_MODELS[model]: ne.model_scores(scores[position], m)
                                    for m, model in enumerate(nf.MODELS)},
                         "energy_grid": np.round(energy[position][where["grid"]], 6).tolist()})
    results = read_json(input_root / study.OUTPUT / "analyze/results.json")
    counts = pd.read_csv(input_root / study01.OUTPUT / "analyze/counts.csv", index_col="group")
    value = {"schema_version": 1, "fs": ne.FS, "arm": ARM, "fit_set": fitsets["fit_sets"]["C"]["counts"],
             "leads": previous["leads"], "components": previous["components"], "models": previous["models"],
             "examples": examples, "summary": summary_block(results, counts, fitsets)}
    validate(value)
    path = input_root / study.OUTPUT / "examples.json"
    write_text_atomic(path, json.dumps(value, separators=(",", ":")))
    details = {"examples": len(examples), "ids_match_nbr01": [e["id"] for e in examples]
               == [e["id"] for e in previous["examples"]], "bytes": path.stat().st_size,
               "summary_statistics": len(value["summary"]["statistics"]),
               "decision": analyzed["decision"]["C6_replaces_P0"]}
    (study.stage_dir(input_root, "export") / "export.json").write_text(json.dumps(details, indent=2))
    return study.complete(input_root, "export", receipt, details)
