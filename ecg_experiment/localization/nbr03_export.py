"""NBR-03 page examples from PTB-XL fold 9, in NBR-01's final schema plus ``dataset``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..eda.ptbxl import load_statements
from ..files import read_json, write_text_atomic
from ..pr74_fresh_baseline import read_signal
from . import nbr01_analysis as na
from . import nbr01_counterpart as nc
from . import nbr01_export as ne
from . import nbr01_features as nf
from . import nbr01_selection as ns
from . import nbr03_ptbxl as ptb
from . import nbr03_study as study

DATASET = "ptbxl_fold9"
AGE_NOTE = "PTB-XL records ages above 89 as 300; they are exported as 90."


def statements(codes: set[str]) -> list[str]:
    """Return readable SCP statements, code first, in alphabetical code order."""
    table = load_statements()
    return [f"{code}: {table.at[code, 'description']}" for code in sorted(codes)]


def export_age(value: str) -> int:
    """Return the exported age, 90 for PTB-XL's above-89 code."""
    age = float(value)
    return 90 if age > 89 else int(round(age))


def summary(results: dict[str, Any], counts: pd.DataFrame, integrity: dict[str, Any]) -> dict[str, Any]:
    """Build NBR-01's page summary plus the all-territory, per-territory and lateral statistics."""
    statistics = results["statistics"]
    block = ne.summary_block(statistics, counts, integrity, "territory")
    extra = [
        key for key in statistics if key.startswith(("all_territories/", "parts/")) or "/lateral/" in key
    ]
    block["statistics"].update(
        {key: {name: statistics[key][name] for name in ("point", "low", "high")} for key in extra}
    )
    decision = results["decision"]
    block["decisions"]["replication"] = (
        "localization replicates" if decision["localization_replicates"] else "not replicated"
    )
    block["decisions"]["replication_qualifier"] = (
        "beyond replaced energy" if decision["beyond_energy"] else "not shown beyond replaced energy"
    )
    block["primary_territories"] = decision["primary_territories"]
    block["source"] = str(study.OUTPUT / "analyze/results.json")
    block["thresholds"] = "PTB fold-8 calibration normals (5% budget), patient-disjoint from fold 9"
    return block


def validate(data: dict[str, Any]) -> None:
    """Check NBR-01's schema and the NBR-03 additions."""
    ne.validate(data)
    if data.get("dataset") != DATASET:
        raise ValueError("dataset")


def export(input_root: Path) -> dict[str, Any]:
    """Write ``examples.json`` for fold 9 and validate it."""
    receipt = study.binding()
    results = study.load_results(input_root)
    integrity = study.require(input_root, "score")
    data = np.load(input_root / study.OUTPUT / "score/scores.npz")
    rows = study.prepared_rows(input_root).set_index("ecg_id").loc[data["ecg_id"]].reset_index()
    limits = ptb.thresholds(input_root)
    where = na.positions()
    examples = []
    for position in ne.draw_examples(data["group"], data["ecg_id"]):
        row = rows.iloc[position]
        plan = nf.load_plan(study.WORK / "full/plans" / f"{row['ecg_id']}.npz")
        raw = read_signal(input_root, row)
        examples.append(
            {
                "id": str(row["ecg_id"]),
                "group": str(row["group"]),
                "statements": statements(row["codes"]),
                "age": export_age(row["age"]),
                "sex": "M" if row["sex"] == "0" else "F",
                "signal": ne.at_250(raw),
                "counterpart": ne.at_250(raw + plan["difference"].astype(np.float64)),
                "components": {
                    name: ne.runs(plan["components"][part]) for part, name in enumerate(nc.COMPONENTS)
                },
                "scores": {
                    ne.EXPORT_MODELS[model]: ne.model_scores(data["scores"][position], m)
                    for m, model in enumerate(nf.MODELS)
                },
                "energy_grid": np.round(data["energy"][position][where["grid"]], 6).tolist(),
            }
        )
    counts = pd.read_csv(input_root / study.OUTPUT / "analyze/counts.csv", index_col="group")
    value = {
        "schema_version": 1,
        "dataset": DATASET,
        "fs": ne.FS,
        "age_note": AGE_NOTE,
        "leads": list(ns.LEAD_NAMES),
        "components": list(nc.COMPONENTS),
        "models": {ne.EXPORT_MODELS[model]: {"threshold": limits[f"{model}_blend"]} for model in nf.MODELS},
        "examples": examples,
        "summary": summary(results, counts, integrity),
    }
    validate(value)
    path = input_root / study.OUTPUT / "examples.json"
    write_text_atomic(path, json.dumps(value, separators=(",", ":")))
    groups = pd.Series([example["group"] for example in examples]).value_counts().to_dict()
    details = {
        "examples": len(examples),
        "groups": groups,
        "bytes": path.stat().st_size,
        "summary_statistics": len(value["summary"]["statistics"]),
    }
    (study.stages.stage_dir(input_root, study.OUTPUT, "export") / "export.json").write_text(
        json.dumps(details, indent=2)
    )
    return study.complete(input_root, "export", receipt, details)


def read_examples(input_root: Path) -> dict[str, Any]:
    """Read the written examples file."""
    return read_json(input_root / study.OUTPUT / "examples.json")
