"""NBR-04 page examples: NBR-01's 16 SPH ECGs with the exported arm's region counterparts and scores."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..files import read_json, write_text_atomic
from ..pr74_fresh_baseline import read_signal
from . import nbr01_analysis as na
from . import nbr01_export as ne
from . import nbr01_features as nf
from . import nbr01_study as study01
from . import nbr02_analysis as na2
from . import nbr02_export as ex2
from . import nbr02_pipeline as pipeline02
from . import nbr04_study as study
from . import nbr04_templates as nt

COUNTERPART_NOTE = (
    "Each lead's counterpart is the template fitted with that whole lead left out of the personalization, "
    "so it shows the normal version of the lead that the other leads predict."
)


def lead_counterpart(raw: np.ndarray, row: Any, model: Any) -> np.ndarray:
    """
    Build the composite counterpart: lead l from the whole-lead-l variant's leave-region-out template.

    Parameters
    ----------
    raw : np.ndarray
        (12, 5000) signal in mV.
    row : Any
        SPH baseline row.
    model : Any
        The arm's model.

    Returns
    -------
    np.ndarray
        (12, 5000) counterpart in mV.
    """
    items = pipeline02.counterpart_variants()
    regions = nt.region_differences(raw, *nf.covariates(row), model, items)
    positions = {item.name: position for position, item in enumerate(items)}
    leads = [
        raw[lead] + regions["differences"][positions[f"lead_{lead}"]][lead].astype(np.float64)
        for lead in range(raw.shape[0])
    ]
    return np.stack(leads)


def summary_block(results: dict[str, Any], arm: str) -> dict[str, Any]:
    """Collect both samples' arm statistics and contrasts, and the decision, keyed as in the aggregate."""
    selected = {}
    for dataset, prefix in (("sph", ""), ("ptb", "ptbxl_fold9/")):
        statistics = results[dataset]
        keys = [
            key
            for name in ("P0", "C6", *study.ARMS)
            for key in ex2.arm_keys(statistics, name)
            if key in statistics
        ]
        keys += [
            key for key in statistics if "-" in key.split("/")[0] or key.startswith(("clip/", "template/"))
        ]
        selected.update(
            {prefix + key: {name: statistics[key][name] for name in ("point", "low", "high")} for key in keys}
        )
    return {
        "source": str(study.OUTPUT / "analyze/results.json"),
        "arms": ["P0", "C6", *study.ARMS],
        "exported_arm": arm,
        "intervals": "97.5% whole-patient bootstrap, 2,000 draws, seed 20261028",
        "statistics": selected,
        "decisions": results["decision"],
    }


def validate(data: dict[str, Any]) -> None:
    """Check NBR-01's schema on the exported arm's view and the NBR-04 additions."""
    arm = str(data.get("arm"))
    summary = data.get("summary", {})
    statistics = summary.get("statistics", {})
    view_statistics = {
        key[len(arm) + 1 :]: value for key, value in statistics.items() if key.startswith(f"{arm}/")
    }
    view = {
        **data,
        "summary": {**summary, "primary_endpoint": "decision", "n": {}, "statistics": view_statistics},
    }
    ne.validate(view)
    if arm not in study.ARMS or data.get("counterpart_note") != COUNTERPART_NOTE:
        raise ValueError("arm or counterpart note")
    if not all(f"{name}/all_age_blend/territory" in statistics for name in ("P0", *study.ARMS)):
        raise ValueError("arm statistics")
    if not {"L6", "L6D", "chosen", "exported_arm"} <= set(summary.get("decisions", {})):
        raise ValueError("decisions")


def export(input_root: Path) -> dict[str, Any]:
    """Write ``examples.json`` for the chosen arm (L6 if none passed) and validate it."""
    receipt = study.binding()
    results = study.load_results(input_root)
    arm = results["decision"]["exported_arm"]
    model = study.arm_models(input_root)[arm]
    base = dict(np.load(input_root / study01.OUTPUT / "score/scores.npz"))
    new = dict(np.load(input_root / study.OUTPUT / "score/sph_scores.npz"))
    index = study.ARMS.index(arm)
    scores, energy = na2.arm_arrays(
        base["scores"], base["energy"], new["scores"][:, index], new["energy"][:, index]
    )
    position_of = {ecg_id: position for position, ecg_id in enumerate(base["ecg_id"])}
    rows = study.datasets(input_root)["sph"]["rows"].set_index("ecg_id")
    previous = read_json(input_root / study01.OUTPUT / "examples.json")
    where = na.positions()
    examples = []
    for example in previous["examples"]:
        position = position_of[example["id"]]
        row = rows.loc[example["id"]]
        raw = read_signal(input_root, row)
        examples.append(
            {
                **example,
                "counterpart": ne.at_250(lead_counterpart(raw, row, model)),
                "scores": {
                    ne.EXPORT_MODELS[name]: ne.model_scores(scores[position], m)
                    for m, name in enumerate(nf.MODELS)
                },
                "energy_grid": np.round(energy[position][where["grid"]], 6).tolist(),
            }
        )
    value = {
        "schema_version": 1,
        "fs": ne.FS,
        "arm": arm,
        "counterpart_note": COUNTERPART_NOTE,
        "leads": previous["leads"],
        "components": previous["components"],
        "models": previous["models"],
        "examples": examples,
        "summary": summary_block(results, arm),
    }
    validate(value)
    path = input_root / study.OUTPUT / "examples.json"
    write_text_atomic(path, json.dumps(value, separators=(",", ":")))
    details = {
        "examples": len(examples),
        "arm": arm,
        "bytes": path.stat().st_size,
        "ids_match_nbr01": [e["id"] for e in examples] == [e["id"] for e in previous["examples"]],
        "summary_statistics": len(value["summary"]["statistics"]),
    }
    directory = study.stages.stage_dir(input_root, study.OUTPUT, "export")
    (directory / "export.json").write_text(json.dumps(details, indent=2))
    return study.complete(input_root, "export", receipt, details)
