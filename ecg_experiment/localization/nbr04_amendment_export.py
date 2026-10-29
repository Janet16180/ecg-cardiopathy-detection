"""NBR-04 Amendment 1 page examples: the chosen arm of three (L6 if none passes) on NBR-01's 16 SPH ECGs."""

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
from . import nbr04_amendment as amendment
from . import nbr04_export as ex4
from . import nbr04_study as study
from . import stages

ARMS = ("P0", "C6", "L6", "D6", "L6D")
WHOLE_RECORD_NOTE = "The counterpart is the arm's whole-record template, as in NBR-02."


def arm_arrays(input_root: Path, arm: str) -> tuple[np.ndarray, np.ndarray]:
    """Return NBR-01-layout SPH scores and energies of one decision arm."""
    base = dict(np.load(input_root / study01.OUTPUT / "score/scores.npz"))
    if arm == "D6":
        new = dict(np.load(input_root / study.OUTPUT / "amendment_score/sph_scores.npz"))
        position = list(new["arms"]).index("D6")
    else:
        new = dict(np.load(input_root / study.OUTPUT / "score/sph_scores.npz"))
        position = study.ARMS.index(arm)
    return na2.arm_arrays(
        base["scores"], base["energy"], new["scores"][:, position], new["energy"][:, position]
    )


def counterpart(input_root: Path, arm: str, row: Any, raw: np.ndarray) -> tuple[np.ndarray, str]:
    """Return the arm's displayed counterpart and its note."""
    if arm == "D6":
        with np.load(amendment.WORK / "full/sph/plans" / f"{row.name}.npz") as prepared:
            return raw + prepared["D6_difference"].astype(np.float64), WHOLE_RECORD_NOTE
    return ex4.lead_counterpart(raw, row, study.arm_models(input_root)[arm]), ex4.COUNTERPART_NOTE


def summary_block(results: dict[str, Any], arm: str) -> dict[str, Any]:
    """Both samples' arm statistics, contrasts and factorial effects, and the amended decision."""
    selected = {}
    for dataset, prefix in (("sph", ""), ("ptb", "ptbxl_fold9/")):
        statistics = results[dataset]
        keys = [key for name in ARMS for key in ex2.arm_keys(statistics, name) if key in statistics]
        keys += [key for key in statistics if "-" in key.split("/")[0] or key.startswith("factorial/")]
        selected.update(
            {
                prefix + key: {
                    name: statistics[key][name] for name in ("point", "low", "high", "low95", "high95")
                }
                for key in keys
            }
        )
    return {
        "source": str(study.OUTPUT / "amendment_analyze/results.json"),
        "arms": list(ARMS),
        "exported_arm": arm,
        "intervals": "low/high 98.33% (decision), low95/high95 95% (factorial); whole-patient bootstrap, "
        "2,000 draws, seed 20261028",
        "statistics": selected,
        "decisions": results["decision"],
    }


def validate(data: dict[str, Any]) -> None:
    """Check NBR-01's schema on the exported arm's view and the amendment's additions."""
    arm = str(data.get("arm"))
    summary = data.get("summary", {})
    statistics = summary.get("statistics", {})
    view_statistics = {
        key[len(arm) + 1 :]: value for key, value in statistics.items() if key.startswith(f"{arm}/")
    }
    ne.validate(
        {
            **data,
            "summary": {**summary, "primary_endpoint": "decision", "n": {}, "statistics": view_statistics},
        }
    )
    if arm not in amendment.DECISION_ARMS or not data.get("counterpart_note"):
        raise ValueError("arm or counterpart note")
    if not all(f"{name}/all_age_blend/territory" in statistics for name in ARMS):
        raise ValueError("arm statistics")
    if not {*amendment.DECISION_ARMS, "passing", "chosen", "exported_arm"} <= set(
        summary.get("decisions", {})
    ):
        raise ValueError("decisions")


def export(input_root: Path) -> dict[str, Any]:
    """Write ``examples.json`` for the amended decision's arm and validate it."""
    receipt = study.binding()
    results = amendment.load_results(input_root)
    arm = results["decision"]["exported_arm"]
    scores, energy = arm_arrays(input_root, arm)
    base_ids = np.load(input_root / study01.OUTPUT / "score/scores.npz")["ecg_id"]
    position_of = {ecg_id: position for position, ecg_id in enumerate(base_ids)}
    rows = study.datasets(input_root)["sph"]["rows"].set_index("ecg_id")
    previous = read_json(input_root / study01.OUTPUT / "examples.json")
    where = na.positions()
    examples, note = [], ""
    for example in previous["examples"]:
        position = position_of[example["id"]]
        row = rows.loc[example["id"]]
        raw = read_signal(input_root, row)
        shown, note = counterpart(input_root, arm, row, raw)
        examples.append(
            {
                **example,
                "counterpart": ne.at_250(shown),
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
        "counterpart_note": note,
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
    directory = stages.stage_dir(input_root, study.OUTPUT, "amendment_export")
    (directory / "export.json").write_text(json.dumps(details, indent=2))
    return study.complete(input_root, "amendment_export", receipt, details)
