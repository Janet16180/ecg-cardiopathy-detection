"""NBR-04 integrity, added with Amendment 1: re-score L6 and L6D from their saved features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..files import write_json_atomic
from . import nbr04_study as study
from . import stages


def rescore(input_root: Path) -> dict[str, Any]:
    """
    Recompute the L6 and L6D scores from the saved measurements and xECG features and compare.

    Parameters
    ----------
    input_root : Path
        Checkout holding outputs.

    Returns
    -------
    dict[str, Any]
        Completion receipt with the largest absolute score difference and energy identity per sample.

    Raises
    ------
    ValueError
        If any recomputed score or energy differs from the saved one.
    """
    receipt = study.binding()
    study.require(input_root, "score")
    details = {}
    for name, spec in study.datasets(input_root).items():
        ids = spec["rows"]["ecg_id"].tolist()
        again = study.score_arms(input_root, ids, study.WORK / "full" / name)
        saved = np.load(input_root / study.OUTPUT / f"score/{name}_scores.npz")
        details[name] = {
            "records": len(ids),
            "max_abs_score_difference": float(np.max(np.abs(again["scores"] - saved["scores"]))),
            "energy_identical": bool(np.array_equal(again["energy"], saved["energy"])),
        }
    directory = stages.stage_dir(input_root, study.OUTPUT, "amendment_rescore")
    write_json_atomic(directory / "rescore.json", details)
    if any(
        value["max_abs_score_difference"] != 0.0 or not value["energy_identical"]
        for value in details.values()
    ):
        raise ValueError(f"Saved L6 or L6D scores do not reproduce: {details}")
    return study.complete(input_root, "amendment_rescore", receipt, details)
