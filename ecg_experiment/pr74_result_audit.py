"""Independently verify corrective evaluation counts from local saved predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .files import sha256_file


def require(condition: bool, message: str) -> None:
    """Stop verification at the first inconsistent claim."""
    if not condition:
        raise ValueError(message)


def confusion(target: np.ndarray, score: np.ndarray, cutoff: float) -> dict[str, int]:
    """Recount the four cells directly, independently of the production metric helper."""
    flagged, positive = score > cutoff, target == 1
    return {"tp": int(np.sum(flagged & positive)), "fn": int(np.sum(~flagged & positive)),
            "fp": int(np.sum(flagged & ~positive)), "tn": int(np.sum(~flagged & ~positive))}


def audit_pool(directory: Path, artifact: dict, site: str) -> dict:
    """Verify patient separation, normal-only calibration, point counts and every saved cutoff."""
    with np.load(directory / f"{site}_predictions.npz", allow_pickle=False) as saved:
        held = saved["evaluation"]
        require(held.dtype == bool and held.any() and (~held).any(), "Invalid evaluation mask")
        target = saved["target"]
        require(np.isin(target, [0, 1]).all(), "Undefined evaluation target")
        require(not set(saved["patient_id"][held]) & set(saved["patient_id"][~held]),
                "Calibration and evaluation patients overlap")
        require(len(np.unique(saved["ecg_id"])) == len(target), "Repeated record identity")
        checked = []
        for model in artifact["models"]:
            result = next(row for row in model["evaluations"] if row["pool"] == site)
            scores = saved[model["id"]]
            require(scores.shape == target.shape and np.isfinite(scores).all(), "Invalid saved scores")
            normal = scores[(~held) & (target == 0)]
            allowed = int(np.floor(0.05 * len(normal)))
            cutoff = float(np.sort(normal)[len(normal) - allowed - 1])
            require(cutoff == result["threshold"], "Operating point is not the calibration order statistic")
            require(int(np.sum(normal > cutoff)) <= allowed, "Normal calibration budget exceeded")
            cells = confusion(target[held], scores[held], cutoff)
            require(all(cells[key] == result["metrics"][key] for key in cells), "Point matrix mismatch")
            auroc = float(roc_auc_score(target[held], scores[held]))
            require(abs(auroc - result["metrics"]["auroc"]) < 1e-12, "AUROC mismatch")
            sweep = result["sweep"]
            for index, threshold in enumerate(sweep["threshold"]):
                actual = confusion(target[held], scores[held], threshold)
                require(all(actual[key] == sweep[key][index] for key in actual), "Sweep matrix mismatch")
            checked.append({"model_id": model["id"], **cells, "auroc": auroc,
                            "thresholds_verified": len(sweep["threshold"])})
    return {"site": site, "evaluation_ecgs": int(held.sum()), "calibration_ecgs": int((~held).sum()),
            "calibration_evaluation_patients_disjoint": True, "models": checked}


def audit(directory: Path) -> dict:
    """Verify completed local outputs without fitting models or accessing closed test data.

    Parameters
    ----------
    directory : Path
        Completed corrective evaluation directory.

    Returns
    -------
    dict
        Aggregate verification evidence and input hashes, without patient identities.
    """
    receipt = json.loads((directory / "completion.json").read_text())
    require(receipt.get("status") == "completed" and receipt.get("stage") == "evaluate",
            "No completed evaluation receipt")
    names = ("screening_evaluation.json", "sph_predictions.npz", "ptb_predictions.npz")
    hashes = {name: sha256_file(directory / name) for name in names}
    for name, digest in hashes.items():
        bindings = [value for path, value in receipt["artifacts"].items() if Path(path).name == name]
        require(bindings == [digest], "Saved evaluation artifact changed")
    artifact = json.loads((directory / "screening_evaluation.json").read_text())
    pools = [audit_pool(directory, artifact, site) for site in ("sph", "ptb")]
    require(hashes == {name: sha256_file(directory / name) for name in names}, "Inputs changed during audit")
    return {"verified": True, "scope": "saved_reused_development_predictions_only",
            "input_sha256": {**hashes, "completion.json": sha256_file(directory / "completion.json")},
            "auditor_sha256": sha256_file(Path(__file__)), "pools": pools,
            "new_model_fits": 0, "closed_test_reads": 0}
