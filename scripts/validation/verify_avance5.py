"""Independently reconstruct the notebook's reported metrics from its local numeric predictions."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file, write_json_atomic


def check_bindings(summary: dict[str, Any]) -> int:
    """Verify current source and numeric-input identities against their captured receipts."""
    checks = 0
    for path, expected in {**summary["source_sha256"], **summary["input_sha256"]}.items():
        if sha256_file(ROOT / path) != expected:
            raise ValueError("Delivery source/input changed")
        checks += 1
    prior = ROOT / "outputs/screen01_matched_transfer_20261027/cuda_runtime_execution"
    for stage in ("qualify", "extract"):
        receipt = json.loads((prior / stage / "completion.json").read_text())
        for path, expected in receipt["artifacts"].items():
            if not path.endswith((".csv", ".npy")):
                continue
            if sha256_file(ROOT / path) != expected:
                raise ValueError("Frozen development cache receipt no longer matches")
            checks += 1
    return checks


def check_predictions(records: list[dict[str, Any]]) -> int:
    """Recompute every matrix/rate/ranking statistic directly with independent sklearn calls."""
    checks = 0
    for site in ("ptb", "sph"):
        with np.load(ROOT / "outputs/avance5_final" / f"{site}_predictions.npz",
                     allow_pickle=False) as predictions:
            y, held = predictions["target"], predictions["evaluation"]
            for row in records:
                if row["sitio"] != site:
                    continue
                scores = predictions[row["id"]]
                negative = scores[~held & (y == 0)]
                if (negative > row["umbral"]).sum() > int(len(negative) * 0.05):
                    raise ValueError("Strict normal-calibration budget exceeded")
                matrix = confusion_matrix(y[held], scores[held] > row["umbral"], labels=[0, 1])
                tn, fp, fn, tp = (int(value) for value in matrix.ravel())
                observed = {"tn": tn, "fp": fp, "fn": fn, "tp": tp,
                            "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
                            "auroc": roc_auc_score(y[held], scores[held]),
                            "average_precision": average_precision_score(y[held], scores[held])}
                for key, value in observed.items():
                    if abs(value - row[key]) > 1e-12:
                        raise ValueError(f"Independent metric mismatch: {site}/{row['id']}/{key}")
                    checks += 1
                checks += 1
    return checks


def main() -> None:
    """Check receipt-bound inputs, strict normal budgets, matrices and both ranking metrics."""
    summary = json.loads((ROOT / "reports/avance5_final/summary.json").read_text())
    checks = check_bindings(summary) + check_predictions(summary["records"])
    notebook = json.loads((ROOT / "notebooks/Avance5.8.ipynb").read_text())
    counts = [cell["execution_count"] for cell in notebook["cells"] if cell["cell_type"] == "code"]
    if counts != list(range(1, len(counts) + 1)):
        raise ValueError("Notebook execution is not sequential")
    if any(output["output_type"] == "error" for cell in notebook["cells"]
           for output in cell.get("outputs", [])):
        raise ValueError("Notebook contains execution errors")
    result = {"passed": True, "independent_checks": checks + 2, "model_site_comparisons": 18,
              "sequential_code_cells": len(counts), "existing_model_deserialization": False,
              "new_model_fit": False, "waveform_read": False}
    write_json_atomic(ROOT / "reports/avance5_final/verification.json", result)
    print(result)


if __name__ == "__main__":
    main()
