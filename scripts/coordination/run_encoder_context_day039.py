"""Run the frozen CPC factorial sequentially and apply its one-shot diagnostic gate."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ecg_experiment import ROOT
from ecg_experiment import encoder_context_study039 as study
from ecg_experiment.encoder_context_analysis039 import CONTEXTS, OUTPUT, cell_path
from ecg_experiment.encoder_context_diagnostics039 import diagnose, feature_health
from ecg_experiment.files import write_json_atomic


def diagnostic_gate(root: Path, tier: int, encoder: str, seed: int,
                    result: dict[str, Any]) -> None:
    """Check an executed cell and run at most one training-only diagnosis per combination.

    Parameters
    ----------
    root : Path
        Repository root.
    tier : int
        Current cohort size in thousands.
    encoder : str
        Current waveform encoder.
    seed : int
        Current initialization seed.
    result : dict[str, Any]
        Audited development result.
    """
    path = cell_path(root, tier, encoder, seed)
    baseline = json.loads((cell_path(root, tier, "cnn", seed) / "result.json").read_text())
    for context in CONTEXTS:
        trigger = {"tier": tier, "encoder": encoder, "context": context, "seed": seed}
        with np.load(path / "features.npz", allow_pickle=False) as features:
            health = feature_health(features[f"train_{context}"])
        score = result["scores"]["limited"][context]["auroc"]
        difference = score - baseline["scores"]["limited"][context]["auroc"]
        bad_score = encoder != "cnn" and (difference <= -0.03 or score <= 0.60)
        if not bad_score and not health["collapsed"]:
            continue
        trigger.update({"auroc": score, "difference": difference,
                        "reason": "training_collapse" if health["collapsed"] else "severe_underperformance"})
        destination = root / OUTPUT / "diagnostics" / f"{encoder}_{context}" / "diagnostic.json"
        if destination.exists():
            continue
        if study.day_ledger(root)["total_seconds"] >= study.DAY_CEILING_SECONDS:
            raise RuntimeError("Day ceiling exhausted before the one-shot diagnosis")
        began = time.monotonic()
        diagnostic_status = "failed"
        try:
            receipt = diagnose(root, encoder, context, [trigger])
            diagnostic_status = "complete"
        finally:
            with study.configured(root, encoder, seed):
                study.record_stage(root, tier, encoder, seed, "diagnostic", time.monotonic() - began,
                                   diagnostic_status, study.base.record_stage)
        print(json.dumps({"diagnostic": f"{encoder}_{context}", "conclusion": receipt["conclusion"]}),
              flush=True)
        if receipt["specific_defect_identified"]:
            raise RuntimeError("A diagnosed defect requires its separate correction protocol before retry")


def main() -> None:
    """Execute the authorized fixed schedule and preserve a durable live status."""
    status_path = ROOT / OUTPUT / "status.json"
    completed = []
    for tier in study.TIERS:
        for seed in study.SEEDS:
            for encoder in study.ENCODERS:
                cell = {"tier": tier, "encoder": encoder, "seed": seed}
                write_json_atomic(status_path, {"status": "running", "current_cell": cell,
                                               "completed_cells": completed}, sort_keys=True)
                succeeded = False
                try:
                    result = study.run(ROOT, tier, encoder, seed)
                    diagnostic_gate(ROOT, tier, encoder, seed, result)
                    succeeded = True
                finally:
                    if not succeeded:
                        write_json_atomic(status_path, {"status": "failed", "current_cell": cell,
                                                       "completed_cells": completed}, sort_keys=True)
                completed.append(cell)
                print(json.dumps({"completed_cell": cell, "scores": result["scores"],
                                  "charged_seconds": study.day_ledger(ROOT)["total_seconds"]}), flush=True)
    write_json_atomic(status_path, {"status": "original_factorial_complete", "completed_cells": completed,
                                   "completed_fits": len(completed) * 2}, sort_keys=True)


if __name__ == "__main__":
    main()
