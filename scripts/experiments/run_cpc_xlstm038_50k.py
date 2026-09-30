"""Run the conditional CPC xLSTM 50k tier with audited 25k provenance."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from ecg_experiment import ROOT
from ecg_experiment import xlstm_study as base
from ecg_experiment import xlstm_study_50k as study

STAGES = ("integrity", "prepare", "profile", "train", "readout", "audit", "run")


def execute(stage: str, root: Path, device: str) -> dict[str, Any]:
    """Execute one conditional 50k stage and record its actual wall time.

    Parameters
    ----------
    stage : str
        Explicit stage other than the convenience ``run`` sequence.
    root : Path
        Repository root containing both committed protocols.
    device : str
        CUDA device for profile, training and readout.

    Returns
    -------
    dict[str, Any]
        Stage receipt or development-only result.
    """
    if stage not in STAGES[:-1]:
        raise ValueError(f"Unknown Experiment 038 50k stage: {stage}")
    started = time.monotonic()
    with study.configured():
        cache = root / "data/processed/clean_50k_v3_cpc/complete.json"
        cache_missing_before = stage == "prepare" and not cache.exists()
        try:
            if stage in ("profile", "train", "readout"):
                base.configure_runtime()
            if stage == "readout":
                study.require_successful_train(root)
            if stage == "integrity":
                result = base.prior_integrity(root)
            elif stage == "prepare":
                result = base.prepare(root, 50)
            elif stage == "profile":
                result = base.profile(root, 50, device)
            elif stage == "train":
                result = base.train(root, 50, device)
            elif stage == "readout":
                result = base.readout(root, 50, device)
            else:
                result = base.audit(root, 50)
        except Exception:
            base.record_stage(root, 50, stage, time.monotonic() - started, "failed",
                              cache_build_included=cache_missing_before and cache.exists())
            raise
        base.record_stage(root, 50, stage, time.monotonic() - started, "complete",
                          cache_build_included=cache_missing_before and cache.exists())
        return result


def main() -> None:
    """Parse one 50k stage or run the complete conditional tier."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--tier", type=int, choices=(50,), default=50)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    stages = ("prepare", "profile", "train", "readout", "audit") if args.stage == "run" else (args.stage,)
    for stage in stages:
        result = execute(stage, args.root, args.device)
        print(json.dumps({"stage": stage, "tier": 50, "result": result}, default=str), flush=True)


if __name__ == "__main__":
    main()
