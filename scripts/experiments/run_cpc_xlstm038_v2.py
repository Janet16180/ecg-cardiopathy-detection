"""Execute Experiment 038 v2 with a checkpointable native GRU backend."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from ecg_experiment import ROOT
from ecg_experiment import xlstm_study as base
from ecg_experiment import xlstm_study_v2 as study_v2

STAGES = ("integrity", "prepare", "profile", "train", "readout", "audit", "run")


def execute(stage: str, root: Path, tier: int, device: str) -> dict[str, Any]:
    """Run one versioned stage and charge its elapsed time even on failure.

    Parameters
    ----------
    stage : str
        Explicit experiment stage other than ``run``.
    root : Path
        Repository root with the committed v2 protocol.
    tier : int
        Curated cohort size in thousands.
    device : str
        CUDA for the real profile, train and readout stages.

    Returns
    -------
    dict[str, Any]
        Stage receipt or development-only result.
    """
    if stage not in STAGES[:-1]:
        raise ValueError(f"Unknown Experiment 038 v2 stage: {stage}")
    started = time.monotonic()
    with study_v2.configured():
        cache = root / f"data/processed/clean_{tier}k_v3_cpc/complete.json"
        cache_missing_before = stage == "prepare" and not cache.exists()
        try:
            study_v2.carry_v1_attempts(root, tier)
            if stage in ("profile", "train", "readout"):
                base.configure_runtime()
            if stage == "integrity":
                result = base.prior_integrity(root)
            elif stage == "prepare":
                result = base.prepare(root, tier)
            elif stage == "profile":
                result = base.profile(root, tier, device)
            elif stage == "train":
                result = base.train(root, tier, device)
            elif stage == "readout":
                result = base.readout(root, tier, device)
            else:
                result = base.audit(root, tier)
        except Exception:
            base.record_stage(root, tier, stage, time.monotonic() - started, "failed",
                              cache_build_included=cache_missing_before and cache.exists())
            raise
        base.record_stage(root, tier, stage, time.monotonic() - started, "complete",
                          cache_build_included=cache_missing_before and cache.exists())
        return result


def main() -> None:
    """Parse an explicit v2 stage or run the complete gated tier sequence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--tier", type=int, choices=(25, 50), default=25)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    stages = ("prepare", "profile", "train", "readout", "audit") if args.stage == "run" else (args.stage,)
    for stage in stages:
        result = execute(stage, args.root, args.tier, args.device)
        print(json.dumps({"stage": stage, "tier": args.tier, "result": result}, default=str), flush=True)


if __name__ == "__main__":
    main()
