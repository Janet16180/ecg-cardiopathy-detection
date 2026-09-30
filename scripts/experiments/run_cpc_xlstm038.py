"""Execute the frozen Experiment 038 CPC GRU/xLSTM comparison."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from ecg_experiment import ROOT
from ecg_experiment import xlstm_study as study

STAGES = ("integrity", "prepare", "profile", "train", "readout", "audit", "run")


def execute(stage: str, root: Path, tier: int, device: str) -> dict[str, Any]:
    """Execute one measured stage and record failed attempts before raising.

    Parameters
    ----------
    stage : str
        One stage from ``STAGES`` except the convenience ``run`` sequence.
    root : Path
        Repository root containing the committed protocol and local data.
    tier : int
        Curated cohort size in thousands: 25 or 50.
    device : str
        Training device, which must be CUDA for profiled stages.

    Returns
    -------
    dict[str, Any]
        The stage's own receipt or result.
    """
    if stage not in STAGES[:-1]:
        raise ValueError(f"Unknown Experiment 038 stage: {stage}")
    if stage in ("profile", "train", "readout"):
        study.configure_runtime()
    cache = root / f"data/processed/clean_{tier}k_v3_cpc/complete.json"
    cache_missing_before = stage == "prepare" and not cache.exists()
    started = time.monotonic()
    try:
        if stage == "integrity":
            result = study.prior_integrity(root)
        elif stage == "prepare":
            result = study.prepare(root, tier)
        elif stage == "profile":
            result = study.profile(root, tier, device)
        elif stage == "train":
            result = study.train(root, tier, device)
        elif stage == "readout":
            result = study.readout(root, tier, device)
        else:
            result = study.audit(root, tier)
    except Exception:
        study.record_stage(root, tier, stage, time.monotonic() - started, "failed",
                           cache_build_included=cache_missing_before and cache.exists())
        raise
    study.record_stage(root, tier, stage, time.monotonic() - started, "complete",
                       cache_build_included=cache_missing_before and cache.exists())
    return result


def main() -> None:
    """Parse one explicit stage or run the full gated tier sequence."""
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
