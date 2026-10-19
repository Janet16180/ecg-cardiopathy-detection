"""Run CPU-only qualification, profiling, preparation or external comparison."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path


def main() -> None:
    """Dispatch one exclusive Experiment 042 execution stage."""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    from ecg_experiment import ROOT
    from ecg_experiment.measurement_screening042 import (
        MODELS,
        OUTPUT,
        PROTOCOL,
        evaluate,
        prepare,
        provenance,
        qualify,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["qualify", "profile", "prepare", "evaluate"])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--models", type=Path, default=MODELS)
    args = parser.parse_args()
    stage = {
        "qualify": "qualification",
        "profile": "profile",
        "prepare": "preparation",
        "evaluate": "evaluation",
    }
    directory = args.output / stage[args.stage]
    existed = directory.exists()
    started = time.perf_counter()
    try:
        if args.stage == "qualify":
            result = qualify(args.output, args.protocol)
        elif args.stage in {"profile", "prepare"}:
            result = prepare(args.output, args.protocol, profile_only=args.stage == "profile")
        else:
            result = evaluate(args.output, args.protocol, args.models)
    except Exception:
        if not existed:
            directory.mkdir(parents=True, exist_ok=True)
            failure = {
                "stage": args.stage,
                "status": "failed",
                "wall_seconds": time.perf_counter() - started,
                "traceback": traceback.format_exc().replace(str(ROOT), "."),
            }
            try:
                failure["provenance"] = provenance(args.protocol)
            except (OSError, ValueError) as error:
                failure["provenance_error"] = str(error).replace(str(ROOT), ".")
            with (directory / "failure.json").open("x") as handle:
                json.dump(failure, handle, indent=2)
        raise
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
