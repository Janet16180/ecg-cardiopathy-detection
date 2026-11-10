"""Run Stage 1 of MLC-02 (inputs, no score): cache, feature profiles and runs, publication, integrity.

Examples
--------
PYTHONPATH=. OPENBLAS_CORETYPE=Haswell OMP_NUM_THREADS=1 .venv/bin/python \
    -m scripts.experiments.run_mlc02_inputs cache --workers 16
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python -m scripts.experiments.run_mlc02_inputs profile \
    --extractor xecg --gpu 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ecg_experiment.mlc02_cache_build import build, verify
from ecg_experiment.mlc02_features import FEATURES, profile_positions
from ecg_experiment.mlc02_stage1 import OUTPUT, PUBLISHED, extract, integrity, key04, profile, publish
from ecg_experiment.mlc02_waveforms import CACHE


def main(argv: list[str] | None = None) -> None:
    """
    Parse the command line and run one step.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or None for ``sys.argv``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("cache", "profile", "extract", "publish", "integrity", "key04"))
    parser.add_argument("--extractor", choices=("xecg", "jepa", "cpc", "measurements"))
    parser.add_argument("--gpu", type=int, choices=(0, 1))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, help="cache: first records per source only (profile)")
    parser.add_argument("--directory", type=Path, default=CACHE, help="cache: output directory")
    parser.add_argument("--profile", action="store_true", help="integrity: compare the profile records")
    args = parser.parse_args(argv)
    command = sys.argv if argv is None else ["run_mlc02_inputs", *argv]
    if args.step == "cache":
        print(json.dumps(build(args.directory, args.workers, args.limit), indent=1))
        receipt = verify(args.directory, args.workers, command)
        print(
            json.dumps({key: receipt[key] for key in ("records", "sha256_cache_ok", "verify_wall_seconds")})
        )
    elif args.step in ("profile", "extract"):
        if args.extractor is None or (args.extractor != "measurements" and args.gpu is None):
            parser.error("profile and extract need --extractor, and --gpu for an encoder")
        function = profile if args.step == "profile" else extract
        result = function(args.extractor, args.gpu, args.workers)
        print(json.dumps({key: value for key, value in result.items() if key != "environment"}, indent=1))
    elif args.step == "publish":
        print(json.dumps(publish(command)["files_sha256"], indent=1))
    elif args.step == "integrity":
        if args.profile:
            index = pd.read_csv(CACHE / "index.csv", dtype={"record_key": str})
            table = integrity(FEATURES / "profile", FEATURES / "profile", profile_positions(index))
        else:
            table = integrity(PUBLISHED, OUTPUT / "integrity", None)
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(table.drop(columns=["features_sha256"]).to_string())
    else:
        record = key04(OUTPUT / "integrity")
        print(
            json.dumps(
                {key: value for key, value in record.items() if key != "git"},
                indent=1,
                default=lambda value: value.tolist() if isinstance(value, np.ndarray) else str(value),
            )
        )


if __name__ == "__main__":
    main()
