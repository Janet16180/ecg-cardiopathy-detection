"""Run candidate S-R of MLC-02 (xresnet1d101 from scratch) on train and validation records only.

Examples
--------
cd /tmp/ecg-mlc02-sr && PYTHONPATH=/tmp/ecg-mlc02-sr OMP_NUM_THREADS=4 .venv/bin/python \
    -m scripts.experiments.run_mlc02_scratch_network prepare
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/tmp/ecg-mlc02-sr OMP_NUM_THREADS=4 .venv/bin/python \
    -m scripts.experiments.run_mlc02_scratch_network plan
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ecg_experiment.files import read_json
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.mlc02_scratch_network import CANDIDATE_DIR, FAMILIES, SEEDS, full_precision
from ecg_experiment.mlc02_scratch_runs import (
    check_files,
    check_predict,
    overfit_check,
    prepare,
    profile,
    publish_lofo,
    publish_main,
    require_free_gpu,
    train_and_record,
)

GPU_LOCK = Path("/tmp/ecg_project_gpu.lock")
STEPS = (
    "prepare",
    "profile",
    "overfit",
    "train",
    "plan",
    "publish",
    "publish-lofo",
    "check-predict",
    "check-files",
)


def needed_mib() -> float:
    """
    Peak reserved GPU memory measured by the profile, or 4 GiB before a profile exists.

    Returns
    -------
    float
        MiB.
    """
    path = CANDIDATE_DIR / "profile.json"
    return read_json(path)["timing"]["peak_reserved_mib"] if path.exists() else 4096.0


def train_runs(runs: list[tuple[int, str | None]]) -> list[dict[str, object]]:
    """
    Train runs one after another under the GPU lock, checking free memory before each.

    Parameters
    ----------
    runs : list[tuple[int, str | None]]
        Seed and held-out family (None for a main run).

    Returns
    -------
    list[dict[str, object]]
        Run records.
    """
    records = []
    with gpu_lock("cuda", path=GPU_LOCK):
        for seed, family in runs:
            print(json.dumps({"run": [seed, family], "gpu": require_free_gpu(needed_mib())}), flush=True)
            records.append(train_and_record(seed, family, "cuda"))
    return records


def main(argv: list[str] | None = None) -> None:
    """
    Parse the command line and run one step.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or None for ``sys.argv``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=STEPS)
    parser.add_argument("--seed", type=int, default=SEEDS[0])
    parser.add_argument("--family", choices=FAMILIES, default=None)
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="check-files only")
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    full_precision()
    if args.step == "prepare":
        result = prepare()
    elif args.step == "check-files" and args.device == "cpu":
        result = check_files("cpu")
    elif args.step in ("profile", "overfit", "check-predict", "check-files"):
        with gpu_lock("cuda", path=GPU_LOCK):
            require_free_gpu(needed_mib())
            if args.step == "profile":
                result = profile("cuda", args.updates)
            elif args.step == "overfit":
                result = overfit_check("cuda")
            elif args.step == "check-files":
                result = check_files("cuda")
            else:
                result = check_predict("cuda")
    elif args.step == "train":
        result = train_runs([(args.seed, args.family)])
    elif args.step == "plan":
        result = train_runs([(seed, None) for seed in SEEDS] + [(SEEDS[0], family) for family in FAMILIES])
        result = {"main": publish_main(), "lofo": publish_lofo()}
    elif args.step == "publish":
        result = publish_main()
    else:
        result = publish_lofo()
    print(json.dumps(result, default=str)[:4000], flush=True)


if __name__ == "__main__":
    main()
