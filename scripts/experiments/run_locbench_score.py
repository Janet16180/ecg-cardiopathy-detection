"""
Score one method's maps on LocBench v1 and optionally compare it, paired, with saved methods.

Usage::

    PYTHONPATH=. .venv/bin/python -m scripts.experiments.run_locbench_score \
        --maps maps.npy --ids maps.ids.json --name my_method --out outputs/my_run/locbench \
        --compare outputs/experiment061_locbench_v1/tables/qrs_prior

``--maps`` is an .npy array (n, 12, 5000), float16 or float32, and ``--ids`` a JSON list of the ``ecg_id`` of
each row; it must cover ``ecg_experiment.locbench.scored_ecg_ids()``. The output folder receives
``summary.json`` (every metric with its interval, paired differences, input hashes) and ``tables/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.locbench import (
    evaluate_method,
    load_benchmark,
    load_tables,
    metric_names,
    paired_difference,
    save_tables,
)
from ecg_experiment.provenance import git_head

LOG = logging.getLogger("locbench_score")


def compare(tables: dict, folders: list[Path]) -> dict[str, dict]:
    """
    Paired differences of every shared metric against each saved method.

    Parameters
    ----------
    tables : dict
        Per-record tables of the scored method.
    folders : list[Path]
        ``save_tables`` folders of the methods to compare with.

    Returns
    -------
    dict[str, dict]
        Per compared method, per metric, the output of ``paired_difference`` (scored minus saved).
    """
    result = {}
    for folder in folders:
        other = load_tables(folder)
        shared = [metric for metric in metric_names(tables) if metric in set(metric_names(other))]
        result[other["name"]] = {metric: paired_difference(tables, other, metric) for metric in shared}
        LOG.info("compared with %s", other["name"])
    return result


def main() -> None:
    """Parse arguments, score the maps and write the summary and tables."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compare", type=Path, action="append", default=[])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    began = time.perf_counter()
    maps = np.load(args.maps, mmap_mode="r")
    ecg_ids = np.array(json.loads(args.ids.read_text()), dtype=np.int64)
    bench = load_benchmark()
    LOG.info("scoring %s: %d maps of dtype %s", args.name, len(maps), maps.dtype)
    summary, tables = evaluate_method(maps, ecg_ids, args.name, bench)
    save_tables(tables, args.out / "tables")
    summary["paired"] = compare(tables, args.compare)
    summary["inputs"] = {
        "maps": str(args.maps), "maps_sha256": sha256_file(args.maps), "maps_dtype": str(maps.dtype),
        "ids": str(args.ids), "ids_sha256": sha256_file(args.ids),
    }
    summary["benchmark_counts"] = bench.counts
    summary["code_commit"] = git_head(ROOT)
    summary["seconds"] = time.perf_counter() - began
    write_json_atomic(args.out / "summary.json", summary)
    LOG.info("A2 %.4f [%.4f, %.4f]", summary["metrics"]["A2"]["value"], summary["metrics"]["A2"]["ci_low"],
             summary["metrics"]["A2"]["ci_high"])


if __name__ == "__main__":
    main()
