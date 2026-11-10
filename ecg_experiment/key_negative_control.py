"""Prospectively added negative-reference composition control for KEY-04."""

from __future__ import annotations

import subprocess
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from . import ROOT
from .condition_key_inputs import json_read
from .condition_key_inputs_v2 import load_inputs
from .condition_keys import GROUPS, macro_interval, ranking_metrics, unit, whole_bank
from .embedding_geometry import prototype
from .files import sha256_file, write_json_atomic, write_npz_atomic
from .paths import to_stored
from .pr74_screening_io import verify_hashes

OUT = ROOT / "outputs/condition_classifiers_negative_control_20261106"
PARENT = ROOT / "outputs/condition_classifiers_20261106"
PROTOCOL = "docs/condition-classifier-negative-control-2026-11-06.md"


def run() -> dict[str, Any]:
    """
    Evaluate all-other-negative prototypes without changing the primary comparison.

    Returns
    -------
    dict[str, Any]
        Frozen secondary control metrics, interval and provenance.
    """
    sources = {}
    for name in (
        PROTOCOL,
        "ecg_experiment/key_negative_control.py",
        "scripts/experiments/run_key_negative_control.py",
    ):
        if subprocess.check_output(["git", "show", "HEAD:" + name], cwd=ROOT) != (ROOT / name).read_bytes():
            raise ValueError(f"Uncommitted control source: {name}")
        sources[name] = sha256_file(ROOT / name)
    parent = json_read(PARENT / "result.json")
    if parent["status"] != "completed":
        raise ValueError("Parent study is not complete")
    verify_hashes(ROOT, parent["source_sha256"])
    verify_hashes(ROOT, parent["qualification"]["input_sha256"])
    verify_hashes(ROOT, parent["artifact_sha256"])
    data = load_inputs()
    with threadpool_limits(2):
        scaler, bank = whole_bank(data["train_pooled"], data["train_flags"])
        reference = unit(scaler.transform(data["train_pooled"].astype(float)))
        query = unit(scaler.transform(data["query_pooled"].astype(float)))
        old = np.load(ROOT / "outputs/condition_keys_20261103_v2/artifacts.npz", allow_pickle=False)
        if not np.allclose(bank["positive"], old["whole_positive"], atol=1e-12, rtol=0):
            raise ValueError("Positive reference keys changed")
        baseline = query @ (bank["positive"] - bank["normal"]).T
        error = float(np.max(np.abs(baseline - old["score_whole"])))
        if error > 1e-12:
            raise ValueError("Whole-record comparator changed")
        negative = np.stack([prototype(reference, data["train_flags"][:, c] == 0) for c in range(4)])
        scores = query @ (bank["positive"] - negative).T
        interval = macro_interval(
            data["query"]["patient_key"].to_numpy(), data["query_flags"], scores, baseline
        )
        result = {
            "study": "KEY-04-negative-control",
            "status": "completed",
            "exploratory": True,
            "protocol": PROTOCOL,
            "source_sha256": sources,
            "parent_result_sha256": sha256_file(PARENT / "result.json"),
            "input_sha256": parent["qualification"]["input_sha256"],
            "baseline_max_score_error": error,
            "positive_keys_unchanged": True,
            "training_counts": {
                g: {
                    "positive": int(data["train_flags"][:, c].sum()),
                    "negative": int((data["train_flags"][:, c] == 0).sum()),
                }
                for c, g in enumerate(GROUPS)
            },
            "metrics": ranking_metrics(data["query_flags"], scores),
            "condition_vs_normal": ranking_metrics(data["query_flags"], scores, True),
            "versus_whole": interval,
        }
    OUT.mkdir(exist_ok=False)
    write_npz_atomic(
        OUT / "artifacts.npz",
        scores=scores,
        positive=bank["positive"],
        negative=negative,
        flags=data["query_flags"],
    )
    result["artifact_sha256"] = {to_stored(OUT / "artifacts.npz"): sha256_file(OUT / "artifacts.npz")}
    write_json_atomic(OUT / "result.json", result)
    write_json_atomic(ROOT / "reports/condition-classifiers-2026-11-06/negative-control.json", result)
    return result
