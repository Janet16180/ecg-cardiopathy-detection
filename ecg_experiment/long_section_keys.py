"""Prospective longer-window prototype comparison using frozen KEY-01 helpers."""

from __future__ import annotations

import subprocess
import time
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from . import ROOT
from .condition_key_inputs import json_read, verify_inputs
from .condition_key_inputs_v2 import load_inputs
from .condition_key_study import deadline
from .condition_keys import BATCH, macro_interval, ranking_metrics, refine, region_vectors, unit, whole_bank
from .files import sha256_file, write_json_atomic, write_npz_atomic
from .paths import to_stored

OUTPUT = ROOT / "outputs/long_section_keys_20261106"
PROTOCOL = "docs/long-section-keys-protocol-2026-11-06.md"
PREVIOUS = ROOT / "outputs/condition_keys_20261103_v2"


def windows(length: int) -> np.ndarray:
    """
    Enumerate all half-open windows of a frozen longer-section length.

    Parameters
    ----------
    length : int
        Ten, twenty or thirty tokens.

    Returns
    -------
    np.ndarray
        Ordered start and exclusive-stop positions.
    """
    if length not in (10, 20, 30):
        raise ValueError("Only prespecified lengths 10, 20 and 30 are admitted")
    return np.array([(start, start + length) for start in range(41 - length)])


def source_identity() -> dict[str, Any]:
    """
    Require committed protocol and scientific sources.

    Returns
    -------
    dict[str, Any]
        Commit and source hashes including reused predecessor sources.
    """
    own = [
        PROTOCOL,
        "ecg_experiment/long_section_keys.py",
        "scripts/experiments/run_long_section_keys.py",
        "tests/test_long_section_keys.py",
    ]
    hashes = dict(json_read(PREVIOUS / "result.json")["source_sha256"])
    for name in own:
        blob = subprocess.check_output(["git", "show", "HEAD:" + name], cwd=ROOT)
        if blob != (ROOT / name).read_bytes():
            raise ValueError(f"Uncommitted source: {name}")
        hashes[name] = sha256_file(ROOT / name)
    for name, digest in hashes.items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError(f"Changed frozen dependency: {name}")
    return {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": hashes,
        "protocol": PROTOCOL,
    }


def replay() -> tuple[dict, Any, dict]:
    """
    Recompute KEY-01 metrics and its primary interval exactly from verified private scores.

    Returns
    -------
    tuple[dict, Any, dict]
        Saved result, private arrays and exact-replay receipt.
    """
    old = json_read(PREVIOUS / "result.json")
    path = PREVIOUS / "artifacts.npz"
    if sha256_file(path) != old["artifact_sha256"][to_stored(path)]:
        raise ValueError("Changed KEY-01 predictions")
    arrays = np.load(path, allow_pickle=False)
    checked = 0
    for name in old["metrics"]:
        for normal, key in ((False, "metrics"), (True, "condition_vs_normal")):
            if ranking_metrics(arrays["query_flags"], arrays["score_" + name], normal) != old[key][name]:
                raise ValueError("Predecessor metric mismatch")
            checked += 1
    interval = macro_interval(
        arrays["query_patient"],
        arrays["query_flags"],
        arrays["score_window_selected_keys"],
        arrays["score_whole"],
    )
    if interval != old["primary"]:
        raise ValueError("Predecessor primary interval mismatch")
    return (
        old,
        arrays,
        {
            "exact_metric_dictionaries": checked,
            "primary_interval_exact": True,
            "result_sha256": sha256_file(PREVIOUS / "result.json"),
            "private_sha256": sha256_file(path),
        },
    )


def score_sections(
    tokens: np.ndarray, index: np.ndarray, scaler: Any, bank: dict, initial: dict, length: int
) -> tuple[dict, np.ndarray]:
    """
    Score fixed query sections, including unchanged-key controls at five seconds.

    Parameters
    ----------
    tokens : np.ndarray
        Cached contextual tokens.
    index : np.ndarray
        Permitted query positions.
    scaler : Any
        Training-only feature transformation.
    bank, initial : dict
        Refined and whole-record keys.
    length : int
        Window length in tokens.

    Returns
    -------
    tuple[dict, np.ndarray]
        Named score arrays and refined-key maximizing indices.
    """
    name = f"selected_{length / 4:g}s"
    scores = {name: np.empty((len(index), 4))}
    choices = np.empty((len(index), 4), dtype=np.int16)
    if length == 20:
        scores.update(
            {key: np.empty((len(index), 4)) for key in ("whole_keys_5s_max", "whole_keys_5s_median")}
        )
    for start in range(0, len(index), BATCH):
        stop = min(start + BATCH, len(index))
        z = region_vectors(tokens[index[start:stop]], scaler, windows(length))
        margins = z @ (bank["positive"] - bank["normal"]).T
        scores[name][start:stop] = margins.max(axis=1)
        choices[start:stop] = margins.argmax(axis=1)
        if length == 20:
            margins = z @ (initial["positive"] - initial["normal"]).T
            scores["whole_keys_5s_max"][start:stop] = margins.max(axis=1)
            scores["whole_keys_5s_median"][start:stop] = np.median(margins, axis=1)
    return scores, choices


def execute(started: float, identity: dict) -> dict:
    """
    Run the unchanged cohort comparison after exact predecessor replay.

    Parameters
    ----------
    started : float
        Monotonic start time.
    identity : dict
        Committed scientific source binding.

    Returns
    -------
    dict
        Completed aggregate result with conditional patient intervals.
    """
    inputs = verify_inputs()
    old, previous, integrity = replay()
    print("KEY-01 metrics and primary interval reproduced exactly", flush=True)
    data = load_inputs()
    if not np.array_equal(data["query"]["record_id"].to_numpy(dtype=str), previous["query_record_id"]):
        raise ValueError("Query ordering changed")
    scaler, initial = whole_bank(data["train_pooled"], data["train_flags"])
    whole = unit(scaler.transform(data["query_pooled"].astype(np.float64)))
    whole = whole @ (initial["positive"] - initial["normal"]).T
    error = float(np.max(np.abs(whole - previous["score_whole"])))
    if error > 1e-12 or ranking_metrics(data["query_flags"], whole) != old["metrics"]["whole"]:
        raise ValueError("Whole-record reproduction failed")
    integrity["whole_score_max_error"] = error
    before = time.perf_counter()
    for length in (10, 20, 30):
        region_vectors(data["train_tokens"][data["train_positions"][:BATCH]], scaler, windows(length))
    elapsed = time.perf_counter() - before
    profile = {"batch_seconds": elapsed, "forecast_seconds": elapsed * (25000 + 10434) / BATCH * 2}
    write_json_atomic(OUTPUT / "profile.json", profile)
    if profile["forecast_seconds"] > 1800:
        raise TimeoutError("Profile exceeds 1800 seconds")
    scores = {"whole": whole, "short_selected": previous["score_window_selected_keys"]}
    private = {"flags": data["query_flags"], "patients": previous["query_patient"]}
    for length in (10, 20, 30):
        bank, selected = refine(
            data["train_tokens"],
            data["train_positions"],
            data["train_flags"],
            scaler,
            initial,
            windows(length),
        )
        current, chosen = score_sections(
            data["query_tokens"], data["query_positions"], scaler, bank, initial, length
        )
        scores.update(current)
        private.update(
            {
                f"positive_{length}": bank["positive"],
                f"normal_{length}": bank["normal"],
                f"support_{length}": selected,
                f"query_{length}": chosen,
            }
        )
        print(f"Finished {length / 4:g}-second sections", flush=True)
        deadline(started)
    metrics = {name: ranking_metrics(data["query_flags"], value) for name, value in scores.items()}
    primary = macro_interval(
        previous["query_patient"], data["query_flags"], scores["selected_5s"], scores["short_selected"]
    )
    versus_whole = macro_interval(
        previous["query_patient"], data["query_flags"], scores["selected_5s"], whole
    )
    gates = {
        "gain_at_least_0.01": primary["difference"] >= 0.01,
        "lower_95_above_zero": primary["ci_low"] > 0,
        "within_0.01_of_whole_point": metrics["selected_5s"]["macro_auroc"]
        >= metrics["whole"]["macro_auroc"] - 0.01,
    }
    private.update({"score_" + name: value for name, value in scores.items()})
    write_npz_atomic(OUTPUT / "artifacts.npz", **private)
    result = {
        "study": "KEY-02",
        "status": "completed",
        **identity,
        "input_sha256": inputs,
        "integrity": integrity,
        "profile": profile,
        "metrics": metrics,
        "primary": primary,
        "versus_whole": versus_whole,
        "decision": {**gates, "advance": all(gates.values())},
        "condition_vs_normal": {
            name: ranking_metrics(data["query_flags"], value, True) for name, value in scores.items()
        },
        "counts": old["counts"],
        "wall_seconds": time.perf_counter() - started,
        "jepa_or_cropa_executed": False,
        "gpu_used": False,
        "artifact_sha256": {to_stored(OUTPUT / "artifacts.npz"): sha256_file(OUTPUT / "artifacts.npz")},
    }
    deadline(started)
    write_json_atomic(OUTPUT / "result.json", result)
    report = ROOT / "reports/long-section-keys-2026-11-06/aggregate.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(report, result)
    return result


def run() -> dict:
    """
    Execute once and preserve failures without modifying the predecessor.

    Returns
    -------
    dict
        Completed result.
    """
    identity = source_identity()
    OUTPUT.mkdir(exist_ok=False)
    write_json_atomic(OUTPUT / "started.json", identity)
    started = time.perf_counter()
    try:
        with threadpool_limits(2):
            return execute(started, identity)
    except Exception as error:
        write_json_atomic(OUTPUT / "failure.json", {"error": str(error)})
        raise
