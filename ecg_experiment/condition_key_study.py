"""Committed-source CPU experiment and aggregate export for KEY-01."""

from __future__ import annotations

import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from . import ROOT
from .condition_key_clusters import cluster_analysis
from .condition_key_inputs import load_inputs, pooling_check, predecessor_replay, verify_inputs
from .condition_keys import (
    BATCH,
    GROUPS,
    macro_interval,
    ranking_metrics,
    refine,
    region_scores,
    region_vectors,
    selected_vectors,
    spans,
    unit,
    whole_bank,
)
from .files import sha256_file, write_json_atomic, write_npz_atomic
from .paths import to_stored

PROTOCOL = Path("docs/condition-keys-protocol-2026-11-03.md")
OUTPUT = Path("outputs/condition_keys_20261103")
REPORT = Path("reports/condition-keys-2026-11-03/aggregate.json")


def source_binding() -> dict[str, Any]:
    """
    Require committed scientific implementation and bind shared source hashes.

    Returns
    -------
    dict[str, Any]
        Commit, protocol and source identities.
    """
    own = sorted((ROOT / "ecg_experiment").glob("condition_key*.py"))
    own += [ROOT / PROTOCOL, ROOT / "scripts/experiments/run_condition_keys.py",
            ROOT / "tests/test_condition_keys.py"]
    for path in own:
        relative = to_stored(path)
        committed = subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=ROOT)
        if committed != path.read_bytes():
            raise ValueError(f"Uncommitted scientific source: {relative}")
    shared = [ROOT / f"ecg_experiment/{name}.py" for name in
              ("embedding_geometry", "intervals", "paths", "files", "pr74_screening_metrics",
               "cpu_screening041")]
    return {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                  text=True).strip(),
            "protocol": to_stored(ROOT / PROTOCOL),
            "source_sha256": {to_stored(p): sha256_file(p) for p in own + shared}}


def deadline(started: float) -> None:
    """
    Enforce the protocol's complete-run ceiling at each stage boundary.

    Parameters
    ----------
    started : float
        Monotonic experiment start time.
    """
    if time.perf_counter() - started > 3600:
        raise TimeoutError("KEY-01 complete run exceeded 3600 seconds")


def profile(data: dict[str, Any], scaler: Any, initial: dict[str, np.ndarray]) -> dict[str, Any]:
    """
    Measure actual first-batch region processing before full bank construction.

    Parameters
    ----------
    data : dict[str, Any]
        Qualified training/query arrays.
    scaler : Any
        Training-only StandardScaler.
    initial : dict[str, np.ndarray]
        Whole-record keys.

    Returns
    -------
    dict[str, Any]
        Actual batch timing and conservative feature processing forecast.
    """
    started = time.perf_counter()
    part = data["train_tokens"][data["train_positions"][:BATCH]]
    for fixed in (False, True):
        z = region_vectors(part, scaler, spans(fixed))
        selected_vectors(z, data["train_flags"][:BATCH], initial)
        region_scores(z, initial)
    seconds = time.perf_counter() - started
    forecast = seconds * (len(data["train"]) + len(data["query"])) / BATCH * 1.5
    return {"records": len(part), "seconds": seconds, "forecast_seconds": forecast,
            "ceiling_seconds": 1800, "admitted": forecast <= 1800,
            "safety_factor": 1.5, "gpu_used": False}


def score_query(
    data: dict[str, Any], scaler: Any, banks: dict[str, dict[str, np.ndarray]]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Score every query without using its annotation in any region choice.

    Parameters
    ----------
    data : dict[str, Any]
        Qualified query cache positions and whole-record vectors.
    scaler : Any
        Training-fitted feature scaler.
    banks : dict[str, dict[str, np.ndarray]]
        Whole, variable-region and fixed-region banks.

    Returns
    -------
    tuple[dict[str, np.ndarray], dict[str, np.ndarray]]
        Scores per arm and maximizing window indices per region arm.
    """
    n = len(data["query"])
    whole = unit(scaler.transform(data["query_pooled"].astype(np.float64)))
    scores = {"whole": whole @ (banks["whole"]["positive"] - banks["whole"]["normal"]).T}
    for name in ("window_whole_keys", "window_selected_keys", "fixed_selected_keys", "shuffled_keys"):
        scores[name] = np.empty((n, 4))
    choices = {name: np.empty((n, 4), np.int16) for name in scores if name != "whole"}
    shuffled = {key: np.roll(value, 1, axis=0) for key, value in banks["window_selected_keys"].items()}
    for start in range(0, n, BATCH):
        stop = min(start + BATCH, n)
        raw = data["query_tokens"][data["query_positions"][start:stop]]
        z = region_vectors(raw, scaler, spans())
        for name, bank in (("window_whole_keys", banks["whole"]),
                           ("window_selected_keys", banks["window_selected_keys"]),
                           ("shuffled_keys", shuffled)):
            scores[name][start:stop], choices[name][start:stop] = region_scores(z, bank)
        fixed = region_vectors(raw, scaler, spans(True))
        scores["fixed_selected_keys"][start:stop], choices["fixed_selected_keys"][start:stop] = (
            region_scores(fixed, banks["fixed_selected_keys"]))
    return scores, choices


def selection_summary(choices: np.ndarray, flags: np.ndarray, windows: np.ndarray,
                      training: bool) -> dict[str, Any]:
    """
    Aggregate region selections without exporting per-record indices.

    Parameters
    ----------
    choices : np.ndarray
        Chosen candidate index per record and group.
    flags : np.ndarray
        Binary group annotations.
    windows : np.ndarray
        Candidate spans.
    training : bool
        Restrict counts to positive support examples for each group.

    Returns
    -------
    dict[str, Any]
        Start and length histograms per group.
    """
    summary = {}
    for column, name in enumerate(GROUPS):
        chosen = windows[choices[:, column]]
        if training:
            chosen = chosen[flags[:, column] == 1]
        lengths = chosen[:, 1] - chosen[:, 0]
        summary[name] = {"records": len(chosen), "starts": np.bincount(chosen[:, 0], minlength=40).tolist(),
                         "lengths": {str(k): int((lengths == k).sum()) for k in np.unique(lengths)}}
    return summary


def decision(primary: dict[str, Any], metrics: dict[str, Any]) -> dict[str, bool]:
    """
    Apply the three frozen region-selection gates.

    Parameters
    ----------
    primary : dict[str, Any]
        Paired primary macro-AUROC contrast.
    metrics : dict[str, Any]
        All-row ranking metrics for each fixed arm.

    Returns
    -------
    dict[str, bool]
        Individual gates and joint advancement verdict.
    """
    gates = {"gain_at_least_0.01": primary["difference"] >= 0.01,
             "lower_95_above_zero": primary["ci_low"] > 0,
             "at_least_fixed_window_point": metrics["window_selected_keys"]["macro_auroc"]
             >= metrics["fixed_selected_keys"]["macro_auroc"]}
    return {**gates, "advance": all(gates.values()), "pipeline_promoted": False}


def run() -> dict[str, Any]:
    """
    Execute KEY-01 once, preserving private arrays and exporting aggregate evidence.

    Returns
    -------
    dict[str, Any]
        Completed aggregate result or a raised exception with a preserved failure receipt.
    """
    directory = ROOT / OUTPUT
    directory.mkdir(exist_ok=False)
    started = time.perf_counter()
    binding = source_binding()
    write_json_atomic(directory / "started.json", binding)
    try:
        with threadpool_limits(2):
            return execute(directory, started, binding)
    except Exception as exc:
        write_json_atomic(directory / "failure.json", {"type": type(exc).__name__, "error": str(exc),
                                                      "wall_seconds": time.perf_counter() - started})
        raise


def execute(directory: Path, started: float, binding: dict[str, Any]) -> dict[str, Any]:
    """
    Run qualified construction, evaluation and descriptive clustering in order.

    Parameters
    ----------
    directory : Path
        Fresh private output directory.
    started : float
        Complete-run monotonic start.
    binding : dict[str, Any]
        Committed protocol and source identity.

    Returns
    -------
    dict[str, Any]
        Final aggregate output, also saved locally and in reports.
    """
    inputs = verify_inputs()
    replay = predecessor_replay()
    data = load_inputs()
    parity = {role: pooling_check(data[f"{role}_tokens"], data[f"{role}_positions"],
                                 data[f"{role}_pooled"]) for role in ("train", "query")}
    integrity = {"predecessor": replay, "pooling": parity, "input_sha256": inputs}
    write_json_atomic(directory / "integrity.json", integrity)
    print("Integrity passed; constructing training-only keys", flush=True)
    scaler, initial = whole_bank(data["train_pooled"], data["train_flags"])
    measured = profile(data, scaler, initial)
    write_json_atomic(directory / "profile.json", measured)
    if not measured["admitted"]:
        raise TimeoutError("Region feature profile exceeds the 1800-second gate")
    deadline(started)
    banks, selections = {"whole": initial}, {}
    for name, fixed in (("window_selected_keys", False), ("fixed_selected_keys", True)):
        banks[name], selections[name] = refine(data["train_tokens"], data["train_positions"],
                                               data["train_flags"], scaler, initial, spans(fixed))
        print(f"Constructed {name}", flush=True)
        deadline(started)
    scores, query_choices = score_query(data, scaler, banks)
    print("Scored SPH evaluation rows; computing frozen metrics", flush=True)
    deadline(started)
    metrics = {name: ranking_metrics(data["query_flags"], value) for name, value in scores.items()}
    normal_metrics = {name: ranking_metrics(data["query_flags"], value, True)
                      for name, value in scores.items()}
    primary = macro_interval(data["query"]["patient_key"].to_numpy(), data["query_flags"],
                             scores["window_selected_keys"], scores["whole"])
    deadline(started)
    print("Primary interval completed; fitting descriptive training-only clusters", flush=True)
    cluster, private = cluster_analysis(unit(scaler.transform(data["train_pooled"].astype(np.float64))),
                                        unit(scaler.transform(data["query_pooled"].astype(np.float64))),
                                        data["train_flags"], data["query_flags"],
                                        data["train"]["family"].to_numpy())
    arrays = {f"{name}_{key}": value for name, bank in banks.items() for key, value in bank.items()}
    arrays.update({"scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
                   "train_flags": data["train_flags"], "query_flags": data["query_flags"],
                   "train_record_id": data["train"]["record_id"].to_numpy(dtype=str),
                   "query_record_id": data["query"]["record_id"].to_numpy(dtype=str),
                   "query_patient": data["query"]["patient_key"].to_numpy(dtype=str), **private})
    arrays.update({f"score_{name}": value for name, value in scores.items()})
    arrays.update({f"support_{name}": value for name, value in selections.items()})
    arrays.update({f"query_{name}": value for name, value in query_choices.items()})
    write_npz_atomic(directory / "artifacts.npz", **arrays)
    selection = {name: {"support_positive": selection_summary(selections[name], data["train_flags"],
                                                              spans(name.startswith("fixed")), True),
                        "query_all": selection_summary(query_choices[name], data["query_flags"],
                                                        spans(name.startswith("fixed")), False)}
                 for name in selections}
    counts = {"training": len(data["train"]), "query": len(data["query"]),
              "query_patients": data["query"]["patient_key"].nunique(),
              "training_normals": int((~data["train_flags"].any(axis=1)).sum()),
              "query_normals": int((~data["query_flags"].any(axis=1)).sum()),
              "training_multiple_groups": int((data["train_flags"].sum(axis=1) > 1).sum()),
              "query_multiple_groups": int((data["query_flags"].sum(axis=1) > 1).sum()),
              "training_sources": data["train"]["source"].value_counts().to_dict(),
              "training_positive": dict(zip(GROUPS, data["train_flags"].sum(axis=0).tolist(), strict=True))}
    similarities = {name: (bank["positive"] @ bank["positive"].T).tolist() for name, bank in banks.items()}
    result = {"study": "KEY-01", "status": "completed", "scope": "reused_external_development",
              **binding, "input_sha256": inputs, "integrity": integrity, "profile": measured,
              "groups": list(GROUPS), "counts": counts, "metrics": metrics,
              "condition_vs_normal": normal_metrics, "primary": primary,
              "decision": decision(primary, metrics), "clusters": cluster, "selections": selection,
              "positive_key_cosines": similarities, "gpu_used": False, "encoder_fitted": False,
              "sam_executed": False, "unseen_disease_evaluated": False,
              "clinical_localization_evaluated": False, "wall_seconds": time.perf_counter() - started,
              "versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "sklearn": sklearn.__version__},
              "artifact_sha256": {to_stored(directory / "artifacts.npz"):
                                   sha256_file(directory / "artifacts.npz")}}
    deadline(started)
    if source_binding() != binding:
        raise ValueError("Scientific sources changed during execution")
    write_json_atomic(directory / "result.json", result)
    (ROOT / REPORT).parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(ROOT / REPORT, result)
    return result
