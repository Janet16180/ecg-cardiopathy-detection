"""KEY-01 orchestration with strict decimal-string membership qualification."""

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
from .condition_key_inputs import json_read, pooling_check, predecessor_replay, verify_inputs
from .condition_key_inputs_v2 import load_inputs
from .condition_key_study import (
    REPORT,
    deadline,
    decision,
    profile,
    score_query,
    selection_summary,
    source_binding,
)
from .condition_keys import GROUPS, macro_interval, ranking_metrics, refine, spans, unit, whole_bank
from .files import sha256_file, write_json_atomic, write_npz_atomic
from .paths import to_stored

OUTPUT = Path("outputs/condition_keys_20261103_v2")
AMENDMENT = Path("docs/condition-keys-execution-v2-2026-11-03.md")


def binding() -> dict[str, Any]:
    """
    Bind both protocols and the committed successor implementation.

    Returns
    -------
    dict[str, Any]
        Original scientific bindings plus correction-specific identities.
    """
    value = source_binding()
    for path in (AMENDMENT, Path("tests/test_condition_key_inputs_v2.py"),
                 Path("scripts/experiments/run_condition_keys_v2.py")):
        committed = subprocess.check_output(["git", "show", "HEAD:" + str(path)], cwd=ROOT)
        if committed != (ROOT / path).read_bytes():
            raise ValueError(f"Uncommitted successor source: {path}")
        value["source_sha256"][to_stored(ROOT / path)] = sha256_file(ROOT / path)
    value["execution_correction"] = str(AMENDMENT)
    return value


def counts(data: dict[str, Any]) -> dict[str, Any]:
    """
    Count fixed cohorts and coexisting annotation groups.

    Parameters
    ----------
    data : dict[str, Any]
        Qualified training and evaluation inputs.

    Returns
    -------
    dict[str, Any]
        Publishable aggregate denominators.
    """
    train, query = data["train_flags"], data["query_flags"]
    return {"training": len(train), "query": len(query),
            "query_patients": int(data["query"]["patient_key"].nunique()),
            "training_normals": int((~train.any(axis=1)).sum()),
            "query_normals": int((~query.any(axis=1)).sum()),
            "training_multiple_groups": int((train.sum(axis=1) > 1).sum()),
            "query_multiple_groups": int((query.sum(axis=1) > 1).sum()),
            "training_sources": data["train"]["source"].value_counts().to_dict(),
            "training_positive": dict(zip(GROUPS, train.sum(axis=0).tolist(), strict=True))}


def construct(data: dict[str, Any], directory: Path, started: float) -> tuple[Any, dict, dict, dict, dict]:
    """
    Profile, construct both region banks and score the fixed query role.

    Parameters
    ----------
    data : dict[str, Any]
        Qualified inputs.
    directory : Path
        Fresh successor output directory.
    started : float
        Complete-run monotonic start.

    Returns
    -------
    tuple[Any, dict, dict, dict, dict]
        Scaler, banks, support selections, query scores and query selections.
    """
    scaler, initial = whole_bank(data["train_pooled"], data["train_flags"])
    measured = profile(data, scaler, initial)
    write_json_atomic(directory / "profile.json", measured)
    print("Real CPU profile:", measured, flush=True)
    if not measured["admitted"]:
        raise TimeoutError("Region processing exceeds the 1800-second profile gate")
    banks, support = {"whole": initial}, {}
    for name, fixed in (("window_selected_keys", False), ("fixed_selected_keys", True)):
        banks[name], support[name] = refine(data["train_tokens"], data["train_positions"],
                                            data["train_flags"], scaler, initial, spans(fixed))
        print(f"Constructed {name}", flush=True)
        deadline(started)
    scores, query = score_query(data, scaler, banks)
    print("All fixed SPH query scores computed", flush=True)
    deadline(started)
    return scaler, banks, support, scores, query


def analyze(data: dict[str, Any], directory: Path, started: float) -> dict[str, Any]:
    """
    Evaluate fixed predictions and describe training-fitted clusters.

    Parameters
    ----------
    data : dict[str, Any]
        Qualified training and evaluation inputs.
    directory : Path
        Fresh successor output directory.
    started : float
        Complete-run start.

    Returns
    -------
    dict[str, Any]
        Statistical results, descriptive clusters and private artifact identity.
    """
    scaler, banks, support, scores, query = construct(data, directory, started)
    metrics = {name: ranking_metrics(data["query_flags"], values) for name, values in scores.items()}
    easier = {name: ranking_metrics(data["query_flags"], values, True) for name, values in scores.items()}
    primary = macro_interval(data["query"]["patient_key"].to_numpy(), data["query_flags"],
                             scores["window_selected_keys"], scores["whole"])
    print("Primary paired interval:", primary, flush=True)
    deadline(started)
    clusters, private = cluster_analysis(unit(scaler.transform(data["train_pooled"].astype(np.float64))),
                                         unit(scaler.transform(data["query_pooled"].astype(np.float64))),
                                         data["train_flags"], data["query_flags"],
                                         data["train"]["family"].to_numpy())
    arrays = {f"{arm}_{kind}": value for arm, bank in banks.items() for kind, value in bank.items()}
    arrays.update({"scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
                   "train_flags": data["train_flags"], "query_flags": data["query_flags"],
                   "train_record_id": data["train"]["record_id"].to_numpy(dtype=str),
                   "query_record_id": data["query"]["record_id"].to_numpy(dtype=str),
                   "query_patient": data["query"]["patient_key"].to_numpy(dtype=str), **private})
    arrays.update({f"score_{arm}": value for arm, value in scores.items()})
    arrays.update({f"support_{arm}": value for arm, value in support.items()})
    arrays.update({f"query_{arm}": value for arm, value in query.items()})
    write_npz_atomic(directory / "artifacts.npz", **arrays)
    selection = {arm: {"support_positive": selection_summary(support[arm], data["train_flags"],
                                                             spans(arm.startswith("fixed")), True),
                       "query_all": selection_summary(query[arm], data["query_flags"],
                                                       spans(arm.startswith("fixed")), False)}
                 for arm in support}
    return {"groups": list(GROUPS), "counts": counts(data), "metrics": metrics,
            "condition_vs_normal": easier, "primary": primary, "decision": decision(primary, metrics),
            "clusters": clusters, "selections": selection,
            "positive_key_cosines": {arm: (bank["positive"] @ bank["positive"].T).tolist()
                                     for arm, bank in banks.items()},
            "artifact_sha256": {to_stored(directory / "artifacts.npz"):
                                 sha256_file(directory / "artifacts.npz")}}


def run() -> dict[str, Any]:
    """
    Execute the corrected qualification and unchanged prospective KEY-01 once.

    Returns
    -------
    dict[str, Any]
        Completed aggregate; errors preserve a separate failure receipt and raise.
    """
    directory = ROOT / OUTPUT
    directory.mkdir(exist_ok=False)
    started = time.perf_counter()
    sources = binding()
    write_json_atomic(directory / "started.json", sources)
    try:
        with threadpool_limits(2):
            inputs = verify_inputs()
            replay = predecessor_replay()
            data = load_inputs()
            parity = {role: pooling_check(data[f"{role}_tokens"], data[f"{role}_positions"],
                                         data[f"{role}_pooled"]) for role in ("train", "query")}
            integrity = {"predecessor": replay, "pooling": parity, "input_sha256": inputs}
            write_json_atomic(directory / "integrity.json", integrity)
            print("Corrected qualification passed", flush=True)
            analysis = analyze(data, directory, started)
            result = {"study": "KEY-01", "status": "completed", "scope": "reused_external_development",
                      **sources, **analysis, "input_sha256": inputs, "integrity": integrity,
                      "profile": json_read(directory / "profile.json"),
                      "gpu_used": False, "encoder_fitted": False, "sam_executed": False,
                      "unseen_disease_evaluated": False, "clinical_localization_evaluated": False,
                      "wall_seconds": time.perf_counter() - started,
                      "versions": {"python": platform.python_version(), "numpy": np.__version__,
                                   "sklearn": sklearn.__version__}}
            deadline(started)
            if binding() != sources:
                raise ValueError("Scientific sources changed during execution")
            write_json_atomic(directory / "result.json", result)
            (ROOT / REPORT).parent.mkdir(parents=True, exist_ok=True)
            write_json_atomic(ROOT / REPORT, result)
            return result
    except Exception as exc:
        write_json_atomic(directory / "failure.json", {"type": type(exc).__name__, "error": str(exc),
                                                      "wall_seconds": time.perf_counter() - started})
        raise
