"""
Stage 3 of MLC-02 on validation data: the selection record, the operating points and the test guards.

The operating points are fixed on validation, and the guards keep the test roles closed until both exist.

Rules (``docs/mlc02-condition-models-protocol-2026-11-10.md``, Stage 3 and Addendum 1):

- The selection metric is ``mlc02_evaluation.selection_metric`` on the validation records, recomputed from
  each candidate's saved validation prediction file and checked against the value its Stage 2 run stored.
- E is the mean probability of the best member of each of the three best candidate families (``FAMILIES``):
  a family is a set of candidates with the same input features and different heads. A family's best member
  has the highest selection metric (the earliest in the protocol table on exact equality), and families are
  ranked by that member's metric. The selection tie margin does not apply inside E's construction.
- The selected model has the highest selection metric; candidates within 0.001 of it are tied and the earliest
  in the protocol table is selected (``mlc02_evaluation.select_candidate``).
- Operating points: per candidate and condition, the threshold of 90% specificity on the validation records
  where the condition is defined, pooled over sources, and a Platt calibration fitted on the same records by
  damped Newton (``mlc02_calibration``; the library's undamped fit does not converge on some of them). Raw
  probabilities stay primary.

Guards: ``select`` and ``fix`` read validation predictions and validation labels only. They refuse to replace
an existing record unless asked, and never once a test prediction file exists (``test_prediction_files``).
``check_record_committed`` is the check ``test-predict`` makes before it opens a test role.
"""

from __future__ import annotations

import math
import os
import platform
import subprocess
import time
from collections.abc import Iterable
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from ecg_experiment import ROOT
from ecg_experiment.files import read_json, sha256_file, write_json_atomic, write_text_atomic
from ecg_experiment.mlc02_bootstrap import labels_digest
from ecg_experiment.mlc02_cache_build import code_sha256, git_state
from ecg_experiment.mlc02_calibration import fit_platt_damped
from ecg_experiment.mlc02_data import (
    CONDITIONS,
    MANIFEST,
    load_manifest,
    load_predictions,
    prediction_path,
    role_data,
    save_predictions,
)
from ecg_experiment.mlc02_evaluation import (
    CANDIDATES,
    TARGET_SPECIFICITY,
    TIE_MARGIN,
    selection_metric,
    selection_record,
    specificity_thresholds,
    threshold_rates,
)
from ecg_experiment.paths import from_stored, to_stored

OUTPUT = ROOT / "outputs/mlc02_condition_models_20261110"
CANDIDATES_DIR = OUTPUT / "candidates"
STAGE3 = OUTPUT / "stage3"
RECORD_DOC = ROOT / "docs/mlc02-selection-record-2026-11-10.md"
RECORD_NAME = "selection_record.json"
OPERATING_NAME = "operating_points.json"
TEST_RECEIPT_NAME = "test_predictions_receipt.json"
PREDICTIONS = "predictions"
RECEIPTS = "receipts"
TEST_ROLES = ("test_same_source", "test_external", "sph_calibration")
ENSEMBLE = "E"
BASE_CANDIDATES = tuple(name for name in CANDIDATES if name != ENSEMBLE)
FAMILIES = {
    "P0": "xecg",
    "H-X": "xecg",
    "F": "xecg",
    "L-J": "jepa",
    "H-J": "jepa",
    "L-C": "cpc",
    "H-C": "cpc",
    "G-M": "measurements",
    "H-All": "all_features",
    "S-R": "scratch_waveform",
}
ENSEMBLE_SIZE = 3
STORED_TOLERANCE = 1e-12
SETTINGS_KEYS = {
    "P0": ("chosen_c",),
    "L-J": ("chosen_c",),
    "L-C": ("chosen_c",),
    "G-M": ("chosen",),
    "H-X": ("chosen", "epoch_by_seed"),
    "H-J": ("chosen", "epoch_by_seed"),
    "H-C": ("chosen", "epoch_by_seed"),
    "H-All": ("chosen", "epoch_by_seed"),
    "F": ("chosen",),
}
SELECTION_RULE = (
    "Highest mean over validation sources of the within-source macro AUROC; candidates within "
    f"{TIE_MARGIN} of the highest are tied and the earliest in the protocol table is selected."
)
ENSEMBLE_RULE = (
    "E is the mean probability, in protocol-table order, of the best member of each of the three best "
    "candidate families. A family is a set of candidates with the same input features and a different head: "
    "xECG (P0, H-X, F), ECG-JEPA (L-J, H-J), ECG-CPC (L-C, H-C); G-M, H-All and S-R are each their own "
    "family. A family's best member is its candidate with the highest validation selection metric (the "
    "earliest in the protocol table on exact equality), and families are ranked by that member's metric. The "
    "0.001 tie margin of the selection rule does not apply inside E's construction. E's prediction files "
    "carry the union of its members' seeds."
)
THRESHOLD_RULE = (
    "Per candidate and condition: the lowest observed validation score whose specificity is at "
    f"least {TARGET_SPECIFICITY} on the validation records where the condition is defined, pooled "
    "over validation sources (mlc02_evaluation.specificity_threshold); p >= threshold is positive."
)
PLATT_RULE = (
    "Per candidate and condition: sigmoid(a + b logit(p)) fitted by maximum likelihood on the "
    "validation records where the condition is defined, by damped Newton "
    "(mlc02_calibration.fit_platt_damped; mlc02_evaluation.fit_platt, the same estimator, does not converge "
    "on 13 of the 140 validation fits). Raw probabilities stay primary (Addendum 1)."
)
CODE = (
    "ecg_experiment/mlc02_stage3.py",
    "ecg_experiment/mlc02_stage3_predict.py",
    "ecg_experiment/mlc02_stage3_evaluate.py",
    "ecg_experiment/mlc02_calibration.py",
    "scripts/experiments/run_mlc02_stage3.py",
    "ecg_experiment/mlc02_data.py",
    "ecg_experiment/mlc02_evaluation.py",
    "ecg_experiment/mlc02_bootstrap.py",
    "ecg_experiment/mlc02_feature_data.py",
    "ecg_experiment/mlc02_feature_models.py",
    "ecg_experiment/mlc02_scratch_network.py",
    "ecg_experiment/intervals.py",
    "ecg_experiment/paths.py",
    "ecg_experiment/files.py",
)
PACKAGES = ("numpy", "scipy", "scikit-learn", "pandas")
ENVIRONMENT = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_CORETYPE",
    "CUDA_VISIBLE_DEVICES",
    "CUBLAS_WORKSPACE_CONFIG",
)


def stored(path: Path | str) -> str:
    """
    Path as written into a record: repository-relative inside the repository, absolute outside it.

    Parameters
    ----------
    path : Path | str
        Any path.

    Returns
    -------
    str
        ``paths.to_stored`` form, or the absolute path for scratch directories outside the repository.
    """
    try:
        return to_stored(path)
    except ValueError:
        return os.path.abspath(path)


def json_ready(value: Any) -> Any:
    """
    Value with NumPy types made plain and non-finite floats written as None, for strict JSON.

    Parameters
    ----------
    value : Any
        Nested dicts, lists, tuples, arrays and scalars.

    Returns
    -------
    Any
        The same structure with only JSON types.
    """
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, Path):
        return stored(value)
    return value


def environment() -> dict[str, Any]:
    """
    Interpreter, package versions and thread settings.

    Returns
    -------
    dict[str, Any]
        ``python``, ``machine``, ``cpus``, ``packages`` and ``variables`` (unset ones as None).
    """
    return {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "cpus": os.cpu_count(),
        "packages": {name: version(name) for name in PACKAGES},
        "variables": {name: os.environ.get(name) for name in ENVIRONMENT},
    }


def code_state(require_committed: bool, code: tuple[str, ...] = CODE) -> dict[str, Any]:
    """
    Commit, modified tracked files and code SHA-256 of a run.

    Parameters
    ----------
    require_committed : bool
        Raise unless every code file is committed and no tracked file is modified.
    code : tuple[str, ...]
        Repository-relative code files.

    Returns
    -------
    dict[str, Any]
        ``commit``, ``modified_tracked``, ``uncommitted_code`` and ``code_sha256``.

    Raises
    ------
    RuntimeError
        If ``require_committed`` and the code is not committed.
    """
    state = git_state()
    status = subprocess.check_output(["git", "status", "--porcelain", "--", *code], cwd=ROOT, text=True)
    uncommitted = [line[3:] for line in status.splitlines()]
    if require_committed and (uncommitted or state["modified_tracked"]):
        raise RuntimeError(f"commit the code before this step: {uncommitted or state['modified_tracked']}")
    return {**state, "uncommitted_code": uncommitted, "code_sha256": code_sha256(code)}


def test_prediction_files(roots: Iterable[Path]) -> list[Path]:
    """
    Every prediction file of a test role under the given directories.

    A file counts when its name starts with ``<test role>__`` (``mlc02_data.prediction_path``), at any depth.

    Parameters
    ----------
    roots : Iterable[Path]
        Directories to search; missing ones are skipped.

    Returns
    -------
    list[Path]
        Sorted, distinct paths.
    """
    found = set()
    for root in roots:
        if Path(root).is_dir():
            for role in TEST_ROLES:
                found.update(Path(root).rglob(f"{role}__*"))
    return sorted(found)


def refuse_after_tests(roots: Iterable[Path], step: str) -> None:
    """
    Refuse a step once any test prediction file exists.

    Parameters
    ----------
    roots : Iterable[Path]
        Directories searched by ``test_prediction_files``.
    step : str
        Step name for the message.

    Raises
    ------
    PermissionError
        If a test prediction file exists.
    """
    found = test_prediction_files(roots)
    if found:
        raise PermissionError(
            f"{step} is closed: test predictions exist ({len(found)} files, first {found[0]})"
        )


def guard_overwrite(path: Path, overwrite_before_tests: bool, roots: Iterable[Path], step: str) -> None:
    """
    Allow replacing ``path`` only on request and only while no test prediction exists.

    Parameters
    ----------
    path : Path
        File the step would write.
    overwrite_before_tests : bool
        The ``--overwrite-before-tests`` flag.
    roots : Iterable[Path]
        Directories searched by ``test_prediction_files``.
    step : str
        Step name for the messages.

    Raises
    ------
    PermissionError
        If any test prediction file exists (always checked).
    FileExistsError
        If ``path`` exists and the flag is not given.
    """
    refuse_after_tests(roots, step)
    if Path(path).exists() and not overwrite_before_tests:
        raise FileExistsError(
            f"{path} exists; pass --overwrite-before-tests to replace it before any test run"
        )


def validation_data(manifest: Path = MANIFEST) -> dict[str, np.ndarray]:
    """
    Manifest arrays of the validation records only.

    The manifest is one file for all roles; the rows of other roles are dropped right after parsing, before
    any use.

    Parameters
    ----------
    manifest : Path
        MLC-01 ``manifest.csv``.

    Returns
    -------
    dict[str, np.ndarray]
        ``load_manifest`` arrays of role ``validation``, in manifest order.
    """
    return role_data(load_manifest(manifest), "validation")


def candidate_info(candidate: str, directory: Path = CANDIDATES_DIR) -> dict[str, Any] | None:
    """
    Seeds, validation file, stored selection metric and chosen settings of one Stage 2 candidate.

    A feature candidate or F is present when ``<id>/model.json`` exists, S-R when ``S-R/settings.json``
    exists.
    The candidate's prediction is the file of all its seeds (``mlc02_data.prediction_path``).

    Parameters
    ----------
    candidate : str
        One of ``BASE_CANDIDATES``.
    directory : Path
        Directory holding ``<id>/``.

    Returns
    -------
    dict[str, Any] | None
        ``id``, ``family``, ``seeds``, ``validation_file``, ``stored_selection_metric`` and ``settings``; None
        if the candidate is absent.

    Raises
    ------
    ValueError
        If the candidate is present but its validation file or record is missing, or its record says it is a
        smoke run or scored a test role.
    """
    folder = Path(directory) / candidate
    if candidate == "S-R":
        if not (folder / "settings.json").exists():
            return None
        meta = read_json(folder / "results.json")
        seeds = tuple(int(run["seed"]) for run in read_json(folder / "settings.json")["seeds"])
        stored_metric = meta["selection"]["mean"]["selection_metric"]
        settings = {"chosen_epoch_by_seed": meta["chosen_epochs"]}
    else:
        if not (folder / "model.json").exists():
            return None
        if not (folder / "candidate.json").exists():
            raise ValueError(f"{candidate} has model.json but no candidate.json: it is not finished")
        meta = read_json(folder / "candidate.json")
        seeds = tuple(int(seed) for seed in read_json(folder / "model.json")["seeds"])
        stored_metric = meta["validation"]["selection_metric"]
        settings = {key: meta[key] for key in SETTINGS_KEYS[candidate]}
    if meta.get("smoke", False) or meta.get("test_roles_scored", False):
        raise ValueError(f"{candidate} is a smoke run or scored a test role")
    path = prediction_path(directory, candidate, "validation", seeds)
    if not path.exists():
        raise ValueError(f"{candidate} is present but {path} is missing")
    return {
        "id": candidate,
        "family": FAMILIES[candidate],
        "seeds": list(seeds),
        "validation_file": path,
        "stored_selection_metric": float(stored_metric),
        "settings": settings,
    }


def ensemble_members(metrics: dict[str, float], size: int = ENSEMBLE_SIZE) -> dict[str, Any]:
    """
    Members of E: the best member of each of the ``size`` best families.

    Parameters
    ----------
    metrics : dict[str, float]
        Base candidate id to validation selection metric; E itself is not allowed.
    size : int
        Number of families.

    Returns
    -------
    dict[str, Any]
        ``members`` (protocol-table order), ``family_best`` (family to its best member) and ``family_ranking``
        (families in rank order, each with ``family``, ``member`` and ``selection_metric``).

    Raises
    ------
    ValueError
        If E or an unknown candidate is given, or fewer than ``size`` families are present.
    """
    unknown = set(metrics) - set(BASE_CANDIDATES)
    if unknown:
        raise ValueError(f"E is built from the base candidates only, got {sorted(unknown)}")
    best: dict[str, str] = {}
    for name in (name for name in BASE_CANDIDATES if name in metrics):
        family = FAMILIES[name]
        if family not in best or metrics[name] > metrics[best[family]]:
            best[family] = name
    if len(best) < size:
        raise ValueError(f"E needs {size} families, found {len(best)}")
    ranking = sorted(best.items(), key=lambda item: (-metrics[item[1]], BASE_CANDIDATES.index(item[1])))
    chosen = {member for _, member in ranking[:size]}
    return {
        "members": [name for name in BASE_CANDIDATES if name in chosen],
        "family_best": best,
        "family_ranking": [
            {"family": family, "member": member, "selection_metric": metrics[member]}
            for family, member in ranking
        ],
    }


def ensemble_probability(probabilities: dict[str, np.ndarray], members: list[str]) -> np.ndarray:
    """
    E's probability: the mean of its members' probabilities, summed in protocol-table order.

    Parameters
    ----------
    probabilities : dict[str, np.ndarray]
        Candidate id to aligned ``[records, 14]`` probabilities.
    members : list[str]
        E's members.

    Returns
    -------
    np.ndarray
        Float64 ``[records, 14]``.
    """
    ordered = [name for name in BASE_CANDIDATES if name in members]
    return np.mean(np.stack([probabilities[name] for name in ordered]), axis=0)


def ensemble_seeds(record_candidates: list[dict[str, Any]], members: list[str]) -> list[int]:
    """
    Seeds written into E's prediction files: the sorted union of its members' seeds.

    Parameters
    ----------
    record_candidates : list[dict[str, Any]]
        Candidate entries with ``id`` and ``seeds``.
    members : list[str]
        E's members.

    Returns
    -------
    list[int]
        Distinct seeds.
    """
    return sorted({seed for row in record_candidates if row["id"] in members for seed in row["seeds"]})


def build_selection(
    data: dict[str, np.ndarray], candidates_dir: Path = CANDIDATES_DIR, allow_missing: tuple[str, ...] = ()
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """
    Build the selection record content and the validation probabilities of every candidate, E included.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Validation records (``validation_data``).
    candidates_dir : Path
        Stage 2 candidate directory.
    allow_missing : tuple[str, ...]
        Base candidates that may be absent; any other absent candidate raises.

    Returns
    -------
    tuple[dict[str, Any], dict[str, np.ndarray]]
        The record (without file paths of E) and candidate id to validation probabilities.

    Raises
    ------
    ValueError
        If ``data`` holds another role, a required candidate is absent, or a recomputed selection metric
        differs from the stored one by more than ``STORED_TOLERANCE``.
    """
    if set(data["role"]) != {"validation"}:
        raise ValueError("selection uses validation records only")
    infos = {name: candidate_info(name, candidates_dir) for name in BASE_CANDIDATES}
    missing = [name for name, info in infos.items() if info is None]
    if set(missing) - set(allow_missing):
        raise ValueError(f"candidates absent: {sorted(set(missing) - set(allow_missing))}")
    present = {name: info for name, info in infos.items() if info is not None}
    probabilities = {
        name: load_predictions(info["validation_file"], data["record_key"])["probability"]
        for name, info in present.items()
    }
    tables = {
        name: selection_metric(data["labels"], probability, data["source"])
        for name, probability in probabilities.items()
    }
    for name, info in present.items():
        info["validation_sha256"] = sha256_file(info["validation_file"])
        info["stored_difference"] = tables[name]["selection_metric"] - info["stored_selection_metric"]
        if abs(info["stored_difference"]) > STORED_TOLERANCE:
            raise ValueError(
                f"{name}: recomputed selection metric differs from the stored one by "
                f"{info['stored_difference']}"
            )
    ensemble = ensemble_members({name: table["selection_metric"] for name, table in tables.items()})
    probabilities[ENSEMBLE] = ensemble_probability(probabilities, ensemble["members"])
    tables[ENSEMBLE] = selection_metric(data["labels"], probabilities[ENSEMBLE], data["source"])
    settings = {name: info["settings"] for name, info in present.items()} | {
        ENSEMBLE: {"members": ensemble["members"]}
    }
    record = selection_record(tables, settings)
    for row in record["candidates"]:
        row["per_condition"] = tables[row["id"]]["per_condition"]
        row["family"] = FAMILIES.get(row["id"], "ensemble")
        if row["id"] in present:
            info = present[row["id"]]
            row |= {
                "seeds": info["seeds"],
                "validation_file": stored(info["validation_file"]),
                "validation_sha256": info["validation_sha256"],
                "stored_selection_metric": info["stored_selection_metric"],
                "stored_difference": info["stored_difference"],
            }
        else:
            row["seeds"] = ensemble_seeds(record["candidates"], ensemble["members"])
    record = {
        "experiment": "MLC-02 Stage 3 selection record",
        "provisional": bool(missing),
        "missing_candidates": missing,
        **record,
        "rule": SELECTION_RULE,
        "ensemble_rule": ENSEMBLE_RULE,
        "families": FAMILIES,
        "ensemble": ensemble,
        "validation": {"records": len(data["record_key"]), "labels_sha256": labels_digest(data["labels"])},
        "test_roles_opened": False,
    }
    return record, probabilities


def settings_text(settings: dict[str, Any]) -> str:
    """
    One-line text of a candidate's chosen settings.

    Parameters
    ----------
    settings : dict[str, Any]
        Settings of one candidate in the record.

    Returns
    -------
    str
        ``key: value`` pairs; per-condition dictionaries are written ``condition value``.
    """
    parts = []
    for key, value in settings.items():
        if isinstance(value, dict) and set(value) <= set(CONDITIONS) and value:
            inner = "; ".join(
                f"{name} {', '.join(f'{k}={v}' for k, v in item.items())}"
                if isinstance(item, dict)
                else f"{name} {item}"
                for name, item in value.items()
            )
            parts.append(f"{key}: {inner}")
        elif isinstance(value, dict):
            parts.append(f"{key}: " + ", ".join(f"{k}={v}" for k, v in value.items()))
        elif isinstance(value, list):
            parts.append(f"{key}: " + ", ".join(str(item) for item in value))
        else:
            parts.append(f"{key}: {value}")
    return "; ".join(parts)


def selection_doc(record: dict[str, Any], record_path: str, record_sha256: str) -> str:
    """
    Markdown of a selection record, the text ``test-predict`` expects committed.

    Parameters
    ----------
    record : dict[str, Any]
        Content of ``selection_record.json``.
    record_path : str
        Stored path of the JSON file.
    record_sha256 : str
        Its SHA-256.

    Returns
    -------
    str
        The document.
    """
    sources = [source for source in record["sources"] if source not in record["excluded_sources"]]
    lines = ["# MLC-02 selection record", ""]
    if record["provisional"]:
        lines += [
            f"**PROVISIONAL: candidates missing ({', '.join(record['missing_candidates'])}). This is not "
            "the selection record of the protocol.**",
            "",
        ]
    lines += [
        f"Generated by `scripts.experiments.run_mlc02_stage3 select` from `{record_path}` (SHA-256 "
        f"`{record_sha256}`). Do not edit by hand: `test-predict` runs only when this file is committed "
        "and equal to the text generated from that record.",
        "",
        f"Validation records only ({record['validation']['records']:,}; label SHA-256 "
        f"`{record['validation']['labels_sha256']}`). No test role was opened to write it.",
        "",
        "## Rules",
        "",
        f"- Selection: {record['rule']}",
        f"- {record['ensemble_rule']}",
        "",
        "## Validation table",
        "",
        "Within-source macro AUROC per validation source and the selection metric (their mean).",
        "",
        "| candidate | family | " + " | ".join(sources) + " | selection metric | tied |",
        "|---|---|" + "---:|" * (len(sources) + 1) + "---|",
    ]
    for row in record["candidates"]:
        cells = " | ".join(f"{row['macro_auroc_by_source'][source]:.4f}" for source in sources)
        tied = "yes" if row["id"] in record["tied"] else ""
        lines.append(f"| {row['id']} | {row['family']} | {cells} | {row['selection_metric']:.4f} | {tied} |")
    lines += [
        "",
        f"Highest: {record['highest']} ({record['highest_metric']:.6f}). Tied within {record['margin']}: "
        f"{', '.join(record['tied'])}. **Selected: {record['selected']}.**",
        "",
    ]
    if record["excluded_sources"]:
        lines += [f"Sources without an entering condition: {', '.join(record['excluded_sources'])}.", ""]
    lines += [
        "Entering conditions per source: "
        + "; ".join(f"{source} {len(names)}" for source, names in record["entering_by_source"].items())
        + ".",
        "",
        "## E",
        "",
        "| rank | family | best member | selection metric | in E |",
        "|---:|---|---|---:|---|",
    ]
    for rank, row in enumerate(record["ensemble"]["family_ranking"], start=1):
        member = "yes" if row["member"] in record["ensemble"]["members"] else ""
        lines.append(
            f"| {rank} | {row['family']} | {row['member']} | {row['selection_metric']:.6f} | {member} |"
        )
    lines += ["", f"Members of E: {', '.join(record['ensemble']['members'])}.", "", "## Chosen settings", ""]
    lines += [f"- {row['id']}: {settings_text(row['settings'])}" for row in record["candidates"]]
    lines += [
        "",
        "## Files",
        "",
        "| candidate | seeds | validation file | SHA-256 | stored metric minus recomputed |",
        "|---|---|---|---|---:|",
    ]
    for row in record["candidates"]:
        difference = row.get("stored_difference")
        shown = "" if difference is None else ("0" if difference == 0 else f"{-difference:.1e}")
        lines.append(
            f"| {row['id']} | {', '.join(str(seed) for seed in row['seeds'])} | "
            f"`{row['validation_file']}` | `{row['validation_sha256']}` | {shown} |"
        )
    lines += [
        "",
        f"Manifest: `{record['inputs']['manifest']}` (SHA-256 `{record['inputs']['manifest_sha256']}`). "
        f"Code commit: `{record['inputs']['commit']}`.",
        "",
    ]
    return "\n".join(lines)


def read_record(output: Path) -> tuple[dict[str, Any], str]:
    """
    Read ``selection_record.json``.

    Parameters
    ----------
    output : Path
        Stage 3 directory.

    Returns
    -------
    tuple[dict[str, Any], str]
        The record and its SHA-256.

    Raises
    ------
    FileNotFoundError
        If the record does not exist.
    """
    path = Path(output) / RECORD_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; run select first")
    return read_json(path), sha256_file(path)


def receipt(
    step: str, command: list[str], started: float, code: dict[str, Any], extra: dict[str, Any]
) -> dict[str, Any]:
    """
    Receipt of one Stage 3 step.

    Parameters
    ----------
    step : str
        Step name.
    command : list[str]
        Command line.
    started : float
        ``time.time()`` at the start.
    code : dict[str, Any]
        Output of ``code_state``.
    extra : dict[str, Any]
        Step-specific inputs and outputs.

    Returns
    -------
    dict[str, Any]
        Step, command, times, code state, environment and the extra fields.
    """
    return {
        "step": step,
        "command": command,
        "started_unix": started,
        "wall_seconds": time.time() - started,
        "git": code,
        "environment": environment(),
        **extra,
    }


def run_select(
    output: Path = STAGE3,
    doc: Path = RECORD_DOC,
    candidates_dir: Path = CANDIDATES_DIR,
    data: dict[str, np.ndarray] | None = None,
    manifest: Path = MANIFEST,
    allow_missing: tuple[str, ...] = (),
    overwrite_before_tests: bool = False,
    roots: tuple[Path, ...] | None = None,
    command: list[str] | None = None,
    require_committed: bool = True,
) -> dict[str, Any]:
    """
    Write the selection record, its Markdown document, E's validation predictions and a receipt.

    Parameters
    ----------
    output : Path
        Stage 3 directory; receives ``selection_record.json``, ``predictions/E/validation__...npz`` and
        ``receipts/select.json``.
    doc : Path
        Markdown document to write (committed before ``test-predict``).
    candidates_dir : Path
        Stage 2 candidate directory.
    data : dict[str, np.ndarray] | None
        Validation records; None reads them from ``manifest``.
    manifest : Path
        MLC-01 ``manifest.csv``.
    allow_missing : tuple[str, ...]
        Base candidates that may be absent (a provisional record).
    overwrite_before_tests : bool
        Replace an existing record; refused once any test prediction exists.
    roots : tuple[Path, ...] | None
        Directories searched for test predictions; None is ``(OUTPUT, output, candidates_dir)``.
    command : list[str] | None
        Command line for the receipt.
    require_committed : bool
        Refuse to run from uncommitted code.

    Returns
    -------
    dict[str, Any]
        The record.
    """
    started = time.time()
    output = Path(output)
    roots = (OUTPUT, output, Path(candidates_dir)) if roots is None else roots
    guard_overwrite(output / RECORD_NAME, overwrite_before_tests, roots, "select")
    code = code_state(require_committed)
    data = validation_data(manifest) if data is None else data
    record, probabilities = build_selection(data, candidates_dir, allow_missing)
    ensemble_row = next(row for row in record["candidates"] if row["id"] == ENSEMBLE)
    path = prediction_path(output / PREDICTIONS, ENSEMBLE, "validation", tuple(ensemble_row["seeds"]))
    save_predictions(
        path, data["record_key"], probabilities[ENSEMBLE], ENSEMBLE, tuple(ensemble_row["seeds"])
    )
    ensemble_row |= {"validation_file": stored(path), "validation_sha256": sha256_file(path)}
    record["inputs"] = {
        "manifest": stored(manifest),
        "manifest_sha256": sha256_file(manifest),
        "candidates_dir": stored(candidates_dir),
        "commit": code["commit"],
    }
    record = json_ready(record)
    write_json_atomic(output / RECORD_NAME, record)
    record_sha = sha256_file(output / RECORD_NAME)
    write_text_atomic(doc, selection_doc(record, stored(output / RECORD_NAME), record_sha))
    write_json_atomic(
        output / RECEIPTS / "select.json",
        json_ready(
            receipt(
                "select",
                command or [],
                started,
                code,
                {
                    "inputs": record["inputs"],
                    "outputs": {
                        "record": stored(output / RECORD_NAME),
                        "record_sha256": record_sha,
                        "doc": stored(doc),
                        "doc_sha256": sha256_file(doc),
                        "ensemble_file": stored(path),
                    },
                    "test_roles_opened": False,
                },
            )
        ),
    )
    return record


def validation_probabilities(record: dict[str, Any], data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Load the validation probabilities of every candidate of a record, after checking each file's SHA-256.

    Parameters
    ----------
    record : dict[str, Any]
        Selection record.
    data : dict[str, np.ndarray]
        Validation records.

    Returns
    -------
    dict[str, np.ndarray]
        Candidate id to ``[records, 14]`` probabilities.

    Raises
    ------
    ValueError
        If a file differs from the record.
    """
    result = {}
    for row in record["candidates"]:
        path = from_stored(row["validation_file"])
        if sha256_file(path) != row["validation_sha256"]:
            raise ValueError(f"{path} differs from the selection record")
        result[row["id"]] = load_predictions(path, data["record_key"])["probability"]
    return result


def operating_points(labels: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    """
    Validation-fixed thresholds and Platt calibration of one candidate.

    Parameters
    ----------
    labels : np.ndarray
        ``[records, 14]`` validation states; undefined entries are left out per condition.
    probability : np.ndarray
        Aligned probabilities.

    Returns
    -------
    dict[str, Any]
        ``thresholds`` and ``platt`` (``intercept``, ``slope``) per condition, and ``validation_at_threshold``
        (sensitivity, specificity, positives and negatives on the validation records).
    """
    thresholds = specificity_thresholds(labels, probability)
    platt = fit_platt_damped(labels, probability)
    rates = {}
    for column, name in enumerate(CONDITIONS):
        defined = ~np.isnan(labels[:, column])
        y = labels[defined, column]
        rates[name] = threshold_rates(y, probability[defined, column], thresholds[column]) | {
            "positives": int(y.sum()),
            "negatives": int(len(y) - y.sum()),
        }
    return {
        "thresholds": dict(zip(CONDITIONS, thresholds.tolist(), strict=True)),
        "platt": {
            name: {"intercept": float(a), "slope": float(b)}
            for name, (a, b) in zip(CONDITIONS, platt, strict=True)
        },
        "validation_at_threshold": rates,
    }


def run_fix(
    output: Path = STAGE3,
    data: dict[str, np.ndarray] | None = None,
    manifest: Path = MANIFEST,
    overwrite_before_tests: bool = False,
    roots: tuple[Path, ...] | None = None,
    command: list[str] | None = None,
    require_committed: bool = True,
) -> dict[str, Any]:
    """
    Fit and write the validation operating points of every candidate of the selection record.

    Parameters
    ----------
    output : Path
        Stage 3 directory holding the record; receives ``operating_points.json`` and ``receipts/fix.json``.
    data : dict[str, np.ndarray] | None
        Validation records; None reads them from ``manifest``.
    manifest : Path
        MLC-01 ``manifest.csv``.
    overwrite_before_tests : bool
        Replace existing operating points; refused once any test prediction exists.
    roots : tuple[Path, ...] | None
        Directories searched for test predictions; None is ``(OUTPUT, output)``.
    command : list[str] | None
        Command line for the receipt.
    require_committed : bool
        Refuse to run from uncommitted code.

    Returns
    -------
    dict[str, Any]
        The operating points.
    """
    started = time.time()
    output = Path(output)
    roots = (OUTPUT, output) if roots is None else roots
    guard_overwrite(output / OPERATING_NAME, overwrite_before_tests, roots, "fix")
    code = code_state(require_committed)
    record, record_sha = read_record(output)
    data = validation_data(manifest) if data is None else data
    if (
        set(data["role"]) != {"validation"}
        or labels_digest(data["labels"]) != record["validation"]["labels_sha256"]
    ):
        raise ValueError("fix needs the validation records the selection record was built on")
    probabilities = validation_probabilities(record, data)
    points = {
        "selection_record": stored(output / RECORD_NAME),
        "selection_record_sha256": record_sha,
        "target_specificity": TARGET_SPECIFICITY,
        "threshold_rule": THRESHOLD_RULE,
        "platt_rule": PLATT_RULE,
        "conditions": list(CONDITIONS),
        "candidates": {
            row["id"]: {
                "validation_file": row["validation_file"],
                "validation_sha256": row["validation_sha256"],
                **operating_points(data["labels"], probabilities[row["id"]]),
            }
            for row in record["candidates"]
        },
        "test_roles_opened": False,
    }
    points = json_ready(points)
    write_json_atomic(output / OPERATING_NAME, points)
    write_json_atomic(
        output / RECEIPTS / "fix.json",
        json_ready(
            receipt(
                "fix",
                command or [],
                started,
                code,
                {
                    "inputs": {
                        "selection_record_sha256": record_sha,
                        "manifest": stored(manifest),
                        "manifest_sha256": sha256_file(manifest),
                    },
                    "outputs": {
                        "operating_points": stored(output / OPERATING_NAME),
                        "operating_points_sha256": sha256_file(output / OPERATING_NAME),
                    },
                    "test_roles_opened": False,
                },
            )
        ),
    )
    return points


def read_operating_points(output: Path, record_sha256: str) -> tuple[dict[str, dict[str, np.ndarray]], str]:
    """
    Thresholds and Platt parameters as arrays, after checking they belong to the selection record.

    A threshold stored as None (no observed score reached the target) is infinite.

    Parameters
    ----------
    output : Path
        Stage 3 directory.
    record_sha256 : str
        SHA-256 of the selection record they must have been fitted for.

    Returns
    -------
    tuple[dict[str, dict[str, np.ndarray]], str]
        Candidate id to ``thresholds`` (``[14]``) and ``platt`` (``[14, 2]``), and the file's SHA-256.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If it was fitted for another selection record.
    """
    path = Path(output) / OPERATING_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; run fix first")
    points = read_json(path)
    if points["selection_record_sha256"] != record_sha256:
        raise ValueError("operating points were fitted for another selection record")
    result = {}
    for name, entry in points["candidates"].items():
        thresholds = np.array(
            [math.inf if entry["thresholds"][c] is None else entry["thresholds"][c] for c in CONDITIONS],
            dtype=np.float64,
        )
        platt = np.array([[entry["platt"][c]["intercept"], entry["platt"][c]["slope"]] for c in CONDITIONS])
        result[name] = {"thresholds": thresholds, "platt": platt}
    return result, sha256_file(path)


def check_record_committed(doc: Path, output: Path) -> dict[str, Any]:
    """
    Check that the selection record's document is committed, unmodified and generated from the record.

    Parameters
    ----------
    doc : Path
        Markdown document of the record.
    output : Path
        Stage 3 directory holding ``selection_record.json``.

    Returns
    -------
    dict[str, Any]
        ``doc``, ``doc_sha256``, ``doc_commit`` and ``record_sha256``.

    Raises
    ------
    FileNotFoundError
        If the record or the document does not exist.
    PermissionError
        If the document is outside the repository, untracked, modified, or differs from the text generated
        from the record.
    """
    record, record_sha = read_record(output)
    doc = Path(doc)
    if not doc.exists():
        raise FileNotFoundError(f"{doc} does not exist")
    try:
        relative = to_stored(doc)
    except ValueError as error:
        raise PermissionError(f"{doc} must be inside the repository and committed") from error
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", relative], cwd=ROOT, capture_output=True)
    if tracked.returncode != 0:
        raise PermissionError(f"{relative} is not committed")
    status = subprocess.check_output(["git", "status", "--porcelain", "--", relative], cwd=ROOT, text=True)
    if status.strip():
        raise PermissionError(f"{relative} has uncommitted changes")
    expected = selection_doc(record, stored(Path(output) / RECORD_NAME), record_sha)
    if doc.read_text(encoding="utf-8") != expected:
        raise PermissionError(f"{relative} is not the document generated from {Path(output) / RECORD_NAME}")
    commit = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", relative], cwd=ROOT, text=True)
    return {
        "doc": relative,
        "doc_sha256": sha256_file(doc),
        "doc_commit": commit.strip(),
        "record_sha256": record_sha,
    }
