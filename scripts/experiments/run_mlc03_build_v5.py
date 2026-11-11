"""
Build dataset v5 of MLC-03 Part C (``outputs/mlc01_condition_dataset_20261111_v5/``) from v4 and rules C1-C12.

Steps:

- ``gate``: the integrity gate on train and validation only. The record and label readers, method A, the
  analysis table and method B (on a sample) reproduce Part B's stored inputs; the window summaries reproduce
  Part B's ``item1_whole_record_windows.csv``; the counts of every adopted rule equal Part A's
  ``proposed_rule_counts.csv`` and Part B's ``rules_candidate_counts.csv``. No test role is read.
- ``build``: the gate, then method B and the window summaries on the test roles (one pass), the rules on every
  role, the v5 manifest, the flow from v4, the support tables, the integrity checks and the receipt. It
  refuses an existing output directory and writes through a staging directory.

Run from the worktree:
``PYTHONPATH=<worktree> OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m
scripts.experiments.run_mlc03_build_v5 <step> --workers 14``
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from ecg_experiment import ROOT
from ecg_experiment import mlc03_dataset_v5 as mv
from ecg_experiment import mlc03_label_quality as lq
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.mlc02_cache_build import code_sha256, git_state
from ecg_experiment.mlc02_feature_data import LOCAL
from ecg_experiment.mlc02_waveforms import CACHE, closed_keys, init_worker
from ecg_experiment.paths import to_stored

CODE = (
    "ecg_experiment/mlc03_dataset_v5.py",
    "scripts/experiments/run_mlc03_build_v5.py",
    "ecg_experiment/mlc03_label_quality.py",
    "ecg_experiment/mlc03_measurements.py",
    "ecg_experiment/mlc01_tables.py",
    "ecg_experiment/mlc01_dataset.py",
    "ecg_experiment/mlc02_waveforms.py",
    "ecg_experiment/mlc02_feature_data.py",
    "ecg_experiment/mlc_labels.py",
    "ecg_experiment/measurement_extractor042.py",
)
INPUTS = (
    mv.V4 / "manifest.csv",
    mv.V4 / "support_B.csv",
    mv.FLAGS_OPEN,
    mv.FLAGS_TEST,
    mv.RULE_COUNTS_A,
    mv.ANALYSIS,
    mv.MEASUREMENTS_B,
    mv.WINDOWS,
    mv.SEPARATE_CODING,
    mv.CUTOFFS,
    mv.RULE_COUNTS_B,
    LOCAL / "rows.csv",
    LOCAL / "measurements.npy",
    LOCAL / "measurements_status.csv",
    CACHE / "index_all.csv",
)
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_CORETYPE", "PYTHONPATH")
PACKAGES = ("numpy", "pandas", "scipy", "wfdb", "h5py", "pyarrow")
EDAC = Path("/sys/devices/system/edac/mc/mc0/ue_count")
SAMPLE_PER_SOURCE = 250
SEED = 20261111
CHUNK = 100


def log(message: str) -> None:
    """
    Print a time-stamped progress line.

    Parameters
    ----------
    message : str
        Text.
    """
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


def stored_name(path: Path) -> str:
    """
    Repository-relative name of a path, or the absolute name of a local scratch file outside the repository.

    Parameters
    ----------
    path : Path
        Input file.

    Returns
    -------
    str
        Name for the receipt.
    """
    return to_stored(path) if Path(os.path.abspath(path)).is_relative_to(ROOT) else str(path)


def edac() -> str | None:
    """
    Read the uncorrected memory-error counter.

    Returns
    -------
    str | None
        Counter value, None where the file is absent.
    """
    return EDAC.read_text().strip() if EDAC.exists() else None


def plain(value: object) -> object:
    """
    Convert NumPy scalars inside a nested value to Python ones, for JSON.

    Parameters
    ----------
    value : object
        Nested dictionaries, lists and scalars.

    Returns
    -------
    object
        The same structure with JSON-serializable scalars.
    """
    return json.loads(
        json.dumps(value, default=lambda item: item.item() if hasattr(item, "item") else str(item))
    )


def column_mismatches(first: pd.DataFrame, second: pd.DataFrame) -> dict[str, int]:
    """
    Count differing cells per column of two aligned tables, NaN equal to NaN.

    Parameters
    ----------
    first, second : pd.DataFrame
        Tables with the same rows.

    Returns
    -------
    dict[str, int]
        Columns that differ (or exist in one table only, as -1) with their count of differing cells.

    Raises
    ------
    ValueError
        If the tables have a different number of rows.
    """
    if len(first) != len(second):
        raise ValueError(f"tables have {len(first)} and {len(second)} rows")
    found = {}
    for name in sorted(set(first.columns) | set(second.columns)):
        if name not in first.columns or name not in second.columns:
            found[name] = -1
            continue
        a, b = first[name].reset_index(drop=True), second[name].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            x, y = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
            differ = ~((x == y) | (np.isnan(x) & np.isnan(y)))
        else:
            differ = ~((a.astype(str) == b.astype(str)) | (a.isna() & b.isna())).to_numpy()
        if differ.any():
            found[name] = int(differ.sum())
    return found


def measure_all(keys: list[str], workers: int) -> pd.DataFrame:
    """
    Run method B on the cached windows of some records, in worker processes.

    Parameters
    ----------
    keys : list[str]
        Record keys.
    workers : int
        Processes.

    Returns
    -------
    pd.DataFrame
        One row per key, in key order, ``status`` as text.
    """
    chunks = [keys[i : i + CHUNK] for i in range(0, len(keys), CHUNK)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [row for chunk in pool.map(mv.measure_cached, chunks) for row in chunk]
    frame = pd.DataFrame(rows)
    frame["status"] = frame["status"].astype(str)
    if frame["record_key"].tolist() != list(keys):
        raise ValueError("measured records differ from the requested ones")
    return frame


def windows_all(records: pd.DataFrame, workers: int) -> pd.DataFrame:
    """
    Window summaries of the records longer than 10 s, read through the closed-data guard.

    Parameters
    ----------
    records : pd.DataFrame
        Output of ``mv.load_records``.
    workers : int
        Processes.

    Returns
    -------
    pd.DataFrame
        ``record_key`` and the ``window_summary`` columns, lists as JSON text.
    """
    long = records[records["duration_s"] > 10.0 + 1e-6]
    jobs = list(zip(long["record_key"], long["raw_path"], long["source"], strict=True))
    with ProcessPoolExecutor(
        max_workers=workers, initializer=init_worker, initargs=(closed_keys(), frozenset(long["raw_path"]))
    ) as pool:
        rows = list(pool.map(mv.record_windows, jobs, chunksize=8))
    frame = pd.DataFrame(rows, columns=None if rows else ["record_key"])
    for column in ("ectopy_by_window", "af_by_window", "hr_by_window"):
        if column in frame:
            frame[column] = frame[column].map(json.dumps)
    return frame


def part_a_check(table: pd.DataFrame, masks: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """
    Compare the record rules' flag counts with Part A's ``proposed_rule_counts.csv``.

    C3 and C4 are compared on their flags before the SPH exception, since Part A counted the flags.

    Parameters
    ----------
    table : pd.DataFrame
        Records with ``source`` and ``role``.
    masks : pd.DataFrame
        Output of ``mv.record_rule_masks``.
    flags : pd.DataFrame
        Output of ``mv.aligned_flags``.

    Returns
    -------
    pd.DataFrame
        ``compare_counts`` rows per rule, role and source.
    """
    found = []
    for rule, flag in mv.RECORD_RULES.items():
        selected = flags[flag].fillna(False).astype(bool)
        for (role, source), part in table.groupby(["role", "source"]):
            found.append(
                {
                    "rule": mv.PART_A_RULES[rule],
                    "role": role,
                    "source": source,
                    "affected": int(selected.loc[part.index].sum()),
                    "rule_count": int(masks.loc[part.index, rule].sum()),
                }
            )
    found = pd.DataFrame(found)
    expected = pd.read_csv(mv.RULE_COUNTS_A)
    expected = expected[
        expected["rule"].isin(mv.PART_A_RULES.values()) & expected["role"].isin(set(table["role"]))
    ]
    return mv.compare_counts(found, expected, ["rule", "role", "source"], ["affected"]).merge(
        found[["rule", "role", "source", "rule_count"]], on=["rule", "role", "source"], how="left"
    )


def gate(workers: int) -> dict[str, object]:
    """
    Run the integrity gate on train and validation records (step ``gate``).

    Parameters
    ----------
    workers : int
        Processes for the method B sample and the window summaries.

    Returns
    -------
    dict[str, object]
        Every check with its result, and the open tables for the build.

    Raises
    ------
    ValueError
        If any check fails.
    """
    started = time.time()
    records = mv.load_records(mv.OPEN_ROLES)
    checks: dict[str, object] = {}
    checks["records_equal_part_b_reader"] = column_mismatches(records, lq.load_open_records())
    keys = records["record_key"].to_numpy()
    a = mv.measurements_a(keys)
    checks["method_a_equal_part_b"] = column_mismatches(a, lq.method_a(keys))
    b = pd.read_parquet(mv.MEASUREMENTS_B)
    if b["record_key"].tolist() != records["record_key"].tolist():
        raise ValueError("stored method B rows differ from the open records")
    table = mv.analysis(records, a, b)
    stored = pd.read_parquet(mv.ANALYSIS)
    checks["analysis_equal_part_b"] = column_mismatches(table.drop(columns=["codes"]), stored)
    log("analysis table rebuilt")
    sample = records.groupby("source", sort=False).sample(n=SAMPLE_PER_SOURCE, random_state=SEED).sort_index()
    remeasured = measure_all(sample["record_key"].tolist(), workers)
    reference = b.set_index("record_key").loc[sample["record_key"]].reset_index()
    checks["method_b_sample"] = {
        "records": len(sample),
        "mismatches": column_mismatches(remeasured[reference.columns], reference),
    }
    log("method B sample remeasured")
    windows = windows_all(records, workers)
    stored_windows = pd.read_csv(mv.WINDOWS, dtype={"record_key": str})
    columns = list(windows.columns)
    aligned = stored_windows.set_index("record_key").loc[windows["record_key"]].reset_index()[columns]
    checks["windows_equal_part_b"] = {
        "records": len(windows),
        "stored_records": len(stored_windows),
        "mismatches": column_mismatches(windows[columns], aligned),
    }
    log("window summaries recomputed")
    flags = mv.aligned_flags(table, pd.read_parquet(mv.FLAGS_OPEN))
    record_masks = mv.record_rule_masks(table, flags)
    part_a = part_a_check(table, record_masks, flags)
    parts = mv.part_b_masks(table, flags, stored_windows)
    found = mv.part_b_counts(parts, table)
    expected = pd.read_csv(mv.RULE_COUNTS_B)
    adopted = expected["rule"].isin(mv.PART_B_NAMES.values()) | expected["rule"].str.startswith("R4 ")
    rate_only = ~expected["rule"].str.startswith("R6 ") | expected["condition"].isin(mv.RATE_CONDITIONS)
    expected = expected[adopted & rate_only]
    part_b = mv.compare_counts(found, expected, ["rule", "condition", "source", "role"], mv.COUNT_COLUMNS)
    checks["sources"] = {
        "found": mv.expected_sources(),
        "used": {"C9": sorted(mv.C9_SOURCES), "C10": sorted(mv.C10_SOURCES)},
    }
    checks["part_a_counts_equal"] = bool(part_a["equal"].all())
    checks["part_b_counts_equal"] = bool(part_b["equal"].all())
    checks["part_b_rows"] = int(len(part_b))
    failed = [
        name
        for name in ("records_equal_part_b_reader", "method_a_equal_part_b", "analysis_equal_part_b")
        if checks[name]
    ]
    failed += [name for name in ("method_b_sample", "windows_equal_part_b") if checks[name]["mismatches"]]
    failed += [name for name in ("part_a_counts_equal", "part_b_counts_equal") if not checks[name]]
    if checks["sources"]["found"] != checks["sources"]["used"]:
        failed.append("sources")
    checks["failed"] = failed
    checks["seconds"] = time.time() - started
    if failed:
        print(json.dumps({k: v for k, v in checks.items() if k != "failed"}, indent=1, default=str))
        print(part_b[~part_b["equal"]].to_string())
        print(part_a[~part_a["equal"]].to_string())
        raise ValueError(f"integrity gate failed: {failed}")
    log(f"gate passed: {len(part_b)} Part B rows, {len(part_a)} Part A rows")
    return {
        "checks": checks,
        "part_a": part_a,
        "part_b": part_b,
        "table": table,
        "flags": flags,
        "windows": stored_windows,
    }


def rule_summary(flow: pd.DataFrame) -> pd.DataFrame:
    """
    Count the records or labels selected and changed by each rule, per role.

    Parameters
    ----------
    flow : pd.DataFrame
        Flow of ``mv.apply_rules``.

    Returns
    -------
    pd.DataFrame
        Per rule and role: selected (before overlaps) and changed (after earlier rules), with positives and
        negatives for the label rules.
    """
    record = (
        flow[flow["condition"] == ""]
        .groupby(["rule", "role"], sort=False)[["selected", "already_excluded", "excluded"]]
        .sum()
        .reset_index()
    )
    label = (
        flow[flow["condition"] != ""]
        .groupby(["rule", "role"], sort=False)[
            [
                "selected_positives",
                "selected_negatives",
                "on_excluded_records",
                "already_undefined",
                "undefined_positives",
                "undefined_negatives",
            ]
        ]
        .sum()
        .reset_index()
    )
    return pd.concat([record, label], ignore_index=True)


def exclusion_flow(v4: pd.DataFrame, v5: pd.DataFrame) -> pd.DataFrame:
    """
    Included records per source and role in v4, excluded by each record rule, and in v5.

    Parameters
    ----------
    v4, v5 : pd.DataFrame
        Manifests read as text.

    Returns
    -------
    pd.DataFrame
        One row per source and role.
    """
    rows = []
    for (source, role), part in v4.groupby(["source", "role"], sort=False):
        after = v5.loc[part.index]
        row = {"source": source, "role": role, "included_v4": int((part["included"] == "True").sum())}
        for rule in mv.RECORD_RULES:
            row[f"excluded_{rule}"] = int(
                ((part["included"] == "True") & (after["exclusion_reason"] == rule)).sum()
            )
        row["included_v5"] = int((after["included"] == "True").sum())
        rows.append(row)
    return pd.DataFrame(rows)


def write_csv(frame: pd.DataFrame, path: Path) -> str:
    """
    Write a table and return its SHA-256.

    Parameters
    ----------
    frame : pd.DataFrame
        Table.
    path : Path
        Destination.

    Returns
    -------
    str
        SHA-256 of the written file.
    """
    frame.to_csv(path, index=False)
    return sha256_file(path)


def rule_definitions() -> pd.DataFrame:
    """
    List the rules of Amendment 1 as applied here, with their evidence and reading.

    Returns
    -------
    pd.DataFrame
        One row per rule.
    """
    rows = [
        (
            "C1",
            "exclude record",
            "missing samples in the first 10 s (Part A flag missing_samples)",
            "A-R1",
            "all",
        ),
        ("C2", "exclude record", "constant head or tail (Part A flag edge_hold)", "A-R3", "all"),
        ("C3", "exclude record", "samples at the 16-bit rail (Part A flag adc_rail)", "A-R2", "all but SPH"),
        (
            "C4",
            "exclude record",
            "derived leads inconsistent with I and II (Part A flag identity_violation)",
            "A-R7",
            "all but SPH",
        ),
        (
            "C5",
            "label undefined",
            "a lead the condition is read on is flat (manifest flat_<lead>, B-R5) or flat, "
            "near-flat or without contact (Part A lead lists, B-R11 first part)",
            "B-R5, B-R11",
            "all",
        ),
        (
            "C6",
            "label undefined",
            "both space-B labels of a contradictory pair (lq.CONTRADICTIONS)",
            "B-R4",
            "all",
        ),
        (
            "C7",
            "label undefined",
            "conduction, axis, hypertrophy, ST, T and MI labels with a paced rhythm statement "
            "(lq.UNREADABLE)",
            "B-R8",
            "all",
        ),
        (
            "C8",
            "label undefined",
            "PVC or AF-or-flutter positive of a recording longer than 10 s whose finding is "
            "measured only after the first 10 s",
            "B-R1",
            "all",
        ),
        (
            "C9",
            "label undefined",
            "complete RBBB positive with consensus QRS below 110 ms (beyond the 10 ms tolerance "
            "of the 120 ms bound)",
            "B-R2",
            "Chapman, CPSC 2018",
        ),
        (
            "C10",
            "label undefined",
            "sinus bradycardia negative, sinus evidence, consensus rate below 60 per minute "
            "(B-R3 code: no tolerance)",
            "B-R3",
            "PTB-XL, CPSC-Extra",
        ),
        (
            "C11",
            "label undefined",
            "sinus tachycardia or bradycardia positive outside its bound by more than 5 per "
            "minute; negative inside it by more than 5 per minute with sinus evidence",
            "B-R6",
            "all; negatives not at SPH (no sinus evidence by codes)",
        ),
        (
            "C12",
            "label undefined",
            "LVH negative with a Sokolow-Lyon or Cornell voltage criterion (method B)",
            "B-R10",
            "all",
        ),
    ]
    return pd.DataFrame(rows, columns=["rule", "action", "definition", "evidence", "sources"])


def build(workers: int, output: Path, command: list[str]) -> dict[str, object]:
    """
    Build dataset v5 (step ``build``).

    Parameters
    ----------
    workers : int
        Processes.
    output : Path
        Output directory; it must not exist.
    command : list[str]
        Command line for the receipt.

    Returns
    -------
    dict[str, object]
        The receipt.

    Raises
    ------
    FileExistsError
        If the output or its staging directory exists.
    RuntimeError
        If a tracked file is modified.
    """
    started = time.time()
    staging = output.with_name(output.name + ".staging")
    for path in (output, staging):
        if path.exists():
            raise FileExistsError(f"{path} exists; the builder never overwrites")
    git = git_state()
    if git["modified_tracked"]:
        raise RuntimeError(f"commit the code before building: {git['modified_tracked']}")
    edac_before = edac()
    opened = gate(workers)
    staging.mkdir(parents=True)
    test_records = mv.load_records(mv.TEST_ROLES)
    test_keys = test_records["record_key"].tolist()
    log(f"measuring {len(test_keys)} test-role records")
    b_test = measure_all(test_keys, workers)
    b_test.to_parquet(staging / "measurements_b_test_roles.parquet")
    windows_test = windows_all(test_records, workers)
    windows_test.to_csv(staging / "windows_test_roles.csv", index=False)
    log(f"windows of {len(windows_test)} long test-role records")
    test_table = mv.analysis(test_records, mv.measurements_a(np.asarray(test_keys)), b_test)
    test_flags = mv.aligned_flags(test_table, pd.read_parquet(mv.FLAGS_TEST))
    table = pd.concat([opened["table"], test_table], ignore_index=True)
    flags = pd.concat([opened["flags"], test_flags], ignore_index=True)
    windows = pd.concat([opened["windows"], windows_test], ignore_index=True)
    record_masks = mv.record_rule_masks(table, flags)
    part_a = part_a_check(table, record_masks, flags)
    parts = mv.part_b_masks(table, flags, windows)
    label_masks = mv.label_rule_masks(parts, table)
    v4 = pd.read_csv(mv.V4 / "manifest.csv", dtype=str, keep_default_na=False, low_memory=False)
    v5, flow = mv.apply_rules(v4, record_masks.assign(record_key=table["record_key"]), label_masks)
    every_flag = pd.concat(
        [pd.read_parquet(mv.FLAGS_OPEN), pd.read_parquet(mv.FLAGS_TEST)], ignore_index=True
    )
    flag_table = mv.flag_columns(every_flag)
    if flag_table["record_key"].duplicated().any():
        raise ValueError("a record has two Part A flag rows")
    flag_table = flag_table.set_index("record_key").reindex(v5["record_key"])
    for name in flag_table.columns:
        v5[name] = flag_table[name].map({True: "True", False: "False"}).fillna("").to_numpy()
    hashes = {"manifest.csv": write_csv(v5, staging / "manifest.csv")}
    integrity = mv.compare_manifests(v4, v5[v4.columns])
    support_v4, support_v5 = mv.support(v4), mv.support(v5)
    reference = pd.read_csv(mv.V4 / "support_B.csv")
    keys = ["space", "label", "role", "level", "unit"]
    mine = support_v4[support_v4["role"].isin(reference["role"].unique())]
    integrity["support_v4_reproduced"] = (
        column_mismatches(
            mine.merge(reference[keys], on=keys)[keys + ["records", "positive", "negative", "undefined"]],
            reference[keys + ["records", "positive", "negative", "undefined"]],
        )
        == {}
    )
    comparison = mv.support_comparison(support_v4, support_v5)
    evaluation = comparison[comparison["role"] != "train"]
    changes = evaluation[evaluation["loses_eligibility"] | evaluation["drops_below_100"]]
    summary = rule_summary(flow)
    tables = {
        "rule_flow.csv": flow,
        "rule_summary.csv": summary,
        "exclusion_flow_v5.csv": exclusion_flow(v4, v5),
        "support_v4_v5.csv": comparison,
        "eligibility_changes.csv": changes,
        "part_a_count_check.csv": part_a,
        "part_b_count_check.csv": opened["part_b"],
        "rule_definitions.csv": rule_definitions(),
    }
    for name, frame in tables.items():
        hashes[name] = write_csv(frame, staging / name)
    hashes["measurements_b_test_roles.parquet"] = sha256_file(staging / "measurements_b_test_roles.parquet")
    hashes["windows_test_roles.csv"] = sha256_file(staging / "windows_test_roles.csv")
    integrity["gate"] = opened["checks"]
    integrity["new_columns"] = [name for name in v5.columns if name not in v4.columns]
    write_json_atomic(staging / "integrity.json", plain(integrity), allow_nan=True)
    hashes["integrity.json"] = sha256_file(staging / "integrity.json")
    receipt = {
        "command": command,
        "started_unix": started,
        "wall_seconds": time.time() - started,
        "output": to_stored(output),
        "git": git,
        "code_sha256": code_sha256(CODE),
        "inputs_sha256": {stored_name(path): sha256_file(path) for path in INPUTS},
        "threads": {name: os.environ.get(name) for name in THREADS},
        "workers": workers,
        "packages": {name: version(name) for name in PACKAGES},
        "edac_uncorrected": {"before": edac_before, "after": edac()},
        "test_roles": {
            "labels_used_to_choose": False,
            "models_scored": False,
            "records": len(test_keys),
            "long_records_windowed": len(windows_test),
        },
        "readings": {
            "C10": "B-R3 code: consensus rate below 60 per minute, no tolerance",
            "C11_negatives_at_sph": "not applied: SPH's AHA codes give no sinus evidence by code",
        },
        "records": {
            "v4_included": int((v4["included"] == "True").sum()),
            "v5_included": int((v5["included"] == "True").sum()),
        },
        "rule_summary": summary.to_dict("records"),
        "cells_compared": integrity["cells_compared"],
        "cells_changed": integrity["cells_changed"],
        "outputs_sha256": hashes,
    }
    write_json_atomic(staging / "receipt.json", plain(receipt), allow_nan=True)
    staging.rename(output)
    log(f"built {to_stored(output)} in {time.time() - started:.0f} s")
    return receipt


def main(argv: list[str] | None = None) -> None:
    """
    Command-line entry point.

    Parameters
    ----------
    argv : list[str] | None
        Arguments; None reads ``sys.argv``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("step", choices=("gate", "build"))
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--output", type=Path, default=mv.V5)
    args = parser.parse_args(argv)
    if args.workers > 16:
        raise ValueError("at most 16 workers while another agent runs")
    if args.step == "gate":
        checks = gate(args.workers)["checks"]
        print(json.dumps(plain(checks), indent=1))
    else:
        build(args.workers, args.output.absolute(), [sys.executable, *sys.argv])


if __name__ == "__main__":
    main()
