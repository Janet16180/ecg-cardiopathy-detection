"""
MLC-03 Part B, label quality, on train and validation records only.

Steps (each writes under ``outputs/mlc03_data_quality_20261111/labels/``; scratch in
``/var/tmp/janetrivera/mlc03``):

- ``calibrate``: method B's QRS and P thresholds, chosen on 1,500 PTB-XL training records against PTB-XL+;
- ``measure``: method B on every open record (cached first 10 s), in checkpointed shards;
- ``validate``: measurement error on synthetic records, LUDB and PTB-XL+; method agreement and failures per
  source;
- ``labels``: items 1 (label against measurement), 2 (source coding), 3 (contradictions) and 5 (undefined
  entries);
- ``oof``: P0 integrity checks and five-fold grouped out-of-fold probabilities (item 4);
- ``suspects``: confident learning and the 50 most suspected false positives and negatives per condition and
  source;
- ``sheets``: review sheets of the suspects under ``outputs/mlc03_data_quality_20261111/review/labels/``;
- ``rules``: counts of the candidate cleaning rules.

Run from the worktree:
``PYTHONPATH=<worktree> OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -m
scripts.experiments.run_mlc03_label_quality <step> --workers 7``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ecg_experiment import ROOT
from ecg_experiment import mlc03_label_quality as lq
from ecg_experiment import mlc03_measurements as mm
from ecg_experiment.files import read_json, sha256_file, write_json_atomic
from ecg_experiment.mlc02_cache_build import git_state
from ecg_experiment.mlc02_data import CONDITIONS, MANIFEST
from ecg_experiment.mlc02_feature_runs import environment
from ecg_experiment.paths import to_stored

CODE = (
    "ecg_experiment/mlc03_label_quality.py",
    "ecg_experiment/mlc03_measurements.py",
    "scripts/experiments/run_mlc03_label_quality.py",
    "ecg_experiment/measurement_extractor042.py",
    "ecg_experiment/mlc02_feature_models.py",
    "ecg_experiment/key_classifier_models.py",
    "ecg_experiment/mlc02_waveforms.py",
    "ecg_experiment/mlc_labels.py",
)
QRS_GRID = (0.02, 0.04, 0.06, 0.08, 0.10)
P_GRID = (0.1, 0.2, 0.3, 0.4)
P0 = ROOT / "outputs/mlc02_condition_models_20261110/candidates/P0"
P0_SELECTION_METRIC = 0.9642697112976489
EDAC = Path("/sys/devices/system/edac/mc/mc0/ue_count")


def write_csv(frame: pd.DataFrame, name: str) -> None:
    """
    Write one output table under ``lq.OUTPUT``.

    Parameters
    ----------
    frame : pd.DataFrame
        Table.
    name : str
        Relative file name.
    """
    path = lq.OUTPUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def receipt(step: str, started: float, extra: dict[str, object]) -> None:
    """
    Write the step's receipt with code hashes, environment, inputs and the memory-error counter.

    Parameters
    ----------
    step : str
        Step name.
    started : float
        Start time (``time.time()``).
    extra : dict[str, object]
        Step-specific content.
    """
    record = {
        "step": step,
        "command": sys.argv,
        "started_unix": started,
        "wall_seconds": time.time() - started,
        "git": git_state(),
        "code_sha256": {path: sha256_file(ROOT / path) for path in CODE},
        "manifest": to_stored(MANIFEST),
        "manifest_sha256": sha256_file(MANIFEST),
        "environment": environment(),
        "edac_uncorrected": EDAC.read_text().strip() if EDAC.exists() else None,
        **extra,
    }
    write_json_atomic(lq.OUTPUT / "receipts" / f"{step}.json", record, sort_keys=False)


def parallel(function: object, chunks: list, workers: int) -> list:
    """
    Map a module-level function over chunks in worker processes, keeping order.

    Parameters
    ----------
    function : object
        Picklable function of one chunk.
    chunks : list
        Inputs.
    workers : int
        Processes.

    Returns
    -------
    list
        One result per chunk.
    """
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(function, chunks))


def grid_chunk(keys: list[str]) -> list[dict[str, object]]:
    """
    Measure method B's QRS and PR on cached records for every threshold pair of the calibration grid.

    Parameters
    ----------
    keys : list[str]
        Record keys.

    Returns
    -------
    list[dict[str, object]]
        ``record_key`` and ``qrs_<q>`` and ``pr_<q>_<p>`` values.
    """
    rows = []
    for key, signal in zip(keys, lq.cached_windows(np.asarray(keys)), strict=True):
        row: dict[str, object] = {"record_key": key}
        for q in QRS_GRID:
            for p in P_GRID:
                result = mm.measure_record(signal, (q, p))
                row[f"qrs_{q}"], row[f"pr_{q}_{p}"] = (
                    result.get("qrs_ms", np.nan),
                    result.get("pr_ms", np.nan),
                )
        rows.append(row)
    return rows


def plus_reference(plus: pd.DataFrame) -> pd.DataFrame:
    """
    PTB-XL+ reference values where 12SL and Uni-G agree.

    QRS duration within 10 ms, PR within 20 ms (P waves found by both), heart rate within 3 per minute, axis
    within 15 degrees; the reference is their mean (circular mean for the axis).

    Parameters
    ----------
    plus : pd.DataFrame
        Output of ``lq.load_ptbxl_plus``.

    Returns
    -------
    pd.DataFrame
        ``record_key``, ``qrs_ref``, ``pr_ref``, ``hr_ref`` and ``axis_ref`` (NaN where the programs
        disagree).
    """
    out = pd.DataFrame({"record_key": plus["record_key"]})
    found = (plus["P_Found_Global_12sl"] == 1) & (plus["P_Found_Global_unig"] == 1)
    pairs = {"qrs": ("QRS_Dur_Global", 10.0, True), "pr": ("PR_Int_Global", 20.0, found)}
    for name, (column, tolerance, valid) in pairs.items():
        first, second = plus[f"{column}_12sl"], plus[f"{column}_unig"]
        valid = valid & (first > 0) & (second > 0) & ((first - second).abs() <= tolerance)
        out[f"{name}_ref"] = ((first + second) / 2).where(valid)
    hr = (plus["HR_Ventr_Global_12sl"], plus["HR__Global_unig"])
    out["hr_ref"] = ((hr[0] + hr[1]) / 2).where((hr[0] - hr[1]).abs() <= 3.0)
    first, second = plus["R_AxisFrontal_Global_12sl"], plus["QRS_AxisFront_Global_unig"]
    middle = second + lq.angle_difference(first, second) / 2
    out["axis_ref"] = middle.where(np.abs(lq.angle_difference(first, second)) <= 15.0)
    return out


def calibrate(workers: int) -> None:
    """
    Choose method B's QRS and P thresholds on PTB-XL training records (step ``calibrate``).

    Rule, fixed before the run: the QRS fraction with the largest share within 10 ms of the PTB-XL+ reference
    on 1,500 training records, then, at that QRS fraction, the P fraction with the largest share within 20 ms
    of the PR reference; ties go to the smaller fraction. 800 validation records report the result.

    Parameters
    ----------
    workers : int
        Processes.
    """
    started = time.time()
    records = lq.load_open_records()
    ptb = records[records["source"] == "ptbxl"]
    reference = plus_reference(lq.load_ptbxl_plus(ptb["record_key"].to_numpy()))
    reference = reference.merge(ptb[["record_key", "role"]], on="record_key")
    usable = reference[reference["qrs_ref"].notna()]
    sample = pd.concat(
        [
            usable[usable["role"] == "train"].sample(1500, random_state=lq.SEED),
            usable[usable["role"] == "validation"].sample(800, random_state=lq.SEED),
        ]
    )
    keys = sample["record_key"].tolist()
    rows = [
        row
        for chunk in parallel(grid_chunk, [keys[i : i + 50] for i in range(0, len(keys), 50)], workers)
        for row in chunk
    ]
    table = sample.merge(pd.DataFrame(rows), on="record_key")
    summary = []
    for role in ("train", "validation"):
        part = table[table["role"] == role]
        for q in QRS_GRID:
            summary.append(
                {
                    "role": role,
                    "quantity": "qrs",
                    "qrs_fraction": q,
                    "p_fraction": np.nan,
                    **lq.error_summary(part[f"qrs_{q}"], part["qrs_ref"], 10.0),
                }
            )
            for p in P_GRID:
                summary.append(
                    {
                        "role": role,
                        "quantity": "pr",
                        "qrs_fraction": q,
                        "p_fraction": p,
                        **lq.error_summary(part[f"pr_{q}_{p}"], part["pr_ref"], 20.0),
                    }
                )
    summary = pd.DataFrame(summary)
    train = summary[summary["role"] == "train"]
    qrs_rows = train[train["quantity"] == "qrs"]
    q_best = float(qrs_rows.loc[qrs_rows["within"].idxmax(), "qrs_fraction"])
    pr_rows = train[(train["quantity"] == "pr") & (train["qrs_fraction"] == q_best)]
    p_best = float(pr_rows.loc[pr_rows["within"].idxmax(), "p_fraction"])
    write_csv(table, "measurement_validation/calibration_records.csv")
    write_csv(summary, "measurement_validation/calibration_grid.csv")
    chosen = {"qrs_fraction": q_best, "p_fraction": p_best, "train_records": 1500, "validation_records": 800}
    write_json_atomic(lq.OUTPUT / "measurement_validation/calibration.json", chosen)
    receipt("calibrate", started, {"chosen": chosen})
    print(json.dumps(chosen))


def measure(workers: int, chunk: int = 200) -> None:
    """
    Run method B on every open record, with checkpointed shards (step ``measure``).

    Parameters
    ----------
    workers : int
        Processes.
    chunk : int
        Records per shard.

    Raises
    ------
    ValueError
        If the module's thresholds differ from the calibration's choice.
    """
    started = time.time()
    chosen = read_json(lq.OUTPUT / "measurement_validation/calibration.json")
    if (chosen["qrs_fraction"], chosen["p_fraction"]) != (mm.QRS_FRACTION, mm.P_FRACTION):
        raise ValueError("mlc03_measurements thresholds differ from calibration.json")
    keys = lq.load_open_records()["record_key"].tolist()
    shards = lq.SCRATCH / "measure_shards"
    shards.mkdir(parents=True, exist_ok=True)
    todo = [
        (i, keys[i : i + chunk])
        for i in range(0, len(keys), chunk)
        if not (shards / f"{i:06d}.parquet").exists()
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for (start, _), rows in zip(todo, pool.map(lq.measure_keys, [part for _, part in todo]), strict=True):
            frame = pd.DataFrame(rows)
            frame["status"] = frame["status"].astype(str)
            frame.to_parquet(shards / f"{start:06d}.parquet")
    frames = [pd.read_parquet(shards / f"{i:06d}.parquet") for i in range(0, len(keys), chunk)]
    table = pd.concat(frames, ignore_index=True)
    if table["record_key"].tolist() != keys:
        raise ValueError("measured records differ from the open records")
    table.to_parquet(lq.OUTPUT / "measurements_b.parquet")
    receipt(
        "measure",
        started,
        {
            "records": len(table),
            "status": table["status"].value_counts().to_dict(),
            "thresholds": chosen,
            "output_sha256": sha256_file(lq.OUTPUT / "measurements_b.parquet"),
        },
    )


def build_table() -> pd.DataFrame:
    """
    Build the analysis table of every open record (``lq.analysis_table``) from the stored A and B results.

    Returns
    -------
    pd.DataFrame
        One row per open record.
    """
    records = lq.load_open_records()
    keys = records["record_key"].to_numpy()
    b = lq.b_columns(pd.read_parquet(lq.OUTPUT / "measurements_b.parquet"))
    return lq.analysis_table(records, lq.method_a(keys), b)


def synthetic_chunk(cases: list[dict[str, float]]) -> list[dict[str, object]]:
    """
    Methods A and B on synthetic records against their known truth.

    Parameters
    ----------
    cases : list[dict[str, float]]
        Output of ``mm.synthetic_cases``.

    Returns
    -------
    list[dict[str, object]]
        Settings, truth (rate, axis, ectopic beats, ST and T levels) and both methods' values per case.
    """
    from ecg_experiment.measurement_extractor042 import extract_array

    rows = []
    for case in cases:
        settings = {key: value for key, value in case.items() if key != "seed"}
        signal, truth = mm.synthetic_record(seed=int(case["seed"]), **settings)
        b, a = mm.measure_record(signal), extract_array(np.ascontiguousarray(signal.T))
        onsets = truth["onsets"]
        inside = (onsets >= mm.PRE) & (onsets + mm.POST <= signal.shape[1])
        row = {
            **case,
            "hr_true": 60000.0 / (np.diff(onsets).mean() * 1000.0 / mm.FS),
            "axis_true": truth["axis_net_deg"],
            "ectopic_true": int((truth["ectopic"] & inside).sum()),
            **{f"{k}_true": v for k, v in mm.synthetic_levels(truth).items()},
            "hr_a": a.get("hr_bpm", np.nan),
            "qrs_a": a.get("qrs_dur_ms", np.nan),
            "pr_a": a.get("pr_ms", np.nan),
            "axis_a": float(
                np.degrees(np.arctan2(a.get("qrs_area_aVF", np.nan), a.get("qrs_area_I", np.nan)))
            ),
            "st60_II_a": a.get("st_II", np.nan),
            "t_II_a": a.get("t_amp_II", np.nan),
            "status_a": a["status"],
        }
        row |= {
            f"{k}_b": b.get(k, np.nan)
            for k in (
                "hr_bpm",
                "qrs_ms",
                "pr_ms",
                "axis_deg",
                "st60_II",
                "st60_V5",
                "t_II",
                "t_V5",
                "v1_positive_peaks",
                "v1_terminal_mv",
                "term_I",
                "term_V6",
                "p_snr",
                "p_amp_mv",
                "rr_nrmssd",
                "n_premature_aberrant",
            )
        }
        row["status_b"] = b["status"]
        rows.append(row)
    return rows


NAMES = {"hr_bpm": "hr", "qrs_ms": "qrs", "pr_ms": "pr"}


def ludb_chunk(ids: list[int]) -> list[dict[str, object]]:
    """
    Methods A and B on LUDB records against the manual delineation.

    Parameters
    ----------
    ids : list[int]
        LUDB record numbers.

    Returns
    -------
    list[dict[str, object]]
        Reference and both methods' rate, QRS duration, PR interval and P-wave measures per record.
    """
    import wfdb

    from ecg_experiment.measurement_extractor042 import extract_array

    rows = []
    for number in ids:
        path = str(lq.LUDB / "data" / str(number))
        signal = wfdb.rdrecord(path).p_signal.T[:, :5000]
        b, a = mm.measure_record(signal), extract_array(np.ascontiguousarray(signal.T))
        rows.append(
            {
                "id": number,
                **{f"{NAMES.get(k, k)}_ref": v for k, v in mm.ludb_reference(path).items()},
                "hr_a": a.get("hr_bpm", np.nan),
                "qrs_a": a.get("qrs_dur_ms", np.nan),
                "pr_a": a.get("pr_ms", np.nan),
                "status_a": a["status"],
                "hr_b": b.get("hr_bpm", np.nan),
                "qrs_b": b.get("qrs_ms", np.nan),
                "pr_b": b.get("pr_ms", np.nan),
                "status_b": b["status"],
                "p_snr": b.get("p_snr", np.nan),
                "p_amp_mv": b.get("p_amp_mv", np.nan),
                "rr_nrmssd_b": b.get("rr_nrmssd", np.nan),
                "n_premature_aberrant": b.get("n_premature_aberrant", np.nan),
            }
        )
    return rows


def ludb_findings(ludb: pd.DataFrame) -> pd.DataFrame:
    """
    Compare method B's rhythm findings on LUDB with its stated rhythm and extrasystoles.

    Parameters
    ----------
    ludb : pd.DataFrame
        ``ludb_chunk`` rows with LUDB's ``Rhythms`` and ``Extrasystolies`` text.

    Returns
    -------
    pd.DataFrame
        Records with and without each stated finding (atrial fibrillation or flutter; ventricular
        extrasystole) and the share method B flags (irregular RR without a P wave; at least one premature
        aberrant beat).
    """
    rhythm = ludb["Rhythms"].fillna("").str.lower()
    extra = ludb["Extrasystolies"].fillna("").str.lower()
    measured = ludb["status_b"] == "ok"
    no_p = ~((ludb["p_snr"] >= lq.P_SNR_MIN) & (ludb["p_amp_mv"] >= lq.P_AMP_MIN))
    flags = {
        "af_or_flutter": (
            rhythm.str.contains("atrial fibrillation|atrial flutter"),
            (ludb["rr_nrmssd_b"] >= lq.IRREGULAR_NRMSSD) & no_p,
        ),
        "ventricular extrasystole": (
            extra.str.contains("ventricular extrasystole"),
            ludb["n_premature_aberrant"] >= 1,
        ),
    }
    rows = []
    for name, (stated, flagged) in flags.items():
        rows.append(
            {
                "finding": name,
                "with_finding": int((stated & measured).sum()),
                "flagged_with": float(flagged[stated & measured].mean()),
                "without_finding": int((~stated & measured).sum()),
                "flagged_without": float(flagged[~stated & measured].mean()),
            }
        )
    return pd.DataFrame(rows)


SYNTHETIC_CHECKS = (
    ("hr", "hr_true", "hr_a", "hr_bpm_b", 5.0, False),
    ("qrs", "qrs_ms", "qrs_a", "qrs_ms_b", 10.0, False),
    ("pr", "pr_ms", "pr_a", "pr_ms_b", 20.0, False),
    ("axis", "axis_true", "axis_a", "axis_deg_b", 15.0, True),
    ("st60_II", "st60_II_true", "st60_II_a", "st60_II_b", 0.05, False),
    ("t_II", "t_II_true", "t_II_a", "t_II_b", 0.1, False),
)


def synthetic_summary(table: pd.DataFrame) -> pd.DataFrame:
    """
    Error of methods A and B on synthetic records, overall and by noise level.

    Parameters
    ----------
    table : pd.DataFrame
        ``synthetic_chunk`` rows.

    Returns
    -------
    pd.DataFrame
        ``lq.error_summary`` per quantity, method and noise level (PR without AF and ectopy).
    """
    rows = []
    for noise, part in [("all", table), *table.groupby("noise_mv")]:
        for name, truth, a, b, tolerance, circular in SYNTHETIC_CHECKS:
            usable = part[part["af"] != 1] if name == "pr" else part
            for method, column in (("A", a), ("B", b)):
                rows.append(
                    {
                        "quantity": name,
                        "method": method,
                        "noise_mv": noise,
                        "tolerance": tolerance,
                        **lq.error_summary(usable[column], usable[truth], tolerance, circular),
                    }
                )
    return pd.DataFrame(rows)


def synthetic_detection(table: pd.DataFrame) -> pd.DataFrame:
    """
    Count method B's categorical findings on synthetic records with and without them.

    Parameters
    ----------
    table : pd.DataFrame
        ``synthetic_chunk`` rows.

    Returns
    -------
    pd.DataFrame
        For RBBB morphology, P-wave presence, irregular rhythm and premature aberrant beats: records with and
        without the finding and the share flagged in each.
    """
    b = table.rename(
        columns={
            "v1_positive_peaks_b": "v1_positive_peaks",
            "v1_terminal_mv_b": "v1_terminal_mv",
            "term_I_b": "term_I",
            "term_V6_b": "term_V6",
        }
    )
    morphology = ((b["v1_positive_peaks"] >= 2) | (b["v1_terminal_mv"] > 0)) & (
        (b["term_I"] < 0) | (b["term_V6"] < 0)
    )
    flags = {
        "rbbb_morphology": (table["rbbb"] == 1, morphology),
        "p_present": (
            table["af"] != 1,
            (table["p_snr_b"] >= lq.P_SNR_MIN) & (table["p_amp_mv_b"] >= lq.P_AMP_MIN),
        ),
        "irregular": (table["af"] == 1, table["rr_nrmssd_b"] >= lq.IRREGULAR_NRMSSD),
        "premature_aberrant": (table["ectopic_true"] >= 1, table["n_premature_aberrant_b"] >= 1),
    }
    rows = []
    for name, (truth, flagged) in flags.items():
        measured = table["status_b"] == "ok"
        rows.append(
            {
                "finding": name,
                "with_finding": int((truth & measured).sum()),
                "flagged_with": float(flagged[truth & measured].mean()),
                "without_finding": int((~truth & measured).sum()),
                "flagged_without": float(flagged[~truth & measured].mean()),
            }
        )
    return pd.DataFrame(rows)


def plus_comparison(table: pd.DataFrame, calibration: set[str]) -> pd.DataFrame:
    """
    Error of methods A and B and of their consensus against PTB-XL+ on open PTB-XL records.

    Parameters
    ----------
    table : pd.DataFrame
        Analysis table joined with ``plus_reference`` and the raw PTB-XL+ columns.
    calibration : set[str]
        Keys of the calibration sample, reported apart.

    Returns
    -------
    pd.DataFrame
        ``lq.error_summary`` per quantity, method and subset (training records outside the calibration
        sample, validation records outside it, the calibration sample).
    """
    t = table.copy()
    t["sokolow_ref"] = mean_if_close(
        -t["S_Amp_V1_12sl"] + t[["R_Amp_V5_12sl", "R_Amp_V6_12sl"]].max(axis=1),
        -t["S_Amp_V1_unig"] + t[["R_Amp_V5_unig", "R_Amp_V6_unig"]].max(axis=1),
        0.3,
    )
    t["cornell_ref"] = mean_if_close(
        t["R_Amp_aVL_12sl"] - t["S_Amp_V3_12sl"], t["R_Amp_aVL_unig"] - t["S_Amp_V3_unig"], 0.3
    )
    for lead in ("II", "V5"):
        signed = np.where(
            t[f"T+_Amp_{lead}_unig"].abs() >= t[f"T-_Amp_{lead}_unig"].abs(),
            t[f"T+_Amp_{lead}_unig"],
            t[f"T-_Amp_{lead}_unig"],
        )
        t[f"t_{lead}_ref"] = mean_if_close(t[f"T_Amp_{lead}_12sl"], pd.Series(signed, index=t.index), 0.1)
        t[f"qdur_{lead}_ref"] = mean_if_close(t[f"Q_Dur_{lead}_12sl"], t[f"Q_Dur_{lead}_unig"], 8.0)
    t["a_sokolow"] = t["a_s_depth_V1"] + t[["a_r_amp_V5", "a_r_amp_V6"]].max(axis=1)
    t["a_cornell"] = t["a_r_amp_aVL"] + t["a_s_depth_V3"]
    checks = (
        ("hr", "hr_ref", "hr_a", "hr_b", "hr", 5.0, False),
        ("qrs", "qrs_ref", "qrs_a", "qrs_b", "qrs", 10.0, False),
        ("pr", "pr_ref", "pr_a", "pr_b", "pr", 20.0, False),
        ("axis", "axis_ref", "axis_a", "axis_b", "axis", 15.0, True),
        ("sokolow", "sokolow_ref", "a_sokolow", "sokolow_mv", None, 0.5, False),
        ("cornell", "cornell_ref", "a_cornell", "cornell_mv", None, 0.5, False),
        ("st80_V5", "ST_Amp80ms_V5_unig", None, "st80_V5", None, 0.05, False),
        ("t_V5", "t_V5_ref", "a_t_amp_V5", "t_V5", None, 0.1, False),
        ("t_II", "t_II_ref", "a_t_amp_II", "t_II", None, 0.1, False),
        ("qdur_II", "qdur_II_ref", None, "qdur_II", None, 10.0, False),
        ("qdur_V5", "qdur_V5_ref", None, "qdur_V5", None, 10.0, False),
    )
    subsets = {
        "train outside calibration": (t["role"] == "train") & ~t["record_key"].isin(calibration),
        "validation outside calibration": (t["role"] == "validation") & ~t["record_key"].isin(calibration),
        "calibration sample": t["record_key"].isin(calibration),
    }
    rows = []
    for subset, mask in subsets.items():
        part = t[mask]
        for name, reference, a, b, agreed, tolerance, circular in checks:
            for method, column in (("A", a), ("B", b), ("consensus", agreed)):
                if column is not None:
                    rows.append(
                        {
                            "quantity": name,
                            "method": method,
                            "subset": subset,
                            "tolerance": tolerance,
                            **lq.error_summary(part[column], part[reference], tolerance, circular),
                        }
                    )
    return pd.DataFrame(rows)


def mean_if_close(first: pd.Series, second: pd.Series, tolerance: float) -> pd.Series:
    """
    Mean of two programs' values where they differ by at most a tolerance.

    Parameters
    ----------
    first, second : pd.Series
        Aligned values.
    tolerance : float
        Largest accepted difference.

    Returns
    -------
    pd.Series
        Their mean, NaN elsewhere.
    """
    return ((first + second) / 2).where((first - second).abs() <= tolerance)


def plus_agreement(table: pd.DataFrame) -> pd.DataFrame:
    """
    Compare method B's P-wave and V1 R' findings with PTB-XL+ where 12SL and Uni-G agree.

    Parameters
    ----------
    table : pd.DataFrame
        Analysis table joined with PTB-XL+ columns.

    Returns
    -------
    pd.DataFrame
        Two-by-two counts of method B's flag against the programs' shared finding.
    """
    pairs = {
        "p_present": (table["p_present"], table["P_Found_Global_12sl"], table["P_Found_Global_unig"]),
        "v1_two_positive_peaks": (
            table["v1_positive_peaks"] >= 2,
            table["Rp_Amp_V1_12sl"] > 0,
            table["Rp_Amp_V1_unig"] > 0,
        ),
    }
    rows = []
    measured = table["status_b"] == "ok"
    for name, (flag, first, second) in pairs.items():
        first, second = first.astype(bool), second.astype(bool)
        agreed = measured & (first == second)
        for reference in (True, False):
            for found in (True, False):
                rows.append(
                    {
                        "finding": name,
                        "reference": reference,
                        "method_b": found,
                        "records": int((agreed & (first == reference) & (flag == found)).sum()),
                    }
                )
    return pd.DataFrame(rows)


def source_agreement(table: pd.DataFrame) -> pd.DataFrame:
    """
    Measurement failures and agreement of methods A and B per source.

    Parameters
    ----------
    table : pd.DataFrame
        Analysis table.

    Returns
    -------
    pd.DataFrame
        Per source and quantity: records, share measured by A, by B, by both, share agreeing among both, and
        the consensus share; plus each method's failure statuses.
    """
    rows = []
    for source, part in table.groupby("source", sort=False):
        status = {f"a_status_{k}": int(v) for k, v in part["a_status"].value_counts().items()}
        status |= {f"b_status_{k}": int(v) for k, v in part["status_b"].value_counts().items()}
        for name in lq.TOLERANCE:
            rows.append(
                {
                    "source": source,
                    "quantity": name,
                    "records": len(part),
                    "a_measured": float(part[f"{name}_a"].notna().mean()),
                    "b_measured": float(part[f"{name}_b"].notna().mean()),
                    "both": float(part[f"{name}_both"].mean()),
                    "agree_given_both": float(
                        part[f"{name}_agree"].sum() / max(part[f"{name}_both"].sum(), 1)
                    ),
                    "consensus": float(part[name].notna().mean()),
                    "tolerance": lq.TOLERANCE[name],
                    **status,
                }
            )
        rows.append(
            {
                "source": source,
                "quantity": "p_present",
                "records": len(part),
                "b_measured": float((part["status_b"] == "ok").mean()),
                "consensus": float(part["p_present"].mean()),
            }
        )
    return pd.DataFrame(rows)


def validate(workers: int) -> None:
    """
    Measurement validation on synthetic records, LUDB and PTB-XL+ (step ``validate``).

    Parameters
    ----------
    workers : int
        Processes.
    """
    started = time.time()
    out = "measurement_validation/"
    cases = mm.synthetic_cases(400, lq.SEED)
    synthetic = pd.DataFrame(
        [
            row
            for chunk in parallel(
                synthetic_chunk, [cases[i : i + 25] for i in range(0, len(cases), 25)], workers
            )
            for row in chunk
        ]
    )
    write_csv(synthetic, out + "synthetic_records.csv")
    write_csv(synthetic_summary(synthetic), out + "synthetic_summary.csv")
    write_csv(synthetic_detection(synthetic), out + "synthetic_detection.csv")
    ludb = pd.DataFrame(
        [
            row
            for chunk in parallel(ludb_chunk, [list(range(i, i + 10)) for i in range(1, 201, 10)], workers)
            for row in chunk
        ]
    )
    ludb = ludb.merge(
        pd.read_csv(lq.LUDB / "ludb.csv").rename(columns={"ID": "id"})[["id", "Rhythms", "Extrasystolies"]],
        on="id",
    )
    write_csv(ludb, out + "ludb_records.csv")
    write_csv(ludb_findings(ludb), out + "ludb_findings.csv")
    rows = [
        {
            "quantity": q,
            "method": m,
            "tolerance": tol,
            **lq.error_summary(ludb[f"{q}_{m.lower()}"], ludb[f"{q}_ref"], tol),
        }
        for q, tol in (("hr", 5.0), ("qrs", 10.0), ("pr", 20.0))
        for m in ("A", "B")
    ]
    write_csv(pd.DataFrame(rows), out + "ludb_summary.csv")
    table = build_table()
    ptb = table[table["source"] == "ptbxl"]
    plus = lq.load_ptbxl_plus(ptb["record_key"].to_numpy())
    joined = ptb.merge(plus, on="record_key").merge(plus_reference(plus), on="record_key")
    calibration = set(pd.read_csv(lq.OUTPUT / out / "calibration_records.csv")["record_key"])
    write_csv(plus_comparison(joined, calibration), out + "ptbxl_plus_summary.csv")
    write_csv(plus_agreement(joined), out + "ptbxl_plus_findings.csv")
    write_csv(source_agreement(table), out + "source_agreement.csv")
    receipt(
        "validate",
        started,
        {
            "synthetic_cases": len(synthetic),
            "ludb_records": len(ludb),
            "ptbxl_records": len(joined),
            "ludb_sha256sums": sha256_file(lq.LUDB / "SHA256SUMS.txt"),
            "ptbxl_plus_sha256": {
                name: sha256_file(lq.PTBXL_PLUS / f"{name}_features.csv") for name in lq.PLUS_COLUMNS
            },
        },
    )


def labels(workers: int) -> None:
    """
    Items 1, 2, 3 and 5 on the analysis table (step ``labels``).

    Parameters
    ----------
    workers : int
        Unused; the step runs in one process.
    """
    started = time.time()
    table = build_table()
    table.drop(columns=["codes"]).to_parquet(lq.OUTPUT / "analysis.parquet")
    item1 = pd.DataFrame(
        [row for condition in CONDITIONS for row in lq.label_measurement_rows(table, condition)]
    )
    write_csv(item1, "item1_label_vs_measurement.csv")
    write_csv(
        pd.DataFrame([{"condition": c, **dict(spec)} for c, spec in lq.CHECKS.items()]),
        "item1_definitions.csv",
    )
    prevalence = pd.concat([lq.matched_prevalence(table, name) for name in lq.MATCHED], ignore_index=True)
    write_csv(prevalence, "item2_matched_prevalence.csv")
    pairs = lq.inconsistent_bins(prevalence)
    write_csv(pairs, "item2_source_pairs.csv")
    cutoffs = [
        lq.implied_cutoff(table, c, s) for c in ("sinus_bradycardia", "sinus_tachycardia") for s in lq.SOURCES
    ]
    write_csv(pd.DataFrame(cutoffs), "item2_implied_cutoffs.csv")
    write_csv(lq.code_usage(table), "item2_code_usage.csv")
    write_csv(lq.separate_coding(table), "item2_separate_coding.csv")
    write_csv(pd.DataFrame(lq.contradiction_rows(table)), "item3_contradictions.csv")
    write_csv(lq.undefined_rows(table, lq.minimum_positive_pairs()), "item5_undefined.csv")
    receipt(
        "labels",
        started,
        {
            "records": len(table),
            "roles": table["role"].value_counts().to_dict(),
            "inconsistent_pairs": int(pairs["inconsistent"].sum()),
            "analysis_sha256": sha256_file(lq.OUTPUT / "analysis.parquet"),
        },
    )


def p0_settings() -> dict[str, float]:
    """
    P0's chosen C per condition, read from its MLC-02 candidate file.

    Returns
    -------
    dict[str, float]
        C per condition, in ``CONDITIONS`` order.

    Raises
    ------
    ValueError
        If the candidate is not P0's logistic model on xECG features.
    """
    candidate = read_json(P0 / "candidate.json")
    if candidate["candidate"] != "P0" or candidate["inputs"] != ["xecg"] or candidate["kind"] != "logistic":
        raise ValueError("candidate.json is not P0")
    return {name: float(candidate["chosen_c"][name]) for name in CONDITIONS}


def fit_task(task: tuple[str, int, int, float]) -> dict[str, object]:
    """
    One P0 fit: a fold (or the MLC-02 train/validation split) and one condition.

    Parameters
    ----------
    task : tuple[str, int, int, float]
        Split name (``fold`` or ``mlc02``), fold index, condition index and C.

    Returns
    -------
    dict[str, object]
        The task, held-out probabilities, iterations, training records and seconds.
    """
    kind, fold, column, c = task
    data = np.load(lq.SCRATCH / "oof_inputs.npz")
    x = np.load(lq.SCRATCH / "oof_xecg.npy", mmap_mode="r")
    if kind == "fold":
        train, test = data["fold"] != fold, data["fold"] == fold
    else:
        train, test = data["role"] == "train", data["role"] == "validation"
    started = time.time()
    fitted = lq.fit_split(np.asarray(x), data["labels"][:, column], train, test, c)
    return {
        "kind": kind,
        "fold": fold,
        "column": column,
        "c": c,
        "seconds": time.time() - started,
        "iterations": fitted["iterations"],
        "records": fitted["records"],
        "probability": fitted["probability"],
        "test": np.flatnonzero(test),
    }


def stored_p0_check(table: pd.DataFrame) -> dict[str, object]:
    """
    Integrity check: P0's MLC-02 validation selection metric from its stored predictions.

    Parameters
    ----------
    table : pd.DataFrame
        Open records with labels, in manifest order.

    Returns
    -------
    dict[str, object]
        The recomputed metric, the stored one and their difference.

    Raises
    ------
    ValueError
        If the metric differs from 0.9642697112976489.
    """
    from ecg_experiment.mlc02_data import load_predictions
    from ecg_experiment.mlc02_evaluation import selection_metric

    validation = table[table["role"] == "validation"]
    stored = load_predictions(P0 / "validation__seeds-42.npz", validation["record_key"].to_numpy())
    labels = validation[[f"b_{name}" for name in CONDITIONS]].to_numpy(float)
    metric = selection_metric(labels, stored["probability"], validation["source"].to_numpy())[
        "selection_metric"
    ]
    if metric != P0_SELECTION_METRIC:
        raise ValueError(f"P0 selection metric {metric} differs from {P0_SELECTION_METRIC}")
    return {
        "recomputed": metric,
        "expected": P0_SELECTION_METRIC,
        "difference": metric - P0_SELECTION_METRIC,
        "stored_file_sha256": sha256_file(P0 / "validation__seeds-42.npz"),
    }


def oof(workers: int) -> None:
    """
    P0 integrity checks and five-fold grouped out-of-fold probabilities (step ``oof``).

    Parameters
    ----------
    workers : int
        Processes; each limited to the BLAS and OpenMP threads of the environment.
    """
    from ecg_experiment.mlc02_data import load_predictions
    from ecg_experiment.mlc02_evaluation import selection_metric

    started = time.time()
    table = lq.load_open_records()
    integrity = {"stored_predictions": stored_p0_check(table)}
    keys = table["record_key"].to_numpy()
    folds = lq.group_folds(table["group_id"].to_numpy(), table["source"].to_numpy())
    labels = table[[f"b_{name}" for name in CONDITIONS]].to_numpy(float)
    np.save(lq.SCRATCH / "oof_xecg.npy", lq.load_feature_rows("xecg.npy", keys))
    np.savez(lq.SCRATCH / "oof_inputs.npz", fold=folds, labels=labels, role=table["role"].to_numpy(dtype=str))
    settings = p0_settings()
    tasks = [("mlc02", -1, j, settings[name]) for j, name in enumerate(CONDITIONS)]
    tasks += [("fold", k, j, settings[name]) for k in range(5) for j, name in enumerate(CONDITIONS)]
    results = parallel(fit_task, tasks, workers)
    refit = np.zeros(((table["role"] == "validation").sum(), len(CONDITIONS)))
    probability = np.full(labels.shape, np.nan)
    for result in results:
        if result["kind"] == "mlc02":
            refit[:, result["column"]] = result["probability"]
        else:
            probability[result["test"], result["column"]] = result["probability"]
    validation = table["role"] == "validation"
    stored = load_predictions(P0 / "validation__seeds-42.npz", keys[validation.to_numpy()])["probability"]
    integrity["refit"] = {
        "max_abs_difference": float(np.abs(refit - stored).max()),
        "selection_metric": selection_metric(
            labels[validation.to_numpy()], refit, table.loc[validation, "source"].to_numpy()
        )["selection_metric"],
    }
    np.savez(
        lq.OUTPUT / "oof_p0.npz",
        record_key=keys,
        fold=folds,
        probability=probability,
        condition=np.asarray(CONDITIONS),
    )
    log = pd.DataFrame([{k: v for k, v in r.items() if k not in ("probability", "test")} for r in results])
    write_csv(log, "oof_fits.csv")
    receipt(
        "oof",
        started,
        {
            "integrity": integrity,
            "c": settings,
            "folds": 5,
            "fold_seed": lq.SEED,
            "records_per_fold": np.bincount(folds).tolist(),
            "oof_sha256": sha256_file(lq.OUTPUT / "oof_p0.npz"),
        },
    )
    print(json.dumps(integrity, indent=1))


RELEVANT = {
    "sinus_tachycardia": ("hr_a", "hr_b", "hr", "p_present", "p_snr", "sinus_codes"),
    "sinus_bradycardia": ("hr_a", "hr_b", "hr", "p_present", "p_snr", "sinus_codes"),
    "sinus_arrhythmia": ("rr_range_dominant_b", "rr_range_a", "hr", "p_present", "p_consistency"),
    "af_or_flutter": ("rr_nrmssd_b", "rr_nrmssd_a", "p_present", "p_snr", "p_consistency", "hr"),
    "premature_ventricular_complex": ("n_premature_aberrant", "n_aberrant", "n_beats", "rr_nrmssd_b"),
    "first_degree_av_block": ("pr_a", "pr_b", "pr", "p_present", "hr"),
    "complete_rbbb": (
        "qrs_a",
        "qrs_b",
        "qrs",
        "v1_positive_peaks",
        "v1_terminal_mv",
        "term_I",
        "term_V6",
        "rbbb_morphology",
    ),
    "incomplete_rbbb": (
        "qrs_a",
        "qrs_b",
        "qrs",
        "v1_positive_peaks",
        "v1_terminal_mv",
        "term_I",
        "term_V6",
        "rbbb_morphology",
    ),
    "lafb": ("axis_a", "axis_b", "axis", "qrs"),
    "left_axis_deviation": ("axis_a", "axis_b", "axis", "qrs"),
    "lvh": ("sokolow_mv", "cornell_mv", "voltage_criterion", "sex", "qrs"),
    "st_depression": ("st_min_mv", "st_dep_group", "st60_V5", "st60_II", "st60_V2"),
    "t_wave_abnormality": ("t_inverted", "t_low", "t_I", "t_II", "t_V5", "t_V6"),
    "mi_any": ("q_group", "qdur_II", "qdur_aVF", "qdur_V2", "qdur_V3", "qdur_V5", "r_V2", "r_V3"),
}
COMMON = ("status_b", "a_status", "duration_s", "n_flat_leads", "raw_codes")


def suspects_step(workers: int) -> None:
    """
    Confident learning per condition and source, and the most suspected labels (step ``suspects``).

    Parameters
    ----------
    workers : int
        Unused; the step runs in one process.
    """
    started = time.time()
    table = pd.read_parquet(lq.OUTPUT / "analysis.parquet")
    stored = np.load(lq.OUTPUT / "oof_p0.npz")
    if not np.array_equal(stored["record_key"], table["record_key"].to_numpy()):
        raise ValueError("out-of-fold probabilities are not aligned with the analysis table")
    probability = stored["probability"]
    learning, listed = [], []
    for column, condition in enumerate(CONDITIONS):
        label = table[f"b_{condition}"]
        for source in (*lq.SOURCES, "pooled"):
            mask = label.notna() & ((table["source"] == source) if source != "pooled" else True)
            y, p = label[mask].to_numpy(int), probability[mask.to_numpy(), column]
            if min(y.sum(), (1 - y).sum()) < 5:
                continue
            result = lq.confident_learning(y, p)
            learning.append(cl_row(condition, source, table[mask], y, result))
            if source == "pooled":
                continue
            for kind, positions in lq.suspects(y, p).items():
                part = table[mask].iloc[positions]
                flags = result["flag_" + kind][positions]
                for rank, (_, row) in enumerate(part.iterrows(), start=1):
                    listed.append(
                        {
                            "condition": condition,
                            "source": source,
                            "kind": kind,
                            "rank": rank,
                            "record_key": row["record_key"],
                            "role": row["role"],
                            "label": int(y[positions[rank - 1]]),
                            "probability": float(p[positions[rank - 1]]),
                            "cl_flag": bool(flags[rank - 1]),
                            **{name: row[name] for name in (*RELEVANT[condition], *COMMON)},
                        }
                    )
    write_csv(pd.DataFrame(learning), "item4_confident_learning.csv")
    suspects = pd.DataFrame(listed)
    for condition, part in suspects.groupby("condition", sort=False):
        write_csv(part.dropna(axis=1, how="all"), f"suspects/{condition}.csv")
    receipt("suspects", started, {"rows": len(suspects)})


def cl_row(
    condition: str, source: str, part: pd.DataFrame, y: np.ndarray, result: dict[str, object]
) -> dict[str, object]:
    """
    One row of the confident-learning table.

    Parameters
    ----------
    condition, source : str
        Condition and source (or ``pooled``).
    part : pd.DataFrame
        Analysis rows of the defined records.
    y : np.ndarray
        Their labels.
    result : dict[str, object]
        Output of ``lq.confident_learning``.

    Returns
    -------
    dict[str, object]
        Counts, thresholds, joints and estimated wrong labels, also by role.
    """
    joint = result["confident_joint"]
    train = (part["role"] == "train").to_numpy()
    return {
        "condition": condition,
        "source": source,
        "defined": len(y),
        "positives": int(y.sum()),
        "threshold_positive": result["threshold_positive"],
        "threshold_negative": result["threshold_negative"],
        "joint_00": joint[0, 0],
        "joint_01": joint[0, 1],
        "joint_10": joint[1, 0],
        "joint_11": joint[1, 1],
        "estimated_false_positives": result["estimated_false_positives"],
        "estimated_false_negatives": result["estimated_false_negatives"],
        "estimated_wrong": result["estimated_false_positives"] + result["estimated_false_negatives"],
        "estimated_wrong_share": (result["estimated_false_positives"] + result["estimated_false_negatives"])
        / len(y),
        "flagged_false_positives_train": int((result["flag_false_positive"] & train).sum()),
        "flagged_false_positives_validation": int((result["flag_false_positive"] & ~train).sum()),
        "flagged_false_negatives_train": int((result["flag_false_negative"] & train).sum()),
        "flagged_false_negatives_validation": int((result["flag_false_negative"] & ~train).sum()),
    }


def describe_codes(raw: str, source: str, names: dict[str, str]) -> str:
    """
    Write a record's codes with SNOMED abbreviations, for a review sheet.

    Parameters
    ----------
    raw : str
        ``raw_codes`` field.
    source : str
        Manifest source.
    names : dict[str, str]
        SNOMED code to official abbreviation.

    Returns
    -------
    str
        PTB-XL statements as stored; SNOMED codes as ``abbreviation(code)``.
    """
    if source == "ptbxl":
        return raw
    return ", ".join(f"{names.get(code, '?')}({code})" for code in raw.split(",") if code)


def format_value(value: object) -> str:
    """
    Short text of a measurement for a sheet.

    Parameters
    ----------
    value : object
        Number, boolean or text.

    Returns
    -------
    str
        Rounded number, ``yes``/``no`` or the text.
    """
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, float, np.floating, np.integer)):
        return "NA" if not np.isfinite(value) else f"{value:.3g}"
    return str(value)


def sheet_task(rows: list[dict[str, object]]) -> int:
    """
    Plot the review sheets of some suspects.

    Parameters
    ----------
    rows : list[dict[str, object]]
        Suspect rows with ``path``, ``heading`` and ``lines``.

    Returns
    -------
    int
        Sheets written.
    """
    signals = lq.cached_windows(np.array([row["record_key"] for row in rows]))
    for row, signal in zip(rows, signals, strict=True):
        lq.plot_review(signal, row["heading"], row["lines"], Path(row["path"]))
    return len(rows)


def sheets(workers: int) -> None:
    """
    Review sheets of every listed suspect (step ``sheets``).

    Parameters
    ----------
    workers : int
        Processes.
    """
    from ecg_experiment.challenge_labels import load_official

    started = time.time()
    names = load_official()["abbreviation"].to_dict()
    table = pd.read_parquet(lq.OUTPUT / "analysis.parquet").set_index("record_key")
    jobs = []
    for condition in CONDITIONS:
        suspects = pd.read_csv(lq.OUTPUT / f"suspects/{condition}.csv", dtype={"record_key": str})
        for _, row in suspects.iterrows():
            record = table.loc[row["record_key"]]
            kind = "fp" if row["kind"] == "false_positive" else "fn"
            measures = "; ".join(f"{name} {format_value(record[name])}" for name in RELEVANT[condition])
            general = (
                f"HR A/B {format_value(record['hr_a'])}/{format_value(record['hr_b'])}; QRS A/B "
                f"{format_value(record['qrs_a'])}/{format_value(record['qrs_b'])} ms; PR A/B "
                f"{format_value(record['pr_a'])}/{format_value(record['pr_b'])} ms; axis A/B "
                f"{format_value(record['axis_a'])}/{format_value(record['axis_b'])}; P present "
                f"{format_value(record['p_present'])}; status A/B {record['a_status']}/{record['status_b']}"
            )
            jobs.append(
                {
                    "record_key": row["record_key"],
                    "path": str(
                        lq.REVIEW
                        / condition
                        / row["source"]
                        / f"{kind}_{row['rank']:02d}_{row['record_key'].replace(':', '_')}.png"
                    ),
                    "heading": (
                        f"{condition} suspected {row['kind'].replace('_', ' ')} #{row['rank']} | "
                        f"{row['record_key']} ({row['source']}, {row['role']}, "
                        f"{record['duration_s']:.0f} s) | "
                        f"label {row['label']} | P0 out-of-fold p = {row['probability']:.3f} | "
                        f"confident-learning flag {format_value(row['cl_flag'])}"
                    ),
                    "lines": [
                        measures,
                        general,
                        "codes: " + describe_codes(record["raw_codes"], row["source"], names),
                    ],
                }
            )
    written = sum(parallel(sheet_task, [jobs[i : i + 40] for i in range(0, len(jobs), 40)], workers))
    receipt("sheets", started, {"sheets": written})


def windows(workers: int) -> None:
    """
    Run method B on every 10 s window of the open records longer than 10 s (step ``windows``).

    Parameters
    ----------
    workers : int
        Processes.
    """
    started = time.time()
    table = lq.load_open_records()
    long = table[table["duration_s"] > 10.0 + 1e-6]
    jobs = list(zip(long["record_key"], long["raw_path"], long["source"], strict=True))
    with ProcessPoolExecutor(
        max_workers=workers, initializer=lq.open_window_guard, initargs=(long[["raw_path"]],)
    ) as pool:
        rows = list(pool.map(lq.whole_record_windows, jobs, chunksize=8))
    result = pd.DataFrame(rows)
    for column in ("ectopy_by_window", "af_by_window", "hr_by_window"):
        result[column] = result[column].map(json.dumps)
    result = long[
        ["record_key", "source", "role", "duration_s", "b_premature_ventricular_complex", "b_af_or_flutter"]
    ].merge(result, on="record_key")
    write_csv(result, "item1_whole_record_windows.csv")
    summary = []
    for condition, first, later in (
        ("premature_ventricular_complex", result["ectopy_first"] >= 1, result["ectopy_later"] >= 1),
        ("af_or_flutter", result["af_first"].astype(bool), result["af_later_windows"] >= 1),
    ):
        pattern = pd.Series(
            np.select([first, later], ["in_first_10s", "only_after_10s"], "in_no_window"), index=result.index
        )
        label = result[f"b_{condition}"].map({1.0: "positive", 0.0: "negative"}).fillna("undefined")
        for (source, state), part in pattern.groupby([result["source"], label]):
            counts = part.value_counts()
            summary.append(
                {
                    "condition": condition,
                    "source": source,
                    "label": state,
                    "records": len(part),
                    **{k: int(counts.get(k, 0)) for k in ("in_first_10s", "only_after_10s", "in_no_window")},
                }
            )
    write_csv(pd.DataFrame(summary), "item1_whole_record_summary.csv")
    receipt("windows", started, {"records": len(result), "windows": int(result["windows"].sum())})


MEASURED_DEFINITION = (
    "sinus_tachycardia",
    "sinus_bradycardia",
    "first_degree_av_block",
    "complete_rbbb",
    "lafb",
    "left_axis_deviation",
)


def cl_flags(table: pd.DataFrame) -> pd.DataFrame:
    """
    Within-source confident-learning flags of every defined label.

    Parameters
    ----------
    table : pd.DataFrame
        Analysis table aligned with ``oof_p0.npz``.

    Returns
    -------
    pd.DataFrame
        ``fp_<condition>`` and ``fn_<condition>`` booleans.
    """
    probability = np.load(lq.OUTPUT / "oof_p0.npz")["probability"]
    flags = pd.DataFrame(
        False, index=table.index, columns=[f"{kind}_{c}" for c in CONDITIONS for kind in ("fp", "fn")]
    )
    for column, condition in enumerate(CONDITIONS):
        label = table[f"b_{condition}"]
        for source in lq.SOURCES:
            mask = label.notna() & (table["source"] == source)
            y = label[mask].to_numpy(int)
            if min(y.sum(), (1 - y).sum()) < 5:
                continue
            result = lq.confident_learning(y, probability[mask.to_numpy(), column])
            flags.loc[mask, f"fp_{condition}"] = result["flag_false_positive"]
            flags.loc[mask, f"fn_{condition}"] = result["flag_false_negative"]
    return flags


def candidate_rules(
    table: pd.DataFrame, whole: pd.DataFrame, flags: pd.DataFrame, coding: pd.DataFrame, cutoffs: pd.DataFrame
) -> list[tuple[str, str, pd.Series]]:
    """
    Candidate cleaning rules as record masks per condition; none creates or flips a positive.

    Parameters
    ----------
    table : pd.DataFrame
        Analysis table.
    whole : pd.DataFrame
        ``item1_whole_record_windows.csv``.
    flags : pd.DataFrame
        Output of ``cl_flags``.
    coding : pd.DataFrame
        ``item2_separate_coding.csv``.
    cutoffs : pd.DataFrame
        ``item2_implied_cutoffs.csv``.

    Returns
    -------
    list[tuple[str, str, pd.Series]]
        ``(rule, condition, mask)``; ``rule_counts`` keeps only the defined entries of the mask.
    """
    rules = []
    later = (
        table["record_key"]
        .map(
            whole.set_index("record_key")["ectopy_first"].lt(1)
            & whole.set_index("record_key")["ectopy_later"].ge(1)
        )
        .fillna(False)
    )
    rules.append(
        (
            "R1 finding only after the first 10 s",
            "premature_ventricular_complex",
            later.astype(bool) & (table["b_premature_ventricular_complex"] == 1),
        )
    )
    af_later = (
        table["record_key"]
        .map(
            ~whole.set_index("record_key")["af_first"].astype(bool)
            & whole.set_index("record_key")["af_later_windows"].ge(1)
        )
        .fillna(False)
    )
    rules.append(
        (
            "R1 finding only after the first 10 s",
            "af_or_flutter",
            af_later.astype(bool) & (table["b_af_or_flutter"] == 1),
        )
    )
    uncoded = coding[(coding["condition"] == "complete_rbbb") & coding["not_coded_separately"]]["source"]
    short = table["source"].isin(uncoded) & (table["qrs"] < 110.0) & (table["b_complete_rbbb"] == 1)
    rules.append(
        ("R2 complete RBBB measurably narrow where completeness is not coded", "complete_rbbb", short)
    )
    rules.append(
        (
            "R2b complete RBBB where completeness is not coded (whole source)",
            "complete_rbbb",
            table["source"].isin(uncoded),
        )
    )
    for condition in ("sinus_bradycardia", "sinus_tachycardia"):
        rules.append(
            (
                f"R3 {condition.replace('_', ' ')} negative meeting 60/100 per minute in a source whose "
                "cut-off differs",
                condition,
                cutoff_mask(table, condition, cutoffs),
            )
        )
    for first, second, _ in lq.CONTRADICTIONS:
        both = (table[first] == 1) & (table[second] == 1)
        for column in (first, second):
            if column.startswith("b_"):
                rules.append((f"R4 contradictory pair {first[2:]} + {second[2:]}", column[2:], both))
    paced = table["c_paced_rhythm"] == 1
    for column in lq.UNREADABLE:
        rules.append(("R8 paced rhythm with a conduction or repolarization label", column[2:], paced))
    for condition, leads in lq.REQUIRED_LEADS.items():
        rules.append(
            (
                "R5 a lead the condition is read on is flat",
                condition,
                table[[f"flat_{lead}" for lead in leads]].any(axis=1),
            )
        )
    for condition in MEASURED_DEFINITION:
        masks = lq.criterion(table, condition)
        label = table[f"b_{condition}"]
        positive = (label == 1) & masks["outside_tol"]
        negative = (
            (label == 0) & masks["inside_tol"] & lq.population(table, lq.CHECKS[condition]["population"])
        )
        rules.append(
            ("R6 positive contradicted by the consensus measurement beyond tolerance", condition, positive)
        )
        rules.append(("R6 negative meeting the measured definition beyond tolerance", condition, negative))
        rules.append(
            (
                "R7 as R6 for positives, and flagged by confident learning",
                condition,
                positive & flags[f"fp_{condition}"],
            )
        )
        rules.append(
            (
                "R7 as R6 for negatives, and flagged by confident learning",
                condition,
                negative & flags[f"fn_{condition}"],
            )
        )
    return rules


def cutoff_mask(table: pd.DataFrame, condition: str, cutoffs: pd.DataFrame) -> pd.Series:
    """
    Negatives of a rate label that meet the standard cut-off in sources whose implied cut-off differs.

    A source qualifies when its implied cut-off's 95% interval lies more than 5 per minute from the standard
    one (60 for bradycardia, 100 for tachycardia); its negatives with sinus evidence and a consensus rate
    beyond the standard cut-off are selected.

    Parameters
    ----------
    table : pd.DataFrame
        Analysis table.
    condition : str
        ``sinus_bradycardia`` or ``sinus_tachycardia``.
    cutoffs : pd.DataFrame
        ``item2_implied_cutoffs.csv``.

    Returns
    -------
    pd.Series
        Selected records.
    """
    standard = 60.0 if condition == "sinus_bradycardia" else 100.0
    rows = cutoffs[cutoffs["condition"] == condition]
    shifted = rows[(rows["ci_high"] < standard - 5) | (rows["ci_low"] > standard + 5)]["source"]
    beyond = table["hr"] < standard if condition == "sinus_bradycardia" else table["hr"] > standard
    return (
        table["source"].isin(shifted)
        & lq.population(table, "sinus")
        & beyond
        & (table[f"b_{condition}"] == 0)
    )


def rules(workers: int) -> None:
    """
    Count the candidate cleaning rules on train and validation (step ``rules``).

    Parameters
    ----------
    workers : int
        Unused.
    """
    started = time.time()
    table = pd.read_parquet(lq.OUTPUT / "analysis.parquet")
    whole = pd.read_csv(lq.OUTPUT / "item1_whole_record_windows.csv", dtype={"record_key": str})
    coding = pd.read_csv(lq.OUTPUT / "item2_separate_coding.csv")
    cutoffs = pd.read_csv(lq.OUTPUT / "item2_implied_cutoffs.csv")
    rows = [
        row
        for rule, condition, mask in candidate_rules(table, whole, cl_flags(table), coding, cutoffs)
        for row in lq.rule_counts(table, rule, condition, mask)
    ]
    counts = pd.DataFrame(rows)
    counts = counts[(counts["positives_to_undefined"] + counts["negatives_to_undefined"]) > 0]
    write_csv(counts, "rules_candidate_counts.csv")
    receipt("rules", started, {"rows": len(counts)})


JUDGEMENTS = ("label error", "model error", "cannot tell")


def tally(workers: int) -> None:
    """
    Count the review judgements recorded by hand in ``review_judgements.csv`` (step ``tally``).

    Each row of that file is one sheet read by eye: condition, kind, source, rank, record key, one of
    ``JUDGEMENTS`` and the measurable reason. The judgements describe measurable features only; they are not
    clinical readings.

    Parameters
    ----------
    workers : int
        Unused.

    Raises
    ------
    ValueError
        If a judgement is not one of ``JUDGEMENTS`` or a row names a sheet that does not exist.
    """
    started = time.time()
    judgements = pd.read_csv(lq.REVIEW / "review_judgements.csv", dtype={"record_key": str})
    if not judgements["judgement"].isin(JUDGEMENTS).all():
        raise ValueError("unknown judgement")
    for _, row in judgements.iterrows():
        kind = "fp" if row["kind"] == "false_positive" else "fn"
        sheet = (
            lq.REVIEW
            / row["condition"]
            / row["source"]
            / f"{kind}_{int(row['rank']):02d}_{row['record_key'].replace(':', '_')}.png"
        )
        if not sheet.exists():
            raise ValueError(f"no sheet {sheet}")
    counts = (
        pd.crosstab([judgements["condition"], judgements["kind"]], judgements["judgement"])
        .reindex(columns=list(JUDGEMENTS), fill_value=0)
        .reset_index()
    )
    write_csv(counts, "item4_review_tally.csv")
    receipt(
        "tally",
        started,
        {
            "sheets_read": len(judgements),
            "judgements_sha256": sha256_file(lq.REVIEW / "review_judgements.csv"),
        },
    )


def main(argv: list[str] | None = None) -> None:
    """
    Command-line entry point.

    Parameters
    ----------
    argv : list[str] | None
        Arguments; ``sys.argv[1:]`` when None.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("step", choices=sorted(STEPS))
    parser.add_argument("--workers", type=int, default=7)
    arguments = parser.parse_args(argv)
    lq.OUTPUT.mkdir(parents=True, exist_ok=True)
    STEPS[arguments.step](arguments.workers)


STEPS = {
    "calibrate": calibrate,
    "measure": measure,
    "validate": validate,
    "windows": windows,
    "labels": labels,
    "oof": oof,
    "suspects": suspects_step,
    "sheets": sheets,
    "tally": tally,
    "rules": rules,
}


if __name__ == "__main__":
    main()
