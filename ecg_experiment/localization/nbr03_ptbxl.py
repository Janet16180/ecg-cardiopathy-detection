"""NBR-03 PTB-XL fold 9: data rules, SCP groups, sample, per-record preparation and historical parity."""

from __future__ import annotations

import ast
import gc
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from .. import normative_map062 as nm
from ..eda.ptbxl import diagnostic_classes, load_statements, superclass_label
from ..external_encoders import xecg_input
from ..files import read_json, sha256_file, write_npz_atomic
from ..full_development import predict
from ..gpu import gpu_lock
from ..heads01_features import lock_path
from ..locbench_data import load_partition
from ..pr74_fresh_baseline import read_signal
from ..pr74_screening_features import load_encoder, measurements
from . import nbr01_counterpart as nc
from . import nbr01_features as nf

ROWS = Path("outputs/pr74_fresh_baseline_exact_20261028/features/ptb_rows.csv")
FEATURES = Path("outputs/pr74_fresh_baseline_exact_20261028/features/ptb_xecg.npy")
FEATURE_RECEIPT = Path("outputs/pr74_fresh_baseline_exact_20261028/features/completion.json")
PREDICTIONS = Path("outputs/pr74_from_raw_validation_20261028/evaluate/ptb_predictions.npz")
COHORT = Path("outputs/cohort_age_label_priority_20261028/all_age.csv")
COHORT_RECEIPT = Path("outputs/cohort_age_label_priority_20261028/receipt.json")
DATABASE = Path("data/raw/ptb-xl/1.0.3/ptbxl_database.csv")
FOLD = "9"
RESERVED_FOLD = "10"
INFERIOR = {"IMI", "ISCIN", "INJIN"}
ANTERIOR = {"AMI", "ASMI", "ISCAN", "ISCAS", "INJAS"}
LATERAL = {"LMI", "ISCLA", "INJLA"}
MULTI_TERRITORY = {"ALMI", "ILMI", "ISCAL", "ISCIL", "IPLMI", "IPMI", "PMI", "INJAL", "INJIL"}
TERRITORY_CODES = {"inferior": INFERIOR, "anterior": ANTERIOR, "lateral": LATERAL}
STT = {"NDT", "NST_", "ISC_"}
OTHER_STTC = {"DIG", "LNGQT", "EL", "ANEUR"}
HISTORICAL_BATCH = 16
_STATE: dict[str, Any] = {}


def read_database(input_root: Path) -> pd.DataFrame:
    """Read the PTB-XL database indexed by ECG ID."""
    database = pd.read_csv(input_root / DATABASE, dtype={"ecg_id": str}, keep_default_na=False)
    return database.set_index("ecg_id")


def scp_codes(database: pd.DataFrame, ecg_id: str) -> set[str]:
    """Return the SCP statements of one PTB-XL record, whatever their likelihood."""
    return set(ast.literal_eval(database.at[str(int(float(ecg_id))), "scp_codes"]))


def group_of(codes: set[str], target: int, classes: dict[str, str]) -> str | None:
    """
    Return the single NBR-03 group of one PTB-XL ECG, or None.

    Parameters
    ----------
    codes : set[str]
        SCP statements, any likelihood (as the project's binary label reads PTB-XL).
    target : int
        Project binary label.
    classes : dict[str, str]
        Diagnostic superclass of each diagnostic statement.

    Returns
    -------
    str or None
        ``inferior``, ``anterior`` or ``lateral`` (exactly one territory, no multi-territory statement),
        ``lbbb`` (CLBBB, no MI or territorial statement), ``rbbb`` (CRBBB, no CLBBB), ``stt`` (NDT, NST_ or
        ISC_ with no MI, territorial ischaemia, other ST-T, conduction or hypertrophy statement) or ``normal``
        (NORM as the only diagnostic statement and a negative label).
    """
    if codes & MULTI_TERRITORY:
        return None
    territories = [name for name, members in TERRITORY_CODES.items() if codes & members]
    if len(territories) == 1:
        return territories[0]
    superclasses = {classes[code] for code in codes if code in classes}
    if territories or "MI" in superclasses:
        return None
    if "CLBBB" in codes:
        return "lbbb"
    if "CRBBB" in codes:
        return "rbbb"
    if codes & STT and not codes & OTHER_STTC and not superclasses & {"CD", "HYP"}:
        return "stt"
    diagnostic = {code for code in codes if code in classes}
    if diagnostic == {"NORM"} and target == 0:
        return "normal"
    return None


def fold_rows(input_root: Path) -> pd.DataFrame:
    """
    Fold-9 rows of the frozen PTB baseline table, with SCP statements and NBR-03 groups.

    Parameters
    ----------
    input_root : Path
        Checkout holding outputs and the PTB-XL database.

    Returns
    -------
    pd.DataFrame
        Fold-9 rows with ``position`` (row in the 19,229-row PTB table), ``codes`` and ``group``.

    Raises
    ------
    ValueError
        If a reserved fold-10 row is present, a fold-9 row is not in the development role, or a label
        disagrees with ``superclass_label``.
    """
    rows = pd.read_csv(input_root / ROWS, dtype=str, keep_default_na=False)
    if rows["strat_fold"].eq(RESERVED_FOLD).any():
        raise ValueError("Fold 10 is reserved for final_frozen_test")
    rows["position"] = np.arange(len(rows))
    rows = rows.loc[rows["strat_fold"].eq(FOLD)].reset_index(drop=True)
    if not rows["role"].eq("development").all() or not rows["evaluation"].eq("True").all():
        raise ValueError("Fold-9 rows must be the development evaluation rows")
    database = read_database(input_root)
    classes = diagnostic_classes(load_statements())
    rows["codes"] = [scp_codes(database, ecg_id) for ecg_id in rows["ecg_id"]]
    labels = [superclass_label(dict.fromkeys(codes, 100.0), classes) for codes in rows["codes"]]
    if [float(value) for value in rows["target"]] != labels:
        raise ValueError("Fold-9 targets disagree with the project's binary label")
    pairs = zip(rows["codes"], rows["target"], strict=True)
    rows["group"] = [group_of(codes, int(target), classes) for codes, target in pairs]
    return rows


def data_rules(input_root: Path, rows: pd.DataFrame) -> dict[str, Any]:
    """
    Verify from receipts and identities that fold 9 is unseen and patient-disjoint from every fit set.

    Parameters
    ----------
    input_root : Path
        Checkout holding outputs.
    rows : pd.DataFrame
        Output of ``fold_rows``.

    Returns
    -------
    dict[str, Any]
        Every check and ``passed``.
    """
    cohort_receipt = read_json(input_root / COHORT_RECEIPT)
    feature_receipt = read_json(input_root / FEATURE_RECEIPT)
    table = pd.read_csv(input_root / ROWS, dtype=str, keep_default_na=False)
    cohort = pd.read_csv(input_root / COHORT, dtype=str, keep_default_na=False)
    database = read_database(input_root)
    partition = load_partition()
    locbench_fit = partition.loc[(partition.partition == "fit") & partition.clean_normal, "ecg_id"]
    patients = set(rows["patient_id"].map(lambda value: str(int(float(value)))))

    def patient_set(ecg_ids: pd.Series) -> set[str]:
        return {
            str(int(float(value)))
            for value in database.loc[ecg_ids.map(lambda v: str(int(float(v)))), "patient_id"]
        }

    ptb_cohort = cohort.loc[cohort["source"].eq("ptbxl")]
    fitted = table.loc[table["fit"].eq("True")]
    calibration = table.loc[table["role"].eq("calibration")]
    result = {
        "all_age_csv_matches_receipt": (
            sha256_file(input_root / COHORT) == cohort_receipt["artifacts"][str(COHORT)]
        ),
        "ptb_rows_match_receipt": sha256_file(input_root / ROWS) == feature_receipt["artifacts"][str(ROWS)],
        "all_age_ptb_folds": sorted(ptb_cohort["strat_fold"].unique().tolist()),
        "ptb_baseline_fit_folds": sorted(fitted["strat_fold"].unique().tolist()),
        "ptb_threshold_calibration_folds": sorted(calibration["strat_fold"].unique().tolist()),
        "fold10_rows_in_table": int(table["strat_fold"].eq(RESERVED_FOLD).sum()),
        "fold9_rows": len(rows),
        "fold9_patients": len(patients),
        "patients_in_all_age_training": len(patients & patient_set(ptb_cohort["record"])),
        "patients_in_ptb_baseline_fit": len(patients & patient_set(fitted["ecg_id"])),
        "patients_in_ptb_calibration": len(patients & patient_set(calibration["ecg_id"])),
        "patients_in_locbench_fit": len(patients & patient_set(locbench_fit.astype(str))),
    }
    zero = (
        "fold10_rows_in_table",
        "patients_in_all_age_training",
        "patients_in_ptb_baseline_fit",
        "patients_in_ptb_calibration",
        "patients_in_locbench_fit",
    )
    result["passed"] = (
        all(result[key] == 0 for key in zero)
        and result["all_age_csv_matches_receipt"]
        and result["ptb_rows_match_receipt"]
        and set(result["all_age_ptb_folds"]) <= set("1234567")
        and set(result["ptb_baseline_fit_folds"]) <= set("1234567")
        and result["ptb_threshold_calibration_folds"] == ["8"]
    )
    return result


def covariates(row: pd.Series) -> tuple[float, float]:
    """Age (NaN above 89, as LocBench and 062 read PTB-XL) and PTB-XL sex (0 male, 1 female) of one row."""
    age = float(row["age"])
    return (float("nan") if age > 89 else age), float(row["sex"])


def prepare_record(input_root: Path, row: pd.Series, model: nm.PopulationModel) -> dict[str, Any] | None:
    """NBR-01's ``prepare_record`` for a PTB-XL row: plan, energies, measurements and variant digest."""
    if row["strat_fold"] != FOLD:
        raise ValueError("Only fold 9 may be read")
    raw = read_signal(input_root, row)
    age, sex = covariates(row)
    plan = nc.record_plan(raw, age, sex, model)
    if plan is None:
        return None
    signals, energy = nc.variant_signals(raw, plan, nc.variants())
    return {
        **{key: plan[key] for key in nf.PLAN_KEYS},
        "energy": energy,
        "measurements": measurements(list(signals)).to_numpy(),
        "digest": nf.signal_digest(signals),
    }


def _prepare_job(position: int) -> tuple[str, str, float]:
    torch.set_num_threads(1)
    row = _STATE["rows"].iloc[position]
    path = _STATE["work"] / "plans" / f"{row['ecg_id']}.npz"
    if path.exists():
        return row["ecg_id"], "done", 0.0
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        prepared = prepare_record(_STATE["input_root"], row, _STATE["model"])
    if prepared is None:
        write_npz_atomic(path, status=np.array("excluded"))
        return row["ecg_id"], "excluded", time.perf_counter() - started
    write_npz_atomic(path, status=np.array("prepared"), **prepared)
    return row["ecg_id"], "prepared", time.perf_counter() - started


def prepare_all(
    input_root: Path, rows: pd.DataFrame, model: nm.PopulationModel, work: Path, workers: int
) -> list[tuple[str, str, float]]:
    """Prepare every row in forked single-thread workers, skipping records already written."""
    (work / "plans").mkdir(parents=True, exist_ok=True)
    _STATE.update(rows=rows.reset_index(drop=True), work=work, input_root=input_root, model=model)
    with get_context("fork").Pool(workers) as pool:
        return pool.map(_prepare_job, range(len(rows)), chunksize=1)


def prepared(path: Path) -> bool:
    """Return whether a written plan holds a prepared record rather than an exclusion."""
    with np.load(path) as data:
        return str(data["status"]) == "prepared"


def thresholds(input_root: Path) -> dict[str, float]:
    """Frozen PTB thresholds (5% budget on fold-8 calibration normals) of every component and blend."""
    bundle = joblib.load(input_root / nf.BUNDLE)
    return {
        f"{name}_{kind}": float(bundle["thresholds"][f"{name}_{kind}"]["ptb"]["threshold"])
        for name in nf.MODELS
        for kind in nf.KINDS
    }


def historical_parity(input_root: Path, rows: pd.DataFrame, work: Path, device: int) -> dict[str, Any]:
    """
    Re-encode records whose unreplaced features are not bit-identical inside their historical PTB batch.

    The saved PTB features were extracted in consecutive batches of sixteen rows of the frozen PTB table;
    the final batch of 13 rows was padded to sixteen with its last waveform
    (``pr74-batch-shape-reproduction-2026-10-28.md``).

    Parameters
    ----------
    input_root : Path
        Checkout holding inputs.
    rows : pd.DataFrame
        Analysed rows with ``position``.
    work : Path
        Local work directory with ``pooled``.
    device : int
        CUDA device.

    Returns
    -------
    dict[str, Any]
        Differing records and whether each reproduces in its historical batch.
    """
    saved = np.load(input_root / FEATURES, mmap_mode="r")
    table = pd.read_csv(input_root / ROWS, dtype=str, keep_default_na=False)
    gaps = np.array(
        [
            float(
                np.max(
                    np.abs(
                        np.load(work / "pooled" / f"{ecg_id}.npy")[0].astype(np.float64)
                        - saved[int(position)]
                    )
                )
            )
            for ecg_id, position in zip(rows["ecg_id"], rows["position"], strict=True)
        ]
    )
    differing = rows.loc[gaps > 0].assign(gap=gaps[gaps > 0])
    predictions = np.load(input_root / PREDICTIONS)
    index = {str(value): position for position, value in enumerate(predictions["ecg_id"])}
    bundle = joblib.load(input_root / nf.BUNDLE)
    records = []
    if len(differing):
        with gpu_lock("cuda", blocking=True, path=lock_path(device)), torch.cuda.device(device):
            torch.set_num_threads(2)
            model = load_encoder(input_root)
            for _, row in differing.iterrows():
                position = int(row["position"])
                start = HISTORICAL_BATCH * (position // HISTORICAL_BATCH)
                batch = table.iloc[start : start + HISTORICAL_BATCH]
                inputs = np.stack([xecg_input(read_signal(input_root, item)) for _, item in batch.iterrows()])
                vector = nf.encode(model, inputs)[position - start]
                score_gap = {
                    name: float(
                        abs(
                            predict(bundle["fitted"][name]["xecg"], vector[None])[0]
                            - predictions[f"{name}_xecg"][index[row["ecg_id"]]]
                        )
                    )
                    for name in nf.MODELS
                }
                identical = bool(np.array_equal(vector, saved[position]))
                records.append(
                    {
                        "ecg_id": row["ecg_id"],
                        "position": position,
                        "feature_gap": float(row["gap"]),
                        "historical_batch_bit_identical": identical,
                        "historical_batch_xecg_score_gap": score_gap,
                    }
                )
            del model
            gc.collect()
            torch.cuda.empty_cache()
    passed = all(item["historical_batch_bit_identical"] for item in records) and all(
        value <= 1e-12 for item in records for value in item["historical_batch_xecg_score_gap"].values()
    )
    return {
        "records_checked": len(rows),
        "bit_identical_records": int((gaps == 0).sum()),
        "differing": records,
        "passed": passed,
    }
