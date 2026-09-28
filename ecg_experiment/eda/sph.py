"""Load SPH metadata and signals and compute the cached per-record analyses.

SPH (Shandong Provincial Hospital, Liu et al., Scientific Data 2022) has 25,770
twelve-lead ECGs of 24,666 patients at 500 Hz, 10 to 60 s long, stored as
float16 mV in one HDF5 file per record. Cardiologists labeled them with AHA
statement codes, where ``+`` adds a modifier such as "frequent" or "old".
"""

from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd

from ecg_experiment.ecg_quality import EXCLUSION_REASONS
from ecg_experiment.eda.ptbxl import OUTPUT_DIR, ROOT
from ecg_experiment.eda.signals import compute_features, record_summary
from ecg_experiment.sph import SINUS_VARIANTS, SUPERCLASSES, assess_window, base_codes, labels

SPH_DIR = ROOT / "data/raw/sph/sph"
RECORDS_DIR = SPH_DIR / "17912444/records"
SAMPLING_RATE = 500
FEATURE_CACHE = OUTPUT_DIR / "features" / "sph_500hz.parquet"
QUALITY_CACHE = OUTPUT_DIR / "features" / "sph_quality.parquet"


def load_codes() -> pd.DataFrame:
    """
    Load the AHA code dictionary shipped with SPH.

    Returns
    -------
    pd.DataFrame
        Indexed by the code as a string, with ``Category`` and ``Description``.
    """
    return pd.read_csv(SPH_DIR / "17912507/code.csv", dtype=str).set_index("Code")


def load_metadata() -> pd.DataFrame:
    """
    Load the record table with parsed codes and the Experiment 022 labels.

    Returns
    -------
    pd.DataFrame
        One row per ECG indexed by ``ecg_id``, with ``patient_id``, ``age``,
        ``sex``, ``duration_s``, ``date``, ``codes`` (base codes), ``modifiers``,
        ``primary`` and ``secondary`` labels, one flag per superclass,
        ``sinus_variant_only`` and ``ecgs_of_patient``.
    """
    meta = pd.read_csv(SPH_DIR / "17912441/metadata.csv", dtype={"AHA_Code": str})
    meta = meta.rename(columns={"ECG_ID": "ecg_id", "Patient_ID": "patient_id", "Age": "age", "Sex": "sex"})
    meta = meta.set_index("ecg_id")
    meta["duration_s"] = meta["N"] / SAMPLING_RATE
    meta["date"] = pd.to_datetime(meta["Date"])
    meta["codes"] = meta["AHA_Code"].apply(lambda text: sorted(base_codes(text)))
    meta["modifiers"] = meta["AHA_Code"].apply(
        lambda text: sorted({part for code in text.split(";") for part in code.split("+")[1:]}))
    mapped = pd.DataFrame([labels(set(codes)) for codes in meta["codes"]], index=meta.index)
    meta = meta.join(mapped)
    meta["sinus_variant_only"] = meta["codes"].apply(lambda codes: set(codes) <= SINUS_VARIANTS)
    meta["ecgs_of_patient"] = meta.groupby("patient_id")["N"].transform("size")
    return meta.drop(columns=["N", "Date"])


def code_counts(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Count records per base code, with its description and label group.

    Parameters
    ----------
    meta : pd.DataFrame
        Output of ``load_metadata``.

    Returns
    -------
    pd.DataFrame
        One row per code, sorted by count.
    """
    codes = load_codes()
    groups = {code: name for name, members in SUPERCLASSES.items() for code in members} | {"1": "normal"}
    counts = meta["codes"].explode().value_counts().rename("records").to_frame()
    counts["description"] = codes["Description"].reindex(counts.index)
    counts["category"] = codes["Category"].reindex(counts.index)
    counts["label_group"] = [groups.get(code, "ignored") for code in counts.index]
    counts["share"] = counts["records"] / len(meta)
    return counts


def read_signal(ecg_id: str) -> tuple[np.ndarray, int]:
    """
    Read one full SPH record in mV and canonical lead order.

    Parameters
    ----------
    ecg_id : str
        Record identifier, such as ``"A00001"``.

    Returns
    -------
    tuple[np.ndarray, int]
        Array of shape ``(samples, 12)`` and the sampling rate.
    """
    with h5py.File(RECORDS_DIR / f"{ecg_id}.h5", "r") as handle:
        signal = handle["ecg"][:].astype(np.float64).T
    return signal, SAMPLING_RATE


def sph_summary(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Per-record signal summary of every full-length record, joined with metadata.

    Parameters
    ----------
    meta : pd.DataFrame
        Output of ``load_metadata``.

    Returns
    -------
    pd.DataFrame
        ``record_summary`` columns plus metadata, indexed by ``ecg_id``.
    """
    items = [(ecg_id, ecg_id) for ecg_id in meta.index]
    summary = record_summary(compute_features(items, read_signal, FEATURE_CACHE))
    return summary.join(meta.drop(columns="duration_s"))


def window_quality(meta: pd.DataFrame, workers: int = 6) -> pd.DataFrame:
    """
    Apply the project quality policy to the first 10 s of every record, or load the cache.

    Parameters
    ----------
    meta : pd.DataFrame
        Output of ``load_metadata``.
    workers : int
        Number of processes.

    Returns
    -------
    pd.DataFrame
        One boolean column per exclusion reason and review flag, ``excluded``,
        and the SHA-256 of the window, indexed by ``ecg_id``.
    """
    if QUALITY_CACHE.exists():
        return pd.read_parquet(QUALITY_CACHE)
    with Pool(workers) as pool:
        rows = pool.map(assess_window, list(meta.index), chunksize=64)
    table = pd.DataFrame(rows).set_index("ecg_id")
    table["excluded"] = table[list(EXCLUSION_REASONS)].any(axis=1)
    QUALITY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(QUALITY_CACHE)
    return table
