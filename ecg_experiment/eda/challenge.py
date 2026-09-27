"""Load the PhysioNet Challenge sources: Georgia, CPSC 2018, CPSC-Extra and Chapman/Shaoxing."""

import hashlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from ecg_experiment.eda.ptbxl import OUTPUT_DIR, ROOT
from ecg_experiment.eda.signals import canonical_order, compute_features, record_summary

RAW_ROOTS = {
    "georgia": ROOT / "data/raw/challenge-2020/1.0.2",
    "cpsc_2018": ROOT / "data/raw/challenge-2020/1.0.2",
    "cpsc_2018_extra": ROOT / "data/raw/challenge-2020/1.0.2",
    "chapman_shaoxing": ROOT / "data/raw/challenge-2021/1.0.3",
}
MAPPING_PATH = ROOT / "third_party/ecg-fm-benchmarking/Label mappings 2021.xlsx"
SINUS_RHYTHM = "426783006"
FEATURE_CACHE = OUTPUT_DIR / "features" / "challenge.parquet"
HEADER_CACHE = OUTPUT_DIR / "features" / "challenge_headers.parquet"
HASH_CACHE = OUTPUT_DIR / "features" / "challenge_hashes.parquet"


def parse_header(path: Path) -> dict[str, object]:
    """
    Parse one Challenge WFDB header without reading the signal.

    Parameters
    ----------
    path : Path
        ``.hea`` file.

    Returns
    -------
    dict[str, object]
        Sampling rate, length, lead names, units, gains and the ``#`` comment
        fields (age, sex, diagnosis codes).
    """
    lines = path.read_text().splitlines()
    name, leads, fs, samples = lines[0].split()[:4]
    signal_lines = [line.split() for line in lines[1:1 + int(leads)]]
    comments = dict(line.lstrip("# ").split(": ", 1)
                    for line in lines if line.startswith("#") and ": " in line)
    gains = [line[2].split("(")[0].split("/")[0] for line in signal_lines]
    return {
        "record": name,
        "fs": int(float(fs)),
        "samples": int(samples),
        "lead_names": ",".join(line[-1] for line in signal_lines),
        "units": ",".join(sorted({line[2].split("/")[-1] for line in signal_lines})),
        "gain": ",".join(sorted(set(gains))),
        "age": comments.get("Age", ""),
        "sex": comments.get("Sex", ""),
        "dx": comments.get("Dx", ""),
    }


def load_headers() -> pd.DataFrame:
    """
    Parse every header of the four sources, caching the result.

    Returns
    -------
    pd.DataFrame
        One row per record indexed by ``source:record``, with raw strings for
        age and sex plus parsed ``age_years``, ``duration_s`` and ``dx_codes``.
    """
    if HEADER_CACHE.exists():
        table = pd.read_parquet(HEADER_CACHE)
        table["dx_codes"] = table["dx_codes"].apply(list)
        return table
    rows = []
    for source, root in RAW_ROOTS.items():
        for header in sorted((root / "training" / source).rglob("*.hea")):
            row = parse_header(header)
            row["source"] = source
            row["path"] = str(header.with_suffix("").relative_to(root))
            rows.append(row)
    table = pd.DataFrame(rows)
    table.index = table["source"] + ":" + table["record"]
    table["age_years"] = pd.to_numeric(table["age"], errors="coerce")
    table["duration_s"] = table["samples"] / table["fs"]
    table["dx_codes"] = table["dx"].str.split(",").apply(
        lambda codes: [code.strip() for code in codes if code])
    HEADER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(HEADER_CACHE)
    return table


def snomed_names() -> dict[str, str]:
    """
    Map SNOMED codes to readable diagnosis names using the benchmark mapping sheets.

    Returns
    -------
    dict[str, str]
        Diagnosis name keyed by SNOMED code as a string. The Georgia sheet is
        preferred because its names are the most descriptive.
    """
    sheets = pd.read_excel(MAPPING_PATH, sheet_name=None)
    names = {}
    for sheet in ("Ningbo", "Chapman", "CPSC-Extra", "CPSC", "G12EC"):
        table = sheets[sheet].dropna(subset=["SNOMED code"])
        codes = table["SNOMED code"].astype("int64").astype(str)
        names |= dict(zip(codes, table["Diagnosis in the dataset"].str.strip(), strict=True))
    return names


def read_signal(path: str) -> tuple[np.ndarray, int]:
    """
    Read one Challenge record in mV and canonical lead order.

    Parameters
    ----------
    path : str
        ``source:relative/path`` as built by ``feature_items``.

    Returns
    -------
    tuple[np.ndarray, int]
        Array of shape ``(samples, 12)`` and the sampling rate.
    """
    source, relative = path.split(":", 1)
    signal, fields = wfdb.rdsamp(str(RAW_ROOTS[source] / relative))
    return canonical_order(signal, fields["sig_name"]), int(fields["fs"])


def challenge_summary(headers: pd.DataFrame) -> pd.DataFrame:
    """
    Per-record signal summary of every record, joined with header fields.

    Parameters
    ----------
    headers : pd.DataFrame
        Output of ``load_headers``.

    Returns
    -------
    pd.DataFrame
        ``record_summary`` columns plus header fields.
    """
    items = [(record_id, f"{row.source}:{row.path}") for record_id, row in headers.iterrows()]
    summary = record_summary(compute_features(items, read_signal, FEATURE_CACHE))
    return summary.join(headers.drop(columns=["samples", "fs", "duration_s"]))


def signal_hash(path: str) -> str:
    """
    Hash one record's samples, rounded to the 1 microvolt storage resolution.

    Parameters
    ----------
    path : str
        ``source:relative/path`` as accepted by ``read_signal``.

    Returns
    -------
    str
        SHA-256 hex digest of the int32 microvolt samples in canonical lead order.
    """
    signal, _ = read_signal(path)
    microvolts = np.round(np.nan_to_num(signal, nan=-99999) * 1000).astype(np.int32)
    return hashlib.sha256(microvolts.tobytes()).hexdigest()


def signal_hashes(headers: pd.DataFrame, workers: int = 6) -> pd.Series:
    """
    Signal hash of every record, cached.

    Parameters
    ----------
    headers : pd.DataFrame
        Output of ``load_headers``.
    workers : int
        Number of processes.

    Returns
    -------
    pd.Series
        Hash indexed like ``headers``.
    """
    if HASH_CACHE.exists():
        return pd.read_parquet(HASH_CACHE)["hash"]
    paths = [f"{row.source}:{row.path}" for row in headers.itertuples()]
    with Pool(workers) as pool:
        hashes = pd.Series(pool.map(signal_hash, paths, chunksize=64), index=headers.index, name="hash")
    hashes.to_frame().to_parquet(HASH_CACHE)
    return hashes
