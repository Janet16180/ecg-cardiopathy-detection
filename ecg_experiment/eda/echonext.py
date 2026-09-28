"""Load EchoNext metadata and compute cached waveform statistics straight from the PhysioNet ZIP.

EchoNext (Columbia University Irving Medical Center, PhysioNet 1.1.0) has 100,000
twelve-lead ECGs labeled with structural heart disease measured on an
echocardiogram. Waveforms are 10 s at 250 Hz and were already median filtered,
clipped at the 0.1 and 99.9 percentiles and standardized by the authors, so
their values have no physical unit. It is credentialed data: analyses report
aggregates only.

The test split stays closed: only its metadata counts are read, never its
waveforms. The ``no_split`` ECGs share patients with ``val`` and ``test``.
"""

import hashlib
import io
import zipfile
from collections.abc import Iterator
from multiprocessing import Pool

import numpy as np
import pandas as pd

from ecg_experiment.eda.ptbxl import OUTPUT_DIR, ROOT
from ecg_experiment.eda.signals import LEADS, longest_constant_run, signal_features

RELEASE = "echonext-a-dataset-for-detecting-echocardiogram-confirmed-structural-heart-disease-from-ecgs-1.1.0"
ARCHIVE = ROOT / "data/raw/echonext" / f"{RELEASE}.zip"
SAMPLING_RATE = 250
SPLITS = ["train", "val", "test", "no_split"]
OPEN_SPLITS = ["train", "val"]
CHUNK = 1024
FEATURE_DIR = OUTPUT_DIR / "features"
COMPOSITE = "shd_moderate_or_greater_flag"
COMPONENTS = {
    "lvef_lte_45_flag": "lvef_value",
    "lvwt_gte_13_flag": "ivs_measurement",
    "aortic_stenosis_moderate_or_greater_flag": "aortic_stenosis_value",
    "aortic_regurgitation_moderate_or_greater_flag": "aortic_regurgitation_value",
    "mitral_regurgitation_moderate_or_greater_flag": "mitral_regurgitation_value",
    "tricuspid_regurgitation_moderate_or_greater_flag": "tricuspid_regurgitation_value",
    "pulmonary_regurgitation_moderate_or_greater_flag": "pulmonary_regurgitation_value",
    "rv_systolic_dysfunction_moderate_or_greater_flag": "rv_systolic_function_value",
    "pericardial_effusion_moderate_large_flag": "pericardial_effusion_value",
    "pasp_gte_45_flag": "pasp_value",
    "tr_max_gte_32_flag": "tr_max_velocity_value",
}
MEASUREMENTS = ["ventricular_rate", "atrial_rate", "pr_interval", "qrs_duration", "qt_corrected"]


def member(name: str) -> str:
    """
    Full path of a file inside the release ZIP.

    Parameters
    ----------
    name : str
        File name, such as ``"README.md"``.

    Returns
    -------
    str
        Path inside the archive.
    """
    return f"{RELEASE}/{name}"


def load_metadata() -> pd.DataFrame:
    """
    Load the metadata table with each ECG's row in its split's waveform file.

    Returns
    -------
    pd.DataFrame
        One row per ECG indexed by ``ecg_key``, with every original column,
        ``row`` (position in ``EchoNext_<split>_waveforms.npy``) and
        ``ecgs_of_patient``.

    Raises
    ------
    ValueError
        If a split is not one contiguous block of the table.
    """
    with zipfile.ZipFile(ARCHIVE) as archive:
        meta = pd.read_csv(io.BytesIO(archive.read(member("echonext_metadata_100k.csv"))))
    position = meta.pop("Unnamed: 0")
    blocks = position.groupby(meta["split"]).agg(["min", "max", "size"])
    if ((blocks["max"] - blocks["min"] + 1) != blocks["size"]).any():
        raise ValueError("A split is not contiguous in the metadata table")
    meta["row"] = meta.groupby("split").cumcount()
    meta["ecgs_of_patient"] = meta.groupby("patient_key")["split"].transform("size")
    return meta.set_index("ecg_key")


def checksums() -> dict[str, str]:
    """
    Read the release's own SHA-256 list.

    Returns
    -------
    dict[str, str]
        Digest keyed by file name.
    """
    with zipfile.ZipFile(ARCHIVE) as archive:
        lines = archive.read(member("SHA256SUMS.txt")).decode().splitlines()
    return {name: digest for digest, name in (line.split() for line in lines)}


def split_overlap(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Patients shared between each pair of splits.

    Parameters
    ----------
    meta : pd.DataFrame
        Output of ``load_metadata``.

    Returns
    -------
    pd.DataFrame
        Square table of shared patient counts.
    """
    patients = meta.groupby("split")["patient_key"].apply(set)
    return pd.DataFrame([[len(patients[a] & patients[b]) for b in SPLITS] for a in SPLITS],
                        index=SPLITS, columns=SPLITS)


def label_consistency(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Compare each binary label with the echo value it is derived from.

    Parameters
    ----------
    meta : pd.DataFrame
        Output of ``load_metadata``.

    Returns
    -------
    pd.DataFrame
        Per label: positives, positives and negatives with a missing source
        value, and disagreements with the documented threshold where one is numeric.
    """
    thresholds = {
        "lvef_lte_45_flag": meta["lvef_value"] <= 45,
        "lvwt_gte_13_flag": meta[["ivs_measurement", "lvpw_measurement"]].max(axis=1) >= 1.3,
        "pasp_gte_45_flag": meta["pasp_value"] >= 45,
        "tr_max_gte_32_flag": meta["tr_max_velocity_value"] >= 3.2,
    }
    rows = []
    for flag, source in COMPONENTS.items():
        missing = meta[source].isna()
        row = {"label": flag, "positives": int(meta[flag].sum()),
               "positive_without_value": int((meta[flag].eq(1) & missing).sum()),
               "negative_without_value": int((meta[flag].eq(0) & missing).sum())}
        if flag in thresholds:
            known = ~missing
            row["disagrees_with_threshold"] = int((meta.loc[known, flag].astype(bool)
                                                   != thresholds[flag][known]).sum())
        rows.append(row)
    any_component = meta[list(COMPONENTS)].max(axis=1)
    rows.append({"label": COMPOSITE, "positives": int(meta[COMPOSITE].sum()),
                 "positive_without_any_component": int((meta[COMPOSITE].eq(1) & any_component.eq(0)).sum()),
                 "component_without_composite": int((meta[COMPOSITE].eq(0) & any_component.eq(1)).sum())})
    return pd.DataFrame(rows).set_index("label")


def _record_features(signal: np.ndarray) -> dict[str, np.ndarray]:
    features = signal_features(signal, SAMPLING_RATE)
    leads = signal.T
    at_bound = (leads == leads.max(axis=1, keepdims=True)) | (leads == leads.min(axis=1, keepdims=True))
    features["bound_fraction"] = at_bound.mean(axis=1)
    features["longest_constant"] = np.array([longest_constant_run(values) for values in leads])
    digest = hashlib.sha256(signal.tobytes()).hexdigest()
    features["signal_sha256"] = np.full(len(LEADS), digest, dtype=object)
    return features


def member_sha256(name: str) -> str:
    """
    SHA-256 of one file inside the release ZIP, read as a stream.

    Parameters
    ----------
    name : str
        File name, such as ``"EchoNext_val_waveforms.npy"``.

    Returns
    -------
    str
        Hex digest.
    """
    digest = hashlib.sha256()
    with zipfile.ZipFile(ARCHIVE) as archive, archive.open(member(name)) as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def waveform_chunks(split: str) -> Iterator[np.ndarray]:
    """
    Stream one split's waveforms from the ZIP without extracting it.

    Parameters
    ----------
    split : str
        Split name.

    Yields
    ------
    np.ndarray
        Float64 arrays of up to ``CHUNK`` records, shape ``(records, 2500, 12)``.
    """
    name = member(f"EchoNext_{split}_waveforms.npy")
    with zipfile.ZipFile(ARCHIVE) as archive, archive.open(name) as handle:
        if np.lib.format.read_magic(handle) != (1, 0):
            raise ValueError(f"Unexpected .npy format version in {name}")
        shape, _, dtype = np.lib.format.read_array_header_1_0(handle)
        record_bytes = int(np.prod(shape[1:])) * dtype.itemsize
        for start in range(0, shape[0], CHUNK):
            count = min(CHUNK, shape[0] - start)
            data = handle.read(count * record_bytes)
            yield np.frombuffer(data, dtype=dtype).reshape(count, *shape[2:])  # drops the singleton axis


def waveform_features(split: str, workers: int = 6) -> tuple[pd.DataFrame, str]:
    """
    Per-lead statistics of every waveform in an open split, or load them from cache.

    Parameters
    ----------
    split : str
        ``"train"`` or ``"val"``.
    workers : int
        Number of processes.

    Returns
    -------
    tuple[pd.DataFrame, str]
        One row per record and lead with ``row`` (position in the split), and
        the SHA-256 of the whole waveform file.

    Raises
    ------
    ValueError
        If ``split`` is closed or the file differs from the release checksum.
    """
    if split not in OPEN_SPLITS:
        raise ValueError(f"Split {split} stays closed")
    cache = FEATURE_DIR / f"echonext_{split}.parquet"
    expected = checksums()[f"EchoNext_{split}_waveforms.npy"]
    if cache.exists():
        return pd.read_parquet(cache), expected
    digest = member_sha256(f"EchoNext_{split}_waveforms.npy")
    if digest != expected:
        raise ValueError(f"EchoNext {split} waveforms differ from the release checksum")
    parts, offset = [], 0
    with Pool(workers) as pool:
        for chunk in waveform_chunks(split):
            features = pool.map(_record_features, list(chunk), chunksize=32)
            for index, item in enumerate(features):
                item["row"] = np.full(len(LEADS), offset + index)
            parts += features
            offset += len(chunk)
    table = pd.DataFrame({key: np.concatenate([part[key] for part in parts]) for key in parts[0]})
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(cache)
    return table, digest


def record_table(features: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse per-lead waveform statistics to one row per record.

    Parameters
    ----------
    features : pd.DataFrame
        Output of ``waveform_features``.

    Returns
    -------
    pd.DataFrame
        Indexed by ``row``, with limb residuals, estimated heart rate, median
        lead statistics, constant and flat lead counts and the signal hash.
    """
    features = features.assign(constant=features["longest_constant"] >= features["samples"],
                               flat=features["longest_constant"] >= SAMPLING_RATE)
    grouped = features.groupby("row", sort=True)
    first = grouped[["einthoven_residual", "avr_residual", "heart_rate", "signal_sha256"]].first()
    medians = grouped[["std", "mean", "baseline_fraction", "high_frequency_fraction", "powerline_50_fraction",
                       "powerline_60_fraction", "bound_fraction"]].median()
    counts = grouped[["constant", "flat"]].sum().rename(columns={"constant": "constant_leads",
                                                                 "flat": "flat_leads"})
    table = first.join(medians).join(counts)
    table["peak_abs"] = grouped["maximum"].max().combine(grouped["minimum"].min().abs(), max)
    return table



def _longest_run(values: np.ndarray) -> tuple[int, int, float]:
    changes = np.flatnonzero(np.r_[True, values[1:] != values[:-1]])
    lengths = np.diff(np.r_[changes, len(values)])
    best = int(lengths.argmax())
    return int(changes[best]), int(lengths[best]), float(values[changes[best]])


def flat_runs(split: str, rows: set[int]) -> pd.DataFrame:
    """
    Locate each lead's longest constant run in selected waveforms of an open split, or load the cache.

    Parameters
    ----------
    split : str
        ``"train"`` or ``"val"``.
    rows : set[int]
        Positions in the split to measure, such as the records with a flat lead.

    Returns
    -------
    pd.DataFrame
        One row per record and lead with ``row``, ``lead``, ``start`` and ``length`` in samples and the
        constant ``value``.

    Raises
    ------
    ValueError
        If ``split`` is closed.
    """
    if split not in OPEN_SPLITS:
        raise ValueError(f"Split {split} stays closed")
    cache = FEATURE_DIR / f"echonext_{split}_flat_runs.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    found, offset = [], 0
    for chunk in waveform_chunks(split):
        for row in sorted(item for item in rows if offset <= item < offset + len(chunk)):
            for index, lead in enumerate(LEADS):
                start, length, value = _longest_run(chunk[row - offset, :, index])
                found.append({"row": row, "lead": lead, "start": start, "length": length, "value": value})
        offset += len(chunk)
    table = pd.DataFrame(found)
    table.to_parquet(cache)
    return table
