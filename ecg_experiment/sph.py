"""SPH (Shandong Provincial Hospital) records and the Experiment 022 label mapping."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .ecg_quality import EXCLUSION_REASONS, REVIEW_FLAGS, assess

ROOT = Path(__file__).resolve().parents[1]
SPH = ROOT / "data/raw/sph"
EXTRACTED = SPH / "sph"
RECORDS = EXTRACTED / "17912444/records"
METADATA = EXTRACTED / "17912441/metadata.csv"
WINDOW = 5000
NORMAL = "1"
SUPERCLASSES = {
    "MI": {"160", "161", "165", "166"},
    "STTC": {"145", "146", "147", "148", "153"},
    "CD": {"82", "83", "84", "87", "88", "101", "102", "104", "105", "106", "108"},
    "HYP": {"140", "142", "143"},
}
SINUS_VARIANTS = {"21", "22", "23"}


def verify_small_files() -> dict[str, str]:
    """
    Check the extracted metadata and code tables against the download manifest MD5.

    Returns
    -------
    dict[str, str]
        MD5 of every verified file keyed by its manifest name.

    Raises
    ------
    ValueError
        If a file differs from the manifest.
    """
    manifest = json.loads((EXTRACTED / "download_manifest.json").read_text())
    verified = {}
    for item in manifest["files"]:
        if item["name"].endswith(".tar.gz"):
            continue
        digest = hashlib.md5((EXTRACTED / item["name"]).read_bytes()).hexdigest()
        if digest != item["md5"]:
            raise ValueError(f"SPH file differs from the manifest: {item['name']}")
        verified[item["name"]] = digest
    return verified


def base_codes(text: str) -> set[str]:
    """
    AHA statement codes of one record without their ``+`` modifiers.

    Parameters
    ----------
    text : str
        The ``AHA_Code`` field, such as ``"22;145+362"``.

    Returns
    -------
    set[str]
        Base codes, such as ``{"22", "145"}``.
    """
    return {code.split("+")[0] for code in text.split(";")}


def labels(codes: set[str]) -> dict[str, float]:
    """
    Primary and secondary labels and superclass membership of one record.

    Parameters
    ----------
    codes : set[str]
        Output of ``base_codes``.

    Returns
    -------
    dict[str, float]
        ``primary`` and ``secondary`` (1, 0 or NaN) and a 0/1 flag per superclass.
    """
    flags = {name: float(bool(codes & members)) for name, members in SUPERCLASSES.items()}
    positive = any(flags.values())
    primary = 1.0 if positive else 0.0 if codes == {NORMAL} else np.nan
    secondary = primary
    if not positive and codes <= SINUS_VARIANTS:
        secondary = 0.0
    return {"primary": primary, "secondary": secondary, **flags}


def table() -> pd.DataFrame:
    """
    One row per SPH record with patient, demographics and labels.

    Returns
    -------
    pd.DataFrame
        Indexed by ``ECG_ID``, with ``patient_id``, ``age``, ``male``, ``primary``, ``secondary`` and the
        superclass flags.
    """
    meta = pd.read_csv(METADATA, dtype={"ECG_ID": str, "AHA_Code": str, "Patient_ID": str})
    mapped = pd.DataFrame([labels(base_codes(text)) for text in meta["AHA_Code"]])
    frame = pd.DataFrame({"patient_id": meta["Patient_ID"].to_numpy(),
                          "age": meta["Age"].to_numpy(dtype=np.float64),
                          "male": (meta["Sex"] == "M").to_numpy(dtype=np.float64),
                          "samples": meta["N"].to_numpy()}, index=pd.Index(meta["ECG_ID"], name="ecg_id"))
    return frame.join(mapped.set_index(frame.index))


def read_window(ecg_id: str) -> np.ndarray:
    """
    Read the first 10 s of one record at 500 Hz in mV.

    Parameters
    ----------
    ecg_id : str
        Record identifier, such as ``"A00001"``.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(12, 5000)``.

    Raises
    ------
    ValueError
        If the record is shorter than 10 s or not finite.
    """
    with h5py.File(RECORDS / f"{ecg_id}.h5", "r") as handle:
        signal = handle["ecg"][:, :WINDOW].astype(np.float32)
    if signal.shape != (12, WINDOW) or not np.isfinite(signal).all():
        raise ValueError(f"Invalid SPH record {ecg_id}: {signal.shape}")
    return signal


def assess_window(ecg_id: str) -> dict[str, object]:
    """
    Apply the project quality policy to the first 10 s of one record and hash the window.

    Parameters
    ----------
    ecg_id : str
        Record identifier.

    Returns
    -------
    dict[str, object]
        ``ecg_id``, ``signal_sha256`` of the float32 window, and one boolean per exclusion reason and
        review flag.
    """
    window = read_window(ecg_id)
    reasons, flags = assess(window)
    return {"ecg_id": ecg_id, "signal_sha256": hashlib.sha256(window.tobytes()).hexdigest(),
            **{name: name in reasons for name in EXCLUSION_REASONS},
            **{name: name in flags for name in REVIEW_FLAGS}}


def records_sha256(ecg_ids: list[str]) -> str:
    """
    One SHA-256 over every record file, in the given order.

    Parameters
    ----------
    ecg_ids : list[str]
        Record identifiers.

    Returns
    -------
    str
        Hex digest.
    """
    digest = hashlib.sha256()
    for ecg_id in ecg_ids:
        digest.update(ecg_id.encode())
        digest.update((RECORDS / f"{ecg_id}.h5").read_bytes())
    return digest.hexdigest()


def einthoven_residual(signal: np.ndarray) -> float:
    """
    Largest violation of III = II - I and aVR = -(I + II) / 2, in mV.

    Parameters
    ----------
    signal : np.ndarray
        Array of shape ``(12, samples)`` in canonical lead order.

    Returns
    -------
    float
        Maximum absolute residual.
    """
    lead_i, lead_ii, lead_iii, avr = signal[:4].astype(np.float64)
    return float(max(np.abs(lead_iii - (lead_ii - lead_i)).max(), np.abs(avr + (lead_i + lead_ii) / 2).max()))


def duplicate_status(signal_sha256: pd.Series, codes: pd.Series) -> pd.Series:
    """
    Resolve records whose 10 s windows are bit-identical.

    In each identical group, the lowest record ID is kept when every copy has the same codes. When the
    copies disagree, the label is ambiguous and every copy is dropped.

    Parameters
    ----------
    signal_sha256 : pd.Series
        Window hash per record, indexed by record ID.
    codes : pd.Series
        Raw ``AHA_Code`` text per record, on the same index.

    Returns
    -------
    pd.Series
        ``unique``, ``kept``, ``dropped_copy`` or ``dropped_conflict`` per record.
    """
    frame = pd.DataFrame({"sha": signal_sha256, "codes": codes}).sort_index()
    grouped = frame.groupby("sha")
    size = grouped["codes"].transform("size")
    conflict = grouped["codes"].transform("nunique") > 1
    first = ~frame["sha"].duplicated(keep="first")
    status = pd.Series("unique", index=frame.index)
    status[(size > 1) & first & ~conflict] = "kept"
    status[(size > 1) & ~first & ~conflict] = "dropped_copy"
    status[(size > 1) & conflict] = "dropped_conflict"
    return status.reindex(signal_sha256.index)
