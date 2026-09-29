"""Map PhysioNet Challenge 2021 SNOMED codes to the project's binary ECG endpoint.

The endpoint is the one used for PTB-XL and SPH: negative for a normal ECG, positive when any myocardial
infarction (MI), ST/T change (STTC), conduction disorder (CD) or hypertrophy (HYP) statement is present, and
undefined otherwise. The Challenge has no "normal ECG" code: 426783006 is sinus rhythm, a rhythm statement.
The primary negative is therefore sinus rhythm as the only code. The secondary negative also allows the
benign sinus variants (bradycardia, tachycardia, arrhythmia), like SPH's secondary label.

Code names and the four scoring equivalences come from the official ``dx_mapping_scored.csv`` and
``dx_mapping_unscored.csv`` of github.com/physionetchallenges/evaluation-2021, checked by SHA-256. Codes of
neither table are ``unknown`` and, like every code outside the four superclasses and the sinus codes, leave a
record undefined unless a superclass code makes it positive.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAPPING_DIR = ROOT / "data/raw/challenge-2021/evaluation-2021"
MAPPING_SHA256 = {
    "dx_mapping_scored.csv": "fad13ad9f7ca230e7e6392ac8a264cb7cd157879525129f964c5f708eabb41d0",
    "dx_mapping_unscored.csv": "ce53c9e35406e922f1634a38bb13c31e8cf7603291641a16c201d8e679f4fe94",
}
SINUS_RHYTHM = "426783006"
SINUS_VARIANTS = {"426177001", "427084000", "427393009"}
SUPERCLASSES = {
    "MI": {"164865005", "164867002", "54329005", "57054005"},
    "STTC": {"164934002", "59931005", "429622005", "164931005", "164930006", "55930002", "428750005",
             "111975006", "164861001", "413844008", "413444003", "425623009", "425419005", "426434006",
             "704997005", "370365005"},
    "CD": {"270492004", "164947007", "195042002", "54016002", "426183003", "27885002", "233917008",
           "445118002", "445211001", "164909002", "733534002", "251120003", "59118001", "713427006",
           "713426002", "6374002", "698252002", "82226007", "74390002", "195060002"},
    "HYP": {"164873001", "89792004", "266249003", "67741000119109", "253352002", "446813000", "446358003",
            "253339007", "195126007"},
}


def load_official(directory: Path = MAPPING_DIR) -> pd.DataFrame:
    """
    Read the official scored and unscored code tables after checking their hashes.

    Parameters
    ----------
    directory : Path
        Folder holding the two CSV files.

    Returns
    -------
    pd.DataFrame
        One row per SNOMED code (as text) with ``name``, ``abbreviation``, ``scored`` and ``notes``.

    Raises
    ------
    ValueError
        If a file differs from the recorded SHA-256 or a code appears in both tables.
    """
    parts = []
    for name, expected in MAPPING_SHA256.items():
        path = directory / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Official mapping changed: {path}")
        table = pd.read_csv(path, dtype={"SNOMEDCTCode": str})
        notes = table["Notes"] if "Notes" in table else pd.Series("", index=table.index)
        parts.append(pd.DataFrame({
            "code": table["SNOMEDCTCode"], "name": table["Dx"].str.strip(),
            "abbreviation": table["Abbreviation"], "scored": name == "dx_mapping_scored.csv",
            "notes": notes.fillna(""),
        }))
    official = pd.concat(parts, ignore_index=True)
    if official["code"].duplicated().any():
        raise ValueError("A code is listed as both scored and unscored")
    return official.set_index("code")


def equivalences(official: pd.DataFrame) -> dict[str, str]:
    """
    Scoring equivalences stated in the official notes, as code to canonical code.

    A note reads "We score 733534002 and 164909002 as the same diagnosis"; the first code is canonical.

    Parameters
    ----------
    official : pd.DataFrame
        Output of ``load_official``.

    Returns
    -------
    dict[str, str]
        Every code of an equivalence pair, including the canonical one, mapped to the canonical code.
    """
    mapping = {}
    for note in official["notes"]:
        found = re.search(r"score (\d+) and (\d+) as the same", note)
        if found:
            first, second = found.groups()
            mapping[first] = first
            mapping[second] = first
    return mapping


def code_group(code: str, official: pd.DataFrame) -> str:
    """
    Endpoint group of one SNOMED code.

    Parameters
    ----------
    code : str
        SNOMED CT code as text.
    official : pd.DataFrame
        Output of ``load_official``.

    Returns
    -------
    str
        ``sinus_rhythm``, ``sinus_variant``, a superclass name, ``other`` for any other official code, or
        ``unknown`` for a code of neither official table.
    """
    group = "unknown"
    if code == SINUS_RHYTHM:
        group = "sinus_rhythm"
    elif code in SINUS_VARIANTS:
        group = "sinus_variant"
    elif any(code in members for members in SUPERCLASSES.values()):
        group = next(name for name, members in SUPERCLASSES.items() if code in members)
    elif code in official.index:
        group = "other"
    return group


def mapping_table(official: pd.DataFrame) -> pd.DataFrame:
    """
    Every official code with its endpoint group and canonical code.

    Parameters
    ----------
    official : pd.DataFrame
        Output of ``load_official``.

    Returns
    -------
    pd.DataFrame
        ``name``, ``abbreviation``, ``scored``, ``canonical`` and ``group``, indexed by code.
    """
    canonical = equivalences(official)
    table = official[["name", "abbreviation", "scored"]].copy()
    table["canonical"] = [canonical.get(code, code) for code in table.index]
    table["group"] = [code_group(code, official) for code in table.index]
    return table


def record_labels(codes: list[str], official: pd.DataFrame) -> dict[str, float]:
    """
    Primary and secondary labels and superclass membership of one record.

    Parameters
    ----------
    codes : list[str]
        The record's SNOMED codes.
    official : pd.DataFrame
        Output of ``load_official``.

    Returns
    -------
    dict[str, float]
        ``primary`` and ``secondary`` (1, 0 or NaN) and a 0/1 flag per superclass.
    """
    groups = {code_group(code, official) for code in codes}
    flags = {name: float(name in groups) for name in SUPERCLASSES}
    positive = any(flags.values())
    primary = 1.0 if positive else 0.0 if groups == {"sinus_rhythm"} else np.nan
    secondary = primary
    if not positive and groups and groups <= {"sinus_rhythm", "sinus_variant"}:
        secondary = 0.0
    return {"primary": primary, "secondary": secondary, **flags}


def label_table(dx_codes: pd.Series, official: pd.DataFrame) -> pd.DataFrame:
    """
    Labels of many records.

    Parameters
    ----------
    dx_codes : pd.Series
        List of SNOMED codes per record.
    official : pd.DataFrame
        Output of ``load_official``.

    Returns
    -------
    pd.DataFrame
        ``record_labels`` columns on the index of ``dx_codes``.
    """
    rows = [record_labels(list(codes), official) for codes in dx_codes]
    return pd.DataFrame(rows, index=dx_codes.index)
