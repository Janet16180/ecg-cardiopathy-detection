"""
Data for the localization benchmark (LocBench v1).

Partitions, clean normals, noise leads and the statement-to-region table.

PTB-XL folds 1-8 are split by patient into ``fit`` (two thirds) and ``evaluation`` (one third).
Methods fit only on ``fit`` records; label-free methods only on its clean normals. The frozen table
is ``docs/locbench-partition-v1.json``; see ``docs/experiment-061-locbench.md``.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ecg_experiment import ROOT
from ecg_experiment.waveforms import LEADS

PARTITION_PATH = ROOT / "docs/locbench-partition-v1.json"
PARTITION_SEED = 61062
NOISE_COLUMNS = ("baseline_drift", "static_noise", "burst_noise", "electrodes_problems")
CLEAN_RHYTHMS = frozenset({"SR", "SARRH", "SBRAD"})
UPPER_LEADS = tuple(lead.upper() for lead in LEADS)

ANTERIOR_SEPTAL = ("V1", "V2", "V3", "V4")
INFERIOR = ("II", "III", "aVF")
HIGH_LATERAL = ("I", "aVL", "V5", "V6")
ANTEROLATERAL = ("I", "aVL", "V3", "V4", "V5", "V6")
LIMB = ("I", "II", "III", "aVR", "aVL", "aVF")

# Statement -> (leads where the finding is expected, ECG components where it is expected).
# Components: P, PR (P onset to QRS onset), QRS, ST (QRS offset to T onset), T.
REGION_TABLE: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "IMI": (INFERIOR, ("QRS",)),
    "ASMI": (("V1", "V2", "V3"), ("QRS",)),
    "AMI": (ANTERIOR_SEPTAL, ("QRS",)),
    "ALMI": (ANTEROLATERAL, ("QRS",)),
    "LMI": (HIGH_LATERAL, ("QRS",)),
    "CLBBB": (LEADS, ("QRS",)),
    "CRBBB": (("I", "V1", "V2", "V6"), ("QRS",)),
    "LAFB": (LIMB, ("QRS",)),
    "1AVB": (LEADS, ("PR",)),
    "WPW": (LEADS, ("PR", "QRS")),
    "LNGQT": (LEADS, ("T",)),
    "INJAS": (ANTERIOR_SEPTAL, ("ST",)),
    "INJAL": (ANTEROLATERAL, ("ST",)),
    "INJIN": (INFERIOR, ("ST",)),
    "ISCAS": (ANTERIOR_SEPTAL, ("ST", "T")),
    "ISCAL": (ANTEROLATERAL, ("ST", "T")),
    "ISCIN": (INFERIOR, ("ST", "T")),
    "ISCLA": (HIGH_LATERAL, ("ST", "T")),
}


def parse_noise_leads(text: object) -> tuple[str, ...] | None:
    """
    Leads named in one PTB-XL noise annotation.

    Lead names, ranges in canonical order (``I-AVF``) and ``alles`` (all leads) are read;
    severity words and question marks are ignored.

    Parameters
    ----------
    text : object
        Cell of a noise column; NaN for no annotation.

    Returns
    -------
    tuple of str or None
        Canonical lead names in canonical order, or None if no lead is named.
    """
    if not isinstance(text, str):
        return None
    found: set[int] = set()
    for token in re.split(r"[,\s]+", text.upper().replace("?", " ")):
        if token == "ALLES":
            found.update(range(len(LEADS)))
        elif "-" in token:
            start, _, end = token.partition("-")
            if start in UPPER_LEADS and end in UPPER_LEADS:
                found.update(range(UPPER_LEADS.index(start), UPPER_LEADS.index(end) + 1))
        elif token in UPPER_LEADS:
            found.add(UPPER_LEADS.index(token))
    return tuple(LEADS[k] for k in sorted(found)) if found else None


def noisy_leads(row: pd.Series) -> tuple[str, ...] | None:
    """
    Union of leads named in a record's four noise columns.

    Parameters
    ----------
    row : pd.Series
        PTB-XL metadata row.

    Returns
    -------
    tuple of str or None
        Leads in canonical order, or None if no column names a lead.
    """
    leads: set[str] = set()
    for column in NOISE_COLUMNS:
        parsed = parse_noise_leads(row[column])
        if parsed:
            leads.update(parsed)
    return tuple(lead for lead in LEADS if lead in leads) or None


def is_clean_normal(row: pd.Series, statements: pd.DataFrame) -> bool:
    """
    Decide whether a record may enter a normal reference.

    A clean normal has NORM at likelihood 100, no other diagnostic or form statement, only
    clean sinus rhythm statements, and no noise, extra-beat or pacemaker annotation.

    Parameters
    ----------
    row : pd.Series
        PTB-XL metadata row with parsed ``scp_codes``.
    statements : pd.DataFrame
        SCP statement table.

    Returns
    -------
    bool
        Whether the record may enter a normal reference.
    """
    codes = row["scp_codes"]
    if codes.get("NORM", 0) != 100:
        return False
    for code in codes:
        if code == "NORM":
            continue
        if statements.loc[code, "rhythm"] == 1:
            if code not in CLEAN_RHYTHMS:
                return False
        else:
            return False
    quality = list(NOISE_COLUMNS) + ["extra_beats", "pacemaker"]
    return bool(row[quality].isna().all())


def build_partition(meta: pd.DataFrame, statements: pd.DataFrame, seed: int = PARTITION_SEED) -> pd.DataFrame:
    """
    Patient split of PTB-XL folds 1-8 into ``fit`` and ``evaluation``, with record annotations.

    Parameters
    ----------
    meta : pd.DataFrame
        ``eda.ptbxl.load_metadata`` output.
    statements : pd.DataFrame
        SCP statement table.
    seed : int
        Shuffle seed for the sorted patient IDs.

    Returns
    -------
    pd.DataFrame
        One row per folds 1-8 record: ``ecg_id``, ``patient_id``, ``strat_fold``, ``partition``,
        ``clean_normal``, ``noisy_leads`` (``;``-joined, empty if none) and ``region_statements``
        (``;``-joined table statements with likelihood at least 50).
    """
    table = meta[meta.strat_fold <= 8]
    patients = np.unique(table.patient_id.to_numpy())
    rng = np.random.default_rng(seed)
    shuffled = patients.copy()
    rng.shuffle(shuffled)
    evaluation = set(shuffled[: len(shuffled) // 3])
    rows = pd.DataFrame(
        {
            "ecg_id": table.index.to_numpy(),
            "patient_id": table.patient_id.to_numpy().astype(int),
            "strat_fold": table.strat_fold.to_numpy(),
        }
    )
    rows["partition"] = np.where(rows.patient_id.isin(evaluation), "evaluation", "fit")
    rows["clean_normal"] = [is_clean_normal(row, statements) for _, row in table.iterrows()]
    rows["noisy_leads"] = [";".join(noisy_leads(row) or ()) for _, row in table.iterrows()]
    rows["region_statements"] = [
        ";".join(code for code in REGION_TABLE if row["scp_codes"].get(code, 0) >= 50)
        for _, row in table.iterrows()
    ]
    return rows


def load_partition() -> pd.DataFrame:
    """
    Frozen partition table.

    Returns
    -------
    pd.DataFrame
        ``docs/locbench-partition-v1.json``; empty strings mean no noisy lead or region statement.
    """
    return pd.read_json(PARTITION_PATH, orient="split", dtype=False)
