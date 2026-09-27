"""Load PTB-XL metadata and derive binary labels independently of the project pipeline."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from ecg_experiment.eda.signals import canonical_order

ROOT = Path(__file__).resolve().parents[2]
PTBXL_DIR = ROOT / "data/raw/ptb-xl/1.0.3"
MANIFEST_DIR = ROOT / "data/processed/ptbxl"
OUTPUT_DIR = ROOT / "outputs/eda"

QUALITY_COLUMNS = ["baseline_drift", "static_noise", "burst_noise", "electrodes_problems",
                   "extra_beats", "pacemaker"]


def load_metadata() -> pd.DataFrame:
    """
    Load the PTB-XL record table with parsed SCP codes and derived columns.

    Returns
    -------
    pd.DataFrame
        One row per ECG indexed by ``ecg_id``. ``scp_codes`` is a dict of
        code to likelihood, and ``age_capped`` replaces the 300 placeholder
        used for patients older than 89 with NaN.
    """
    table = pd.read_csv(PTBXL_DIR / "ptbxl_database.csv", index_col="ecg_id")
    table["scp_codes"] = table["scp_codes"].apply(ast.literal_eval)
    table["age_capped"] = table["age"].where(table["age"] < 300)
    table["recording_date"] = pd.to_datetime(table["recording_date"])
    return table


def load_statements() -> pd.DataFrame:
    """
    Load the SCP statement table.

    Returns
    -------
    pd.DataFrame
        One row per SCP code with the diagnostic, form and rhythm flags.
    """
    return pd.read_csv(PTBXL_DIR / "scp_statements.csv", index_col=0)


def diagnostic_classes(statements: pd.DataFrame) -> dict[str, str]:
    """
    Map diagnostic SCP codes to their superclass.

    Parameters
    ----------
    statements : pd.DataFrame
        Output of ``load_statements``.

    Returns
    -------
    dict[str, str]
        Superclass (NORM, MI, STTC, CD, HYP) keyed by SCP code.
    """
    diagnostic = statements[statements["diagnostic"] == 1]
    return diagnostic["diagnostic_class"].to_dict()


def project_label(codes: dict[str, float], classes: dict[str, str]) -> float:
    """
    Reproduce the binary rule described in the project documentation.

    NORM records are negative only when NORM and SR are their sole codes.
    Records without NORM are positive when any diagnostic code belongs to an
    abnormal superclass. Everything else is unlabeled.

    Parameters
    ----------
    codes : dict[str, float]
        SCP code to likelihood for one record.
    classes : dict[str, str]
        Output of ``diagnostic_classes``.

    Returns
    -------
    float
        1.0, 0.0, or NaN when unlabeled.
    """
    listed = set(codes)
    label = np.nan
    if "NORM" in listed:
        label = 0.0 if listed <= {"NORM", "SR"} else np.nan
    elif any(classes.get(code, "NORM") != "NORM" for code in listed):
        label = 1.0
    return label


def superclass_label(codes: dict[str, float], classes: dict[str, str]) -> float:
    """
    Binary label from diagnostic superclasses, as in the PTB-XL benchmark.

    Negative when NORM is the only diagnostic superclass, positive when any
    abnormal superclass is present, unlabeled when no diagnostic code exists.
    Rhythm and form codes are ignored.

    Parameters
    ----------
    codes : dict[str, float]
        SCP code to likelihood for one record.
    classes : dict[str, str]
        Output of ``diagnostic_classes``.

    Returns
    -------
    float
        1.0, 0.0, or NaN when unlabeled.
    """
    superclasses = {classes[code] for code in codes if code in classes}
    label = np.nan
    if superclasses == {"NORM"}:
        label = 0.0
    elif superclasses:
        label = 1.0
    return label


def superclasses(codes: dict[str, float], classes: dict[str, str]) -> str:
    """
    Join the sorted diagnostic superclasses of a record.

    Parameters
    ----------
    codes : dict[str, float]
        SCP code to likelihood for one record.
    classes : dict[str, str]
        Output of ``diagnostic_classes``.

    Returns
    -------
    str
        For example ``"CD+MI"``, or ``"none"`` without diagnostic codes.
    """
    found = sorted({classes[code] for code in codes if code in classes})
    return "+".join(found) or "none"


def load_manifest(name: str, fraction: str = "1") -> pd.DataFrame:
    """
    Load one of the project's seed-42 split manifests.

    Parameters
    ----------
    name : str
        Manifest stem, such as ``labeled_train`` or ``test``.
    fraction : str
        Label fraction directory suffix, ``"1"`` or ``"0.1"``.

    Returns
    -------
    pd.DataFrame
        Manifest indexed by ``ecg_id``.
    """
    path = MANIFEST_DIR / f"seed42_fraction{fraction}" / f"{name}.csv"
    return pd.read_csv(path, index_col="ecg_id")


def read_signal(filename: str) -> tuple[np.ndarray, int]:
    """
    Read one PTB-XL record in physical units and canonical lead order.

    Parameters
    ----------
    filename : str
        Relative record path from the metadata table (``filename_hr`` or
        ``filename_lr``), without extension.

    Returns
    -------
    tuple[np.ndarray, int]
        Array of shape ``(samples, 12)`` in mV, and the sampling rate.
    """
    signal, fields = wfdb.rdsamp(str(PTBXL_DIR / filename))
    return canonical_order(signal, fields["sig_name"]), int(fields["fs"])


SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def superclass_flags(meta: pd.DataFrame, classes: dict[str, str]) -> pd.DataFrame:
    """
    One boolean column per diagnostic superclass, plus ``abnormal``.

    A superclass is present when any of its diagnostic codes is listed,
    whatever the likelihood, following the PTB-XL benchmark convention.

    Parameters
    ----------
    meta : pd.DataFrame
        Output of ``load_metadata``.
    classes : dict[str, str]
        Output of ``diagnostic_classes``.

    Returns
    -------
    pd.DataFrame
        Boolean columns ``NORM``, ``MI``, ``STTC``, ``CD``, ``HYP`` and
        ``abnormal`` (any superclass other than NORM), indexed like ``meta``.
    """
    present = meta["scp_codes"].apply(lambda codes: {classes[code] for code in codes if code in classes})
    flags = pd.DataFrame({name: present.apply(lambda found, name=name: name in found)
                          for name in SUPERCLASSES}, index=meta.index)
    flags["abnormal"] = flags[SUPERCLASSES[1:]].any(axis=1)
    return flags


def implausible_rows(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Flag metadata values that are placeholders or physically implausible.

    Children legitimately have small heights and weights, so the height and
    weight rules only apply to adults (18 or older).

    Parameters
    ----------
    meta : pd.DataFrame
        Output of ``load_metadata``.

    Returns
    -------
    pd.DataFrame
        Boolean column per rule, indexed like ``meta``.
    """
    adult = meta["age_capped"].ge(18) | meta["age"].eq(300)
    bmi = meta["weight"] / (meta["height"] / 100) ** 2
    return pd.DataFrame({
        "age 300 (placeholder for > 89)": meta["age"].eq(300),
        "adult height < 120 cm": adult & meta["height"].lt(120),
        "adult weight < 30 kg": adult & meta["weight"].lt(30),
        "weight > 180 kg": meta["weight"].gt(180),
        "BMI < 12 or > 70": bmi.lt(12) | bmi.gt(70),
    }, index=meta.index)
