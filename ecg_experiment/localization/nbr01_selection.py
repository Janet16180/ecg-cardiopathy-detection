"""NBR-01 SPH label groups, their expected regions and the one-ECG-per-patient sample."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROWS = Path("outputs/pr74_fresh_baseline_exact_20261028/features/sph_rows.csv")
CODES = Path("data/raw/sph/sph/17912507/code.csv")
LEAD_NAMES = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
GROUPS = ("inferior", "anterior", "lateral", "lbbb", "rbbb", "stt", "normal")
TERRITORY_LEADS = {"inferior": ("II", "III", "aVF"), "anterior": ("V1", "V2", "V3", "V4"),
                   "lateral": ("I", "aVL", "V5", "V6")}
BBB_LEADS = {"rbbb": ("V1", "V2"), "lbbb": ("I", "aVL", "V5", "V6")}
INFERIOR_MI, ANTERIOR_MI, EXTENSIVE_ANTERIOR_MI = {"161"}, {"160", "165"}, {"166"}
MI = INFERIOR_MI | ANTERIOR_MI | EXTENSIVE_ANTERIOR_MI
LBBB, RBBB = {"104"}, {"106"}
CONDUCTION = {"101", "102", "104", "105", "106", "108"}
HYPERTROPHY = {"142", "143", "153"}
STT = {"145", "146", "147"}
NORMAL = {"1"}
SEED = 20261028


def statement_codes(value: str) -> set[str]:
    """
    Return the base AHA codes of one SPH ``AHA_Code`` field, without modifiers.

    Parameters
    ----------
    value : str
        Field such as ``"22;145+362"``.

    Returns
    -------
    set[str]
        Base codes, such as ``{"22", "145"}``.
    """
    return {token.split("+")[0].strip() for token in str(value).split(";") if token.strip()}


def group_of(codes: set[str], target: int) -> str | None:
    """
    Return the single NBR-01 group of one SPH ECG, or None if it belongs to none.

    Parameters
    ----------
    codes : set[str]
        Base AHA codes.
    target : int
        Project binary label of the ECG.

    Returns
    -------
    str or None
        ``inferior`` (inferior MI, no anterior MI), ``anterior`` (anterior or anteroseptal MI, no inferior
        or extensive anterior MI), ``lbbb`` (LBBB, no MI), ``rbbb`` (RBBB, no LBBB, no MI), ``stt`` (ST
        deviation, ST-T or T-wave statement with no MI, conduction or hypertrophy statement), ``normal``
        (only "Normal ECG" and a negative label). SPH has no lateral MI statement, so ``lateral`` is never
        returned.
    """
    if codes & INFERIOR_MI and not codes & (ANTERIOR_MI | EXTENSIVE_ANTERIOR_MI):
        return "inferior"
    if codes & ANTERIOR_MI and not codes & (INFERIOR_MI | EXTENSIVE_ANTERIOR_MI):
        return "anterior"
    if codes & MI:
        return None
    if codes & LBBB:
        return "lbbb"
    if codes & RBBB:
        return "rbbb"
    if codes & STT and not codes & (CONDUCTION | HYPERTROPHY):
        return "stt"
    if codes == NORMAL and target == 0:
        return "normal"
    return None


def evaluation_rows(input_root: Path) -> pd.DataFrame:
    """
    SPH evaluation ECGs with their NBR-01 group.

    Parameters
    ----------
    input_root : Path
        Checkout holding ``outputs``.

    Returns
    -------
    pd.DataFrame
        Every ``evaluation``-role row of the frozen baseline table with ``codes`` and ``group`` columns,
        and ``position``, its row in the 17,402-row SPH feature and prediction order.
    """
    rows = pd.read_csv(input_root / ROWS, dtype=str, keep_default_na=False)
    rows["position"] = np.arange(len(rows))
    rows = rows.loc[rows["role"].eq("evaluation")].reset_index(drop=True)
    rows["codes"] = rows["aha_codes"].map(statement_codes)
    rows["group"] = [group_of(codes, int(target)) for codes, target in zip(rows["codes"], rows["target"],
                                                                          strict=True)]
    return rows


def group_counts(rows: pd.DataFrame) -> dict[str, dict[str, int]]:
    """ECG and patient counts of each group, from labels only."""
    return {group: {"ecgs": int(rows["group"].eq(group).sum()),
                    "patients": int(rows.loc[rows["group"].eq(group), "patient_key"].nunique())}
            for group in GROUPS}


def draw_sample(rows: pd.DataFrame, cap: int, normals: int, seed: int = SEED) -> pd.DataFrame:
    """
    One ECG per patient, at most ``cap`` patients per abnormal group and ``normals`` normal patients.

    Groups are drawn in ``GROUPS`` order with one generator. In each group the sorted unique patients not
    already drawn are permuted, the first ones kept, and one of each kept patient's group ECGs is drawn
    with the same generator, as in Experiment 070's ``one_per_patient``.

    Parameters
    ----------
    rows : pd.DataFrame
        Output of ``evaluation_rows``.
    cap : int
        Patient cap of every abnormal group.
    normals : int
        Patient count of the normal group.
    seed : int
        Generator seed.

    Returns
    -------
    pd.DataFrame
        The drawn rows with their group, in draw order.
    """
    rng = np.random.default_rng(seed)
    used: set[str] = set()
    parts = []
    for group in GROUPS:
        eligible = rows.loc[rows["group"].eq(group)]
        patients = np.array(sorted(set(eligible["patient_key"]) - used), dtype=object)
        chosen = rng.permutation(patients)[: normals if group == "normal" else cap]
        for patient in chosen:
            ids = np.sort(eligible.loc[eligible["patient_key"].eq(patient), "ecg_id"].to_numpy())
            parts.append(eligible.loc[eligible["ecg_id"].eq(ids[rng.integers(0, len(ids))])])
            used.add(patient)
    return pd.concat(parts, ignore_index=True) if parts else rows.iloc[:0]


def statement_text(value: str, input_root: Path) -> list[str]:
    """
    Readable SPH statements of one ``AHA_Code`` field, modifiers appended in lower case.

    Parameters
    ----------
    value : str
        Field such as ``"22;145+362"``.
    input_root : Path
        Checkout holding ``data/raw/sph``.

    Returns
    -------
    list[str]
        Statements, such as ``["Sinus bradycardia", "ST deviation, depression"]``.
    """
    table = pd.read_csv(input_root / CODES, dtype=str)
    names = dict(zip(table["Code"], table["Description"], strict=True))
    statements = []
    for token in str(value).split(";"):
        parts = [part.strip() for part in token.split("+") if part.strip()]
        if parts:
            statements.append(", ".join([names[parts[0]]] + [names[part].lower() for part in parts[1:]]))
    return statements
