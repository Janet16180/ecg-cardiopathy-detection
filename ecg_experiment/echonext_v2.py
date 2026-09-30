"""EchoNext rows, version 2: the v1 cache rows with separate training and evaluation use, and checks.

Version 1 (``docs/clean-sph-echonext-v1.md``) has one ``use`` flag and drops the release's
``most_recent_ecg``, which the EDA needs for one-ECG-per-patient analyses. Version 2 adds it and splits
``use`` following ``docs/clean-echonext-v2.md``:

- ``use_training`` excludes every v1 exclusion reason, as ``use`` did.
- ``use_evaluation`` excludes only recordings with part of the signal missing (nonfinite samples, a constant
  lead, a flat segment). It keeps noise-dominated ECGs, which are real recordings of very sick patients, as
  the quality policy prescribes for held-out data.

The waveforms stay in the v1 arrays; row ``row`` of ``<split>.npy`` is the ECG of each row.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew

from .eda.echonext import ARCHIVE, OPEN_SPLITS, member
from .eda.signals import LEADS, heart_rate

MISSING_SIGNAL = ("nonfinite", "constant_lead", "flat_segment")
REASONS = (*MISSING_SIGNAL, "noise_dominated")
COMPOSITE = "shd_moderate_or_greater_flag"
LIMB_RELATIONS = {"II ~ I + III": ("II", "I", "III"), "aVR ~ I + II": ("aVR", "I", "II"),
                  "aVL ~ I + III": ("aVL", "I", "III"), "aVF ~ II + III": ("aVF", "II", "III")}
EXPECTED_SIGNS = {"II ~ I + III": (1, 1), "aVR ~ I + II": (-1, -1), "aVL ~ I + III": (1, -1),
                  "aVF ~ II + III": (1, 1)}


def read_release_columns(columns: list[str], archive: Path = ARCHIVE) -> pd.DataFrame:
    """
    Read selected columns of the EchoNext metadata for every split, so no other column is ever loaded.

    Parameters
    ----------
    columns : list[str]
        Columns to read besides ``ecg_key`` and ``split``.
    archive : Path
        The release ZIP.

    Returns
    -------
    pd.DataFrame
        ``ecg_key``, ``split`` and ``columns`` as strings, in release order.
    """
    with zipfile.ZipFile(archive) as release:
        data = release.read(member("echonext_metadata_100k.csv"))
    return pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False,
                       usecols=["ecg_key", "split", *columns])


def rows_v2(rows: pd.DataFrame, release: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``most_recent_ecg``, ``ecgs_in_split``, ``use_training`` and ``use_evaluation`` to the v1 rows.

    Parameters
    ----------
    rows : pd.DataFrame
        The v1 ``rows.csv`` as strings, with ``use`` as booleans.
    release : pd.DataFrame
        ``ecg_key``, ``split`` and ``most_recent_ecg`` of the release.

    Returns
    -------
    pd.DataFrame
        The v1 columns without ``use``, then the four new columns.

    Raises
    ------
    ValueError
        If the rows are not exactly the release's open-split ECGs, a reason is unknown, or ``use`` differs
        from the reasons.
    """
    release = release[release["split"].isin(OPEN_SPLITS)].set_index("ecg_key")
    if sorted(rows["ecg_key"]) != sorted(release.index) or \
            (release.loc[rows["ecg_key"], "split"].to_numpy() != rows["split"].to_numpy()).any():
        raise ValueError("The v1 rows are not the release's train and val ECGs")
    reasons = rows["exclusion_reasons"].str.split(";").explode()
    if not set(reasons[reasons != ""]) <= set(REASONS):
        raise ValueError("Unknown exclusion reason")
    missing = rows["exclusion_reasons"].apply(lambda text: any(name in text.split(";")
                                                               for name in MISSING_SIGNAL))
    if ((rows["exclusion_reasons"] == "") != rows["use"]).any():
        raise ValueError("The v1 use flag differs from its exclusion reasons")
    out = rows.drop(columns="use")
    out["most_recent_ecg"] = release.loc[rows["ecg_key"], "most_recent_ecg"].to_numpy()
    out["ecgs_in_split"] = rows.groupby(["split", "patient_key"])["ecg_key"].transform("size")
    out["use_training"] = rows["exclusion_reasons"] == ""
    out["use_evaluation"] = ~missing
    return out


def check_alignment(rows: pd.DataFrame, signals: np.ndarray, count: int, seed: int) -> dict[str, float]:
    """
    Compare the waveform heart rate with the cart's ventricular rate, and with shuffled rows as a control.

    Parameters
    ----------
    rows : pd.DataFrame
        Rows of one split with ``row`` and ``ventricular_rate``.
    signals : np.ndarray
        That split's ``(N, 12, 2500)`` array.
    count : int
        Rows sampled.
    seed : int
        Sampling seed.

    Returns
    -------
    dict[str, float]
        Share within 10 bpm, correlation, the shuffled share and the rows compared.
    """
    sample = rows.sample(min(count, len(rows)), random_state=seed)
    estimated = np.array([heart_rate(signals[int(row)].T.astype(np.float64), 250) for row in sample["row"]])
    cart = pd.to_numeric(sample["ventricular_rate"]).to_numpy()
    found = np.isfinite(estimated) & np.isfinite(cart)
    estimated, cart = estimated[found], cart[found]
    shuffled = np.random.default_rng(seed).permutation(cart)
    return {"within_10_bpm": float(np.mean(np.abs(estimated - cart) <= 10)),
            "correlation": float(np.corrcoef(estimated, cart)[0, 1]),
            "within_10_bpm_shuffled": float(np.mean(np.abs(estimated - shuffled) <= 10)),
            "rows": int(found.sum())}


def check_lead_order(signals: np.ndarray) -> dict[str, object]:
    """
    Test the standard lead order: limb-lead linear relations, QRS polarity and precordial neighbours.

    Parameters
    ----------
    signals : np.ndarray
        ``(N, 12, 2500)`` sample of usable ECGs.

    Returns
    -------
    dict[str, object]
        Fitted coefficients and R2 per limb relation and whether the signs match, the median skewness per
        lead, and the correlation of every precordial lead with V1 and with V6.
    """
    flat = signals.astype(np.float64).transpose(1, 0, 2).reshape(len(LEADS), -1)
    lead = {name: flat[index] for index, name in enumerate(LEADS)}
    relations = {}
    for name, (target, first, second) in LIMB_RELATIONS.items():
        design = np.column_stack([lead[first], lead[second]])
        coefficients = np.linalg.lstsq(design, lead[target], rcond=None)[0]
        residual = lead[target] - design @ coefficients
        relations[name] = {"coefficients": coefficients.round(3).tolist(),
                           "r2": float(1 - residual.var() / lead[target].var()),
                           "signs_match": bool((np.sign(coefficients) == EXPECTED_SIGNS[name]).all())}
    skewness = np.median(skew(signals.astype(np.float64), axis=2), axis=0)
    precordial = np.corrcoef(flat[6:])
    return {"limb_relations": relations,
            "median_skewness": dict(zip(LEADS, skewness.round(3).tolist(), strict=True)),
            "precordial_correlation_with_v1": precordial[0].round(3).tolist(),
            "precordial_correlation_with_v6": precordial[-1].round(3).tolist()}


def reason_profile(rows: pd.DataFrame) -> dict[str, dict[str, object]]:
    """
    Aggregate profile of the ECGs removed by each exclusion reason, and of the usable training ECGs.

    Parameters
    ----------
    rows : pd.DataFrame
        Output of ``rows_v2``.

    Returns
    -------
    dict[str, dict[str, object]]
        Per reason and for ``usable_train``: ECGs per split, patients, composite positive share, median QRS,
        share with QRS of 120 ms or more, share without PR, median LVEF, inpatient share and median year.
    """
    groups = {name: rows[rows["exclusion_reasons"].str.split(";").apply(lambda items, n=name: n in items)]
              for name in REASONS}
    groups["usable_train"] = rows[(rows["split"] == "train") & rows["use_training"]]
    profile = {}
    for name, part in groups.items():
        number = {column: pd.to_numeric(part[column], errors="coerce")
                  for column in (COMPOSITE, "qrs_duration", "lvef_value", "acquisition_year")}
        profile[name] = {
            "ecgs": part["split"].value_counts().to_dict(), "patients": int(part["patient_key"].nunique()),
            "composite_positive": float(number[COMPOSITE].mean()) if len(part) else None,
            "median_qrs_ms": float(number["qrs_duration"].median()) if len(part) else None,
            "qrs_120_or_more": float((number["qrs_duration"] >= 120).mean()) if len(part) else None,
            "pr_missing": float((part["pr_interval"] == "").mean()) if len(part) else None,
            "median_lvef": float(number["lvef_value"].median()) if len(part) else None,
            "inpatient": float((part["location_setting"] == "inpatient").mean()) if len(part) else None,
            "median_year": float(number["acquisition_year"].median()) if len(part) else None,
        }
    return profile


def bound_shares(rows: pd.DataFrame, signals: np.ndarray, count: int, seed: int) -> dict[str, float]:
    """
    Share of samples at a lead's minimum or maximum, the trace of the release's clipping, per sampled ECG.

    Parameters
    ----------
    rows : pd.DataFrame
        Usable rows of one split.
    signals : np.ndarray
        That split's array.
    count : int
        Rows sampled.
    seed : int
        Sampling seed.

    Returns
    -------
    dict[str, float]
        Median share and the share of ECGs above 1% and 5%.
    """
    shares = []
    for row in rows.sample(min(count, len(rows)), random_state=seed)["row"]:
        signal = signals[int(row)]
        high = signal == signal.max(axis=1, keepdims=True)
        at_bound = high | (signal == signal.min(axis=1, keepdims=True))
        shares.append(at_bound.mean())
    shares = np.array(shares)
    return {"median": float(np.median(shares)), "above_1_percent": float((shares > 0.01).mean()),
            "above_5_percent": float((shares > 0.05).mean()), "rows": len(shares)}


def one_per_patient(rows: pd.DataFrame) -> dict[str, float]:
    """
    Composite positive share of all training ECGs, of each training patient's most recent ECG, and of val.

    Parameters
    ----------
    rows : pd.DataFrame
        Output of ``rows_v2``.

    Returns
    -------
    dict[str, float]
        The three shares, over rows usable for evaluation, and the number of most recent training ECGs.
    """
    usable = rows[rows["use_evaluation"]]
    train = usable[usable["split"] == "train"]
    recent = train[train["most_recent_ecg"] == "1"]
    positive = lambda part: float(pd.to_numeric(part[COMPOSITE]).mean())  # noqa: E731
    return {"train_all": positive(train), "train_most_recent": positive(recent),
            "val": positive(usable[usable["split"] == "val"]), "train_most_recent_ecgs": len(recent),
            "train_patients": int(train["patient_key"].nunique())}
