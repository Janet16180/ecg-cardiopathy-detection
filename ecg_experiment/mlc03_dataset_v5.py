"""
Dataset v5 of MLC-03 Part C: the MLC-01 v4 dataset with the cleaning rules C1-C12 of Amendment 1.

``docs/mlc03-data-quality-protocol-2026-11-11.md`` (Amendment 1) is the contract. The rules reuse the code
that produced their evidence: the Part A flags (``outputs/mlc03_data_quality_20261111/signal/``) and the Part
B rule logic (``mlc03_label_quality``: ``CONTRADICTIONS``, ``UNREADABLE``, ``REQUIRED_LEADS``, ``criterion``,
``population``, ``analysis_table``). The record masks of Part B's rules were written in
``scripts/experiments/run_mlc03_label_quality.py``; scripts are not imported, so ``part_b_masks`` restates
them and ``part_b_counts`` checks them against Part B's ``rules_candidate_counts.csv`` before anything is
built.

Rules C1-C4 exclude records; C5-C12 set labels of the 14 space-B conditions undefined. No rule creates or
flips a positive. Exclusions come first, in rule order; the label rules then act, in rule order, on the
records that stay included. A rule column holds the rule's own selection on the v4 states of every
v4-included record, before any overlap, so its sum equals the Part A or Part B count.

The test roles (PTB-XL fold 10, both SPH halves) get the same code. Their method B measurements are computed
here (``measure_cached``, ``record_windows``), because Part B measured train and validation only; their labels
are inputs of the rules, never a choice.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ecg_experiment import ROOT, mlc01_tables
from ecg_experiment import mlc03_label_quality as lq
from ecg_experiment.measurement_extractor042 import FEATURE_NAMES
from ecg_experiment.mlc02_data import CONDITIONS
from ecg_experiment.mlc02_feature_data import LOCAL, aligned_rows
from ecg_experiment.mlc03_measurements import LEADS

V4 = ROOT / "outputs/mlc01_condition_dataset_20261110_v4"
V5 = ROOT / "outputs/mlc01_condition_dataset_20261111_v5"
QUALITY = ROOT / "outputs/mlc03_data_quality_20261111"
FLAGS_OPEN = QUALITY / "signal/flags_train_validation.parquet"
FLAGS_TEST = QUALITY / "signal/flags_test_roles.parquet"
RULE_COUNTS_A = QUALITY / "signal/proposed_rule_counts.csv"
ANALYSIS = QUALITY / "labels/analysis.parquet"
MEASUREMENTS_B = QUALITY / "labels/measurements_b.parquet"
WINDOWS = QUALITY / "labels/item1_whole_record_windows.csv"
SEPARATE_CODING = QUALITY / "labels/item2_separate_coding.csv"
CUTOFFS = QUALITY / "labels/item2_implied_cutoffs.csv"
RULE_COUNTS_B = QUALITY / "labels/rules_candidate_counts.csv"
OPEN_ROLES = ("train", "validation")
TEST_ROLES = ("test_same_source", "test_external", "sph_calibration")
ROLES = (*OPEN_ROLES, *TEST_ROLES)
SIGNAL_FLAGS = (
    "missing_samples",
    "edge_hold",
    "adc_rail",
    "flat_lead",
    "near_flat_lead",
    "duplicate_lead",
    "identity_violation",
    "reversal_la_ra",
    "reversal_ra_ll",
    "reversal_neutral",
    "amplitude_outlier",
    "baseline_wander",
    "mains_interference",
    "hf_noise",
    "baseline_step",
    "isolated_spike",
    "contact_loss",
    "rhythm_unmeasurable",
)
RECORD_RULES = {"C1": "missing_samples", "C2": "edge_hold", "C3": "adc_rail", "C4": "identity_violation"}
PART_A_RULES = {"C1": "R1", "C2": "R3", "C3": "R2", "C4": "R7"}
INTEGER_ONLY = ("C3", "C4")
FLOAT_STORAGE = ("sph",)
LEAD_LOSS = ("flat_lead__leads", "near_flat_lead__leads", "contact_loss__leads")
AXIS_LEADS = ("I", "II", "III", "aVF")
C9_SOURCES = ("chapman_shaoxing", "cpsc_2018")
C10_SOURCES = ("ptbxl", "cpsc_2018_extra")
NO_SINUS_EVIDENCE = ("sph",)
RATE_CONDITIONS = ("sinus_tachycardia", "sinus_bradycardia")
LABEL_RULES = {
    "C5": tuple(lq.REQUIRED_LEADS),
    "C6": tuple(dict.fromkeys(c[2:] for pair in lq.CONTRADICTIONS for c in pair[:2] if c.startswith("b_"))),
    "C7": tuple(column[2:] for column in lq.UNREADABLE),
    "C8": ("premature_ventricular_complex", "af_or_flutter"),
    "C9": ("complete_rbbb",),
    "C10": ("sinus_bradycardia",),
    "C11": RATE_CONDITIONS,
    "C12": ("lvh",),
}
RULE_ORDER = (*RECORD_RULES, *LABEL_RULES)
PART_B_NAMES = {
    "R1": "R1 finding only after the first 10 s",
    "R2": "R2 complete RBBB measurably narrow where completeness is not coded",
    "R3": "R3 sinus bradycardia negative meeting 60/100 per minute in a source whose cut-off differs",
    "R5": "R5 a lead the condition is read on is flat",
    "R6+": "R6 positive contradicted by the consensus measurement beyond tolerance",
    "R6-": "R6 negative meeting the measured definition beyond tolerance",
    "R8": "R8 paced rhythm with a conduction or repolarization label",
    "R10": "R10 LVH negative meeting a voltage criterion (voltage alone stays undefined)",
    "R11": "R11 Part A: a lead the condition is read on is flat or loses contact",
}
COUNT_COLUMNS = ["positives_to_undefined", "negatives_to_undefined", "positives", "negatives"]


def rule_column(rule: str, condition: str | None = None) -> str:
    """
    Manifest column of one rule, per record (C1-C4) or per record and condition (C5-C12).

    Parameters
    ----------
    rule : str
        ``C1`` to ``C12``.
    condition : str | None
        Space-B condition of a label rule.

    Returns
    -------
    str
        ``rule_C1`` or ``rule_C5__complete_rbbb``.
    """
    return f"rule_{rule}" if condition is None else f"rule_{rule}__{condition}"


def load_records(roles: tuple[str, ...], path: Path = lq.MANIFEST) -> pd.DataFrame:
    """
    Identity, codes, label states and flat-lead flags of the included records of some roles.

    The same columns and types as ``mlc03_label_quality.load_open_records``, which reads train and validation
    only; this reader also takes the test roles, for the mechanical application of the frozen rules.

    Parameters
    ----------
    roles : tuple[str, ...]
        Roles to keep, from ``ROLES``.
    path : Path
        MLC-01 v4 manifest.

    Returns
    -------
    pd.DataFrame
        ``lq.STRUCTURE`` columns, ``b_<condition>``, ``c_<label>`` (``lq.EXTRA_LABELS``) and ``flat_<lead>``,
        in manifest order.

    Raises
    ------
    ValueError
        If a role is unknown.
    """
    unknown = sorted(set(roles) - set(ROLES))
    if unknown or not roles:
        raise ValueError(f"roles must be a nonempty subset of {ROLES}, got {list(roles)}")
    labels = [f"b_{name}" for name in CONDITIONS] + [f"c_{name}" for name in lq.EXTRA_LABELS]
    flats = [f"flat_{lead}" for lead in LEADS]
    frame = pd.read_csv(
        path,
        usecols=[*lq.STRUCTURE, *labels, *flats],
        dtype=dict.fromkeys(lq.STRUCTURE, str),
        keep_default_na=False,
        low_memory=False,
    )
    frame = frame[(frame["included"] == "True") & frame["role"].isin(roles)].reset_index(drop=True)
    for column in labels:
        frame[column] = pd.to_numeric(frame[column].replace("", np.nan)).astype(float)
    for column in ("age", "duration_s", "n_flat_leads", "n_missing_leads"):
        frame[column] = pd.to_numeric(frame[column].replace("", np.nan))
    for column in flats:
        frame[column] = frame[column].astype(str) == "True"
    return frame


def measurements_a(keys: np.ndarray, directory: Path = LOCAL) -> pd.DataFrame:
    """
    Read method A (the stored MLC-02 measurements) of any included records, with Part B's derived quantities.

    The derivation is ``mlc03_label_quality.method_a``'s, without its open-role guard.

    Parameters
    ----------
    keys : np.ndarray
        Record keys of included records.
    directory : Path
        MLC-02 feature directory with ``rows.csv``, ``measurements.npy`` and ``measurements_status.csv``.

    Returns
    -------
    pd.DataFrame
        ``record_key``, the 130 columns prefixed ``a_``, ``a_status``, ``hr_a``, ``qrs_a``, ``pr_a``,
        ``axis_a``, ``rr_nrmssd_a`` and ``rr_range_a``.
    """
    rows = aligned_rows(Path(directory) / "rows.csv")
    rows = rows[rows["included_v4"].astype(bool)]
    position = pd.Series(rows.index.to_numpy(), index=rows["record_key"].to_numpy())
    array = np.load(Path(directory) / "measurements.npy", mmap_mode="r")
    values = pd.DataFrame(
        np.asarray(array[position.loc[np.asarray(keys)].to_numpy()], dtype=np.float64),
        columns=[f"a_{name}" for name in FEATURE_NAMES],
    )
    status = pd.read_csv(Path(directory) / "measurements_status.csv", dtype=str).set_index("record_key")
    values.insert(0, "record_key", np.asarray(keys))
    values["a_status"] = status.loc[np.asarray(keys), "status"].to_numpy()
    values["hr_a"], values["qrs_a"], values["pr_a"] = (
        values["a_hr_bpm"],
        values["a_qrs_dur_ms"],
        values["a_pr_ms"],
    )
    values["axis_a"] = np.degrees(np.arctan2(values["a_qrs_area_aVF"], values["a_qrs_area_I"]))
    values["rr_nrmssd_a"] = values["a_rr_rmssd_ms"] / values["a_rr_mean_ms"]
    values["rr_range_a"] = values["a_rr_range_rel"] * values["a_rr_mean_ms"]
    return values


CACHE_STATE: dict[str, object] = {}


def measure_cached(keys: list[str]) -> list[dict[str, object]]:
    """
    Run method B on the cached first 10 s of any included records, as ``lq.measure_keys`` does.

    Each window is checked against the cache index's first-10-s hash before it is measured.

    Parameters
    ----------
    keys : list[str]
        Record keys.

    Returns
    -------
    list[dict[str, object]]
        ``record_key`` and ``measure_record``'s values, one dictionary per key.

    Raises
    ------
    ValueError
        If a cached window does not reproduce its hash.
    """
    from ecg_experiment.mlc01_dataset import array_sha256, microvolts
    from ecg_experiment.mlc02_waveforms import load_cache, windows
    from ecg_experiment.mlc03_measurements import measure_record

    if "index" not in CACHE_STATE:
        index, blocks = load_cache(index_name="index_all.csv")
        CACHE_STATE.update(
            index=index,
            blocks=blocks,
            position=pd.Series(index["position"].to_numpy(), index=index["record_key"].to_numpy()),
        )
    index = CACHE_STATE["index"]
    positions = CACHE_STATE["position"].loc[np.asarray(keys)].to_numpy()
    signals = windows(index, CACHE_STATE["blocks"], positions)
    expected = index["sha256_10s"].to_numpy()[positions]
    rows = []
    for key, signal, digest in zip(keys, signals, expected, strict=True):
        if array_sha256(microvolts(signal.T)) != digest:
            raise ValueError(f"cached window of {key} does not reproduce its manifest hash")
        rows.append({"record_key": key, **measure_record(signal)})
    return rows


def window_summary(signal: np.ndarray) -> dict[str, object]:
    """
    Run method B on every complete 10 s window of a whole recording, as ``lq.whole_record_windows`` does.

    Parameters
    ----------
    signal : np.ndarray
        ``[12, samples]`` in mV at 500 Hz, ``LEADS`` order.

    Returns
    -------
    dict[str, object]
        ``windows``, per window lists of ``ectopy_by_window`` (premature aberrant beats, edge beats included),
        ``af_by_window`` (irregular RR without a detectable P wave) and ``hr_by_window``, and
        ``ectopy_first``, ``ectopy_later``, ``af_first`` and ``af_later_windows``.
    """
    from ecg_experiment.mlc03_measurements import measure_record

    results = [
        measure_record(signal[:, start : start + 5000]) for start in range(0, signal.shape[1] - 4999, 5000)
    ]
    ectopy = [
        float(r.get("n_premature_aberrant", np.nan)) + float(r.get("n_premature_aberrant_edge", 0.0))
        for r in results
    ]
    af = [
        bool(
            r.get("rr_nrmssd", 0.0) >= lq.IRREGULAR_NRMSSD
            and r.get("status") == "ok"
            and not (r.get("p_snr", 0.0) >= lq.P_SNR_MIN and r.get("p_amp_mv", 0.0) >= lq.P_AMP_MIN)
        )
        for r in results
    ]
    return {
        "windows": len(results),
        "ectopy_by_window": ectopy,
        "af_by_window": af,
        "hr_by_window": [float(r.get("hr_bpm", np.nan)) for r in results],
        "ectopy_first": ectopy[0],
        "ectopy_later": float(np.nansum(ectopy[1:])) if len(ectopy) > 1 else 0.0,
        "af_first": af[0],
        "af_later_windows": int(sum(af[1:])),
    }


def record_windows(job: tuple[str, str, str]) -> dict[str, object]:
    """
    Read one whole recording through the closed-data guard and summarise its 10 s windows.

    WFDB for the Challenge sources and PTB-XL, HDF5 for SPH, as in ``mlc03_signal_runs.full_record_job``; the
    guard (``mlc02_waveforms.init_worker``) must be set in the process first.

    Parameters
    ----------
    job : tuple[str, str, str]
        Record key, manifest ``raw_path`` and source.

    Returns
    -------
    dict[str, object]
        ``record_key`` and ``window_summary``.

    Raises
    ------
    ValueError
        If the recording is not 500 Hz.
    """
    from ecg_experiment.eda.signals import canonical_order
    from ecg_experiment.mlc01_dataset import read_signal
    from ecg_experiment.mlc02_waveforms import guard_path

    key, raw_path, source = job
    path = guard_path(raw_path, source)
    signal, rate, names = read_signal({"format": "h5" if source == "sph" else "wfdb", "stem": str(path)})
    if rate != 500:
        raise ValueError(f"{key} is not 500 Hz")
    return {"record_key": key, **window_summary(canonical_order(signal, names).T)}


def analysis(records: pd.DataFrame, a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """
    Part B's analysis table (``lq.analysis_table``) of any records.

    Parameters
    ----------
    records : pd.DataFrame
        Output of ``load_records``.
    a : pd.DataFrame
        Output of ``measurements_a`` for the same keys.
    b : pd.DataFrame
        Raw method B rows (``measure_cached``) for the same keys.

    Returns
    -------
    pd.DataFrame
        One row per record, in ``records`` order.
    """
    return lq.analysis_table(records, a, lq.b_columns(b))


def aligned_flags(table: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """
    Part A flags of the included records, aligned with an analysis table.

    Parameters
    ----------
    table : pd.DataFrame
        Records with ``record_key``.
    flags : pd.DataFrame
        Part A flag table (train and validation, or test roles).

    Returns
    -------
    pd.DataFrame
        One row per table row, same index.

    Raises
    ------
    ValueError
        If a record has no included flag row.
    """
    included = flags[flags["included"].astype(bool)].set_index("record_key")
    missing = set(table["record_key"]) - set(included.index)
    if missing:
        raise ValueError(f"{len(missing)} records have no Part A flags")
    aligned = included.loc[table["record_key"].to_numpy()].reset_index()
    aligned.index = table.index
    return aligned


def record_rule_masks(table: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """
    Select the records of the exclusion rules C1-C4.

    Parameters
    ----------
    table : pd.DataFrame
        Records with ``source``.
    flags : pd.DataFrame
        Output of ``aligned_flags``.

    Returns
    -------
    pd.DataFrame
        One boolean column per rule; C3 and C4 never select a source with float storage (SPH).
    """
    out = pd.DataFrame(index=table.index)
    for rule, flag in RECORD_RULES.items():
        mask = flags[flag].fillna(False).astype(bool)
        if rule in INTEGER_ONLY:
            mask = mask & ~table["source"].isin(FLOAT_STORAGE)
        out[rule] = mask
    return out


def lost_leads(flags: pd.DataFrame) -> pd.Series:
    """
    Leads that Part A found flat, near-flat or without contact, per record.

    Parameters
    ----------
    flags : pd.DataFrame
        Output of ``aligned_flags``.

    Returns
    -------
    pd.Series
        A set of lead names per record.
    """
    lost = pd.Series([set() for _ in range(len(flags))], index=flags.index)
    for column in LEAD_LOSS:
        found = flags[column].fillna("").astype(str).str.split(";").map(set)
        lost = lost.combine(found, lambda first, second: first | second)
    return lost.map(lambda leads: leads - {""})


def later_finding(table: pd.DataFrame, windows: pd.DataFrame, condition: str) -> pd.Series:
    """
    Select records whose PVC or AF pattern is measured after the first 10 s only (Part B rule R1).

    Parameters
    ----------
    table : pd.DataFrame
        Analysis table.
    windows : pd.DataFrame
        Window summaries (``window_summary`` columns with ``record_key``) of the records longer than 10 s.
    condition : str
        ``premature_ventricular_complex`` or ``af_or_flutter``.

    Returns
    -------
    pd.Series
        Boolean mask over the table.
    """
    summary = windows.set_index("record_key")
    if condition == "premature_ventricular_complex":
        later = summary["ectopy_first"].lt(1) & summary["ectopy_later"].ge(1)
    else:
        later = ~summary["af_first"].astype(bool) & summary["af_later_windows"].ge(1)
    return table["record_key"].map(later).fillna(False).astype(bool)


def part_b_masks(
    table: pd.DataFrame, flags: pd.DataFrame, windows: pd.DataFrame
) -> list[tuple[str, str, str, pd.Series]]:
    """
    Record masks of the Part B rules that Amendment 1 adopted, as Part B's ``candidate_rules`` wrote them.

    Parameters
    ----------
    table : pd.DataFrame
        Analysis table.
    flags : pd.DataFrame
        Output of ``aligned_flags``.
    windows : pd.DataFrame
        Window summaries of the records longer than 10 s.

    Returns
    -------
    list[tuple[str, str, str, pd.Series]]
        ``(v5 rule, Part B rule name, condition, mask)``. C5 is B-R5 with the first part of B-R11, C6 is
        B-R4, C7 is B-R8, C8 is B-R1, C9 is B-R2, C10 is B-R3, C11 is B-R6 for the rate conditions and C12 is
        B-R10. A mask may cover undefined labels; only defined ones count.
    """
    masks = []
    for condition, leads in lq.REQUIRED_LEADS.items():
        flat = table[[f"flat_{lead}" for lead in leads]].any(axis=1)
        masks.append(("C5", PART_B_NAMES["R5"], condition, flat))
    lost = lost_leads(flags)
    read_on = {**lq.REQUIRED_LEADS, "lafb": AXIS_LEADS, "left_axis_deviation": AXIS_LEADS}
    for condition, leads in read_on.items():
        mask = lost.map(lambda found, leads=leads: bool(found & set(leads))).astype(bool)
        masks.append(("C5", PART_B_NAMES["R11"], condition, mask))
    for first, second, _ in lq.CONTRADICTIONS:
        both = (table[first] == 1) & (table[second] == 1)
        for column in (first, second):
            if column.startswith("b_"):
                masks.append(("C6", f"R4 contradictory pair {first[2:]} + {second[2:]}", column[2:], both))
    paced = table["c_paced_rhythm"] == 1
    for column in lq.UNREADABLE:
        masks.append(("C7", PART_B_NAMES["R8"], column[2:], paced))
    for condition in LABEL_RULES["C8"]:
        positive = table[f"b_{condition}"] == 1
        masks.append(
            ("C8", PART_B_NAMES["R1"], condition, later_finding(table, windows, condition) & positive)
        )
    narrow = table["source"].isin(C9_SOURCES) & (table["qrs"] < 110.0) & (table["b_complete_rbbb"] == 1)
    masks.append(("C9", PART_B_NAMES["R2"], "complete_rbbb", narrow))
    slow = (
        table["source"].isin(C10_SOURCES)
        & lq.population(table, "sinus")
        & (table["hr"] < 60.0)
        & (table["b_sinus_bradycardia"] == 0)
    )
    masks.append(("C10", PART_B_NAMES["R3"], "sinus_bradycardia", slow))
    sinus = lq.population(table, lq.CHECKS["sinus_tachycardia"]["population"])
    sinus = sinus & ~table["source"].isin(NO_SINUS_EVIDENCE)
    for condition in RATE_CONDITIONS:
        regions = lq.criterion(table, condition)
        label = table[f"b_{condition}"]
        masks.append(("C11", PART_B_NAMES["R6+"], condition, (label == 1) & regions["outside_tol"]))
        masks.append(("C11", PART_B_NAMES["R6-"], condition, (label == 0) & regions["inside_tol"] & sinus))
    voltage = (table["b_lvh"] == 0) & (table["status_b"] == "ok") & table["voltage_criterion"].astype(bool)
    masks.append(("C12", PART_B_NAMES["R10"], "lvh", voltage))
    return masks


def label_rule_masks(
    parts: list[tuple[str, str, str, pd.Series]], table: pd.DataFrame
) -> dict[tuple[str, str], pd.Series]:
    """
    One mask per label rule and condition: the union of its Part B parts, on defined labels only.

    Parameters
    ----------
    parts : list[tuple[str, str, str, pd.Series]]
        Output of ``part_b_masks``.
    table : pd.DataFrame
        The same analysis table.

    Returns
    -------
    dict[tuple[str, str], pd.Series]
        ``(rule, condition)`` in ``LABEL_RULES`` order to a boolean mask.
    """
    out = {}
    for rule, conditions in LABEL_RULES.items():
        for condition in conditions:
            masks = [mask for name, _, target, mask in parts if name == rule and target == condition]
            union = np.logical_or.reduce([mask.fillna(False).astype(bool).to_numpy() for mask in masks])
            out[(rule, condition)] = pd.Series(union, index=table.index) & table[f"b_{condition}"].notna()
    return out


def part_b_counts(parts: list[tuple[str, str, str, pd.Series]], table: pd.DataFrame) -> pd.DataFrame:
    """
    Part B's ``rule_counts`` of every adopted part, for comparison with ``rules_candidate_counts.csv``.

    Parameters
    ----------
    parts : list[tuple[str, str, str, pd.Series]]
        Output of ``part_b_masks``.
    table : pd.DataFrame
        The same analysis table.

    Returns
    -------
    pd.DataFrame
        ``rule``, ``condition``, ``source``, ``role`` and ``COUNT_COLUMNS``; rows where the part changes no
        label are dropped, as Part B did.
    """
    rows = [row for _, name, condition, mask in parts for row in lq.rule_counts(table, name, condition, mask)]
    counts = pd.DataFrame(rows)
    return counts[(counts["positives_to_undefined"] + counts["negatives_to_undefined"]) > 0].reset_index(
        drop=True
    )


def compare_counts(
    found: pd.DataFrame, expected: pd.DataFrame, keys: list[str], values: list[str]
) -> pd.DataFrame:
    """
    Cell-by-cell comparison of two count tables on their keys.

    Parameters
    ----------
    found, expected : pd.DataFrame
        Count tables.
    keys : list[str]
        Key columns.
    values : list[str]
        Count columns compared.

    Returns
    -------
    pd.DataFrame
        The outer join with ``<value>_found``, ``<value>_expected`` and ``equal`` (missing rows count as 0).
    """
    merged = found[keys + values].merge(
        expected[keys + values], on=keys, how="outer", suffixes=("_found", "_expected")
    )
    for value in values:
        for side in ("found", "expected"):
            merged[f"{value}_{side}"] = merged[f"{value}_{side}"].fillna(0).astype(int)
    merged["equal"] = np.logical_and.reduce(
        [merged[f"{value}_found"] == merged[f"{value}_expected"] for value in values]
    )
    return merged


def expected_sources() -> dict[str, list[str]]:
    """
    Derive the sources of C9 and C10 from Part B's outputs, to check the constants of this module.

    Returns
    -------
    dict[str, list[str]]
        ``C9``: sources where complete RBBB is not coded separately; ``C10``: sources whose implied
        bradycardia cut-off lies more than 5 per minute below 60 (``cutoff_mask`` of Part B).
    """
    coding = pd.read_csv(SEPARATE_CODING)
    uncoded = coding[(coding["condition"] == "complete_rbbb") & coding["not_coded_separately"].astype(bool)]
    cutoffs = pd.read_csv(CUTOFFS)
    rows = cutoffs[cutoffs["condition"] == "sinus_bradycardia"]
    shifted = rows[(rows["ci_high"] < 55.0) | (rows["ci_low"] > 65.0)]
    return {"C9": sorted(uncoded["source"]), "C10": sorted(shifted["source"])}


def flag_columns(flags: pd.DataFrame) -> pd.DataFrame:
    """
    Collect the 18 Part A flags of every flagged candidate, keyed by record.

    Parameters
    ----------
    flags : pd.DataFrame
        Part A flag tables of all roles, concatenated.

    Returns
    -------
    pd.DataFrame
        ``record_key`` and one ``flag_<name>`` column per flag.
    """
    return flags[["record_key", *SIGNAL_FLAGS]].rename(
        columns={name: f"flag_{name}" for name in SIGNAL_FLAGS}
    )


def apply_rules(
    manifest: pd.DataFrame, record_masks: pd.DataFrame, label_masks: dict[tuple[str, str], pd.Series]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply the rules to the v4 manifest, read as text, and count the flow from v4.

    Exclusions (C1-C4) come first: a selected record is no longer included and its exclusion reason is the
    first rule that selects it. The label rules (C5-C12), in order, then set the selected defined labels of
    the records still included to undefined (an empty cell); the likelihood flag of such a label is cleared,
    since it marks PTB-XL positives only. No other cell changes.

    Parameters
    ----------
    manifest : pd.DataFrame
        v4 ``manifest.csv`` read with ``dtype=str`` and ``keep_default_na=False``.
    record_masks : pd.DataFrame
        C1-C4 masks with a ``record_key`` column, for the v4-included records.
    label_masks : dict[tuple[str, str], pd.Series]
        Output of ``label_rule_masks``, with the records in ``record_masks`` order.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        The v5 manifest (v4 columns with changed cells, then one column per rule) and the flow: per rule,
        condition, source and role, the records or labels the rule selects, those already excluded or
        undefined by an earlier rule, and those it changes.

    Raises
    ------
    ValueError
        If a mask covers a record that was not included in v4.
    """
    out = manifest.copy()
    row = pd.Series(np.arange(len(out)), index=out["record_key"].to_numpy())
    positions = row.loc[record_masks["record_key"].to_numpy()].to_numpy()
    if not (out["included"].to_numpy()[positions] == "True").all():
        raise ValueError("a rule mask covers a record that is not included in v4")
    source = out["source"].to_numpy()[positions]
    role = out["role"].to_numpy()[positions]
    flow = []
    excluded = np.zeros(len(positions), dtype=bool)
    reason = out["exclusion_reason"].to_numpy(dtype=object).copy()
    for rule in RECORD_RULES:
        selected = record_masks[rule].to_numpy(dtype=bool)
        new = selected & ~excluded
        reason[positions[new]] = rule
        for (s, r), index in pd.Series(np.arange(len(positions))).groupby([source, role]).groups.items():
            index = np.asarray(index)
            flow.append(
                {
                    "rule": rule,
                    "condition": "",
                    "source": s,
                    "role": r,
                    "records": len(index),
                    "selected": int(selected[index].sum()),
                    "already_excluded": int((selected & excluded)[index].sum()),
                    "excluded": int(new[index].sum()),
                }
            )
        excluded |= selected
    included = np.full(len(out), False)
    included[positions] = True
    included[positions[excluded]] = False
    out["included"] = np.where(
        out["included"] == "True", np.where(included, "True", "False"), out["included"]
    )
    out["exclusion_reason"] = reason
    columns = {rule: np.zeros(len(out), dtype=bool) for rule in RECORD_RULES}
    for rule in RECORD_RULES:
        columns[rule][positions] = record_masks[rule].to_numpy(dtype=bool)
    keep = ~excluded
    for (rule, condition), mask in label_masks.items():
        label = f"b_{condition}"
        state = out[label].to_numpy(dtype=object)[positions]
        v4_state = manifest[label].to_numpy(dtype=object)[positions]
        selected = mask.to_numpy(dtype=bool)
        if not np.isin(v4_state[selected], ("0", "1")).all():
            raise ValueError(f"{rule} selects an undefined {condition} label")
        new = selected & keep & (state != "")
        cells = positions[new]
        values = out[label].to_numpy(dtype=object).copy()
        values[cells] = ""
        out[label] = values
        flags = out[f"lik50_{condition}"].to_numpy(dtype=object).copy()
        flags[cells] = "False"
        out[f"lik50_{condition}"] = flags
        column = np.zeros(len(out), dtype=bool)
        column[positions] = selected
        columns[(rule, condition)] = column
        for (s, r), index in pd.Series(np.arange(len(positions))).groupby([source, role]).groups.items():
            index = np.asarray(index)
            sel, old = selected[index], v4_state[index]
            flow.append(
                {
                    "rule": rule,
                    "condition": condition,
                    "source": s,
                    "role": r,
                    "records": len(index),
                    "selected_positives": int((sel & (old == "1")).sum()),
                    "selected_negatives": int((sel & (old == "0")).sum()),
                    "on_excluded_records": int((sel & ~keep[index]).sum()),
                    "already_undefined": int((sel & keep[index] & (state[index] == "")).sum()),
                    "undefined_positives": int((new[index] & (old == "1")).sum()),
                    "undefined_negatives": int((new[index] & (old == "0")).sum()),
                }
            )
    for key, values in columns.items():
        out[rule_column(*key) if isinstance(key, tuple) else rule_column(key)] = np.where(
            values, "True", "False"
        )
    return out, pd.DataFrame(flow)


def compare_manifests(v4: pd.DataFrame, v5: pd.DataFrame) -> dict[str, object]:
    """
    Cell-by-cell comparison of v5 with v4 on v4's columns.

    Parameters
    ----------
    v4, v5 : pd.DataFrame
        Manifests read as text, rows in the same order.

    Returns
    -------
    dict[str, object]
        Cells compared and changed, changed cells per column, label transitions of the 14 conditions, and the
        checks (identity columns unchanged, no label created or flipped, changes only in the expected
        columns).
    """
    if list(v4["record_key"]) != list(v5["record_key"]):
        raise ValueError("record keys or their order differ")
    columns = list(v4.columns)
    changed = {name: int((v4[name].to_numpy() != v5[name].to_numpy()).sum()) for name in columns}
    transitions = {}
    for condition in CONDITIONS:
        pairs = pd.crosstab(
            v4[f"b_{condition}"].replace("", "undefined"), v5[f"b_{condition}"].replace("", "undefined")
        )
        transitions[condition] = {
            f"{a}->{b}": int(pairs.loc[a, b])
            for a in pairs.index
            for b in pairs.columns
            if pairs.loc[a, b] and a != b
        }
    allowed = {
        "included",
        "exclusion_reason",
        *(f"b_{c}" for c in CONDITIONS),
        *(f"lik50_{c}" for c in CONDITIONS),
    }
    moves = [move for table in transitions.values() for move in table]
    return {
        "records": len(v4),
        "columns": len(columns),
        "cells_compared": len(v4) * len(columns),
        "cells_changed": int(sum(changed.values())),
        "changed_by_column": {name: count for name, count in changed.items() if count},
        "label_transitions": transitions,
        "only_defined_to_undefined": all(move in ("0->undefined", "1->undefined") for move in moves),
        "changes_only_in_rule_columns": {name for name, count in changed.items() if count} <= allowed,
        "identity_unchanged": all(
            changed[name] == 0
            for name in ("record_key", "source", "role", "group_id", "family", "origin", "raw_path")
        ),
        "included_only_removed": bool(((v5["included"] == "True") <= (v4["included"] == "True")).all()),
    }


def support(manifest: pd.DataFrame) -> pd.DataFrame:
    """
    Positives, negatives and undefined labels per condition, role and source or family (``mlc01_tables``).

    Parameters
    ----------
    manifest : pd.DataFrame
        Manifest read as text.

    Returns
    -------
    pd.DataFrame
        ``state_counts`` rows of space B for every role, with ``below_100`` and ``enters_macro`` (at least 20
        positives and 20 negatives).
    """
    frame = manifest[["included", "role", "source", "family", *(f"b_{c}" for c in CONDITIONS)]].copy()
    frame["included"] = frame["included"] == "True"
    for condition in CONDITIONS:
        frame[f"b_{condition}"] = pd.to_numeric(frame[f"b_{condition}"].replace("", np.nan))
    counts = mlc01_tables.state_counts(frame, [f"b_{c}" for c in CONDITIONS], "B")
    counts["below_100"] = counts["positive"] < 100
    counts["enters_macro"] = (counts["positive"] >= 20) & (counts["negative"] >= 20)
    return counts


def support_comparison(v4: pd.DataFrame, v5: pd.DataFrame) -> pd.DataFrame:
    """
    Support of v4 beside v5, with the changes and the eligibility flags.

    Parameters
    ----------
    v4, v5 : pd.DataFrame
        Outputs of ``support``.

    Returns
    -------
    pd.DataFrame
        One row per condition, role, level and unit; ``loses_eligibility`` and ``drops_below_100``.
    """
    keys = ["space", "label", "role", "level", "unit"]
    merged = v4.merge(v5, on=keys, suffixes=("_v4", "_v5"), validate="one_to_one")
    for name in ("records", "positive", "negative", "undefined"):
        merged[f"{name}_change"] = merged[f"{name}_v5"] - merged[f"{name}_v4"]
    merged["loses_eligibility"] = merged["enters_macro_v4"] & ~merged["enters_macro_v5"]
    merged["drops_below_100"] = ~merged["below_100_v4"] & merged["below_100_v5"]
    return merged
