"""
KEY-06: structure of the xECG embedding space beyond the 14 labelled conditions.

Executes ``docs/key06-embedding-structure-protocol-2026-11-11.md``. The protocol leaves some operational
choices open; they are fixed here, before any full run, and listed under departures in the results report.

Records and roles

- Only the included train (discovery) and validation (replication) records of MLC-01 v4 are read.
  ``read_by_role`` reads the identity columns first and parses the other columns of those rows only, so no
  value of a test-role row is parsed; any closed role raises ``PermissionError``. Feature arrays are
  memory-mapped and indexed by those rows; waveforms come through the MLC-03 role-checked cache reader.

Factor groups (step 1)

- ``source``: indicators of the six sources (first dropped).
- ``labels``: positive indicator of each of the 14 conditions (negative and undefined both 0).
- ``heart_rate``: cubic spline of ``hr_bpm`` (five knots at train quantiles, values clipped to the train
  0.5-99.5 percentiles, missing set to the train median) and a missing indicator. ``rr_mean_ms`` and
  ``n_beats`` restate the rate and enter no group.
- ``measurements``: the other 127 measurements, each mapped to normal scores by train quantiles, missing set
  to 0, with a missing indicator for every column that has missing train values.
- ``signal_flags``: the 15 MLC-03 flags that passed its visual check (``RELIABLE_FLAGS``); ``reversal_ra_ll``
  and ``rhythm_unmeasurable`` failed it and ``duplicate_lead`` flags no record.
- ``age_sex``: cubic spline of age (five knots, same rules) with a missing indicator, and indicators of male
  and of unknown sex.
- A share is the variance-weighted out-of-sample R^2 on validation of an ordinary least-squares fit on train,
  over all 1,024 standardized dimensions (the same as over all principal components); it is also given for
  the principal components that keep 90% of the variance and per component. "After the others" is R^2 of all
  groups minus R^2 of all groups but this one.
- Reverse direction: logistic regression (C = 1) for binary factors and ridge regression (alpha = 1, no
  tuning) for continuous ones, fitted on train and read on validation. "Within source" is the stratified
  AUROC (the probability that a positive scores above a negative of the same source) and the within-source
  R^2 (residual sum of squares over the sum of squares around each source's mean). The source itself is
  scored by accuracy and balanced accuracy.

Residual space (step 2)

- The ordinary least-squares fit on train of the embedding on ``source``, ``heart_rate``, ``signal_flags`` and
  ``age_sex`` is subtracted from train and validation records alike.

Clustering (step 3)

- Each space is clustered in its own principal components that keep 90% of the train variance (refitted on
  the label-negative subset for that subset). k-means: ``n_init`` 4, seed ``SEED + 1000 * m + r + 1`` (m the
  method number, r the replicate, -1 for the full fit). HDBSCAN: ``min_cluster_size`` 100 and the library's
  other defaults. Twenty subsamples of 80% of the train records of a subset (one generator, seed
  ``SEED + 7``) are shared by every method and space. A cluster's Jaccard in a subsample is its best overlap
  with any subsample cluster, counting only its members inside the subsample; stability is the median of the
  20 values. Clusters are numbered by decreasing train size. Validation records go to the nearest k-means
  centre, or to the HDBSCAN label of their nearest train record (noise included).
- Label-negative subset: records whose defined labels hold no positive.

Description and enrichment (step 4)

- Statements are the raw codes of a record (PTB-XL SCP statements of any likelihood and its ``heart_axis`` as
  ``axis:X``, SNOMED CT codes of the Challenge sources) that are not a positive code of one of the 14
  conditions; ``axis:missing`` is left out. A statement is tested in a set when at least 30 of the set's train
  records carry it.
- Enrichment is the Mantel-Haenszel odds ratio of the statement (or label) among members against
  non-members, with the six sources as strata. Strata without members, without non-members, or without
  positive or negative records do not count. Where a stratum has an empty cell, 0.5 is added to its four
  cells for the odds ratio (not for the test). The p-value is the Cochran-Mantel-Haenszel test without
  continuity correction. Undefined label entries are left out of that label's tables.
- Benjamini-Hochberg at 5% is applied once over all cluster-statement pairs of the study on train, and once
  over all cluster-label pairs.

Reading the result

- Stable: median Jaccard at least 0.6.
- Not one source: at least two source families each hold at least 10% of the members or at least 50 members.
- Replicated: the strongest enrichment on train is the label or statement with the largest absolute log
  odds ratio among those significant after Benjamini-Hochberg; on validation, its log odds ratio has the same
  sign and its 95% interval (2,000 patient resamples, ``intervals.patient_resample``) excludes 0. A cluster
  with no significant enrichment on train fails this rule.
- Not an artefact: no reliable flag has a train rate among members more than twice its rate among
  non-members (a flag absent outside and present inside counts as more than twice). A second reading that
  also requires at least five flagged members is reported beside it and does not decide.
- Not already labelled: no condition is positive in more than half of the members (undefined members count
  in the denominator).
- Candidates are the clusters that pass all five, in either space and either subset.

Cross-encoder check (step 5), outliers (step 6) and statements (step 7)

- Step 5 predicts membership of a cluster from the standardized ECG-CPC embedding (logistic, C = 1, fitted on
  the train records of the cluster's subset) and scores it by within-source AUROC on the validation records
  assigned to the cluster, for every candidate and, as context, every cluster that is stable and not one
  source.
- Step 6 scores each train record by the mean distance to its 20 nearest train neighbours in the 90%
  principal components of the space; the top 1% are the outliers.
- Step 7 uses statements with at least 100 train positives that are present (at least five train positives,
  the MLC-01 annotation rule) in at least two source families. Records of sources where the statement is
  present form its train and validation sets. Statements present in one family only are listed separately.
"""

from __future__ import annotations

import ast
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import chi2, rankdata

from ecg_experiment import ROOT, challenge_labels, mlc_labels
from ecg_experiment.intervals import patient_groups, patient_resample
from ecg_experiment.mlc02_data import CONDITIONS, MANIFEST, VALIDATION_SOURCES, check_states

FEATURES = ROOT / "outputs/mlc02_condition_models_20261110/features"
FLAGS_TABLE = ROOT / "outputs/mlc03_data_quality_20261111/signal/flags_train_validation.parquet"
P0_VALIDATION = ROOT / "outputs/mlc02_condition_models_20261110/candidates/P0/validation__seeds-42.npz"
SELECTION_RECORD = ROOT / "outputs/mlc02_condition_models_20261110/stage3/selection_record.json"
SCP_STATEMENTS = ROOT / "data/raw/ptb-xl/1.0.3/scp_statements.csv"
OUTPUT = ROOT / "outputs/key06_embedding_structure_20261111"
WORK = Path("/var/tmp/janetrivera/key06")
OPEN_ROLES = ("train", "validation")
CLOSED_ROLES = ("test_same_source", "test_external", "sph_calibration")
RELIABLE_FLAGS = ("missing_samples", "edge_hold", "adc_rail", "flat_lead", "near_flat_lead",
                  "identity_violation", "reversal_la_ra", "reversal_neutral", "amplitude_outlier",
                  "baseline_wander", "mains_interference", "hf_noise", "baseline_step", "isolated_spike",
                  "contact_loss")
RATE_COLUMNS = ("hr_bpm", "rr_mean_ms", "n_beats")
KEY_MEASUREMENTS = ("hr_bpm", "rr_sdnn_ms", "pr_ms", "qrs_dur_ms", "qt_ms", "p_amp_II", "r_amp_V1",
                    "s_depth_V1", "r_amp_V5", "s_depth_V2", "st_V2", "st_V5", "t_amp_V2", "t_amp_V5",
                    "ptp_II")
FACTOR_GROUPS = ("source", "labels", "heart_rate", "measurements", "signal_flags", "age_sex")
RESIDUAL_GROUPS = ("source", "heart_rate", "signal_flags", "age_sex")
SPACES = ("raw", "residual")
SUBSETS = ("all", "label_negative")
METHODS = ("kmeans_10", "kmeans_20", "kmeans_40", "kmeans_80", "hdbscan")
SEED = 20261111
SPLINE_KNOTS = 5
RIDGE_ALPHA = 1.0
LOGISTIC_C = 1.0
VARIANCE_KEPT = 0.90
HDBSCAN_MIN_CLUSTER = 100
SUBSAMPLES = 20
SUBSAMPLE_SHARE = 0.8
KMEANS_INIT = 4
STATEMENT_MIN_TEST = 30
STATEMENT_MIN_PROBE = 100
SOURCE_MIN_POSITIVES = 5
BH_LEVEL = 0.05
STABLE_JACCARD = 0.6
FAMILY_SHARE = 0.10
FAMILY_COUNT = 50
FLAG_RATIO = 2.0
FLAG_MIN_MEMBERS = 5
LABEL_MAJORITY = 0.5
DRAWS = 2000
OUTLIER_SHARE = 0.01
OUTLIER_NEIGHBOURS = 20
REVIEW_NEAREST = 24
REVIEW_RANDOM = 24
OUTLIER_PLOTS = 48


def check_open_roles(roles: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """
    Refuse every role but train and validation.

    Parameters
    ----------
    roles : tuple[str, ...] | list[str]
        Requested roles.

    Returns
    -------
    tuple[str, ...]
        The roles, unchanged.

    Raises
    ------
    PermissionError
        If a test role is requested.
    ValueError
        If a role is unknown or none is given.
    """
    roles = tuple(roles)
    closed = sorted(set(roles) & set(CLOSED_ROLES))
    if closed:
        raise PermissionError(f"KEY-06 reads train and validation records only, not {closed}")
    unknown = sorted(set(roles) - set(OPEN_ROLES))
    if unknown or not roles:
        raise ValueError(f"unknown or no roles: {unknown}")
    return roles


def read_by_role(path: Path, roles: tuple[str, ...], columns: list[str], included: str) -> pd.DataFrame:
    """
    Read some columns of the included rows of some roles of a CSV table, parsing no other row.

    The ``role`` and inclusion columns are read first; the second pass skips every other line before it
    is parsed. ``position`` is the row's index among the table's data rows.

    Parameters
    ----------
    path : Path
        CSV table with a ``role`` column and a ``True``/``False`` inclusion column.
    roles : tuple[str, ...]
        Roles to keep; checked by ``check_open_roles``.
    columns : list[str]
        Columns to read.
    included : str
        Name of the inclusion column.

    Returns
    -------
    pd.DataFrame
        ``position``, ``role`` and the requested columns as text (empty text for missing), in table order.

    Raises
    ------
    ValueError
        If the second pass does not return exactly the selected rows.
    """
    roles = check_open_roles(roles)
    identity = pd.read_csv(path, usecols=["role", included], dtype=str, keep_default_na=False)
    keep = np.flatnonzero(identity["role"].isin(roles).to_numpy() & (identity[included] == "True").to_numpy())
    lines = frozenset((keep + 1).tolist())
    wanted = list(dict.fromkeys(["role", *columns]))
    frame = pd.read_csv(path, usecols=wanted, dtype=str, keep_default_na=False, low_memory=False,
                        skiprows=lambda line: line > 0 and line not in lines)
    if len(frame) != len(keep) or not frame["role"].isin(roles).all():
        raise ValueError(f"{path} did not return the {len(keep)} selected rows")
    frame.insert(0, "position", keep)
    return frame


def to_float(values: pd.Series) -> np.ndarray:
    """
    Convert text values to float, empty text to NaN.

    Parameters
    ----------
    values : pd.Series
        Text column.

    Returns
    -------
    np.ndarray
        Float64 values.
    """
    return pd.to_numeric(values.replace("", np.nan), errors="raise").to_numpy(dtype=np.float64)


def load_records(roles: tuple[str, ...] = OPEN_ROLES, manifest: Path = MANIFEST,
                 features: Path = FEATURES) -> pd.DataFrame:
    """
    Identity, labels and statements of the included records of some roles, aligned with the feature rows.

    Parameters
    ----------
    roles : tuple[str, ...]
        Roles to keep (train and validation only).
    manifest : Path
        MLC-01 v4 ``manifest.csv``.
    features : Path
        MLC-02 feature directory with ``rows.csv``.

    Returns
    -------
    pd.DataFrame
        One row per record in manifest order: ``record_key``, ``source``, ``family``, ``role``, ``group``,
        ``age`` (float), ``sex`` (``F``, ``M`` or ``unknown``), ``raw_codes``, ``heart_axis``, the 14 ``b_``
        label columns (1.0, 0.0, NaN), ``position`` (feature row), ``cache_block``, ``cache_row`` and
        ``sha256_10s``.

    Raises
    ------
    ValueError
        If the feature rows do not reproduce the manifest's keys, sources, roles and waveform hashes.
    """
    label_names = [f"b_{name}" for name in CONDITIONS]
    columns = ["record_key", "source", "family", "group_id", "age", "sex", "raw_codes", "heart_axis",
               "sha256_10s", *label_names]
    frame = read_by_role(manifest, roles, columns, "included").drop(columns="position")
    rows = read_by_role(features / "rows.csv", roles,
                        ["record_key", "source", "sha256_10s", "cache_block", "cache_row"], "included_v4")
    for name in ("record_key", "source", "role", "sha256_10s"):
        if not np.array_equal(frame[name].to_numpy(), rows[name].to_numpy()):
            raise ValueError(f"feature rows do not reproduce the manifest's {name}")
    frame = frame.assign(group=frame["group_id"], age=to_float(frame["age"]),
                         sex=frame["sex"].where(frame["sex"].isin(["F", "M"]), "unknown"),
                         position=rows["position"].to_numpy(), cache_block=rows["cache_block"].to_numpy(),
                         cache_row=rows["cache_row"].astype(int).to_numpy())
    for name in label_names:
        frame[name] = to_float(frame[name])
    check_states(frame[label_names].to_numpy(), "space B")
    return frame.drop(columns="group_id")


def label_matrix(records: pd.DataFrame) -> np.ndarray:
    """
    Return the ``[records, 14]`` label states.

    Parameters
    ----------
    records : pd.DataFrame
        Output of ``load_records``.

    Returns
    -------
    np.ndarray
        1.0 positive, 0.0 negative, NaN undefined.
    """
    return records[[f"b_{name}" for name in CONDITIONS]].to_numpy(dtype=np.float64)


def feature_rows(name: str, positions: np.ndarray, features: Path = FEATURES) -> np.ndarray:
    """
    Rows of one MLC-02 feature array, read through a memory map.

    Parameters
    ----------
    name : str
        ``xecg``, ``cpc`` or ``measurements``.
    positions : np.ndarray
        Strictly increasing feature rows (``position`` of ``load_records``).
    features : Path
        Feature directory.

    Returns
    -------
    np.ndarray
        Float64 ``[len(positions), width]``.

    Raises
    ------
    ValueError
        If the positions are not strictly increasing or the array is not one row per ``rows.csv`` row.
    """
    positions = np.asarray(positions, dtype=np.int64)
    if len(positions) > 1 and np.any(np.diff(positions) <= 0):
        raise ValueError("feature positions must be strictly increasing")
    receipt = json.loads((features / "receipt.json").read_text())
    array = np.load(features / f"{name}.npy", mmap_mode="r")
    if array.shape[0] != receipt["records"]:
        raise ValueError(f"{name}.npy has {array.shape[0]} rows, the receipt {receipt['records']}")
    return np.asarray(array[positions], dtype=np.float64)


def measurement_names(features: Path = FEATURES) -> list[str]:
    """
    Column names of ``measurements.npy``.

    Parameters
    ----------
    features : Path
        Feature directory.

    Returns
    -------
    list[str]
        The 130 names of the feature receipt.
    """
    return list(json.loads((features / "receipt.json").read_text())["measurement_columns"])


def load_flags(records: pd.DataFrame, path: Path = FLAGS_TABLE) -> pd.DataFrame:
    """
    MLC-03 signal flags of the records, in their order.

    Parameters
    ----------
    records : pd.DataFrame
        Output of ``load_records``.
    path : Path
        ``flags_train_validation.parquet`` (train and validation candidates only).

    Returns
    -------
    pd.DataFrame
        ``RELIABLE_FLAGS`` as bool, plus ``reversal_ra_ll``, ``rhythm_unmeasurable`` and ``any_reliable``.

    Raises
    ------
    ValueError
        If the table holds another role or misses a record.
    """
    columns = ["record_key", "role", *RELIABLE_FLAGS, "reversal_ra_ll", "rhythm_unmeasurable"]
    table = pd.read_parquet(path, columns=columns)
    check_open_roles(tuple(table["role"].unique()))
    table = table.set_index("record_key").reindex(records["record_key"].to_numpy())
    if table["role"].isna().any() or not np.array_equal(table["role"].to_numpy(), records["role"].to_numpy()):
        raise ValueError("the flags table does not cover the records")
    flags = table.drop(columns="role").astype(bool).reset_index(drop=True)
    return flags.assign(any_reliable=flags[list(RELIABLE_FLAGS)].any(axis=1))


def record_statements(source: str, raw_codes: str, heart_axis: str) -> list[str]:
    """
    Raw statements of one record as ``<vocabulary>:<code>``.

    Parameters
    ----------
    source : str
        Manifest source (not SPH).
    raw_codes : str
        Manifest ``raw_codes``: a PTB-XL ``scp_codes`` dictionary or comma-separated SNOMED CT codes.
    heart_axis : str
        PTB-XL ``heart_axis`` (empty when missing or for other sources).

    Returns
    -------
    list[str]
        Statements in a stable order; PTB-XL's axis enters as ``scp:axis:X`` unless missing.

    Raises
    ------
    ValueError
        If the source's vocabulary is not SCP or SNOMED CT.
    """
    vocabulary = mlc_labels.VOCABULARY[source]
    if vocabulary == "scp":
        codes = mlc_labels.ptbxl_codes(ast.literal_eval(raw_codes) if raw_codes else {}, heart_axis)
        names = [code for code in codes if code != mlc_labels.AXIS_MISSING]
    elif vocabulary == "snomed":
        names = sorted(mlc_labels.parse_snomed(raw_codes))
    else:
        raise ValueError(f"KEY-06 does not read {source}")
    return [f"{vocabulary}:{code}" for code in names]


def inside_statements() -> frozenset[str]:
    """
    Statements that are positive codes of one of the 14 conditions.

    Returns
    -------
    frozenset[str]
        ``<vocabulary>:<code>`` of every positive SCP and SNOMED CT code of space B.
    """
    return frozenset(f"{vocabulary}:{code}" for name in mlc_labels.SPACE_B
                     for vocabulary, codes in mlc_labels.CONDITIONS[name]["positive"].items()
                     if vocabulary in ("scp", "snomed") for code in codes)


def statement_matrix(records: pd.DataFrame) -> tuple[sparse.csr_matrix, list[str]]:
    """
    Binary matrix of the statements outside the 14 conditions.

    Parameters
    ----------
    records : pd.DataFrame
        Output of ``load_records``.

    Returns
    -------
    tuple[sparse.csr_matrix, list[str]]
        ``[records, statements]`` int8 matrix and the sorted statement names.
    """
    lists = [record_statements(source, codes, axis) for source, codes, axis
             in zip(records["source"], records["raw_codes"], records["heart_axis"], strict=True)]
    names = sorted({name for items in lists for name in items} - inside_statements())
    column = {name: index for index, name in enumerate(names)}
    pairs = [(row, column[name]) for row, items in enumerate(lists) for name in set(items) if name in column]
    rows, cols = (np.array(values, dtype=np.int64) for values in zip(*pairs, strict=True)) if pairs else \
        (np.zeros(0, np.int64), np.zeros(0, np.int64))
    shape = (len(lists), len(names))
    matrix = sparse.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=shape)
    return matrix, names


def statement_mapping_note(name: str) -> str:
    """
    Where a statement sits in the frozen label code, if anywhere.

    Parameters
    ----------
    name : str
        ``<vocabulary>:<code>``.

    Returns
    -------
    str
        For example ``"space C positive: complete_lbbb"`` or ``"ambiguous: lvh"``; empty when unmapped.
    """
    vocabulary, code = name.split(":", 1)
    notes = []
    for condition, definition in mlc_labels.CONDITIONS.items():
        if code in definition["positive"][vocabulary] and condition not in mlc_labels.SPACE_B:
            notes.append(f"space C positive: {condition}")
        if code in definition["ambiguous"][vocabulary]:
            notes.append(f"ambiguous: {condition}")
    return "; ".join(notes)


def statement_descriptions(names: list[str]) -> dict[str, str]:
    """
    Official description of each statement.

    Parameters
    ----------
    names : list[str]
        ``<vocabulary>:<code>`` names.

    Returns
    -------
    dict[str, str]
        Name to description (``scp_statements.csv``, the official SNOMED CT tables, or ``heart_axis = X``).
    """
    scp = pd.read_csv(SCP_STATEMENTS, index_col=0)["description"].to_dict()
    snomed = challenge_labels.load_official()["name"].to_dict()
    result = {}
    for name in names:
        vocabulary, code = name.split(":", 1)
        if code.startswith("axis:"):
            result[name] = f"heart_axis = {code[5:]}"
        else:
            result[name] = str((scp if vocabulary == "scp" else snomed).get(code, ""))
    return result


def standardize(train: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    """
    Standardize columns with the train mean and standard deviation.

    Parameters
    ----------
    train : np.ndarray
        Train rows.
    *others : np.ndarray
        Other rows, transformed with the train statistics.

    Returns
    -------
    list[np.ndarray]
        Train first, then the others.
    """
    mean, scale = train.mean(axis=0), train.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    return [(values - mean) / scale for values in (train, *others)]


def spline_features(train: np.ndarray, validation: np.ndarray,
                    knots: int = SPLINE_KNOTS) -> tuple[np.ndarray, np.ndarray]:
    """
    Cubic spline basis of one variable with a missing indicator, fitted on train.

    Parameters
    ----------
    train, validation : np.ndarray
        Values with NaN for missing.
    knots : int
        Knots at train quantiles.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Basis columns and the missing indicator, for train and validation.
    """
    from sklearn.preprocessing import SplineTransformer

    finite = train[np.isfinite(train)]
    low, high = np.percentile(finite, [0.5, 99.5])
    fill = float(np.median(finite))

    def prepared(values: np.ndarray) -> np.ndarray:
        return np.clip(np.where(np.isfinite(values), values, fill), low, high)[:, None]

    spline = SplineTransformer(n_knots=knots, degree=3, knots="quantile", include_bias=False)
    spline.fit(prepared(train))
    return tuple(np.column_stack([spline.transform(prepared(values)), (~np.isfinite(values)).astype(float)])
                 for values in (train, validation))


def normal_scores(train: np.ndarray, validation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Map each column to normal scores by its train quantiles; NaN stays NaN.

    Parameters
    ----------
    train, validation : np.ndarray
        ``[records, columns]`` values.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Transformed train and validation values.
    """
    from sklearn.preprocessing import QuantileTransformer

    transform = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=None,
                                    random_state=SEED).fit(train)
    return transform.transform(train), transform.transform(validation)


def indicators(values: np.ndarray, categories: tuple[str, ...]) -> np.ndarray:
    """
    Build indicator columns of all but the first category.

    Parameters
    ----------
    values : np.ndarray
        Category of each record.
    categories : tuple[str, ...]
        Every category, the reference first.

    Returns
    -------
    np.ndarray
        Float ``[records, len(categories) - 1]``.

    Raises
    ------
    ValueError
        If a value is not a listed category.
    """
    values = np.asarray(values, dtype=str)
    if not np.isin(values, categories).all():
        raise ValueError(f"values outside {categories}")
    return np.column_stack([values == category for category in categories[1:]]).astype(float)


def factor_blocks(train: dict[str, Any], validation: dict[str, Any],
                  names: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Design matrices of the six factor groups, fitted on train.

    Parameters
    ----------
    train, validation : dict[str, Any]
        ``source``, ``labels`` ([n, 14]), ``measurements`` ([n, 130]), ``flags`` (DataFrame), ``age`` and
        ``sex`` of each split.
    names : list[str]
        Measurement column names.

    Returns
    -------
    dict[str, tuple[np.ndarray, np.ndarray]]
        Group name (``FACTOR_GROUPS``) to its train and validation columns.
    """
    rate = names.index("hr_bpm")
    others = [index for index, name in enumerate(names) if name not in RATE_COLUMNS]
    scores = normal_scores(train["measurements"][:, others], validation["measurements"][:, others])
    missing = np.isnan(train["measurements"][:, others]).any(axis=0)
    measurement = tuple(np.column_stack([np.nan_to_num(score, nan=0.0),
                                         np.isnan(split["measurements"][:, others])[:, missing]])
                        for score, split in zip(scores, (train, validation), strict=True))
    age = spline_features(train["age"], validation["age"])
    return {
        "source": tuple(indicators(split["source"], VALIDATION_SOURCES) for split in (train, validation)),
        "labels": tuple((split["labels"] == 1.0).astype(float) for split in (train, validation)),
        "heart_rate": spline_features(train["measurements"][:, rate], validation["measurements"][:, rate]),
        "measurements": measurement,
        "signal_flags": tuple(split["flags"][list(RELIABLE_FLAGS)].to_numpy(dtype=float)
                              for split in (train, validation)),
        "age_sex": tuple(np.column_stack([spline, indicators(split["sex"], ("F", "M", "unknown"))])
                         for spline, split in zip(age, (train, validation), strict=True)),
    }


def ols_fit(design: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Ordinary least squares with an intercept (minimum-norm solution when columns are collinear).

    Parameters
    ----------
    design : np.ndarray
        ``[records, columns]``.
    targets : np.ndarray
        ``[records, outputs]``.

    Returns
    -------
    np.ndarray
        ``[columns + 1, outputs]``, intercept first.
    """
    full = np.column_stack([np.ones(len(design)), design])
    coefficients, *_ = np.linalg.lstsq(full, targets, rcond=None)
    return coefficients


def ols_predict(design: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """
    Predictions of ``ols_fit`` coefficients.

    Parameters
    ----------
    design : np.ndarray
        ``[records, columns]``.
    coefficients : np.ndarray
        Output of ``ols_fit``.

    Returns
    -------
    np.ndarray
        ``[records, outputs]``.
    """
    return coefficients[0] + design @ coefficients[1:]


def variance_r2(targets: np.ndarray, predictions: np.ndarray) -> float:
    """
    Variance-weighted R^2 over all outputs, around the targets' own mean.

    Parameters
    ----------
    targets, predictions : np.ndarray
        ``[records, outputs]``.

    Returns
    -------
    float
        One minus the residual over the total sum of squares.
    """
    return float(1.0 - ((targets - predictions) ** 2).sum() / ((targets - targets.mean(axis=0)) ** 2).sum())


def pca_fit(train: np.ndarray, share: float = VARIANCE_KEPT) -> dict[str, Any]:
    """
    Principal components of train rows from the covariance eigendecomposition.

    Parameters
    ----------
    train : np.ndarray
        ``[records, dimensions]``.
    share : float
        Variance share the kept components reach.

    Returns
    -------
    dict[str, Any]
        ``mean``, ``components`` (all, ``[dimensions, dimensions]``, by decreasing variance), ``ratio``
        (variance share of each) and ``kept`` (smallest count reaching ``share``).
    """
    mean = train.mean(axis=0)
    centred = train - mean
    values, vectors = np.linalg.eigh(centred.T @ centred / (len(train) - 1))
    order = np.argsort(values)[::-1]
    values, vectors = np.clip(values[order], 0.0, None), vectors[:, order]
    signs = np.sign(vectors[np.argmax(np.abs(vectors), axis=0), np.arange(vectors.shape[1])])
    ratio = values / values.sum()
    kept = int(np.searchsorted(np.cumsum(ratio), share) + 1)
    vectors = vectors * np.where(signs == 0, 1.0, signs)
    return {"mean": mean, "components": vectors, "ratio": ratio, "kept": kept}


def pca_project(values: np.ndarray, pca: dict[str, Any], count: int | None = None) -> np.ndarray:
    """
    Scores on the first principal components.

    Parameters
    ----------
    values : np.ndarray
        ``[records, dimensions]``.
    pca : dict[str, Any]
        Output of ``pca_fit``.
    count : int | None
        Components; ``kept`` when None.

    Returns
    -------
    np.ndarray
        ``[records, count]``.
    """
    count = pca["kept"] if count is None else count
    return (values - pca["mean"]) @ pca["components"][:, :count]


def variance_accounting(blocks: dict[str, tuple[np.ndarray, np.ndarray]], train: np.ndarray,
                        validation: np.ndarray, pca: dict[str, Any], per_component: int = 10
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Share of embedding variance each factor group explains alone and after the others.

    Parameters
    ----------
    blocks : dict[str, tuple[np.ndarray, np.ndarray]]
        Output of ``factor_blocks``.
    train, validation : np.ndarray
        Standardized embeddings.
    pca : dict[str, Any]
        ``pca_fit`` of the train embeddings.
    per_component : int
        Leading components reported one by one.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Per group (and ``all``): columns, R^2 alone and unique on validation, in the full space and in the
        kept components, and R^2 alone on train. Per component and group: validation R^2 alone.
    """
    rotation = pca["components"]
    target_train, target_validation = train @ rotation, validation @ rotation
    kept = pca["kept"]

    def others(group: str) -> tuple[str, ...]:
        return tuple(name for name in FACTOR_GROUPS if name != group)

    def fit(groups: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
        design = [np.column_stack([blocks[group][split] for group in groups]) for split in (0, 1)]
        coefficients = ols_fit(design[0], target_train)
        return ols_predict(design[0], coefficients), ols_predict(design[1], coefficients)

    fits = {(group,): fit((group,)) for group in FACTOR_GROUPS}
    fits[FACTOR_GROUPS] = fit(FACTOR_GROUPS)
    for group in FACTOR_GROUPS:
        fits[others(group)] = fit(others(group))
    score = {groups: (variance_r2(target_validation, pred[1]),
                      variance_r2(target_validation[:, :kept], pred[1][:, :kept]),
                      variance_r2(target_train, pred[0])) for groups, pred in fits.items()}
    rows = []
    for group in (*FACTOR_GROUPS, "all"):
        alone = score[FACTOR_GROUPS if group == "all" else (group,)]
        without = score[others(group)] if group != "all" else (0.0, 0.0, 0.0)
        everything = score[FACTOR_GROUPS]
        members = FACTOR_GROUPS if group == "all" else (group,)
        columns = sum(blocks[name][0].shape[1] for name in members)
        rows.append({"group": group, "columns": columns, "r2_alone_validation": alone[0],
                     "r2_unique_validation": everything[0] - without[0],
                     "r2_alone_validation_kept": alone[1],
                     "r2_unique_validation_kept": everything[1] - without[1],
                     "r2_alone_train": alone[2], "kept_components": kept})
    components = []
    for index in range(per_component):
        column = target_validation[:, index]
        total = ((column - column.mean()) ** 2).sum()
        components.append({"component": index + 1, "variance_share": float(pca["ratio"][index]),
                           **{group: 1.0 - float(((column - fits[(group,)][1][:, index]) ** 2).sum() / total)
                              for group in FACTOR_GROUPS}})
    return pd.DataFrame(rows), pd.DataFrame(components)


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    """
    Area under the ROC curve from ranks (ties averaged).

    Parameters
    ----------
    y : np.ndarray
        Binary targets.
    score : np.ndarray
        Scores.

    Returns
    -------
    float
        AUROC, NaN when a class is absent.
    """
    y = np.asarray(y, dtype=bool)
    positives, negatives = int(y.sum()), int((~y).sum())
    if not positives or not negatives:
        return math.nan
    ranks = rankdata(score)
    return float((ranks[y].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def within_source_auroc(y: np.ndarray, score: np.ndarray,
                        strata: np.ndarray) -> tuple[float, dict[str, float]]:
    """
    Stratified AUROC: the probability that a positive scores above a negative of the same stratum.

    Parameters
    ----------
    y : np.ndarray
        Binary targets.
    score : np.ndarray
        Scores.
    strata : np.ndarray
        Stratum (source) of each record.

    Returns
    -------
    tuple[float, dict[str, float]]
        The pooled value (per-stratum AUROC weighted by positives times negatives; NaN when no stratum has
        both classes) and each stratum's AUROC.
    """
    y = np.asarray(y, dtype=bool)
    numerator, denominator, by_stratum = 0.0, 0, {}
    for stratum in np.unique(strata):
        mask = strata == stratum
        value = auroc(y[mask], score[mask])
        by_stratum[str(stratum)] = value
        if np.isfinite(value):
            pairs = int(y[mask].sum()) * int((~y[mask]).sum())
            numerator, denominator = numerator + value * pairs, denominator + pairs
    return (numerator / denominator if denominator else math.nan), by_stratum


def within_source_r2(y: np.ndarray, prediction: np.ndarray, strata: np.ndarray) -> float:
    """
    R^2 around each stratum's own mean.

    Parameters
    ----------
    y, prediction : np.ndarray
        Targets and predictions.
    strata : np.ndarray
        Stratum of each record.

    Returns
    -------
    float
        One minus the residual sum of squares over the within-stratum total sum of squares.
    """
    total = sum(float(((y[strata == s] - y[strata == s].mean()) ** 2).sum()) for s in np.unique(strata))
    return float(1.0 - ((y - prediction) ** 2).sum() / total)


def bootstrap_weights(groups: np.ndarray, draws: int = DRAWS, seed: int = SEED) -> np.ndarray:
    """
    Record counts of whole-patient bootstrap draws, by the ``intervals`` convention.

    Parameters
    ----------
    groups : np.ndarray
        Patient (or record) group of each record.
    draws : int
        Number of draws.
    seed : int
        Seed of ``numpy.random.default_rng``.

    Returns
    -------
    np.ndarray
        Int32 ``[draws, records]``: how often each record is drawn.
    """
    blocks = patient_groups(np.asarray(groups))
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(patient_resample(blocks, rng), minlength=len(groups)).astype(np.int32)
                     for _ in range(draws)])


def within_source_auroc_interval(y: np.ndarray, score: np.ndarray, strata: np.ndarray, groups: np.ndarray,
                                 draws: int = DRAWS, seed: int = SEED) -> dict[str, Any]:
    """
    Within-source AUROC with a whole-patient bootstrap interval.

    Parameters
    ----------
    y, score, strata, groups : np.ndarray
        Targets, scores, source and patient group of each record.
    draws : int
        Bootstrap draws (0 for the point value only).
    seed : int
        Seed of the draws.

    Returns
    -------
    dict[str, Any]
        ``auroc``, ``ci_low``, ``ci_high``, ``draws``, ``skipped_draws``, ``seed`` and ``by_source``.
    """
    value, by_source = within_source_auroc(y, score, strata)
    values = []
    if draws:
        rng = np.random.default_rng(seed)
        blocks = patient_groups(np.asarray(groups))
        for _ in range(draws):
            rows = patient_resample(blocks, rng)
            drawn, _ = within_source_auroc(y[rows], score[rows], strata[rows])
            if np.isfinite(drawn):
                values.append(drawn)
    low, high = (np.percentile(values, [2.5, 97.5]) if values else (math.nan, math.nan))
    return {"auroc": value, "ci_low": float(low), "ci_high": float(high), "draws": draws,
            "skipped_draws": draws - len(values), "seed": seed, "by_source": by_source}


def mantel_haenszel(a: np.ndarray, members: np.ndarray, positives: np.ndarray, total: np.ndarray
                    ) -> dict[str, np.ndarray]:
    """
    Mantel-Haenszel odds ratio and Cochran-Mantel-Haenszel test over strata on the first axis.

    Cells per stratum: ``a`` members with the feature, ``members - a`` members without it, ``positives - a``
    non-members with it and the rest. A stratum counts only when it has members and non-members, and records
    with and without the feature. The odds ratio adds 0.5 to the four cells of a counting stratum with an
    empty cell; the test uses the counts as they are, without continuity correction.

    Parameters
    ----------
    a : np.ndarray
        ``[strata, ...]`` members with the feature.
    members, positives, total : np.ndarray
        Members, records with the feature and records per stratum, broadcastable to ``a``.

    Returns
    -------
    dict[str, np.ndarray]
        ``odds_ratio`` (corrected), ``log_odds_ratio``, ``statistic``, ``p_value`` (NaN where no stratum
        counts) and ``strata`` (counting strata).
    """
    a, members, positives, total = np.broadcast_arrays(*(np.asarray(value, dtype=np.float64)
                                                         for value in (a, members, positives, total)))
    b, c = members - a, positives - a
    d = total - members - positives + a
    counts = (members > 0) & (members < total) & (positives > 0) & (positives < total)
    empty = counts & ((a == 0) | (b == 0) | (c == 0) | (d == 0))
    shift = np.where(empty, 0.5, 0.0)
    ac, bc, cc, dc = a + shift, b + shift, c + shift, d + shift
    nc = np.where(counts, total + 4 * shift, 1.0)
    numerator = np.where(counts, ac * dc / nc, 0.0).sum(axis=0)
    denominator = np.where(counts, bc * cc / nc, 0.0).sum(axis=0)
    safe = np.where(counts, total, 2.0)
    expected = np.where(counts, members * positives / safe, 0.0)
    variance = np.where(counts, members * (safe - members) * positives * (safe - positives)
                        / (safe ** 2 * (safe - 1)), 0.0)
    observed = np.where(counts, a, 0.0)
    tested = counts.any(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        odds = np.where(tested, numerator / denominator, np.nan)
        difference = observed.sum(axis=0) - expected.sum(axis=0)
        statistic = np.where(tested, difference ** 2 / variance.sum(axis=0), np.nan)
    return {"odds_ratio": odds, "log_odds_ratio": np.log(odds), "statistic": statistic,
            "p_value": np.where(np.isfinite(statistic), chi2.sf(statistic, 1), np.nan),
            "strata": counts.sum(axis=0)}


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg adjusted p-values (q-values); NaN entries are not tests and stay NaN.

    Parameters
    ----------
    p_values : np.ndarray
        P-values of any shape.

    Returns
    -------
    np.ndarray
        Adjusted values of the same shape.
    """
    flat = np.asarray(p_values, dtype=np.float64).ravel()
    result = np.full_like(flat, np.nan)
    tested = np.flatnonzero(np.isfinite(flat))
    if len(tested):
        order = tested[np.argsort(flat[tested], kind="stable")]
        ranked = flat[order] * len(tested) / np.arange(1, len(tested) + 1)
        result[order] = np.minimum(np.minimum.accumulate(ranked[::-1])[::-1], 1.0)
    return result.reshape(np.shape(p_values))


def stratum_codes(sources: np.ndarray) -> np.ndarray:
    """
    Integer stratum of each record in ``VALIDATION_SOURCES`` order.

    Parameters
    ----------
    sources : np.ndarray
        Source of each record.

    Returns
    -------
    np.ndarray
        Int64 codes 0-5.
    """
    lookup = {source: index for index, source in enumerate(VALIDATION_SOURCES)}
    return np.array([lookup[source] for source in sources], dtype=np.int64)


def enrichment_tables(membership: np.ndarray, features: sparse.spmatrix | np.ndarray, strata: np.ndarray,
                      defined: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """
    Stratified enrichment of every feature in every cluster.

    Parameters
    ----------
    membership : np.ndarray
        Cluster index of each record (-1 for none, such as HDBSCAN noise).
    features : sparse.spmatrix | np.ndarray
        ``[records, features]`` binary features.
    strata : np.ndarray
        ``stratum_codes`` of each record.
    defined : np.ndarray | None
        ``[records, features]`` mask of records where a feature is defined; all when None.

    Returns
    -------
    dict[str, np.ndarray]
        ``mantel_haenszel`` output with shape ``[clusters, features]``, plus ``a`` (members with the
        feature), ``members`` (defined members) and ``positives_out`` (non-members with the feature), summed
        over strata.
    """
    clusters = int(membership.max()) + 1
    features = sparse.csr_matrix(features, dtype=np.float64)
    defined = np.ones(features.shape, dtype=bool) if defined is None else np.asarray(defined, dtype=bool)
    shape = (len(VALIDATION_SOURCES), clusters, features.shape[1])
    a, members, positives, total = (np.zeros(shape) for _ in range(4))
    for stratum in range(len(VALIDATION_SOURCES)):
        rows = np.flatnonzero(strata == stratum)
        if not len(rows):
            continue
        onehot = np.zeros((len(rows), clusters))
        inside = membership[rows] >= 0
        onehot[np.flatnonzero(inside), membership[rows][inside]] = 1.0
        mask = defined[rows].astype(np.float64)
        values = features[rows].toarray() * mask
        a[stratum] = onehot.T @ values
        members[stratum] = onehot.T @ mask
        positives[stratum] = values.sum(axis=0)[None, :]
        total[stratum] = mask.sum(axis=0)[None, :]
    result = mantel_haenszel(a, members, positives, total)
    return {**result, "a": a.sum(axis=0), "members": members.sum(axis=0),
            "positives_out": (positives - a).sum(axis=0)}


def bootstrap_log_odds(weights: np.ndarray, member: np.ndarray, feature: np.ndarray, strata: np.ndarray,
                       defined: np.ndarray | None = None) -> np.ndarray:
    """
    Corrected Mantel-Haenszel log odds ratio of one feature in one cluster in every bootstrap draw.

    Parameters
    ----------
    weights : np.ndarray
        Output of ``bootstrap_weights`` over the same records.
    member, feature : np.ndarray
        Boolean membership and feature of each record.
    strata : np.ndarray
        ``stratum_codes`` of each record.
    defined : np.ndarray | None
        Records where the feature is defined; all when None.

    Returns
    -------
    np.ndarray
        One log odds ratio per draw (NaN where no stratum counts).
    """
    defined = np.ones(len(member), dtype=bool) if defined is None else np.asarray(defined, dtype=bool)
    onehot = (strata[:, None] == np.arange(len(VALIDATION_SOURCES))[None, :]) & defined[:, None]
    cells = np.column_stack([onehot * member[:, None] * feature[:, None], onehot * member[:, None],
                             onehot * feature[:, None], onehot]).astype(np.float64)
    sums = (np.asarray(weights, dtype=np.float64) @ cells).reshape(len(weights), 4, len(VALIDATION_SOURCES))
    result = mantel_haenszel(*(np.moveaxis(sums[:, part, :], 1, 0) for part in range(4)))
    return result["log_odds_ratio"]


def logistic_scores(train_x: np.ndarray, train_y: np.ndarray, validation_x: np.ndarray,
                    c: float = LOGISTIC_C) -> tuple[np.ndarray, int]:
    """
    Fit a logistic regression on one thread and score validation records.

    Parameters
    ----------
    train_x, validation_x : np.ndarray
        Standardized features.
    train_y : np.ndarray
        Binary train targets.
    c : float
        Inverse regularization strength.

    Returns
    -------
    tuple[np.ndarray, int]
        Validation decision values and the solver's iterations.
    """
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits

    with threadpool_limits(1):
        model = LogisticRegression(C=c, max_iter=3000, random_state=SEED).fit(train_x, train_y.astype(int))
        return model.decision_function(validation_x), int(model.n_iter_[0])


def binary_probe(name: str, train_x: np.ndarray, train_y: np.ndarray, validation_x: np.ndarray,
                 validation_y: np.ndarray, strata: np.ndarray, groups: np.ndarray, draws: int = 0,
                 seed: int = SEED) -> dict[str, Any]:
    """
    Logistic probe of one binary factor, read by within-source AUROC on validation.

    Parameters
    ----------
    name : str
        Factor name.
    train_x, validation_x : np.ndarray
        Standardized features.
    train_y, validation_y : np.ndarray
        Targets with NaN for undefined records, which are left out.
    strata, groups : np.ndarray
        Source and patient group of each validation record.
    draws : int
        Bootstrap draws of the interval (0 for none).
    seed : int
        Seed of the draws.

    Returns
    -------
    dict[str, Any]
        ``name``, counts, ``iterations`` and the ``within_source_auroc_interval`` fields.
    """
    fit_rows, read_rows = np.isfinite(train_y), np.isfinite(validation_y)
    scores, iterations = logistic_scores(train_x[fit_rows], train_y[fit_rows], validation_x[read_rows])
    y = validation_y[read_rows].astype(bool)
    result = within_source_auroc_interval(y, scores, strata[read_rows], groups[read_rows], draws, seed)
    return {"name": name, "train_positives": int(np.nansum(train_y)), "train_records": int(fit_rows.sum()),
            "validation_positives": int(y.sum()), "validation_records": int(read_rows.sum()),
            "iterations": iterations, **result}


def ridge_predictions(train_x: np.ndarray, train_y: np.ndarray, validation_x: np.ndarray,
                      alpha: float = RIDGE_ALPHA) -> np.ndarray:
    """
    Ridge regression of each target on the rows where it is present.

    Targets with the same missing pattern share one solve.

    Parameters
    ----------
    train_x, validation_x : np.ndarray
        Standardized features.
    train_y : np.ndarray
        ``[records, targets]`` with NaN for missing.
    alpha : float
        Ridge penalty.

    Returns
    -------
    np.ndarray
        ``[validation records, targets]`` predictions.
    """
    present = np.isfinite(train_y)
    patterns: dict[bytes, list[int]] = {}
    for column in range(train_y.shape[1]):
        patterns.setdefault(present[:, column].tobytes(), []).append(column)
    predictions = np.full((len(validation_x), train_y.shape[1]), np.nan)
    for columns in patterns.values():
        rows = present[:, columns[0]]
        x = train_x[rows]
        mean = x.mean(axis=0)
        centred = x - mean
        y = train_y[rows][:, columns]
        offset = y.mean(axis=0)
        gram = centred.T @ centred + alpha * np.eye(x.shape[1])
        coefficients = np.linalg.solve(gram, centred.T @ (y - offset))
        predictions[:, columns] = offset + (validation_x - mean) @ coefficients
    return predictions


def continuous_probes(names: list[str], train_x: np.ndarray, train_y: np.ndarray, validation_x: np.ndarray,
                      validation_y: np.ndarray, strata: np.ndarray) -> pd.DataFrame:
    """
    Within-source R^2 on validation of ridge predictions of continuous factors.

    Parameters
    ----------
    names : list[str]
        Target names.
    train_x, validation_x : np.ndarray
        Standardized features.
    train_y, validation_y : np.ndarray
        ``[records, targets]`` with NaN for missing.
    strata : np.ndarray
        Source of each validation record.

    Returns
    -------
    pd.DataFrame
        ``name``, ``within_source_r2``, ``r2`` (around the pooled mean) and the records used.
    """
    predictions = ridge_predictions(train_x, train_y, validation_x)
    rows = []
    for column, name in enumerate(names):
        present = np.isfinite(validation_y[:, column])
        y, prediction = validation_y[present, column], predictions[present, column]
        rows.append({"name": name, "within_source_r2": within_source_r2(y, prediction, strata[present]),
                     "r2": float(1 - ((y - prediction) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
                     "validation_records": int(present.sum()),
                     "train_records": int(np.isfinite(train_y[:, column]).sum())})
    return pd.DataFrame(rows)


def source_probe(train_x: np.ndarray, train_source: np.ndarray, validation_x: np.ndarray,
                 validation_source: np.ndarray) -> dict[str, float]:
    """
    Multinomial logistic regression of the source, read on validation.

    Parameters
    ----------
    train_x, validation_x : np.ndarray
        Standardized features.
    train_source, validation_source : np.ndarray
        Source of each record.

    Returns
    -------
    dict[str, float]
        ``accuracy``, ``balanced_accuracy``, ``majority_share`` and ``iterations``.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score

    model = LogisticRegression(C=LOGISTIC_C, max_iter=3000, random_state=SEED).fit(train_x, train_source)
    predicted = model.predict(validation_x)
    _, counts = np.unique(validation_source, return_counts=True)
    return {"accuracy": float((predicted == validation_source).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(validation_source, predicted)),
            "majority_share": float(counts.max() / counts.sum()), "iterations": int(model.n_iter_.max())}


def residual_space(blocks: dict[str, tuple[np.ndarray, np.ndarray]], train: np.ndarray,
                   validation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove the linear effects of source, heart rate, signal flags, age and sex, fitted on train.

    Parameters
    ----------
    blocks : dict[str, tuple[np.ndarray, np.ndarray]]
        Output of ``factor_blocks``.
    train, validation : np.ndarray
        Standardized embeddings.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Train and validation residuals and the ``ols_fit`` coefficients.
    """
    design = [np.column_stack([blocks[group][split] for group in RESIDUAL_GROUPS]) for split in (0, 1)]
    coefficients = ols_fit(design[0], train)
    return (train - ols_predict(design[0], coefficients), validation - ols_predict(design[1], coefficients),
            coefficients)


def cluster_seed(method: str, replicate: int) -> int:
    """
    Seed of one clustering fit.

    Parameters
    ----------
    method : str
        One of ``METHODS``.
    replicate : int
        -1 for the full fit, 0-19 for the subsamples.

    Returns
    -------
    int
        ``SEED + 1000 * (method number) + replicate + 1``.
    """
    return SEED + 1000 * (METHODS.index(method) + 1) + replicate + 1


def fit_clusters(points: np.ndarray, method: str, seed: int) -> dict[str, Any]:
    """
    Fit k-means or HDBSCAN on one thread.

    Parameters
    ----------
    points : np.ndarray
        ``[records, components]``.
    method : str
        ``kmeans_<k>`` or ``hdbscan``.
    seed : int
        k-means seed (HDBSCAN is deterministic).

    Returns
    -------
    dict[str, Any]
        ``labels`` (int64, -1 for HDBSCAN noise) and ``centres`` (k-means only, else None).
    """
    from sklearn.cluster import HDBSCAN, KMeans
    from threadpoolctl import threadpool_limits

    with threadpool_limits(1):
        if method == "hdbscan":
            model = HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER, copy=True).fit(points)
            return {"labels": model.labels_.astype(np.int64), "centres": None}
        model = KMeans(n_clusters=int(method.split("_")[1]), n_init=KMEANS_INIT, random_state=seed)
        model.fit(points)
        return {"labels": model.labels_.astype(np.int64), "centres": model.cluster_centers_}


def subsample_indices(count: int, subsamples: int = SUBSAMPLES, share: float = SUBSAMPLE_SHARE,
                      seed: int = SEED + 7) -> np.ndarray:
    """
    Sorted record indices of the stability subsamples.

    Parameters
    ----------
    count : int
        Records in the clustered set.
    subsamples : int
        Number of subsamples.
    share : float
        Share of records in each.
    seed : int
        Seed of one generator that draws them in turn.

    Returns
    -------
    np.ndarray
        ``[subsamples, round(share * count)]``.
    """
    rng = np.random.default_rng(seed)
    size = int(round(share * count))
    return np.stack([np.sort(rng.choice(count, size=size, replace=False)) for _ in range(subsamples)])


def order_by_size(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Renumber clusters by decreasing size (ties by the old number); -1 stays -1.

    Parameters
    ----------
    labels : np.ndarray
        Cluster of each record.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        New labels and the old label of each new cluster.
    """
    ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    order = ids[np.lexsort((ids, -counts))]
    lookup = np.full(int(labels.max()) + 2 if len(ids) else 1, -1, dtype=np.int64)
    lookup[order] = np.arange(len(order))
    return np.where(labels >= 0, lookup[np.maximum(labels, 0)], -1), order


def jaccard_stability(full: np.ndarray, indices: np.ndarray, subsample: np.ndarray) -> np.ndarray:
    """
    Best Jaccard overlap of each full-solution cluster with a cluster of one subsample solution.

    Parameters
    ----------
    full : np.ndarray
        Labels of the full solution (-1 for noise).
    indices : np.ndarray
        Records of the subsample.
    subsample : np.ndarray
        Labels of the subsample solution, aligned with ``indices``.

    Returns
    -------
    np.ndarray
        One value per full cluster (0 when the subsample holds none of it or has no cluster).
    """
    clusters = int(full.max()) + 1
    inside = full[indices]
    found = int(subsample.max()) + 1 if len(subsample) else 0
    if found <= 0:
        return np.zeros(clusters)
    both = (inside >= 0) & (subsample >= 0)
    table = np.zeros((clusters, found))
    np.add.at(table, (inside[both], subsample[both]), 1.0)
    sizes = np.bincount(inside[inside >= 0], minlength=clusters).astype(float)
    other = np.bincount(subsample[subsample >= 0], minlength=found).astype(float)
    union = sizes[:, None] + other[None, :] - table
    with np.errstate(invalid="ignore", divide="ignore"):
        jaccard = np.where(union > 0, table / union, 0.0)
    return jaccard.max(axis=1)


def assign_validation(train_points: np.ndarray, train_labels: np.ndarray, centres: np.ndarray | None,
                      validation_points: np.ndarray) -> np.ndarray:
    """
    Assign validation records to the nearest train cluster.

    Parameters
    ----------
    train_points, validation_points : np.ndarray
        Points in the clustering space.
    train_labels : np.ndarray
        Train labels (renumbered).
    centres : np.ndarray | None
        k-means centres in the renumbered order; None for HDBSCAN, whose records take the label of their
        nearest train record, noise included.

    Returns
    -------
    np.ndarray
        Validation labels.
    """
    if centres is not None:
        distance = ((validation_points ** 2).sum(axis=1)[:, None] - 2 * validation_points @ centres.T
                    + (centres ** 2).sum(axis=1)[None, :])
        return np.argmin(distance, axis=1).astype(np.int64)
    from sklearn.neighbors import NearestNeighbors

    nearest = NearestNeighbors(n_neighbors=1).fit(train_points).kneighbors(validation_points,
                                                                            return_distance=False)
    return train_labels[nearest[:, 0]]


def nearest_to_centre(points: np.ndarray, labels: np.ndarray, cluster: int, count: int) -> np.ndarray:
    """
    Members closest to their cluster's mean.

    Parameters
    ----------
    points : np.ndarray
        Points in the clustering space.
    labels : np.ndarray
        Cluster of each point.
    cluster : int
        Cluster number.
    count : int
        Members returned.

    Returns
    -------
    np.ndarray
        Indices into ``points``, nearest first.
    """
    members = np.flatnonzero(labels == cluster)
    distance = ((points[members] - points[members].mean(axis=0)) ** 2).sum(axis=1)
    return members[np.argsort(distance, kind="stable")[:count]]


def family_rule(families: pd.Series) -> tuple[bool, int]:
    """
    Apply the protocol's "not one source" rule to the families of a cluster's members.

    Parameters
    ----------
    families : pd.Series
        Family of each member.

    Returns
    -------
    tuple[bool, int]
        Whether at least two families each hold 10% of the members or 50 members, and how many do.
    """
    counts = families.value_counts()
    qualifying = int(((counts >= FAMILY_SHARE * len(families)) | (counts >= FAMILY_COUNT)).sum())
    return qualifying >= 2, qualifying


def flag_rates(member: np.ndarray, flags: pd.DataFrame) -> pd.DataFrame:
    """
    Rate of each reliable flag among members and non-members.

    Parameters
    ----------
    member : np.ndarray
        Boolean membership of each record.
    flags : pd.DataFrame
        Output of ``load_flags`` for the same records.

    Returns
    -------
    pd.DataFrame
        ``flag``, ``inside``, ``members``, ``outside``, ``others`` and ``ratio`` (inside rate over outside
        rate; infinite when the flag is absent outside and present inside, NaN when absent in both).
    """
    values = flags[list(RELIABLE_FLAGS)].to_numpy(dtype=bool)
    inside, outside = values[member].sum(axis=0), values[~member].sum(axis=0)
    members, others = int(member.sum()), int((~member).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (inside / max(members, 1)) / (outside / max(others, 1))
    ratio = np.where((inside == 0) & (outside == 0), np.nan, ratio)
    return pd.DataFrame({"flag": RELIABLE_FLAGS, "inside": inside, "members": members, "outside": outside,
                         "others": others, "ratio": ratio})


def artefact_rule(rates: pd.DataFrame, minimum: int = 0) -> tuple[bool, str]:
    """
    Apply the protocol's "not an artefact" rule.

    Parameters
    ----------
    rates : pd.DataFrame
        Output of ``flag_rates``.
    minimum : int
        Flagged members a flag also needs to fail the rule (0 for the protocol's rule).

    Returns
    -------
    tuple[bool, str]
        Whether no flag is more than twice as frequent inside, and the worst flag with its ratio and count.
    """
    failing = rates[(rates["ratio"] > FLAG_RATIO) & (rates["inside"] >= minimum)]
    worst = rates.loc[rates["ratio"].fillna(-1).idxmax()]
    text = f"{worst['flag']} x{worst['ratio']:.2f} ({int(worst['inside'])}/{int(worst['members'])})"
    return failing.empty, text


def label_rule(member: np.ndarray, labels: np.ndarray) -> tuple[bool, str, float]:
    """
    Apply the protocol's "not already labelled" rule.

    Parameters
    ----------
    member : np.ndarray
        Boolean membership of each record.
    labels : np.ndarray
        ``[records, 14]`` label states.

    Returns
    -------
    tuple[bool, str, float]
        Whether no condition is positive in more than half of the members, the most frequent condition and
        its positive share (undefined members in the denominator).
    """
    shares = (labels[member] == 1.0).sum(axis=0) / max(int(member.sum()), 1)
    top = int(np.argmax(shares))
    return bool(shares[top] <= LABEL_MAJORITY), CONDITIONS[top], float(shares[top])


def median_beat_template(window: np.ndarray) -> np.ndarray:
    """
    Median beat of every lead by the measurement extractor's detector and window (-350 to +550 ms).

    Parameters
    ----------
    window : np.ndarray
        Float ``(12, 5000)`` mV, NaN for missing samples.

    Returns
    -------
    np.ndarray
        ``(12, 450)`` mV of the 0.5-40 Hz signal; NaN when fewer than two beats fit.
    """
    from scipy.signal import sosfiltfilt

    from ecg_experiment.measurement_extractor042 import SOS_CLEAN, detect_r_peaks, median_beat
    from ecg_experiment.mlc03_signal_quality import fill_missing

    clean = sosfiltfilt(SOS_CLEAN, fill_missing(window).T, axis=0)
    template, _ = median_beat(clean, detect_r_peaks(clean))
    return template.T


def read_windows(records: pd.DataFrame) -> list[np.ndarray]:
    """
    Read the cached first 10 s of train or validation records, each checked against its manifest hash.

    Parameters
    ----------
    records : pd.DataFrame
        Rows of ``load_records``.

    Returns
    -------
    list[np.ndarray]
        Float64 ``(12, 5000)`` mV windows in the records' order.
    """
    from ecg_experiment.mlc03_signal_runs import open_block, read_checked

    check_open_roles(tuple(records["role"].unique()))
    blocks: dict[str, np.ndarray] = {}
    windows = []
    for block, row, expected, key in zip(records["cache_block"], records["cache_row"], records["sha256_10s"],
                                         records["record_key"], strict=True):
        if block not in blocks:
            blocks[block] = open_block(block, allow_test=False)
        windows.append(read_checked(blocks[block], int(row), expected, key))
    return windows


def draw_beats(axis: Any, templates: list[np.ndarray], lead: int, name: str) -> None:
    """
    Overlay the median beats of one lead with their pointwise median.

    Parameters
    ----------
    axis : Any
        Matplotlib axes.
    templates : list[np.ndarray]
        ``(12, 450)`` median beats (NaN ones are skipped).
    lead : int
        Lead row.
    name : str
        Lead name.
    """
    time_ms = (np.arange(450) - 175) * 2.0
    valid = [template[lead] for template in templates if np.isfinite(template).all()]
    for values in valid:
        axis.plot(time_ms, values, lw=0.5, color="0.45", alpha=0.45)
    if valid:
        stacked = np.stack(valid)
        axis.plot(time_ms, np.median(stacked, axis=0), lw=1.4, color="black")
        low, high = np.percentile(stacked, [1, 99])
        half = max(high - low, 1.0) / 2 * 1.1
        axis.set_ylim((low + high) / 2 - half, (low + high) / 2 + half)
    for tick in range(-300, 600, 100):
        axis.axvline(tick, color="tab:blue", lw=0.3, alpha=0.4)
    axis.axhline(0.0, color="tab:blue", lw=0.4, alpha=0.5)
    axis.set_xlim(time_ms[0], time_ms[-1])
    axis.set_title(f"{name} ({len(valid)} beats; grid 100 ms)", fontsize=8, loc="left")
    axis.tick_params(labelsize=6, length=2)


def draw_cluster_sheet(windows: list[np.ndarray], captions: list[str], path: Path, title: str) -> None:
    """
    One review sheet: median beats of every lead overlaid, then a 10 s lead II strip per record.

    Parameters
    ----------
    windows : list[np.ndarray]
        ``(12, 5000)`` mV windows.
    captions : list[str]
        One caption per window.
    path : Path
        Destination PNG.
    title : str
        Sheet title.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ecg_experiment.eda.signals import LEADS
    from ecg_experiment.mlc03_signal_quality import fill_missing

    templates = [median_beat_template(window) for window in windows]
    figure = plt.figure(figsize=(14, 18))
    outer = figure.add_gridspec(2, 1, height_ratios=[0.34, 0.66], hspace=0.06, left=0.04, right=0.995,
                                top=0.95, bottom=0.015)
    beats = outer[0].subgridspec(3, 4, hspace=0.35, wspace=0.18)
    for lead, name in enumerate(LEADS):
        draw_beats(figure.add_subplot(beats[lead // 4, lead % 4]), templates, lead, name)
    strips = outer[1].subgridspec(max(len(windows), 1), 1, hspace=0.0)
    time = np.arange(5000) / 500.0
    for row, (window, caption) in enumerate(zip(windows, captions, strict=True)):
        axis = figure.add_subplot(strips[row, 0])
        values = fill_missing(window)[1]
        values = values - np.median(values)
        low, high = np.percentile(values, [0.5, 99.5])
        half = max(high - low, 1.0) / 2 * 1.1
        axis.plot(time, values, lw=0.5, color="black")
        axis.set_ylim((low + high) / 2 - half, (low + high) / 2 + half)
        axis.set_xlim(0, 10)
        for second in range(1, 10):
            axis.axvline(second, color="tab:blue", lw=0.3, alpha=0.4)
        axis.set_yticks([])
        axis.tick_params(axis="x", labelbottom=row == len(windows) - 1, labelsize=6, length=2)
        axis.text(0.002, 0.92, f"{caption} | II range {high - low:.1f} mV", transform=axis.transAxes,
                  fontsize=6.5, va="top", bbox={"facecolor": "white", "alpha": 0.7, "lw": 0, "pad": 0.5})
    figure.suptitle(title, fontsize=10, x=0.01, y=0.985, ha="left")
    save_small_png(figure, path)
    plt.close(figure)


def save_small_png(figure: Any, path: Path) -> None:
    """
    Save a figure as a 64-colour PNG to keep review sheets small.

    Parameters
    ----------
    figure : Any
        Matplotlib figure.
    path : Path
        Destination PNG.
    """
    import io

    from PIL import Image

    buffer = io.BytesIO()
    figure.savefig(buffer, dpi=80, format="png")
    buffer.seek(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(buffer).convert("RGB").quantize(colors=64)
    image.save(path, optimize=True)


def caption(record: pd.Series, measures: dict[str, float], flags: pd.Series) -> str:
    """
    Caption of one record on a review sheet.

    Parameters
    ----------
    record : pd.Series
        Row of ``load_records``.
    measures : dict[str, float]
        ``hr_bpm`` and ``qrs_dur_ms`` of the record.
    flags : pd.Series
        Row of ``load_flags``.

    Returns
    -------
    str
        Key, source, age and sex, rate, QRS duration, positive labels and flags.
    """
    positives = [name for name in CONDITIONS if record[f"b_{name}"] == 1.0]
    raised = [name for name in RELIABLE_FLAGS if flags[name]]
    age = "?" if not np.isfinite(record["age"]) else f"{record['age']:.0f}"
    return (f"{record['record_key']} | {record['source']} | {age}{record['sex'][0]} | "
            f"HR {measures['hr_bpm']:.0f} | QRS {measures['qrs_dur_ms']:.0f} ms | "
            f"labels: {', '.join(positives) or '-'} | flags: {', '.join(raised) or '-'}")


def qrs_axis(measurements: np.ndarray, names: list[str]) -> np.ndarray:
    """
    Frontal QRS axis from the net QRS areas of leads I and aVF, in degrees.

    Parameters
    ----------
    measurements : np.ndarray
        ``[records, 130]`` measurements.
    names : list[str]
        Their names.

    Returns
    -------
    np.ndarray
        ``atan2(area aVF, area I)`` in degrees, NaN where either is missing.
    """
    return np.degrees(np.arctan2(measurements[:, names.index("qrs_area_aVF")],
                                 measurements[:, names.index("qrs_area_I")]))


def within_source_differences(membership: np.ndarray, values: np.ndarray, strata: np.ndarray,
                              clusters: int) -> np.ndarray:
    """
    Standardized difference of each measurement between members and non-members, within source.

    Per source with at least two present members and two present non-members, the difference of means is
    divided by the source's standard deviation; the sources are averaged with weights of their present
    members.

    Parameters
    ----------
    membership : np.ndarray
        Cluster of each record (-1 for none).
    values : np.ndarray
        ``[records, measurements]`` normal scores with NaN for missing.
    strata : np.ndarray
        ``stratum_codes`` of each record.
    clusters : int
        Number of clusters.

    Returns
    -------
    np.ndarray
        ``[clusters, measurements]``.
    """
    present = np.isfinite(values)
    filled = np.where(present, values, 0.0)
    numerator, weight = np.zeros((clusters, values.shape[1])), np.zeros((clusters, values.shape[1]))
    for stratum in np.unique(strata):
        rows = strata == stratum
        onehot = np.zeros((int(rows.sum()), clusters))
        inside = membership[rows] >= 0
        onehot[np.flatnonzero(inside), membership[rows][inside]] = 1.0
        count_in = onehot.T @ present[rows]
        sum_in = onehot.T @ filled[rows]
        count_all, sum_all = present[rows].sum(axis=0), filled[rows].sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.nanstd(np.where(present[rows], values[rows], np.nan), axis=0)
            difference = (sum_in / count_in - (sum_all - sum_in) / (count_all - count_in)) / scale
        usable = (count_in >= 2) & (count_all - count_in >= 2) & (scale > 0)
        numerator += np.where(usable, difference * count_in, 0.0)
        weight += np.where(usable, count_in, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(weight > 0, numerator / weight, np.nan)


def knn_distance(points: np.ndarray, neighbours: int = OUTLIER_NEIGHBOURS, jobs: int = 8) -> np.ndarray:
    """
    Mean distance of each record to its nearest other records.

    Parameters
    ----------
    points : np.ndarray
        ``[records, components]``.
    neighbours : int
        Neighbours averaged.
    jobs : int
        Threads of the search.

    Returns
    -------
    np.ndarray
        One mean distance per record, itself excluded.
    """
    from sklearn.neighbors import NearestNeighbors

    distance, index = NearestNeighbors(n_neighbors=neighbours + 1, n_jobs=jobs).fit(points).kneighbors(points)
    others = index != np.arange(len(points))[:, None]
    others[others.sum(axis=1) > neighbours, -1] = False
    return np.array([row[mask][:neighbours].mean() for row, mask in zip(distance, others, strict=True)])


def statement_presence(statements: sparse.csr_matrix, names: list[str],
                       records: pd.DataFrame) -> pd.DataFrame:
    """
    Train and validation positives of every statement per source, and where it is present.

    Parameters
    ----------
    statements : sparse.csr_matrix
        Output of ``statement_matrix`` for ``records``.
    names : list[str]
        Statement names.
    records : pd.DataFrame
        Output of ``load_records``.

    Returns
    -------
    pd.DataFrame
        ``statement``, ``train_<source>`` and ``validation_<source>`` counts, ``train_positives``,
        ``present_sources`` and ``present_families`` (sources with at least five train positives, joined by
        ``;``), ``families`` (their number) and ``probe_rule`` (100 train positives and two families).
    """
    table = pd.DataFrame({"statement": names})
    for role in OPEN_ROLES:
        for source in VALIDATION_SOURCES:
            rows = ((records["role"] == role) & (records["source"] == source)).to_numpy()
            table[f"{role}_{source}"] = np.asarray(statements[rows].sum(axis=0)).ravel()
    train = table[[f"train_{source}" for source in VALIDATION_SOURCES]].to_numpy()
    present = train >= SOURCE_MIN_POSITIVES
    family = {source: mlc_labels.FAMILIES[source] for source in VALIDATION_SOURCES}
    table["train_positives"] = train.sum(axis=1)
    chosen = [[source for source, flag in zip(VALIDATION_SOURCES, row, strict=True) if flag]
              for row in present]
    table["present_sources"] = [";".join(sources) for sources in chosen]
    table["present_families"] = [";".join(sorted({family[source] for source in sources}))
                                 for sources in chosen]
    table["families"] = [len(text.split(";")) if text else 0 for text in table["present_families"]]
    table["probe_rule"] = (table["train_positives"] >= STATEMENT_MIN_PROBE) & (table["families"] >= 2)
    return table


def timed(function: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """
    Call a function and measure its wall time.

    Parameters
    ----------
    function : Any
        Callable.
    *args, **kwargs : Any
        Its arguments.

    Returns
    -------
    tuple[Any, float]
        The result and the seconds it took.
    """
    start = time.perf_counter()
    return function(*args, **kwargs), time.perf_counter() - start


def enrichment_rows(result: dict[str, np.ndarray], names: list[str], kind: str) -> pd.DataFrame:
    """
    Long table of ``enrichment_tables`` output, one row per tested cluster and feature.

    Parameters
    ----------
    result : dict[str, np.ndarray]
        Output of ``enrichment_tables``.
    names : list[str]
        Feature names.
    kind : str
        ``label`` or ``statement``.

    Returns
    -------
    pd.DataFrame
        ``cluster``, ``kind``, ``feature``, ``a``, ``members``, ``positives_out``, ``odds_ratio``,
        ``log_odds_ratio``, ``p_value`` and ``strata``, for pairs where a stratum counts.
    """
    clusters, features = result["odds_ratio"].shape
    columns = ("a", "members", "positives_out", "odds_ratio", "log_odds_ratio", "p_value", "strata")
    frame = pd.DataFrame({"cluster": np.repeat(np.arange(clusters), features), "kind": kind,
                          "feature": np.tile(np.asarray(names, dtype=object), clusters),
                          **{name: result[name].ravel() for name in columns}})
    return frame[frame["strata"] > 0].reset_index(drop=True)


def nan_median(values: np.ndarray) -> float:
    """
    Median of the finite values, NaN when there is none.

    Parameters
    ----------
    values : np.ndarray
        Values.

    Returns
    -------
    float
        The median.
    """
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if len(finite) else math.nan


def cluster_row(member: np.ndarray, records: pd.DataFrame, labels: np.ndarray, flags: pd.DataFrame,
                measurements: np.ndarray, names: list[str], axis: np.ndarray) -> dict[str, Any]:
    """
    Size, sources, rules 2, 4 and 5, labels, flags, demographics and key measurements of one cluster.

    Parameters
    ----------
    member : np.ndarray
        Boolean membership of each record (at least one member).
    records : pd.DataFrame
        Rows of ``load_records``.
    labels : np.ndarray
        ``[records, 14]`` label states.
    flags : pd.DataFrame
        ``load_flags`` rows.
    measurements : np.ndarray
        ``[records, 130]`` raw measurements.
    names : list[str]
        Measurement names.
    axis : np.ndarray
        ``qrs_axis`` of each record.

    Returns
    -------
    dict[str, Any]
        One table row.
    """
    families = records["family"][member]
    passes, qualifying = family_rule(families)
    rates = flag_rates(member, flags)
    artefact, worst = artefact_rule(rates)
    artefact_five, _ = artefact_rule(rates, FLAG_MIN_MEMBERS)
    labelled, top_label, top_share = label_rule(member, labels)
    row = {"size": int(member.sum()), "families_qualifying": qualifying, "rule_source": passes,
           "rule_artefact": artefact, "rule_artefact_min5": artefact_five, "worst_flag": worst,
           "rule_labelled": labelled, "top_label": top_label, "top_label_share": top_share,
           "age_median": nan_median(records["age"][member]),
           "female_share": float((records["sex"][member] == "F").mean()),
           "any_flag_share": float(flags["any_reliable"][member].mean()),
           "axis_median": nan_median(axis[member])}
    for prefix, values in (("family", families), ("source", records["source"][member])):
        row.update({f"{prefix}_{name}": int(count) for name, count in values.value_counts().items()})
    for name in KEY_MEASUREMENTS:
        row[f"median_{name}"] = nan_median(measurements[member, names.index(name)])
    for index, name in enumerate(CONDITIONS):
        row[f"share_{name}"] = float((labels[member, index] == 1.0).mean())
    return row


def cluster_profile(membership: np.ndarray, records: pd.DataFrame, labels: np.ndarray, flags: pd.DataFrame,
                    measurements: np.ndarray, names: list[str], clusters: int) -> pd.DataFrame:
    """
    ``cluster_row`` of every cluster, and of the noise group when present.

    Parameters
    ----------
    membership : np.ndarray
        Cluster of each record (-1 for noise).
    records : pd.DataFrame
        Rows of ``load_records``.
    labels : np.ndarray
        ``[records, 14]`` label states.
    flags : pd.DataFrame
        ``load_flags`` rows.
    measurements : np.ndarray
        ``[records, 130]`` raw measurements.
    names : list[str]
        Measurement names.
    clusters : int
        Number of clusters.

    Returns
    -------
    pd.DataFrame
        One row per cluster, ``cluster`` -1 for noise.
    """
    axis = qrs_axis(measurements, names)
    records = records.reset_index(drop=True)
    flags = flags.reset_index(drop=True)
    rows = []
    for cluster in [*range(clusters), *([-1] if (membership < 0).any() else [])]:
        member = membership == cluster
        row = cluster_row(member, records, labels, flags, measurements, names, axis) if member.any() else \
            {"size": 0}
        rows.append({"cluster": cluster, **row})
    return pd.DataFrame(rows)


def strongest_enrichment(rows: pd.DataFrame, level: float = BH_LEVEL) -> pd.Series | None:
    """
    Find the significant enrichment with the largest absolute log odds ratio.

    Parameters
    ----------
    rows : pd.DataFrame
        Train enrichment rows of one cluster with a ``q_value`` column (labels and statements).
    level : float
        Benjamini-Hochberg level.

    Returns
    -------
    pd.Series | None
        The row, or None when no enrichment is significant.
    """
    significant = rows[rows["q_value"] < level]
    if significant.empty:
        return None
    order = np.lexsort((significant["feature"].to_numpy(dtype=str),
                        -np.abs(significant["log_odds_ratio"].to_numpy())))
    return significant.iloc[order[0]]


def replication(weights: np.ndarray, member: np.ndarray, feature: np.ndarray, strata: np.ndarray,
                defined: np.ndarray, train_log_odds: float) -> dict[str, Any]:
    """
    Whether a train enrichment replicates on validation: same sign and a 95% interval that excludes 0.

    Parameters
    ----------
    weights : np.ndarray
        ``bootstrap_weights`` of the validation records.
    member, feature, defined : np.ndarray
        Validation membership, feature and defined mask.
    strata : np.ndarray
        ``stratum_codes`` of the validation records.
    train_log_odds : float
        The train log odds ratio.

    Returns
    -------
    dict[str, Any]
        ``log_odds_validation``, ``ci_low``, ``ci_high``, ``skipped_draws`` and ``replicated``.
    """
    observed = bootstrap_log_odds(np.ones((1, len(member))), member, feature, strata, defined)[0]
    draws = bootstrap_log_odds(weights, member, feature, strata, defined)
    finite = draws[np.isfinite(draws)]
    low, high = np.percentile(finite, [2.5, 97.5]) if len(finite) else (math.nan, math.nan)
    same = bool(np.isfinite(observed) and np.sign(observed) == np.sign(train_log_odds))
    excludes = bool(np.isfinite(low) and (low > 0 or high < 0))
    return {"log_odds_validation": float(observed), "ci_low": float(low), "ci_high": float(high),
            "skipped_draws": int(len(draws) - len(finite)), "replicated": same and excludes}


def verdict(row: pd.Series) -> tuple[bool, str]:
    """
    Combine the five rules of the protocol into a verdict.

    Parameters
    ----------
    row : pd.Series
        Cluster row with ``rule_stable``, ``rule_source``, ``rule_replicated``, ``rule_artefact`` and
        ``rule_labelled``.

    Returns
    -------
    tuple[bool, str]
        Whether the cluster is a candidate, and the failed rules joined by ``;`` (empty for a candidate).
    """
    names = {"rule_stable": "stable", "rule_source": "one source", "rule_replicated": "not replicated",
             "rule_artefact": "artefact", "rule_labelled": "already labelled"}
    failed = [text for column, text in names.items() if not bool(row[column])]
    return not failed, "; ".join(failed)
