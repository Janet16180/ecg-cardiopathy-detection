"""Compare the project's processed waveform arrays with the raw source files."""

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample_poly

from eda.challenge import RAW_ROOTS
from eda.ptbxl import PTBXL_DIR, ROOT
from eda.signals import canonical_order

PROCESSED_DIR = ROOT / "data/processed"
UNION_DIR = PROCESSED_DIR / "training_union_500hz_v1"
POOL_DIR = PROCESSED_DIR / "cpc_pool_40k"
CACHE_100_DIR = PROCESSED_DIR / "ptbxl/waveforms100"


def read_raw(path: str) -> np.ndarray:
    """
    Read a raw WFDB record in canonical lead order.

    Parameters
    ----------
    path : str
        Record path without extension, absolute or relative to the repository.

    Returns
    -------
    np.ndarray
        Array of shape ``(12, samples)`` in mV.
    """
    signal, fields = wfdb.rdsamp(str(ROOT / path))
    return canonical_order(signal, fields["sig_name"]).T


def union_raw_window(row: pd.Series, views: dict[str, pd.DataFrame]) -> np.ndarray:
    """
    Rebuild one union row from raw files.

    PTB-XL and MIMIC rows point straight at raw records. Challenge rows are
    looked up by record ID in their view manifest, which names the raw record
    and the window start.

    Parameters
    ----------
    row : pd.Series
        Row of ``train_manifest.csv``.
    views : dict[str, pd.DataFrame]
        Challenge view manifests indexed by ``ecg_id`` and keyed by view name.

    Returns
    -------
    np.ndarray
        Raw window of shape ``(12, 5000)``.
    """
    path, start = row["origin_path"], int(row["window_start"])
    if row["source"] not in ("ptbxl", "mimic"):
        view = views[row["view"]].loc[row["record_id"]]
        path = str((RAW_ROOTS[row["source"]] / view["filename_hr"]).relative_to(ROOT))
        start = int(view["window_start"])
    return read_raw(path)[:, start:start + 5000]


def compare_union(per_source: int = 40, seed: int = 0) -> pd.DataFrame:
    """
    Compare sampled rows of the 500 Hz union dataset with their raw windows.

    Parameters
    ----------
    per_source : int
        Rows sampled from each source.
    seed : int
        Sampling seed.

    Returns
    -------
    pd.DataFrame
        Per sampled row: source, view and the largest absolute difference in mV.
    """
    manifest = pd.read_csv(UNION_DIR / "train_manifest.csv", low_memory=False)
    sample = manifest.groupby("source").sample(per_source, random_state=seed)
    views = {view: pd.read_csv(PROCESSED_DIR / "challenge_ecg_views" / view / "manifest.csv",
                               index_col="ecg_id", low_memory=False)
             for view in ("strict_10s", "cpsc_ssl_center_crop")}
    rows = []
    for _, row in sample.iterrows():
        stored = np.load(UNION_DIR / row["shard"], mmap_mode="r")[int(row["shard_index"])]
        raw = union_raw_window(row, views)
        rows.append({"record_id": row["record_id"], "source": row["source"], "view": row["view"],
                     "max_abs_difference_mv": float(np.abs(stored - raw).max())})
    return pd.DataFrame(rows)


def ptbxl_raw_path(ecg_id: int, rate: str = "500") -> str:
    """
    Raw PTB-XL record path of one ``ecg_id``.

    Parameters
    ----------
    ecg_id : int
        PTB-XL record number.
    rate : str
        ``"500"`` or ``"100"``.

    Returns
    -------
    str
        Repository-relative record path without extension.
    """
    suffix = "hr" if rate == "500" else "lr"
    folder = f"records{rate}/{ecg_id // 1000 * 1000:05d}"
    return str((PTBXL_DIR / folder / f"{ecg_id:05d}_{suffix}").relative_to(ROOT))


def compare_pool(records: int = 60, seed: int = 0) -> pd.DataFrame:
    """
    Compare the 250 Hz frozen pool with the raw 500 Hz records.

    Each raw record is downsampled twice: the way the pool receipt describes
    (two 5-second halves resampled separately) and as one whole record. The
    difference between the two isolates artifacts created at the join.

    Parameters
    ----------
    records : int
        Rows sampled from the pool.
    seed : int
        Sampling seed.

    Returns
    -------
    pd.DataFrame
        Per sampled row: difference from the described method, and the
        halves-versus-whole difference near the join and elsewhere.
    """
    rows_table = pd.read_csv(POOL_DIR / "rows.csv")
    pool = np.load(POOL_DIR / "signals.npy", mmap_mode="r")
    mimic_manifest = PROCESSED_DIR / "mimic_ssl_200k/ssl_manifest.csv"
    mimic_paths = pd.read_csv(mimic_manifest, index_col="ecg_id")["filename_hr"]
    rows = []
    for index, row in rows_table.sample(records, random_state=seed).iterrows():
        ecg_id = str(row["ecg_id"])
        path = (f"data/raw/mimic-iv-ecg/1.0/{mimic_paths[ecg_id]}" if ecg_id.startswith("mimic:")
                else ptbxl_raw_path(int(ecg_id)))
        raw = read_raw(path)
        halves = np.concatenate([resample_poly(raw[:, :2500], 1, 2, axis=1),
                                 resample_poly(raw[:, 2500:], 1, 2, axis=1)], axis=1)
        whole = resample_poly(raw, 1, 2, axis=1)
        join = np.abs(halves - whole)[:, 1230:1270]
        interior = np.abs(halves - whole)[:, 100:1150]
        rows.append({"ecg_id": ecg_id, "source": row["source"],
                     "stored_vs_described_mv": float(np.abs(pool[index] - halves).max()),
                     "join_artifact_mv": float(join.max()),
                     "interior_difference_mv": float(interior.max())})
    return pd.DataFrame(rows)


def compare_cache_100(records: int = 200, seed: int = 0) -> pd.DataFrame:
    """
    Compare the 100 Hz PTB-XL training cache with the raw 100 Hz files.

    Parameters
    ----------
    records : int
        Cached records sampled.
    seed : int
        Sampling seed.

    Returns
    -------
    pd.DataFrame
        Per sampled record: largest absolute difference in mV.
    """
    signals = np.load(CACHE_100_DIR / "signals.npy", mmap_mode="r")
    ids = np.load(CACHE_100_DIR / "ecg_ids.npy")
    positions = np.random.default_rng(seed).choice(len(ids), records, replace=False)
    rows = []
    for position in positions:
        number = int(ids[position])
        raw = read_raw(ptbxl_raw_path(number, "100"))
        rows.append({"ecg_id": number, "max_abs_difference_mv": float(np.abs(signals[position] - raw).max())})
    return pd.DataFrame(rows)


def join_example(ecg_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Downsample one PTB-XL record whole and in two halves, as the pool did.

    Parameters
    ----------
    ecg_id : int
        PTB-XL record number.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Stored pool signal, whole-record resampling and half-by-half
        resampling, each ``(12, 2500)`` at 250 Hz.
    """
    raw = read_raw(ptbxl_raw_path(ecg_id))
    whole = resample_poly(raw, 1, 2, axis=1)
    halves = np.concatenate([resample_poly(raw[:, :2500], 1, 2, axis=1),
                             resample_poly(raw[:, 2500:], 1, 2, axis=1)], axis=1)
    rows = pd.read_csv(POOL_DIR / "rows.csv")
    position = int(np.flatnonzero(rows["ecg_id"].astype(str) == str(ecg_id))[0])
    stored = np.load(POOL_DIR / "signals.npy", mmap_mode="r")[position]
    return np.asarray(stored), whole, halves


def near_rail_windows(threshold_mv: float = 32.6) -> pd.DataFrame:
    """
    Find Challenge windows in the union dataset that contain clipped samples.

    Signed 16-bit storage with a gain of 1000/mV saturates at 32.767 mV, and
    clipped stretches often sit just below that value rather than on it.

    Parameters
    ----------
    threshold_mv : float
        Absolute amplitude treated as clipped.

    Returns
    -------
    pd.DataFrame
        One row per affected window: source, view, clipped sample count, and
        whether any sample equals the exact 32.767 mV rail.
    """
    manifest = pd.read_csv(UNION_DIR / "train_manifest.csv", low_memory=False)
    challenge = manifest[manifest["source"].isin(["georgia", "cpsc_2018", "cpsc_2018_extra"])]
    rows = []
    for shard, group in challenge.groupby("shard"):
        signals = np.load(UNION_DIR / shard, mmap_mode="r")
        for _, row in group.iterrows():
            magnitude = np.abs(np.asarray(signals[int(row["shard_index"])]))
            clipped = int((magnitude >= threshold_mv).sum())
            if clipped:
                rows.append({"record_id": row["record_id"], "source": row["source"], "view": row["view"],
                             "clipped_samples": clipped,
                             "exact_rail": bool(np.isclose(magnitude, 32.767, atol=5e-4).any())})
    return pd.DataFrame(rows)


def challenge_union_reasons(headers: pd.DataFrame, summary: pd.DataFrame, hashes: pd.Series) -> pd.Series:
    """
    Explain why each Georgia/CPSC record is absent from the union dataset.

    Reasons are assigned from our own measurements, in this order: shorter
    than 10 s, exact duplicate of a record that was kept, contains NaN
    samples, reaches the 16-bit rail, has a constant lead.

    Parameters
    ----------
    headers : pd.DataFrame
        Output of ``eda.challenge.load_headers``.
    summary : pd.DataFrame
        Output of ``eda.challenge.challenge_summary``.
    hashes : pd.Series
        Output of ``eda.challenge.signal_hashes``.

    Returns
    -------
    pd.Series
        Reason per absent record, ``unexplained`` when none applies.
    """
    union_ids = pd.read_csv(UNION_DIR / "train_manifest.csv", usecols=["record_id"])["record_id"]
    candidates = headers[headers["source"] != "chapman_shaoxing"]
    kept = candidates.index.isin(union_ids)
    absent = candidates.index[~kept]
    kept_hashes = set(hashes[candidates.index[kept]])
    checks = [
        ("shorter than 10 s", summary.loc[absent, "samples"] < 5000),
        ("exact duplicate of a kept record", hashes[absent].isin(kept_hashes)),
        ("NaN samples", summary.loc[absent, "nan_leads"] > 0),
        ("clipped at the 16-bit rail", summary.loc[absent, "peak_abs_mv"] >= 32.6),
        ("constant lead", summary.loc[absent, "constant_leads"] > 0),
    ]
    reasons = pd.Series("unexplained", index=absent)
    for reason, applies in reversed(checks):
        reasons[applies.to_numpy()] = reason
    return reasons
