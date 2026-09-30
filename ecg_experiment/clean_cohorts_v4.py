"""Cohorts v4: cohorts v3 with the usable EchoNext training ECGs ranked right after the curated sources.

Sources enter in order of quality (``docs/clean-cohorts-v4.md``):

1. The curated human-read 500 Hz sources in millivolts, exactly as in v3 (PTB-XL first, then the Challenge
   training group interleaved), so the 25k and 50k tiers equal v3's.
2. EchoNext training ECGs: echo-confirmed labels, known patients, but 250 Hz waveforms that the dataset
   authors filtered, clipped and standardized, so they have no physical unit.
3. The MIMIC top-up, only if the two blocks above fall short of 100k.
4. CODE-15, then the remaining local MIMIC records, then the pending ones, as in v2.

Only EchoNext ``train`` rows that pass the unit-free quality rules are candidates. ``val`` is an evaluation
split, ``test`` is closed and ``no_split`` shares its patients with both, so none of them is ever read here
beyond the patient and split columns needed to prove that no training patient appears in them.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import clean_cohorts_v2 as v2
from . import clean_cohorts_v3 as v3
from .clean_cohorts import UNION, table
from .cohort_tiers import SUMMARY_KEYS, TABLES, tier_sizes, tier_summary
from .cohort_tiers import check_waveforms as check_other_waveforms
from .eda.echonext import ARCHIVE, member
from .files import sha256_file, write_json_atomic
from .paths import to_stored
from .public_sources import signal_sha256

ECHONEXT = v2.ROOT / "data/processed/echonext_250hz_v1"
BACKEND = "echonext_npy"
CHECKED_PER_BACKEND = 20


def check_patients_disjoint(splits: pd.DataFrame) -> None:
    """
    Refuse EchoNext training patients that also appear in another split of the release.

    Parameters
    ----------
    splits : pd.DataFrame
        ``patient_key`` and ``split`` of every ECG of the release.

    Raises
    ------
    ValueError
        If a training patient has an ECG in ``val``, ``test`` or ``no_split``.
    """
    train = splits["split"] == "train"
    if splits.loc[train, "patient_key"].isin(set(splits.loc[~train, "patient_key"])).any():
        raise ValueError("An EchoNext training patient appears in another split")


def read_release_splits(archive: Path = ARCHIVE) -> pd.DataFrame:
    """
    Read only the ECG, patient and split columns of the EchoNext metadata, so no label is loaded.

    Parameters
    ----------
    archive : Path
        The release ZIP.

    Returns
    -------
    pd.DataFrame
        ``ecg_key``, ``patient_key`` and ``split`` as strings.
    """
    with zipfile.ZipFile(archive) as release:
        data = release.read(member("echonext_metadata_100k.csv"))
    return pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False,
                       usecols=["ecg_key", "patient_key", "split"])


def echonext_rows(directory: Path = ECHONEXT) -> pd.DataFrame:
    """
    Candidate rows of the usable EchoNext training ECGs, each with the hash of its stored waveform.

    Parameters
    ----------
    directory : Path
        The EchoNext cache (``docs/clean-sph-echonext-v1.md``) with ``rows.csv`` and ``train.npy``.

    Returns
    -------
    pd.DataFrame
        ``clean_cohorts_v2.COLUMNS`` rows. ``index`` is the row in ``train.npy``; the waveform is float32
        ``(12, 2500)`` at 250 Hz in the release's standardized values.
    """
    rows = pd.read_csv(directory / "rows.csv", dtype=str, keep_default_na=False)
    rows = rows[(rows["split"] == "train") & (rows["use"] == "True")]
    signals = np.load(directory / "train.npy", mmap_mode="r")
    return pd.DataFrame({
        "record_id": "echonext:" + rows["ecg_key"], "patient_id": "echonext:" + rows["patient_key"],
        "source": "echonext", "split": "train", "label_scope": "ssl_only", "backend": BACKEND,
        "path": to_stored(directory / "train.npy"), "index": rows["row"], "window_start": "0",
        "signal_sha256": [signal_sha256(signals[int(row)]) for row in rows["row"]],
        "quality_status": "passed", "review_flags": "", "label_available": False,
        "age": pd.to_numeric(rows["age_at_ecg"]), "male": rows["sex"].map({"male": 1.0, "female": 0.0}),
    }).reset_index(drop=True)


def global_order(curated: pd.DataFrame, echonext: pd.DataFrame, mimic_local: pd.DataFrame,
                 code15: pd.DataFrame, pending: pd.DataFrame, seed: int,
                 curated_tier: int = v2.CURATED_TIER) -> pd.DataFrame:
    """
    Put every candidate into the single selection order that defines all tiers.

    Parameters
    ----------
    curated : pd.DataFrame
        Curated rows after the Challenge exclusion.
    echonext : pd.DataFrame
        Output of ``echonext_rows``.
    mimic_local : pd.DataFrame
        Output of ``clean_cohorts_v2.mimic_local_rows``.
    code15 : pd.DataFrame
        Output of ``clean_cohorts_v2.code15_rows``.
    pending : pd.DataFrame
        Output of ``clean_cohorts_v2.mimic_pending_rows``, already in order.
    seed : int
        Selection seed.
    curated_tier : int
        Size that the MIMIC top-up completes.

    Returns
    -------
    pd.DataFrame
        ``COLUMNS`` plus ``order`` and ``block`` (``curated``, ``echonext``, ``mimic_top_up``, ``code15``,
        ``mimic_local``, ``mimic_pending``).
    """
    ptbxl = v2.source_order(curated[curated["source"] == "ptbxl"], seed)
    others = {name: v2.source_order(curated[curated["source"] == name], seed)
              for name in v2.CURATED if name != "ptbxl"}
    curated_ordered = pd.concat([ptbxl, v2.interleave(others)], ignore_index=True)
    echonext_ordered = v2.source_order(echonext, seed)
    mimic_ordered = v2.source_order(mimic_local, seed)
    top_up = max(0, curated_tier - len(curated_ordered) - len(echonext_ordered))
    blocks = [("curated", curated_ordered), ("echonext", echonext_ordered),
              ("mimic_top_up", mimic_ordered.iloc[:top_up]), ("code15", v2.source_order(code15, seed)),
              ("mimic_local", mimic_ordered.iloc[top_up:]), ("mimic_pending", pending)]
    ordered = pd.concat([rows[list(v2.COLUMNS)].assign(block=name) for name, rows in blocks],
                        ignore_index=True)
    return ordered.assign(order=np.arange(len(ordered)))


def candidate_order(seed: int, splits: pd.DataFrame, echonext: pd.DataFrame) -> pd.DataFrame:
    """
    Assemble, order and deduplicate every candidate with the v3 exclusions and EchoNext as the second block.

    Parameters
    ----------
    seed : int
        Selection seed.
    splits : pd.DataFrame
        Output of ``clean_cohorts_v3.read_splits``.
    echonext : pd.DataFrame
        Output of ``echonext_rows``.

    Returns
    -------
    pd.DataFrame
        The global order.
    """
    known, held_ids, held_hashes = v3.challenge_references(splits)
    curated = v3.exclude_challenge_evaluation(v2.curated_rows(), known, held_ids, held_hashes)
    ordered = global_order(curated, echonext, v2.mimic_local_rows(), v2.code15_rows(),
                           v2.mimic_pending_rows(seed), seed)
    ordered = ordered[~ordered["signal_sha256"].isin(held_hashes)]
    return v2.remove_overlaps(ordered, *v2.heldout_references())


def compare_with_v3(order: pd.DataFrame, previous: pd.DataFrame) -> dict[str, int]:
    """
    Check that v4 is v3 plus the EchoNext block: same curated rows, same records, same order per source.

    Parameters
    ----------
    order : pd.DataFrame
        The v4 order read back as strings.
    previous : pd.DataFrame
        The saved v3 order read as strings.

    Returns
    -------
    dict[str, int]
        Curated rows compared and records shared with v3.

    Raises
    ------
    ValueError
        If the curated block differs, or v4 without EchoNext holds other records or another order.
    """
    curated = order[order["block"] == "curated"].reset_index(drop=True)
    if not curated.equals(previous[previous["block"] == "curated"].reset_index(drop=True)):
        raise ValueError("The v4 curated block differs from v3")
    rest = order[order["source"] != "echonext"]
    for source, rows in previous.groupby("source"):
        if rest.loc[rest["source"] == source, "record_id"].tolist() != rows["record_id"].tolist():
            raise ValueError(f"v4 without EchoNext differs from v3 for {source}")
    if len(rest) != len(previous):
        raise ValueError("v4 holds a non-EchoNext record that v3 does not")
    return {"curated_rows_identical": len(curated), "records_shared_with_v3": len(rest)}


def check_waveforms(rows: pd.DataFrame, seed: int) -> dict[str, int]:
    """
    Spot-check waveform hashes: EchoNext rows here, every other backend with ``cohort_tiers``.

    Parameters
    ----------
    rows : pd.DataFrame
        Passed rows of a tier.
    seed : int
        Sampling seed.

    Returns
    -------
    dict[str, int]
        Rows checked per backend.

    Raises
    ------
    ValueError
        If a waveform differs from its manifest hash.
    """
    echo = rows["backend"] == BACKEND
    checked = check_other_waveforms(rows[~echo], seed)
    if echo.any():
        sample = rows[echo].sample(min(CHECKED_PER_BACKEND, int(echo.sum())), random_state=seed)
        for row in sample.itertuples(index=False):
            signal = np.load(v2.ROOT / row.path, mmap_mode="r")[int(row.index)]
            if signal_sha256(signal) != row.signal_sha256:
                raise ValueError(f"Waveform differs from the manifest: {row.record_id}")
        checked[BACKEND] = len(sample)
    return checked


def publish_tier(output: Path, rows: pd.DataFrame, labels: dict[str, pd.DataFrame],
                 shared: dict[str, object], seed: int) -> dict[str, object]:
    """
    Write one tier directory as ``cohort_tiers.publish_tier`` does, with the EchoNext waveform check.

    Parameters
    ----------
    output : Path
        New directory; it must not exist.
    rows : pd.DataFrame
        The tier's rows in selection order.
    labels : dict[str, pd.DataFrame]
        PTB-XL label tables keyed by budget.
    shared : dict[str, object]
        Metadata common to all tiers.
    seed : int
        Seed for the waveform spot check.

    Returns
    -------
    dict[str, object]
        The tier metadata.

    Raises
    ------
    ValueError
        If the output exists, a label record is missing from the tier, or a record or waveform repeats.
    """
    if output.exists():
        raise ValueError(f"Destination exists: {output}")
    passed = rows[rows["quality_status"] == "passed"]
    if rows["record_id"].duplicated().any() or passed["signal_sha256"].duplicated().any():
        raise ValueError("A tier repeats a record or a waveform")
    if not set(labels["1"]["record_id"]) <= set(rows["record_id"]):
        raise ValueError("A PTB-XL label record is missing from the tier")
    stage = output.with_name(output.name + ".partial")
    stage.mkdir(parents=True)
    rows.to_csv(stage / "train_manifest.csv", index=False)
    for budget, frame in labels.items():
        frame.to_csv(stage / f"labels_fraction{budget}.csv", index=False)
    metadata = shared | tier_summary(rows) | {
        "waveforms_checked": check_waveforms(passed, seed),
        "table_sha256": {name: sha256_file(stage / name) for name in TABLES},
    }
    write_json_atomic(stage / "metadata.json", metadata, sort_keys=True)
    stage.rename(output)
    return metadata


def publish_tiers(order: pd.DataFrame, sizes: dict[str, int], output_root: Path, suffix: str,
                  shared: dict[str, object], seed: int) -> dict[str, object]:
    """
    Write every feasible tier of a candidate order.

    Parameters
    ----------
    order : pd.DataFrame
        The candidate order.
    sizes : dict[str, int]
        Requested tier sizes by name.
    output_root : Path
        Parent of the tier directories.
    suffix : str
        Directory suffix, such as ``v4``.
    shared : dict[str, object]
        Metadata common to all tiers.
    seed : int
        Seed for the waveform spot checks.

    Returns
    -------
    dict[str, object]
        Composition, manifest hash and metadata hash per written tier.
    """
    clean_labeled = set(order.loc[order["source"] == "ptbxl", "record_id"])
    labels = {budget: table(UNION / f"labels_fraction{budget}.csv") for budget in ("1", "0.1")}
    labels = {budget: frame[frame["record_id"].isin(clean_labeled)] for budget, frame in labels.items()}
    written = {}
    for name, size in tier_sizes(sizes, len(order)).items():
        output = output_root / f"clean_{name}_{suffix}"
        tier = shared | {"tier": name, "size": size}
        metadata = publish_tier(output, order.iloc[:size], labels, tier, seed)
        written[name] = {key: metadata[key] for key in SUMMARY_KEYS}
        written[name]["manifest_sha256"] = metadata["table_sha256"]["train_manifest.csv"]
        written[name]["metadata_sha256"] = sha256_file(output / "metadata.json")
    return written
