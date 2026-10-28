"""Recover the frozen Challenge split from checksum-verified public raw records."""

from __future__ import annotations

import json
import os
import resource
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from ecg_experiment import challenge_labels, challenge_splits, ningbo
from ecg_experiment.eda import challenge
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.paths import to_stored
from ecg_experiment.public_sources import signal_sha256

TARGET = "5ecd82154fe9ed574b9706d9b7efbe417a578197845cf2a6a48615ca54959cc8"
RELEASES = {source: "challenge-2021/1.0.3" if source in ("ningbo", "chapman_shaoxing")
            else "challenge-2020/1.0.2" for source in challenge_splits.SOURCES}
ABSENT = {"training/ningbo/g3/JS13118.mat", "training/ningbo/g13/JS23074.mat",
          "training/ningbo/g13/S23074.hea"}
COLUMNS = ("source", "family", "record", "path", "duration_s", *ningbo.LABEL_COLUMNS,
           "signal_sha256", "window_sha256", "duplicate_group", "duplicate_status", "evaluable", "split")


def configure(source_root: str) -> None:
    """Point the frozen signal reader at the explicit shared raw directory."""
    challenge.RAW_ROOTS = {source: Path(source_root) / "data/raw" / release
                           for source, release in RELEASES.items()}


def catalog(source_root: Path, source: str) -> tuple[list[tuple], dict[str, object]]:
    """Inventory raw records and require the historical known absent-file set."""
    raw = source_root / "data/raw" / RELEASES[source]
    manifest = raw / "SHA256SUMS.txt"
    checksums = dict((name.strip(), digest) for digest, name in
                     (line.split(maxsplit=1) for line in manifest.read_text().splitlines())
                     if name.startswith(f"training/{source}/") and name.endswith((".hea", ".mat")))
    present = {path.relative_to(raw).as_posix() for path in (raw / "training" / source).rglob("*")
               if path.suffix in (".hea", ".mat")}
    absent = set(checksums) - present
    expected_absent = ABSENT if source == "ningbo" else set()
    if absent != expected_absent or present - set(checksums):
        raise ValueError(f"Unexpected file inventory for {source}: "
                         f"absent={sorted(absent ^ expected_absent)}, extra={sorted(present - set(checksums))}")
    jobs = []
    for header in sorted(name for name in present if name.endswith(".hea")):
        signal = str(Path(header).with_suffix(".mat"))
        if signal in present:
            jobs.append((source, str(raw), header, checksums[header], checksums[signal]))
    return jobs, {"sha256sums_sha256": sha256_file(manifest), "present_files": len(present),
                  "absent_files": sorted(absent), "records_with_signals": len(jobs)}


def scan(item: tuple[str, str, str, str, str]) -> dict[str, object]:
    """Verify one raw pair and recover the historical hash and header fields."""
    source, directory, relative, header_sha, mat_sha = item
    header = Path(directory) / relative
    if sha256_file(header) != header_sha or sha256_file(header.with_suffix(".mat")) != mat_sha:
        raise ValueError(f"Raw file checksum mismatch: {source}:{relative}")
    row = challenge.parse_header(header)
    path = str(Path(relative).with_suffix(""))
    key = f"{source}:{path}"
    row.update(source=source, path=path, duration_s=10.0 if source == "ningbo" else
               row["samples"] / row["fs"], signal_sha256=challenge.signal_hash(key), window_sha256="")
    if source == "ningbo":
        signal, _ = challenge.read_signal(key)
        row["window_sha256"] = signal_sha256(signal[:5000].T.astype(np.float32))
    row["worker_max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return row


def recover(source_root: Path, output: Path, workers: int, profile_records: int = 0) -> dict[str, object]:
    """Recover verified record tables and require the original serialized split hash.

    Parameters
    ----------
    source_root : Path
        Main checkout containing the raw public files.
    output : Path
        New output directory; completed per-source checkpoints permit resuming.
    workers : int
        Bounded process count, from one to four.
    profile_records : int
        Records per source for a resource profile; zero reconstructs all records.

    Returns
    -------
    dict[str, object]
        Aggregate resource, verification and reconstruction receipt.
    """
    if not 1 <= workers <= 4:
        raise ValueError("Workers must be between one and four")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metadata.json").exists():
        raise ValueError(f"Completed output exists: {output}")
    configure(str(source_root))
    official = challenge_labels.load_official(source_root / "data/raw/challenge-2021/evaluation-2021")
    start = time.monotonic()
    parts, receipts = [], {}
    for source in challenge_splits.SOURCES:
        jobs, verification = catalog(source_root, source)
        if profile_records:
            jobs = jobs[:profile_records]
        checkpoint = output / f"{source}_verified.parquet"
        before = time.monotonic()
        if checkpoint.exists():
            table = pd.read_parquet(checkpoint)
            receipt = json.loads((output / f"{source}_receipt.json").read_text())
            if receipt["checkpoint_sha256"] != sha256_file(checkpoint) or len(table) != len(jobs):
                raise ValueError(f"Checkpoint differs: {source}")
        else:
            records = []
            with Pool(workers, initializer=configure, initargs=(str(source_root),)) as pool:
                for index, row in enumerate(pool.imap(scan, jobs, chunksize=16), 1):
                    records.append(row)
                    if index % 1000 == 0 or index == len(jobs):
                        print(json.dumps({"source": source, "verified_records": index,
                                          "records": len(jobs), "elapsed_s": time.monotonic() - before}), flush=True)
            table = pd.DataFrame(records).set_index("record")
            table.to_parquet(checkpoint)
            receipt = {**verification, "verified_records": len(table), "elapsed_s": time.monotonic() - before,
                       "worker_max_rss_kib": int(table["worker_max_rss_kib"].max()),
                       "checkpoint_sha256": sha256_file(checkpoint)}
            write_json_atomic(output / f"{source}_receipt.json", receipt, sort_keys=True)
        codes = table["dx"].str.split(",").apply(lambda values: [v.strip() for v in values if v])
        labels = challenge_labels.label_table(codes, official)
        rows = table[["source", "path", "duration_s", "signal_sha256", "window_sha256"]].join(labels)
        rows["male"] = ningbo.male(table["sex"])
        parts.append(rows)
        receipts[source] = receipt
    metadata = {"complete": not bool(profile_records), "profile_records_per_source": profile_records,
                "workers": workers, "threads_per_worker": 1, "sources": receipts,
                "official_mapping_sha256": challenge_labels.MAPPING_SHA256,
                "elapsed_s": time.monotonic() - start,
                "parent_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "mode": "mechanical reconstruction; no models, scores or test evaluation",
                "source_root": "shared main checkout; paths in rows remain original release-relative",
                "code_sha256": {name: sha256_file(Path(__file__).parent.parent / name) for name in (
                    "ecg_experiment/challenge_recovery20261028.py", "ecg_experiment/challenge_splits.py",
                    "ecg_experiment/challenge_labels.py", "ecg_experiment/ningbo.py",
                    "ecg_experiment/eda/challenge.py", "ecg_experiment/public_sources.py")}}
    if profile_records:
        write_json_atomic(output / "metadata.json", metadata, sort_keys=True)
        return metadata
    rows = pd.concat(parts)
    if rows.index.duplicated().any():
        raise ValueError("A record name appears twice")
    rows["family"] = rows["source"].map(challenge_splits.FAMILIES)
    rows["label_key"] = ningbo.label_key(rows)
    rows["duplicate_status"] = ningbo.duplicate_status(rows)["duplicate_status"]
    rows["duplicate_group"] = challenge_splits.duplicate_groups(rows[["signal_sha256", "window_sha256"]])
    if rows.groupby("duplicate_group")["signal_sha256"].nunique().max() > 1:
        raise ValueError("Window hashes join different signals")
    strata = rows["source"] + ":" + challenge_splits.label_category(rows["primary"])
    rows["split"] = challenge_splits.assign_splits(rows["duplicate_group"], strata)
    if rows.groupby("duplicate_group")["split"].nunique().max() > 1:
        raise ValueError("A duplicate group crosses splits")
    rows["evaluable"] = rows["duplicate_status"].isin(["unique", "kept"]) & rows["primary"].notna()
    rows = rows.rename_axis("record").reset_index().sort_values(["source", "record"])[list(COLUMNS)]
    rows.to_csv(output / "rows.csv", index=False)
    digest = sha256_file(output / "rows.csv")
    groups = rows.groupby("duplicate_group")
    sizes = groups.size()
    metadata.update(rows_sha256=digest, target_rows_sha256=TARGET, original_rows_reproduced=digest == TARGET,
                    records=len(rows), seed=challenge_splits.SEED,
                    duplicate_groups={"groups": len(sizes), "multi_record_groups": int((sizes > 1).sum()),
                                      "records_in_multi_record_groups": int(sizes[sizes > 1].sum()),
                                      "cross_source_groups": int((groups["source"].nunique() > 1).sum())},
                    duplicate_status=rows.groupby("source")["duplicate_status"].value_counts()
                    .unstack(fill_value=0).to_dict(orient="index"),
                    counts_all=challenge_splits.split_counts(rows),
                    counts_evaluable=challenge_splits.split_counts(rows[rows["evaluable"]]),
                    ningbo_reconstruction="minimal split dependencies only; not the original full Ningbo "
                    "manifest, near-duplicate audit or external-overlap receipt")
    write_json_atomic(output / "metadata.json", metadata, sort_keys=True)
    if digest != TARGET:
        raise ValueError(f"Original split hash not reproduced: {digest}")
    return metadata
