"""Extract frozen CPC, ECG-JEPA and xECG features of the PhysioNet Challenge 2021 sources.

The encoders and their input paths are those of Experiment 022. Records are read in source order. A
longer record gives its centred ten-second window; a record shorter than ten seconds, not at 500 Hz,
without twelve leads or with a nonfinite window is skipped with its reason. No readout is fitted and no
label is read.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from torch import nn

from ecg_experiment.challenge_features import (
    RAW_ROOTS,
    canonical_window,
    encoder_inputs,
    files_digest,
    read_verified,
    record_stems,
    skip_reasons,
    window_start,
)
from ecg_experiment.downloads import parse_checksums
from ecg_experiment.external_encoders import (
    JEPA_CHECKPOINT,
    jepa_features,
    load_jepa,
    load_xecg_backbone,
    source_files,
    xecg_features,
)
from ecg_experiment.external_readout import ENCODER, load_encoder
from ecg_experiment.files import sha256_file, write_json_atomic, write_npz_atomic
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.ptb_cpc_features import NORMALIZATION, pooled_features
from ecg_experiment.public_sources import signal_sha256
from ecg_experiment.xecg import DEFAULT_CHECKPOINT_DIR

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/processed/ningbo_clean_v1"
OUTPUT = ROOT / "outputs/features_challenge_v1"
REFERENCE_PROFILE = ROOT / "outputs/experiment022_sph_external_v3/profile.json"
SOURCES = ("ningbo", "chapman_shaoxing", "georgia", "cpsc_2018", "cpsc_2018_extra")
ENCODER_NAMES = ("cpc", "jepa", "xecg")
CHUNK = 512
PROFILE_RECORDS = 512
INTEGRITY_RECORDS = 128
CEILING_SECONDS = 5_400
READER_THREADS = 4
WINDOW_RULE = ("centred 10 s window, start = (samples - 5000) // 2, the ssl_center_crop rule of "
               "public_sources.load_view; skipped if shorter than 10 s, not 500 Hz, not 12 leads or "
               "nonfinite in the window")
XECG_CHECKPOINT_FILES = ("model.safetensors", "config.json", "xECG.py", "download_provenance.json")
CODE = ("ecg_experiment/challenge_features.py", "ecg_experiment/external_encoders.py",
        "ecg_experiment/external_readout.py", "ecg_experiment/ptb_cpc_features.py", "ecg_experiment/cpc.py",
        "ecg_experiment/cpc_input_audit.py", "ecg_experiment/xecg.py", "ecg_experiment/eda/signals.py",
        "ecg_experiment/wfdb_records.py", "scripts/data/extract_challenge_features.py", "pyproject.toml",
        "uv.lock")
EXTRACTORS: dict[str, Callable[[nn.Module, np.ndarray], np.ndarray]] = {
    "cpc": pooled_features, "jepa": jepa_features, "xecg": xecg_features}

# record, release-relative stem, manifest float32 window hash ("" when the source has no manifest)
Item = tuple[str, str, str]


def encoder_hashes() -> dict[str, object]:
    """
    Hash the encoder checkpoints and input-path code and require them to equal Experiment 022's.

    Returns
    -------
    dict[str, object]
        CPC encoder and normalization, ECG-JEPA and xECG checkpoint and encoder source hashes.

    Raises
    ------
    ValueError
        If any hash differs from the Experiment 022 profile.
    """
    reference = json.loads(REFERENCE_PROFILE.read_text())["identity"]
    caches = reference["feature_caches"]
    hashes = {
        "cpc_encoder": sha256_file(ENCODER), "cpc_normalization": sha256_file(NORMALIZATION),
        "jepa_checkpoint": sha256_file(JEPA_CHECKPOINT),
        "xecg_checkpoint": {name: sha256_file(DEFAULT_CHECKPOINT_DIR / name)
                            for name in XECG_CHECKPOINT_FILES},
        "encoder_sources": {str(path.relative_to(ROOT)): sha256_file(path) for path in source_files()},
    }
    expected = {
        "cpc_encoder": reference["cpc_encoder"], "cpc_normalization": reference["cpc_normalization"],
        "jepa_checkpoint": caches["jepa"]["checkpoint"],
        "xecg_checkpoint": {name: caches["xecg"][f"checkpoint_{name}"] for name in XECG_CHECKPOINT_FILES},
        "encoder_sources": reference["encoder_sources"],
    }
    if hashes != expected:
        raise ValueError("Encoder checkpoints or input code differ from Experiment 022")
    return {**hashes, "experiment022_profile_sha256": sha256_file(REFERENCE_PROFILE)}


def ningbo_items() -> tuple[list[Item], dict[str, str]]:
    """
    Every row of the clean Ningbo manifest, after checking the manifest against its receipt.

    Returns
    -------
    tuple[list[Item], dict[str, str]]
        Items in manifest order and the manifest hashes.

    Raises
    ------
    ValueError
        If ``rows.csv`` differs from its metadata.
    """
    metadata = json.loads((MANIFEST / "metadata.json").read_text())
    hashes = {"rows_csv": sha256_file(MANIFEST / "rows.csv"),
              "metadata_json": sha256_file(MANIFEST / "metadata.json")}
    if hashes["rows_csv"] != metadata["rows_sha256"]:
        raise ValueError("Ningbo manifest differs from its metadata")
    rows = pd.read_csv(MANIFEST / "rows.csv", dtype=str, usecols=["record", "path", "window_sha256"])
    return list(rows.itertuples(index=False, name=None)), hashes


def source_items(checksums: dict[str, dict[str, str]]) -> tuple[dict[str, list[Item]], dict[str, str]]:
    """
    List the records of every source in extraction order.

    Parameters
    ----------
    checksums : dict[str, dict[str, str]]
        Official checksums per source.

    Returns
    -------
    tuple[dict[str, list[Item]], dict[str, str]]
        Ningbo manifest rows, then every record of each other source in path order, and the Ningbo
        manifest hashes.
    """
    ningbo, manifest_hashes = ningbo_items()
    items = {"ningbo": ningbo}
    for source in SOURCES[1:]:
        items[source] = [(Path(stem).name, stem, "") for stem in record_stems(checksums[source], source)]
    return items, manifest_hashes


def load_checksums() -> dict[str, dict[str, str]]:
    """
    Official checksums of each source's release.

    Returns
    -------
    dict[str, dict[str, str]]
        Digests keyed by release-relative file name, per source.
    """
    return {source: parse_checksums((root / "SHA256SUMS.txt").read_text())
            for source, root in RAW_ROOTS.items()}


def identity(items: dict[str, list[Item]], manifest_hashes: dict[str, str],
             checksums: dict[str, dict[str, str]]) -> dict[str, object]:
    """
    Hash every input of the extraction.

    Parameters
    ----------
    items : dict[str, list[Item]]
        Records per source, from ``source_items``.
    manifest_hashes : dict[str, str]
        Ningbo manifest hashes, from ``source_items``.
    checksums : dict[str, dict[str, str]]
        Output of ``load_checksums``.

    Returns
    -------
    dict[str, object]
        The identity a profile and a run must share.
    """
    releases = {str(root.relative_to(ROOT)): sha256_file(root / "SHA256SUMS.txt")
                for root in RAW_ROOTS.values()}
    records = {source: {"candidates": len(rows),
                        "record_files_sha256": files_digest([stem for _, stem, _ in rows], checksums[source])}
               for source, rows in items.items()}
    return {"encoders": encoder_hashes(), "ningbo_manifest": manifest_hashes, "sha256sums": releases,
            "records": records, "window_rule": WINDOW_RULE,
            "code": {name: sha256_file(ROOT / name) for name in CODE}}


def load_models() -> dict[str, nn.Module]:
    """
    Load the three frozen encoders on the GPU.

    Returns
    -------
    dict[str, nn.Module]
        Encoders keyed by ``cpc``, ``jepa`` and ``xecg``.
    """
    return {"cpc": load_encoder(), "jepa": load_jepa(), "xecg": load_xecg_backbone()}


def prepare(item: Item, source: str, checksums: dict[str, str],
            ) -> tuple[dict[str, np.ndarray] | None, str, int]:
    """
    Read one record and build its encoder inputs, or give the reason it is skipped.

    Parameters
    ----------
    item : Item
        Record, stem and manifest window hash.
    source : str
        Source name.
    checksums : dict[str, str]
        Official checksums of the source's release.

    Returns
    -------
    tuple[dict[str, np.ndarray] | None, str, int]
        Inputs per encoder, an empty reason and the window start; or ``None``, the ``;``-joined skip
        reasons and ``-1``.

    Raises
    ------
    ValueError
        If a Ningbo window differs from the manifest.
    """
    _, stem, window_sha256 = item
    signal, sampling_rate, names = read_verified(RAW_ROOTS[source], stem, checksums)
    reasons = skip_reasons(sampling_rate, signal)
    if reasons:
        return None, ";".join(reasons), -1
    start = window_start(len(signal))
    window = canonical_window(signal, names, start)
    if window_sha256 and signal_sha256(window.astype(np.float32)) != window_sha256:
        raise ValueError(f"Ningbo window differs from the manifest: {stem}")
    return encoder_inputs(window), "", start


def run_encoders(models: dict[str, nn.Module], batch: list[dict[str, np.ndarray]],
                 seconds: dict[str, float]) -> dict[str, np.ndarray]:
    """
    Features of one chunk of prepared records.

    Parameters
    ----------
    models : dict[str, nn.Module]
        Output of ``load_models``.
    batch : list[dict[str, np.ndarray]]
        Outputs of ``prepare``.
    seconds : dict[str, float]
        Running seconds per encoder, updated in place.

    Returns
    -------
    dict[str, np.ndarray]
        Float32 features per encoder.
    """
    features = {}
    for name in ENCODER_NAMES:
        began = time.monotonic()
        features[name] = EXTRACTORS[name](models[name], np.stack([inputs[name] for inputs in batch]))
        torch.cuda.synchronize()
        seconds[name] += time.monotonic() - began
    return features


def append_features(chunks: dict[str, list[np.ndarray]], features: dict[str, np.ndarray]) -> None:
    """
    Append one chunk's features to the running lists.

    Parameters
    ----------
    chunks : dict[str, list[np.ndarray]]
        Feature chunks per encoder, updated in place.
    features : dict[str, np.ndarray]
        Output of ``run_encoders``.
    """
    for name, values in features.items():
        chunks[name].append(values)


def extract(models: dict[str, nn.Module], source: str, items: list[Item], checksums: dict[str, str],
            ) -> tuple[list[tuple[str, int]], dict[str, np.ndarray], dict[str, str], dict[str, float]]:
    """
    Extract every featurizable record of a source in item order, in chunks of ``CHUNK`` records.

    Records are read by ``READER_THREADS`` threads, one block of ``CHUNK`` items at a time.

    Parameters
    ----------
    models : dict[str, nn.Module]
        Output of ``load_models``.
    source : str
        Source name.
    items : list[Item]
        Records to read.
    checksums : dict[str, str]
        Official checksums of the source's release.

    Returns
    -------
    tuple[list[tuple[str, int]], dict[str, np.ndarray], dict[str, str], dict[str, float]]
        Extracted records with their window starts, their features per encoder, skip reason per skipped
        record, and seconds spent reading (including input conversion) and per encoder.
    """
    records, skipped, batch = [], {}, []
    chunks = {name: [] for name in ENCODER_NAMES}
    seconds = dict.fromkeys(("read", *ENCODER_NAMES), 0.0)
    read = partial(prepare, source=source, checksums=checksums)
    with ThreadPoolExecutor(READER_THREADS) as pool:
        for start in range(0, len(items), CHUNK):
            began = time.monotonic()
            block = items[start:start + CHUNK]
            prepared = list(pool.map(read, block))
            seconds["read"] += time.monotonic() - began
            for item, (inputs, reason, start) in zip(block, prepared, strict=True):
                if inputs is None:
                    skipped[item[0]] = reason
                    continue
                records.append((item[0], start))
                batch.append(inputs)
                if len(batch) == CHUNK:
                    append_features(chunks, run_encoders(models, batch, seconds))
                    batch = []
    if batch:
        append_features(chunks, run_encoders(models, batch, seconds))
    return records, {name: np.concatenate(values) for name, values in chunks.items()}, skipped, seconds


def integrity(models: dict[str, nn.Module], source: str, items: list[Item], checksums: dict[str, str],
              saved: dict[str, np.ndarray]) -> dict[str, object]:
    """
    Re-extract the first ``INTEGRITY_RECORDS`` saved records and require identical features.

    The first records fill the same encoder batches as in the full pass, so any difference is a fault.

    Parameters
    ----------
    models : dict[str, nn.Module]
        Output of ``load_models``.
    source : str
        Source name.
    items : list[Item]
        Records of the source.
    checksums : dict[str, str]
        Official checksums of the source's release.
    saved : dict[str, np.ndarray]
        Contents of the written npz.

    Returns
    -------
    dict[str, object]
        Checked records and the largest absolute difference per encoder.

    Raises
    ------
    ValueError
        If any recomputed feature differs from the saved value.
    """
    wanted = set(saved["record"][:INTEGRITY_RECORDS])
    subset = [item for item in items if item[0] in wanted]
    records, features, _, _ = extract(models, source, subset, checksums)
    differences = {name: float(np.abs(features[name].astype(np.float64) - saved[name][:len(records)]).max())
                   for name in ENCODER_NAMES}
    expected = list(zip(saved["record"][:INTEGRITY_RECORDS].tolist(),
                        saved["window_start"][:INTEGRITY_RECORDS].tolist(), strict=True))
    if records != expected or any(differences.values()):
        raise ValueError(f"Re-extracted {source} features differ: {differences}")
    return {"records": [record for record, _ in records], "max_abs_difference": differences}


def check_saved(path: Path, records: list[tuple[str, int]]) -> dict[str, np.ndarray]:
    """
    Reload a written npz and require unique records in input order and finite float32 features.

    Parameters
    ----------
    path : Path
        Written npz.
    records : list[tuple[str, int]]
        Extracted records and window starts in input order.

    Returns
    -------
    dict[str, np.ndarray]
        The saved arrays.

    Raises
    ------
    ValueError
        If the order, uniqueness, shape, dtype or finiteness is wrong.
    """
    with np.load(path) as handle:
        saved = {name: handle[name] for name in handle.files}
    names = [record for record, _ in records]
    order_ok = saved["record"].tolist() == names and saved["window_start"].tolist() == [s for _, s in records]
    if not order_ok or len(set(names)) != len(names):
        raise ValueError(f"Saved record order differs from the input order: {path}")
    for name in ENCODER_NAMES:
        values = saved[name]
        if values.dtype != np.float32 or len(values) != len(records) or not np.isfinite(values).all():
            raise ValueError(f"Invalid saved {name} features in {path}")
    return saved


def completed(run_identity: dict[str, object]) -> dict[str, object]:
    """
    Find the sources a previous run with the same identity finished, with unchanged npz files.

    Parameters
    ----------
    run_identity : dict[str, object]
        Output of ``identity``.

    Returns
    -------
    dict[str, object]
        Metadata per completed source.
    """
    path = OUTPUT / "metadata.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    if previous.get("identity") != run_identity:
        return {}
    return {source: entry for source, entry in previous["sources"].items()
            if sha256_file(OUTPUT / f"{source}.npz") == entry["npz_sha256"]}


def extract_source(models: dict[str, nn.Module], source: str, items: list[Item],
                   checksums: dict[str, str]) -> dict[str, object]:
    """
    Extract, write and verify one source.

    Parameters
    ----------
    models : dict[str, nn.Module]
        Output of ``load_models``.
    source : str
        Source name.
    items : list[Item]
        Records of the source.
    checksums : dict[str, str]
        Official checksums of the source's release.

    Returns
    -------
    dict[str, object]
        Counts, skipped records, timings, integrity result and the npz hash.
    """
    began = time.monotonic()
    records, features, skipped, seconds = extract(models, source, items, checksums)
    path = OUTPUT / f"{source}.npz"
    starts = np.array([start for _, start in records], dtype=np.int64)
    names = np.array([record for record, _ in records])
    write_npz_atomic(path, record=names, window_start=starts, **features)
    saved = check_saved(path, records)
    checks = integrity(models, source, items, checksums, saved)
    return {"candidates": len(items), "extracted": len(records), "skipped": len(skipped),
            "skip_reason_counts": dict(Counter(skipped.values())), "skipped_records": skipped,
            "window_rule": WINDOW_RULE, "windowed_records": int((starts > 0).sum()),
            "nonzero_window_start": {record: start for record, start in records if start > 0},
            "dimensions": {name: int(features[name].shape[1]) for name in ENCODER_NAMES},
            "seconds_by_part": seconds, "seconds": time.monotonic() - began,
            "seconds_per_candidate": sum(seconds.values()) / len(items), "integrity": checks,
            "npz_sha256": sha256_file(path)}


def profile(items: dict[str, list[Item]], checksums: dict[str, dict[str, str]],
            run_identity: dict[str, object]) -> None:
    """
    Time the first ``PROFILE_RECORDS`` Ningbo records and gate the full extraction.

    Parameters
    ----------
    items : dict[str, list[Item]]
        Output of ``source_items``.
    checksums : dict[str, dict[str, str]]
        Output of ``load_checksums``.
    run_identity : dict[str, object]
        Output of ``identity``.
    """
    with gpu_lock("cuda", blocking=False):
        began = time.monotonic()
        models = load_models()
        load_seconds = time.monotonic() - began
        records, _, skipped, seconds = extract(models, "ningbo", items["ningbo"][:PROFILE_RECORDS],
                                               checksums["ningbo"])
    per_record = sum(seconds.values()) / PROFILE_RECORDS
    candidates = sum(len(rows) for rows in items.values())
    integrity_records = INTEGRITY_RECORDS * len(SOURCES)
    projected = load_seconds + per_record * (candidates + integrity_records)
    receipt = {"identity": run_identity, "profile_records": PROFILE_RECORDS, "extracted": len(records),
               "skipped": skipped, "seconds_by_part": seconds, "seconds_per_record": per_record,
               "model_load_seconds": load_seconds, "candidates": candidates,
               "projected_total_seconds": projected, "ceiling_seconds": CEILING_SECONDS,
               "gate_passed": projected <= CEILING_SECONDS}
    write_json_atomic(OUTPUT / "profile.json", receipt)
    print(json.dumps({"stage": "profile", "seconds_per_record": per_record,
                      "projected_total_seconds": projected, "gate_passed": receipt["gate_passed"]}),
          flush=True)


def run(items: dict[str, list[Item]], checksums: dict[str, dict[str, str]],
        run_identity: dict[str, object]) -> None:
    """
    Extract every source in order, writing its npz and the metadata as each source finishes.

    Parameters
    ----------
    items : dict[str, list[Item]]
        Output of ``source_items``.
    checksums : dict[str, dict[str, str]]
        Output of ``load_checksums``.
    run_identity : dict[str, object]
        Output of ``identity``.

    Raises
    ------
    ValueError
        If no matching passed profile exists.
    """
    receipt = json.loads((OUTPUT / "profile.json").read_text())
    if receipt["identity"] != run_identity or not receipt["gate_passed"]:
        raise ValueError("A matching passed profile is required")
    sources = completed(run_identity)
    metadata = {"status": "running", "identity": run_identity,
                "profile_sha256": sha256_file(OUTPUT / "profile.json"), "sources": sources}
    with gpu_lock("cuda", blocking=False):
        models = load_models()
        for source in SOURCES:
            if source in sources:
                continue
            sources[source] = extract_source(models, source, items[source], checksums[source])
            write_json_atomic(OUTPUT / "metadata.json", metadata)
            print(json.dumps({"source": source, "extracted": sources[source]["extracted"],
                              "skipped": sources[source]["skipped"], "seconds": sources[source]["seconds"]}),
                  flush=True)
    metadata["status"] = "complete"
    write_json_atomic(OUTPUT / "metadata.json", metadata)
    print(json.dumps({"stage": "complete"}), flush=True)


def main() -> None:
    """Profile or run the Challenge feature extraction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("profile", "run"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    checksums = load_checksums()
    items, manifest_hashes = source_items(checksums)
    run_identity = identity(items, manifest_hashes, checksums)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        if args.stage == "profile":
            profile(items, checksums, run_identity)
        else:
            run(items, checksums, run_identity)


if __name__ == "__main__":
    main()
