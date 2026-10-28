"""HEADS-01 training superclass labels and frozen xECG token extraction."""

from __future__ import annotations

import ast
import gc
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .challenge_labels import load_official, record_labels
from .eda.ptbxl import diagnostic_classes, load_statements, superclass_label
from .external_encoders import xecg_input
from .files import sha256_file, write_json_atomic
from .gpu import GPU_LOCK_PATH, gpu_lock
from .heads01_heads import SUPERCLASSES
from .pr74_fresh_baseline import read_signal
from .pr74_screening_features import load_encoder, raw_pair, read_training_signal

BATCH = 16
TOKENS = 40
WIDTH = 1024
PTB_DATABASE = Path("data/raw/ptb-xl/1.0.3/ptbxl_database.csv")


def lock_path(device: int) -> Path:
    """Return the shared project lock of one GPU, as in ``pr74_fresh_baseline``."""
    return GPU_LOCK_PATH if device == 0 else Path(f"/tmp/ecg_project_gpu{device}.lock")


def header_codes(input_root: Path, row: pd.Series) -> list[str]:
    """
    Read the SNOMED codes of one admitted Challenge training header after its checksum check.

    Parameters
    ----------
    input_root : Path
        Checkout holding ``data/raw``.
    row : pd.Series
        Admitted training candidate row.

    Returns
    -------
    list[str]
        Codes of the header's ``Dx`` line.

    Raises
    ------
    ValueError
        If the record is not training data, the header changed or has no single ``Dx`` line.
    """
    if row["split"] != "train":
        raise ValueError("Only original training headers may be read")
    raw, stem, expected = raw_pair(input_root, row)
    path = raw / (stem + ".hea")
    data = path.read_bytes()
    if sha256_file(path) != expected[".hea"]:
        raise ValueError("Challenge training header changed")
    lines = [line.lstrip("#").strip() for line in data.decode().splitlines() if line.startswith("#")]
    found = [line[3:].strip() for line in lines if line.startswith("Dx:")]
    if len(found) != 1:
        raise ValueError(f"Expected one Dx line in {stem}")
    return [code.strip() for code in found[0].split(",") if code.strip()]


def challenge_flags(input_root: Path, rows: pd.DataFrame) -> pd.DataFrame:
    """Primary label and superclass flags of admitted Challenge training rows."""
    official = load_official(input_root / "data/raw/challenge-2021/evaluation-2021")
    labels = [record_labels(header_codes(input_root, row), official) for _, row in rows.iterrows()]
    return pd.DataFrame(labels, index=rows["record_id"].to_numpy())


def ptb_flags(input_root: Path, rows: pd.DataFrame) -> pd.DataFrame:
    """Binary label and superclass flags of admitted PTB training rows only."""
    meta = pd.read_csv(input_root / PTB_DATABASE, dtype={"ecg_id": str}, keep_default_na=False)
    meta = meta.set_index("ecg_id")
    ids = rows["record"].map(lambda value: str(int(value)))
    codes = meta.loc[ids, "scp_codes"].map(ast.literal_eval).to_numpy()
    classes = diagnostic_classes(load_statements())
    groups = [{classes[code] for code in record if code in classes} for record in codes]
    table = pd.DataFrame({name: [float(name in group) for group in groups] for name in SUPERCLASSES},
                         index=rows["record_id"].to_numpy())
    table.insert(0, "primary", [superclass_label(record, classes) for record in codes])
    return table


def superclass_table(input_root: Path, pool: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct superclass flags for every admitted training record and enforce exact agreement.

    Parameters
    ----------
    input_root : Path
        Checkout holding ``data/raw``.
    pool : pd.DataFrame
        Admitted candidate union with ``record_id``, ``source`` and integer ``target``.

    Returns
    -------
    pd.DataFrame
        ``record_id``, ``source``, ``target`` and one 0/1 column per superclass, in pool order.

    Raises
    ------
    ValueError
        If any reconstructed label or flag union differs from the cohort target.
    """
    is_ptb = pool["source"].eq("ptbxl")
    parts = [challenge_flags(input_root, pool.loc[~is_ptb]), ptb_flags(input_root, pool.loc[is_ptb])]
    flags = pd.concat(parts).loc[pool["record_id"]]
    target = pool["target"].astype(int).to_numpy()
    union = flags[list(SUPERCLASSES)].to_numpy().max(axis=1)
    if not np.array_equal(flags["primary"].to_numpy(), target) or not np.array_equal(union, target):
        raise ValueError("Reconstructed superclass labels disagree with the cohort target")
    table = pool[["record_id", "source"]].reset_index(drop=True).copy()
    table["target"] = target
    for name in SUPERCLASSES:
        table[name] = flags[name].to_numpy().astype(int)
    return table


@torch.inference_mode()
def token_features(model: torch.nn.Module, inputs: np.ndarray) -> tuple[np.ndarray, ...]:
    """
    Pooled vector, token maximum and tokens of the frozen xECG for one microbatch.

    Parameters
    ----------
    model : torch.nn.Module
        Output of ``load_encoder``.
    inputs : np.ndarray
        Float32 ``[records, 1000, 12]`` inputs, at most one microbatch.

    Returns
    -------
    tuple[np.ndarray, ...]
        Float32 pooled ``[records, 1024]``, float32 maximum ``[records, 1024]`` and float16
        tokens ``[records, 40, 1024]``.

    Raises
    ------
    ValueError
        If the output shape is unexpected or nonfinite.
    """
    tensor = torch.from_numpy(np.ascontiguousarray(inputs)).to("cuda")
    pooled, tokens = model(tensor)
    if tokens.shape[1:] != (TOKENS, WIDTH) or pooled.shape[1:] != (WIDTH,):
        raise ValueError(f"Unexpected xECG token shape {tuple(tokens.shape)}")
    masked = tokens.masked_fill(model.get_padding_mask(tensor), float("-inf"))
    result = (pooled.float().cpu().numpy(), masked.max(dim=1).values.float().cpu().numpy(),
              tokens.half().cpu().numpy())
    if not all(np.isfinite(part).all() for part in result):
        raise ValueError("Nonfinite xECG tokens")
    return result


def read_any(input_root: Path, row: pd.Series) -> np.ndarray:
    """Read one admitted waveform through the predecessor's verified path for its source role."""
    if row["block"] == "challenge":
        return read_training_signal(input_root, row)
    return read_signal(input_root, row)


def run_blocks(input_root: Path, rows: pd.DataFrame, directory: Path, device: int,
               progress: bool = True) -> dict[str, Any]:
    """
    Extract every row of a part in source blocks of sixteen under the device's GPU lock.

    Parameters
    ----------
    input_root : Path
        Checkout holding inputs.
    rows : pd.DataFrame
        Part rows with a ``block`` column, in extraction order.
    directory : Path
        New output directory for ``pooled.npy``, ``max.npy`` and ``tokens.npy``.
    device : int
        CUDA device index.
    progress : bool
        Write ``progress.json`` and print every 512 records.

    Returns
    -------
    dict[str, Any]
        Timing, peak memory and initial free memory.

    Raises
    ------
    RuntimeError
        If fewer than 4 GiB are free on the device.
    """
    n = len(rows)
    pooled = np.lib.format.open_memmap(directory / "pooled.npy", "w+", np.float32, (n, WIDTH))
    maximum = np.lib.format.open_memmap(directory / "max.npy", "w+", np.float32, (n, WIDTH))
    tokens = np.lib.format.open_memmap(directory / "tokens.npy", "w+", np.float16, (n, TOKENS, WIDTH))
    starts = [start for _, block in rows.groupby("block", sort=False)
              for start in range(block.index[0], block.index[-1] + 1, BATCH)]
    ends = {block.index[-1] + 1 for _, block in rows.groupby("block", sort=False)}
    with gpu_lock("cuda", blocking=False, path=lock_path(device)), torch.cuda.device(device):
        torch.set_num_threads(2)
        free, _ = torch.cuda.mem_get_info()
        if free < 4 * 1024**3:
            raise RuntimeError("Less than four GiB free on the selected GPU")
        torch.cuda.reset_peak_memory_stats()
        model = load_encoder(input_root)
        started = time.perf_counter()
        for count, start in enumerate(starts):
            stop = min(start + BATCH, min(end for end in ends if end > start))
            part = rows.iloc[start:stop]
            signals = np.stack([xecg_input(read_any(input_root, row)) for _, row in part.iterrows()])
            pooled[start:stop], maximum[start:stop], tokens[start:stop] = token_features(model, signals)
            if progress and count % 32 == 0:
                state = {"records": stop, "total": n, "wall_seconds": time.perf_counter() - started}
                write_json_atomic(directory / "progress.json", state)
                print(f"Tokens {stop}/{n} after {state['wall_seconds']:.1f} s", flush=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    for array in (pooled, maximum, tokens):
        array.flush()
    return {"wall_seconds": elapsed, "peak_gpu_bytes": int(peak), "initial_free_gpu_bytes": int(free),
            "records": n}


def pooled_parity(pooled: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    """
    Compare extracted pooled vectors with the predecessor's saved features.

    Parameters
    ----------
    pooled, expected : np.ndarray
        Aligned ``[records, 1024]`` arrays.

    Returns
    -------
    dict[str, Any]
        Maximum absolute difference and the count of bit-identical rows.

    Raises
    ------
    ValueError
        If any element differs by more than 1e-5.
    """
    difference = np.abs(np.asarray(pooled, dtype=np.float64) - np.asarray(expected, dtype=np.float64))
    worst = float(difference.max())
    if worst > 1e-5:
        raise ValueError(f"Token mean differs from saved xECG features by {worst}")
    identical = int(np.all(np.asarray(pooled) == np.asarray(expected), axis=1).sum())
    return {"max_absolute_difference": worst, "bit_identical_rows": identical, "rows": len(pooled)}
