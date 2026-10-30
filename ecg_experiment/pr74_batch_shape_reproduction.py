"""Reproduce admitted PTB encoder vectors with their historical full microbatch shape."""

from __future__ import annotations

import gc
import hashlib
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from . import ROOT
from .cuda_runtime20261028 import configure_cuda_runtime
from .external_encoders import xecg_features, xecg_input
from .files import sha256_file
from .gpu import gpu_lock
from .pr74_fresh_baseline import OUTPUT as SOURCE
from .pr74_fresh_baseline import finish_stage, read_signal, source_binding, stage_inputs
from .pr74_screening_features import checkpoint_hashes, load_encoder
from .pr74_screening_io import relative, stored

OUTPUT = Path("outputs/pr74_fresh_baseline_exact_20261028")
PROTOCOL = Path("docs/pr74-batch-shape-reproduction-2026-10-28.md")


def binding() -> dict[str, Any]:
    """Bind the committed numerical amendment before reading or changing derived artifacts."""
    receipt = source_binding()
    paths = [
        Path(__file__),
        ROOT / PROTOCOL,
        ROOT / "scripts/experiments/reproduce_pr74_batch_shape.py",
        ROOT / "tests/test_pr74_batch_shape_reproduction.py",
    ]
    for path in paths:
        name = stored(path.relative_to(ROOT))
        blob = subprocess.check_output(["git", "show", "HEAD:" + name], cwd=ROOT)
        if hashlib.sha256(blob).hexdigest() != sha256_file(path):
            raise ValueError("Batch-shape reproduction source must be committed")
        receipt["source_sha256"][name] = sha256_file(path)
    receipt["protocol"] = {"path": str(PROTOCOL), "sha256": sha256_file(ROOT / PROTOCOL)}
    return receipt


def padded_vectors(model: torch.nn.Module, inputs: np.ndarray) -> np.ndarray:
    """Encode an admitted partial batch as 16 examples and discard every repeated padding output."""
    size = len(inputs)
    if not 0 < size < 16 or inputs.shape[1:] != (1000, 12) or not np.isfinite(inputs).all():
        raise ValueError("Expected one finite partial admitted PTB microbatch")
    padded = np.concatenate([inputs, np.repeat(inputs[-1:], 16 - size, axis=0)], axis=0)
    return xecg_features(model, padded)[:size]


def reproduce(input_root: Path, source: Path = SOURCE, output: Path = OUTPUT) -> dict[str, Any]:
    """Write an honest new feature stage without touching completed source tables or excluded records."""
    if relative(source) == relative(output):
        raise ValueError("Batch correction must use a new output namespace")
    receipt = binding()
    configure_cuda_runtime()
    _, hashes = stage_inputs(input_root, source, "features")
    profiled, bound = stage_inputs(input_root, source, "profile")
    hashes.update(bound)
    hashes.update(checkpoint_hashes(input_root))
    if not all(profiled["resource_gates"].values()):
        raise ValueError("The existing real resource profile must pass")
    origin = input_root / source / "features"
    rows = pd.read_csv(origin / "ptb_rows.csv", dtype=str, keep_default_na=False)
    size = len(rows) % 16
    if len(rows) != 19229 or size != 13 or not rows["strat_fold"].isin([str(i) for i in range(1, 10)]).all():
        raise ValueError("Only the exact admitted PTB development population may be corrected")
    started = time.perf_counter()
    with gpu_lock("cuda", blocking=False), torch.cuda.device(0):
        torch.set_num_threads(2)
        free, _ = torch.cuda.mem_get_info()
        if free < 4 * 1024**3:
            raise RuntimeError("Less than four GiB free for the admitted partial microbatch")
        model = load_encoder(input_root)
        signals = [read_signal(input_root, row) for _, row in rows.tail(size).iterrows()]
        vectors = padded_vectors(model, np.stack([xecg_input(signal) for signal in signals]))
        torch.cuda.synchronize()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    destination = input_root / relative(output) / "features"
    destination.mkdir(parents=True, exist_ok=False)
    for site in ("ptb", "sph"):
        for suffix in ("rows.csv", "measurements.csv", "xecg.npy"):
            shutil.copyfile(origin / (site + "_" + suffix), destination / (site + "_" + suffix))
    values = np.load(destination / "ptb_xecg.npy", mmap_mode="r+")
    if values.shape != (19229, 1024):
        raise ValueError("The parent PTB vector population changed")
    maximum = float(np.max(np.abs(values[-size:].astype(float) - vectors.astype(float))))
    changed = int(np.count_nonzero(np.any(values[-size:] != vectors, axis=1)))
    values[-size:] = vectors
    values.flush()
    return finish_stage(
        input_root,
        output,
        "features",
        receipt,
        hashes,
        {
            "records": {"ptb": len(rows), "sph": 17402},
            "device": 0,
            "execution_mode": "fresh_parent_with_admitted_ptb_batch_shape_reproduction",
            "parent_features": str(source),
            "recomputed_ptb_records": size,
            "padding_inputs_discarded": 16 - size,
            "changed_ptb_vectors": changed,
            "maximum_absolute_vector_change": maximum,
            "sph_vectors_changed": 0,
            "excluded_ptb_fold10_records_read": 0,
            "encoder_fitted": False,
            "wall_seconds": time.perf_counter() - started,
        },
    )
