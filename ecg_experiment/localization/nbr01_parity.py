"""NBR-01 Amendment 1: re-encode unreplaced records inside their historical SPH batch of sixteen."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from ..external_encoders import xecg_features, xecg_input
from ..full_development import predict
from ..gpu import gpu_lock
from ..heads01_features import lock_path
from ..pr74_fresh_baseline import read_signal
from ..pr74_screening_features import load_encoder
from . import nbr01_features as nf
from .nbr01_selection import ROWS

FEATURES = Path("outputs/pr74_fresh_baseline_exact_20261028/features/sph_xecg.npy")
HISTORICAL_BATCH = 16


def differing_records(input_root: Path, rows: pd.DataFrame, work: Path) -> pd.DataFrame:
    """
    Return the prepared rows whose unreplaced xECG features are not bit-identical to the saved SPH features.

    Parameters
    ----------
    input_root : Path
        Checkout holding the saved features.
    rows : pd.DataFrame
        Prepared rows with ``ecg_id`` and ``position``.
    work : Path
        Local work directory with ``pooled``.

    Returns
    -------
    pd.DataFrame
        The differing rows, with ``max_abs_difference``.
    """
    saved = np.load(input_root / FEATURES, mmap_mode="r")
    gaps = np.array([float(np.max(np.abs(np.load(work / "pooled" / f"{ecg_id}.npy")[0].astype(np.float64)
                                         - saved[int(position)])))
                     for ecg_id, position in zip(rows["ecg_id"], rows["position"], strict=True)])
    return rows.assign(max_abs_difference=gaps).loc[gaps > 0].reset_index(drop=True)


def historical_parity(input_root: Path, rows: pd.DataFrame, work: Path, device: int) -> dict[str, Any]:
    """
    Re-encode each differing record inside its historical batch and compare features and scores.

    The saved SPH features were extracted in consecutive batches of sixteen rows of the frozen SPH table,
    so record ``p`` was encoded with rows ``16 * (p // 16)`` to ``16 * (p // 16) + 15``.

    Parameters
    ----------
    input_root : Path
        Checkout holding inputs.
    rows : pd.DataFrame
        Prepared rows.
    work : Path
        Local work directory with ``pooled``.
    device : int
        CUDA device.

    Returns
    -------
    dict[str, Any]
        Bit-identity of the re-encoded features and the largest score gaps against the saved predictions.
    """
    differing = differing_records(input_root, rows, work)
    table = pd.read_csv(input_root / ROWS, dtype=str, keep_default_na=False)
    saved = np.load(input_root / FEATURES, mmap_mode="r")
    predictions = np.load(input_root / nf.PREDICTIONS)
    bundle = joblib.load(input_root / nf.BUNDLE)
    records = []
    with gpu_lock("cuda", blocking=True, path=lock_path(device)), torch.cuda.device(device):
        torch.set_num_threads(2)
        model = load_encoder(input_root)
        for _, row in differing.iterrows():
            position = int(row["position"])
            start = HISTORICAL_BATCH * (position // HISTORICAL_BATCH)
            batch = table.iloc[start:start + HISTORICAL_BATCH]
            vectors = xecg_features(model, np.stack([xecg_input(read_signal(input_root, item))
                                                     for _, item in batch.iterrows()]))
            vector = vectors[position - start]
            gaps = {name: float(abs(predict(bundle["fitted"][name]["xecg"], vector[None])[0]
                                    - predictions[f"{name}_xecg"][position])) for name in nf.MODELS}
            records.append({"ecg_id": row["ecg_id"], "position": position,
                            "nbr01_max_abs_feature_difference": float(row["max_abs_difference"]),
                            "historical_batch_bit_identical": bool(np.array_equal(vector, saved[position])),
                            "historical_batch_xecg_score_gap": gaps})
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return {"records_checked": len(rows), "bit_identical_records": len(rows) - len(differing),
            "differing": records,
            "passed": all(item["historical_batch_bit_identical"] for item in records)
            and all(value <= 1e-12 for item in records
                    for value in item["historical_batch_xecg_score_gap"].values())}
