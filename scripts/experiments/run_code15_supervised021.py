"""Experiment 021: supervised CODE-15 continuation of the starting CPC encoder and its PTB-XL readout."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits
from torch import nn
from torch.utils.data import DataLoader, Subset

from ecg_experiment.code15_cpc import LABELS, MAX_AMPLITUDE_MV, Code15Dataset, amplitude_factor, lead_std
from ecg_experiment.cpc import CPCEncoder
from ecg_experiment.cpc_pool import Pool
from ecg_experiment.files import sha256_file, write_json_atomic, write_torch_atomic
from ecg_experiment.full_development import cohorts, fit_logistic, patient_bootstrap, predict, ptb_table
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.ptb_cpc_features import NORMALIZATION, POOL_DIR, pooled_features, ptb_signals
from ecg_experiment.reproducibility import capture_rng_state, restore_rng_state, seed_everything
from ecg_experiment.training import checked_step

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data/processed/code15_cpc_250hz_v1"
BASE = ROOT / "outputs/experiment004_cpc_40k"
PRIOR = ROOT / "outputs/experiment020_full_development_v2"
OUTPUT = ROOT / "outputs/experiment021_code15_supervised_v1"
MODEL_SEED = 21042
ORDER_SEED = 21047
BOOTSTRAP_SEED = 21045
EXPOSURES = 115_359
BATCH = 128
PROFILE_BATCHES = 24
CEILING_SECONDS = 7_200
SOURCES = (
    "ecg_experiment/code15_cpc.py", "ecg_experiment/ptb_cpc_features.py",
    "ecg_experiment/full_development.py", "ecg_experiment/cpc.py", "ecg_experiment/training.py",
    "scripts/experiments/run_code15_supervised021.py", "docs/experiment-021-code15-supervised.md",
    "pyproject.toml", "uv.lock",
)


class SupervisedCPC(nn.Module):
    """The CPC encoder with a linear head from its pooled features to the CODE-15 labels."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = CPCEncoder()
        self.head = nn.Linear(512, len(LABELS))

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Return one logit per label."""
        _, contexts = self.encoder(signal)
        return self.head(CPCEncoder.pooled(contexts))


def cache_rows() -> pd.DataFrame:
    """Read the verified cache rows."""
    metadata = json.loads((CACHE / "metadata.json").read_text())
    if not metadata["complete"] or sha256_file(CACHE / "rows.csv") != metadata["rows_sha256"]:
        raise ValueError("CODE-15 cache is incomplete or changed")
    return pd.read_csv(CACHE / "rows.csv")


def factor(rows: pd.DataFrame, pool: Pool) -> float:
    """Compute, or reuse, the train-only amplitude factor."""
    path = OUTPUT / "amplitude_factor.json"
    if path.exists():
        return json.loads(path.read_text())["factor"]
    ptb_rows = [i for i, row in enumerate(pool.rows) if row["source"] == "ptbxl" and row["split"] == "train"]
    ptb = lead_std(pool.signals[ptb_rows[0]:ptb_rows[-1] + 1])[np.asarray(ptb_rows) - ptb_rows[0]]
    train = rows[rows["split"] == "train"]
    code = np.concatenate([lead_std(np.load(CACHE / shard, mmap_mode="r"))[group["index"].to_numpy()]
                           for shard, group in train.groupby("shard")])
    value = amplitude_factor(ptb, code)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, {"factor": value, "ptb_training_records": len(ptb),
                             "code15_training_records": len(code)})
    return value


def clean_rows(rows: pd.DataFrame, split: str, scale: float) -> pd.DataFrame:
    """Apply the quality exclusions to one patient split."""
    failed = rows[["nonfinite", "constant_lead", "flat_segment"]].any(axis=1)
    clean = ~failed & (rows["peak"] * scale <= MAX_AMPLITUDE_MV)
    return rows[(rows["split"] == split) & clean].reset_index(drop=True)


def identity(train: pd.DataFrame, scale: float) -> dict[str, object]:
    """Hash the cache, starting encoder and sources."""
    metadata = json.loads((CACHE / "metadata.json").read_text())
    return {
        "cache_metadata": sha256_file(CACHE / "metadata.json"), "cache_rows": metadata["rows_sha256"],
        "encoder": sha256_file(BASE / "cpc_ssl/encoder.pt"), "normalization": sha256_file(NORMALIZATION),
        "sources": {name: sha256_file(ROOT / name) for name in SOURCES},
        "amplitude_factor": scale, "training_records": len(train), "exposures": EXPOSURES, "batch": BATCH,
        "model_seed": MODEL_SEED, "order_seed": ORDER_SEED,
    }


def initial_model() -> tuple[SupervisedCPC, torch.optim.Optimizer]:
    """Start from the unchanged starting encoder with a seeded head."""
    seed_everything(MODEL_SEED)
    model = SupervisedCPC().cuda()
    saved = torch.load(BASE / "cpc_ssl/encoder.pt", map_location="cpu", weights_only=True)
    model.encoder.load_state_dict(saved["encoder"], strict=True)
    return model, torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)


def loader(dataset: Code15Dataset, indices: np.ndarray) -> DataLoader:
    """Read rows in a fixed order."""
    return DataLoader(Subset(dataset, indices.tolist()), batch_size=BATCH, shuffle=False, num_workers=4,
                      pin_memory=True)


def update(model: SupervisedCPC, optimizer: torch.optim.Optimizer,
           batch: tuple[torch.Tensor, torch.Tensor]) -> float:
    """Apply one supervised update."""
    signals, labels = batch
    logits = model(signals.cuda(non_blocking=True))
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels.cuda(non_blocking=True))
    checked_step(loss, model, optimizer, "CODE-15 supervised")
    return float(loss.detach())


def profile(dataset: Code15Dataset, order: np.ndarray, run_identity: dict[str, object],
            preflight: float) -> None:
    """Time real updates and gate the full training."""
    model, optimizer = initial_model()
    start = time.monotonic()
    losses = [update(model, optimizer, batch) for batch in loader(dataset, order[:PROFILE_BATCHES * BATCH])]
    torch.cuda.synchronize()
    measured = time.monotonic() - start
    projected = 2 * preflight + measured + 1.25 * math.ceil(EXPOSURES / BATCH) * measured / len(losses) + 300
    receipt = {"identity": run_identity, "profile_seconds": measured, "preflight_seconds": preflight,
               "projected_total_seconds": projected, "ceiling_seconds": CEILING_SECONDS,
               "gate_passed": projected <= CEILING_SECONDS, "last_profile_loss": losses[-1],
               "peak_gpu_bytes": torch.cuda.max_memory_allocated()}
    write_json_atomic(OUTPUT / "profile.json", receipt)
    print(json.dumps({"stage": "profile", "gate_passed": receipt["gate_passed"],
                      "projected_total_seconds": projected}), flush=True)


def train_stage(dataset: Code15Dataset, order: np.ndarray, run_identity: dict[str, object]) -> None:
    """Train 902 resumable updates and save the final encoder."""
    receipt = json.loads((OUTPUT / "profile.json").read_text())
    if receipt["identity"] != run_identity or not receipt["gate_passed"]:
        raise ValueError("A matching passed profile is required")
    checkpoint = OUTPUT / "latest.pt"
    model, optimizer = initial_model()
    completed, loss_sum = 0, 0.0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["identity"] != run_identity:
            raise ValueError("Checkpoint identity changed")
        completed, loss_sum = saved["completed_batches"], saved["loss_sum"]
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng_state(saved["rng"])
    total = math.ceil(EXPOSURES / BATCH)
    start = time.monotonic()
    for offset, batch in enumerate(loader(dataset, order[completed * BATCH:]), start=1):
        loss_sum += update(model, optimizer, batch)
        current = completed + offset
        if current % 100 == 0 or current == total:
            torch.cuda.synchronize()
            write_torch_atomic(checkpoint, {"identity": run_identity, "completed_batches": current,
                                            "loss_sum": loss_sum, "model": model.state_dict(),
                                            "optimizer": optimizer.state_dict(), "rng": capture_rng_state()})
            print(json.dumps({"stage": "train", "updates": current, "total": total,
                              "seconds": time.monotonic() - start}), flush=True)
    write_json_atomic(OUTPUT / "complete.json", {
        "identity": run_identity, "completed_updates": total, "mean_loss": loss_sum / total,
        "checkpoint_sha256": sha256_file(checkpoint), "profile_sha256": sha256_file(OUTPUT / "profile.json")})


def trained_model(run_identity: dict[str, object]) -> SupervisedCPC:
    """Load the final checkpoint after checking its completion receipt."""
    completion = json.loads((OUTPUT / "complete.json").read_text())
    checkpoint = sha256_file(OUTPUT / "latest.pt")
    if completion["identity"] != run_identity or completion["checkpoint_sha256"] != checkpoint:
        raise ValueError("Checkpoint does not match its completion receipt")
    model = SupervisedCPC().cuda().eval()
    model.load_state_dict(torch.load(OUTPUT / "latest.pt", map_location="cpu", weights_only=False)["model"])
    model.requires_grad_(False)
    return model


@torch.inference_mode()
def monitoring_scores(model: SupervisedCPC, dataset: Code15Dataset) -> dict[str, dict[str, float]]:
    """AUROC and average precision of each CODE-15 label on monitoring patients."""
    logits, labels = [], []
    for signals, targets in DataLoader(dataset, batch_size=256, num_workers=4):
        logits.append(model(signals.cuda()).float().cpu().numpy())
        labels.append(targets.numpy())
    logits, labels = np.concatenate(logits), np.concatenate(labels)
    return {name: {"auroc": float(roc_auc_score(labels[:, i], logits[:, i])),
                   "average_precision": float(average_precision_score(labels[:, i], logits[:, i])),
                   "positives": int(labels[:, i].sum())} for i, name in enumerate(LABELS)}


def ptb_predictions(features: dict[str, np.ndarray], train: pd.DataFrame) -> dict[str, np.ndarray]:
    """Fit the Experiment 020 CPC heads on the new features."""
    masks = {"cpc_project_full": (train["target"].notna(), "target"),
             "cpc_project_limited": (train["limited"], "target"),
             "cpc_standard": (train["standard"].notna(), "standard")}
    predictions = {}
    for name, (mask, label) in masks.items():
        mask = mask.to_numpy()
        head = fit_logistic(features["train"][mask], train.loc[mask, label].to_numpy(dtype=np.int64))
        predictions[name] = predict(head, features["development"])
    return predictions


def contrasts(dev: pd.DataFrame, new: dict[str, np.ndarray]) -> dict[str, dict[str, float | int]]:
    """Compare with the starting encoder's saved Experiment 020 predictions."""
    result = json.loads((PRIOR / "result.json").read_text())
    if sha256_file(PRIOR / "development_predictions.npz") != result["predictions_sha256"]:
        raise ValueError("Experiment 020 predictions changed")
    with np.load(PRIOR / "development_predictions.npz", allow_pickle=True) as saved:
        if not np.array_equal(saved["record_ids"], dev.index.to_numpy()):
            raise ValueError("Development order differs from Experiment 020")
        prior = {name: saved[name] for name in new}
    standard = dev["standard"].notna().to_numpy()
    original = dev["original"].to_numpy()
    plans = {
        "cpc_standard_full_development": ("cpc_standard", standard, "standard"),
        "cpc_standard_added": ("cpc_standard", standard & ~original, "standard"),
        "cpc_project_full_original": ("cpc_project_full", original, "target"),
        "cpc_project_limited_original": ("cpc_project_limited", original, "target"),
    }
    output = {}
    for name, (head, mask, label) in plans.items():
        y = dev.loc[mask, label].to_numpy(dtype=np.int64)
        comparison = patient_bootstrap(dev.loc[mask, "patient_id"].to_numpy(), y, new[head][mask],
                                       prior[head][mask], seed=BOOTSTRAP_SEED)
        output[name] = {**comparison, "new_auroc": float(roc_auc_score(y, new[head][mask])),
                        "starting_auroc": float(roc_auc_score(y, prior[head][mask])), "records": len(y)}
    return output


def readout(dataset: Code15Dataset, run_identity: dict[str, object]) -> None:
    """Score CODE-15 monitoring patients and read the encoder out on PTB-XL development."""
    pool = Pool(POOL_DIR)
    groups = cohorts(ptb_table())
    train, dev = groups["train"], groups["development"]
    with gpu_lock("cuda", blocking=False):
        model = trained_model(run_identity)
        monitoring = monitoring_scores(model, dataset)
        features = {name: pooled_features(model.encoder, ptb_signals(pool, frame))
                    for name, frame in (("train", train), ("development", dev))}
    with threadpool_limits(limits=1):
        predictions = ptb_predictions(features, train)
        comparison = contrasts(dev, predictions)
    predictions_path = OUTPUT / "development_predictions.npz"
    np.savez_compressed(predictions_path, record_ids=dev.index.to_numpy(), **predictions)
    write_json_atomic(OUTPUT / "result.json", {
        "status": "complete_development_only", "identity": run_identity, "code15_monitoring": monitoring,
        "ptb_contrasts": comparison, "predictions_sha256": sha256_file(predictions_path),
        "calibration_test_evaluated": False})
    print(json.dumps({"stage": "readout", "ptb_contrasts": comparison}, indent=1), flush=True)


def main() -> None:
    """Run one stage of Experiment 021."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("profile", "train", "readout"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    start = time.monotonic()
    rows = cache_rows()
    scale = factor(rows, Pool(POOL_DIR))
    clean = clean_rows(rows, "train", scale)
    if len(clean) < EXPOSURES:
        raise ValueError("Fewer clean training records than planned exposures")
    run_identity = identity(clean, scale)
    order = np.random.default_rng(ORDER_SEED).permutation(len(clean))[:EXPOSURES]
    preflight = time.monotonic() - start
    if args.stage == "readout":
        monitoring = clean_rows(rows, "monitoring", scale)
        readout(Code15Dataset(CACHE, monitoring, scale, NORMALIZATION), run_identity)
        return
    dataset = Code15Dataset(CACHE, clean, scale, NORMALIZATION)
    with gpu_lock("cuda", blocking=False):
        if args.stage == "profile":
            profile(dataset, order, run_identity, preflight)
        else:
            train_stage(dataset, order, run_identity)

if __name__ == "__main__":
    main()
