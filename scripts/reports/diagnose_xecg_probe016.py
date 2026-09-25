"""Read-only initial-state diagnostic for Experiment 016's probe-head decline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from ecg_experiment.files import read_csv, sha256_file, sha256_json, write_json_atomic
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.reproducibility import seed_everything
from ecg_experiment.training import require_cuda
from ecg_experiment.xecg import XECGBinaryClassifier, load_xecg

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/experiment016_xecg_probe_finetune"
MANIFEST = ROOT / "data/processed/ptbxl/seed42_fraction1/labeled_train.csv"
CACHE = ROOT / "data/processed/ptbxl/xecg_views"
CHECKPOINT = ROOT / "third_party/checkpoints/xecg"


def _sample() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select 64 fixed training records per class from the existing manifest."""
    rows = read_csv(MANIFEST)
    positives = [row for row in rows if row["target"] == "1"][:64]
    negatives = [row for row in rows if row["target"] == "0"][:64]
    selected = positives + negatives
    if len(selected) != 128:
        raise ValueError("Expected 64 training records in each class")
    metadata = json.loads((CACHE / "metadata.json").read_text())
    index = {int(value): i for i, value in enumerate(metadata["ecg_ids"])}
    ids = np.array([int(row["ecg_id"]) for row in selected])
    targets = np.array([int(row["target"]) for row in selected], dtype=np.float32)
    views = np.load(CACHE / "views.npy", mmap_mode="r")
    signals = np.asarray(views[[index[int(identifier)] for identifier in ids]], dtype=np.float32)
    if signals.shape != (128, 1000, 12) or not np.isfinite(signals).all():
        raise ValueError("Malformed xECG diagnostic sample")
    return signals, targets, ids


@torch.inference_mode()
def _pooled(model: XECGBinaryClassifier, signals: np.ndarray) -> np.ndarray:
    """Return pooled representations in fixed 16-record batches."""
    result = []
    for start in range(0, len(signals), 16):
        batch = torch.from_numpy(signals[start:start + 16]).to("cuda")
        pooled, _ = model.backbone(batch)
        result.append(pooled.float().cpu().numpy())
    return np.concatenate(result)


def _bce(logits: np.ndarray, targets: np.ndarray) -> float:
    """Compute binary cross-entropy from finite logits and fixed labels."""
    values = nn.functional.binary_cross_entropy_with_logits(
        torch.from_numpy(logits.astype(np.float32)), torch.from_numpy(targets))
    return float(values)


def diagnose() -> dict[str, object]:
    """Compare the same initial probe head under eval and train modes without updates."""
    require_cuda("cuda")
    torch.set_num_threads(1)
    signals, targets, ids = _sample()
    probe_path = OUTPUT / "probe.npz"
    receipt = json.loads((OUTPUT / "probe.json").read_text())
    if sha256_file(probe_path) != receipt["probe_sha256"]:
        raise ValueError("Frozen Experiment 016 probe changed")
    with gpu_lock("cuda", blocking=False):
        seed_everything(42)
        model = XECGBinaryClassifier(load_xecg(CHECKPOINT, backend="vanilla",
                                               device="cuda", drop_path_prob=0.5)).to("cuda")
        random_weight = model.head.weight.detach().cpu().numpy().reshape(-1).copy()
        random_bias = float(model.head.bias.detach().cpu().item())
        with np.load(probe_path) as probe:
            weight = probe["raw_weight"].astype(np.float32)
            bias = float(probe["raw_bias"])
        with torch.no_grad():
            model.head.weight.copy_(torch.from_numpy(weight[None]).to("cuda"))
            model.head.bias.fill_(bias)
        model.eval()
        eval_features = _pooled(model, signals)
        model.train()
        torch.manual_seed(42)
        train_features = _pooled(model, signals)
    eval_logits = eval_features @ weight + bias
    train_logits = train_features @ weight + bias
    random_eval_logits = eval_features @ random_weight + random_bias
    cosine = np.sum(eval_features * train_features, axis=1) / (
        np.linalg.norm(eval_features, axis=1) * np.linalg.norm(train_features, axis=1))
    result = {
        "kind": "read_only_initial_state_diagnostic",
        "sample": "first 64 positive and first 64 negative labeled training records",
        "sample_ecg_ids_sha256": sha256_json(ids.tolist()),
        "training_manifest_sha256": sha256_file(MANIFEST),
        "records": len(ids), "seed": 42, "drop_path_prob": 0.5,
        "probe_eval_bce": _bce(eval_logits, targets),
        "probe_train_mode_bce": _bce(train_logits, targets),
        "random_eval_bce": _bce(random_eval_logits, targets),
        "probe_eval_auroc": float(roc_auc_score(targets, eval_logits)),
        "probe_train_mode_auroc": float(roc_auc_score(targets, train_logits)),
        "mean_abs_logit_shift": float(np.mean(np.abs(train_logits - eval_logits))),
        "median_feature_cosine": float(np.median(cosine)),
        "probe_sha256": sha256_file(probe_path),
        "source_sha256": sha256_file(Path(__file__)),
    }
    write_json_atomic(OUTPUT / "initial_mode_diagnostic.json", result)
    return result


def main() -> None:
    """Run the frozen-weight diagnostic on the local V100."""
    print(json.dumps(diagnose()), flush=True)


if __name__ == "__main__":
    main()
