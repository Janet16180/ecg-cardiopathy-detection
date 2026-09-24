#!/usr/bin/env python3
"""Run matched local CPC/CPC+CMSC pretraining and PTB-XL fine-tuning."""

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ecg_experiment.cpc import CPCClassifier, CPCPretrainer
from ecg_experiment.data import read_manifest
from ecg_experiment.evaluation import metrics
from ecg_experiment.run import cpu_state, evaluate_predictions, partition_validation


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / "data/processed/cpc_pool_40k"
DEFAULT_MANIFEST = ROOT / "data/processed/ptbxl"
DEFAULT_OUTPUT = ROOT / "outputs/experiment004_cpc_40k"
GPU_LOCK = Path("/tmp/ecg_project_gpu.lock")
SSL_SEED = 42


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state(generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])


class Pool:
    def __init__(self, directory):
        directory = Path(directory)
        self.directory = directory
        if not (directory / "complete.json").exists():
            raise FileNotFoundError(f"CPC cache is incomplete: {directory}")
        self.metadata = json.loads((directory / "complete.json").read_text())
        self.signals = np.load(directory / "signals.npy", mmap_mode="r")
        self.ids = np.load(directory / "ecg_ids.npy", allow_pickle=False)
        self.rows = read_manifest(directory / "rows.csv")
        if self.signals.dtype != np.float32 or self.signals.ndim != 3 or self.signals.shape[1:] != (12, 2500):
            raise ValueError("Expected float32 cache [N,12,2500]")
        if len(self.ids) != len(self.rows) or len(self.ids) != len(self.signals):
            raise ValueError("Cache ID, row, and signal counts differ")
        self.index = {str(ecg_id): i for i, ecg_id in enumerate(self.ids)}
        if len(self.index) != len(self.ids):
            raise ValueError("Duplicate cache ECG IDs")
        if any(str(self.ids[i]) != row["ecg_id"] for i, row in enumerate(self.rows)):
            raise ValueError("Cache rows are not aligned to signals")
        if any(row["split"] not in ("train", "validation", "test") for row in self.rows):
            raise ValueError("Unknown cache split")
        self.train_rows = [row for row in self.rows if row["split"] == "train"]

    def indices(self, rows):
        return [self.index[row["ecg_id"]] for row in rows]

    def normalization(self, output_dir, source_hashes=None):
        """Global per-lead mV mean/std over training records and every sample."""
        path = Path(output_dir) / "normalization.json"
        signal_path = str((self.directory / "signals.npy").resolve())
        signal_hash = source_hashes[signal_path] if source_hashes is not None else digest_file(signal_path)
        source = {"train_ids_sha256": digest_json([r["ecg_id"] for r in self.train_rows]),
                  "rows_sha256": digest_file(self.directory / "rows.csv"),
                  "signals_sha256": signal_hash,
                  "method": "global per-lead mean and population std, training waveforms only"}
        if path.exists():
            info = json.loads(path.read_text())
            if info["source"] != source:
                raise ValueError("Existing normalization uses different training data")
            return np.asarray(info["mean"], dtype=np.float32), np.asarray(info["std"], dtype=np.float32)
        totals = np.zeros(12, dtype=np.float64)
        squares = np.zeros(12, dtype=np.float64)
        indices = self.indices(self.train_rows)
        for start in range(0, len(indices), 128):
            block = np.asarray(self.signals[indices[start:start + 128]], dtype=np.float64)
            totals += block.sum(axis=(0, 2))
            squares += np.square(block).sum(axis=(0, 2))
        count = len(indices) * 2500
        mean = totals / count
        std = np.maximum(np.sqrt(np.maximum(squares / count - mean ** 2, 0)), 1e-6)
        atomic_json(path, {"source": source, "count": count,
                           "mean": mean.tolist(), "std": std.tolist()})
        return mean.astype(np.float32), std.astype(np.float32)


class PoolDataset(Dataset):
    def __init__(self, pool, rows, mean, std):
        self.pool, self.rows = pool, rows
        self.indices = pool.indices(rows)
        self.mean = mean[:, None]
        self.std = std[:, None]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        signal = np.array(self.pool.signals[self.indices[index]], copy=True)
        signal -= self.mean
        signal /= self.std
        return torch.from_numpy(signal), float(self.rows[index].get("target", -1)), self.rows[index]["patient_id"]


def loader(pool, rows, mean, std, batch_size, shuffle, generator, device):
    return DataLoader(PoolDataset(pool, rows, mean, std), batch_size=batch_size,
                      shuffle=shuffle, generator=generator, num_workers=0,
                      pin_memory=(device == "cuda"), drop_last=False)


def manifest_rows(pool, manifest_dir, budget):
    directory = Path(manifest_dir) / f"seed42_fraction{budget}"
    files = {name: directory / f"{name}.csv" for name in ("all_train_ssl", "labeled_train", "validation", "test")}
    rows = {name: read_manifest(path) for name, path in files.items()}
    for name, expected_split in (("all_train_ssl", "train"), ("labeled_train", "train"),
                                 ("validation", "validation"), ("test", "test")):
        for row in rows[name]:
            cache_row = pool.rows[pool.index[row["ecg_id"]]]
            if (cache_row["source"], cache_row["split"], cache_row["patient_id"]) != ("ptbxl", expected_split, row["patient_id"]):
                raise ValueError(f"PTB manifest/cache mismatch for {row['ecg_id']}")
    return rows, {name: digest_file(path) for name, path in files.items()}


def fingerprint(pool, source_hashes, mean, std, settings):
    data = {"cache": source_hashes, "normalization": {"mean": mean.tolist(), "std": std.tolist()},
            "settings": settings, "code": {name: digest_file(ROOT / name) for name in
            ("ecg_experiment/cpc.py", "scripts/experiments/run_cpc_experiment.py")}}
    return digest_json(data), data


def make_source_hashes(pool, manifest_dir):
    paths = [pool.directory / name for name in ("complete.json", "rows.csv", "ecg_ids.npy", "signals.npy")]
    paths += [Path(manifest_dir) / f"seed42_fraction{budget}" / f"{name}.csv"
              for budget in ("0.1", "1") for name in ("all_train_ssl", "labeled_train", "validation", "test")]
    hashes = {str(path.resolve()): digest_file(path) for path in paths}
    for filename, key in (("signals.npy", "signals_sha256"), ("rows.csv", "rows_sha256"),
                          ("ecg_ids.npy", "ecg_ids_sha256")):
        declared = pool.metadata.get(key)
        if declared is not None and hashes[str((pool.directory / filename).resolve())] != declared:
            raise ValueError(f"CPC cache hash mismatch for {filename}")
    for filename, expected in pool.metadata.get("ptb_manifest_sha256", {}).items():
        path = Path(manifest_dir) / "seed42_fraction1" / filename
        if str(path.resolve()) in hashes and hashes[str(path.resolve())] != expected:
            raise ValueError(f"PTB manifest hash mismatch for {filename}")
    return hashes


def resume_or_new(directory, fingerprint_value, model, optimizer, generator):
    state_path = directory / "epoch_state.pt"
    if not state_path.exists():
        if (directory / "history.json").exists():
            raise ValueError(f"History exists without resumable state in {directory}")
        return 0, [], None, -1.0, 0
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if state["fingerprint"] != fingerprint_value:
        raise ValueError(f"Resume fingerprint mismatch: {directory}")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    restore_rng(state["rng"], generator)
    return state["epoch"], state["history"], state.get("best_model"), state.get("best_auc", -1.0), state.get("best_epoch", 0)


def save_epoch(directory, fingerprint_value, epoch, model, optimizer, generator,
               history, best_model=None, best_auc=-1.0, best_epoch=0):
    atomic_torch(directory / "epoch_state.pt", {"fingerprint": fingerprint_value, "epoch": epoch,
                "model": cpu_state(model), "optimizer": optimizer.state_dict(),
                "rng": rng_state(generator), "history": history, "best_model": best_model,
                "best_auc": best_auc, "best_epoch": best_epoch})
    atomic_json(directory / "history.json", history)


def pretrain(args, pool, mean, std, source_hashes, variant):
    seed_all(SSL_SEED)
    directory = args.output_dir / f"{variant}_ssl"
    directory.mkdir(parents=True, exist_ok=True)
    settings = {"stage": "pretrain", "variant": variant, "seed": SSL_SEED,
                "epochs": args.ssl_epochs, "batch_size": args.ssl_batch_size,
                "optimizer": "AdamW", "lr": 1e-3, "weight_decay": 0.01,
                "warmup_epochs": 2, "cosine_floor": 0.1,
                "horizons": [4, 8, 12], "temperature": 0.1,
                "negative_exclusion_tokens": 3, "cmsc_weight": 0.1 if variant == "hybrid" else 0}
    fp, inputs = fingerprint(pool, source_hashes, mean, std, settings)
    if (directory / "config.json").exists() and json.loads((directory / "config.json").read_text())["fingerprint"] != fp:
        raise ValueError(f"Existing pretraining config fingerprint mismatch: {directory}")
    atomic_json(directory / "config.json", {"fingerprint": fp, "inputs": inputs,
                "description": "Local compact causal CNN/GRU CPC; not published S4 ECG-CPC"})
    complete = directory / "encoder.pt"
    if complete.exists():
        saved = torch.load(complete, map_location="cpu", weights_only=True)
        if saved["fingerprint"] != fp:
            raise ValueError(f"Completed SSL fingerprint mismatch: {complete}")
        return complete
    model = CPCPretrainer(hybrid=(variant == "hybrid")).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    generator = torch.Generator().manual_seed(SSL_SEED)
    data = loader(pool, pool.train_rows, mean, std, args.ssl_batch_size, True, generator, args.device)
    start_epoch, history, _, _, _ = resume_or_new(directory, fp, model, optimizer, generator)
    started = time.monotonic()
    previous_elapsed = history[-1]["elapsed_seconds"] if history else 0.0
    for epoch in range(start_epoch, args.ssl_epochs):
        model.train()
        lr = 1e-3 * min(1.0, (epoch + 1) / 2) * (0.1 + 0.9 * (1 + math.cos(math.pi * epoch / args.ssl_epochs)) / 2)
        optimizer.param_groups[0]["lr"] = lr
        totals = {"loss": 0.0, "cpc": 0.0, "cmsc": 0.0}
        count = 0
        updates = 0
        for signal, _, patients in data:
            signal = signal.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, details = model(signal, patients)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite SSL loss")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise RuntimeError("Nonfinite SSL gradients")
            optimizer.step()
            for name, value in {"loss": float(loss.detach()), **details}.items():
                totals[name] += value * len(signal)
            count += len(signal)
            updates += 1
        record = {"epoch": epoch + 1, "lr": lr, "elapsed_seconds": previous_elapsed + time.monotonic() - started,
                  "optimizer_updates": updates, "record_exposures": count,
                  **{key: value / count for key, value in totals.items()}}
        history.append(record)
        save_epoch(directory, fp, epoch + 1, model, optimizer, generator, history)
        print(json.dumps({"stage": "pretrain", "variant": variant, **record}), flush=True)
    atomic_torch(complete, {"fingerprint": fp, "encoder": cpu_state(model.encoder),
                            "variant": variant, "epochs": args.ssl_epochs, "seed": SSL_SEED,
                            "training_records": len(pool.train_rows), "parameters": sum(p.numel() for p in model.parameters())})
    return complete


@torch.inference_mode()
def predict(model, data, device):
    model.eval()
    return np.concatenate([model(signal.to(device, non_blocking=True)).cpu().numpy()
                           for signal, _, _ in data])


def fine_tune(args, pool, mean, std, source_hashes, variant, budget):
    rows, manifest_hashes = manifest_rows(pool, args.manifest_dir, budget)
    seed_all(SSL_SEED)
    directory = args.output_dir / f"{variant}_fraction{budget}_seed42"
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / f"{variant}_ssl" / "encoder.pt" if variant != "scratch" else None
    checkpoint_hash = digest_file(checkpoint) if checkpoint else None
    settings = {"stage": "train", "variant": variant, "budget": budget, "seed": SSL_SEED,
                "epochs": args.epochs, "patience": args.patience, "batch_size": args.batch_size,
                "encoder_lr": 3e-4, "head_lr": 1e-3, "weight_decay": 0.01,
                "augmentation": "none", "ssl_checkpoint_sha256": checkpoint_hash,
                "manifest_sha256": manifest_hashes}
    fp, inputs = fingerprint(pool, source_hashes, mean, std, settings)
    completion = directory / "completion.json"
    if completion.exists():
        saved = json.loads(completion.read_text())
        if saved["fingerprint"] != fp:
            raise ValueError(f"Completed training fingerprint mismatch: {directory}")
        for name, expected_hash in saved["artifacts"].items():
            if digest_file(directory / name) != expected_hash:
                raise ValueError(f"Completed artifact changed: {directory / name}")
        return
    model = CPCClassifier().to(args.device)
    if checkpoint:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["variant"] != variant or saved["epochs"] != args.ssl_epochs:
            raise ValueError("SSL checkpoint variant or duration mismatch")
        model.encoder.load_state_dict(saved["encoder"])
    optimizer = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": 3e-4},
                                   {"params": model.head.parameters(), "lr": 1e-3}], weight_decay=0.01)
    generator = torch.Generator().manual_seed(SSL_SEED)
    development, calibration = partition_validation(rows["validation"])
    train_data = loader(pool, rows["labeled_train"], mean, std, args.batch_size, True, generator, args.device)
    dev_data = loader(pool, development, mean, std, args.batch_size, False, None, args.device)
    start_epoch, history, best_model, best_auc, best_epoch = resume_or_new(directory, fp, model, optimizer, generator)
    atomic_json(directory / "config.json", {"fingerprint": fp, "inputs": inputs,
                "model_parameters": sum(p.numel() for p in model.parameters()),
                "encoder_parameters": sum(p.numel() for p in model.encoder.parameters()),
                "labeled_training_records": len(rows["labeled_train"]),
                "development_records": len(development), "calibration_records": len(calibration),
                "test_records": len(rows["test"]), "description": "Mean/max context pooling over each half, then average halves"})
    dev_y = np.array([int(row["target"]) for row in development])
    started = time.monotonic()
    previous_elapsed = history[-1]["elapsed_seconds"] if history else 0.0
    for epoch in range(start_epoch, args.epochs):
        if epoch - best_epoch >= args.patience:
            break
        model.train()
        total = 0.0
        updates = 0
        exposures = 0
        for signal, target, _ in train_data:
            signal = signal.to(args.device, non_blocking=True)
            target = target.to(args.device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(signal), target)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite supervised loss")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise RuntimeError("Nonfinite supervised gradients")
            optimizer.step()
            total += float(loss.detach()) * len(signal)
            updates += 1
            exposures += len(signal)
        dev_auc = float(roc_auc_score(dev_y, predict(model, dev_data, args.device)))
        if dev_auc > best_auc:
            best_auc, best_epoch, best_model = dev_auc, epoch + 1, cpu_state(model)
        record = {"epoch": epoch + 1, "train_loss": total / len(rows["labeled_train"]),
                  "development_auroc": dev_auc, "best_epoch": best_epoch,
                  "optimizer_updates": updates, "record_exposures": exposures,
                  "elapsed_seconds": previous_elapsed + time.monotonic() - started}
        history.append(record)
        save_epoch(directory, fp, epoch + 1, model, optimizer, generator, history,
                   best_model, best_auc, best_epoch)
        print(json.dumps({"stage": "train", "variant": variant, "budget": budget, **record}), flush=True)
    if best_model is None:
        raise RuntimeError("No supervised epoch completed")
    model.load_state_dict(best_model)
    atomic_torch(directory / "model.pt", {"fingerprint": fp, "model": best_model,
                 "best_epoch": best_epoch, "best_development_auroc": best_auc})
    calibration_logits = predict(model, loader(pool, calibration, mean, std, args.batch_size, False, None, args.device), args.device)
    test_logits = predict(model, loader(pool, rows["test"], mean, std, args.batch_size, False, None, args.device), args.device)
    evaluate_predictions(f"{variant}_fraction{budget}", calibration_logits, test_logits,
                         calibration, rows["test"], directory, SSL_SEED, args.bootstrap)
    artifact_names = ("config.json", "history.json", "model.pt", "metrics.json",
                      "test_predictions.csv", "calibration_predictions.npz")
    atomic_json(completion, {"fingerprint": fp,
                             "artifacts": {name: digest_file(directory / name) for name in artifact_names}})


def paired_comparison(left, right, repeats=500):
    def read_predictions(directory):
        with (directory / "test_predictions.csv").open(newline="") as handle:
            return list(csv.DictReader(handle))
    left_rows, right_rows = read_predictions(left), read_predictions(right)
    if [(r["ecg_id"], r["patient_id"], r["target"]) for r in left_rows] != [
            (r["ecg_id"], r["patient_id"], r["target"]) for r in right_rows]:
        raise ValueError("Cannot pair test predictions with different rows")
    y = np.array([int(r["target"]) for r in left_rows])
    ids = np.array([r["patient_id"] for r in left_rows])
    left_p = np.array([float(r["probability"]) for r in left_rows])
    right_p = np.array([float(r["probability"]) for r in right_rows])
    left_t = json.loads((left / "metrics.json").read_text())["threshold"]
    right_t = json.loads((right / "metrics.json").read_text())["threshold"]
    groups = [np.flatnonzero(ids == patient) for patient in np.unique(ids)]
    rng = np.random.default_rng(2026)
    names = ("auroc", "average_precision", "sensitivity", "specificity")
    point_left, point_right = metrics(y, left_p, left_t), metrics(y, right_p, right_t)
    samples = {name: [] for name in names}
    for _ in range(repeats):
        sampled = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        if len(np.unique(y[sampled])) < 2:
            continue
        a, b = metrics(y[sampled], left_p[sampled], left_t), metrics(y[sampled], right_p[sampled], right_t)
        for name in names:
            samples[name].append(a[name] - b[name])
    return {name: {"difference": point_left[name] - point_right[name],
                   "ci95": np.quantile(samples[name], [0.025, 0.975]).tolist()}
            for name in names}


def report(args):
    comparisons = {}
    lines = ["# Local CPC experiment on the 40k ECG training pool", "",
             "Compact causal CNN/GRU CPC experiment; this is not an S4 ECG-CPC reproduction.", "",
             f"All models use the same architecture and PTB-XL label manifests. CPC and CPC+CMSC use {args.ssl_epochs} fixed SSL epochs. The CMSC arm adds a 0.1-weighted cross-half contrastive loss. Thresholds use held-out calibration patients; CIs resample test patients.", ""]
    for budget in ("0.1", "1"):
        paths = {variant: args.output_dir / f"{variant}_fraction{budget}_seed42"
                 for variant in ("scratch", "cpc", "hybrid")}
        if not all((path / "completion.json").exists() for path in paths.values()):
            continue
        lines += [f"## {budget} label fraction", "", "| Arm | AUROC | AP | Sensitivity | Specificity |", "| --- | ---: | ---: | ---: | ---: |"]
        for variant, path in paths.items():
            result = json.loads((path / "metrics.json").read_text())["test"]
            lines.append(f"| {variant} | {result['auroc']:.3f} | {result['average_precision']:.3f} | {result['sensitivity']:.3f} | {result['specificity']:.3f} |")
        lines.append("")
        for left, right in (("hybrid", "cpc"), ("cpc", "scratch"), ("hybrid", "scratch")):
            comparison = paired_comparison(paths[left], paths[right], args.bootstrap)
            comparisons[f"fraction{budget}_{left}_minus_{right}"] = comparison
            auc = comparison["auroc"]
            lines.append(f"{left} minus {right}: AUROC {auc['difference']:+.3f} "
                         f"(paired patient 95% CI {auc['ci95'][0]:+.3f} to {auc['ci95'][1]:+.3f}).")
        lines.append("")
    atomic_json(args.output_dir / "paired_comparisons.json", comparisons)
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")


def profile_updates(args, pool, mean, std):
    """Measure five warmup and N real optimizer updates with no saved checkpoint."""
    for variant in (("cpc", "hybrid") if args.variant == "all" else (args.variant,)):
        seed_all(SSL_SEED)
        model = CPCPretrainer(hybrid=(variant == "hybrid")).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        generator = torch.Generator().manual_seed(SSL_SEED)
        data = loader(pool, pool.train_rows, mean, std, args.ssl_batch_size, True, generator, args.device)
        measured_records = 0
        started = None
        for update, (signal, _, patients) in enumerate(data, 1):
            if update == 6:
                if args.device == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                started = time.monotonic()
            signal = signal.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, details = model(signal, patients)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite profile loss")
            loss.backward()
            optimizer.step()
            if update > 5:
                measured_records += len(signal)
            if update >= 5 + args.profile_updates:
                break
        if started is None:
            raise RuntimeError("Pool is too small for the requested profile")
        if args.device == "cuda":
            torch.cuda.synchronize()
        seconds = time.monotonic() - started
        print(json.dumps({"stage": "profile", "variant": variant,
                          "model_parameters": sum(p.numel() for p in model.parameters()),
                          "encoder_parameters": sum(p.numel() for p in model.encoder.parameters()),
                          "warmup_updates": 5, "measured_updates": update - 5,
                          "measured_records": measured_records, "measured_seconds": seconds,
                          "seconds_per_update": seconds / (update - 5),
                          "records_per_second": measured_records / seconds,
                          "estimated_ssl_epoch_seconds": len(pool.train_rows) / (measured_records / seconds),
                          "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9 if args.device == "cuda" else None,
                          "loss": float(loss.detach()), **details}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("profile", "pretrain", "train", "all"), default="all")
    parser.add_argument("--variant", choices=("cpc", "hybrid", "scratch", "all"), default="all")
    parser.add_argument("--labels", choices=("0.1", "1", "all"), default="all")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ssl-batch-size", type=int, default=128)
    parser.add_argument("--ssl-epochs", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--profile-updates", type=int, default=20,
                        help="Measured SSL updates after five warmup updates in profile stage")
    args = parser.parse_args()
    if min(args.threads, args.batch_size, args.ssl_batch_size, args.ssl_epochs, args.epochs, args.patience, args.bootstrap) < 1:
        parser.error("Thread, batch, epoch, patience, and bootstrap counts must be positive")
    if args.stage in ("profile", "pretrain") and args.variant == "scratch":
        parser.error("Scratch has no profile or pretraining stage")
    if args.profile_updates < 1:
        parser.error("profile-updates must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    torch.set_num_threads(args.threads)
    if args.stage != "profile":
        args.output_dir.mkdir(parents=True, exist_ok=True)
    with GPU_LOCK.open("a+") as lock:
        if args.device == "cuda":
            print(f"Waiting for GPU lock {GPU_LOCK}", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
        pool = Pool(args.cache_dir)
        source_hashes = make_source_hashes(pool, args.manifest_dir)
        if args.stage == "profile":
            with tempfile.TemporaryDirectory(prefix="cpc_profile_") as temporary:
                mean, std = pool.normalization(temporary, source_hashes)
                profile_updates(args, pool, mean, std)
            return
        mean, std = pool.normalization(args.output_dir, source_hashes)
        if args.stage in ("pretrain", "all"):
            for variant in (("cpc", "hybrid") if args.variant == "all" else (args.variant,)):
                if variant != "scratch":
                    pretrain(args, pool, mean, std, source_hashes, variant)
        if args.stage in ("train", "all"):
            for budget in (("1", "0.1") if args.labels == "all" else (args.labels,)):
                for variant in (("scratch", "cpc", "hybrid") if args.variant == "all" else (args.variant,)):
                    fine_tune(args, pool, mean, std, source_hashes, variant, budget)
            report(args)


if __name__ == "__main__":
    main()
