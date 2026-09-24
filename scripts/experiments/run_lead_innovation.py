#!/usr/bin/env python3
"""Run the prespecified eight-lead multiscale SSL ablation and classifiers."""

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ecg_experiment.data import ECGDataset, Waveforms, read_manifest
from ecg_experiment.lead_innovation import LEAD_INDICES, LeadClassifier, LeadMultiscaleEncoder, LeadSSL
from ecg_experiment.run import cpu_state, evaluate_predictions, partition_validation, save_json


NAMES = {"supervised": "lead_multiscale_supervised",
         "ordinary": "lead_multiscale_latent",
         "innovation": "lead_multiscale_innovation"}
SSL_SEED = 42


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EightLeadDataset(Dataset):
    def __init__(self, waveforms, rows, scale):
        self.base = ECGDataset(waveforms, rows, scale)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        signal, target = self.base[index]
        return signal[list(LEAD_INDICES)], target


def make_loader(waveforms, rows, scale, batch_size, shuffle=False):
    return DataLoader(EightLeadDataset(waveforms, rows, scale), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0, drop_last=False)


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    return np.concatenate([model(signal.to(device)).cpu().numpy() for signal, _ in loader])


def ssl_checkpoint(args):
    return args.output_dir / f"lead_multiscale_{args.variant}_ssl" / "encoder.pt"


def pretrain(args, waveforms, all_train, scale):
    seed_everything(SSL_SEED)
    checkpoint = ssl_checkpoint(args)
    directory = checkpoint.parent
    directory.mkdir(parents=True, exist_ok=True)
    expected_ids = [row["ecg_id"] for row in all_train]
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (state["variant"] != args.variant or state["seed"] != SSL_SEED
                or state["epochs"] != args.ssl_epochs or state["training_ecg_ids"] != expected_ids
                or not np.allclose(state["scale"], scale)):
            raise ValueError(f"SSL checkpoint does not match inputs/settings: {checkpoint}")
        print(f"Using completed SSL checkpoint {checkpoint}", flush=True)
        return checkpoint
    if (directory / "history.json").exists():
        raise FileExistsError(f"Incomplete SSL run in {directory}; use a new output directory")
    model = LeadSSL(0.0 if args.variant == "ordinary" else 0.5).to(args.device)
    loader = make_loader(waveforms, all_train, scale, args.ssl_batch_size, shuffle=True)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters()
                                   if parameter.requires_grad), lr=3e-4, weight_decay=0.01)
    history = []
    started = time.monotonic()
    for epoch in range(args.ssl_epochs):
        model.train()
        lr = 3e-4 * min(1.0, (epoch + 1) / 2) * (
            0.1 + 0.9 * (1 + math.cos(math.pi * epoch / args.ssl_epochs)) / 2)
        for group in optimizer.param_groups:
            group["lr"] = lr
        totals, observations = {}, 0
        for signal, _ in loader:
            signal = signal.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss, details = model(signal)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite SSL loss in {args.variant}")
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise RuntimeError(f"Nonfinite SSL gradients in {args.variant}")
            optimizer.step()
            model.update_teacher(0.99 + 0.009 * epoch / max(1, args.ssl_epochs - 1))
            for key, value in {"loss": float(loss.detach()), **details}.items():
                totals[key] = totals.get(key, 0.0) + value * len(signal)
            observations += len(signal)
        record = {"epoch": epoch + 1, "lr": lr, "seconds": time.monotonic() - started,
                  **{key: value / observations for key, value in totals.items()}}
        history.append(record)
        save_json(directory / "history.json", history)
        print(json.dumps({"stage": f"lead_{args.variant}_ssl", **record}), flush=True)
    torch.save({"encoder": cpu_state(model.encoder), "scale": scale.tolist(),
                "variant": args.variant, "seed": SSL_SEED, "epochs": args.ssl_epochs,
                "training_ecg_ids": expected_ids}, checkpoint)
    save_json(directory / "config.json", {
        "architecture": repr(model), "variant": args.variant,
        "ordinary_weight": 1.0, "innovation_weight": model.innovation_weight,
        "variance_weight": 0.1, "ssl_epochs": args.ssl_epochs,
        "ssl_batch_size": args.ssl_batch_size, "seed": SSL_SEED,
        "training_records": len(all_train), "seconds": time.monotonic() - started,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "encoder_parameters": sum(p.numel() for p in model.encoder.parameters()),
        "mask": "One whole lead and two distinct leads with contiguous 2-second spans; raw samples hidden before both branches",
        "target": "Stop-gradient teacher tokens batch-centered per lead/time coordinate before LayerNorm; innovation subtracts the seven other centered leads at the same time",
        "target_batch_dependence": "Teacher content targets depend on the current training batch; batches require at least two records",
        "checkpoint": str(checkpoint)})
    return checkpoint


def supervised(args, waveforms, train, validation, test, all_train, scale):
    seed_everything(args.seed)
    name = NAMES[args.variant]
    directory = args.output_dir / f"{name}_seed{args.seed}"
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    encoder = LeadMultiscaleEncoder()
    checkpoint = None
    if args.variant != "supervised":
        checkpoint = ssl_checkpoint(args)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (state["variant"] != args.variant or state["seed"] != SSL_SEED
                or state["epochs"] != args.ssl_epochs
                or state["training_ecg_ids"] != [row["ecg_id"] for row in all_train]
                or not np.allclose(state["scale"], scale)):
            raise ValueError(f"SSL checkpoint does not match this experiment: {checkpoint}")
        encoder.load_state_dict(state["encoder"])
    model = LeadClassifier(encoder).to(args.device)
    development, calibration = partition_validation(validation)
    train_loader = make_loader(waveforms, train, scale, args.batch_size, shuffle=True)
    dev_loader = make_loader(waveforms, development, scale, args.batch_size)
    development_y = np.array([int(row["target"]) for row in development])
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": 3e-4},
        {"params": model.head.parameters(), "lr": 1e-3}], weight_decay=0.01)
    best_auc, best_epoch, best_state, history = -1.0, 0, None, []
    started = time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for signal, target in train_loader:
            signal, target = signal.to(args.device), target.to(args.device)
            gain = torch.empty(len(signal), 1, 1, device=args.device).uniform_(0.9, 1.1)
            signal = signal * gain + torch.randn_like(signal) * 0.01
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(signal), target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite supervised loss in {name}")
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise RuntimeError(f"Nonfinite supervised gradients in {name}")
            optimizer.step()
            total_loss += float(loss.detach()) * len(signal)
        dev_auc = float(roc_auc_score(development_y, predict(model, dev_loader, args.device)))
        if dev_auc > best_auc:
            best_auc, best_epoch, best_state = dev_auc, epoch + 1, cpu_state(model)
        record = {"epoch": epoch + 1, "loss": total_loss / len(train),
                  "development_auroc": dev_auc, "seconds": time.monotonic() - started}
        history.append(record)
        save_json(directory / "history.json", history)
        print(json.dumps({"stage": name, "label_seed": args.seed, **record}), flush=True)
        if epoch + 1 - best_epoch >= args.patience:
            break
    model.load_state_dict(best_state)
    torch.save({"model": best_state, "scale": scale.tolist(), "best_epoch": best_epoch,
                "label_seed": args.seed, "architecture": name,
                "ssl_checkpoint": str(checkpoint) if checkpoint else None}, directory / "model.pt")
    evaluate_predictions(name,
        predict(model, make_loader(waveforms, calibration, scale, args.batch_size), args.device),
        predict(model, make_loader(waveforms, test, scale, args.batch_size), args.device),
        calibration, test, directory, args.seed, args.bootstrap)
    save_json(directory / "config.json", {
        "architecture": repr(model), "variant": args.variant,
        "parameters": sum(p.numel() for p in model.parameters()),
        "encoder_parameters": sum(p.numel() for p in model.encoder.parameters()),
        "label_seed": args.seed, "labeled_training_records": len(train),
        "development_records": len(development), "calibration_records": len(calibration),
        "test_records": len(test), "epochs_budget": args.epochs,
        "patience": args.patience, "best_epoch": best_epoch,
        "best_development_auroc": best_auc, "batch_size": args.batch_size,
        "ssl_checkpoint": str(checkpoint) if checkpoint else None,
        "ssl_seed": SSL_SEED if checkpoint else None,
        "ssl_epochs": args.ssl_epochs if checkpoint else None,
        "label_manifest_sha256": hashlib.sha256((args.manifest_dir / "labeled_train.csv").read_bytes()).hexdigest(),
        "normalization": "Per-record per-lead demeaning, training-only per-lead RMS scaling, clipping +/-20; then I, II, V1-V6",
        "pretraining_exposure": "Own SSL uses official PTB-XL folds 1-8 only; no validation/test waveforms",
        "hypothesis": "Lead-residual latent prediction may preserve lead-specific morphology beyond shared rhythm; this is not physiological proof",
        "device": args.device, "torch_version": torch.__version__,
        "seconds": time.monotonic() - started})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("ssl", "train"), required=True)
    parser.add_argument("--variant", choices=tuple(NAMES), required=True)
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/processed/ptbxl/seed42_fraction0.1"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/processed/ptbxl/waveforms100"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiment001"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ssl-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--ssl-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    if min(args.threads, args.batch_size, args.ssl_batch_size, args.epochs,
           args.ssl_epochs, args.patience, args.bootstrap) < 1:
        parser.error("threads, batch sizes, epochs, patience, and bootstrap must be positive")
    if args.stage == "ssl" and args.ssl_batch_size < 2:
        parser.error("SSL batch size must be at least two for batch-centered teacher targets")
    if args.stage == "ssl" and args.variant == "supervised":
        parser.error("supervised variant has no SSL stage")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    waveforms = Waveforms(args.cache_dir)
    all_train = read_manifest(args.manifest_dir / "all_train_ssl.csv")
    scale = waveforms.training_scale(all_train)
    if args.stage == "ssl":
        pretrain(args, waveforms, all_train, scale)
    else:
        supervised(args, waveforms,
                   read_manifest(args.manifest_dir / "labeled_train.csv"),
                   read_manifest(args.manifest_dir / "validation.csv"),
                   read_manifest(args.manifest_dir / "test.csv"), all_train, scale)


if __name__ == "__main__":
    main()
