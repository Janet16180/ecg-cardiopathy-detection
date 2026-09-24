"""Train and evaluate the compact PTB-XL experiment on fixed patient manifests."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from .data import ECGDataset, Waveforms, build_cache, read_manifest
from .evaluation import evaluate_predictions, partition_validation
from .files import write_json_atomic
from .models import CNN, JEPA, Classifier, MaskedAutoencoder, PatchTransformer
from .reproducibility import cpu_state, seed_everything


def make_loader(waveforms, rows, scale, batch_size, shuffle=False):
    return DataLoader(ECGDataset(waveforms, rows, scale), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0, drop_last=False)


def pretrain(args, waveforms, train, scale):
    seed_everything(args.seed)
    directory = args.output_dir / f"{args.model}_ssl"
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "encoder.pt"
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (state["model"] != args.model or state["seed"] != args.seed
                or state["epochs"] != args.ssl_epochs
                or state["training_ecg_ids"] != [r["ecg_id"] for r in train]
                or not np.allclose(scale, state["scale"])):
            raise ValueError("Existing SSL checkpoint differs from this experiment's inputs/settings")
        print(f"Using completed SSL checkpoint {checkpoint}", flush=True)
        return checkpoint
    model = (MaskedAutoencoder() if args.model == "mae" else JEPA()).to(args.device)
    loader = make_loader(waveforms, train, scale, args.ssl_batch_size, shuffle=True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=3e-4, weight_decay=0.01)
    history = []
    start = time.monotonic()
    for epoch in range(args.ssl_epochs):
        model.train()
        lr = 3e-4 * min(1.0, (epoch + 1) / 2) * (0.1 + 0.9 * (1 + math.cos(math.pi * epoch / args.ssl_epochs)) / 2)
        for group in optimizer.param_groups:
            group["lr"] = lr
        totals = {}
        observations = 0
        for signal, _ in loader:
            signal = signal.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss, details = model(signal)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite SSL loss in {args.model}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if isinstance(model, JEPA):
                model.update_teacher(0.99 + 0.009 * epoch / max(1, args.ssl_epochs - 1))
            totals["loss"] = totals.get("loss", 0) + float(loss.detach()) * len(signal)
            for key, value in details.items():
                totals[key] = totals.get(key, 0) + value * len(signal)
            observations += len(signal)
        row = {"epoch": epoch + 1, "lr": lr, "seconds": time.monotonic() - start,
               **{key: value / observations for key, value in totals.items()}}
        history.append(row)
        print(json.dumps({"stage": args.model + "_ssl", **row}), flush=True)
        write_json_atomic(directory / "history.json", history)
    torch.save({"encoder": cpu_state(model.encoder), "scale": scale.tolist(),
                "model": args.model, "seed": args.seed, "epochs": args.ssl_epochs,
                "training_records": len(train), "training_ecg_ids": [r["ecg_id"] for r in train]}, checkpoint)
    write_json_atomic(directory / "config.json", {"architecture": repr(model), "ssl_epochs": args.ssl_epochs,
               "ssl_batch_size": args.ssl_batch_size, "seed": args.seed,
               "seconds": time.monotonic() - start, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})
    return checkpoint


@torch.inference_mode()
def predict(model, loader):
    model.eval()
    device = next(model.parameters()).device
    return np.concatenate([model(signal.to(device)).cpu().numpy() for signal, _ in loader])


def supervised(args, waveforms, train, validation, test, scale, ssl_checkpoint=None):
    seed_everything(args.seed)
    name = {"cnn": "cnn_supervised", "transformer": "transformer_supervised",
            "mae": "mae_finetuned", "jepa": "jepa_finetuned"}[args.model]
    directory = args.output_dir / f"{name}_seed{args.seed}"
    if directory.exists() and (directory / "metrics.json").exists():
        raise FileExistsError(f"Completed results already exist at {directory}; choose a new output directory")
    directory.mkdir(parents=True, exist_ok=True)
    encoder = CNN() if args.model == "cnn" else PatchTransformer()
    if ssl_checkpoint:
        state = torch.load(ssl_checkpoint, map_location="cpu", weights_only=True)
        expected_training = [r["ecg_id"] for r in read_manifest(args.manifest_dir / "all_train_ssl.csv")]
        if state["model"] != args.model or state["training_ecg_ids"] != expected_training:
            raise ValueError("SSL checkpoint has a different model or training patient manifest")
        encoder.load_state_dict(state["encoder"])
        if not np.allclose(scale, state["scale"]):
            raise ValueError("SSL and supervised normalization do not match")
    model = Classifier(encoder).to(args.device)
    development, calibration = partition_validation(validation)
    train_loader = make_loader(waveforms, train, scale, args.batch_size, shuffle=True)
    dev_loader = make_loader(waveforms, development, scale, args.batch_size)
    dev_y = np.array([int(row["target"]) for row in development])
    # Match the optimizer across transformer scratch/MAE/JEPA to isolate SSL initialization.
    encoder_lr = 3e-4
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": encoder_lr},
        {"params": model.head.parameters(), "lr": 1e-3}], weight_decay=0.01)
    best_auc, best_epoch, best_state, history = -1, 0, None, []
    start = time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        # Gentle noise/gain augmentation preserves lead relationships and rhythm timing.
        for signal, target in train_loader:
            signal, target = signal.to(args.device), target.to(args.device)
            gain = torch.empty(len(signal), 1, 1, device=args.device).uniform_(0.9, 1.1)
            signal = signal * gain + torch.randn_like(signal) * 0.01
            optimizer.zero_grad(set_to_none=True)
            logits = model(signal)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite supervised loss in {name}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(signal)
        dev_auc = float(roc_auc_score(dev_y, predict(model, dev_loader)))
        if dev_auc > best_auc:
            best_auc, best_epoch, best_state = dev_auc, epoch + 1, cpu_state(model)
        row = {"epoch": epoch + 1, "loss": total_loss / len(train), "development_auroc": dev_auc,
               "seconds": time.monotonic() - start}
        history.append(row)
        print(json.dumps({"stage": name, "label_seed": args.seed, **row}), flush=True)
        write_json_atomic(directory / "history.json", history)
        if epoch + 1 - best_epoch >= args.patience:
            break
    model.load_state_dict(best_state)
    torch.save({"model": best_state, "scale": scale.tolist(), "best_epoch": best_epoch,
                "label_seed": args.seed, "architecture": name}, directory / "model.pt")
    # No test labels have been loaded into model fitting or checkpoint selection.
    result = evaluate_predictions(name,
        predict(model, make_loader(waveforms, calibration, scale, args.batch_size)),
        predict(model, make_loader(waveforms, test, scale, args.batch_size)),
        calibration, test, directory, args.seed, args.bootstrap)
    config = {"architecture": repr(model), "parameters": sum(p.numel() for p in model.parameters()),
              "epochs_budget": args.epochs, "patience": args.patience, "best_epoch": best_epoch,
              "best_development_auroc": best_auc, "labeled_training_records": len(train),
              "development_records": len(development), "calibration_records": len(calibration),
              "test_records": len(test), "seed": args.seed, "torch_threads": args.threads,
              "device": args.device,
              "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
              "torch_version": str(torch.__version__), "seconds": time.monotonic() - start,
              "ssl_checkpoint": str(ssl_checkpoint) if ssl_checkpoint else None,
              "label_manifest_sha256": hashlib.sha256((args.manifest_dir / "labeled_train.csv").read_bytes()).hexdigest(),
              "normalization": "Per-record per-lead demeaning, training-only per-lead RMS scaling, clipping +/-20",
              "pretraining_exposure": "Own SSL uses official folds1-8 only; no validation/test waveforms"}
    write_json_atomic(directory / "config.json", config)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["cache", "ssl", "train"], default="train")
    parser.add_argument("--model", choices=["cnn", "transformer", "mae", "jepa"], default="cnn")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/ptb-xl/1.0.3"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/processed/ptbxl/seed42_fraction0.1"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/processed/ptbxl/waveforms100"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiment001"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ssl-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--ssl-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--ssl-checkpoint", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.stage == "cache":
        build_cache(args.raw_dir, args.manifest_dir, args.cache_dir)
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Run with GPU access or explicitly --device cpu.")
    waveforms = Waveforms(args.cache_dir)
    all_train = read_manifest(args.manifest_dir / "all_train_ssl.csv")
    scale = waveforms.training_scale(all_train)
    ssl_checkpoint = args.ssl_checkpoint
    if args.model in {"mae", "jepa"} and not ssl_checkpoint:
        ssl_checkpoint = pretrain(args, waveforms, all_train, scale)
    if args.stage == "ssl":
        if args.model not in {"mae", "jepa"}:
            raise ValueError("SSL requires --model mae or jepa")
        return
    supervised(args, waveforms, read_manifest(args.manifest_dir / "labeled_train.csv"),
               read_manifest(args.manifest_dir / "validation.csv"),
               read_manifest(args.manifest_dir / "test.csv"), scale, ssl_checkpoint)


if __name__ == "__main__":
    main()
