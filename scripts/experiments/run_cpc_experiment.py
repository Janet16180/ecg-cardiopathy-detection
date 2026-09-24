#!/usr/bin/env python3
"""Run matched local CPC/CPC+CMSC pretraining and PTB-XL fine-tuning."""

import argparse
import fcntl
import json
import math
import tempfile
import time
from pathlib import Path

import torch
from torch import nn

from ecg_experiment.cpc import CPCPretrainer
from ecg_experiment.cpc_pool import (
    DEFAULT_CACHE, DEFAULT_MANIFEST, ROOT, SSL_SEED, Pool, fine_tune, fingerprint, loader,
    make_source_hashes, resume_or_new, save_epoch,
)
from ecg_experiment.evaluation import paired_comparison
from ecg_experiment.files import write_json_atomic, write_torch_atomic
from ecg_experiment.gpu import GPU_LOCK_PATH
from ecg_experiment.reproducibility import cpu_state, seed_everything


DEFAULT_OUTPUT = ROOT / "outputs/experiment004_cpc_40k"


def pretrain(args, pool, mean, std, source_hashes, variant):
    seed_everything(SSL_SEED)
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
    write_json_atomic(directory / "config.json", {"fingerprint": fp, "inputs": inputs,
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
    write_torch_atomic(complete, {"fingerprint": fp, "encoder": cpu_state(model.encoder),
                            "variant": variant, "epochs": args.ssl_epochs, "seed": SSL_SEED,
                            "training_records": len(pool.train_rows), "parameters": sum(p.numel() for p in model.parameters())})
    return complete


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
    write_json_atomic(args.output_dir / "paired_comparisons.json", comparisons)
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")


def profile_updates(args, pool, mean, std):
    """Measure five warmup and N real optimizer updates with no saved checkpoint."""
    for variant in (("cpc", "hybrid") if args.variant == "all" else (args.variant,)):
        seed_everything(SSL_SEED)
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
    with GPU_LOCK_PATH.open("a+") as lock:
        if args.device == "cuda":
            print(f"Waiting for GPU lock {GPU_LOCK_PATH}", flush=True)
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
