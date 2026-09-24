#!/usr/bin/env python3
"""Run matched sampled InfoNCE-16 and word2vec-style SGNS-16 ECG CPC arms."""

import argparse
import fcntl
import json
import math
import tempfile
import time
from pathlib import Path

import torch
from torch import nn

from ecg_experiment.cpc_word2vec import SampledCPCPretrainer
from scripts import run_cpc_experiment as base


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/experiment005_cpc_word2vec"
VARIANTS = ("sampled_info", "sgns")
SAMPLER_SEED = 424242


def source_hashes_with_new_code(pool, manifest_dir):
    hashes = base.make_source_hashes(pool, manifest_dir)
    for name in ("ecg_experiment/cpc_word2vec.py", "scripts/run_cpc_word2vec.py"):
        path = ROOT / name
        hashes[str(path.resolve())] = base.digest_file(path)
    return hashes


def settings(args, variant):
    return {"stage": "pretrain", "variant": variant, "seed": base.SSL_SEED,
            "sampler_seed": SAMPLER_SEED, "epochs": args.ssl_epochs,
            "batch_size": args.ssl_batch_size, "optimizer": "AdamW",
            "lr": 1e-3, "weight_decay": 0.01, "warmup_epochs": 2,
            "cosine_floor": 0.1, "horizons": [4, 8, 12],
            "temperature": 0.1, "negative_count": 16,
            "negative_sampling": "Uniform with replacement, same half, exclude target and positions within +/-3 tokens",
            "objective": ("-log(exp(pos)/(exp(pos)+sum_16 exp(neg)))" if variant == "sampled_info"
                          else "softplus(-pos)+sum_16 softplus(neg); mean over queries and horizons"),
            "normalization": "Global train-only per-lead mean/std; no augmentation"}


def resume_or_new(directory, fingerprint, model, optimizer, loader_generator,
                  sampler_generator):
    state_path = directory / "epoch_state.pt"
    if not state_path.exists():
        if (directory / "history.json").exists():
            raise ValueError(f"History exists without resumable state: {directory}")
        return 0, []
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if state["fingerprint"] != fingerprint:
        raise ValueError(f"Resume fingerprint mismatch: {directory}")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    base.restore_rng(state["rng"], loader_generator)
    sampler_generator.set_state(state["sampler_rng"])
    return state["epoch"], state["history"]


def save_epoch(directory, fingerprint, epoch, model, optimizer, loader_generator,
               sampler_generator, history):
    base.atomic_torch(directory / "epoch_state.pt", {
        "fingerprint": fingerprint, "epoch": epoch,
        "model": base.cpu_state(model), "optimizer": optimizer.state_dict(),
        "rng": base.rng_state(loader_generator),
        "sampler_rng": sampler_generator.get_state(), "history": history})
    base.atomic_json(directory / "history.json", history)


def pretrain(args, pool, mean, std, source_hashes, variant):
    base.seed_all(base.SSL_SEED)
    directory = args.output_dir / f"{variant}_ssl"
    directory.mkdir(parents=True, exist_ok=True)
    fp, inputs = base.fingerprint(pool, source_hashes, mean, std, settings(args, variant))
    config_path = directory / "config.json"
    if config_path.exists() and json.loads(config_path.read_text())["fingerprint"] != fp:
        raise ValueError(f"Existing SSL config fingerprint mismatch: {directory}")
    base.atomic_json(config_path, {"fingerprint": fp, "inputs": inputs,
             "description": "Same compact causal CNN/GRU encoder and heads as experiment 004; sampled losses only",
             "loss_scale_note": "SGNS sums sixteen negative terms; raw loss is not comparable to InfoNCE"})
    complete = directory / "encoder.pt"
    if complete.exists():
        saved = torch.load(complete, map_location="cpu", weights_only=True)
        if saved["fingerprint"] != fp:
            raise ValueError(f"Completed SSL fingerprint mismatch: {complete}")
        return complete
    model = SampledCPCPretrainer(variant).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    loader_generator = torch.Generator().manual_seed(base.SSL_SEED)
    sampler_generator = torch.Generator().manual_seed(SAMPLER_SEED)
    data = base.loader(pool, pool.train_rows, mean, std, args.ssl_batch_size,
                       True, loader_generator, args.device)
    start_epoch, history = resume_or_new(directory, fp, model, optimizer,
                                         loader_generator, sampler_generator)
    started = time.monotonic()
    previous_elapsed = history[-1]["elapsed_seconds"] if history else 0.0
    for epoch in range(start_epoch, args.ssl_epochs):
        model.train()
        lr = 1e-3 * min(1.0, (epoch + 1) / 2) * (
            0.1 + 0.9 * (1 + math.cos(math.pi * epoch / args.ssl_epochs)) / 2)
        optimizer.param_groups[0]["lr"] = lr
        totals = {name: 0.0 for name in ("loss", "positive_score", "negative_score",
                                         "token_variance", "mean_pair_cosine")}
        count = updates = clipped = 0
        norm_total = 0.0
        for signal, _, _ in data:
            signal = signal.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, details = model(signal, sampler_generator)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite sampled CPC loss")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise RuntimeError("Nonfinite sampled CPC gradients")
            optimizer.step()
            clipped += int(float(norm) > 1.0)
            norm_total += float(norm)
            updates += 1
            count += len(signal)
            for name, value in {"loss": float(loss.detach()), **details}.items():
                totals[name] += value * len(signal)
        record = {"epoch": epoch + 1, "lr": lr,
                  "elapsed_seconds": previous_elapsed + time.monotonic() - started,
                  "optimizer_updates": updates, "record_exposures": count,
                  "mean_gradient_norm_before_clip": norm_total / updates,
                  "gradient_clip_fraction": clipped / updates,
                  **{name: value / count for name, value in totals.items()}}
        history.append(record)
        save_epoch(directory, fp, epoch + 1, model, optimizer,
                   loader_generator, sampler_generator, history)
        print(json.dumps({"stage": "pretrain", "variant": variant, **record}), flush=True)
    base.atomic_torch(complete, {"fingerprint": fp,
                "encoder": base.cpu_state(model.encoder), "variant": variant,
                "epochs": args.ssl_epochs, "seed": base.SSL_SEED,
                "sampler_seed": SAMPLER_SEED, "training_records": len(pool.train_rows),
                "parameters": sum(p.numel() for p in model.parameters())})
    return complete


def profile(args, pool, mean, std):
    for variant in (VARIANTS if args.variant == "all" else (args.variant,)):
        base.seed_all(base.SSL_SEED)
        model = SampledCPCPretrainer(variant).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        loader_generator = torch.Generator().manual_seed(base.SSL_SEED)
        sampler_generator = torch.Generator().manual_seed(SAMPLER_SEED)
        data = base.loader(pool, pool.train_rows, mean, std, args.ssl_batch_size,
                           True, loader_generator, args.device)
        started = None
        measured_records = 0
        for update, (signal, _, _) in enumerate(data, 1):
            if update == 6:
                if args.device == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                started = time.monotonic()
            signal = signal.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, details = model(signal, sampler_generator)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite profile loss")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise RuntimeError("Nonfinite profile gradients")
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


def report(args):
    comparisons = {}
    cache_metadata = json.loads((args.cache_dir / "complete.json").read_text())
    train_count = cache_metadata.get("split_counts", {}).get("train")
    cohort = (f"{train_count:,} combined MIMIC and PTB-XL training ECGs" if train_count is not None
              else "MIMIC plus PTB-XL training ECGs")
    lines = [f"# Sampled CPC objectives on {cohort}", "",
             "Both arms use the same compact causal CNN/GRU, temporal predictions, and sixteen sampled same-half negatives per query. Sampling permits replacement; positions within three tokens of the positive are excluded.", "",
             f"Both arms use {args.ssl_epochs} fixed SSL epochs. InfoNCE uses positive-versus-sixteen softmax classification. SGNS uses softplus(-positive) plus the sum of sixteen negative softplus terms. Their raw losses have different scales and class-prior optima; compare downstream metrics.", ""]
    lines += ["This is a one-seed exploratory comparison on the PTB-XL test set already used by earlier project experiments; the test set is not a fresh external confirmation cohort.", ""]
    for budget in ("1", "0.1"):
        paths = {variant: args.output_dir / f"{variant}_fraction{budget}_seed42" for variant in VARIANTS}
        if not all((path / "completion.json").exists() for path in paths.values()):
            continue
        lines += [f"## {budget} label fraction", "",
                  "| Arm | AUROC | AP | Sensitivity | Specificity |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for variant, path in paths.items():
            result = json.loads((path / "metrics.json").read_text())["test"]
            lines.append(f"| {variant} | {result['auroc']:.3f} | {result['average_precision']:.3f} | {result['sensitivity']:.3f} | {result['specificity']:.3f} |")
        comparison = base.paired_comparison(paths["sgns"], paths["sampled_info"], args.bootstrap)
        comparisons[f"fraction{budget}_sgns_minus_sampled_info"] = comparison
        lines += ["", "SGNS minus sampled InfoNCE (paired test-patient bootstrap):", ""]
        for name, result in comparison.items():
            lines.append(f"- {name}: {result['difference']:+.3f} "
                         f"(95% CI {result['ci95'][0]:+.3f} to {result['ci95'][1]:+.3f})")
        lines.append("")
    base.atomic_json(args.output_dir / "paired_comparisons.json", comparisons)
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("profile", "pretrain", "train", "all"), default="all")
    parser.add_argument("--variant", choices=(*VARIANTS, "all"), default="all")
    parser.add_argument("--labels", choices=("0.1", "1", "all"), default="all")
    parser.add_argument("--cache-dir", type=Path, default=base.DEFAULT_CACHE)
    parser.add_argument("--manifest-dir", type=Path, default=base.DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ssl-batch-size", type=int, default=128)
    parser.add_argument("--ssl-epochs", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--profile-updates", type=int, default=20)
    args = parser.parse_args()
    if min(args.threads, args.batch_size, args.ssl_batch_size, args.ssl_epochs,
           args.epochs, args.patience, args.bootstrap, args.profile_updates) < 1:
        parser.error("Thread, batch, epoch, patience, bootstrap, and profile counts must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    torch.set_num_threads(args.threads)
    if args.stage != "profile":
        args.output_dir.mkdir(parents=True, exist_ok=True)
    with base.GPU_LOCK.open("a+") as lock:
        if args.device == "cuda":
            print(f"Waiting for GPU lock {base.GPU_LOCK}", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
        pool = base.Pool(args.cache_dir)
        source_hashes = source_hashes_with_new_code(pool, args.manifest_dir)
        if args.stage == "profile":
            with tempfile.TemporaryDirectory(prefix="cpc_word2vec_profile_") as temporary:
                mean, std = pool.normalization(temporary, source_hashes)
                profile(args, pool, mean, std)
            return
        mean, std = pool.normalization(args.output_dir, source_hashes)
        if args.stage in ("pretrain", "all"):
            for variant in (VARIANTS if args.variant == "all" else (args.variant,)):
                pretrain(args, pool, mean, std, source_hashes, variant)
        if args.stage in ("train", "all"):
            for budget in (("1", "0.1") if args.labels == "all" else (args.labels,)):
                for variant in (VARIANTS if args.variant == "all" else (args.variant,)):
                    base.fine_tune(args, pool, mean, std, source_hashes, variant, budget)
            report(args)


if __name__ == "__main__":
    main()
