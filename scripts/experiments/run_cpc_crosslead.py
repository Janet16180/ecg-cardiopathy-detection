#!/usr/bin/env python3
"""Experiment010: matched full, within-lead, and cross-lead CPC continuation."""

import argparse
import copy
import fcntl
import json
import math
import time
from pathlib import Path

import torch
from torch import nn

from ecg_experiment.cpc_crosslead import AUX_WEIGHT, GROUP_A, GROUP_B, VARIANTS, CrossLeadPretrainer
from scripts.experiments import run_cpc_experiment as base
from scripts.experiments.run_cpc_tokenization import bootstrap


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs/experiment010_cpc_crosslead"
DEFAULT_BOOTSTRAP = ROOT / "outputs/experiment004_cpc_40k/cpc_ssl"
PROFILE_WARMUP = 2


def source_hashes(args, pool):
    hashes = base.make_source_hashes(pool, args.manifest_dir)
    paths = [ROOT / name for name in (
        "ecg_experiment/cpc_crosslead.py", "scripts/experiments/run_cpc_crosslead.py",
        # bootstrap() is imported, so its implementation is part of this run.
        "scripts/experiments/run_cpc_tokenization.py")]
    paths += [args.bootstrap_dir / name for name in
              ("encoder.pt", "epoch_state.pt", "config.json")]
    paths.append(args.bootstrap_dir.parent / "normalization.json")
    hashes.update({str(path.resolve()): base.digest_file(path) for path in paths})
    return hashes


def fixed_normalization(pool, args, hashes, bootstrap_config):
    """Read the exact 004 normalization; Pool.normalization validates its source."""
    path = args.bootstrap_dir.parent / "normalization.json"
    if not path.exists():
        raise FileNotFoundError(f"Experiment004 normalization is missing: {path}")
    mean, std = pool.normalization(path.parent, hashes)
    original = bootstrap_config["inputs"]["normalization"]
    if mean.tolist() != original["mean"] or std.tolist() != original["std"]:
        # 004 fingerprints float32 arrays; do not silently use a changed file.
        raise ValueError("Experiment004 normalization differs from bootstrap config")
    return mean, std


def ssl_settings(args, variant):
    return {"stage": "pretrain", "variant": variant, "seed": base.SSL_SEED,
            "epochs": args.ssl_epochs, "batch_size": args.ssl_batch_size,
            "bootstrap": "Experiment004 CPC completed epoch20 encoder and all three heads",
            "optimizer": "AdamW", "lr": 1e-3, "weight_decay": 0.01,
            "warmup_epochs": 2, "cosine_floor": 0.1, "horizons": [4, 8, 12],
            "temperature": 0.1, "negative_exclusion_tokens": 3,
            "views": {"full": list(range(12)), "A": list(GROUP_A), "B": list(GROUP_B)},
            "pass_order": ["full", "A", "B"],
            "auxiliary_weight": 0.0 if variant == "native" else AUX_WEIGHT,
            "auxiliary_target": "opposite lead group" if variant == "crosslead" else "same lead group",
            "normalization": "exact Experiment004 train-only normalization; zero absent leads after normalization"}


def ssl_fingerprint(args, pool, mean, std, hashes, variant):
    return base.fingerprint(pool, hashes, mean, std, ssl_settings(args, variant))


def profile_path(args, variant):
    return args.output_dir / f"profile_{variant}.json"


def require_profile(args, variant, fingerprint):
    path = profile_path(args, variant)
    if not path.exists():
        raise RuntimeError(f"CUDA SSL requires a matching completed profile: {path}")
    record = json.loads(path.read_text())
    if (record.get("fingerprint") != fingerprint or record.get("device") != "cuda"
            or record.get("measured_updates", 0) < 5 or not record.get("resume_roundtrip")):
        raise ValueError(f"CUDA profile gate does not match inputs or is incomplete: {path}")


def new_model(variant, encoder, heads, device):
    model = CrossLeadPretrainer(variant)
    model.load_bootstrap(encoder, heads)
    return model.to(device)


def checked_step(model, optimizer, signal):
    optimizer.zero_grad(set_to_none=True)
    loss, details = model(signal)
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite cross-lead CPC loss")
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(grad_norm):
        raise RuntimeError("Nonfinite cross-lead CPC gradient norm")
    optimizer.step()
    if not torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all():
        raise RuntimeError("Nonfinite cross-lead CPC parameters")
    return float(loss.detach()), details, float(grad_norm)


def profile_roundtrip(model, optimizer, signal, variant, generator):
    """Verify a saved model/optimizer/RNG state reproduces the next dropout update."""
    snapshot = {"model": base.cpu_state(model), "optimizer": copy.deepcopy(optimizer.state_dict()),
                "rng": base.rng_state(generator)}
    duplicate = CrossLeadPretrainer(variant).to(signal.device)
    duplicate.load_state_dict(snapshot["model"])
    copy_optimizer = torch.optim.AdamW(duplicate.parameters(), lr=1e-3, weight_decay=0.01)
    copy_optimizer.load_state_dict(snapshot["optimizer"])
    base.restore_rng(snapshot["rng"], generator)
    checked_step(model, optimizer, signal)
    expected = base.cpu_state(model)
    base.restore_rng(snapshot["rng"], generator)
    checked_step(duplicate, copy_optimizer, signal)
    for key, tensor in expected.items():
        torch.testing.assert_close(tensor, duplicate.state_dict()[key].cpu(), atol=0, rtol=0)
    # Restore so the profile itself has no hidden difference after the check.
    base.restore_rng(snapshot["rng"], generator)
    return True


def profile(args, pool, mean, std, hashes, encoder, heads, variants):
    for variant in variants:
        base.seed_all(base.SSL_SEED)
        model = new_model(variant, encoder, heads, args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        generator = torch.Generator().manual_seed(base.SSL_SEED)
        data = base.loader(pool, pool.train_rows, mean, std, args.ssl_batch_size,
                           True, generator, args.device)
        model.train()
        started = None
        measured_records = 0
        last_signal = None
        for update, (signal, _, _) in enumerate(data, 1):
            if update == PROFILE_WARMUP + 1:
                if args.device == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                started = time.monotonic()
            signal = signal.to(args.device, non_blocking=True)
            loss, details, grad_norm = checked_step(model, optimizer, signal)
            last_signal = signal
            if update > PROFILE_WARMUP:
                measured_records += len(signal)
            if update >= PROFILE_WARMUP + args.profile_updates:
                break
        if started is None or update < PROFILE_WARMUP + args.profile_updates:
            raise RuntimeError("Pool is too small for the requested profile")
        if args.device == "cuda":
            torch.cuda.synchronize()
        seconds = time.monotonic() - started
        roundtrip = profile_roundtrip(model, optimizer, last_signal, variant, generator)
        fp, inputs = ssl_fingerprint(args, pool, mean, std, hashes, variant)
        record = {"stage": "profile", "variant": variant, "fingerprint": fp,
                  "inputs": inputs, "device": args.device,
                  "model_parameters": sum(p.numel() for p in model.parameters()),
                  "encoder_parameters": sum(p.numel() for p in model.encoder.parameters()),
                  "warmup_updates": PROFILE_WARMUP, "measured_updates": args.profile_updates,
                  "measured_records": measured_records, "measured_seconds": seconds,
                  "seconds_per_update": seconds / args.profile_updates,
                  "estimated_ssl_epoch_seconds": len(pool.train_rows) * seconds / measured_records,
                  "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9 if args.device == "cuda" else None,
                  "resume_roundtrip": roundtrip, "loss": loss, "gradient_norm": grad_norm, **details}
        base.atomic_json(profile_path(args, variant), record)
        print(json.dumps({k: v for k, v in record.items() if k != "inputs"}), flush=True)


def resume_or_new(directory, fingerprint, model, optimizer, generator):
    path = directory / "epoch_state.pt"
    if not path.exists():
        if (directory / "history.json").exists():
            raise ValueError(f"History exists without resumable state: {directory}")
        return 0, []
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["fingerprint"] != fingerprint:
        raise ValueError(f"Resume fingerprint mismatch: {directory}")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    base.restore_rng(state["rng"], generator)
    return state["epoch"], state["history"]


def save_epoch(directory, fingerprint, epoch, model, optimizer, generator, history):
    base.atomic_torch(directory / "epoch_state.pt", {
        "fingerprint": fingerprint, "epoch": epoch, "model": base.cpu_state(model),
        "optimizer": optimizer.state_dict(), "rng": base.rng_state(generator),
        "history": history})
    base.atomic_json(directory / "history.json", history)


def pretrain(args, pool, mean, std, hashes, encoder, heads, variant):
    base.seed_all(base.SSL_SEED)
    directory = args.output_dir / f"{variant}_ssl"
    directory.mkdir(parents=True, exist_ok=True)
    fp, inputs = ssl_fingerprint(args, pool, mean, std, hashes, variant)
    if args.device == "cuda":
        require_profile(args, variant, fp)
    config_path = directory / "config.json"
    if config_path.exists() and json.loads(config_path.read_text())["fingerprint"] != fp:
        raise ValueError(f"Existing SSL config fingerprint mismatch: {directory}")
    base.atomic_json(config_path, {"fingerprint": fp, "inputs": inputs})
    complete = directory / "encoder.pt"
    if complete.exists():
        saved = torch.load(complete, map_location="cpu", weights_only=True)
        if saved["fingerprint"] != fp or saved["variant"] != variant or saved["epochs"] != args.ssl_epochs:
            raise ValueError(f"Completed SSL checkpoint mismatch: {complete}")
        return complete
    model = new_model(variant, encoder, heads, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    generator = torch.Generator().manual_seed(base.SSL_SEED)
    data = base.loader(pool, pool.train_rows, mean, std, args.ssl_batch_size,
                       True, generator, args.device)
    start_epoch, history = resume_or_new(directory, fp, model, optimizer, generator)
    started = time.monotonic()
    previous_elapsed = history[-1]["elapsed_seconds"] if history else 0.0
    for epoch in range(start_epoch, args.ssl_epochs):
        model.train()
        lr = 1e-3 * min(1.0, (epoch + 1) / 2) * (
            0.1 + 0.9 * (1 + math.cos(math.pi * epoch / args.ssl_epochs)) / 2)
        optimizer.param_groups[0]["lr"] = lr
        totals = {name: 0.0 for name in (
            "loss", "full_cpc", "aux_a", "aux_b", "token_variance",
            "adjacent_token_cosine", "view_a_token_variance", "view_b_token_variance")}
        count = updates = clipped = 0
        norm_total = 0.0
        for signal, _, _ in data:
            signal = signal.to(args.device, non_blocking=True)
            loss, details, norm = checked_step(model, optimizer, signal)
            clipped += int(norm > 1.0)
            norm_total += norm
            updates += 1
            count += len(signal)
            for name, value in {"loss": loss, **details}.items():
                totals[name] += value * len(signal)
        record = {"epoch": epoch + 1, "lr": lr,
                  "elapsed_seconds": previous_elapsed + time.monotonic() - started,
                  "optimizer_updates": updates, "record_exposures": count,
                  "mean_gradient_norm_before_clip": norm_total / updates,
                  "gradient_clip_fraction": clipped / updates,
                  **{name: value / count for name, value in totals.items()}}
        history.append(record)
        save_epoch(directory, fp, epoch + 1, model, optimizer, generator, history)
        print(json.dumps({"stage": "pretrain", "variant": variant, **record}), flush=True)
    base.atomic_torch(complete, {"fingerprint": fp, "encoder": base.cpu_state(model.encoder),
                      "variant": variant, "epochs": args.ssl_epochs, "seed": base.SSL_SEED,
                      "training_records": len(pool.train_rows),
                      "parameters": sum(p.numel() for p in model.parameters())})
    return complete


def require_all_ssl(args, pool, mean, std, hashes):
    for variant in VARIANTS:
        fp, _ = ssl_fingerprint(args, pool, mean, std, hashes, variant)
        directory = args.output_dir / f"{variant}_ssl"
        config = json.loads((directory / "config.json").read_text())
        saved = torch.load(directory / "encoder.pt", map_location="cpu", weights_only=True)
        state = torch.load(directory / "epoch_state.pt", map_location="cpu", weights_only=False)
        if (config["fingerprint"] != fp or saved["fingerprint"] != fp
                or state["fingerprint"] != fp or state["epoch"] != args.ssl_epochs
                or saved["epochs"] != args.ssl_epochs or saved["variant"] != variant):
            raise ValueError(f"Incomplete or changed SSL arm: {variant}")
        encoder = {key.removeprefix("encoder."): value for key, value in state["model"].items()
                   if key.startswith("encoder.")}
        if encoder.keys() != saved["encoder"].keys() or any(
                not torch.equal(value, saved["encoder"][key]) for key, value in encoder.items()):
            raise ValueError(f"SSL final encoder and epoch state disagree: {variant}")


def report(args):
    comparisons = {}
    lines = ["# Experiment010: cross-lead CPC", "",
             "Three arms continue the same completed Experiment004 epoch20 CPC encoder and three prediction heads for ten SSL epochs. All use full twelve-lead CPC. Within-lead and cross-lead add 0.1-weighted CPC on masked A and B views; cross-lead uses the opposite view's future targets.", "",
             "This is a one-seed exploratory comparison on the previously used PTB-XL test set, not a fresh external confirmation cohort. Intervals resample test patients in matched pairs.", ""]
    for budget in ("1", "0.1"):
        paths = {variant: args.output_dir / f"{variant}_fraction{budget}_seed42" for variant in VARIANTS}
        if not all((path / "completion.json").exists() for path in paths.values()):
            continue
        lines += [f"## {budget} label fraction", "", "| Arm | AUROC | AP | Sensitivity | Specificity |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for variant, path in paths.items():
            result = json.loads((path / "metrics.json").read_text())["test"]
            lines.append(f"| {variant} | {result['auroc']:.3f} | {result['average_precision']:.3f} | {result['sensitivity']:.3f} | {result['specificity']:.3f} |")
        lines.append("")
        for left, right in (("crosslead", "withinlead"), ("withinlead", "native")):
            comparison = base.paired_comparison(paths[left], paths[right], args.bootstrap)
            comparisons[f"fraction{budget}_{left}_minus_{right}"] = comparison
            lines += [f"{left} minus {right} (paired patient bootstrap):", ""]
            for name, result in comparison.items():
                lines.append(f"- {name}: {result['difference']:+.3f} (95% CI {result['ci95'][0]:+.3f} to {result['ci95'][1]:+.3f})")
            lines.append("")
    base.atomic_json(args.output_dir / "paired_comparisons.json", comparisons)
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")


def run(args):
    if args.ssl_epochs != 10:
        raise ValueError("Experiment010 protocol requires exactly ten continuation epochs")
    if args.stage == "all" and args.variant != "all":
        raise ValueError("The all stage requires all three SSL arms before transfer")
    if args.device == "cuda" and args.stage != "check" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with base.GPU_LOCK.open("a+") as lock:
        if args.device == "cuda" and args.stage != "check":
            print(f"Waiting for GPU lock {base.GPU_LOCK}", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
        pool = base.Pool(args.cache_dir)
        hashes = source_hashes(args, pool)
        encoder, heads, bootstrap_config = bootstrap(args, hashes)
        mean, std = fixed_normalization(pool, args, hashes, bootstrap_config)
        variants = VARIANTS if args.variant == "all" else (args.variant,)
        if args.stage == "check":
            for variant in variants:
                fp, _ = ssl_fingerprint(args, pool, mean, std, hashes, variant)
                print(json.dumps({"stage": "check", "variant": variant, "fingerprint": fp,
                                  "profile_ready": profile_path(args, variant).exists(),
                                  "ssl_complete": (args.output_dir / f"{variant}_ssl/encoder.pt").exists()}), flush=True)
            return
        if args.stage == "profile":
            profile(args, pool, mean, std, hashes, encoder, heads, variants)
            return
        if args.stage in ("pretrain", "all"):
            for variant in variants:
                pretrain(args, pool, mean, std, hashes, encoder, heads, variant)
        if args.stage in ("train", "all"):
            require_all_ssl(args, pool, mean, std, hashes)
            for budget in (("1", "0.1") if args.labels == "all" else (args.labels,)):
                for variant in variants:
                    base.fine_tune(args, pool, mean, std, hashes, variant, budget)
            report(args)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("check", "profile", "pretrain", "train", "all"), default="all")
    parser.add_argument("--variant", choices=(*VARIANTS, "all"), default="all")
    parser.add_argument("--labels", choices=("0.1", "1", "all"), default="all")
    parser.add_argument("--cache-dir", type=Path, default=base.DEFAULT_CACHE)
    parser.add_argument("--manifest-dir", type=Path, default=base.DEFAULT_MANIFEST)
    parser.add_argument("--bootstrap-dir", type=Path, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ssl-batch-size", type=int, default=128)
    parser.add_argument("--ssl-epochs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--profile-updates", type=int, default=5)
    args = parser.parse_args(argv)
    if min(args.threads, args.batch_size, args.ssl_batch_size, args.ssl_epochs,
           args.epochs, args.patience, args.bootstrap) < 1 or args.profile_updates < 5:
        parser.error("Counts must be positive and profile-updates must be at least five")
    run(args)


if __name__ == "__main__":
    main()
