#!/usr/bin/env python3
"""Continue the same CPC with cluster targets or causal chunk tokenization."""

import argparse
import fcntl
import json
import math
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from ecg_experiment.cpc_tokenization import (
    CLUSTERS, CLUSTER_WEIGHT, FIT_RECORDS, TOKENS_PER_RECORD,
    TokenizationClassifier, TokenizationPretrainer, cnn_tokens, fit_kmeans,
    snapshot_teacher_convs,
)
from scripts import run_cpc_experiment as base


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/experiment006_cpc_tokenization"
DEFAULT_BEATS = ROOT / "data/processed/cpc_beats_40k"
DEFAULT_BOOTSTRAP = ROOT / "outputs/experiment004_cpc_40k/cpc_ssl"
VARIANTS = ("continuation", "clusteraux", "fixedchunk", "beatchunk", "learnedchunk")
CHUNK_VARIANTS = ("fixedchunk", "beatchunk", "learnedchunk")


class BeatPoolDataset(base.PoolDataset):
    def __init__(self, pool, rows, mean, std, boundaries):
        super().__init__(pool, rows, mean, std)
        self.boundaries = boundaries

    def __getitem__(self, index):
        signal, target, patient = super().__getitem__(index)
        return signal, target, patient, torch.from_numpy(
            np.asarray(self.boundaries[self.indices[index]], dtype=np.bool_).copy())


def loader(pool, rows, mean, std, boundaries, batch_size, shuffle, generator, device):
    return DataLoader(BeatPoolDataset(pool, rows, mean, std, boundaries),
                      batch_size=batch_size, shuffle=shuffle, generator=generator,
                      num_workers=0, pin_memory=(device == "cuda"), drop_last=False)


def load_beats(path, pool):
    from ecg_experiment.ecg_tokenizers import load_beat_metadata
    info = json.loads((path / "complete.json").read_text())
    identity = info["identity"]
    if (identity["cache_signals_sha256"] != pool.metadata["signals_sha256"]
            or identity["cache_complete_sha256"] != base.digest_file(pool.directory / "complete.json")
            or identity["detector_code_sha256"] != base.digest_file(ROOT / "ecg_experiment/ecg_tokenizers.py")):
        raise ValueError("Beat metadata uses different waveforms or detector code")
    return load_beat_metadata(path, pool.ids)


def source_hashes(args, pool):
    hashes = base.make_source_hashes(pool, args.manifest_dir)
    paths = [ROOT / name for name in (
        "ecg_experiment/cpc_tokenization.py", "scripts/run_cpc_tokenization.py",
        "ecg_experiment/ecg_tokenizers.py", "scripts/prepare_beat_tokens.py")]
    paths += [args.beat_dir / name for name in (
        "complete.json", "ecg_ids.npy", "boundaries.npy", "peaks.npy",
        "confirmations.npy", "counts.npy", "forced.npy")]
    paths += [args.bootstrap_dir / name for name in ("encoder.pt", "epoch_state.pt", "config.json")]
    hashes.update({str(path.resolve()): base.digest_file(path) for path in paths})
    return hashes


def bootstrap(args, source_hashes):
    encoder_path = args.bootstrap_dir / "encoder.pt"
    state_path = args.bootstrap_dir / "epoch_state.pt"
    final = torch.load(encoder_path, map_location="cpu", weights_only=True)
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if final["variant"] != "cpc" or final["epochs"] != 20 or state["epoch"] != 20:
        raise ValueError("Experiment 004 CPC bootstrap is not its completed 20-epoch state")
    if final["fingerprint"] != state["fingerprint"]:
        raise ValueError("Experiment 004 final encoder and epoch state disagree")
    encoder = {name.removeprefix("encoder."): value for name, value in state["model"].items()
               if name.startswith("encoder.")}
    if encoder.keys() != final["encoder"].keys() or any(
            not torch.equal(encoder[name], final["encoder"][name]) for name in encoder):
        raise ValueError("Experiment 004 final encoder differs from full epoch state")
    heads = {name.removeprefix("heads."): value for name, value in state["model"].items()
             if name.startswith("heads.")}
    if len(heads) != 3:
        raise ValueError("Expected all three pretrained CPC prediction heads")
    config = json.loads((args.bootstrap_dir / "config.json").read_text())
    if config["fingerprint"] != final["fingerprint"]:
        raise ValueError("Experiment 004 bootstrap config fingerprint mismatch")
    for path, declared in config["inputs"]["cache"].items():
        if path in source_hashes and source_hashes[path] != declared:
            raise ValueError(f"Experiment 004 bootstrap input changed: {path}")
    return encoder, heads, config


def load_bootstrap_weights(model, encoder_state, head_state):
    if model.variant in CHUNK_VARIANTS:
        model.encoder.load_from_cpc_state_dict(encoder_state)
    else:
        model.encoder.load_state_dict(encoder_state)
    model.heads.load_state_dict(head_state)


def classifier_with_matched_head(variant, device):
    """Use identical binary-head weights despite variant constructor draw counts."""
    cpu_rng = torch.random.get_rng_state()
    try:
        torch.random.manual_seed(42)
        shared_head = base.cpu_state(nn.Linear(512, 1))
    finally:
        torch.random.set_rng_state(cpu_rng)
    model = TokenizationClassifier(variant)
    model.head.load_state_dict(shared_head)
    return model.to(device)


def reset_cluster_heads(model, optimizer, generator, seed=43):
    """New K-means IDs get new class heads; leave CPC weights and RNG untouched."""
    snapshot = base.rng_state(generator)
    try:
        torch.manual_seed(seed)
        for head in model.cluster_heads:
            head.reset_parameters()
            for parameter in head.parameters():
                optimizer.state.pop(parameter, None)
    finally:
        base.restore_rng(snapshot, generator)


def boundary_disagreement(encoder):
    gates = encoder.last_gates.detach() >= 0.5
    fixed = torch.zeros(79, dtype=torch.bool, device=gates.device)
    fixed[15::16] = True
    fixed[-1] = True
    return float((gates != fixed.reshape(1, 1, 79)).float().mean())


def codebook_selection(pool):
    rng = np.random.default_rng(42)
    count = min(FIT_RECORDS, len(pool.train_rows))
    rows = rng.choice(len(pool.train_rows), size=count, replace=False)
    positions = np.stack([rng.choice(158, size=min(TOKENS_PER_RECORD, 158),
                                     replace=False) for _ in range(count)])
    return rows.astype(np.int64), positions.astype(np.int64)


@torch.inference_mode()
def selected_features(args, pool, mean, std, teacher_convs, row_indices, positions):
    """Extract the same fixed train-record/token sample for each codebook round."""
    teacher_convs.eval().to(args.device)
    selected_rows = [pool.train_rows[int(i)] for i in row_indices]
    data = base.loader(pool, selected_rows, mean, std, 64, False,
                       torch.Generator().manual_seed(4242), args.device)
    features = np.empty((len(row_indices) * positions.shape[1], 256), dtype=np.float32)
    offset = 0
    for signal, _, _ in data:
        batch = len(signal)
        tokens = cnn_tokens(teacher_convs, signal.to(args.device, non_blocking=True))
        flat = tokens.reshape(batch, 158, 256)
        indices = torch.from_numpy(positions[offset:offset + batch]).to(args.device)
        chosen = flat.gather(1, indices.unsqueeze(-1).expand(-1, -1, 256))
        start = offset * positions.shape[1]
        features[start:start + batch * positions.shape[1]] = chosen.cpu().reshape(-1, 256).numpy()
        offset += batch
    if offset != len(row_indices):
        raise RuntimeError("Incomplete codebook feature extraction")
    return features


def fit_round(args, pool, mean, std, teacher, row_indices, positions, seed):
    """No fitting RNG leaks into training dropout or shuffled data order."""
    dummy_loader_generator = torch.Generator().manual_seed(0)
    snapshot = base.rng_state(dummy_loader_generator)
    try:
        features = selected_features(args, pool, mean, std, teacher, row_indices, positions)
        centers, fit = fit_kmeans(features, seed=seed)
    finally:
        base.restore_rng(snapshot, dummy_loader_generator)
    return torch.from_numpy(centers).to(args.device), fit


def initial_codebook(args, pool, mean, std, source_hashes, encoder_state):
    path = args.output_dir / "cluster_codebook_round0.pt"
    input_fp = base.digest_json({"source_hashes": source_hashes,
                                 "normalization": [mean.tolist(), std.tolist()],
                                 "fit_records": FIT_RECORDS,
                                 "tokens_per_record": TOKENS_PER_RECORD,
                                 "clusters": CLUSTERS, "seed": 42})
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["input_fingerprint"] != input_fp:
            raise ValueError("Existing initial codebook has different inputs")
        return saved
    teacher = TokenizationPretrainer("continuation").encoder
    teacher.load_state_dict(encoder_state)
    teacher_convs = snapshot_teacher_convs(teacher)
    rows, positions = codebook_selection(pool)
    centers, fit = fit_round(args, pool, mean, std, teacher_convs, rows, positions, 42)
    saved = {"input_fingerprint": input_fp, "centers": centers.cpu(),
             "teacher_state": base.cpu_state(teacher_convs),
             "row_indices": torch.from_numpy(rows),
             "positions": torch.from_numpy(positions), "fit": fit,
             "training_ecg_ids": [pool.train_rows[int(i)]["ecg_id"] for i in rows]}
    base.atomic_torch(path, saved)
    return saved


def ssl_settings(args, variant, codebook_hash=None):
    return {"stage": "continued_pretrain", "variant": variant,
            "seed": 42, "epochs": args.ssl_epochs, "batch_size": args.ssl_batch_size,
            "bootstrap": "Experiment 004 CPC completed encoder plus all three prediction heads",
            "learning_rate": 1e-3, "weight_decay": 0.01, "warmup_epochs": 2,
            "cosine_floor": 0.1, "horizons": [4, 8, 12],
            "first_query": 24, "negative_exclusion_tokens": 3,
            "cluster_codebook_sha256": codebook_hash,
            "cluster_k": CLUSTERS if variant == "clusteraux" else None,
            "cluster_ce_weight": CLUSTER_WEIGHT if variant == "clusteraux" else None,
            "cluster_refresh_after_epoch": 5 if variant == "clusteraux" else None,
            "augmentation": "none"}


def save_ssl_epoch(directory, fingerprint, epoch, model, optimizer, generator,
                   history, centers=None, teacher=None, codebook_round=0,
                   round_fits=None):
    base.atomic_torch(directory / "epoch_state.pt", {
        "fingerprint": fingerprint, "epoch": epoch,
        "model": base.cpu_state(model), "optimizer": optimizer.state_dict(),
        "rng": base.rng_state(generator), "history": history,
        "centers": centers.detach().cpu() if centers is not None else None,
        "teacher_state": base.cpu_state(teacher) if teacher is not None else None,
        "codebook_round": codebook_round, "round_fits": round_fits or []})
    base.atomic_json(directory / "history.json", history)


def pretrain(args, pool, mean, std, boundaries, source_hashes, bootstrap_weights,
             variant, codebook=None):
    encoder_state, head_state, _ = bootstrap_weights
    base.seed_all(42)
    directory = args.output_dir / f"{variant}_ssl"
    directory.mkdir(parents=True, exist_ok=True)
    codebook_hash = base.digest_file(args.output_dir / "cluster_codebook_round0.pt") if codebook else None
    fp, inputs = base.fingerprint(pool, source_hashes, mean, std,
                                  ssl_settings(args, variant, codebook_hash))
    config_path = directory / "config.json"
    if config_path.exists() and json.loads(config_path.read_text())["fingerprint"] != fp:
        raise ValueError(f"Existing SSL config differs: {directory}")
    base.atomic_json(config_path, {"fingerprint": fp, "inputs": inputs,
        "description": "All arms continue completed Experiment 004 CPC encoder and future heads",
        "cluster_teacher": "Frozen CNN snapshot per five-epoch round; refit codebook after epoch five and reset auxiliary class heads" if codebook else None})
    complete = directory / "encoder.pt"
    if complete.exists():
        saved = torch.load(complete, map_location="cpu", weights_only=True)
        if saved["fingerprint"] != fp:
            raise ValueError(f"Completed SSL fingerprint mismatch: {complete}")
        return complete
    model = TokenizationPretrainer(variant).to(args.device)
    load_bootstrap_weights(model, encoder_state, head_state)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=1e-3, weight_decay=0.01)
    generator = torch.Generator().manual_seed(42)
    data = loader(pool, pool.train_rows, mean, std, boundaries, args.ssl_batch_size,
                  True, generator, args.device)
    centers = codebook["centers"].to(args.device) if codebook else None
    teacher = snapshot_teacher_convs(model.encoder).to(args.device) if codebook else None
    if codebook:
        teacher.load_state_dict(codebook["teacher_state"])
    codebook_round = 0
    round_fits = [codebook["fit"]] if codebook else []
    state_path = directory / "epoch_state.pt"
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["fingerprint"] != fp:
            raise ValueError(f"SSL resume fingerprint mismatch: {directory}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        base.restore_rng(state["rng"], generator)
        start_epoch, history = state["epoch"], state["history"]
        centers = state["centers"].to(args.device) if codebook else None
        if codebook:
            teacher.load_state_dict(state["teacher_state"])
        codebook_round, round_fits = state["codebook_round"], state["round_fits"]
    else:
        if (directory / "history.json").exists():
            raise ValueError("History exists without resumable SSL state")
        start_epoch, history = 0, []
        # This reset makes dropout sequences identical despite variant-specific construction.
        base.seed_all(42)
    started = time.monotonic()
    previous_elapsed = history[-1]["elapsed_seconds"] if history else 0.0
    for epoch in range(start_epoch, args.ssl_epochs):
        if codebook and epoch == 5 and codebook_round == 0:
            teacher = snapshot_teacher_convs(model.encoder).to(args.device)
            centers, fit = fit_round(args, pool, mean, std, teacher,
                                      codebook["row_indices"].numpy(),
                                      codebook["positions"].numpy(), 43)
            reset_cluster_heads(model, optimizer, generator, seed=43)
            round_fits.append(fit)
            codebook_round = 1
            save_ssl_epoch(directory, fp, epoch, model, optimizer, generator, history,
                           centers, teacher, codebook_round, round_fits)
        model.train()
        lr = 1e-3 * min(1.0, (epoch + 1) / 2) * (
            0.1 + 0.9 * (1 + math.cos(math.pi * epoch / args.ssl_epochs)) / 2)
        optimizer.param_groups[0]["lr"] = lr
        totals = {}
        count = updates = clipped = 0
        norm_total = 0.0
        for signal, _, _, beat_boundaries in data:
            signal = signal.to(args.device, non_blocking=True)
            beat_boundaries = beat_boundaries.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, details = model(signal, beat_boundaries, centers, teacher)
            if variant in CHUNK_VARIANTS:
                details["boundary_disagreement_with_fixed"] = boundary_disagreement(model.encoder)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite continued CPC loss in {variant}")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise RuntimeError(f"Nonfinite continued CPC gradients in {variant}")
            optimizer.step()
            clipped += int(float(norm) > 1.0)
            norm_total += float(norm)
            updates += 1
            count += len(signal)
            for name, value in {"loss": float(loss.detach()), **details}.items():
                totals[name] = totals.get(name, 0.0) + float(value) * len(signal)
        record = {"epoch": epoch + 1, "lr": lr,
                  "elapsed_seconds": previous_elapsed + time.monotonic() - started,
                  "optimizer_updates": updates, "record_exposures": count,
                  "mean_gradient_norm_before_clip": norm_total / updates,
                  "gradient_clip_fraction": clipped / updates,
                  "codebook_round": codebook_round if codebook else None,
                  **{name: value / count for name, value in totals.items()}}
        history.append(record)
        save_ssl_epoch(directory, fp, epoch + 1, model, optimizer, generator,
                       history, centers, teacher, codebook_round, round_fits)
        print(json.dumps({"stage": "pretrain", "variant": variant, **record}), flush=True)
    base.atomic_torch(complete, {"fingerprint": fp, "encoder": base.cpu_state(model.encoder),
        "variant": variant, "epochs": args.ssl_epochs, "seed": 42,
        "bootstrap_epoch": 20, "training_records": len(pool.train_rows),
        "codebook_round": codebook_round if codebook else None,
        "round_fits": round_fits, "parameters": sum(p.numel() for p in model.parameters())})
    return complete


@torch.inference_mode()
def predict(model, data, device):
    model.eval()
    return np.concatenate([model(signal.to(device, non_blocking=True),
                        beat_boundaries.to(device, non_blocking=True)).cpu().numpy()
                        for signal, _, _, beat_boundaries in data])


def fine_tune(args, pool, mean, std, boundaries, source_hashes, variant, budget):
    rows, manifest_hashes = base.manifest_rows(pool, args.manifest_dir, budget)
    base.seed_all(42)
    directory = args.output_dir / f"{variant}_fraction{budget}_seed42"
    directory.mkdir(parents=True, exist_ok=True)
    ssl_path = args.output_dir / f"{variant}_ssl" / "encoder.pt"
    settings = {"stage": "train", "variant": variant, "budget": budget,
                "seed": 42, "epochs": args.epochs, "patience": args.patience,
                "batch_size": args.batch_size, "encoder_lr": 3e-4,
                "head_lr": 1e-3, "weight_decay": 0.01,
                "augmentation": "none", "manifest_sha256": manifest_hashes,
                "ssl_checkpoint_sha256": base.digest_file(ssl_path)}
    fp, inputs = base.fingerprint(pool, source_hashes, mean, std, settings)
    completion = directory / "completion.json"
    if completion.exists():
        saved = json.loads(completion.read_text())
        if saved["fingerprint"] != fp:
            raise ValueError(f"Completed fine-tune fingerprint mismatch: {directory}")
        for name, declared in saved["artifacts"].items():
            if base.digest_file(directory / name) != declared:
                raise ValueError(f"Completed fine-tune artifact changed: {directory / name}")
        return
    model = classifier_with_matched_head(variant, args.device)
    ssl = torch.load(ssl_path, map_location="cpu", weights_only=True)
    if ssl["variant"] != variant or ssl["epochs"] != args.ssl_epochs:
        raise ValueError("SSL encoder variant or duration mismatch")
    model.encoder.load_state_dict(ssl["encoder"])
    optimizer = torch.optim.AdamW([{"params": (p for p in model.encoder.parameters() if p.requires_grad),
                                   "lr": 3e-4}, {"params": model.head.parameters(), "lr": 1e-3}],
                                  weight_decay=0.01)
    generator = torch.Generator().manual_seed(42)
    development, calibration = base.partition_validation(rows["validation"])
    train_data = loader(pool, rows["labeled_train"], mean, std, boundaries,
                        args.batch_size, True, generator, args.device)
    dev_data = loader(pool, development, mean, std, boundaries,
                      args.batch_size, False, None, args.device)
    start_epoch, history, best_model, best_auc, best_epoch = base.resume_or_new(
        directory, fp, model, optimizer, generator)
    if start_epoch == 0:
        base.seed_all(42)
    config_path = directory / "config.json"
    if config_path.exists() and json.loads(config_path.read_text())["fingerprint"] != fp:
        raise ValueError(f"Existing fine-tune config differs: {directory}")
    base.atomic_json(config_path, {"fingerprint": fp, "inputs": inputs,
        "architecture": variant, "model_parameters": sum(p.numel() for p in model.parameters()),
        "encoder_parameters": sum(p.numel() for p in model.encoder.parameters()),
        "labeled_training_records": len(rows["labeled_train"]),
        "development_records": len(development), "calibration_records": len(calibration),
        "test_records": len(rows["test"])})
    dev_y = np.array([int(row["target"]) for row in development])
    started = time.monotonic()
    previous_elapsed = history[-1]["elapsed_seconds"] if history else 0.0
    for epoch in range(start_epoch, args.epochs):
        if epoch - best_epoch >= args.patience:
            break
        model.train()
        total = 0.0
        updates = exposures = 0
        disagreement_total = 0.0
        for signal, target, _, beat_boundaries in train_data:
            signal = signal.to(args.device, non_blocking=True)
            target = target.to(args.device, dtype=torch.float32, non_blocking=True)
            beat_boundaries = beat_boundaries.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(signal, beat_boundaries), target)
            if variant in CHUNK_VARIANTS:
                disagreement_total += boundary_disagreement(model.encoder) * len(signal)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite fine-tune loss in {variant}")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise RuntimeError(f"Nonfinite fine-tune gradients in {variant}")
            optimizer.step()
            total += float(loss.detach()) * len(signal)
            updates += 1
            exposures += len(signal)
        dev_auc = float(roc_auc_score(dev_y, predict(model, dev_data, args.device)))
        if dev_auc > best_auc:
            best_auc, best_epoch, best_model = dev_auc, epoch + 1, base.cpu_state(model)
        record = {"epoch": epoch + 1, "train_loss": total / len(rows["labeled_train"]),
                  "development_auroc": dev_auc, "best_epoch": best_epoch,
                  "optimizer_updates": updates, "record_exposures": exposures,
                  "elapsed_seconds": previous_elapsed + time.monotonic() - started}
        if variant in CHUNK_VARIANTS:
            record["boundary_disagreement_with_fixed"] = disagreement_total / exposures
        history.append(record)
        base.save_epoch(directory, fp, epoch + 1, model, optimizer, generator,
                        history, best_model, best_auc, best_epoch)
        print(json.dumps({"stage": "train", "variant": variant,
                          "budget": budget, **record}), flush=True)
    if best_model is None:
        raise RuntimeError("No fine-tune epoch completed")
    model.load_state_dict(best_model)
    base.atomic_torch(directory / "model.pt", {"fingerprint": fp, "model": best_model,
                    "best_epoch": best_epoch, "best_development_auroc": best_auc})
    calibration_logits = predict(model, loader(pool, calibration, mean, std, boundaries,
                                 args.batch_size, False, None, args.device), args.device)
    test_logits = predict(model, loader(pool, rows["test"], mean, std, boundaries,
                            args.batch_size, False, None, args.device), args.device)
    base.evaluate_predictions(f"{variant}_fraction{budget}", calibration_logits,
        test_logits, calibration, rows["test"], directory, 42, args.bootstrap)
    names = ("config.json", "history.json", "model.pt", "metrics.json",
             "test_predictions.csv", "calibration_predictions.npz")
    base.atomic_json(completion, {"fingerprint": fp,
                      "artifacts": {name: base.digest_file(directory / name) for name in names}})


def profile(args, pool, mean, std, boundaries, bootstrap_weights):
    encoder_state, head_state, _ = bootstrap_weights
    for variant in (VARIANTS if args.variant == "all" else (args.variant,)):
        base.seed_all(42)
        model = TokenizationPretrainer(variant).to(args.device)
        load_bootstrap_weights(model, encoder_state, head_state)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                      lr=1e-3, weight_decay=0.01)
        generator = torch.Generator().manual_seed(42)
        data = loader(pool, pool.train_rows, mean, std, boundaries,
                      args.ssl_batch_size, True, generator, args.device)
        teacher = snapshot_teacher_convs(model.encoder).to(args.device) if variant == "clusteraux" else None
        centers = torch.randn(CLUSTERS, 256, generator=torch.Generator().manual_seed(42)).to(args.device)
        started = None
        measured = 0
        for update, (signal, _, _, beat_boundaries) in enumerate(data, 1):
            if update == 6:
                if args.device == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                started = time.monotonic()
            signal = signal.to(args.device, non_blocking=True)
            beat_boundaries = beat_boundaries.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, details = model(signal, beat_boundaries, centers, teacher)
            if variant in CHUNK_VARIANTS:
                details["boundary_disagreement_with_fixed"] = boundary_disagreement(model.encoder)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite profile loss")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise RuntimeError("Nonfinite profile gradients")
            optimizer.step()
            if update > 5:
                measured += len(signal)
            if update >= 5 + args.profile_updates:
                break
        if started is None:
            raise RuntimeError("Insufficient ECGs for profile")
        if args.device == "cuda":
            torch.cuda.synchronize()
        seconds = time.monotonic() - started
        print(json.dumps({"stage": "profile", "variant": variant,
             "warmup_updates": 5, "measured_updates": update - 5,
             "seconds_per_update": seconds / (update - 5),
             "records_per_second": measured / seconds,
             "estimated_ssl_epoch_seconds": len(pool.train_rows) * seconds / measured,
             "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9 if args.device == "cuda" else None,
             "model_parameters": sum(p.numel() for p in model.parameters()),
             "loss": float(loss.detach()), **details}), flush=True)


def report(args):
    comparisons = {}
    metadata = json.loads((args.cache_dir / "complete.json").read_text())
    count = metadata.get("split_counts", {}).get("train")
    title = (f"{count:,} combined MIMIC and PTB-XL training ECGs" if count is not None
             else "combined MIMIC and PTB-XL training ECGs")
    lines = [f"# Continued CPC tokenization on {title}", "",
             "All five arms begin from the same completed 20-epoch local CPC encoder and future-prediction heads, then continue for the same fixed SSL budget. Every arm uses query positions 24 onward in each five-second half, horizons 4/8/12, and same-half temporal negatives outside ±3 tokens of the target.", "",
             "The cluster arm predicts K=64 train-only CNN code IDs from a frozen teacher snapshot, refreshing that snapshot and codebook after epoch five. Chunk arms update context at emitted boundaries and pool emitted contexts for classification while preserving original causal CNN tokens as CPC targets. Fixed chunks provide the matched boundary control for beat and learned chunks; the native continuation uses its original grid-context readout.", "",
             "This is a one-seed exploratory comparison on a PTB-XL test set already used in earlier project experiments, not a fresh external confirmation cohort.", ""]
    comparisons_to_make = (("clusteraux", "continuation"),
                           ("fixedchunk", "continuation"),
                           ("beatchunk", "fixedchunk"),
                           ("learnedchunk", "fixedchunk"))
    for budget in ("1", "0.1"):
        paths = {variant: args.output_dir / f"{variant}_fraction{budget}_seed42"
                 for variant in VARIANTS}
        if not all((path / "completion.json").exists() for path in paths.values()):
            continue
        lines += [f"## {budget} label fraction", "",
                  "| Arm | AUROC | AP | Sensitivity | Specificity |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for variant, path in paths.items():
            result = json.loads((path / "metrics.json").read_text())["test"]
            lines.append(f"| {variant} | {result['auroc']:.3f} | {result['average_precision']:.3f} | {result['sensitivity']:.3f} | {result['specificity']:.3f} |")
        lines.append("")
        for left, right in comparisons_to_make:
            comparison = base.paired_comparison(paths[left], paths[right], args.bootstrap)
            comparisons[f"fraction{budget}_{left}_minus_{right}"] = comparison
            auc = comparison["auroc"]
            lines.append(f"{left} minus {right}: AUROC {auc['difference']:+.3f} "
                         f"(paired patient 95% CI {auc['ci95'][0]:+.3f} to {auc['ci95'][1]:+.3f}).")
        lines.append("")
    base.atomic_json(args.output_dir / "paired_comparisons.json", comparisons)
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("profile", "pretrain", "train", "all"), default="all")
    parser.add_argument("--variant", choices=(*VARIANTS, "all"), default="all")
    parser.add_argument("--labels", choices=("0.1", "1", "all"), default="all")
    parser.add_argument("--cache-dir", type=Path, default=base.DEFAULT_CACHE)
    parser.add_argument("--beat-dir", type=Path, default=DEFAULT_BEATS)
    parser.add_argument("--bootstrap-dir", type=Path, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--manifest-dir", type=Path, default=base.DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ssl-batch-size", type=int, default=128)
    parser.add_argument("--ssl-epochs", type=int, default=10)
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
        hashes = source_hashes(args, pool)
        beats = load_beats(args.beat_dir, pool)
        if args.stage == "profile":
            with tempfile.TemporaryDirectory(prefix="cpc_tokenization_profile_") as temporary:
                mean, std = pool.normalization(temporary, hashes)
                boot = bootstrap(args, hashes)
                profile(args, pool, mean, std, beats, boot)
            return
        mean, std = pool.normalization(args.output_dir, hashes)
        boot = bootstrap(args, hashes)
        selected = VARIANTS if args.variant == "all" else (args.variant,)
        codebook = initial_codebook(args, pool, mean, std, hashes, boot[0]) if (
            args.stage in ("pretrain", "all") and "clusteraux" in selected) else None
        if args.stage in ("pretrain", "all"):
            for variant in selected:
                pretrain(args, pool, mean, std, beats, hashes, boot, variant,
                         codebook if variant == "clusteraux" else None)
        if args.stage in ("train", "all"):
            for budget in (("1", "0.1") if args.labels == "all" else (args.labels,)):
                for variant in selected:
                    fine_tune(args, pool, mean, std, beats, hashes, variant, budget)
            report(args)


if __name__ == "__main__":
    main()
