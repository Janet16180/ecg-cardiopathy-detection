#!/usr/bin/env python3
"""Experiment 015: matched cached-JEPA representation distillation into local CPC."""

import argparse
import fcntl
import hashlib
import json
import os
import random
import signal
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.nn import functional as F

from ecg_experiment.bounded_waveform_cache import BoundedWaveformCache
from ecg_experiment.cpc import CPCClassifier, CPCEncoder
from ecg_experiment.data import read_manifest
from ecg_experiment.evaluation import select_threshold
from ecg_experiment.run import cpu_state, partition_validation
from scripts.run_cpc_experiment import Pool, atomic_json, atomic_torch, digest_file, seed_all


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/processed/cpc_pool_40k"
MANIFEST = ROOT / "data/processed/ptbxl"
TEACHER = ROOT / "data/processed/pretrained/ecg-jepa-full-public"
SSL = ROOT / "outputs/experiment004_cpc_40k/cpc_ssl/encoder.pt"
NORMALIZATION = ROOT / "outputs/experiment004_cpc_40k/normalization.json"
OUTPUT = ROOT / "outputs/experiment015_jepa_cpc_distillation"
GPU_LOCK = Path("/tmp/ecg_project_gpu.lock")
SEED = 42
EPOCHS = 5
BATCH = 128
SAVE_EVERY = 20
STOP = False


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_identity(path):
    stat = Path(path).stat()
    return {"device": stat.st_dev, "inode": stat.st_ino, "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}


def stop_handler(_signum, _frame):
    global STOP
    STOP = True


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CPCEncoder()
        self.head = nn.Linear(512, 1)
        self.projector = nn.Linear(512, 768)

    def forward(self, waveforms):
        _, contexts = self.encoder(waveforms)
        pooled = self.encoder.pooled(contexts)
        return self.head(pooled).squeeze(-1), self.projector(pooled)

    def classifier_state(self):
        return {key: value for key, value in cpu_state(self).items()
                if key.startswith(("encoder.", "head."))}


def masked_loss(logits, projected, targets, exposed, teacher, weight):
    """Run both graphs in both arms; only exposed labels contribute to BCE."""
    if exposed.dtype != torch.bool or exposed.shape != logits.shape:
        raise ValueError("Exposed-label mask must be boolean and match logits")
    if not exposed.any():
        raise ValueError("Every fixed batch must contain exposed labels")
    bce = F.binary_cross_entropy_with_logits(logits[exposed], targets[exposed])
    cosine = 1 - F.cosine_similarity(F.normalize(projected, dim=-1),
                                     F.normalize(teacher.detach(), dim=-1), dim=-1).mean()
    loss = bce + weight * cosine
    return loss, bce, cosine


def fixed_batches(size, exposed, epoch, batch_size=BATCH):
    """Each row appears once; the 1,518 labels are spread across all 120 batches."""
    rng = np.random.default_rng(SEED + 1009 * epoch)
    groups = [np.empty(0, dtype=np.int64) for _ in range((size + batch_size - 1) // batch_size)]
    labeled = rng.permutation(np.flatnonzero(exposed))
    hidden = rng.permutation(np.flatnonzero(~exposed))
    if not len(labeled):
        raise ValueError("No exposed labels")
    for i, part in enumerate(np.array_split(labeled, len(groups))):
        groups[i] = part
    offset = 0
    for i, group in enumerate(groups):
        capacity = min(batch_size, size - i * batch_size) - len(group)
        groups[i] = rng.permutation(np.concatenate((group, hidden[offset:offset + capacity])))
        offset += capacity
    batches = [group.tolist() for group in groups]
    flat = [i for group in batches for i in group]
    if len(flat) != size or len(set(flat)) != size or set(flat) != set(range(size)) or not all(exposed[group].any() for group in batches):
        raise AssertionError("Batch schedule lost or duplicated rows or labels")
    return batches


def load_inputs(args):
    started = time.monotonic()
    pool = Pool(args.cache_dir)
    full = read_manifest(args.manifest_dir / "seed42_fraction1/labeled_train.csv")
    limited = read_manifest(args.manifest_dir / "seed42_fraction0.1/labeled_train.csv")
    validation = read_manifest(args.manifest_dir / "seed42_fraction1/validation.csv")
    development, calibration = partition_validation(validation)
    test = read_manifest(args.manifest_dir / "seed42_fraction1/test.csv")
    if (len(full), len(limited), len(development), len(calibration), len(test)) != (15360, 1518, 1306, 564, 1896):
        raise ValueError("Frozen partition counts changed")
    train_ids = [row["ecg_id"] for row in full]
    limited_ids = {row["ecg_id"] for row in limited}
    if len(set(train_ids)) != len(full) or not limited_ids <= set(train_ids):
        raise ValueError("Training IDs duplicate or limited labels not subset")
    full_by_id = {r["ecg_id"]: r for r in full}
    if any((r["patient_id"], r["target"]) !=
           (full_by_id[r["ecg_id"]]["patient_id"], full_by_id[r["ecg_id"]]["target"])
           for r in limited):
        raise ValueError("Limited label manifest differs from full training identities/targets")
    for name in ("validation", "test"):
        if read_manifest(args.manifest_dir / f"seed42_fraction0.1/{name}.csv") != (validation if name == "validation" else test):
            raise ValueError(f"Label-budget {name} manifests differ")
    for rows, split in ((full, "train"), (development, "validation"), (calibration, "validation"), (test, "test")):
        for row in rows:
            cached = pool.rows[pool.index[row["ecg_id"]]]
            if (cached["source"], cached["split"], cached["patient_id"]) != ("ptbxl", split, row["patient_id"]):
                raise ValueError(f"Manifest/cache mismatch: {row['ecg_id']}")
    all_patients = [set(r["patient_id"] for r in rows) for rows in (full, development, calibration, test)]
    if any(all_patients[i] & all_patients[j] for i in range(4) for j in range(i + 1, 4)):
        raise ValueError("Patient partition overlap")
    teacher_meta = json.loads((args.teacher_dir / "metadata.json").read_text())
    if teacher_meta.get("feature_dimension") != 768 or teacher_meta.get("record_count") != 19126:
        raise ValueError("Unexpected JEPA teacher cache metadata")
    full_manifest_hash = digest_file(args.manifest_dir / "seed42_fraction1/labeled_train.csv")
    if teacher_meta.get("manifest_sha256", {}).get("labeled_train.csv") != full_manifest_hash:
        raise ValueError("Teacher training manifest differs from this label pool")
    teacher_ids = [str(item) for item in np.load(args.teacher_dir / "ecg_ids.npy", allow_pickle=False)]
    if len(set(teacher_ids)) != len(teacher_ids):
        raise ValueError("Duplicate teacher ECG IDs")
    teacher_index = {ecg_id: i for i, ecg_id in enumerate(teacher_ids)}
    if not set(train_ids) <= set(teacher_index):
        raise ValueError("Teacher misses eligible training ECGs")
    teacher_file = np.load(args.teacher_dir / "features.npy", mmap_mode="r")
    if teacher_file.shape != (19126, 768) or teacher_file.dtype != np.float32:
        raise ValueError("Unexpected teacher feature shape/dtype")
    vectors = np.array(teacher_file[[teacher_index[ecg_id] for ecg_id in train_ids]], copy=True)
    if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) < 1e-12):
        raise ValueError("Invalid teacher vectors")
    norm = json.loads(args.normalization.read_text())
    base_config = json.loads((args.ssl.parent / "config.json").read_text())
    base_cache = base_config["inputs"]["cache"]
    saved_ssl = torch.load(args.ssl, map_location="cpu", weights_only=True)
    final_ssl = torch.load(args.ssl.parent / "epoch_state.pt", map_location="cpu", weights_only=False)
    if (saved_ssl["fingerprint"] != base_config["fingerprint"] or
            final_ssl["fingerprint"] != base_config["fingerprint"] or final_ssl["epoch"] != 20 or
            any(not torch.equal(value, final_ssl["model"][f"encoder.{key}"])
                for key, value in saved_ssl["encoder"].items())):
        raise ValueError("Experiment 004 CPC encoder does not match final SSL state")
    del saved_ssl, final_ssl
    if norm["source"]["signals_sha256"] != base_cache[str((args.cache_dir / "signals.npy").resolve())]:
        raise ValueError("Normalization is not from the CPC SSL pool")
    if norm["source"]["train_ids_sha256"] != sha_json([r["ecg_id"] for r in pool.train_rows]):
        raise ValueError("Normalization training identities changed")
    mean = np.asarray(norm["mean"], np.float32)[:, None]
    std = np.asarray(norm["std"], np.float32)[:, None]
    if np.any(std <= 0) or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Invalid normalization")
    manifest_paths = [args.manifest_dir / f"seed42_fraction{budget}" / f"{name}.csv"
                      for budget in ("1", "0.1") for name in ("labeled_train", "validation", "test")]
    prior_path = args.output_dir / "provenance/verification.json"
    previous = json.loads(prior_path.read_text()) if prior_path.exists() else {}
    pool_hashes, pool_stats = {}, {}
    pool_hash_mode = "verified_receipt_reuse"
    for filename, metadata_key in (("signals.npy", "signals_sha256"),
                                   ("rows.csv", "rows_sha256"),
                                   ("ecg_ids.npy", "ecg_ids_sha256")):
        path = args.cache_dir / filename
        before = file_identity(path)
        prior_hash = previous.get("provenance", {}).get("pool_content_sha256", {}).get(filename)
        prior_stat = previous.get("pool_file_stats", {}).get(filename)
        if prior_hash == pool.metadata[metadata_key] and prior_stat == before:
            actual = prior_hash
        else:
            actual = digest_file(path)
            if file_identity(path) != before:
                raise ValueError(f"Waveform pool {filename} changed during hashing")
            pool_hash_mode = "full_content_sha256"
        if actual != pool.metadata[metadata_key]:
            raise ValueError(f"Waveform pool {filename} changed")
        pool_hashes[filename] = actual
        pool_stats[filename] = before
    provenance = {"sources": {str(path.resolve()): digest_file(path) for path in
                              [args.ssl, args.ssl.parent / "epoch_state.pt",
                               args.ssl.parent / "config.json", args.normalization,
                               args.teacher_dir / "metadata.json",
                               args.teacher_dir / "ecg_ids.npy", args.teacher_dir / "features.npy", *manifest_paths]},
                  "code": {name: digest_file(ROOT / name) for name in
                           ("scripts/run_jepa_cpc_distillation.py", "ecg_experiment/cpc.py",
                            "ecg_experiment/bounded_waveform_cache.py",
                            "scripts/run_cpc_experiment.py", "ecg_experiment/data.py",
                            "ecg_experiment/run.py", "ecg_experiment/evaluation.py",
                            "docs/astra-next-model-ideas.md", "docs/experiment-015-distillation.md")},
                  "pool_complete_sha256": digest_file(args.cache_dir / "complete.json"),
                  "pool_content_sha256": pool_hashes,
                  "base_cpc_config_fingerprint": base_config["fingerprint"],
                  "teacher_metadata": teacher_meta,
                  "settings": {"seed": SEED, "epochs": EPOCHS, "batch_size": BATCH,
                               "save_every_updates": SAVE_EVERY, "teacher_weight": 0.1,
                               "encoder_lr": 3e-4, "head_and_projection_lr": 1e-3,
                               "weight_decay": 0.01}}
    precheck_seconds = time.monotonic() - started
    if args.stage == "check":
        sample = np.array(pool.signals[pool.indices(full[:1])[0]], copy=True)
        if sample.shape != (12, 2500) or not np.isfinite(sample).all():
            raise ValueError("Invalid waveform")
        return locals()
    preload_started = time.monotonic()
    combined = full + development
    waveforms = BoundedWaveformCache(pool, combined, max_bytes=args.max_cache_bytes,
                                     reserve_bytes=args.reserve_bytes, expected_source="ptbxl")
    if len(waveforms.signals) != len(combined):
        raise ValueError("Bounded cache row mismatch")
    preload_seconds = time.monotonic() - preload_started
    return locals()


def make_model(ssl_path, device):
    seed_all(SEED)
    model = Student()
    checkpoint = torch.load(ssl_path, map_location="cpu", weights_only=True)
    if checkpoint.get("variant") != "cpc" or checkpoint.get("epochs") != 20:
        raise ValueError("Expected Experiment 004 ordinary 20-epoch CPC checkpoint")
    model.encoder.load_state_dict(checkpoint["encoder"], strict=True)
    return model.to(device)


def make_optimizer(model):
    return torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": 3e-4},
                              {"params": model.head.parameters(), "lr": 1e-3},
                              {"params": model.projector.parameters(), "lr": 1e-3}], weight_decay=0.01)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_state(directory, fingerprint, model, optimizer, epoch, batch, history, best, totals, elapsed):
    state = {"fingerprint": fingerprint, "model": cpu_state(model),
             "optimizer": optimizer.state_dict(), "rng": rng_state(),
             "epoch": epoch, "batch": batch, "history": history, "best": best,
             "totals": totals, "elapsed_seconds": elapsed}
    atomic_torch(directory / "resume.pt", state)
    atomic_json(directory / "history.json", history)


def load_state(directory, fingerprint, model, optimizer):
    path = directory / "resume.pt"
    if not path.exists():
        if (directory / "history.json").exists():
            raise ValueError("History without checkpoint")
        return 0, 0, [], {"auc": -1.0, "epoch": 0, "model": None}, {}, 0.0
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["fingerprint"] != fingerprint:
        raise ValueError("Resume fingerprint mismatch")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    restore_rng(state["rng"])
    return state["epoch"], state["batch"], state["history"], state["best"], state["totals"], state["elapsed_seconds"]


def verify_roundtrip(directory, args, provenance, budget, arm):
    path = directory / "resume.pt"
    original = torch.load(path, map_location="cpu", weights_only=False)
    probe = make_model(args.ssl, "cpu")
    optimizer = make_optimizer(probe)
    identity = {"provenance": provenance, "budget": budget, "arm": arm,
                "train_ids": [r["ecg_id"] for r in read_manifest(args.manifest_dir / "seed42_fraction1/labeled_train.csv")],
                "development_ids": [r["ecg_id"] for r in partition_validation(
                    read_manifest(args.manifest_dir / "seed42_fraction1/validation.csv"))[0]]}
    epoch, batch, _, _, _, _ = load_state(directory, sha_json(identity), probe, optimizer)
    if (epoch, batch) != (1, 0):
        raise ValueError("Profile did not save a complete epoch")
    if any(not torch.equal(value, probe.state_dict()[key]) for key, value in original["model"].items()):
        raise ValueError("Profile model checkpoint roundtrip differs")
    saved_opt = original["optimizer"]["state"]
    loaded_opt = optimizer.state_dict()["state"]
    for key, state in saved_opt.items():
        for field, value in state.items():
            other = loaded_opt[key][field]
            if torch.is_tensor(value) and not torch.equal(value, other):
                raise ValueError("Profile optimizer checkpoint roundtrip differs")
    return {"epoch": epoch, "batch": batch, "model_tensors": len(original["model"]),
            "optimizer_slots": len(saved_opt)}


def normalized_batch(cache, indices, mean, std, device):
    array = np.array(cache.signals[indices], copy=True)
    array -= mean
    array /= std
    return torch.from_numpy(array).to(device, non_blocking=True)


@torch.inference_mode()
def predict_development(model, data, device):
    model.eval()
    values = []
    representations = []
    for start in range(15360, 15360 + len(data["development"]), BATCH):
        stop = min(start + BATCH, 15360 + len(data["development"]))
        x = normalized_batch(data["waveforms"], list(range(start, stop)), data["mean"], data["std"], device)
        _, contexts = model.encoder(x)
        pooled = model.encoder.pooled(contexts)
        logits = model.head(pooled).squeeze(-1)
        values.extend(logits.cpu().numpy().tolist())
        representations.append(pooled.cpu().numpy())
    return np.asarray(values), np.concatenate(representations)


def development_screen(rows, logits):
    y = np.asarray([int(r["target"]) for r in rows])
    groups = np.asarray([r["patient_id"] for r in rows])
    auc = float(roc_auc_score(y, logits))
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    predictions = []
    probabilities = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
    for train, held in splitter.split(logits, y, groups):
        threshold = select_threshold(y[train], probabilities[train], 0.95)
        predictions.extend((int(y[i]), bool(probabilities[i] >= threshold)) for i in held)
    tp = sum(label == 1 and pred for label, pred in predictions)
    fn = sum(label == 1 and not pred for label, pred in predictions)
    tn = sum(label == 0 and not pred for label, pred in predictions)
    fp = sum(label == 0 and pred for label, pred in predictions)
    return {"auroc": auc, "fold_sensitivity": tp / (tp + fn),
            "fold_specificity": tn / (tn + fp), "fold_confusion": {"tp": tp, "fn": fn, "tn": tn, "fp": fp}}


def run_arm(args, data, budget, arm, directory, max_epochs, provenance, deadline=None):
    directory.mkdir(parents=True, exist_ok=True)
    weight = 0.1 if arm == "distill" else 0.0
    full = data["full"]
    limited_ids = {row["ecg_id"] for row in data["limited"]}
    exposed = np.asarray([budget == "1" or row["ecg_id"] in limited_ids for row in full], dtype=bool)
    targets = np.asarray([float(row["target"]) if mask else 0.0 for row, mask in zip(full, exposed)], dtype=np.float32)
    if int(exposed.sum()) != (15360 if budget == "1" else 1518):
        raise ValueError("Unexpected exposed label count")
    identity = {"provenance": provenance, "budget": budget, "arm": arm,
                "train_ids": [r["ecg_id"] for r in full],
                "development_ids": [r["ecg_id"] for r in data["development"]]}
    fingerprint = sha_json(identity)
    completion = directory / "completion.json"
    if completion.exists():
        done = json.loads(completion.read_text())
        if done["fingerprint"] != fingerprint:
            raise ValueError("Completed arm fingerprint mismatch")
        for name, field in (("history.json", "history_sha256"), ("best_student.pt", "best_student_sha256")):
            if digest_file(directory / name) != done[field]:
                raise ValueError(f"Completed artifact changed: {name}")
        return json.loads((directory / "history.json").read_text())
    config_path = directory / "config.json"
    if config_path.exists() and json.loads(config_path.read_text())["fingerprint"] != fingerprint:
        raise ValueError("Existing output uses different inputs or code")
    model = make_model(args.ssl, args.device)
    optimizer = make_optimizer(model)
    epoch, batch_pos, history, best, totals, elapsed = load_state(directory, fingerprint, model, optimizer)
    atomic_json(config_path, {"fingerprint": fingerprint, "identity": identity,
                              "model_parameters": sum(p.numel() for p in model.parameters()),
                              "inference_parameters": sum(p.numel() for p in model.encoder.parameters()) + sum(p.numel() for p in model.head.parameters()),
                              "exposed_training_labels": int(exposed.sum())})
    started = time.monotonic()
    while epoch < max_epochs:
        model.train()
        batches = fixed_batches(len(full), exposed, epoch)
        for position in range(batch_pos, len(batches)):
            indices = batches[position]
            x = normalized_batch(data["waveforms"], indices, data["mean"], data["std"], args.device)
            labels = torch.from_numpy(targets[indices]).to(args.device)
            mask = torch.from_numpy(exposed[indices]).to(args.device)
            teacher = torch.from_numpy(data["vectors"][indices]).to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits, projection = model(x)
            loss, bce, cosine = masked_loss(logits, projection, labels, mask, teacher, weight)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise RuntimeError("Nonfinite gradient")
            optimizer.step()
            totals["updates"] = totals.get("updates", 0) + 1
            totals["record_exposures"] = totals.get("record_exposures", 0) + len(indices)
            totals["label_exposures"] = totals.get("label_exposures", 0) + int(mask.sum())
            totals["bce_sum"] = totals.get("bce_sum", 0.0) + float(bce.detach())
            totals["cosine_sum"] = totals.get("cosine_sum", 0.0) + float(cosine.detach())
            batch_pos = position + 1
            timed_out = deadline is not None and time.monotonic() >= deadline
            if batch_pos % SAVE_EVERY == 0 or STOP or timed_out:
                save_state(directory, fingerprint, model, optimizer, epoch, batch_pos, history,
                           best, totals, elapsed + time.monotonic() - started)
            if STOP or timed_out:
                raise SystemExit(75)
        logits, representations = predict_development(model, data, args.device)
        screen = development_screen(data["development"], logits)
        screen["mean_feature_variance"] = float(np.var(representations, axis=0).mean())
        if not np.isfinite(screen["mean_feature_variance"]):
            raise RuntimeError("Nonfinite development representations")
        if any(not torch.isfinite(param).all() for param in model.parameters()):
            raise RuntimeError("Nonfinite model weights")
        if screen["auroc"] > best["auc"]:
            best = {"auc": screen["auroc"], "epoch": epoch + 1, "model": model.classifier_state()}
            atomic_torch(directory / "best_student.pt", {"fingerprint": fingerprint,
                         "epoch": epoch + 1, "model": best["model"]})
        record = {"epoch": epoch + 1, "arm": arm, "budget": budget,
                  "train_bce_mean_batch": totals["bce_sum"] / totals["updates"],
                  "train_cosine_mean_batch": totals["cosine_sum"] / totals["updates"],
                  "optimizer_updates": totals["updates"], "record_exposures": totals["record_exposures"],
                  "label_exposures": totals["label_exposures"], "development": screen,
                  "best_epoch": best["epoch"],
                  "elapsed_seconds": elapsed + time.monotonic() - started}
        history.append(record)
        print(json.dumps(record), flush=True)
        epoch += 1
        batch_pos, totals = 0, {}
        save_state(directory, fingerprint, model, optimizer, epoch, 0, history, best, totals,
                   elapsed + time.monotonic() - started)
    if epoch == EPOCHS and max_epochs == EPOCHS:
        atomic_json(completion, {"fingerprint": fingerprint,
                    "best_epoch": best["epoch"], "best_development_auroc": best["auc"],
                    "history_sha256": digest_file(directory / "history.json"),
                    "best_student_sha256": digest_file(directory / "best_student.pt")})
    return history


def report_pilot(output_dir):
    results = {}
    lines = ["# Experiment 015 development-only distillation pilot", "",
             "Five fixed epochs per arm; no calibration or test predictions were used.", "",
             "| Label budget | Arm | Best epoch | Development AUROC | Patient-fold specificity | Patient-fold sensitivity | Mean feature variance | Updates | Labeled exposures | Wall seconds |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for budget in ("1", "0.1"):
        for arm in ("control", "distill"):
            directory = output_dir / f"{arm}_fraction{budget}_seed42"
            if not (directory / "completion.json").exists():
                return
            history = json.loads((directory / "history.json").read_text())
            if len(history) != EPOCHS:
                raise ValueError("Completed pilot lacks five epochs")
            best_epoch = json.loads((directory / "completion.json").read_text())["best_epoch"]
            row = history[best_epoch - 1]
            results[(budget, arm)] = row
            dev = row["development"]
            lines.append(f"| {budget} | {arm} | {best_epoch} | {dev['auroc']:.4f} | "
                         f"{dev['fold_specificity']:.4f} | {dev['fold_sensitivity']:.4f} | "
                         f"{dev['mean_feature_variance']:.6g} | "
                         f"{sum(r['optimizer_updates'] for r in history)} | "
                         f"{sum(r['label_exposures'] for r in history)} | "
                         f"{history[-1]['elapsed_seconds']:.1f} |")
    lines += ["", "## Prespecified development screen", ""]
    positive = False
    sensitivity_safe = True
    for budget in ("1", "0.1"):
        a, b = results[(budget, "distill")]["development"], results[(budget, "control")]["development"]
        delta_auc = a["auroc"] - b["auroc"]
        delta_spec = a["fold_specificity"] - b["fold_specificity"]
        delta_sens = a["fold_sensitivity"] - b["fold_sensitivity"]
        variance_ratio = a["mean_feature_variance"] / max(b["mean_feature_variance"], 1e-20)
        positive_here = (delta_auc >= 0.002 and delta_spec >= 0.02 and
                         delta_sens >= -0.005 and variance_ratio >= 0.1)
        positive |= positive_here
        sensitivity_safe &= delta_sens >= -0.005
        lines.append(f"Budget {budget}: distill minus control AUROC {delta_auc:+.4f}, "
                     f"patient-fold specificity {delta_spec:+.4f}, sensitivity {delta_sens:+.4f}, "
                     f"feature-variance ratio {variance_ratio:.3f}; "
                     f"positive-screen criteria {'met' if positive_here else 'not met'}.")
    other_budget_safe = all(results[(budget, "distill")]["development"]["auroc"] -
                            results[(budget, "control")]["development"]["auroc"] >= -0.002
                            for budget in ("1", "0.1"))
    lines += ["", ("A second matched seed is warranted before calibration/test."
                   if positive and other_budget_safe and sensitivity_safe else
                   "The prespecified development go/no-go screen did not pass; retain all runs as exploratory evidence."),
              "", "The teacher has external pretraining exposure and uncertain overlap. "
              "The endpoint is a diagnostic annotation proxy, not verified health or referral need.", ""]
    (output_dir / "report.md").write_text("\n".join(lines))
    atomic_json(output_dir / "completion.json", {"status": "development_pilot_complete",
                "report_sha256": digest_file(output_dir / "report.md"),
                "advance_to_second_seed": bool(positive and other_budget_safe and sensitivity_safe),
                "arm_completions_sha256": {
                    f"{arm}_fraction{budget}_seed42": digest_file(output_dir / f"{arm}_fraction{budget}_seed42/completion.json")
                    for budget in ("1", "0.1") for arm in ("control", "distill")}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("check", "profile", "train"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--manifest-dir", type=Path, default=MANIFEST)
    parser.add_argument("--teacher-dir", type=Path, default=TEACHER)
    parser.add_argument("--ssl", type=Path, default=SSL)
    parser.add_argument("--normalization", type=Path, default=NORMALIZATION)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--max-cache-bytes", type=int, default=2_400_000_000)
    parser.add_argument("--reserve-bytes", type=int, default=1_000_000_000)
    parser.add_argument("--max-wall-seconds", type=int, default=7200)
    args = parser.parse_args()
    if args.threads < 1 or args.max_cache_bytes < 2_100_000_000 or args.reserve_bytes < 0 or args.max_wall_seconds < 1:
        parser.error("Invalid threads or cache memory limits")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    torch.set_num_threads(args.threads)
    signal.signal(signal.SIGTERM, stop_handler)
    deadline = time.monotonic() + args.max_wall_seconds
    with GPU_LOCK.open("a+") as lock:
        if args.device == "cuda":
            print(f"Waiting for GPU lock {GPU_LOCK}", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
        data = load_inputs(args)
        if args.stage == "check":
            receipt = {"stage": "check", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                       "command": [sys.executable, "-m", "scripts.run_jepa_cpc_distillation", *sys.argv[1:]],
                       "train": len(data["full"]), "limited": len(data["limited"]),
                       "development": len(data["development"]),
                       "calibration": len(data["calibration"]), "test": len(data["test"]),
                       "precheck_seconds": data["precheck_seconds"],
                       "pool_hash_mode": data["pool_hash_mode"],
                       "pool_file_stats": data["pool_stats"],
                       "fingerprint": sha_json(data["provenance"]),
                       "provenance": data["provenance"]}
            atomic_json(args.output_dir / "provenance/verification.json", receipt)
            print(json.dumps({key: value for key, value in receipt.items() if key != "provenance"}), flush=True)
            return
        check_path = args.output_dir / "provenance/verification.json"
        if args.device == "cuda":
            if not check_path.exists() or json.loads(check_path.read_text())["fingerprint"] != sha_json(data["provenance"]):
                raise ValueError("Verified CPU input/source receipt is missing or changed")
        print(json.dumps({"stage": args.stage, "precheck_seconds": data["precheck_seconds"],
                          "preload_seconds": data["preload_seconds"],
                          "pool_hash_mode": data["pool_hash_mode"],
                          "cache_bytes": int(data["waveforms"].signals.nbytes)}), flush=True)
        if args.stage == "profile":
            with tempfile.TemporaryDirectory(prefix="experiment015_profile_") as temporary:
                durations = {}
                roundtrips = {}
                for arm in ("control", "distill"):
                    start = time.monotonic()
                    run_arm(args, data, "1", arm, Path(temporary) / arm, 1, data["provenance"], deadline)
                    roundtrips[arm] = verify_roundtrip(Path(temporary) / arm, args, data["provenance"], "1", arm)
                    durations[arm] = time.monotonic() - start
                estimate = data["precheck_seconds"] + data["preload_seconds"] + 2 * EPOCHS * sum(durations.values())
                receipt = {"stage": "profile", "full_epoch_seconds": durations,
                           "checkpoint_roundtrips": roundtrips,
                           "precheck_seconds": data["precheck_seconds"],
                           "preload_seconds": data["preload_seconds"],
                           "conservative_four_run_seconds": estimate,
                           "planning_gate_seconds": 7200,
                           "gate_passed": estimate <= 7200,
                           "peak_gpu_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
                           "fingerprint": sha_json(data["provenance"])}
                args.output_dir.mkdir(parents=True, exist_ok=True)
                atomic_json(args.output_dir / "profile.json", receipt)
                print(json.dumps(receipt), flush=True)
            return
        receipt_path = args.output_dir / "profile.json"
        if args.device == "cuda":
            if not receipt_path.exists():
                raise ValueError("A complete real-data GPU profile is required before training")
            receipt = json.loads(receipt_path.read_text())
            if receipt["fingerprint"] != sha_json(data["provenance"]) or not receipt["gate_passed"]:
                raise ValueError("GPU profile fingerprint or two-hour planning gate failed")
        for budget in ("1", "0.1"):
            for arm in ("control", "distill"):
                if time.monotonic() >= deadline:
                    raise SystemExit(75)
                run_arm(args, data, budget, arm,
                        args.output_dir / f"{arm}_fraction{budget}_seed42", EPOCHS, data["provenance"], deadline)
        report_pilot(args.output_dir)


if __name__ == "__main__":
    main()
