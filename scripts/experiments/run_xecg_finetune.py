#!/usr/bin/env python3
"""Reproducible xECG fine-tuning on the fixed PTB-XL proxy task.

The official pretrained encoder sees a full ten-second, twelve-lead ECG in
physical mV at 100 Hz. Development patients select the epoch; calibration
patients set the probability threshold; test patients are evaluated once.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ecg_experiment.data import read_manifest
from ecg_experiment.evaluation import evaluate_predictions, partition_validation
from ecg_experiment.files import sha256_file
from ecg_experiment.provenance import git_head
from ecg_experiment.reproducibility import cpu_state, seed_everything


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_ROOT = ROOT / "data/processed/ptbxl"
DEFAULT_RAW_DIR = ROOT / "data/raw/ptb-xl/1.0.3"
DEFAULT_CHECKPOINT_DIR = ROOT / "third_party/checkpoints/xecg"
DEFAULT_CACHE_DIR = ROOT / "data/processed/ptbxl/xecg_views"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/experiment007_xecg"
EXPECTED_COUNTS = {"full": 15360, "ten_percent": 1518}


def budgets(value: str) -> tuple[str, ...]:
    if value == "all":
        return ("full", "ten_percent")
    if value in EXPECTED_COUNTS:
        return (value,)
    raise ValueError(f"Unknown label budget: {value}")


def manifest_dir(root: Path, budget: str) -> Path:
    return root / ("seed42_fraction1" if budget == "full" else "seed42_fraction0.1")


def _identities(rows):
    return {row["ecg_id"]: (row["patient_id"], row["filename_hr"]) for row in rows}


def load_manifests(root: Path, budget: str):
    """Check both budgets and all patient boundaries before fitting either."""
    full, small = manifest_dir(root, "full"), manifest_dir(root, "ten_percent")
    names = ("labeled_train", "validation", "test")
    rows = {name: read_manifest(manifest_dir(root, budget) / f"{name}.csv") for name in names}
    full_train = read_manifest(full / "labeled_train.csv")
    small_train = read_manifest(small / "labeled_train.csv")
    if len(full_train) != EXPECTED_COUNTS["full"] or len(small_train) != EXPECTED_COUNTS["ten_percent"]:
        raise ValueError("Labeled manifests do not match the fixed PTB-XL budgets")
    full_by_id = {row["ecg_id"]: row for row in full_train}
    if len(full_by_id) != len(full_train) or len(_identities(small_train)) != len(small_train):
        raise ValueError("Duplicate training ECG identifiers")
    for row in small_train:
        if row["ecg_id"] not in full_by_id or row != full_by_id[row["ecg_id"]]:
            raise ValueError("Ten-percent training manifest differs from full training manifest")
    for name in ("validation", "test"):
        other = read_manifest(full / f"{name}.csv")
        if rows[name] != other:
            raise ValueError(f"{name} differs across label budgets")
    if len(rows["validation"]) != 1870 or len(rows["test"]) != 1896:
        raise ValueError("Unexpected official validation or test budget")
    ids = [row["ecg_id"] for name in names for row in rows[name]]
    if len(ids) != len(set(ids)):
        raise ValueError("Training, validation, and test ECG identifiers overlap")
    patients = [{row["patient_id"] for row in rows[name]} for name in names]
    if any(patients[i] & patients[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Training, validation, and test patients overlap")
    if len(rows["labeled_train"]) != EXPECTED_COUNTS[budget]:
        raise ValueError("Selected training budget has wrong size")
    if {row["target"] for row in rows["labeled_train"]} != {"0", "1"}:
        raise ValueError("Training labels must contain both classes")
    development, calibration = partition_validation(rows["validation"])
    if len(development) != 1306 or len(calibration) != 564:
        raise ValueError("Development/calibration split differs from the fixed protocol")
    hashes = {f"{name}.csv": sha256_file(manifest_dir(root, budget) / f"{name}.csv") for name in names}
    hashes["full_labeled_train.csv"] = sha256_file(full / "labeled_train.csv")
    hashes["ten_percent_labeled_train.csv"] = sha256_file(small / "labeled_train.csv")
    return rows, development, calibration, hashes


class CachedECGs(Dataset):
    def __init__(self, cache, index, rows):
        self.cache = cache
        self.indices = [index[int(row["ecg_id"])] for row in rows]
        self.targets = [float(row["target"]) for row in rows]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        signal = np.array(self.cache[self.indices[i]], dtype=np.float32, copy=True)
        if signal.shape != (1000, 12) or not np.isfinite(signal).all():
            raise ValueError("xECG cache must contain finite [1000,12] physical-mV ECGs")
        return torch.from_numpy(signal), torch.tensor(self.targets[i], dtype=torch.float32)


def layerwise_parameter_groups(model: nn.Module, core_lr=3e-5, head_lr=1e-3,
                               weight_decay=0.1, decay=0.75):
    """Assign nine xLSTM blocks monotonically increasing LRs, as official PTB-XL."""
    blocks = model.backbone.core.model.blocks
    if len(blocks) != 9:
        raise ValueError(f"Expected nine xECG backbone blocks, found {len(blocks)}")
    groups = []
    used = set()

    def add(name, params, lr):
        selected = [param for param in params if param.requires_grad]
        for param in selected:
            if id(param) in used:
                raise ValueError(f"Overlapping optimizer parameter group: {name}")
            used.add(id(param))
        if selected:
            groups.append({"params": selected, "lr": lr, "weight_decay": weight_decay, "name": name})

    add("patch_embedding", model.backbone.patch_embedding.parameters(), core_lr * decay ** 10)
    for index, block in enumerate(blocks):
        add(f"core_block_{index}", block.parameters(), core_lr * decay ** (9 - index))
    remaining = [param for param in model.backbone.parameters() if param.requires_grad and id(param) not in used]
    add("core_remaining", remaining, core_lr)
    add("binary_head", model.head.parameters(), head_lr)
    expected = {id(param) for param in model.parameters() if param.requires_grad}
    if expected != used:
        raise ValueError("Optimizer groups do not cover the full xECG classifier")
    return groups


def make_scheduler(optimizer, steps_per_epoch: int, epochs: int):
    warmup, total = steps_per_epoch, steps_per_epoch * epochs

    def factor(step):
        if step < warmup:
            return float(step) / max(1, warmup)
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def save_atomic_torch(path: Path, value):
    temporary = path.with_name(path.name + ".partial")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_atomic_json(path: Path, value):
    temporary = path.with_name(path.name + ".partial")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_resume(path, fingerprint, model, optimizer, scheduler, generator,
                best_state, best_epoch, best_auc, history, elapsed):
    numpy_state = np.random.get_state()
    save_atomic_torch(path, {
        "version": 1, "fingerprint": fingerprint, "model": cpu_state(model),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "loader_rng": generator.get_state(), "best_model": best_state,
        "best_epoch": best_epoch, "best_auc": best_auc, "history": history,
        "elapsed_seconds": elapsed, "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python_rng": random.getstate(),
        "numpy_rng": (numpy_state[0], numpy_state[1].tolist(), numpy_state[2], numpy_state[3], numpy_state[4]),
    })


def load_resume(path, fingerprint, model, optimizer, scheduler, generator):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("version") != 1 or saved.get("fingerprint") != fingerprint:
        raise ValueError("xECG resume configuration differs from current inputs")
    history = saved["history"]
    if (not history or any(row.get("epoch") != index + 1 for index, row in enumerate(history))
            or saved["best_epoch"] not in range(1, len(history) + 1)
            or saved["best_model"] is None):
        raise ValueError("Malformed xECG resume checkpoint")
    model.load_state_dict(saved["model"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    generator.set_state(saved["loader_rng"])
    torch.set_rng_state(saved["torch_rng"])
    if torch.cuda.is_available():
        if len(saved["cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from resume checkpoint")
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    random.setstate(saved["python_rng"])
    state = saved["numpy_rng"]
    np.random.set_state((state[0], np.array(state[1], dtype=np.uint32), state[2], state[3], state[4]))
    return saved


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    return np.concatenate([model(signal.to(device)).float().cpu().numpy() for signal, _ in loader])


def train_epoch(model, loader, optimizer, scheduler, device, effective_batch_size,
                grad_clip=3.0):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss, seen, steps = 0.0, 0, 0
    for signal, target in loader:
        signal, target = signal.to(device), target.to(device)
        group_start = (seen // effective_batch_size) * effective_batch_size
        group_size = min(effective_batch_size, len(loader.dataset) - group_start)
        logits = model(signal)
        summed = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="sum")
        if not torch.isfinite(summed):
            raise RuntimeError("Nonfinite xECG fine-tuning loss")
        (summed / group_size).backward()
        total_loss += float(summed.detach())
        seen += len(target)
        if seen - group_start == group_size:
            norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if not torch.isfinite(norm):
                raise RuntimeError("Nonfinite xECG gradient norm")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
    if seen != len(loader.dataset) or steps != math.ceil(seen / effective_batch_size):
        raise RuntimeError("xECG optimizer update count differs from planned budget")
    return total_loss / seen, steps


def fit(model, optimizer, scheduler, train_loader, dev_loader, development_y,
        output_dir: Path, fingerprint, generator, device, epochs, patience,
        effective_batch_size, resume=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_path = output_dir / "resume.pt"
    history, best_epoch, best_auc, best_state = [], 0, -1.0, None
    started = time.monotonic()
    if resume:
        if not resume_path.is_file():
            raise FileNotFoundError(f"No complete xECG epoch to resume: {resume_path}")
        saved = load_resume(resume_path, fingerprint, model, optimizer, scheduler, generator)
        history, best_epoch, best_auc, best_state = (saved["history"], saved["best_epoch"],
                                                      saved["best_auc"], saved["best_model"])
        started -= saved["elapsed_seconds"]
    elif resume_path.exists():
        raise FileExistsError(f"Incomplete xECG run exists; use --resume: {resume_path}")
    for epoch in range(len(history), epochs):
        if epoch - best_epoch >= patience:
            break
        loss, updates = train_epoch(model, train_loader, optimizer, scheduler, device, effective_batch_size)
        dev_logits = predict(model, dev_loader, device)
        if not np.isfinite(dev_logits).all():
            raise RuntimeError("Nonfinite development logits")
        auc = float(roc_auc_score(development_y, dev_logits))
        if auc > best_auc:
            best_auc, best_epoch, best_state = auc, epoch + 1, cpu_state(model)
        row = {"epoch": epoch + 1, "loss": loss, "development_auroc": auc,
               "optimizer_updates": updates, "seconds": time.monotonic() - started}
        history.append(row)
        save_atomic_json(output_dir / "history.json", history)
        save_resume(resume_path, fingerprint, model, optimizer, scheduler, generator,
                    best_state, best_epoch, best_auc, history, row["seconds"])
        print(json.dumps({"stage": "xecg_train", **row}), flush=True)
        if epoch + 1 - best_epoch >= patience:
            break
    if best_state is None:
        raise RuntimeError("No xECG checkpoint selected")
    model.load_state_dict(best_state, strict=True)
    return {"history": history, "best_epoch": best_epoch, "best_development_auroc": best_auc,
            "elapsed_seconds": time.monotonic() - started}


def completion_hashes(output_dir: Path):
    return {name: sha256_file(output_dir / name) for name in
            ("model.pt", "metrics.json", "test_predictions.csv", "calibration_predictions.npz",
             "config.json", "history.json")}


def valid_completion(output_dir: Path, fingerprint):
    marker = output_dir / "complete.json"
    if not marker.is_file():
        return False
    data = json.loads(marker.read_text())
    if data.get("fingerprint") != fingerprint:
        raise ValueError(f"Completed xECG run differs from requested inputs: {output_dir}")
    if data.get("sha256") != completion_hashes(output_dir):
        raise ValueError(f"Completed xECG artifact hash mismatch: {output_dir}")
    return True


@contextlib.contextmanager
def gpu_lock(device):
    if device != "cuda":
        yield
        return
    path = Path("/tmp/ecg_project_gpu.lock")
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"GPU is reserved by another project run: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def checkpoint_fingerprint(checkpoint_dir: Path):
    required = ("model.safetensors", "config.json", "xECG.py", "download_provenance.json")
    for name in required:
        if not (checkpoint_dir / name).is_file():
            raise FileNotFoundError(f"Missing official xECG checkpoint component: {checkpoint_dir / name}")
    return {name: sha256_file(checkpoint_dir / name) for name in required}


def source_tree_sha256(root: Path):
    digest = hashlib.sha256()
    files = sorted(root.rglob("*.py"))
    if not files:
        raise FileNotFoundError(f"No Python source files in {root}")
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def resume_for_budget(directory: Path, requested: bool) -> bool:
    """A completed first budget may precede an untouched second budget."""
    resume_path = directory / "resume.pt"
    if resume_path.is_file():
        if not requested:
            raise FileExistsError(f"Incomplete xECG run exists; use --resume: {resume_path}")
        return True
    if directory.is_dir() and any(directory.iterdir()):
        raise FileExistsError(f"Partial xECG run lacks an epoch checkpoint: {directory}")
    return False


def cache_fingerprint(cache_dir: Path, raw_dir: Path, full_manifest_dir: Path):
    metadata_path = cache_dir / "metadata.json"
    views_path = cache_dir / "views.npy"
    if not all(path.is_file() for path in (metadata_path, views_path)):
        raise FileNotFoundError(f"xECG 500 Hz to 100 Hz cache is missing from {cache_dir}; run --stage prep")
    metadata = json.loads(metadata_path.read_text())
    ids = [int(ecg_id) for ecg_id in metadata["ecg_ids"]]
    views = np.load(views_path, mmap_mode="r")
    if views.shape != (len(ids), 1000, 12) or views.dtype != np.float32:
        raise ValueError("Malformed xECG FFT cache")
    if len(ids) != 19126 or len(set(ids)) != len(ids):
        raise ValueError("xECG FFT cache lacks the exact PTB-XL train/validation/test ECG union")
    if metadata.get("shape") != list(views.shape) or metadata.get("dtype") != "float32":
        raise ValueError("xECG cache metadata does not match waveform array")
    if metadata.get("raw_dir") != str(raw_dir.resolve()):
        raise ValueError("xECG cache was made from another PTB-XL waveform directory")
    requested = {name: sha256_file(full_manifest_dir / name) for name in
                 ("labeled_train.csv", "validation.csv", "test.csv")}
    if metadata.get("manifest_sha256") != requested:
        raise ValueError("xECG cache was made from another full-label manifest")
    if (metadata.get("preparation_sha256") != sha256_file(ROOT / "scripts/data/prepare_xecg.py")
            or metadata.get("adapter_sha256") != sha256_file(ROOT / "ecg_experiment/xecg.py")):
        raise ValueError("xECG cache preprocessing source differs from current source")
    return views, {ecg_id: index for index, ecg_id in enumerate(ids)}, {
        "metadata_sha256": sha256_file(metadata_path),
        "views_sha256": sha256_file(views_path), "metadata": metadata,
    }


def run_profile(args, rows, development, views, index, checkpoint_hashes, cache_hashes):
    from ecg_experiment.xecg import XECGBinaryClassifier, load_xecg

    seed_everything(args.seed)
    backbone = load_xecg(args.checkpoint_dir, backend="vanilla", device=args.device,
                         drop_path_prob=0.5)
    model = XECGBinaryClassifier(backbone).to(args.device)
    optimizer = torch.optim.AdamW(layerwise_parameter_groups(model), weight_decay=0.1)
    scheduler = make_scheduler(optimizer, math.ceil(len(rows["labeled_train"]) / args.effective_batch_size), args.epochs)
    train_loader = DataLoader(CachedECGs(views, index, rows["labeled_train"]),
                              batch_size=args.microbatch_size, shuffle=False, num_workers=0)
    dev_loader = DataLoader(CachedECGs(views, index, development),
                            batch_size=args.microbatch_size, shuffle=False, num_workers=0)
    signal, target = next(iter(train_loader))
    signal, target = signal.to(args.device), target.to(args.device)
    model.train()
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    step_seconds = []
    for _ in range(2):
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        logits = model(signal)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
        if logits.shape != target.shape or not torch.isfinite(loss):
            raise RuntimeError("xECG profile produced invalid logits/loss")
        loss.backward()
        if not all(param.grad is None or torch.isfinite(param.grad).all() for param in model.parameters()):
            raise RuntimeError("xECG profile produced nonfinite gradients")
        norm = nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        if not torch.isfinite(norm):
            raise RuntimeError("xECG profile produced nonfinite gradient norm")
        optimizer.step()
        scheduler.step()
        if not all(torch.isfinite(param).all() for param in model.parameters()):
            raise RuntimeError("xECG profile produced nonfinite parameters after optimizer step")
        if args.device == "cuda":
            torch.cuda.synchronize()
        step_seconds.append(time.monotonic() - started)
    if args.device == "cuda":
        torch.cuda.synchronize()
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    with torch.inference_mode():
        dev_signal, _ = next(iter(dev_loader))
        dev_logits = model(dev_signal.to(args.device))
    if not torch.isfinite(dev_logits).all():
        raise RuntimeError("xECG profile produced nonfinite development logits")
    result = {
        "device": args.device, "device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "microbatch_size": args.microbatch_size, "effective_batch_size": args.effective_batch_size,
        "first_optimizer_step_seconds": step_seconds[0],
        "steady_state_optimizer_step_seconds": step_seconds[1],
        "optimizer_state_bytes": sum(value.numel() * value.element_size()
                                     for state in optimizer.state.values() for value in state.values()
                                     if isinstance(value, torch.Tensor)),
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
        "logit_shape": list(logits.shape), "loss": float(loss.detach()),
        "checkpoint_sha256": checkpoint_hashes, "cache_sha256": {k: v for k, v in cache_hashes.items() if k != "metadata"},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_atomic_json(args.output_dir / f"profile_{args.device}.json", result)
    print(json.dumps({"stage": "xecg_profile", **result}), flush=True)


def train_budget(args, budget, rows, development, calibration, hashes, views, index,
                 checkpoint_hashes, cache_hashes):
    from ecg_experiment.xecg import XECGBinaryClassifier, load_xecg

    directory = args.output_dir / f"xecg_{budget}_seed{args.seed}"
    fingerprint = {
        "budget": budget, "seed": args.seed, "epochs": args.epochs, "patience": args.patience,
        "microbatch_size": args.microbatch_size, "effective_batch_size": args.effective_batch_size,
        "bootstrap": args.bootstrap, "device": args.device, "threads": args.threads,
        "raw_dir": str(args.raw_dir.resolve()), "cache_dir": str(args.cache_dir.resolve()),
        "checkpoint_dir": str(args.checkpoint_dir.resolve()), "checkpoint_sha256": checkpoint_hashes,
        "cache_sha256": {k: v for k, v in cache_hashes.items() if k != "metadata"},
        "manifest_sha256": hashes,
        "runner_source_sha256": sha256_file(Path(__file__)),
        "adapter_source_sha256": sha256_file(ROOT / "ecg_experiment/xecg.py"),
        "xlstm_python_source_sha256": source_tree_sha256(ROOT / "third_party/xecg-deps/xlstm"),
        "official_ptbxl_config_sha256": sha256_file(ROOT / "third_party/bench-xecg/configs/ptb-xl/xlstm_ft.yaml"),
        "official_ptbxl_defaults_sha256": sha256_file(ROOT / "third_party/bench-xecg/config_defaults/train_ptb_xl_defaults.yaml"),
        "evaluation_source_sha256": sha256_file(ROOT / "ecg_experiment/evaluation.py"),
        "run_source_sha256": sha256_file(ROOT / "ecg_experiment/run.py"),
        "reproducibility_source_sha256": sha256_file(ROOT / "ecg_experiment/reproducibility.py"),
        "pretrained_checkpoint_source_commit": git_head(ROOT / "third_party/bench-xecg"),
    }
    if valid_completion(directory, fingerprint):
        print(json.dumps({"stage": "xecg_reuse", "budget": budget, "directory": str(directory)}), flush=True)
        return
    resume_this_budget = resume_for_budget(directory, args.resume)
    if args.resume and (directory / "config.json").is_file():
        existing = json.loads((directory / "config.json").read_text())
        if existing.get("fingerprint") != fingerprint:
            raise ValueError(f"xECG resume configuration differs: {directory}")
    seed_everything(args.seed)
    backbone = load_xecg(args.checkpoint_dir, backend="vanilla", device=args.device,
                         drop_path_prob=0.5)
    model = XECGBinaryClassifier(backbone).to(args.device)
    groups = layerwise_parameter_groups(model)
    optimizer = torch.optim.AdamW(groups, weight_decay=0.1)
    steps_per_epoch = math.ceil(len(rows["labeled_train"]) / args.effective_batch_size)
    scheduler = make_scheduler(optimizer, steps_per_epoch, args.epochs)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(CachedECGs(views, index, rows["labeled_train"]),
                              batch_size=args.microbatch_size, shuffle=True,
                              num_workers=0, generator=generator)
    dev_loader = DataLoader(CachedECGs(views, index, development),
                            batch_size=args.microbatch_size, num_workers=0)
    development_y = np.array([int(row["target"]) for row in development])
    directory.mkdir(parents=True, exist_ok=True)
    config = {"fingerprint": fingerprint, "model": "xECG", "task": "PTB-XL diagnostic abnormality proxy",
              "preprocessing": "Official 500 Hz physical-mV ECG FFT-resampled to 100 Hz; full 10 seconds, canonical 12 leads; no normalization",
              "augmentation": "none", "head": "Identity + Linear(1024,1)",
              "optimizer": "AdamW", "core_lr": 3e-5, "head_lr": 1e-3,
              "weight_decay": 0.1, "layerwise_lr_decay": 0.75, "gradient_clip": 3.0,
              "scheduler": "one-epoch linear warmup then half-cosine per optimizer step",
              "drop_path_prob": 0.5, "precision": "float32",
              "checkpoint_selection": "best development AUROC; first epoch wins ties; patience 8",
              "records": {"train": len(rows["labeled_train"]), "development": len(development),
                          "calibration": len(calibration), "test": len(rows["test"])},
              "pretraining_exposure_caveat": "Released xECG sources do not report PTB-XL or MIMIC as pretraining sources. Cross-source overlap is not independently audited; our PTB-XL test cohort has already been used in earlier experiments.",
              "protocol_deviations": "Binary proxy task; 40-epoch ceiling; effective batch 64 by float32 microbatch accumulation; no augmentation or class weighting."}
    save_atomic_json(directory / "config.json", config)
    result = fit(model, optimizer, scheduler, train_loader, dev_loader, development_y,
                 directory, fingerprint, generator, args.device, args.epochs, args.patience,
                 args.effective_batch_size, resume=resume_this_budget)
    save_atomic_torch(directory / "model.pt", {"model": cpu_state(model), "fingerprint": fingerprint,
                                                    "best_epoch": result["best_epoch"]})
    # Calibration and test labels enter only after the best development epoch is fixed.
    calibration_loader = DataLoader(CachedECGs(views, index, calibration),
                                    batch_size=args.microbatch_size, num_workers=0)
    test_loader = DataLoader(CachedECGs(views, index, rows["test"]),
                             batch_size=args.microbatch_size, num_workers=0)
    evaluate_predictions("xecg_finetuned_" + budget,
                         predict(model, calibration_loader, args.device),
                         predict(model, test_loader, args.device),
                         calibration, rows["test"], directory, args.seed, args.bootstrap)
    config.update({"best_epoch": result["best_epoch"],
                   "best_development_auroc": result["best_development_auroc"],
                   "elapsed_seconds": result["elapsed_seconds"],
                   "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None})
    save_atomic_json(directory / "config.json", config)
    save_atomic_json(directory / "complete.json", {"fingerprint": fingerprint,
                                                     "sha256": completion_hashes(directory)})
    (directory / "resume.pt").unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("check", "prep", "profile", "train", "all"), default="all")
    parser.add_argument("--budget", choices=("all", "full", "ten_percent"), default="all")
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--microbatch-size", type=int, default=16)
    parser.add_argument("--effective-batch-size", type=int, default=64)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.seed != 42:
        parser.error("This fixed patient and label protocol requires --seed 42")
    if min(args.epochs, args.patience, args.microbatch_size, args.effective_batch_size,
           args.bootstrap, args.threads) < 1:
        parser.error("Epochs, patience, batch sizes, bootstrap, and threads must be positive")
    if args.effective_batch_size % args.microbatch_size:
        parser.error("Effective batch size must be divisible by microbatch size")
    if args.stage in ("profile", "train", "all") and args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(args.threads)
    selected = budgets(args.budget)
    selected_manifests = {budget: load_manifests(args.manifest_root, budget) for budget in selected}
    checkpoint_hashes = checkpoint_fingerprint(args.checkpoint_dir)
    print(json.dumps({"stage": "xecg_check", "budgets": list(selected),
                      "checkpoint_sha256": checkpoint_hashes,
                      "records": {budget: len(selected_manifests[budget][0]["labeled_train"])
                                  for budget in selected}}), flush=True)
    if args.stage == "check":
        return
    if args.stage in ("prep", "all"):
        from scripts.data.prepare_xecg import cache_xecg_views
        cache_xecg_views(args.raw_dir, manifest_dir(args.manifest_root, "full"), args.cache_dir)
    views, index, cache_hashes = cache_fingerprint(
        args.cache_dir, args.raw_dir, manifest_dir(args.manifest_root, "full"))
    for budget, (rows, development, calibration, _) in selected_manifests.items():
        for row in rows["labeled_train"] + development + calibration + rows["test"]:
            if int(row["ecg_id"]) not in index:
                raise ValueError(f"xECG FFT cache missing ECG {row['ecg_id']}")
    if args.stage == "prep":
        return
    with gpu_lock(args.device):
        if args.stage in ("profile", "all"):
            rows, development, _, _ = selected_manifests[selected[0]]
            run_profile(args, rows, development, views, index, checkpoint_hashes, cache_hashes)
        if args.stage in ("train", "all"):
            for budget in selected:
                train_budget(args, budget, *selected_manifests[budget], views, index,
                             checkpoint_hashes, cache_hashes)


if __name__ == "__main__":
    main()
