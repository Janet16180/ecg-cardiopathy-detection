#!/usr/bin/env python3
"""Four controlled xECG continuation arms, followed by the fixed 007 transfers."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ecg_experiment.xecg import load_xecg, XECGBinaryClassifier
from ecg_experiment.xecg_adaptation import (
    ARMS, AdaptationConfig, AdaptationModel, ShuffledStream, contiguous_masks,
    ema_momentum_at, learning_rate_at, representation_diagnostics,
    ssl_parameter_groups, update_ema,
)
from scripts.experiments import run_xecg_finetune as ft


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs/experiment008_vision_ssl"
DEFAULT_SSL_CACHE = ROOT / "data/processed/xecg_ssl_40k"


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def rng_state():
    numpy = np.random.get_state()
    return {"python": random.getstate(), "numpy": (numpy[0], numpy[1].tolist(), *numpy[2:]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(saved):
    random.setstate(saved["python"])
    state = saved["numpy"]
    np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))
    torch.set_rng_state(saved["torch"])
    if torch.cuda.is_available():
        if len(saved["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume CUDA device count changed")
        torch.cuda.set_rng_state_all(saved["cuda"])
    elif saved["cuda"]:
        raise ValueError("CUDA resume requested without CUDA")


def save_ssl_resume(path, fingerprint, model, optimizer, stream, mask_generator,
                    step, history, trace, elapsed):
    ft.save_atomic_torch(path, {
        "version": 1, "fingerprint": fingerprint, "step": step,
        "student": ft.cpu_state(model.student), "ema": ft.cpu_state(model.ema),
        "optimizer": optimizer.state_dict(), "sampler": stream.state_dict(),
        "mask_rng": mask_generator.get_state(), "rng": rng_state(),
        # The LR/EMA schedules are pure functions of this counter and config.
        "scheduler": {"next_update": step}, "history": history,
        "trace_sha256": trace, "elapsed_seconds": elapsed,
    })


def load_ssl_resume(path, fingerprint, model, optimizer, stream, mask_generator):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("version") != 1 or saved.get("fingerprint") != fingerprint:
        raise ValueError("SSL resume arm, sources, or protocol differ")
    if (not isinstance(saved["step"], int) or saved["step"] < 0
            or saved.get("scheduler") != {"next_update": saved["step"]}
            or len(saved["history"]) != saved["step"]
            or any(row["update"] != index + 1 for index, row in enumerate(saved["history"]))):
        raise ValueError("Malformed SSL resume step/history/scheduler")
    model.student.load_state_dict(saved["student"], strict=True)
    model.ema.load_state_dict(saved["ema"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    stream.load_state_dict(saved["sampler"])
    mask_generator.set_state(saved["mask_rng"])
    restore_rng(saved["rng"])
    return saved


def load_ssl_cache(path: Path):
    """Recheck the root-owned cache against its audited train-only identities."""
    from scripts.data.prepare_xecg_ssl import FIELDS, selected_rows

    metadata = json.loads((path / "metadata.json").read_text())
    with (path / "rows.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("SSL row columns must contain identities only, without targets")
        rows = list(reader)
    expected_rows, sources = selected_rows()
    if rows != [{key: row[key] for key in FIELDS} for row in expected_rows]:
        raise ValueError("SSL rows differ from the audited training-only pool")
    if (len(rows) != 56875 or any(row["split"] != "train" for row in rows)
            or metadata.get("all_train_only") is not True
            or metadata.get("source_sha256") != sources):
        raise ValueError("SSL cohort/source identity changed")
    if Counter(row["source"] for row in rows) != {"ptbxl": 17418, "mimic": 39457}:
        raise ValueError("Unexpected SSL source composition")
    for field, key in (("ecg_id", "ecg_ids"), ("patient_id", "patient_ids"), ("source", "sources")):
        if metadata.get(key) != [row[field] for row in rows]:
            raise ValueError(f"SSL metadata order differs for {field}")
    if len(set(metadata["ecg_ids"])) != len(rows):
        raise ValueError("Duplicate SSL ECG identity")
    checksums = {"rows_sha256": ft.sha256(path / "rows.csv"),
                 "views_sha256": ft.sha256(path / "views.npy"),
                 "raw_sha256_file_sha256": ft.sha256(path / "raw_sha256.npy")}
    if any(metadata.get(key) != value for key, value in checksums.items()):
        raise ValueError("SSL cache checksum mismatch")
    views = np.load(path / "views.npy", mmap_mode="r")
    if views.shape != (56875, 1000, 12) or views.dtype != np.float32 or metadata.get("shape") != list(views.shape):
        raise ValueError("SSL cache must contain full float32 ten-second ECGs")
    return views, rows, {**checksums, "metadata_sha256": ft.sha256(path / "metadata.json")}


def source_identity():
    files = ("ecg_experiment/xecg_adaptation.py", "scripts/experiments/run_xecg_adaptation.py",
             "ecg_experiment/xecg.py", "scripts/experiments/run_xecg_finetune.py",
             "scripts/data/prepare_xecg_ssl.py", "ecg_experiment/run.py", "ecg_experiment/evaluation.py",
             "scripts/reports/report_xecg_adaptation.py", "scripts/experiments/run_cpc_experiment.py")
    return {**{name: ft.sha256(ROOT / name) for name in files},
            "xlstm_python_tree": ft.source_tree_sha256(ROOT / "third_party/xecg-deps/xlstm")}


def make_fingerprint(args, config, cache_hashes, arm):
    protocol = {"config": config.as_dict(), "views": "two contiguous masks; eight of forty tokens; clean teachers",
                "coding_rate_batch": "actual microbatch", "selection": "final student",
                "arm": arm, "backend": "vanilla", "precision": "float32"}
    return {"experiment": 8, "arm": arm, "protocol": protocol, "protocol_sha256": json_digest(protocol),
            "sources": source_identity(), "release": ft.checkpoint_fingerprint(args.checkpoint_dir),
            "ssl_cache": cache_hashes, "device": args.device, "threads": args.threads,
            "torch_version": str(torch.__version__), "numpy_version": np.__version__}


def make_ssl_state(args, config, pool_size):
    ft.seed_all(config.seed)
    backbone = load_xecg(args.checkpoint_dir, backend="vanilla", device=args.device, drop_path_prob=0.0)
    model = AdaptationModel(backbone).to(device=args.device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(ssl_parameter_groups(model.student, config), lr=config.learning_rate)
    stream = ShuffledStream(pool_size, config.seed)
    masks = torch.Generator().manual_seed(config.seed + 10001)
    return model, optimizer, stream, masks


def tensor_batch(views, indices, device):
    batch = np.array(views[indices.tolist()], dtype=np.float32, copy=True)
    if batch.ndim != 3 or batch.shape[-1] != 12 or not np.isfinite(batch).all():
        raise ValueError("SSL input batch is malformed or nonfinite")
    return torch.from_numpy(batch).to(device)


def diagnostic_indices(rows, seed=42):
    """Fixed four PTB and four MIMIC ECGs, without consuming training RNG."""
    generator = np.random.default_rng(seed + 20001)
    selected = []
    for source in ("ptbxl", "mimic"):
        candidates = [i for i, row in enumerate(rows) if row["source"] == source]
        if len(candidates) < 4:
            raise ValueError("Diagnostics require four training records per source")
        selected.extend(generator.choice(candidates, 4, replace=False).tolist())
    return torch.tensor(selected, dtype=torch.long)


def ssl_update(model, optimizer, views, stream, mask_generator, config, arm, step,
               device, trace="0" * 64):
    """One effective batch; expansion statistics remain local to each microbatch."""
    indices = stream.take(config.effective_batch_size)
    tokens = views.shape[1] // model.student.patch_size
    masks = contiguous_masks(config.effective_batch_size, tokens, config.mask_tokens, mask_generator)
    trace_digest = hashlib.sha256(bytes.fromhex(trace))
    trace_digest.update(indices.numpy().tobytes())
    trace_digest.update(masks.numpy().tobytes())
    learning_rate = learning_rate_at(step, config)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    optimizer.zero_grad(set_to_none=True)
    model.train()
    sums = {}
    count = config.effective_batch_size // config.microbatch_size
    for start in range(0, len(indices), config.microbatch_size):
        signal = tensor_batch(views, indices[start:start + config.microbatch_size], device)
        loss, terms = model(signal, masks[start:start + config.microbatch_size].to(device), arm, config)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite adaptation objective")
        (loss / count).backward()
        for name, value in {"total": loss.detach(), **terms}.items():
            sums[name] = sums.get(name, 0.0) + float(value) / count
    norm = torch.nn.utils.clip_grad_norm_(model.student.parameters(), config.gradient_clip)
    if not torch.isfinite(norm):
        raise RuntimeError("Nonfinite adaptation gradient norm")
    optimizer.step()
    if not all(torch.isfinite(value).all() for value in model.student.parameters()):
        raise RuntimeError("Nonfinite adapted parameters")
    momentum = ema_momentum_at(step, config)
    update_ema(model.ema, model.student, momentum)
    optimizer.zero_grad(set_to_none=True)
    return {"update": step + 1, "losses": sums, "gradient_norm": float(norm),
            "learning_rate": learning_rate, "ema_momentum": momentum,
            "record_draws": (step + 1) * config.effective_batch_size,
            "student_view_draws": (step + 1) * config.effective_batch_size * 2}, trace_digest.hexdigest(), indices


def ssl_completion(directory: Path, fingerprint):
    marker = directory / "complete.json"
    if not marker.is_file():
        return None
    saved = json.loads(marker.read_text())
    if saved.get("fingerprint") != fingerprint:
        raise ValueError(f"Completed SSL arm has different inputs: {directory}")
    if saved.get("sha256") != {name: ft.sha256(directory / name) for name in ("encoder.pt", "history.json", "config.json")}:
        raise ValueError(f"Completed SSL artifact checksum mismatch: {directory}")
    return saved


def pretrain_arm(args, config, views, rows, cache_hashes, arm):
    fingerprint = make_fingerprint(args, config, cache_hashes, arm)
    directory = args.output_dir / "pretrain" / arm
    if ssl_completion(directory, fingerprint):
        print(json.dumps({"stage": "ssl_reuse", "arm": arm}), flush=True)
        return
    resume_path = directory / "resume.pt"
    if directory.exists() and any(directory.iterdir()) and not args.resume:
        raise FileExistsError(f"Existing incomplete SSL arm; use --resume: {directory}")
    if (directory / "config.json").exists():
        if json.loads((directory / "config.json").read_text())["fingerprint"] != fingerprint:
            raise ValueError("Incomplete SSL configuration differs")
    model, optimizer, stream, masks = make_ssl_state(args, config, len(views))
    history, trace, start_step = [], "0" * 64, 0
    started = time.monotonic()
    if resume_path.is_file():
        saved = load_ssl_resume(resume_path, fingerprint, model, optimizer, stream, masks)
        start_step, history, trace = saved["step"], saved["history"], saved["trace_sha256"]
        started -= saved["elapsed_seconds"]
        if start_step > config.updates:
            raise ValueError("SSL checkpoint exceeds the update budget")
    directory.mkdir(parents=True, exist_ok=True)
    ft.save_atomic_json(directory / "config.json", {"fingerprint": fingerprint,
                        "selection": "Final update student encoder", "mask_seed": config.seed + 10001,
                        "source_counts": dict(Counter(row["source"] for row in rows)),
                        "teachers": "clean EMA and frozen release; no gradients; frozen forward in every arm"})
    fixed_indices = diagnostic_indices(rows, config.seed)
    diagnostic_signal = tensor_batch(views, fixed_indices, args.device)
    np.savez(directory / "diagnostic_waveforms.npz", signals=diagnostic_signal.cpu().numpy(),
             ecg_ids=np.asarray([rows[i]["ecg_id"] for i in fixed_indices.tolist()]),
             sources=np.asarray([rows[i]["source"] for i in fixed_indices.tolist()]))
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for step in range(start_step, config.updates):
        row, trace, indices = ssl_update(model, optimizer, views, stream, masks, config, arm, step, args.device, trace)
        selected = [rows[i] for i in indices.tolist()]
        row.update({"source_counts": dict(Counter(r["source"] for r in selected)),
                    "repeated_patient_draws": len(selected) - len({(r["source"], r["patient_id"]) for r in selected}),
                    "elapsed_seconds": time.monotonic() - started})
        if (step + 1) % 100 == 0 or step + 1 == config.updates:
            row["diagnostics"] = representation_diagnostics(model, diagnostic_signal)
        history.append(row)
        if (step + 1) % 10 == 0 or step == start_step:
            print(json.dumps({"stage": "ssl_train", "arm": arm, **row}), flush=True)
        if (step + 1) % 100 == 0 or step + 1 == config.updates:
            save_ssl_resume(resume_path, fingerprint, model, optimizer, stream, masks,
                            step + 1, history, trace, time.monotonic() - started)
            ft.save_atomic_json(directory / "history.json", history)
    ft.save_atomic_torch(directory / "encoder.pt", {"student": ft.cpu_state(model.student),
                         "fingerprint": fingerprint, "updates": config.updates,
                         "selection": "final_student", "trace_sha256": trace})
    ft.save_atomic_json(directory / "history.json", history)
    receipt = {"fingerprint": fingerprint, "updates": config.updates, "trace_sha256": trace,
               "elapsed_seconds": time.monotonic() - started,
               "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
               "sha256": {name: ft.sha256(directory / name) for name in ("encoder.pt", "history.json", "config.json")}}
    ft.save_atomic_json(directory / "complete.json", receipt)
    resume_path.unlink(missing_ok=True)
    print(json.dumps({"stage": "ssl_complete", "arm": arm, "updates": config.updates,
                      "encoder_sha256": receipt["sha256"]["encoder.pt"]}), flush=True)


def require_all_ssl(args, config, cache_hashes):
    """Barrier: every arm is final and immutable before any downstream test."""
    receipts = {}
    for arm in ARMS:
        receipt = ssl_completion(args.output_dir / "pretrain" / arm,
                                 make_fingerprint(args, config, cache_hashes, arm))
        if receipt is None or receipt.get("updates") != config.updates:
            raise RuntimeError("All four final SSL arms must exist before supervised transfers")
        receipts[arm] = receipt
    if len({receipt["trace_sha256"] for receipt in receipts.values()}) != 1:
        raise ValueError("Record and mask streams differed across SSL arms")
    marker = {arm: receipt["sha256"]["encoder.pt"] for arm, receipt in receipts.items()}
    ft.save_atomic_json(args.output_dir / "all_ssl_complete.json", marker)
    return receipts


def profile(args, config, views, cache_hashes):
    """Exercise every objective and steady Adam memory without retaining weights."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results, fingerprints = {}, {}
    for arm in ARMS:
        model, optimizer, stream, masks = make_ssl_state(args, config, len(views))
        fingerprint = make_fingerprint(args, config, cache_hashes, arm)
        fingerprints[arm] = fingerprint
        trace, history, timings = "0" * 64, [], []
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for step in range(2):
            started = time.monotonic()
            row, trace, _ = ssl_update(model, optimizer, views, stream, masks, config, arm, step, args.device, trace)
            history.append(row)
            if args.device == "cuda":
                torch.cuda.synchronize()
            timings.append(time.monotonic() - started)
        path = args.output_dir / f"profile_{arm}_resume.pt"
        save_ssl_resume(path, fingerprint, model, optimizer, stream, masks, 2, history, trace, sum(timings))
        saved = load_ssl_resume(path, fingerprint, model, optimizer, stream, masks)
        if saved["step"] != 2 or saved["trace_sha256"] != trace:
            raise RuntimeError("Profile resume roundtrip failed")
        path.unlink()
        results[arm] = {"first_update_seconds": timings[0], "steady_update_seconds": timings[1],
                        "projected_ssl_arm_seconds": timings[1] * config.updates,
                        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
                        "optimizer_state_bytes": sum(value.numel() * value.element_size() for state in optimizer.state.values()
                                                     for value in state.values() if isinstance(value, torch.Tensor)),
                        "final_losses": history[-1]["losses"], "resume_roundtrip": True}
        del model, optimizer, saved
        if args.device == "cuda":
            torch.cuda.empty_cache()
        print(json.dumps({"stage": "adaptation_profile", "arm": arm, **results[arm]}), flush=True)
    baseline_seconds = []
    for budget in ("full", "ten_percent"):
        path = ft.DEFAULT_OUTPUT_DIR / f"xecg_{budget}_seed42" / "config.json"
        if path.is_file():
            elapsed = json.loads(path.read_text()).get("elapsed_seconds")
            if elapsed is not None:
                baseline_seconds.append(float(elapsed))
    projected_ssl = sum(r["projected_ssl_arm_seconds"] for r in results.values())
    projected_transfer = 4 * sum(baseline_seconds) if len(baseline_seconds) == 2 else None
    result = {"config": config.as_dict(), "cache": cache_hashes, "sources": source_identity(),
              "device": args.device, "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
              "arm_fingerprints": fingerprints,
              "arms": results, "projected_total_ssl_seconds": projected_ssl,
              "projected_supervised_seconds": projected_transfer,
              "projected_total_suite_seconds": projected_ssl + projected_transfer if projected_transfer is not None else None,
              "supervised_projection_assumption": "Four times the measured Experiment 007 two-budget training duration; early stopping may differ"}
    ft.save_atomic_json(args.output_dir / f"profile_{args.device}.json", result)
    return result


def require_profile(args, config, cache_hashes):
    path = args.output_dir / f"profile_{args.device}.json"
    if not path.is_file():
        raise RuntimeError("Run --stage profile successfully before CUDA adaptation")
    receipt = json.loads(path.read_text())
    expected = {arm: make_fingerprint(args, config, cache_hashes, arm) for arm in ARMS}
    if receipt.get("arm_fingerprints") != expected or set(receipt.get("arms", {})) != set(ARMS):
        raise ValueError("Profile does not match the current four-arm configuration/sources")
    if not all(receipt["arms"][arm].get("resume_roundtrip") is True for arm in ARMS):
        raise ValueError("Profile did not finish all resume checks")


def transfer(args, config, arm, budget, manifests, views, index, cache_hashes, ssl_receipt):
    rows, development, calibration, manifest_hashes = manifests
    directory = args.output_dir / "transfer" / arm / f"xecg_{budget}_seed42"
    encoder_path = args.output_dir / "pretrain" / arm / "encoder.pt"
    fingerprint = {"experiment": 8, "arm": arm, "budget": budget,
                   "adaptation_fingerprint": ssl_receipt["fingerprint"],
                   "adapted_encoder_sha256": ssl_receipt["sha256"]["encoder.pt"],
                   "all_ssl_checkpoint_sha256": json.loads((args.output_dir / "all_ssl_complete.json").read_text()),
                   "manifest_sha256": manifest_hashes, "cache": {k: v for k, v in cache_hashes.items() if k != "metadata"},
                   "epochs": args.epochs, "patience": args.patience, "seed": 42,
                   "microbatch_size": args.finetune_microbatch_size, "effective_batch_size": 64,
                   "bootstrap": args.bootstrap, "device": args.device, "sources": source_identity()}
    if ft.valid_completion(directory, fingerprint):
        print(json.dumps({"stage": "transfer_reuse", "arm": arm, "budget": budget}), flush=True)
        return
    resume = ft.resume_for_budget(directory, args.resume)
    if (directory / "config.json").exists() and json.loads((directory / "config.json").read_text())["fingerprint"] != fingerprint:
        raise ValueError("Transfer resume configuration differs")
    ft.seed_all(42)
    backbone = load_xecg(args.checkpoint_dir, backend="vanilla", device=args.device, drop_path_prob=0.5)
    adapted = torch.load(encoder_path, map_location="cpu", weights_only=True)
    if (adapted["fingerprint"] != ssl_receipt["fingerprint"] or adapted["updates"] != config.updates
            or adapted["selection"] != "final_student" or ft.sha256(encoder_path) != fingerprint["adapted_encoder_sha256"]):
        raise ValueError("Adapted encoder source/selection mismatch")
    backbone.load_state_dict(adapted["student"], strict=True)
    del adapted
    model = XECGBinaryClassifier(backbone).to(args.device)
    optimizer = torch.optim.AdamW(ft.layerwise_parameter_groups(model), weight_decay=0.1)
    scheduler = ft.make_scheduler(optimizer, math.ceil(len(rows["labeled_train"]) / 64), args.epochs)
    generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(ft.CachedECGs(views, index, rows["labeled_train"]),
                             batch_size=args.finetune_microbatch_size, shuffle=True, num_workers=0, generator=generator)
    dev_loader = DataLoader(ft.CachedECGs(views, index, development), batch_size=args.finetune_microbatch_size, num_workers=0)
    directory.mkdir(parents=True, exist_ok=True)
    description = {"fingerprint": fingerprint, "head": "Identity + Linear(1024,1)",
                   "supervised_policy": "Experiment 007 helpers, unchanged", "precision": "float32",
                   "selected_ssl_encoder": "final update student", "records": {"train": len(rows["labeled_train"]),
                   "development": len(development), "calibration": len(calibration), "test": len(rows["test"])} }
    ft.save_atomic_json(directory / "config.json", description)
    result = ft.fit(model, optimizer, scheduler, train_loader, dev_loader,
                    np.asarray([int(r["target"]) for r in development]), directory, fingerprint,
                    generator, args.device, args.epochs, args.patience, 64, resume=resume)
    ft.save_atomic_torch(directory / "model.pt", {"model": ft.cpu_state(model), "fingerprint": fingerprint,
                                                 "best_epoch": result["best_epoch"]})
    calibration_loader = DataLoader(ft.CachedECGs(views, index, calibration), batch_size=args.finetune_microbatch_size, num_workers=0)
    test_loader = DataLoader(ft.CachedECGs(views, index, rows["test"]), batch_size=args.finetune_microbatch_size, num_workers=0)
    ft.evaluate_predictions(f"xecg_adaptation_{arm}_{budget}",
                            ft.predict(model, calibration_loader, args.device),
                            ft.predict(model, test_loader, args.device), calibration, rows["test"],
                            directory, 42, args.bootstrap)
    description.update({key: value for key, value in result.items() if key != "history"})
    ft.save_atomic_json(directory / "config.json", description)
    ft.save_atomic_json(directory / "complete.json", {"fingerprint": fingerprint, "sha256": ft.completion_hashes(directory)})
    (directory / "resume.pt").unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("check", "profile", "pretrain", "train", "all"), default="all")
    parser.add_argument("--arm", choices=(*ARMS, "all"), default="all")
    parser.add_argument("--ssl-cache-dir", type=Path, default=DEFAULT_SSL_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint-dir", type=Path, default=ft.DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--manifest-root", type=Path, default=ft.DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--raw-dir", type=Path, default=ft.DEFAULT_RAW_DIR)
    parser.add_argument("--cache-dir", type=Path, default=ft.DEFAULT_CACHE_DIR)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--microbatch-size", type=int, default=8)
    parser.add_argument("--finetune-microbatch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.threads < 1 or min(args.microbatch_size, args.finetune_microbatch_size, args.bootstrap) < 1:
        parser.error("Threads, batch sizes and bootstrap must be positive")
    if 64 % args.microbatch_size or 64 % args.finetune_microbatch_size:
        parser.error("Both microbatch sizes must divide 64")
    if args.epochs != 40 or args.patience != 8:
        parser.error("Experiment 008 fixes the Experiment 007 policy at 40 epochs/patience 8")
    if args.stage != "check" and args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(args.threads)
    config = AdaptationConfig(microbatch_size=args.microbatch_size)
    selected = ARMS if args.arm == "all" else (args.arm,)
    manifests = {budget: ft.load_manifests(args.manifest_root, budget) for budget in ("full", "ten_percent")}
    ssl_views, ssl_rows, ssl_hashes = load_ssl_cache(args.ssl_cache_dir)
    print(json.dumps({"stage": "adaptation_check", "records": len(ssl_rows),
                      "arms": selected, "protocol": config.as_dict()}), flush=True)
    if args.stage == "check":
        return
    with ft.gpu_lock(args.device):
        if args.stage == "profile":
            profile(args, config, ssl_views, ssl_hashes)
            return
        if args.stage in ("pretrain", "all"):
            if args.device == "cuda":
                require_profile(args, config, ssl_hashes)
            for arm in selected:
                pretrain_arm(args, config, ssl_views, ssl_rows, ssl_hashes, arm)
        if args.stage in ("train", "all"):
            receipts = require_all_ssl(args, config, ssl_hashes)
            views, index, cache_hashes = ft.cache_fingerprint(args.cache_dir, args.raw_dir,
                                                            ft.manifest_dir(args.manifest_root, "full"))
            for arm in selected:
                for budget in ("full", "ten_percent"):
                    transfer(args, config, arm, budget, manifests[budget], views, index, cache_hashes, receipts[arm])
            if all((args.output_dir / "transfer" / arm / f"xecg_{budget}_seed42" / "complete.json").is_file()
                   for arm in ARMS for budget in ("full", "ten_percent")):
                from scripts.reports.report_xecg_adaptation import report
                report(args.output_dir, args.bootstrap)
                ft.save_atomic_json(args.output_dir / "complete.json", {
                    "all_ssl_sha256": ft.sha256(args.output_dir / "all_ssl_complete.json"),
                    "report_sha256": ft.sha256(args.output_dir / "report.md"),
                    "paired_comparisons_sha256": ft.sha256(args.output_dir / "paired_comparisons.json")})


if __name__ == "__main__":
    main()
