"""
Experiment 064 runner: beat pool, GPU profile, SimDINOv2 + iBOT pretraining, kNN maps.

Usage::

    PYTHONPATH=. python -m scripts.experiments.run_dino_knn064 pool
    PYTHONPATH=. python -m scripts.experiments.run_dino_knn064 profile
    PYTHONPATH=. python -m scripts.experiments.run_dino_knn064 pretrain --hours 2
    PYTHONPATH=. python -m scripts.experiments.run_dino_knn064 maps
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ecg_experiment import dino_knn064 as dk
from ecg_experiment.ecgad import load_split
from ecg_experiment.files import sha256_file
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.locbench_cache import open_cache
from ecg_experiment.locbench_data import load_partition
from ecg_experiment.paths import to_stored

BEAT_DIR = Path(os.environ.get("ECG064_BEATS", "/tmp/ecg064-beats"))
OUTPUT = Path("outputs/experiment064_dino_knn_v1")
THREADS = 8
SEED = 64064
BATCH = 256
PEAK_LR = 5e-4
WEIGHT_DECAY = 0.05
CLIP_NORM = 3.0
WARMUP_SHARE = 0.05
CHECKPOINT_SECONDS = 900
MIN_CAP_GB = 4.0


def _pool_record(row: tuple[str, str, str, str]) -> tuple[str, np.ndarray | None]:
    record_id, backend, path, index = row
    signal = dk.read_pool_signal(backend, path, index)
    if dk.has_flat_lead(signal):
        return record_id, None
    return record_id, dk.pool_windows(dk.preprocess(signal), dk.peaks_250(signal)).astype(np.float16)


def build_pool() -> None:
    """Write the pretraining beat cache to local disk."""
    rows, counts = dk.pool_rows()
    rows = rows.sort_values(["path", "index"]).reset_index(drop=True)
    BEAT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    windows, table = [], []
    items = list(rows[["record_id", "backend", "path", "index"]].itertuples(index=False, name=None))
    with Pool(THREADS) as pool:
        for position, (record_id, beats) in enumerate(pool.imap(_pool_record, items, chunksize=32)):
            table.append(
                {
                    "record_id": record_id,
                    "flat_lead": beats is None,
                    "beats": 0 if beats is None else len(beats),
                }
            )
            if beats is not None and len(beats):
                windows.append(beats)
            if position % 5000 == 0:
                print(f"{position}/{len(items)} records, {time.time() - started:.0f} s", flush=True)
    beats = np.concatenate(windows)
    np.save(BEAT_DIR / "beats.npy", beats)
    table = pd.DataFrame(table).merge(rows[["record_id", "source"]], on="record_id")
    table.to_csv(BEAT_DIR / "records.csv", index=False)
    summary = {
        "manifest": to_stored(dk.ROOT / dk.POOL_MANIFEST),
        "manifest_sha256": sha256_file(dk.ROOT / dk.POOL_MANIFEST),
        "selection_counts": counts,
        "records_flat_lead": int(table.flat_lead.sum()),
        "records_without_beats": int(((table.beats == 0) & ~table.flat_lead).sum()),
        "records_used": int((table.beats > 0).sum()),
        "beats": len(beats),
        "beats_per_source": table.groupby("source").beats.sum().astype(int).to_dict(),
        "records_per_source": table[table.beats > 0].groupby("source").size().astype(int).to_dict(),
        "amplitude_mv_percentiles": np.percentile(
            np.abs(beats[:: max(1, len(beats) // 2000)].astype(np.float32)), [50, 99, 99.99]
        ).tolist(),
        "seconds": time.time() - started,
    }
    (BEAT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def pick_device(cap_gb: float) -> torch.device:
    """Choose the GPU with the most free memory and cap this process at ``cap_gb`` or free memory - 0.7 GB."""
    free = [torch.cuda.mem_get_info(i)[0] / 2**30 for i in range(torch.cuda.device_count())]
    index = int(np.argmax(free))
    cap = min(cap_gb, free[index] - 0.7)
    if cap < MIN_CAP_GB:
        raise RuntimeError(f"GPU {index} has only {free[index]:.1f} GB free")
    total = torch.cuda.get_device_properties(index).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(cap / total, index)
    print(f"GPU {index}: {free[index]:.1f} GB free, cap {cap:.1f} GB", flush=True)
    return torch.device(f"cuda:{index}")


def new_models(device: torch.device) -> tuple[dk.Network, dk.Network]:
    """Seeded student and its frozen EMA teacher copy."""
    torch.manual_seed(SEED)
    student = dk.Network().to(device)
    student.backbone.checkpoint = True
    teacher = copy.deepcopy(student)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return student, teacher


def new_optimizer(student: dk.Network) -> torch.optim.Optimizer:
    """AdamW with weight decay on matrices only."""
    decay = [p for p in student.parameters() if p.ndim >= 2]
    rest = [p for p in student.parameters() if p.ndim < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": WEIGHT_DECAY}, {"params": rest, "weight_decay": 0.0}], lr=PEAK_LR
    )


def train_steps(
    beats: np.ndarray,
    student: dk.Network,
    teacher: dk.Network,
    optimizer: torch.optim.Optimizer,
    first: int,
    last: int,
    total: int,
    device: torch.device,
    log: Path | None,
    checkpoint: Path | None,
    batch_size: int = BATCH,
) -> list[dict[str, float]]:
    """Run optimizer steps ``first`` to ``last - 1`` of a ``total``-step schedule."""
    loss_module = dk.new_loss_module()
    generator = torch.Generator(device=device).manual_seed(SEED + first)
    order = np.random.default_rng(SEED + first)
    warmup = max(1, int(WARMUP_SHARE * total))
    records, saved = [], time.time()
    student.train()
    for step in range(first, last):
        index = np.sort(order.integers(0, len(beats), batch_size))
        batch = torch.from_numpy(beats[index]).to(device, non_blocking=True)
        for group in optimizer.param_groups:
            group["lr"] = dk.learning_rate(step, total, PEAK_LR, warmup)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            parts = dk.simdino_loss(student, teacher, batch, generator, loss_module)
        optimizer.zero_grad(set_to_none=True)
        parts["total"].backward()
        norm = torch.nn.utils.clip_grad_norm_(student.parameters(), CLIP_NORM)
        optimizer.step()
        dk.ema_update(teacher, student, dk.ema_momentum(step + 1, total))
        if step % 100 == 0 or step == last - 1:
            row = {"step": step, "lr": optimizer.param_groups[0]["lr"], "grad_norm": float(norm)}
            row |= {key: float(value) for key, value in parts.items()}
            records.append(row)
            if not np.isfinite(row["total"]):
                raise FloatingPointError(f"Nonfinite loss at step {step}")
            if log is not None:
                with log.open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
        if checkpoint is not None and time.time() - saved > CHECKPOINT_SECONDS:
            save_state(checkpoint, student, teacher, optimizer, step + 1, total, batch_size)
            saved = time.time()
    return records


def save_state(
    path: Path,
    student: dk.Network,
    teacher: dk.Network,
    optimizer: torch.optim.Optimizer,
    step: int,
    total: int,
    batch_size: int,
) -> None:
    """Atomic checkpoint of both networks, the optimizer, the step counter and the batch size."""
    state = {
        "student": student.state_dict(),
        "teacher": teacher.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "total": total,
        "batch": batch_size,
    }
    torch.save(state, path.with_suffix(".part"))
    os.replace(path.with_suffix(".part"), path)


def profile_batch(beats: np.ndarray, device: torch.device, batch_size: int) -> dict[str, object]:
    """Time 50 updates after 10 warm-up updates at one batch size."""
    student, teacher = new_models(device)
    optimizer = new_optimizer(student)
    torch.cuda.reset_peak_memory_stats(device)
    train_steps(beats, student, teacher, optimizer, 0, 10, 10000, device, None, None, batch_size)
    torch.cuda.synchronize(device)
    started = time.time()
    records = train_steps(beats, student, teacher, optimizer, 10, 60, 10000, device, None, None, batch_size)
    torch.cuda.synchronize(device)
    seconds = (time.time() - started) / 50
    return {
        "batch": batch_size,
        "seconds_per_update": seconds,
        "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 2**30,
        "updates_in_2h": int(7200 / seconds),
        "epochs_in_2h": int(7200 / seconds) * batch_size / len(beats),
        "last_losses": records[-1],
    }


def profile(cap_gb: float) -> None:
    """Real GPU profile at batch 256 and, if that does not fit the cap, at batch 128."""
    beats = np.load(BEAT_DIR / "beats.npy", mmap_mode="r")
    result: dict[str, object] = {"beats_in_pool": len(beats), "attempts": []}
    with gpu_lock("cuda"):
        device = pick_device(cap_gb)
        result["device"] = torch.cuda.get_device_name(device)
        result["cap_gb"] = torch.cuda.get_per_process_memory_fraction(device) * (
            torch.cuda.get_device_properties(device).total_memory / 2**30
        )
        for batch_size in (BATCH, BATCH // 2):
            try:
                attempt = profile_batch(beats, device, batch_size)
            except torch.OutOfMemoryError:
                result["attempts"].append({"batch": batch_size, "out_of_memory": True})
                torch.cuda.empty_cache()
                continue
            result["attempts"].append(attempt)
            break
    student = dk.Network()
    result["student_parameters"] = sum(p.numel() for p in student.parameters())
    result["backbone_parameters"] = sum(p.numel() for p in student.backbone.parameters())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "gpu_profile.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def pretrain(steps: int, batch_size: int, cap_gb: float) -> None:
    """Full pretraining run (resumes from the local checkpoint if present)."""
    beats = np.load(BEAT_DIR / "beats.npy")
    checkpoint = BEAT_DIR / "checkpoint.pt"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    log = OUTPUT / "pretrain_log.jsonl"
    with gpu_lock("cuda"):
        device = pick_device(cap_gb)
        student, teacher = new_models(device)
        optimizer = new_optimizer(student)
        first = 0
        if checkpoint.exists():
            state = torch.load(checkpoint, map_location=device)
            if state["total"] != steps or state["batch"] != batch_size:
                raise ValueError("Checkpoint belongs to another schedule")
            student.load_state_dict(state["student"])
            teacher.load_state_dict(state["teacher"])
            optimizer.load_state_dict(state["optimizer"])
            first = state["step"]
        started = time.time()
        train_steps(
            beats, student, teacher, optimizer, first, steps, steps, device, log, checkpoint, batch_size
        )
        save_state(checkpoint, student, teacher, optimizer, steps, steps, batch_size)
    torch.save(
        {"teacher": teacher.state_dict(), "student": student.state_dict(), "steps": steps},
        OUTPUT / "model.pt",
    )
    summary = {
        "steps": steps,
        "resumed_from": first,
        "seconds_this_session": time.time() - started,
        "beats": len(beats),
        "batch": batch_size,
        "seed": SEED,
    }
    (OUTPUT / "pretrain_summary.json").write_text(json.dumps(summary, indent=2))


NEIGHBOURS = 3
REACH = 1
CALIBRATION_SHARE = 0.2
FEATURE_BATCH = 1024
TARGET_CHUNK = 200
VARIANTS = ("dino_z", "dino_dist", "dino_z_rr", "raw_z", "raw_dist")
_CACHE: dict[str, object] = {}


def _cached_beats(position: int) -> dk.Beats:
    if "signals" not in _CACHE:
        _CACHE["signals"], _ = open_cache()
    return dk.score_beats(np.asarray(_CACHE["signals"][position], dtype=np.float32))


def record_beats(positions: list[int], pool: Pool) -> list[dk.Beats]:
    """Scoring windows of cache rows, computed on CPU workers."""
    return pool.map(_cached_beats, positions, chunksize=8)


def stack(beats: list[dk.Beats]) -> dk.Beats:
    """Concatenate the beats of several records."""
    return dk.Beats(
        np.concatenate([b.windows for b in beats]),
        np.concatenate([b.valid for b in beats]),
        np.concatenate([b.starts for b in beats]),
        np.concatenate([b.prematurity for b in beats]),
    )


def features(backbone: dk.BeatViT | None, beats: dk.Beats, device: torch.device) -> torch.Tensor:
    """Patch descriptors on the GPU, computed in batches."""
    parts = []
    for begin in range(0, len(beats.windows), FEATURE_BATCH):
        part = dk.Beats(
            *(getattr(beats, f)[begin : begin + FEATURE_BATCH] for f in dk.Beats.__dataclass_fields__)
        )
        parts.append(dk.patch_features(backbone, part, device))
    return torch.cat(parts)


def build_bank(backbone: dk.BeatViT | None, beats: dk.Beats, device: torch.device) -> torch.Tensor:
    """Every patch of every bank beat whose 24 patches all lie inside the record, as ``[12, 24, n, dim]``."""
    full = beats.valid.all(axis=1)
    kept = dk.Beats(beats.windows[full], beats.valid[full], beats.starts[full], beats.prematurity[full])
    parts = []
    for begin in range(0, len(kept.windows), FEATURE_BATCH):
        part = dk.Beats(
            *(getattr(kept, f)[begin : begin + FEATURE_BATCH] for f in dk.Beats.__dataclass_fields__)
        )
        parts.append(dk.patch_features(backbone, part, device).permute(1, 2, 0, 3))
    return torch.cat(parts, dim=2).contiguous()


def calibration(distance: torch.Tensor, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of calibration-normal distances per lead and phase bin."""
    d = distance.cpu().numpy()
    mask = np.broadcast_to(valid[:, None, :], d.shape)
    masked = np.where(mask, d, np.nan)
    return np.nanmean(masked, axis=0), np.nanstd(masked, axis=0)


def split_normals(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit clean normals split by patient into bank (80 percent) and calibration (20 percent) records."""
    partition = load_partition()
    normals = partition[(partition.partition == "fit") & partition.clean_normal]
    patients = np.unique(normals.patient_id)
    rng = np.random.default_rng(seed)
    held = set(rng.choice(patients, int(round(CALIBRATION_SHARE * len(patients))), replace=False))
    calibration_rows = normals[normals.patient_id.isin(held)]
    return normals[~normals.patient_id.isin(held)], calibration_rows


def target_ids() -> np.ndarray:
    """ECGAD development records followed by every LocBench ``evaluation`` record."""
    split = load_split()
    partition = load_partition()
    development = split.ecg_id[split.partition == "development"].astype(int).to_numpy()
    evaluation = partition.ecg_id[partition.partition == "evaluation"].astype(int).to_numpy()
    return np.concatenate([development, evaluation])


def load_backbone(model_path: Path | None, device: torch.device) -> dk.BeatViT:
    """Pretrained teacher backbone, or the untrained seeded network if no model is given."""
    _, teacher = new_models(device)
    if model_path is not None:
        teacher.load_state_dict(torch.load(model_path, map_location=device)["teacher"])
    return teacher.backbone.eval()


def fit_reference(
    backbone: dk.BeatViT, bank: dk.Beats, held: dk.Beats, device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, tuple]]:
    """Patch banks from bank normals and distance statistics from calibration normals."""
    banks, stats = {}, {}
    for name, net, cosine in (("dino", backbone, True), ("raw", None, False)):
        banks[name] = build_bank(net, bank, device)
        distance = dk.knn_distance(features(net, held, device), banks[name], NEIGHBOURS, REACH, cosine)
        stats[name] = calibration(distance, held.valid)
    stats["prematurity"] = (float(held.prematurity.mean()), float(held.prematurity.std()))
    return banks, stats


def chunk_scores(
    backbone: dk.BeatViT,
    beats: dk.Beats,
    banks: dict[str, torch.Tensor],
    stats: dict[str, tuple],
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Patch scores ``[beats, 12, 24]`` of every variant."""
    scores = {}
    for name, net, cosine in (("dino", backbone, True), ("raw", None, False)):
        distance = dk.knn_distance(features(net, beats, device), banks[name], NEIGHBOURS, REACH, cosine)
        distance = distance.cpu().numpy()
        mean, std = stats[name]
        scores[f"{name}_dist"] = distance
        scores[f"{name}_z"] = (distance - mean) / std
    mean_p, std_p = stats["prematurity"]
    rr = np.maximum(0.0, (beats.prematurity - mean_p) / std_p).astype(np.float32)
    scores["dino_z_rr"] = scores["dino_z"] + rr[:, None, None]
    return scores


def make_maps(model_path: Path | None, out: Path, limit: int | None, cap_gb: float) -> None:
    """Bank, calibration and float16 maps for every target record and variant."""
    _, index = open_cache()
    position = dict(zip(index.ecg_id.astype(int), range(len(index)), strict=True))
    bank_rows, calibration_rows = split_normals(SEED)
    targets = target_ids()
    if limit is not None:
        bank_rows, calibration_rows, targets = (
            bank_rows[:limit],
            calibration_rows[: limit // 4],
            targets[:limit],
        )
    if set(targets) & (set(bank_rows.ecg_id) | set(calibration_rows.ecg_id)):
        raise ValueError("Target record inside the normal bank")
    out.mkdir(parents=True, exist_ok=True)
    local = BEAT_DIR / ("maps_" + out.name)
    local.mkdir(parents=True, exist_ok=True)
    started = time.time()
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    shape = (len(targets), dk.LEADS, 5000)
    maps = {v: np.lib.format.open_memmap(local / f"maps_{v}.npy", "w+", np.float16, shape) for v in VARIANTS}
    beat_counts = np.zeros(len(targets), int)
    with Pool(THREADS) as pool, gpu_lock("cuda"):
        device = pick_device(cap_gb)
        backbone = load_backbone(model_path, device)
        bank = stack(record_beats([position[i] for i in bank_rows.ecg_id], pool))
        held = stack(record_beats([position[i] for i in calibration_rows.ecg_id], pool))
        banks, stats = fit_reference(backbone, bank, held, device)
        del bank, held
        print(f"Bank and calibration ready in {time.time() - started:.0f} s", flush=True)
        for begin in range(0, len(targets), TARGET_CHUNK):
            per_record = record_beats([position[i] for i in targets[begin : begin + TARGET_CHUNK]], pool)
            scores = chunk_scores(backbone, stack(per_record), banks, stats, device)
            offsets = np.cumsum([0] + [len(b.windows) for b in per_record])
            for j, record in enumerate(per_record):
                beat_counts[begin + j] = len(record.windows)
                for variant in VARIANTS:
                    part = scores[variant][offsets[j] : offsets[j + 1]]
                    maps[variant][begin + j] = dk.signed_log1p(dk.beat_map(part, record)).astype(np.float16)
            print(
                f"{begin + len(per_record)}/{len(targets)} records, {time.time() - started:.0f} s", flush=True
            )
    for variant in VARIANTS:
        maps[variant].flush()
        shutil.move(local / f"maps_{variant}.npy", out / f"maps_{variant}.npy")
    np.save(out / "ecg_ids.npy", targets)
    np.save(out / "beat_counts.npy", beat_counts)
    np.savez(
        out / "calibration.npz",
        **{f"{k}_{s}": v[i] for k, v in stats.items() for i, s in enumerate(("mean", "std"))},
    )
    receipt = {
        "model": None if model_path is None else to_stored(model_path),
        "model_sha256": None if model_path is None else sha256_file(model_path),
        "bank_records": len(bank_rows),
        "calibration_records": len(calibration_rows),
        "bank_beats": int(banks["dino"].shape[2]),
        "neighbours": NEIGHBOURS,
        "reach": REACH,
        "targets": len(targets),
        "beats_per_record": np.percentile(beat_counts, [0, 50, 100]).tolist(),
        "records_without_beats": int((beat_counts == 0).sum()),
        "prematurity_calibration": stats["prematurity"],
        "variants": list(VARIANTS),
        "seconds": time.time() - started,
    }
    (out / "maps_receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("pool")
    sub.add_parser("profile").add_argument("--cap-gb", type=float, default=9.0)
    run = sub.add_parser("pretrain")
    run.add_argument("--steps", type=int, required=True)
    run.add_argument("--batch", type=int, default=BATCH)
    run.add_argument("--cap-gb", type=float, default=9.0)
    maps = sub.add_parser("maps")
    maps.add_argument("--model", type=Path)
    maps.add_argument("--out", type=Path, default=OUTPUT)
    maps.add_argument("--limit", type=int)
    maps.add_argument("--cap-gb", type=float, default=9.0)
    args = parser.parse_args()
    torch.set_num_threads(THREADS)
    if args.command == "pool":
        build_pool()
    elif args.command == "profile":
        profile(args.cap_gb)
    elif args.command == "pretrain":
        pretrain(args.steps, args.batch, args.cap_gb)
    elif args.command == "maps":
        make_maps(args.model, args.out, args.limit, args.cap_gb)


if __name__ == "__main__":
    main()
