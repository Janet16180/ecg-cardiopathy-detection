"""
Experiment 063 runner: masked cross-lead and cross-beat reconstruction trained on clean normals.

Subcommands: ``prepare`` (training arrays on local disk), ``profile`` (GPU profile), ``train``,
``maps`` (LocBench maps for a record set), ``calibration`` (z-score calibration on held-out fit normals).
See ``docs/experiment-063-masked-reconstruction.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ecg_experiment import ROOT
from ecg_experiment.ecgad import load_split
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.inpaint063 import (
    CROP,
    FAMILIES,
    VARIANTS,
    InpaintNet,
    beta_nll,
    electrode_noise,
    gaussian_nll,
    pad_length,
    peaks_250,
    preprocess,
    record_errors,
    train_mask,
    variant_maps,
)
from ecg_experiment.locbench_cache import open_cache
from ecg_experiment.locbench_data import load_partition
from ecg_experiment.paths import to_stored

OUT = ROOT / "outputs/experiment063_inpaint_v1"
LOCAL = Path(os.environ.get("ECG_063_LOCAL", "/tmp/ecg-063-local"))
SEED = 63063
VAL_SHARE = 0.1
BATCH = 64
NOISE_PROBABILITY = 0.6
THREADS = 8


def log(message: str) -> None:
    """
    Print a timestamped line and append it to the run log.

    Parameters
    ----------
    message : str
        Text to record.
    """
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "run.log").open("a") as handle:
        handle.write(line + "\n")


def cache_rows(ecg_ids: np.ndarray) -> np.ndarray:
    """
    Row positions of PTB-XL records in the LocBench cache.

    Parameters
    ----------
    ecg_ids : np.ndarray
        PTB-XL record IDs.

    Returns
    -------
    np.ndarray
        Cache row of each record, in input order.
    """
    _, index = open_cache()
    return pd.Series(np.arange(len(index)), index=index.ecg_id.to_numpy()).loc[ecg_ids].to_numpy()


def load_records(ecg_ids: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Preprocessed 250 Hz signals and frozen R peaks of a record list.

    Parameters
    ----------
    ecg_ids : np.ndarray
        PTB-XL record IDs.

    Returns
    -------
    tuple[np.ndarray, list[np.ndarray]]
        Signals (n, 12, 2500) and the R peaks of each record at 250 Hz.
    """
    signals, _ = open_cache()
    rows = cache_rows(ecg_ids)
    order = np.argsort(rows)
    raw = np.empty((len(rows), 12, signals.shape[2]), dtype=np.float32)
    raw[order] = signals[rows[order]]
    with ThreadPoolExecutor(THREADS) as pool:
        peaks = list(pool.map(peaks_250, raw))
        parts = list(pool.map(preprocess, np.array_split(raw, max(1, len(raw) // 64))))
    return np.concatenate(parts), peaks


def training_split() -> pd.DataFrame:
    """
    Fit-partition clean normals with a patient-held-out validation flag.

    Returns
    -------
    pd.DataFrame
        ``ecg_id``, ``patient_id`` and ``val`` (10% of patients, seed 63063).
    """
    partition = load_partition()
    table = partition[(partition.partition == "fit") & partition.clean_normal][["ecg_id", "patient_id"]]
    table = table.sort_values("ecg_id").reset_index(drop=True)
    patients = np.unique(table.patient_id.to_numpy())
    rng = np.random.default_rng(SEED)
    held = set(rng.choice(patients, int(round(VAL_SHARE * len(patients))), replace=False))
    table["val"] = table.patient_id.isin(held)
    ecgad = set(load_split().ecg_id)
    if table.ecg_id.isin(ecgad).any():
        raise ValueError("ECGAD record in the training set")
    return table


def prepare() -> None:
    """Write the preprocessed training and validation arrays to local disk."""
    LOCAL.mkdir(parents=True, exist_ok=True)
    table = training_split()
    signals, peaks = load_records(table.ecg_id.to_numpy())
    lengths = np.array([len(p) for p in peaks])
    packed = np.full((len(peaks), lengths.max()), -10_000, dtype=np.int64)
    for k, p in enumerate(peaks):
        packed[k, : len(p)] = p
    np.savez(
        LOCAL / "train.npz", signals=signals, peaks=packed, val=table.val.to_numpy(), ecg_id=table.ecg_id
    )
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "training_split.csv", index=False)
    spread = np.percentile(lengths, [0, 50, 100])
    log(
        f"prepare: {len(table)} records, {int(table.val.sum())} validation records from "
        f"{table[table.val].patient_id.nunique()} patients; peaks per record {spread}"
    )


def choose_device(cap_gb: float) -> str:
    """
    Pick the GPU with the most free memory and cap this process's share of it.

    Parameters
    ----------
    cap_gb : float
        Memory cap in GB.

    Returns
    -------
    str
        ``"cuda"``, with the chosen GPU set as the current device.
    """
    free = [torch.cuda.mem_get_info(k) for k in range(torch.cuda.device_count())]
    best = int(np.argmax([f for f, _ in free]))
    torch.cuda.set_device(best)
    available, total = free[best]
    cap = min(cap_gb * 2**30, available - 2**30)
    if cap < 2**30:
        raise RuntimeError(f"Not enough free GPU memory: {available / 2**30:.1f} GB on GPU {best}")
    torch.cuda.set_per_process_memory_fraction(cap / total, best)
    log(f"device cuda:{best}, free {available / 2**30:.1f} GB, cap {cap / 2**30:.1f} GB")
    return "cuda"


def load_training() -> dict[str, np.ndarray]:
    """
    Arrays written by ``prepare``.

    Returns
    -------
    dict[str, np.ndarray]
        ``signals``, ``peaks`` (padded with -10000), ``val``, ``ecg_id``.
    """
    with np.load(LOCAL / "train.npz") as data:
        return {key: data[key] for key in data.files}


def make_batch(
    signals: np.ndarray, peaks: np.ndarray, indices: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """
    One augmented training batch: random crop, amplitude scale, electrode noise, and a random mask.

    Parameters
    ----------
    signals : np.ndarray
        Training signals (n, 12, 2500).
    peaks : np.ndarray
        Padded R peaks (n, K).
    indices : np.ndarray
        Records of the batch.
    rng : np.random.Generator
        Random source.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Signals (B, 12, CROP) float32 and masks (B, 12, CROP) bool.
    """
    starts = rng.integers(0, signals.shape[2] - CROP + 1, len(indices))
    x = np.stack([signals[i, :, s : s + CROP] for i, s in zip(indices, starts, strict=True)])
    x = x * np.exp(rng.uniform(math.log(0.8), math.log(1.25), (len(indices), 1, 1))).astype(np.float32)
    noisy = rng.random(len(indices)) < NOISE_PROBABILITY
    if noisy.any():
        x[noisy] += electrode_noise(int(noisy.sum()), CROP, rng)
    masks = np.stack(
        [
            train_mask(peaks[i][peaks[i] > -10_000] - s, CROP, rng)
            for i, s in zip(indices, starts, strict=True)
        ]
    )
    return x, masks


def validation_set(data: dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build fixed validation inputs: full held-out records with two seeded training-style masks each, no noise.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Output of ``load_training``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Signals (V, 12, padded) and masks (V, 12, padded), on CPU.
    """
    rng = np.random.default_rng(SEED + 1)
    rows = np.flatnonzero(data["val"])
    length = data["signals"].shape[2]
    padded = pad_length(length)
    xs, ms = [], []
    for _ in range(2):
        for i in rows:
            x = np.pad(data["signals"][i], ((0, 0), (0, padded - length)), mode="reflect")
            m = np.zeros((12, padded), dtype=bool)
            m[:, :length] = train_mask(data["peaks"][i][data["peaks"][i] > -10_000], length, rng)
            xs.append(x)
            ms.append(m)
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ms))


@torch.no_grad()
def evaluate(model: InpaintNet, xs: torch.Tensor, ms: torch.Tensor, device: str) -> dict[str, float]:
    """
    Compute validation Gaussian NLL and mean squared error over hidden samples.

    Parameters
    ----------
    model : InpaintNet
        Model.
    xs, ms : torch.Tensor
        Output of ``validation_set``.
    device : str
        Torch device.

    Returns
    -------
    dict[str, float]
        ``nll``, ``mse`` and ``mean_z2`` (mean squared z, 1 when calibrated).
    """
    model.eval()
    totals = np.zeros(4)
    for start in range(0, len(xs), BATCH):
        x, m = xs[start : start + BATCH].to(device), ms[start : start + BATCH].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mean, logvar = model(x, m)
        mean, logvar = mean.float(), logvar.float()
        n = float(m.sum())
        totals += [
            float(gaussian_nll(x, mean, logvar, m)) * n,
            float(((x - mean) ** 2)[m].mean()) * n,
            float(((x - mean) ** 2 * torch.exp(-logvar))[m].mean()) * n,
            n,
        ]
    model.train()
    return {"nll": totals[0] / totals[3], "mse": totals[1] / totals[3], "mean_z2": totals[2] / totals[3]}


def lr_at(step: int, total: int, peak: float, warmup: int = 500) -> float:
    """
    Linear warm-up then cosine decay to zero.

    Parameters
    ----------
    step, total : int
        Current and total steps.
    peak : float
        Peak learning rate.
    warmup : int
        Warm-up steps.

    Returns
    -------
    float
        Learning rate.
    """
    if step < warmup:
        return peak * (step + 1) / warmup
    return peak * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def setup(cap_gb: float) -> tuple[str, InpaintNet, torch.optim.Optimizer]:
    """
    Choose the GPU and build the model and optimizer.

    Parameters
    ----------
    cap_gb : float
        GPU memory cap.

    Returns
    -------
    tuple[str, InpaintNet, torch.optim.Optimizer]
        Device, model and AdamW optimizer.
    """
    device = choose_device(cap_gb)
    model = InpaintNet().to(device)
    return device, model, torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)


def update(
    model: InpaintNet,
    optimizer: torch.optim.Optimizer,
    batch: tuple[np.ndarray, np.ndarray],
    lr: float,
    device: str,
) -> float:
    """
    Take one optimizer step on the beta-NLL of hidden samples.

    Parameters
    ----------
    model : InpaintNet
        Model in train mode.
    optimizer : torch.optim.Optimizer
        Optimizer.
    batch : tuple[np.ndarray, np.ndarray]
        Signals and masks from ``make_batch``.
    lr : float
        Learning rate for this step.
    device : str
        Torch device.

    Returns
    -------
    float
        Loss value.
    """
    x = torch.from_numpy(batch[0]).to(device, non_blocking=True)
    m = torch.from_numpy(batch[1]).to(device)
    for group in optimizer.param_groups:
        group["lr"] = lr
    with torch.autocast("cuda", dtype=torch.bfloat16):
        mean, logvar = model(x, m)
    loss = beta_nll(x, mean.float(), logvar.float(), m)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss)


def batches(data: dict[str, np.ndarray], steps: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Training batches built one step ahead in a background thread.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Output of ``load_training``.
    steps : int
        Number of batches.

    Yields
    ------
    tuple[np.ndarray, np.ndarray]
        Signals and masks of one batch.
    """
    rows = np.flatnonzero(~data["val"])
    rng = np.random.default_rng(SEED)
    with ThreadPoolExecutor(1) as pool:

        def submit(step: int) -> Future:
            generator = np.random.default_rng([SEED, step])
            return pool.submit(make_batch, data["signals"], data["peaks"], rng.choice(rows, BATCH), generator)

        future = submit(0)
        for step in range(steps):
            batch = future.result()
            if step + 1 < steps:
                future = submit(step + 1)
            yield batch


def profile(steps: int, cap_gb: float) -> None:
    """
    Time training updates and map inference on the GPU and write ``profile.json``.

    Parameters
    ----------
    steps : int
        Updates to time (the first ten are warm-up).
    cap_gb : float
        GPU memory cap.
    """
    torch.set_num_threads(THREADS)
    torch.manual_seed(SEED)
    data = load_training()
    with gpu_lock("cuda"):
        device, model, optimizer = setup(cap_gb)
        torch.cuda.reset_peak_memory_stats()
        for step, batch in enumerate(batches(data, steps)):
            if step == 10:
                torch.cuda.synchronize()
                warm = time.time()
            update(model, optimizer, batch, 1e-4, device)
        torch.cuda.synchronize()
        per_step = (time.time() - warm) / (steps - 10)
        model.eval()
        signals, peaks = data["signals"][:5], [p[p > -10_000] for p in data["peaks"][:5]]
        record_errors(model, signals[0], peaks[0], device)
        torch.cuda.synchronize()
        started = time.time()
        for signal, peak in zip(signals[1:], peaks[1:], strict=True):
            variant_maps(record_errors(model, signal, peak, device))
        per_record = (time.time() - started) / 4
        result = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "batch": BATCH,
            "crop": CROP,
            "seconds_per_update": per_step,
            "peak_memory_gb": torch.cuda.max_memory_allocated() / 2**30,
            "inference_seconds_per_record": per_record,
            "projected_minutes_per_10k_updates": per_step * 10_000 / 60,
            "projected_minutes_maps_6200_records": per_record * 6200 / 60,
            "gpu": torch.cuda.get_device_name(),
        }
    (OUT / "profile.json").write_text(json.dumps(result, indent=2))
    log(f"profile: {json.dumps(result)}")


def train(steps: int, eval_every: int, patience: int, tag: str, cap_gb: float) -> None:
    """
    Train the model on fit clean normals with early stopping on the held-out patients.

    Parameters
    ----------
    steps : int
        Maximum updates (also sets the cosine schedule).
    eval_every : int
        Updates between validation passes.
    patience : int
        Validation passes without improvement before stopping.
    tag : str
        Output name: ``model_<tag>.pt`` and ``train_<tag>.json``.
    cap_gb : float
        GPU memory cap.
    """
    torch.set_num_threads(THREADS)
    torch.manual_seed(SEED)
    data = load_training()
    xs, ms = validation_set(data)
    history, best, stale = [], math.inf, 0
    with gpu_lock("cuda"):
        device, model, optimizer = setup(cap_gb)
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        for step, batch in enumerate(batches(data, steps)):
            loss = update(model, optimizer, batch, lr_at(step, steps, 1e-3), device)
            if (step + 1) % eval_every:
                continue
            record = {"step": step + 1, "train_loss": loss, "seconds": time.time() - started}
            record.update(evaluate(model, xs, ms, device))
            history.append(record)
            log(f"train {tag}: {json.dumps(record)}")
            if record["nll"] < best:
                best, stale = record["nll"], 0
                torch.save(model.state_dict(), OUT / f"model_{tag}.pt")
            else:
                stale += 1
            if stale >= patience:
                log(f"train {tag}: early stop at step {step + 1}")
                break
        peak = torch.cuda.max_memory_allocated() / 2**30
    summary = {
        "tag": tag,
        "parameters": sum(p.numel() for p in model.parameters()),
        "max_steps": steps,
        "steps_run": history[-1]["step"],
        "best_val_nll": best,
        "best_step": min(history, key=lambda r: r["nll"])["step"],
        "seconds": time.time() - started,
        "peak_memory_gb": peak,
        "train_records": int((~data["val"]).sum()),
        "val_records": int(data["val"].sum()),
        "history": history,
    }
    (OUT / f"train_{tag}.json").write_text(json.dumps(summary, indent=2))
    log(
        f"train {tag}: done, best val nll {best:.4f} at step {summary['best_step']}, "
        f"{summary['seconds'] / 60:.1f} min"
    )


def record_set(name: str) -> np.ndarray:
    """
    PTB-XL record IDs of a named map set.

    Parameters
    ----------
    name : str
        ``ecgad_dev`` (ECGAD development), ``evaluation`` (LocBench evaluation partition) or ``val``
        (held-out fit clean normals of this experiment).

    Returns
    -------
    np.ndarray
        Sorted record IDs.
    """
    if name == "ecgad_dev":
        split = load_split()
        return np.sort(split[split.partition == "development"].ecg_id.to_numpy())
    if name == "evaluation":
        partition = load_partition()
        return np.sort(partition[partition.partition == "evaluation"].ecg_id.to_numpy())
    if name == "val":
        table = training_split()
        return np.sort(table[table.val].ecg_id.to_numpy())
    raise ValueError(f"Unknown record set {name}")


def maps(name: str, tag: str, cap_gb: float, limit: int | None) -> None:
    """
    Write LocBench maps (n, 12, 5000) for every declared variant: float32 for ECGAD, float16 otherwise.

    Parameters
    ----------
    name : str
        Record set (see ``record_set``).
    tag : str
        Model tag.
    cap_gb : float
        GPU memory cap.
    limit : int or None
        Only the first records (smoke tests); outputs then go under ``smoke/``.
    """
    torch.set_num_threads(THREADS)
    ecg_ids = record_set(name)[:limit]
    target = OUT / "smoke" if limit else OUT / f"maps_{tag}"
    target.mkdir(parents=True, exist_ok=True)
    signals, peaks = load_records(ecg_ids)
    dtype = np.float32 if name == "ecgad_dev" else np.float16
    outputs = {
        variant: np.lib.format.open_memmap(
            target / f"{name}_{variant}.npy.part", mode="w+", dtype=dtype, shape=(len(ecg_ids), 12, 5000)
        )
        for variant in VARIANTS
    }
    family_cover = np.zeros(len(FAMILIES))
    started = time.time()
    with gpu_lock("cuda"):
        device = choose_device(cap_gb)
        model = InpaintNet().to(device)
        model.load_state_dict(torch.load(OUT / f"model_{tag}.pt", map_location=device))
        model.eval()
        with ThreadPoolExecutor(4) as pool:
            pending = []
            for k in range(len(ecg_ids)):
                errors = record_errors(model, signals[k], peaks[k], device)
                family_cover += [np.isfinite(errors[f"z_{f}"]).mean() for f in FAMILIES]
                pending.append((k, pool.submit(variant_maps, errors)))
                if len(pending) >= 16:
                    for j, fut in pending:
                        for variant, value in fut.result().items():
                            outputs[variant][j] = value
                    pending = []
                if (k + 1) % 500 == 0:
                    log(f"maps {name}: {k + 1}/{len(ecg_ids)} in {(time.time() - started) / 60:.1f} min")
            for j, fut in pending:
                for variant, value in fut.result().items():
                    outputs[variant][j] = value
    for variant, array in outputs.items():
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite values in {variant}")
        array.flush()
        del array
        os.replace(target / f"{name}_{variant}.npy.part", target / f"{name}_{variant}.npy")
    manifest = {
        "set": name,
        "model": to_stored(OUT / f"model_{tag}.pt"),
        "records": len(ecg_ids),
        "ecg_ids": [int(e) for e in ecg_ids],
        "variants": {v: to_stored(target / f"{name}_{v}.npy") for v in VARIANTS},
        "family_coverage": dict(zip(FAMILIES, (family_cover / len(ecg_ids)).tolist(), strict=True)),
        "seconds": time.time() - started,
    }
    (target / f"{name}_manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"maps {name}: {len(ecg_ids)} records in {manifest['seconds'] / 60:.1f} min -> {to_stored(target)}")


def calibration(tag: str, cap_gb: float) -> None:
    """
    Z-score calibration on the held-out fit clean normals under the fixed inference masks.

    Parameters
    ----------
    tag : str
        Model tag.
    cap_gb : float
        GPU memory cap.
    """
    ecg_ids = record_set("val")
    signals, peaks = load_records(ecg_ids)
    stats = {f: {"z": [], "abs": []} for f in FAMILIES}
    with gpu_lock("cuda"):
        device = choose_device(cap_gb)
        model = InpaintNet().to(device)
        model.load_state_dict(torch.load(OUT / f"model_{tag}.pt", map_location=device))
        model.eval()
        for k in range(len(ecg_ids)):
            errors = record_errors(model, signals[k], peaks[k], device)
            for f in FAMILIES:
                z = errors[f"z_{f}"][:, 50:2450]
                stats[f]["z"].append(z[np.isfinite(z)])
                stats[f]["abs"].append(errors[f"abs_{f}"][:, 50:2450][np.isfinite(z)])
    summary = {}
    for f in FAMILIES:
        z, a = np.concatenate(stats[f]["z"]), np.concatenate(stats[f]["abs"])
        summary[f] = {
            "mean_z": float(z.mean()),
            "median_z": float(np.median(z)),
            "share_z_above_2": float((z > 2).mean()),
            "share_z_above_3": float((z > 3).mean()),
            "mean_abs_error_mv": float(a.mean()),
        }
    result = {"tag": tag, "records": len(ecg_ids), "families": summary}
    (OUT / f"calibration_{tag}.json").write_text(json.dumps(result, indent=2))
    log(f"calibration {tag}: {json.dumps(summary)}")


def main() -> None:
    """Parse arguments and run a subcommand."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "profile", "train", "maps", "calibration"])
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--tag", default="v1")
    parser.add_argument("--set", default="ecgad_dev")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cap-gb", type=float, default=8.0)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "profile":
        profile(args.steps, args.cap_gb)
    elif args.command == "train":
        train(args.steps, args.eval_every, args.patience, args.tag, args.cap_gb)
    elif args.command == "maps":
        maps(args.set, args.tag, args.cap_gb, args.limit)
    else:
        calibration(args.tag, args.cap_gb)


if __name__ == "__main__":
    main()
