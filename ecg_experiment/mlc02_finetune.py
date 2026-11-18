"""
Candidate F of MLC-02: the released xECG encoder fine-tuned end to end under H-X's head, and ``predict``.

The protocol's row F fine-tunes the encoder of the best single-encoder head on validation (H-X, xECG) with a
14-output head started from that head, layer-wise rate decay 0.75, top rate 1e-5 or 3e-5, up to 5 epochs and
mixed precision. The choices it leaves open were fixed here before the first training run (10 November 2026):

- Encoder: the released xECG with the vanilla xLSTM backend, the input path of Stage 1, and the released
  configuration's drop-path (0.0) and dropout (0.0).
- Head: H-X's network (1,024 inputs, 512 ReLU units, dropout 0.3, 14 logits) on pooled features standardized
  with H-X's training standardizer, which stays fixed. Its weights start from H-X's weights of the run's seed
  (H-X saved one head per seed), so that at step 0 the model reproduces H-X's validation predictions.
- Loss: binary cross-entropy averaged over the defined entries of each effective batch of 32 records;
  undefined labels are masked; no class weights, no augmentation.
- AdamW (default betas and epsilon) with weight decay 0.01 on weights and none on biases, normalisation
  weights and the unused mask token. Rates: the top rate ``r`` on the top xLSTM block, ``r * 0.75**(8 - i)``
  on block ``i``, ``r * 0.75**9`` on the patch embedding, ``10 r`` on the head.
- Schedule: linear warm-up over the first 5% of updates, then cosine decay to zero at the last update of the
  fifth epoch; gradient clipping at a global norm of 1.0.
- Precision: training forwards under BF16 autocast with FP32 master weights; every evaluation and ``predict``
  in FP32.
- Evaluation on validation after every half epoch (10 points); the kept point is the first with the highest
  selection metric, chosen separately for each seed.

A trained candidate lives in ``candidates/F/``: ``weights__seed-<s>.safetensors`` (the kept point of each
seed) and ``model.json`` (settings and the SHA-256 of every file). ``predict`` turns float32 ``[n, 12, 5000]``
waveforms at 500 Hz into ``[n, 14]`` probabilities, the mean of the seeds.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from ecg_experiment.cuda_runtime20261028 import configure_cuda_runtime
from ecg_experiment.files import (
    read_json,
    sha256_file,
    write_csv_atomic,
    write_json_atomic,
    write_npz_atomic,
    write_torch_atomic,
)
from ecg_experiment.mlc02_data import CONDITIONS, check_probabilities
from ecg_experiment.mlc02_evaluation import selection_metric
from ecg_experiment.mlc02_feature_models import CANDIDATES, build_network, load_candidate, masked_bce
from ecg_experiment.mlc02_finetune_data import WORK, window_inputs
from ecg_experiment.xecg import DEFAULT_CHECKPOINT_DIR, EMBEDDING_SIZE, load_xecg

CANDIDATE = "F"
BASE = "H-X"
DIRECTORY = CANDIDATES / CANDIDATE
RUNS = WORK / "runs"
TOP_RATES = (1e-5, 3e-5)
LAYER_DECAY = 0.75
HEAD_FACTOR = 10.0
WEIGHT_DECAY = 0.01
CLIP_NORM = 1.0
BATCH = 32
EPOCHS = 5
WARMUP_FRACTION = 0.05
EVALUATIONS_PER_EPOCH = 2
EVAL_BATCH = 16
DROP_PATH = 0.0
BLOCKS = 9
NORM_CLASSES = ("LayerNorm", "MultiHeadLayerNorm")
CHECKPOINT_VERSION = 1
LOG_EVERY = 100
UE_COUNT = Path("/sys/devices/system/edac/mc/mc0/ue_count")


class FineTuned(nn.Module):
    """Released xECG encoder, a fixed standardizer of its pooled features, and H-X's head."""

    def __init__(self, encoder: nn.Module, mean: np.ndarray, scale: np.ndarray) -> None:
        """
        Assemble the model.

        Parameters
        ----------
        encoder : nn.Module
            Released xECG model with average pooling.
        mean, scale : np.ndarray
            Training standardizer of the 1,024 pooled features.
        """
        super().__init__()
        self.encoder = encoder
        self.register_buffer("mean", torch.as_tensor(np.asarray(mean), dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(np.asarray(scale), dtype=torch.float32))
        self.head = build_network(EMBEDDING_SIZE)

    def pooled(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Pooled encoder features.

        Parameters
        ----------
        signal : torch.Tensor
            ``[batch, 1000, 12]`` xECG inputs.

        Returns
        -------
        torch.Tensor
            ``[batch, 1024]``.
        """
        pooled, _ = self.encoder(signal)
        return pooled

    def head_logits(self, pooled: torch.Tensor) -> torch.Tensor:
        """
        Logits of the head on pooled features.

        Parameters
        ----------
        pooled : torch.Tensor
            ``[batch, 1024]`` pooled features.

        Returns
        -------
        torch.Tensor
            ``[batch, 14]``.
        """
        return self.head((pooled - self.mean) / self.scale)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Logits of the 14 conditions.

        Parameters
        ----------
        signal : torch.Tensor
            ``[batch, 1000, 12]`` xECG inputs.

        Returns
        -------
        torch.Tensor
            ``[batch, 14]``.
        """
        return self.head_logits(self.pooled(signal))


def pretrained_drop_path() -> float:
    """
    Drop-path probability of the released xECG configuration.

    Returns
    -------
    float
        ``drop_path_prob`` of the release's ``config.json``.
    """
    return float(json.loads((DEFAULT_CHECKPOINT_DIR / "config.json").read_text())["drop_path_prob"])


def xecg_encoder() -> nn.Module:
    """
    Released xECG encoder on the CPU with the vanilla backend and the released drop-path.

    Returns
    -------
    nn.Module
        Strictly loaded model; ``load_xecg`` checks the release hashes.
    """
    configure_cuda_runtime()
    return load_xecg(DEFAULT_CHECKPOINT_DIR, backend="vanilla", device="cpu", drop_path_prob=DROP_PATH)


def load_head(model: FineTuned, weights: dict[str, np.ndarray]) -> None:
    """
    Copy H-X's head weights into the model.

    Parameters
    ----------
    model : FineTuned
        Model to change in place.
    weights : dict[str, np.ndarray]
        ``w1``, ``b1``, ``w2`` and ``b2`` of ``mlc02_feature_models.network_weights``.

    Raises
    ------
    ValueError
        If a shape differs.
    """
    targets = {"w1": model.head[0].weight, "b1": model.head[0].bias, "w2": model.head[3].weight,
               "b2": model.head[3].bias}
    with torch.no_grad():
        for name, target in targets.items():
            value = torch.as_tensor(np.asarray(weights[name]), dtype=torch.float32)
            if value.shape != target.shape:
                raise ValueError(f"H-X {name} has shape {tuple(value.shape)}, expected {tuple(target.shape)}")
            target.copy_(value)


def initial_model(
    seed: int, device: str, base_directory: Path = CANDIDATES
) -> tuple[FineTuned, dict[str, Any]]:
    """
    Released encoder with H-X's standardizer and its head of one seed.

    Parameters
    ----------
    seed : int
        H-X seed whose head starts the run.
    device : str
        Torch device.
    base_directory : Path
        Directory holding ``H-X/model.json``.

    Returns
    -------
    tuple[FineTuned, dict[str, Any]]
        The model on ``device`` and its origin (H-X files and their SHA-256, head seed).

    Raises
    ------
    ValueError
        If H-X has no head of that seed.
    """
    base = load_candidate(BASE, base_directory)
    if seed not in base["seeds"]:
        raise ValueError(f"H-X has no head of seed {seed}")
    scaler = base["artefacts"]["scaler.npz"]
    model = FineTuned(xecg_encoder(), scaler["mean"], scaler["scale"])
    load_head(model, base["artefacts"][f"weights__seed-{seed}.npz"])
    model.requires_grad_(True)
    return model.to(device), {"base": BASE, "head_seed": seed, "base_files_sha256": base["files"]}


def layer_name(name: str) -> str:
    """
    Layer of a parameter for the layer-wise rates.

    Parameters
    ----------
    name : str
        Parameter name in ``FineTuned``.

    Returns
    -------
    str
        ``head``, ``block_<i>`` or ``embedding`` (patch embedding and mask token).

    Raises
    ------
    ValueError
        If the parameter belongs to none of them.
    """
    if name.startswith("head."):
        return "head"
    prefix = "encoder.core.model.blocks."
    if name.startswith(prefix):
        return f"block_{int(name[len(prefix):].split('.', 1)[0])}"
    if name.startswith("encoder.patch_embedding.") or name == "encoder.mask_token":
        return "embedding"
    raise ValueError(f"parameter {name} has no layer-wise rate")


def layer_rate(layer: str, top_rate: float) -> float:
    """
    Learning rate of a layer.

    Parameters
    ----------
    layer : str
        Output of ``layer_name``.
    top_rate : float
        Rate of the top xLSTM block.

    Returns
    -------
    float
        ``10 r`` for the head, ``r * 0.75**(8 - i)`` for block ``i`` and ``r * 0.75**9`` for the embedding.
    """
    if layer == "head":
        return HEAD_FACTOR * top_rate
    if layer == "embedding":
        return top_rate * LAYER_DECAY**BLOCKS
    return top_rate * LAYER_DECAY ** (BLOCKS - 1 - int(layer.removeprefix("block_")))


def decayed(name: str, modules: dict[str, nn.Module]) -> bool:
    """
    Whether weight decay applies to a parameter.

    Parameters
    ----------
    name : str
        Parameter name.
    modules : dict[str, nn.Module]
        ``named_modules`` of the model.

    Returns
    -------
    bool
        False for biases, normalisation weights and the mask token; True for every other weight.
    """
    owner = modules[name.rsplit(".", 1)[0]]
    return not (
        name.endswith(("bias", "_bias_"))
        or type(owner).__name__ in NORM_CLASSES
        or name == "encoder.mask_token"
    )


def parameter_groups(model: nn.Module, top_rate: float) -> list[dict[str, Any]]:
    """
    AdamW groups with layer-wise rates and weight decay on weights only.

    Parameters
    ----------
    model : nn.Module
        ``FineTuned`` model.
    top_rate : float
        Rate of the top xLSTM block.

    Returns
    -------
    list[dict[str, Any]]
        One group per layer and decay setting: ``name``, ``params``, ``parameter_names``, ``lr``, ``base_lr``
        and ``weight_decay``.

    Raises
    ------
    ValueError
        If a parameter has no layer or the groups do not cover every parameter exactly once.
    """
    modules = dict(model.named_modules())
    groups: dict[tuple[str, bool], dict[str, Any]] = {}
    for name, parameter in model.named_parameters():
        layer, decay = layer_name(name), decayed(name, modules)
        rate = layer_rate(layer, top_rate)
        group = groups.setdefault((layer, decay), {
            "name": f"{layer}/{'decay' if decay else 'no_decay'}", "params": [], "parameter_names": [],
            "lr": rate, "base_lr": rate, "weight_decay": WEIGHT_DECAY if decay else 0.0})
        group["params"].append(parameter)
        group["parameter_names"].append(name)
    covered = [id(parameter) for group in groups.values() for parameter in group["params"]]
    if sorted(covered) != sorted(id(parameter) for parameter in model.parameters()):
        raise ValueError("the parameter groups do not cover every parameter exactly once")
    return list(groups.values())


def make_optimizer(model: nn.Module, top_rate: float) -> torch.optim.AdamW:
    """
    AdamW over ``parameter_groups``.

    Parameters
    ----------
    model : nn.Module
        ``FineTuned`` model.
    top_rate : float
        Rate of the top xLSTM block.

    Returns
    -------
    torch.optim.AdamW
        Optimizer with default betas and epsilon, one tensor at a time (``foreach=False``).
    """
    return torch.optim.AdamW(parameter_groups(model, top_rate), foreach=False)


def plan(records: int) -> dict[str, Any]:
    """
    Compute the updates and evaluation points of one run.

    Parameters
    ----------
    records : int
        Training records.

    Returns
    -------
    dict[str, Any]
        ``per_epoch`` (updates, the last batch partial), ``total``, ``warmup`` and ``points`` (updates after
        which validation is scored: every half epoch, without repeats in runs of a few records).
    """
    per_epoch = math.ceil(records / BATCH)
    total = EPOCHS * per_epoch
    marks = [math.ceil(per_epoch * (k + 1) / EVALUATIONS_PER_EPOCH) for k in range(EVALUATIONS_PER_EPOCH)]
    points = sorted({epoch * per_epoch + mark for epoch in range(EPOCHS) for mark in marks})
    return {"records": records, "per_epoch": per_epoch, "total": total,
            "warmup": math.ceil(WARMUP_FRACTION * total), "points": points}


def rate_factor(update: int, total: int, warmup: int) -> float:
    """
    Multiplier of every base rate at one update.

    Parameters
    ----------
    update : int
        Update number, from 1 to ``total``.
    total : int
        Updates of the run.
    warmup : int
        Warm-up updates.

    Returns
    -------
    float
        ``update / warmup`` during warm-up, then a cosine from 1 down to 0 at ``total``.
    """
    if update <= warmup:
        return update / warmup
    return 0.5 * (1.0 + math.cos(math.pi * (update - warmup) / (total - warmup)))


def set_rates(optimizer: torch.optim.Optimizer, factor: float) -> None:
    """
    Scale every group's base rate.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer of ``make_optimizer``.
    factor : float
        Output of ``rate_factor``.
    """
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * factor


def train_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    microbatch: int,
    device: str,
    autocast: bool = True,
) -> tuple[float, float]:
    """
    One optimizer update over an effective batch, accumulated over microbatches.

    The loss is the binary cross-entropy summed over the defined entries of the effective batch and divided by
    their number, so the accumulated gradient equals that of the whole batch at once.

    Parameters
    ----------
    model : nn.Module
        Model in any mode; it is put in training mode.
    optimizer : torch.optim.Optimizer
        Optimizer with the rates of this update.
    inputs : np.ndarray
        Float32 ``[records, 1000, 12]`` training inputs.
    labels : np.ndarray
        Float32 ``[records, 14]`` states with NaN where undefined.
    indices : np.ndarray
        Records of the effective batch.
    microbatch : int
        Records per forward pass.
    device : str
        Torch device.
    autocast : bool
        Run forwards under BF16 autocast.

    Returns
    -------
    tuple[float, float]
        Batch loss and the gradient norm before clipping.

    Raises
    ------
    RuntimeError
        If the loss or the gradient norm is not finite, or the batch has no defined label.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    defined = ~np.isnan(labels[indices])
    total = int(defined.sum())
    if total == 0:
        raise RuntimeError("an effective batch has no defined label")
    batch_loss = 0.0
    for start in range(0, len(indices), microbatch):
        part = indices[start:start + microbatch]
        signal = torch.from_numpy(inputs[part]).to(device)
        target = torch.from_numpy(labels[part]).to(device)
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=autocast):
            logits = model(signal)
        loss = masked_bce(logits.float(), target) * (int(defined[start:start + microbatch].sum()) / total)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite fine-tuning loss")
        loss.backward()
        batch_loss += float(loss.detach())
    norm = nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM, error_if_nonfinite=True)
    optimizer.step()
    return batch_loss, float(norm)


def evaluate(
    model: FineTuned, inputs: np.ndarray, device: str, batch: int = EVAL_BATCH, pooled: bool = False
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    FP32 evaluation-mode probabilities, and optionally the pooled features.

    Parameters
    ----------
    model : FineTuned
        Model.
    inputs : np.ndarray
        Float32 ``[records, 1000, 12]`` inputs.
    device : str
        Torch device.
    batch : int
        Records per forward pass (Stage 1 used 16).
    pooled : bool
        Also return the pooled features.

    Returns
    -------
    tuple[np.ndarray, np.ndarray | None]
        Float64 ``[records, 14]`` probabilities and float32 ``[records, 1024]`` features (or None).
    """
    model.eval()
    probabilities, features = [], []
    with torch.inference_mode():
        for start in range(0, len(inputs), batch):
            signal = torch.from_numpy(np.ascontiguousarray(inputs[start:start + batch])).to(device)
            values = model.pooled(signal)
            probabilities.append(torch.sigmoid(model.head_logits(values).double()).cpu().numpy())
            if pooled:
                features.append(values.float().cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(features) if pooled else None


def point_row(
    point: int, update: int, per_epoch: int, probability: np.ndarray, validation: dict[str, Any]
) -> dict[str, Any]:
    """
    Score one evaluation point: validation selection metric and per-source macro AUROC.

    Parameters
    ----------
    point : int
        Evaluation point, 0 for the start.
    update : int
        Updates done.
    per_epoch : int
        Updates per epoch.
    probability : np.ndarray
        Validation probabilities.
    validation : dict[str, Any]
        ``labels`` and ``sources`` of the validation records.

    Returns
    -------
    dict[str, Any]
        ``point``, ``update``, ``epoch``, ``selection_metric`` and ``macro_<source>`` per source.
    """
    selection = selection_metric(validation["labels"], probability, validation["sources"])
    return {"point": point, "update": update, "epoch": update / per_epoch,
            "selection_metric": selection["selection_metric"],
            **{f"macro_{source}": value for source, value in selection["macro_auroc_by_source"].items()}}


def peak_bytes(device: str) -> int | None:
    """
    Peak allocated GPU memory of this process since the last reset.

    Parameters
    ----------
    device : str
        Torch device.

    Returns
    -------
    int | None
        Bytes, or None on the CPU.
    """
    return int(torch.cuda.max_memory_allocated()) if torch.device(device).type == "cuda" else None


def ue_count() -> str | None:
    """
    Uncorrectable ECC memory errors counted by EDAC on memory controller 0.

    Returns
    -------
    str | None
        The counter's text, or None where it is not readable.
    """
    return UE_COUNT.read_text().strip() if UE_COUNT.exists() else None


def save_weights(model: nn.Module, path: Path) -> str:
    """
    Save the model's state as safetensors through a temporary file.

    Parameters
    ----------
    model : nn.Module
        Model.
    path : Path
        Destination.

    Returns
    -------
    str
        SHA-256 of the written file.
    """
    temporary = path.with_name(f".{path.name}.tmp")
    save_file({name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()},
              str(temporary))
    os.replace(temporary, path)
    return sha256_file(path)


def save_state(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, generator: torch.Generator,
               progress: dict[str, Any], device: str) -> None:
    """
    Save everything needed to resume after an evaluation point.

    Parameters
    ----------
    path : Path
        Destination (``resume.pt``).
    model : nn.Module
        Model.
    optimizer : torch.optim.Optimizer
        Optimizer.
    generator : torch.Generator
        Record-order generator.
    progress : dict[str, Any]
        Run progress: ``fingerprint``, ``update``, ``order``, ``history``, ``best``, ``start`` and
        ``resumes``.
    device : str
        Torch device; on CUDA the dropout generator state is saved too.
    """
    cuda = torch.device(device).type == "cuda"
    state = {"version": CHECKPOINT_VERSION, **progress,
             "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
             "optimizer": optimizer.state_dict(), "generator": generator.get_state(),
             "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state() if cuda else None}
    write_torch_atomic(path, state)


def restore_state(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, generator: torch.Generator,
                  fingerprint: dict[str, Any], device: str) -> dict[str, Any]:
    """
    Restore a run saved by ``save_state``.

    Parameters
    ----------
    path : Path
        ``resume.pt``.
    model : nn.Module
        Freshly built model, changed in place.
    optimizer : torch.optim.Optimizer
        Freshly built optimizer, changed in place.
    generator : torch.Generator
        Record-order generator, changed in place.
    fingerprint : dict[str, Any]
        Identity the saved run must have.
    device : str
        Torch device.

    Returns
    -------
    dict[str, Any]
        The saved progress.

    Raises
    ------
    ValueError
        If the checkpoint has another version or fingerprint.
    """
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("version") != CHECKPOINT_VERSION or saved.get("fingerprint") != fingerprint:
        raise ValueError(f"{path} was saved by a run with another identity")
    model.load_state_dict(saved["model"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    generator.set_state(saved["generator"])
    torch.set_rng_state(saved["cpu_rng"])
    if torch.device(device).type == "cuda":
        torch.cuda.set_rng_state(saved["cuda_rng"])
    keys = ("fingerprint", "update", "order", "history", "best", "start", "resumes")
    return {key: saved[key] for key in keys}


def write_history(path: Path, history: list[dict[str, Any]]) -> None:
    """
    Write the evaluation-point log of a run as CSV.

    Parameters
    ----------
    path : Path
        Destination.
    history : list[dict[str, Any]]
        One row per evaluation point.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(path, history, list(dict.fromkeys(key for row in history for key in row)))


def start_run(model: FineTuned, validation: dict[str, Any], device: str, directory: Path,
              check: Callable[[np.ndarray, np.ndarray], dict[str, Any]] | None,
              fingerprint: dict[str, Any], per_epoch: int) -> dict[str, Any]:
    """
    Score the starting model on validation, check it, and open the run's progress.

    Parameters
    ----------
    model : FineTuned
        Model at step 0.
    validation : dict[str, Any]
        ``inputs``, ``labels`` and ``sources`` of the validation records.
    device : str
        Torch device.
    directory : Path
        Run directory.
    check : Callable | None
        Called with the step-0 probabilities and pooled features; it raises when a check fails and returns
        its record.
    fingerprint : dict[str, Any]
        Identity of the run.
    per_epoch : int
        Updates per epoch.

    Returns
    -------
    dict[str, Any]
        Progress at update 0, with the step-0 point (not eligible for choice) in ``history``.
    """
    started = time.perf_counter()
    probability, pooled = evaluate(model, validation["inputs"], device, pooled=True)
    seconds = time.perf_counter() - started
    start = check(probability, pooled) if check is not None else {}
    write_npz_atomic(directory / "validation__point-00.npz", probability=probability, update=np.int64(0))
    row = {**point_row(0, 0, per_epoch, probability, validation), "eligible": False,
           "evaluation_seconds": seconds}
    return {"fingerprint": fingerprint, "update": 0, "order": None, "history": [row], "best": None,
            "start": {**start, "evaluation_seconds": seconds}, "resumes": []}


def train_segment(model: nn.Module, optimizer: torch.optim.Optimizer, generator: torch.Generator,
                  progress: dict[str, Any], schedule: dict[str, Any], train: dict[str, np.ndarray],
                  setting: dict[str, Any], log: Callable[[str], None]) -> dict[str, Any]:
    """
    Train from the current update to the next evaluation point.

    A new record order is drawn at the start of every epoch from the run's CPU generator; the last batch of an
    epoch is partial.

    Parameters
    ----------
    model : nn.Module
        Model.
    optimizer : torch.optim.Optimizer
        Optimizer of ``make_optimizer``.
    generator : torch.Generator
        Record-order generator.
    progress : dict[str, Any]
        Run progress, changed in place.
    schedule : dict[str, Any]
        Output of ``plan``.
    train : dict[str, np.ndarray]
        ``inputs`` and ``labels`` of the training records.
    setting : dict[str, Any]
        ``microbatch``, ``device`` and ``autocast``.
    log : Callable[[str], None]
        Progress printer.

    Returns
    -------
    dict[str, Any]
        Loss, gradient-norm, clipping and timing summary of the segment.
    """
    target = min(point for point in schedule["points"] if point > progress["update"])
    per_epoch, losses, norms = schedule["per_epoch"], [], []
    started = time.perf_counter()
    while progress["update"] < target:
        update = progress["update"]
        if update % per_epoch == 0:
            progress["order"] = torch.randperm(len(train["inputs"]), generator=generator)
        position = (update % per_epoch) * BATCH
        indices = progress["order"][position:position + BATCH].numpy()
        factor = rate_factor(update + 1, schedule["total"], schedule["warmup"])
        set_rates(optimizer, factor)
        loss, norm = train_update(model, optimizer, train["inputs"], train["labels"], indices,
                                  setting["microbatch"], setting["device"], setting["autocast"])
        progress["update"] = update + 1
        losses.append(loss)
        norms.append(norm)
        if progress["update"] % LOG_EVERY == 0:
            seconds = (time.perf_counter() - started) / len(losses)
            log(f"update {progress['update']}/{schedule['total']} loss {np.mean(losses[-LOG_EVERY:]):.4f} "
                f"norm {np.mean(norms[-LOG_EVERY:]):.3f} factor {factor:.4f} {seconds:.3f} s/update")
    return {"train_loss_mean": float(np.mean(losses)), "grad_norm_mean": float(np.mean(norms)),
            "grad_norm_max": float(np.max(norms)),
            "clipped_fraction": float(np.mean(np.array(norms) > CLIP_NORM)),
            "train_updates": len(losses), "train_seconds": time.perf_counter() - started,
            "rate_factor": factor}


def record_point(model: FineTuned, optimizer: torch.optim.Optimizer, generator: torch.Generator,
                 progress: dict[str, Any], schedule: dict[str, Any], validation: dict[str, Any],
                 segment: dict[str, Any], setting: dict[str, Any]) -> None:
    """
    Score an evaluation point, keep the best weights, save the resumable state and the log.

    Parameters
    ----------
    model : FineTuned
        Model after the point's update.
    optimizer : torch.optim.Optimizer
        Optimizer.
    generator : torch.Generator
        Record-order generator.
    progress : dict[str, Any]
        Run progress, changed in place.
    schedule : dict[str, Any]
        Output of ``plan``.
    validation : dict[str, Any]
        ``inputs``, ``labels`` and ``sources`` of the validation records.
    segment : dict[str, Any]
        Output of ``train_segment``.
    setting : dict[str, Any]
        ``device``, ``directory`` and ``published`` (where the log is copied).
    """
    directory, device = setting["directory"], setting["device"]
    point = schedule["points"].index(progress["update"]) + 1
    started = time.perf_counter()
    probability, _ = evaluate(model, validation["inputs"], device)
    seconds = time.perf_counter() - started
    write_npz_atomic(directory / f"validation__point-{point:02d}.npz", probability=probability,
                     update=np.int64(progress["update"]))
    row = {**point_row(point, progress["update"], schedule["per_epoch"], probability, validation),
           "eligible": True, **segment, "evaluation_seconds": seconds,
           "peak_allocated_bytes": peak_bytes(device),
           "ue_count": ue_count(), "unix_time": time.time()}
    best = progress["best"]
    if best is None or row["selection_metric"] > best["selection_metric"]:
        digest = save_weights(model, directory / "best.safetensors")
        progress["best"] = {key: row[key] for key in ("point", "update", "epoch", "selection_metric")}
        progress["best"]["weights_sha256"] = digest
    progress["history"].append(row)
    save_state(directory / "resume.pt", model, optimizer, generator, progress, device)
    for folder in (directory, setting["published"]):
        write_history(folder / "evaluation_points.csv", progress["history"])


def train_run(rate: float, seed: int, train: dict[str, np.ndarray], validation: dict[str, Any],
              setting: dict[str, Any], fingerprint: dict[str, Any],
              check: Callable[[np.ndarray, np.ndarray], dict[str, Any]] | None = None,
              log: Callable[[str], None] = print) -> dict[str, Any]:
    """
    One fine-tuning run, resumable from its last evaluation point.

    Parameters
    ----------
    rate : float
        Top rate.
    seed : int
        Run seed: H-X head, record order and dropout.
    train : dict[str, np.ndarray]
        ``inputs`` and ``labels`` of the training records.
    validation : dict[str, Any]
        ``inputs``, ``labels`` and ``sources`` of the validation records.
    setting : dict[str, Any]
        ``microbatch``, ``device``, ``autocast``, ``directory`` (local run directory) and ``published`` (where
        the evaluation log and ``run.json`` are copied).
    fingerprint : dict[str, Any]
        Identity a resumed checkpoint must match.
    check : Callable | None
        Step-0 check of ``start_run``.
    log : Callable[[str], None]
        Progress printer.

    Returns
    -------
    dict[str, Any]
        ``run.json`` content: settings, schedule, step-0 checks, every evaluation point and the best one.
    """
    directory = Path(setting["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    Path(setting["published"]).mkdir(parents=True, exist_ok=True)
    schedule = plan(len(train["inputs"]))
    torch.manual_seed(seed)
    model, origin = initial_model(seed, setting["device"])
    optimizer = make_optimizer(model, rate)
    generator = torch.Generator().manual_seed(seed)
    if (directory / "resume.pt").exists():
        progress = restore_state(directory / "resume.pt", model, optimizer, generator, fingerprint,
                                 setting["device"])
        progress["resumes"].append(
            {"update": progress["update"], "unix_time": time.time(), "ue_count": ue_count()})
        log(f"resumed at update {progress['update']}")
    else:
        progress = start_run(model, validation, setting["device"], directory, check, fingerprint,
                             schedule["per_epoch"])
        log(f"step 0 selection metric {progress['history'][0]['selection_metric']:.4f}")
    while progress["update"] < schedule["total"]:
        segment = train_segment(model, optimizer, generator, progress, schedule, train, setting, log)
        record_point(model, optimizer, generator, progress, schedule, validation, segment, setting)
        log(f"point {len(progress['history']) - 1} update {progress['update']} "
            f"selection {progress['history'][-1]['selection_metric']:.4f}")
    result = {"rate": rate, "seed": seed, "origin": origin, "schedule": schedule, "fingerprint": fingerprint,
              "groups": [{key: group[key] for key in ("name", "base_lr", "weight_decay")}
                         | {"parameters": len(group["params"]),
                            "elements": int(sum(p.numel() for p in group["params"]))}
                         for group in optimizer.param_groups],
              "start": progress["start"], "history": progress["history"], "best": progress["best"],
              "resumes": progress["resumes"], "complete": True}
    for folder in (directory, Path(setting["published"])):
        write_json_atomic(folder / "run.json", result)
    return result


def write_model(directory: Path, seeds: tuple[int, ...], settings: dict[str, Any]) -> dict[str, Any]:
    """
    Write ``model.json``, the index of the trained candidate.

    Parameters
    ----------
    directory : Path
        Candidate directory holding ``weights__seed-<s>.safetensors``.
    seeds : tuple[int, ...]
        Seeds whose mean probability ``predict`` returns.
    settings : dict[str, Any]
        Chosen settings.

    Returns
    -------
    dict[str, Any]
        The written index.
    """
    files = [f"weights__seed-{seed}.safetensors" for seed in seeds]
    index = {"candidate": CANDIDATE, "kind": "fine-tuned encoder", "base": BASE,
             "input": "float32 [n, 12, 5000] waveforms, 500 Hz, mV, leads I, II, III, aVR, aVL, aVF, V1-V6",
             "conditions": list(CONDITIONS), "seeds": list(seeds), "settings": settings,
             "files": {name: sha256_file(Path(directory) / name) for name in files}}
    write_json_atomic(Path(directory) / "model.json", index)
    return index


def load_trained(directory: Path, seed: int, device: str) -> FineTuned:
    """
    One seed's trained model.

    Parameters
    ----------
    directory : Path
        Candidate directory.
    seed : int
        Seed.
    device : str
        Torch device.

    Returns
    -------
    FineTuned
        Model in evaluation mode.
    """
    model = FineTuned(xecg_encoder(), np.zeros(EMBEDDING_SIZE), np.ones(EMBEDDING_SIZE))
    model.load_state_dict(load_file(str(Path(directory) / f"weights__seed-{seed}.safetensors")), strict=True)
    return model.to(device).eval()


def checked_index(directory: Path) -> dict[str, Any]:
    """
    Read ``model.json`` after checking the SHA-256 of every file.

    Parameters
    ----------
    directory : Path
        Candidate directory.

    Returns
    -------
    dict[str, Any]
        The index.

    Raises
    ------
    ValueError
        If the index is not F's or a file differs.
    """
    index = read_json(Path(directory) / "model.json")
    if index["candidate"] != CANDIDATE or tuple(index["conditions"]) != CONDITIONS:
        raise ValueError(f"{directory} does not hold candidate {CANDIDATE}")
    found = {name: sha256_file(Path(directory) / name) for name in index["files"]}
    if found != index["files"]:
        raise ValueError(f"files of {CANDIDATE} differ from model.json")
    return index


def predict_inputs(inputs: np.ndarray, directory: Path = DIRECTORY, seeds: tuple[int, ...] | None = None,
                   device: str = "cuda", batch: int = EVAL_BATCH) -> np.ndarray:
    """
    Probabilities of the trained candidate for prepared xECG inputs.

    Parameters
    ----------
    inputs : np.ndarray
        Float32 ``[n, 1000, 12]`` inputs of ``mlc02_finetune_data.window_inputs``.
    directory : Path
        Candidate directory.
    seeds : tuple[int, ...] | None
        Seeds to average; None gives the candidate's prediction (all of its seeds).
    device : str
        Torch device.
    batch : int
        Records per forward pass.

    Returns
    -------
    np.ndarray
        Float64 ``[n, 14]`` in ``CONDITIONS`` order.

    Raises
    ------
    ValueError
        If a seed is not one of the candidate's.
    """
    index = checked_index(directory)
    chosen = tuple(index["seeds"]) if seeds is None else tuple(seeds)
    if not chosen or not set(chosen) <= set(index["seeds"]):
        raise ValueError(f"seeds {chosen} are not among {index['seeds']}")
    probabilities = []
    for seed in chosen:
        model = load_trained(directory, seed, device)
        probabilities.append(evaluate(model, inputs, device, batch)[0])
        del model
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
    return check_probabilities(np.mean(np.stack(probabilities), axis=0), len(inputs), len(CONDITIONS))


def predict(waveforms: np.ndarray, directory: Path = DIRECTORY, seeds: tuple[int, ...] | None = None,
            device: str = "cuda", batch: int = EVAL_BATCH) -> np.ndarray:
    """
    Probabilities of the trained candidate for raw waveforms.

    Parameters
    ----------
    waveforms : np.ndarray
        Float32 ``[n, 12, 5000]``: the first 10 s at 500 Hz in mV, leads I, II, III, aVR, aVL, aVF, V1-V6
        (the MLC-02 waveform cache layout); NaN marks samples WFDB read as missing.
    directory : Path
        Candidate directory.
    seeds : tuple[int, ...] | None
        Seeds to average; None gives the candidate's prediction, the mean of its three seeds.
    device : str
        Torch device.
    batch : int
        Records per forward pass.

    Returns
    -------
    np.ndarray
        Float64 ``[n, 14]`` in ``CONDITIONS`` order.
    """
    return predict_inputs(window_inputs(waveforms), directory, seeds, device, batch)
