"""Matched training and checkpoint state for the xECG DropPath rescue."""

from __future__ import annotations

import gc
import math
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from .files import write_torch_atomic
from .reproducibility import cpu_state, flat_rng_state, restore_flat_rng_state, seed_everything
from .xecg import XECGBinaryClassifier, load_xecg
from .xecg_droppath_rescue import PairedDropPath, install_droppath


def cpu_nested(value: Any) -> Any:
    """Clone a nested optimizer state onto CPU without retaining CUDA tensors."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_nested(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_nested(item) for item in value)
    return value


def same_nested(left: Any, right: Any) -> bool:
    """Compare CPU and CUDA tensor trees bitwise, including sequence lengths."""
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(same_nested(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(same_nested(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


def _numeric_tree(left: Any, right: Any, path: str, summary: dict) -> None:  # noqa: C901
    """Compare float tensors tightly; require exact structure and scalar state."""
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.shape != right.shape or left.dtype != right.dtype:
            raise RuntimeError(f"Replay tensor identity differs: {path}")
        if torch.equal(left, right):
            return
        if not left.is_floating_point() or path.endswith("/step"):
            raise RuntimeError(f"Replay integer or update-step tensor differs: {path}")
        if not torch.allclose(left, right, atol=1e-8, rtol=1e-6):
            error = (left - right).abs()
            raise RuntimeError(f"Replay numeric tolerance exceeded at {path}: max_abs={error.max().item()}")
        error = (left - right).abs()
        scale = torch.maximum(left.abs(), right.abs()).clamp_min(1e-12)
        summary["tensors_with_numeric_difference"] += 1
        summary["max_abs_difference"] = max(summary["max_abs_difference"], float(error.max()))
        summary["max_relative_difference"] = max(
            summary["max_relative_difference"], float((error / scale).max())
        )
        return
    if isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise RuntimeError(f"Replay mapping identity differs: {path}")
        for key in left:
            _numeric_tree(left[key], right[key], f"{path}/{key}", summary)
        return
    if isinstance(left, (tuple, list)):
        if type(left) is not type(right) or len(left) != len(right):
            raise RuntimeError(f"Replay sequence identity differs: {path}")
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            _numeric_tree(a, b, f"{path}/{index}", summary)
        return
    if left != right:
        raise RuntimeError(f"Replay scalar state differs: {path}")


def compare_replay_snapshots(first: dict, second: dict) -> dict:
    """Require exact order/RNG/schedule and tightly bounded float update drift."""
    if first.keys() != second.keys():
        raise RuntimeError("Replay snapshot fields differ")
    summary = {
        "atol": 1e-8,
        "rtol": 1e-6,
        "tensors_with_numeric_difference": 0,
        "max_abs_difference": 0.0,
        "max_relative_difference": 0.0,
        "exact_fields": [],
    }
    for key in first:
        if key in {"model", "optimizer"}:
            _numeric_tree(first[key], second[key], key, summary)
        elif not same_nested(first[key], second[key]):
            raise RuntimeError(f"Replay exact state differs: {key}")
        else:
            summary["exact_fields"].append(key)
    return summary


def sequential_replay(run_once: Callable[[], dict], device: str) -> dict:
    """Run two next updates sequentially and apply the documented v5 criterion."""
    first = run_once()
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    second = run_once()
    return compare_replay_snapshots(first, second)


def build_model(
    checkpoint_dir: Path, probe_path: Path, arm: str, device: str, train_records: int
) -> tuple[
    nn.Module, PairedDropPath, torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR, torch.Generator
]:
    """Construct equal initial weights and independent optimizer/RNG streams."""
    seed_everything(42)
    backbone = load_xecg(checkpoint_dir, backend="vanilla", device=device, drop_path_prob=0.5)
    mask = install_droppath(backbone, arm, seed=42)
    model = XECGBinaryClassifier(backbone).to(device)
    with np.load(probe_path) as probe, torch.no_grad():
        model.head.weight.copy_(
            torch.as_tensor(probe["raw_weight"], dtype=torch.float32, device=device).view(1, -1)
        )
        model.head.bias.fill_(float(probe["raw_bias"]))
    optimizer = torch.optim.AdamW(layerwise_parameter_groups(model), weight_decay=0.1)
    scheduler = make_scheduler(optimizer, math.ceil(train_records / 64))
    permutation = torch.Generator(device="cpu").manual_seed(42)
    return model, mask, optimizer, scheduler, permutation


def layerwise_parameter_groups(model: nn.Module) -> list[dict]:
    """Match frozen 016's nine-block AdamW grouping and layerwise LR decay."""
    core = model.backbone.core.model.blocks
    if len(core) != 9:
        raise ValueError("Expected nine xECG blocks")
    groups: list[dict] = []
    used: set[int] = set()

    def add(name: str, parameters: Iterable[nn.Parameter], lr: float) -> None:
        chosen = [parameter for parameter in parameters if parameter.requires_grad]
        if any(id(parameter) in used for parameter in chosen):
            raise ValueError(f"Overlapping xECG parameter group: {name}")
        used.update(id(parameter) for parameter in chosen)
        if chosen:
            groups.append({"name": name, "params": chosen, "lr": lr, "weight_decay": 0.1})

    add("patch_embedding", model.backbone.patch_embedding.parameters(), 3e-5 * 0.75**10)
    for index, block in enumerate(core):
        add(f"core_block_{index}", block.parameters(), 3e-5 * 0.75 ** (9 - index))
    add(
        "core_remaining",
        (parameter for parameter in model.backbone.parameters() if id(parameter) not in used),
        3e-5,
    )
    add("binary_head", model.head.parameters(), 1e-3)
    if used != {id(parameter) for parameter in model.parameters() if parameter.requires_grad}:
        raise ValueError("Optimizer groups do not cover classifier")
    return groups


def make_scheduler(
    optimizer: torch.optim.Optimizer, steps_per_epoch: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Match frozen 016's one-epoch warmup and following half-cosine decay."""

    def factor(step: int) -> float:
        if step < steps_per_epoch:
            return step / steps_per_epoch
        progress = min(1.0, (step - steps_per_epoch) / steps_per_epoch)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@torch.inference_mode()
def predict(model: nn.Module, dataset: Dataset, device: str) -> np.ndarray:
    """Read development logits in manifest order and reject nonfinite values."""
    model.eval()
    chunks = []
    for signal, _ in DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0):
        chunks.append(model(signal.to(device)).float().cpu().numpy())
    result = np.concatenate(chunks)
    if not np.isfinite(result).all():
        raise RuntimeError("Nonfinite xECG logits")
    return result


def save_checkpoint(
    path: Path,
    fingerprint: dict,
    model: nn.Module,
    mask: PairedDropPath,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    permutation: torch.Generator,
    epoch: int,
    next_index: int,
    order: torch.Tensor | None,
    updates: int,
) -> dict:
    """Save all state needed to resume at the exact next optimizer update."""
    state = {
        "version": 1,
        "fingerprint": fingerprint,
        "model": cpu_state(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "mask_rng": mask.get_rng_state(),
        "mask_draws": mask.draws,
        "permutation_rng": permutation.get_state(),
        "epoch": epoch,
        "next_index": next_index,
        "order": order,
        "updates": updates,
        **flat_rng_state(),
    }
    write_torch_atomic(path, state)
    reloaded = torch.load(path, map_location="cpu", weights_only=True)
    if reloaded["fingerprint"] != fingerprint or reloaded["updates"] != updates:
        raise RuntimeError("Checkpoint identity or update position changed on reload")
    for name, tensor in state["model"].items():
        if not torch.equal(tensor, reloaded["model"][name]):
            raise RuntimeError(f"Checkpoint tensor changed on CPU serialization: {name}")
    if not torch.equal(state["mask_rng"], reloaded["mask_rng"]):
        raise RuntimeError("Mask RNG changed on checkpoint reload")
    return reloaded


def restore_checkpoint(
    path: Path,
    fingerprint: dict,
    model: nn.Module,
    mask: PairedDropPath,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    permutation: torch.Generator,
) -> dict:
    """Restore model, optimizer, scheduler, sampler and global/mask RNG."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("version") != 1 or saved.get("fingerprint") != fingerprint:
        raise ValueError("Resume checkpoint differs from frozen inputs")
    model.load_state_dict(saved["model"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    mask.set_rng_state(saved["mask_rng"])
    mask.draws = int(saved["mask_draws"])
    permutation.set_state(saved["permutation_rng"])
    restore_flat_rng_state(saved)
    return saved


def one_epoch(  # noqa: C901 - explicit update/resume invariants belong together
    model: nn.Module,
    mask: PairedDropPath,
    dataset: Dataset,
    device: str,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    permutation: torch.Generator,
    epoch: int,
    *,
    checkpoint: Path | None,
    fingerprint: dict,
    initial: dict | None = None,
    after_update: Callable[[nn.Module, float], None] | None = None,
    max_updates: int | None = None,
) -> dict:
    """Run exactly ceil(N/64) updates, including the final 63-record update."""
    count = len(dataset)
    order = torch.randperm(count, generator=permutation) if initial is None else initial["order"]
    start = 0 if initial is None else int(initial["next_index"])
    updates = epoch * math.ceil(count / 64) if initial is None else int(initial["updates"])
    if order is None or len(order) != count or start % 64 != 0 and start != count:
        raise ValueError("Malformed sampled record order or update position")
    model.train()
    elapsed = time.monotonic()
    loss_sum = 0.0
    seen = 0
    for completed_here, group_start in enumerate(range(start, count, 64), start=1):
        group = order[group_start : group_start + 64].tolist()
        optimizer.zero_grad(set_to_none=True)
        for signal, target in DataLoader(Subset(dataset, group), batch_size=16, shuffle=False, num_workers=0):
            signal, target = signal.to(device), target.to(device)
            loss = nn.functional.binary_cross_entropy_with_logits(model(signal), target, reduction="sum")
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite xECG loss")
            (loss / len(group)).backward()
            loss_sum += float(loss.detach())
            seen += len(target)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError("Nonfinite xECG gradient norm")
        optimizer.step()
        scheduler.step()
        updates += 1
        if after_update is not None and updates == 1:
            after_update(model, float(grad_norm))
        if checkpoint is not None and (updates % 40 == 0 or group_start + 64 >= count):
            save_checkpoint(
                checkpoint,
                fingerprint,
                model,
                mask,
                optimizer,
                scheduler,
                permutation,
                epoch,
                min(group_start + 64, count),
                order,
                updates,
            )
        if updates % 40 == 0:
            print({"epoch": epoch + 1, "updates": updates, "arm": mask.arm}, flush=True)
        if max_updates is not None and completed_here >= max_updates:
            break
    expected = (epoch + 1) * math.ceil(count / 64)
    if max_updates is None and (updates != expected or seen != count - start):
        raise RuntimeError("Training exposure or update count changed")
    return {
        "epoch": epoch + 1,
        "updates": updates,
        "seen": seen,
        "mean_loss": loss_sum / max(1, seen),
        "seconds": time.monotonic() - elapsed,
        "mask_draws": mask.draws,
    }
