"""Matched moving/frozen xECG training mechanics for Experiment 016 v9."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from .reproducibility import flat_rng_state, restore_flat_rng_state, seed_everything
from .xecg import XECGBinaryClassifier, load_xecg
from .xecg_droppath_rescue import install_droppath
from .xecg_rescue_training import layerwise_parameter_groups, make_scheduler

TRAIN_RECORDS = 15359
DEVELOPMENT_RECORDS = 1306
UPDATES = 240
MICROBATCH = 16
EFFECTIVE_BATCH = 64
CLIP_NORM = 3.0
ARMS = ("M", "F")
DIAGNOSTIC_UPDATES = (0, 1, 60, 120, 240)


def build_model(
    release: Path,
    probe_path: Path,
    arm: str,
    seed: int,
    device: str,
) -> tuple[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR, torch.Generator]:
    """Load common released tensors, then apply only the encoder-LR intervention."""
    if arm not in ARMS:
        raise ValueError("Unknown encoder-motion arm")
    seed_everything(42)
    backbone = load_xecg(release, backend="vanilla", device=device, drop_path_prob=0.5)
    install_droppath(backbone, "off", seed=seed)
    model = XECGBinaryClassifier(backbone).to(device)
    if any(isinstance(module, nn.Dropout) and module.p != 0 for module in model.modules()):
        raise ValueError("Nonzero ordinary dropout in encoder-motion arms")
    if any(backbone.core.dropout_rates):
        raise ValueError("Stochastic depth is not Off")
    with np.load(probe_path) as probe, torch.no_grad():
        model.head.weight.copy_(
            torch.as_tensor(probe["raw_weight"], dtype=torch.float32, device=device)[None]
        )
        model.head.bias.fill_(float(probe["raw_bias"]))
    groups = layerwise_parameter_groups(model)
    if arm == "F":
        for group in groups:
            if group["name"] != "binary_head":
                group["lr"] = 0.0
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)
    scheduler = make_scheduler(optimizer, UPDATES)
    if len(optimizer.param_groups) != 12:
        raise RuntimeError("Expected patch, nine blocks, remaining backbone and head groups")
    if any(not parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Frozen arm must still compute all encoder gradients")
    if arm == "F" and any(
        group["lr"] != 0 or scheduler.base_lrs[index] != 0
        for index, group in enumerate(optimizer.param_groups)
        if group["name"] != "binary_head"
    ):
        raise RuntimeError("Frozen encoder LR or scheduler base LR is nonzero")
    seed_everything(seed)
    permutation = torch.Generator(device="cpu").manual_seed(seed)
    return model, optimizer, scheduler, permutation


def model_identity(model: nn.Module) -> str:
    """Digest every model parameter and buffer in stable state-dict order."""
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def encoder_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Clone every encoder parameter and buffer for bitwise freeze verification."""
    return {name: value.detach().clone() for name, value in model.backbone.state_dict().items()}


def encoder_equal(model: nn.Module, baseline: dict[str, torch.Tensor]) -> bool:
    """Check every encoder state tensor bitwise, including buffers."""
    state = model.backbone.state_dict()
    return state.keys() == baseline.keys() and all(
        torch.equal(value, baseline[name]) for name, value in state.items()
    )


def _gradient_norm(parameters: list[nn.Parameter]) -> float:
    """Measure an unmodified FP32 preclip gradient norm."""
    squares = [
        torch.sum(parameter.grad.detach().float() ** 2)
        for parameter in parameters
        if parameter.grad is not None
    ]
    return float(torch.sqrt(torch.sum(torch.stack(squares))).item()) if squares else 0.0


def train_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    dataset: Dataset,
    group: torch.Tensor,
    ecg_ids: np.ndarray,
    device: str,
    update_index: int,
) -> dict:
    """Apply one complete weighted 16×4 FP32 update and log the full gradient policy."""
    if group.ndim != 1 or not 0 < len(group) <= EFFECTIVE_BATCH:
        raise ValueError("Malformed effective batch")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    seen = 0
    for signal, target in DataLoader(
        Subset(dataset, group.tolist()), batch_size=MICROBATCH, shuffle=False, num_workers=0
    ):
        signal = signal.to(device)
        target = target.to(device)
        if signal.dtype != torch.float32:
            raise RuntimeError("Encoder-motion training must use FP32 waveforms")
        loss = nn.functional.binary_cross_entropy_with_logits(model(signal), target, reduction="sum")
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training BCE")
        (loss / len(group)).backward()
        loss_sum += float(loss.detach())
        seen += len(target)
    if seen != len(group):
        raise RuntimeError("Microbatch accumulation omitted records")
    encoder_params = list(model.backbone.parameters())
    head_params = list(model.head.parameters())
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    if missing != ["backbone.mask_token"]:
        raise RuntimeError(f"Unexpected absent full-model gradients: {missing}")
    encoder_norm = _gradient_norm(encoder_params)
    head_norm = _gradient_norm(head_params)
    global_norm = float(nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM))
    if not math.isfinite(global_norm):
        raise RuntimeError("Nonfinite preclip gradient norm")
    if abs(global_norm - math.hypot(encoder_norm, head_norm)) > 2e-4:
        raise RuntimeError("Global clip excluded encoder or head gradient")
    factor = min(1.0, CLIP_NORM / (global_norm + 1e-6))
    applied_lrs = {group_row["name"]: float(group_row["lr"]) for group_row in optimizer.param_groups}
    head_before = [parameter.detach().clone() for parameter in head_params]
    optimizer.step()
    scheduler.step()
    head_update = math.sqrt(
        sum(
            float(torch.sum((parameter.detach() - before) ** 2))
            for parameter, before in zip(head_params, head_before, strict=True)
        )
    )
    batch_ids = np.asarray(ecg_ids[group.numpy()], dtype=np.int64)
    return {
        "update_index": update_index,
        "updates_completed": update_index + 1,
        "records": len(group),
        "batch_ids_sha256": hashlib.sha256(batch_ids.tobytes()).hexdigest(),
        "mean_training_bce": loss_sum / len(group),
        "preclip_global_norm": global_norm,
        "preclip_encoder_norm": encoder_norm,
        "preclip_head_norm": head_norm,
        "unused_gradient_parameters": missing,
        "clip_factor": factor,
        "head_update_norm": head_update,
        "applied_learning_rates": applied_lrs,
    }


@torch.inference_mode()
def infer_features(model: nn.Module, dataset: Dataset, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract eval-mode raw pooled features and native head logits in row order."""
    prior_mode = model.training
    model.eval()
    features, logits = [], []
    for signal, _ in DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0):
        pooled, _ = model.backbone(signal.to(device))
        features.append(pooled.float().cpu().numpy())
        logits.append(model.head(pooled).squeeze(-1).float().cpu().numpy())
    model.train(prior_mode)
    return np.concatenate(features), np.concatenate(logits)


def fixed_diagnostic(
    model: nn.Module,
    data: Dataset,
    device: str,
    labels: np.ndarray,
    release_features: np.ndarray,
    original_weight: np.ndarray,
    original_bias: float,
    baseline_encoder: dict[str, torch.Tensor],
    baseline_head: dict[str, torch.Tensor],
    update: int,
) -> dict:
    """Evaluate only fixed training ECGs while restoring model mode and all RNG."""
    rng = flat_rng_state()
    mask = model.backbone.core.drop_path
    mask_rng = mask.get_rng_state()
    mask_draws = mask.draws
    current_features, current_logits = infer_features(model, data, device)
    restore_flat_rng_state(rng)
    mask.set_rng_state(mask_rng)
    mask.draws = mask_draws
    old_logits = current_features.astype(np.float64) @ original_weight + original_bias
    cosine = np.sum(current_features * release_features, axis=1) / (
        np.linalg.norm(current_features, axis=1) * np.linalg.norm(release_features, axis=1) + 1e-12
    )
    encoder_change = (
        sum(
            float(torch.sum((value.detach() - baseline_encoder[name]) ** 2))
            for name, value in model.backbone.state_dict().items()
        )
        ** 0.5
    )
    head_change = (
        sum(
            float(torch.sum((value.detach() - baseline_head[name]) ** 2))
            for name, value in model.head.state_dict().items()
        )
        ** 0.5
    )
    return {
        "update": update,
        "current_head_bce": float(np.mean(np.logaddexp(0, current_logits) - labels * current_logits)),
        "old_probe_bce": float(np.mean(np.logaddexp(0, old_logits) - labels * old_logits)),
        "mean_current_head_logit": float(np.mean(current_logits)),
        "mean_old_probe_logit": float(np.mean(old_logits)),
        "current_head_logits": current_logits.astype(np.float32).tolist(),
        "old_probe_logits": old_logits.astype(np.float64).tolist(),
        "mean_abs_current_logit_shift_from_release": float(
            np.mean(
                np.abs(
                    current_logits - (release_features.astype(np.float64) @ original_weight + original_bias)
                )
            )
        ),
        "mean_abs_old_probe_logit_shift_from_release": float(
            np.mean(
                np.abs(old_logits - (release_features.astype(np.float64) @ original_weight + original_bias))
            )
        ),
        "median_feature_cosine_to_release": float(np.median(cosine)),
        "encoder_parameter_change_norm": encoder_change,
        "head_parameter_change_norm": head_change,
        "encoder_bitwise_unchanged": encoder_equal(model, baseline_encoder),
        "rng_restored": True,
    }
