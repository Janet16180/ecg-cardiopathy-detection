"""Seed-43 construction and gates for the one-epoch xECG mechanism screen."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from .reproducibility import seed_everything
from .xecg import XECGBinaryClassifier, load_xecg
from .xecg_droppath_rescue import install_droppath
from .xecg_rescue_training import layerwise_parameter_groups, make_scheduler

SEED = 43
EXECUTION_EPOCHS = 1
SCHEDULER_HORIZON_EPOCHS = 2
UPDATES = 240
HISTORICAL_COST_SECONDS = 1981.6532189800346
SLOW_PASS_SECONDS = 722.3458946699975
REPORT_ALLOWANCE_SECONDS = 300.0
COST_MARGIN = 1.25
COST_CEILING_SECONDS = 7200.0


def build_model_seed43(
    checkpoint_dir: Path, probe_path: Path, arm: str, device: str, train_records: int
) -> tuple:
    """Load identical seed-42 tensors, then isolate seed-43 training streams."""
    seed_everything(42)
    backbone = load_xecg(checkpoint_dir, backend="vanilla", device=device, drop_path_prob=0.5)
    mask = install_droppath(backbone, arm, seed=SEED)
    model = XECGBinaryClassifier(backbone).to(device)
    with np.load(probe_path) as probe, torch.no_grad():
        model.head.weight.copy_(
            torch.as_tensor(probe["raw_weight"], dtype=torch.float32, device=device).view(1, -1)
        )
        model.head.bias.fill_(float(probe["raw_bias"]))
    optimizer = torch.optim.AdamW(layerwise_parameter_groups(model), weight_decay=0.1)
    steps_per_epoch = math.ceil(train_records / 64)
    if steps_per_epoch != UPDATES:
        raise ValueError("One-epoch update count differs from frozen 240")
    scheduler = make_scheduler(optimizer, steps_per_epoch)
    seed_everything(SEED)
    permutation = torch.Generator(device="cpu").manual_seed(SEED)
    return model, mask, optimizer, scheduler, permutation


def warmup_multipliers() -> list[float]:
    """Return the first 240 multipliers of the unchanged two-epoch schedule."""
    return [step / UPDATES for step in range(UPDATES)]


def projected_cost(new_verification_seconds: float, completed_passes: list[float]) -> dict:
    """Conservatively replace completed inherited pass estimates with actual time."""
    if len(completed_passes) > 3 or new_verification_seconds < 0:
        raise ValueError("Invalid 016 v6 cost input")
    remaining = 3 - len(completed_passes)
    estimate = (
        HISTORICAL_COST_SECONDS
        + new_verification_seconds
        + COST_MARGIN * (sum(completed_passes) + remaining * SLOW_PASS_SECONDS + REPORT_ALLOWANCE_SECONDS)
    )
    return {
        "historical_preparation_and_profile_seconds": HISTORICAL_COST_SECONDS,
        "new_verification_seconds": new_verification_seconds,
        "completed_pass_seconds": completed_passes,
        "remaining_profiled_passes": remaining,
        "inherited_slowest_pass_seconds": SLOW_PASS_SECONDS,
        "report_allowance_seconds": REPORT_ALLOWANCE_SECONDS,
        "margin": COST_MARGIN,
        "projected_total_seconds": estimate,
        "ceiling_seconds": COST_CEILING_SECONDS,
        "passed": estimate <= COST_CEILING_SECONDS,
    }


def decisions(metrics: dict[str, dict], probe: dict, bootstrap: dict, inherited_shift: dict) -> dict:
    """Apply the frozen one-epoch point-estimate screens without model selection."""
    legacy = metrics["legacy"]
    residual = metrics["residual"]
    off = metrics["off"]
    delta = residual["auroc"] - legacy["auroc"]
    off_delta = off["auroc"] - legacy["auroc"]
    residual_off = residual["auroc"] - off["auroc"]
    shift_verified = inherited_shift["residual"] < inherited_shift["legacy"]
    support = delta >= 0.005 and shift_verified
    support_interval = bootstrap["contrasts"]["residual_minus_legacy"]["interval_95"]
    practical = []
    for arm, advantage in (("residual", delta), ("off", off_delta)):
        if (
            advantage >= 0.005
            and metrics[arm]["auroc"] >= probe["auroc"] - 0.002
            and probe["mean_fold_sensitivity"] - metrics[arm]["mean_fold_sensitivity"] <= 0.005
        ):
            practical.append(arm)
    preferred = None
    if practical:
        preferred = (
            "residual"
            if "residual" in practical and ("off" not in practical or residual_off >= 0.002)
            else "off"
        )
    return {
        "residual_minus_legacy": delta,
        "off_minus_legacy": off_delta,
        "residual_minus_off": residual_off,
        "inherited_initial_shift_verified": shift_verified,
        "mechanistic_support_point_screen": support,
        "mechanistic_interval_lower_above_zero": support_interval[0] > 0,
        "mechanistic_interpretation": (
            "positive point-estimate screen; sampling uncertainty unresolved"
            if support and support_interval[0] <= 0
            else "positive point-estimate screen with positive conditional interval"
            if support
            else "mechanistic point-estimate screen negative"
        ),
        "partial_if_residual_below_probe": support and residual["auroc"] < probe["auroc"] - 0.002,
        "practical_rescue_arms": practical,
        "preferred_if_any": preferred,
    }
