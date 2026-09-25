"""Experimental complete-block stochastic-depth controls for xECG 016.

The released model and its loader are untouched. Install this module only on a
freshly loaded backbone; evaluation always executes the complete block.
"""

from __future__ import annotations

import torch
from torch import nn


class PairedDropPath(nn.Module):
    """Draw per-record masks from an isolated, checkpointable RNG stream."""

    def __init__(self, arm: str, seed: int = 42, device: str | torch.device = "cpu") -> None:
        super().__init__()
        if arm not in {"legacy", "residual", "off"}:
            raise ValueError(f"Unknown DropPath arm: {arm}")
        self.arm = arm
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.draws = 0

    def get_rng_state(self) -> torch.Tensor:
        """Return the isolated mask stream state for checkpointing."""
        return self.generator.get_state().clone()

    def set_rng_state(self, state: torch.Tensor) -> None:
        """Restore the isolated mask stream state."""
        self.generator.set_state(state.cpu())

    def forward(self, x: torch.Tensor, block: nn.Module, probability: float) -> torch.Tensor:
        """Apply the selected whole-block rule to a per-record mask."""
        if probability == 0 or not self.training:
            return block(x)
        if not 0 < probability < 1:
            raise ValueError("DropPath probability must lie in [0, 1)")
        keep = 1 - probability
        mask = torch.rand(x.shape[0], generator=self.generator, device=x.device) < keep
        mask = mask.view(x.shape[0], *([1] * (x.ndim - 1)))
        self.draws += x.shape[0]
        complete = block(x)
        if self.arm == "legacy":
            return torch.where(mask, complete / keep, x)
        if self.arm == "residual":
            return x + mask * ((complete - x) / keep)
        return complete


def install_droppath(backbone: nn.Module, arm: str, seed: int = 42) -> PairedDropPath:
    """Replace only the freshly constructed backbone's DropPath wrapper."""
    core = backbone.core
    if len(core.model.blocks) != 9 or len(core.dropout_rates) != 9:
        raise ValueError("Expected the nine-block released xECG backbone")
    wrapper = PairedDropPath(arm, seed, next(backbone.parameters()).device)
    core.drop_path = wrapper
    if arm == "off":
        core.dropout_rates = [0.0] * 9
    return wrapper
