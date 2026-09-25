"""Matched within-view and disjoint-lead future forecasting for compact CPC."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812 - conventional PyTorch alias

from .cpc import LEADS, CPCEncoder, cpc_loss, prediction_heads

VARIANTS = ("native", "withinlead", "crosslead")
# Canonical I, II, III, aVR, aVL, aVF, V1..V6. Derived limb leads are
# excluded from both partial views, but retained in the ordinary full view.
GROUP_A = (0, 6, 8, 10)
GROUP_B = (1, 7, 9, 11)
AUX_WEIGHT = 0.1


def lead_view(normalized_signal: torch.Tensor, leads: Sequence[int]) -> torch.Tensor:
    """
    Keep only the given leads, zeroing the rest.

    Masking happens AFTER fixed training-set normalization: zero means missing here.

    Parameters
    ----------
    normalized_signal : torch.Tensor
        Normalized signals of shape [batch, 12, samples].
    leads : Sequence[int]
        Canonical lead indices to keep.

    Returns
    -------
    torch.Tensor
        Masked signals with the input shape.

    Raises
    ------
    ValueError
        If the signal is not twelve-lead.
    """
    if normalized_signal.ndim != 3 or normalized_signal.shape[1] != LEADS:
        raise ValueError("Expected normalized [batch,12,samples]")
    mask = normalized_signal.new_zeros(1, LEADS, 1)
    mask[:, list(leads)] = 1
    return normalized_signal * mask


def _unnormalized_token_variance(tokens: torch.Tensor) -> float:
    """Mean unbiased per-dimension variance of raw tokens."""
    return float(tokens.detach().flatten(0, 2).var(dim=0).mean())


@torch.no_grad()
def _diagnostics(full: torch.Tensor, a: torch.Tensor, b: torch.Tensor, z: torch.Tensor,
                 za: torch.Tensor, zb: torch.Tensor) -> dict[str, float]:
    """Detached losses and token-collapse statistics for the full and partial views."""
    normalized = F.normalize(z.detach(), dim=-1)
    adjacent_cosine = (normalized[:, :, 1:] * normalized[:, :, :-1]).sum(-1).mean()
    return {"full_cpc": float(full.detach()), "aux_a": float(a.detach()),
            "aux_b": float(b.detach()),
            "token_variance": _unnormalized_token_variance(z),
            "adjacent_token_cosine": float(adjacent_cosine),
            "view_a_token_variance": _unnormalized_token_variance(za),
            "view_b_token_variance": _unnormalized_token_variance(zb)}


class CrossLeadPretrainer(nn.Module):
    """Full-view CPC plus within-view or cross-view auxiliary forecasting."""

    def __init__(self, variant: str) -> None:
        """
        Build the shared encoder and three sets of prediction heads.

        Parameters
        ----------
        variant : str
            One of ``VARIANTS``.

        Raises
        ------
        ValueError
            If the variant is unknown.
        """
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown cross-lead variant: {variant}")
        self.variant = variant
        self.encoder = CPCEncoder()
        self.heads = prediction_heads()
        self.heads_a = prediction_heads()
        self.heads_b = prediction_heads()

    def load_bootstrap(self, encoder: dict[str, torch.Tensor], heads: dict[str, torch.Tensor]) -> None:
        """
        Load a native encoder and copy its heads into all three head sets.

        Parameters
        ----------
        encoder : dict[str, torch.Tensor]
            ``CPCEncoder`` state dict.
        heads : dict[str, torch.Tensor]
            State dict of a CPC head ``ModuleList``.
        """
        self.encoder.load_state_dict(encoder, strict=True)
        for group in (self.heads, self.heads_a, self.heads_b):
            group.load_state_dict(heads, strict=True)

    def forward(self, signal: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute the full-view loss plus the weighted auxiliary view losses.

        Parameters
        ----------
        signal : torch.Tensor
            Normalized signals of shape [batch, 12, 2500].

        Returns
        -------
        tuple[torch.Tensor, dict[str, float]]
            Scalar loss and detached diagnostics.
        """
        # Identical pass order, dropout calls, parameter shapes and data draws
        # in every arm. Native computes auxiliary diagnostics with zero weight.
        z, c = self.encoder(signal)
        za, ca = self.encoder(lead_view(signal, GROUP_A))
        zb, cb = self.encoder(lead_view(signal, GROUP_B))
        full = cpc_loss(z, c, self.heads)
        target_a, target_b = (zb, za) if self.variant == "crosslead" else (za, zb)
        a = cpc_loss(target_a, ca, self.heads_a)
        b = cpc_loss(target_b, cb, self.heads_b)
        auxiliary = (a + b) / 2
        loss = full + (0.0 if self.variant == "native" else AUX_WEIGHT) * auxiliary
        return loss, _diagnostics(full, a, b, z, za, zb)
