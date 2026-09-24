"""Matched within-view and disjoint-lead future forecasting for compact CPC."""

import torch
from torch import nn
from torch.nn import functional as F

from ecg_experiment.cpc import CPCEncoder, HORIZONS, cpc_loss

VARIANTS = ("native", "withinlead", "crosslead")
# Canonical I, II, III, aVR, aVL, aVF, V1..V6. Derived limb leads are
# excluded from both partial views, but retained in the ordinary full view.
GROUP_A = (0, 6, 8, 10)
GROUP_B = (1, 7, 9, 11)
AUX_WEIGHT = 0.1


def lead_view(normalized_signal, leads):
    """Mask AFTER fixed training-set normalization: zero means missing here."""
    if normalized_signal.ndim != 3 or normalized_signal.shape[1] != 12:
        raise ValueError("Expected normalized [batch,12,samples]")
    mask = normalized_signal.new_zeros(1, 12, 1)
    mask[:, list(leads)] = 1
    return normalized_signal * mask


class CrossLeadPretrainer(nn.Module):
    def __init__(self, variant):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown cross-lead variant: {variant}")
        self.variant = variant
        self.encoder = CPCEncoder()
        self.heads = nn.ModuleList(nn.Linear(256, 256, bias=False) for _ in HORIZONS)
        self.heads_a = nn.ModuleList(nn.Linear(256, 256, bias=False) for _ in HORIZONS)
        self.heads_b = nn.ModuleList(nn.Linear(256, 256, bias=False) for _ in HORIZONS)

    def load_bootstrap(self, encoder, heads):
        self.encoder.load_state_dict(encoder, strict=True)
        for group in (self.heads, self.heads_a, self.heads_b):
            group.load_state_dict(heads, strict=True)

    def forward(self, signal):
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
        with torch.no_grad():
            normalized = F.normalize(z.detach(), dim=-1)
            details = {"full_cpc": float(full.detach()), "aux_a": float(a.detach()),
                       "aux_b": float(b.detach()),
                       "token_variance": float(z.detach().flatten(0, 2).var(dim=0).mean()),
                       "adjacent_token_cosine": float((normalized[:, :, 1:] * normalized[:, :, :-1]).sum(-1).mean()),
                       "view_a_token_variance": float(za.detach().flatten(0, 2).var(dim=0).mean()),
                       "view_b_token_variance": float(zb.detach().flatten(0, 2).var(dim=0).mean())}
        return loss, details
