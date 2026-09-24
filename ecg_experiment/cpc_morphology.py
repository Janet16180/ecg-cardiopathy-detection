"""Experiment 017 causal local morphology branches for native-grid CPC."""

import torch
from torch import nn
from torch.nn import functional as F

from ecg_experiment.cpc import CPCEncoder


TEMPLATES = 32
LEADS = 12
SUPPORT = 50
HALF = 1250
TOKEN_STRIDE = 16
TOKEN_COUNT = 79
WINDOW_ELEMENTS = LEADS * SUPPORT


def native_windows(signal):
    """Independent five-second halves, matching CPCEncoder's input layout."""
    if signal.ndim != 3 or tuple(signal.shape[1:]) != (LEADS, 2500):
        raise ValueError("Expected native 250 Hz signals [batch,12,2500]")
    batch = len(signal)
    return signal.reshape(batch, LEADS, 2, HALF).permute(0, 2, 1, 3).reshape(batch * 2, LEADS, HALF)


def local_response(halves, bank, kind):
    """Mean dot or negative mean squared distance, causal and right aligned.

    At token t, only samples through right edge 16*t of its own half enter.
    Missing samples before the half start are zero padded in both arms.
    """
    if kind not in ("template", "conv"):
        raise ValueError(f"Unknown morphology branch: {kind}")
    if tuple(bank.shape) != (TEMPLATES, LEADS, SUPPORT):
        raise ValueError("Expected 32 twelve-lead templates of 50 samples")
    padded = F.pad(halves, (SUPPORT - 1, 0))
    dot = F.conv1d(padded, bank, stride=TOKEN_STRIDE) / WINDOW_ELEMENTS
    if kind == "conv":
        return 2 * dot
    # This exact expansion avoids materializing B x 79 x 32 x 12 x 50 windows.
    energy = F.conv1d(padded.square(), halves.new_ones(1, LEADS, SUPPORT),
                      stride=TOKEN_STRIDE) / WINDOW_ELEMENTS
    template_energy = bank.square().sum(dim=(1, 2)).view(1, TEMPLATES, 1) / WINDOW_ELEMENTS
    return -(energy - 2 * dot + template_energy)


class MorphologyCPCEncoder(CPCEncoder):
    """Native CPC with a zero-start local residual before the original GRU."""

    def __init__(self, kind="none", initial_bank=None):
        super().__init__()
        if kind not in ("none", "conv", "template"):
            raise ValueError(f"Unknown branch: {kind}")
        self.kind = kind
        if kind != "none":
            if initial_bank is None or tuple(initial_bank.shape) != (TEMPLATES, LEADS, SUPPORT):
                raise ValueError("Branched encoder needs 32 training-only windows [32,12,50]")
            self.bank = nn.Parameter(torch.as_tensor(initial_bank, dtype=torch.float32).clone())
            self.branch_hidden = nn.Linear(TEMPLATES, 64)
            self.branch_final = nn.Linear(64, 256)
            nn.init.zeros_(self.branch_final.weight)
            nn.init.zeros_(self.branch_final.bias)

    def forward(self, signal):
        halves = native_windows(signal)
        batch = len(signal)
        tokens = self.convs(halves).transpose(1, 2)
        if tokens.shape[1] != TOKEN_COUNT:
            raise AssertionError("Native CPC token grid changed")
        if self.kind != "none":
            response = local_response(halves, self.bank, self.kind).transpose(1, 2)
            if response.shape[1] != TOKEN_COUNT:
                raise AssertionError("Morphology token grid changed")
            residual = self.branch_final(F.silu(self.branch_hidden(response)))
            tokens = tokens + residual
        contexts, _ = self.context(tokens)
        return tokens.reshape(batch, 2, TOKEN_COUNT, 256), contexts.reshape(batch, 2, TOKEN_COUNT, 256)


class MorphologyCPCClassifier(nn.Module):
    def __init__(self, kind="none", initial_bank=None):
        super().__init__()
        self.encoder = MorphologyCPCEncoder(kind, initial_bank)
        self.head = nn.Linear(512, 1)

    def forward(self, signal):
        _, contexts = self.encoder(signal)
        return self.head(self.encoder.pooled(contexts)).squeeze(-1)
