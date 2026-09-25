"""Experiment 017 causal local morphology branches for native-grid CPC."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812 - conventional PyTorch alias

from .cpc import LEADS, TOKEN_COUNT, WIDTH, CPCEncoder, check_signal_batch, split_halves

TEMPLATES = 32
SUPPORT = 50
TOKEN_STRIDE = 16
WINDOW_ELEMENTS = LEADS * SUPPORT
BRANCH_HIDDEN = 64
RESPONSE_KINDS = ("template", "conv")
BRANCH_KINDS = ("none", *RESPONSE_KINDS)


def local_response(halves: torch.Tensor, bank: torch.Tensor, kind: str) -> torch.Tensor:
    """
    Mean dot or negative mean squared distance, causal and right aligned.

    At token t, only samples through right edge 16*t of its own half enter.
    Missing samples before the half start are zero padded in both arms.

    Parameters
    ----------
    halves : torch.Tensor
        Halves of shape [records, 12, 1250].
    bank : torch.Tensor
        Templates of shape [32, 12, 50].
    kind : str
        ``"template"`` (negative mean squared distance) or ``"conv"`` (twice the mean dot).

    Returns
    -------
    torch.Tensor
        Responses of shape [records, 32, 79].

    Raises
    ------
    ValueError
        If ``kind`` is unknown or the bank shape is wrong.
    """
    if kind not in RESPONSE_KINDS:
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

    def __init__(self, kind: str = "none", initial_bank: np.ndarray | torch.Tensor | None = None) -> None:
        """
        Build the encoder and, for a branched arm, its morphology residual.

        Parameters
        ----------
        kind : str
            ``"none"``, ``"conv"`` or ``"template"``.
        initial_bank : np.ndarray | torch.Tensor | None
            Training-only initial windows [32, 12, 50]; required unless ``kind`` is ``"none"``.

        Raises
        ------
        ValueError
            If ``kind`` is unknown or a branched arm lacks a correctly shaped bank.
        """
        super().__init__()
        if kind not in BRANCH_KINDS:
            raise ValueError(f"Unknown branch: {kind}")
        self.kind = kind
        if kind == "none":
            return
        if initial_bank is None or tuple(initial_bank.shape) != (TEMPLATES, LEADS, SUPPORT):
            raise ValueError("Branched encoder needs 32 training-only windows [32,12,50]")
        self.bank = nn.Parameter(torch.as_tensor(initial_bank, dtype=torch.float32).clone())
        self.branch_hidden = nn.Linear(TEMPLATES, BRANCH_HIDDEN)
        self.branch_final = nn.Linear(BRANCH_HIDDEN, WIDTH)
        nn.init.zeros_(self.branch_final.weight)
        nn.init.zeros_(self.branch_final.bias)

    def forward(self, signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode both halves, adding the morphology residual to the CNN tokens.

        Parameters
        ----------
        signal : torch.Tensor
            Normalized signals of shape [batch, 12, 2500].

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tokens and contexts, each of shape [batch, 2, 79, 256].

        Raises
        ------
        ValueError
            If the signal shape is not [batch, 12, 2500].
        RuntimeError
            If the CNN or morphology token grid differs from 79 steps.
        """
        check_signal_batch(signal)
        halves = split_halves(signal)
        batch = len(signal)
        tokens = self.convs(halves).transpose(1, 2)
        if tokens.shape[1] != TOKEN_COUNT:
            raise RuntimeError("Native CPC token grid changed")
        if self.kind != "none":
            response = local_response(halves, self.bank, self.kind).transpose(1, 2)
            if response.shape[1] != TOKEN_COUNT:
                raise RuntimeError("Morphology token grid changed")
            residual = self.branch_final(F.silu(self.branch_hidden(response)))
            tokens = tokens + residual
        contexts, _ = self.context(tokens)
        return (tokens.reshape(batch, 2, TOKEN_COUNT, WIDTH),
                contexts.reshape(batch, 2, TOKEN_COUNT, WIDTH))


class MorphologyCPCClassifier(nn.Module):
    """Linear binary head on pooled morphology-CPC contexts."""

    def __init__(self, kind: str = "none", initial_bank: np.ndarray | torch.Tensor | None = None) -> None:
        """
        Build the classifier.

        Parameters
        ----------
        kind : str
            Morphology branch kind passed to ``MorphologyCPCEncoder``.
        initial_bank : np.ndarray | torch.Tensor | None
            Initial template bank for a branched arm.
        """
        super().__init__()
        self.encoder = MorphologyCPCEncoder(kind, initial_bank)
        self.head = nn.Linear(2 * WIDTH, 1)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Predict one logit per record.

        Parameters
        ----------
        signal : torch.Tensor
            Normalized signals of shape [batch, 12, 2500].

        Returns
        -------
        torch.Tensor
            Logits of shape [batch].
        """
        _, contexts = self.encoder(signal)
        return self.head(self.encoder.pooled(contexts)).squeeze(-1)
