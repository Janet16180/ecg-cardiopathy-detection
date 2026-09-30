"""Causal CPC with the original xLSTM mLSTM residual blocks.

Equations and architecture: Beck et al., arXiv:2405.04517; official reference:
https://github.com/NX-AI/xlstm/tree/ab22eadbd245f293dd8dec38ed29963d73758a12
This independent PyTorch implementation follows the parallel stabilized backend,
blockwise Q/K/V, depthwise convolution, gated skip and residual-weight norms.
The CPC adaptation fixes two width-256 blocks, expansion two, four memory heads,
projection blocks of four, and residual dropout 0.1. No custom CUDA is needed.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from ecg_experiment.cpc import WIDTH, CPCPretrainer

OFFICIAL_COMMIT = "ab22eadbd245f293dd8dec38ed29963d73758a12"
BLOCK_COUNT = 2
INNER_WIDTH = 2 * WIDTH
HEAD_COUNT = 4
PROJECTION_BLOCK = 4
CONV_KERNEL = 4
DROPOUT = 0.1


def mlstm_parallel(queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                   input_gate: torch.Tensor, forget_gate: torch.Tensor,
                   eps: float = 1e-6) -> torch.Tensor:
    """
    Retrieve causal matrix-memory values with log-space gate stabilization.

    Parameters
    ----------
    queries, keys, values : torch.Tensor
        Projected tensors [batch, heads, time, head_width].
    input_gate, forget_gate : torch.Tensor
        Gate preactivations [batch, heads, time].
    eps : float
        Additive denominator stabilizer from the official implementation.

    Returns
    -------
    torch.Tensor
        Retrieved values with the same shape as the projected tensors.
    """
    length, head_width = queries.shape[-2:]
    cumulative = F.logsigmoid(forget_gate).cumsum(dim=-1)
    log_decay = cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2)
    log_decay = log_decay + input_gate.unsqueeze(-2)
    causal = torch.ones(length, length, dtype=torch.bool, device=queries.device).tril()
    log_decay = log_decay.masked_fill(~causal, -torch.inf)
    maximum = log_decay.amax(dim=-1, keepdim=True)
    decay = (log_decay - maximum).exp()
    combination = (queries @ (keys / math.sqrt(head_width)).transpose(-1, -2)) * decay
    normalizer = torch.maximum(combination.sum(dim=-1, keepdim=True).abs(), (-maximum).exp())
    return (combination / (normalizer + eps)) @ values


class ResidualLayerNorm(nn.Module):
    """Layer normalization with the official zero-initialized residual scale."""

    def __init__(self, width: int) -> None:
        """
        Build bias-free normalization over the final dimension.

        Parameters
        ----------
        width : int
            Number of independently scaled features.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize each token independently without mixing time positions.

        Parameters
        ----------
        x : torch.Tensor
            Tokens with the configured feature width in the final dimension.

        Returns
        -------
        torch.Tensor
            Normalized tokens with the input shape.
        """
        return F.layer_norm(x, (x.shape[-1],), 1 + self.weight, None, 1e-5)


class BlockwiseLinear(nn.Module):
    """Independent square projections over contiguous groups of four features."""

    def __init__(self, width: int = INNER_WIDTH) -> None:
        """
        Create the block-diagonal projection with xLSTM initialization.

        Parameters
        ----------
        width : int
            Feature width, divisible by the fixed projection block size.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(width // PROJECTION_BLOCK,
                                               PROJECTION_BLOCK, PROJECTION_BLOCK))
        nn.init.normal_(self.weight, std=math.sqrt(2 / (5 * WIDTH)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project groups without mixing groups or time positions.

        Parameters
        ----------
        x : torch.Tensor
            Tokens with the configured feature width in the final dimension.

        Returns
        -------
        torch.Tensor
            Projected tokens with the input shape.
        """
        groups = x.reshape(*x.shape[:-1], -1, PROJECTION_BLOCK)
        return torch.einsum("...gi,goi->...go", groups, self.weight).flatten(-2)


class MLSTMBlock(nn.Module):
    """Original pre-normalized mLSTM residual block with a gated inner skip."""

    def __init__(self) -> None:
        """Build the fixed width-256, four-head mLSTM architecture."""
        super().__init__()
        self.norm = ResidualLayerNorm(WIDTH)
        self.up = nn.Linear(WIDTH, 2 * INNER_WIDTH, bias=False)
        self.conv = nn.Conv1d(INNER_WIDTH, INNER_WIDTH, CONV_KERNEL, groups=INNER_WIDTH)
        self.query = BlockwiseLinear()
        self.key = BlockwiseLinear()
        self.value = BlockwiseLinear()
        self.input_gate = nn.Linear(3 * INNER_WIDTH, HEAD_COUNT)
        self.forget_gate = nn.Linear(3 * INNER_WIDTH, HEAD_COUNT)
        self.head_scale = nn.Parameter(torch.zeros(HEAD_COUNT, INNER_WIDTH // HEAD_COUNT))
        self.skip = nn.Parameter(torch.ones(INNER_WIDTH))
        self.down = nn.Linear(INNER_WIDTH, WIDTH, bias=False)
        self.dropout = nn.Dropout(DROPOUT)
        nn.init.normal_(self.up.weight, std=math.sqrt(2 / (5 * WIDTH)))
        nn.init.normal_(self.down.weight, std=2 / BLOCK_COUNT / math.sqrt(WIDTH))
        nn.init.zeros_(self.input_gate.weight)
        nn.init.normal_(self.input_gate.bias, std=0.1)
        nn.init.zeros_(self.forget_gate.weight)
        with torch.no_grad():
            self.forget_gate.bias.copy_(torch.linspace(3, 6, HEAD_COUNT))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply one causal residual block.

        Parameters
        ----------
        x : torch.Tensor
            Tokens of shape [batch, time, 256].

        Returns
        -------
        torch.Tensor
            Updated causal tokens with the input shape.
        """
        batch, length, _ = x.shape
        inner, output_gate = self.up(self.norm(x)).chunk(2, dim=-1)
        convolved = F.silu(self.conv(F.pad(inner.transpose(1, 2), (CONV_KERNEL - 1, 0))))
        convolved = convolved.transpose(1, 2)
        query, key, value = self.query(convolved), self.key(convolved), self.value(inner)
        gate_input = torch.cat((query, key, value), dim=-1)
        input_gate = self.input_gate(gate_input).transpose(1, 2)
        forget_gate = self.forget_gate(gate_input).transpose(1, 2)
        query = query.reshape(batch, length, HEAD_COUNT, -1).transpose(1, 2)
        key = key.reshape(batch, length, HEAD_COUNT, -1).transpose(1, 2)
        value = value.reshape(batch, length, HEAD_COUNT, -1).transpose(1, 2)
        memory = mlstm_parallel(query, key, value, input_gate, forget_gate)
        memory = F.layer_norm(memory, (INNER_WIDTH // HEAD_COUNT,), eps=1e-5)
        memory = memory * (1 + self.head_scale[None, :, None, :])
        memory = memory.transpose(1, 2).reshape(batch, length, INNER_WIDTH)
        gated = (memory + self.skip * convolved) * F.silu(output_gate)
        return x + self.dropout(self.down(gated))


class MLSTMContext(nn.Module):
    """Two mLSTM blocks and post-normalization with the existing GRU interface."""

    def __init__(self) -> None:
        """Build the fixed context stack, starting each forward with zero memory."""
        super().__init__()
        self.blocks = nn.Sequential(*(MLSTMBlock() for _ in range(BLOCK_COUNT)))
        self.post_norm = ResidualLayerNorm(WIDTH)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, None]:
        """
        Return contexts and a placeholder state for CPCEncoder compatibility.

        Parameters
        ----------
        tokens : torch.Tensor
            CNN tokens of shape [batch, time, 256].

        Returns
        -------
        tuple[torch.Tensor, None]
            Causal contexts with the input shape and no retained memory state.
        """
        return self.post_norm(self.blocks(tokens)), None


def create_model(arm: str, seed: int, device: torch.device | str = "cpu") -> CPCPretrainer:
    """
    Build a CPC arm with exactly matched initial CNN and prediction-head tensors.

    Parameters
    ----------
    arm : str
        ``gru`` for the existing architecture or ``xlstm`` for two mLSTM blocks.
    seed : int
        Common initialization seed; the new context uses seed plus one.
    device : torch.device or str
        Target device after deterministic CPU construction.

    Returns
    -------
    CPCPretrainer
        Ordinary CPC model using unchanged encoder and loss interfaces.

    Raises
    ------
    ValueError
        If the arm name is unknown.
    """
    if arm not in {"gru", "xlstm"}:
        raise ValueError(f"Unknown CPC arm: {arm}")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = CPCPretrainer(hybrid=False)
        if arm == "xlstm":
            torch.random.default_generator.manual_seed(seed + 1)
            model.encoder.context = MLSTMContext()
    return model.to(device)
