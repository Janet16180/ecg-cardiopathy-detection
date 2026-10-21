"""Experiment 043: an xECG-style encoder on every open EchoNext ECG, then unsupervised clusters.

xECG (Lunelli et al., arXiv:2509.10151) is a stack of sLSTM and mLSTM blocks that reads an ECG as a
sequence of time patches, in both directions, and is pretrained without labels with SimDINOv2: a student
network learns to give the same description as a slowly moving teacher for different crops of one ECG.

This module is a compact version written from the papers for a laptop. It keeps the patch tokens, both
xLSTM block types, the alternating directions, the six views, the patch masking and the three loss terms.
The protocol lists every difference from the published model.

Data loading, the float16 cache, the closed ``test`` split and the clustering come from
``echonext_signal_clusters``.
"""

from __future__ import annotations

import copy
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .echonext_signal_clusters import LEADS, SAMPLES, SignalDataset, epoch_order, save_artifacts

PATCH = 50
KERNEL = 4
GLOBAL_VIEWS = 2
LOCAL_VIEWS = 4
GLOBAL_SAMPLES = 2000
LOCAL_SAMPLES = 1000
MASK_PROBABILITY = 0.3
LEAD_DROPOUT = 0.2
KEPT_LEAD = 1
JITTER_PROBABILITY = 0.1
JITTER_AMPLITUDE = 0.6
SCALING_PROBABILITY = 0.1
SCALING_RANGE = 0.2
EPSILON = 0.5
MOMENTUM_START = 0.99
GRADIENT_NORM = 3.0


def patchify(signals: torch.Tensor) -> torch.Tensor:
    """
    Cut waveforms into consecutive time patches that hold every lead.

    Parameters
    ----------
    signals : torch.Tensor
        Shape ``(batch, 12, samples)``; ``samples`` is a multiple of 50.

    Returns
    -------
    torch.Tensor
        Shape ``(batch, samples / 50, 600)``: each patch is 12 leads x 50 samples, flattened.
    """
    batch, leads, samples = signals.shape
    patches = signals.reshape(batch, leads, samples // PATCH, PATCH)
    return patches.permute(0, 2, 1, 3).reshape(batch, samples // PATCH, leads * PATCH)


def mlstm_memory(queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, input_raw: torch.Tensor,
                 forget_raw: torch.Tensor) -> torch.Tensor:
    """
    Read the mLSTM matrix memory at every step, computed for all steps at once.

    The recurrence stores ``C_t = f_t C_{t-1} + i_t v_t k_t^T`` and reads ``C_t q_t / max(|n_t . q_t|, 1)``
    with ``i_t = exp(input_raw)``, ``f_t = sigmoid(forget_raw)`` and ``n_t = f_t n_{t-1} + i_t k_t``.
    Unrolled, step ``t`` reads every earlier step ``s`` with weight ``i_s f_{s+1} ... f_t``, which is
    evaluated here in log space so the exponential gate cannot overflow.

    Parameters
    ----------
    queries, keys, values : torch.Tensor
        Shape ``(batch, heads, steps, size)``.
    input_raw, forget_raw : torch.Tensor
        Gate pre-activations of shape ``(batch, heads, steps)``.

    Returns
    -------
    torch.Tensor
        Shape ``(batch, heads, steps, size)``.
    """
    steps, size = queries.shape[-2:]
    kept = torch.cumsum(nn.functional.logsigmoid(forget_raw), dim=-1)
    log_weight = kept.unsqueeze(-1) - kept.unsqueeze(-2) + input_raw.unsqueeze(-2)
    past = torch.ones(steps, steps, dtype=torch.bool, device=queries.device).tril()
    log_weight = log_weight.masked_fill(~past, float("-inf"))
    largest = log_weight.max(dim=-1, keepdim=True).values
    scores = queries @ keys.transpose(-2, -1) / math.sqrt(size) * torch.exp(log_weight - largest)
    normalizer = torch.maximum(scores.sum(dim=-1, keepdim=True).abs(), torch.exp(-largest))
    return scores / (normalizer + 1e-6) @ values


def slstm_states(inputs: torch.Tensor, recurrent: torch.Tensor) -> torch.Tensor:
    """
    Run the sLSTM recurrence step by step.

    Each step computes ``c_t = f_t c_{t-1} + i_t tanh(z_t)``, ``n_t = f_t n_{t-1} + i_t`` and
    ``h_t = o_t c_t / n_t`` with an exponential input gate. The gates also receive the previous ``h``, so
    the steps cannot be computed in parallel. A running maximum keeps the exponentials finite.

    Parameters
    ----------
    inputs : torch.Tensor
        Input part of the pre-activations, shape ``(batch, steps, heads, 4, size)``; the four slots are
        the input gate, forget gate, cell input and output gate.
    recurrent : torch.Tensor
        Weights from the previous hidden state, shape ``(heads, size, 4 * size)``: each head only sees
        its own state.

    Returns
    -------
    torch.Tensor
        Hidden states of shape ``(batch, steps, heads, size)``.
    """
    batch, steps, heads, _, size = inputs.shape
    hidden = inputs.new_zeros(heads, batch, size)
    cell = normalizer = stabilizer = hidden
    states = []
    for step, step_input in enumerate(inputs.permute(1, 2, 0, 3, 4).unbind(dim=0)):
        feedback = (hidden @ recurrent).view(heads, batch, 4, size)
        input_raw, forget_raw, cell_raw, output_raw = (step_input + feedback).unbind(dim=2)
        if step == 0:
            stabilizer, cell, normalizer = input_raw, torch.tanh(cell_raw), torch.ones_like(input_raw)
        else:
            log_forget = stabilizer + nn.functional.logsigmoid(forget_raw)
            stabilizer = torch.maximum(input_raw, log_forget)
            input_gate, forget_gate = torch.exp(input_raw - stabilizer), torch.exp(log_forget - stabilizer)
            cell = forget_gate * cell + input_gate * torch.tanh(cell_raw)
            normalizer = forget_gate * normalizer + input_gate
        hidden = torch.sigmoid(output_raw) * cell / normalizer
        states.append(hidden)
    return torch.stack(states, dim=0).permute(2, 0, 1, 3)


class CausalMix(nn.Module):
    """
    Mix each token with the three before it, feature by feature, and apply SiLU.

    It is the causal depthwise convolution of the xLSTM blocks, written as a weighted sum of four shifted
    copies of the sequence.

    Parameters
    ----------
    channels : int
        Features per token.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(KERNEL, channels) * KERNEL ** -0.5)
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Mix the tokens.

        Parameters
        ----------
        tokens : torch.Tensor
            Shape ``(batch, steps, channels)``.

        Returns
        -------
        torch.Tensor
            Same shape; step ``t`` depends on steps ``t - 3`` to ``t`` only.
        """
        steps = tokens.shape[1]
        padded = nn.functional.pad(tokens, (0, 0, KERNEL - 1, 0))
        mixed = sum(self.weight[lag] * padded[:, lag:lag + steps] for lag in range(KERNEL))
        return nn.functional.silu(mixed + self.bias)


class SLSTMBlock(nn.Module):
    """
    sLSTM block: scalar memory cells with exponential gating, then a small feed-forward layer.

    On an Apple GPU the step-by-step recurrence runs on the CPU: it is thousands of tiny operations in
    sequence, and the fixed cost the GPU pays per operation is larger than the work itself.

    Parameters
    ----------
    width : int
        Features per token.
    heads : int
        Groups of cells; the recurrent weights connect cells of the same head only.
    """

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        size = width // heads
        self.heads = heads
        self.norm = nn.LayerNorm(width)
        self.mix = CausalMix(width)
        self.input_forget = nn.Linear(width, 2 * width)
        self.cell_output = nn.Linear(width, 2 * width)
        self.recurrent = nn.Parameter(torch.randn(heads, size, 4 * size) * 0.02)
        self.state_norm = nn.GroupNorm(heads, width)
        self.mlp_norm = nn.LayerNorm(width)
        hidden = round(width * 4 / 3)
        self.mlp = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width))
        with torch.no_grad():
            self.input_forget.bias[:width].zero_()
            self.input_forget.bias[width:].copy_(torch.linspace(3.0, 6.0, width))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Update the tokens from first to last.

        Parameters
        ----------
        tokens : torch.Tensor
            Shape ``(batch, steps, width)``.

        Returns
        -------
        torch.Tensor
            Same shape.
        """
        batch, steps, width = tokens.shape
        normed = self.norm(tokens)
        gates = torch.cat([self.input_forget(self.mix(normed)), self.cell_output(normed)], dim=-1)
        inputs = gates.view(batch, steps, 4, self.heads, width // self.heads).transpose(2, 3)
        loop = torch.device("cpu") if tokens.device.type == "mps" else tokens.device
        hidden = slstm_states(inputs.to(loop), self.recurrent.to(loop)).to(tokens.device)
        hidden = hidden.reshape(batch * steps, width)
        tokens = tokens + self.state_norm(hidden).view(batch, steps, width)
        return tokens + self.mlp(self.mlp_norm(tokens))


class MLSTMBlock(nn.Module):
    """
    mLSTM block: widen the tokens, read a matrix memory with queries, keys and values, narrow them back.

    Parameters
    ----------
    width : int
        Features per token.
    heads : int
        Independent memories.
    """

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        inner = 2 * width
        self.heads = heads
        self.norm = nn.LayerNorm(width)
        self.up = nn.Linear(width, 2 * inner)
        self.mix = CausalMix(inner)
        self.query = nn.Linear(inner, inner)
        self.key = nn.Linear(inner, inner)
        self.value = nn.Linear(inner, inner)
        self.gates = nn.Linear(3 * inner, 2 * heads)
        self.state_norm = nn.GroupNorm(heads, inner)
        self.skip = nn.Parameter(torch.ones(inner))
        self.down = nn.Linear(inner, width)
        with torch.no_grad():
            self.gates.weight.zero_()
            self.gates.bias[:heads].zero_()
            self.gates.bias[heads:].copy_(torch.linspace(3.0, 6.0, heads))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Update the tokens from first to last.

        Parameters
        ----------
        tokens : torch.Tensor
            Shape ``(batch, steps, width)``.

        Returns
        -------
        torch.Tensor
            Same shape.
        """
        batch, steps, _ = tokens.shape
        content, gate = self.up(self.norm(tokens)).chunk(2, dim=-1)
        mixed = self.mix(content)
        query, key, value = self.query(mixed), self.key(mixed), self.value(content)
        gates = self.gates(torch.cat([query, key, value], dim=-1)).transpose(1, 2)
        input_raw, forget_raw = gates.chunk(2, dim=1)
        query, key, value = (part.view(batch, steps, self.heads, -1).transpose(1, 2)
                             for part in (query, key, value))
        hidden = mlstm_memory(query, key, value, input_raw, forget_raw).transpose(1, 2)
        hidden = self.state_norm(hidden.reshape(batch * steps, -1)).view(batch, steps, -1)
        return tokens + self.down((hidden + self.skip * mixed) * nn.functional.silu(gate))


class XLSTMEncoder(nn.Module):
    """
    Patch embedding and a stack of xLSTM blocks that alternate reading direction.

    Blocks at even positions read the patches from first to last; blocks at odd positions read them from
    last to first. No position is added: a recurrent block already knows the order.

    Parameters
    ----------
    width : int
        Features per token.
    heads : int
        Heads of every block.
    pattern : str
        One letter per block, ``"s"`` for sLSTM and ``"m"`` for mLSTM.
    """

    def __init__(self, width: int, heads: int, pattern: str) -> None:
        super().__init__()
        kinds = {"s": SLSTMBlock, "m": MLSTMBlock}
        self.embed = nn.Linear(LEADS * PATCH, width)
        self.mask_token = nn.Parameter(torch.zeros(width))
        self.blocks = nn.ModuleList(kinds[kind](width, heads) for kind in pattern)
        self.norm = nn.LayerNorm(width)

    def forward(self, signals: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Encode waveforms into one contextual token per patch.

        Parameters
        ----------
        signals : torch.Tensor
            Shape ``(batch, 12, samples)``.
        mask : torch.Tensor or None
            Boolean ``(batch, patches)``; true patches are replaced by the learned mask token.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, patches, width)``.
        """
        tokens = self.embed(patchify(signals))
        if mask is not None:
            tokens = torch.where(mask.unsqueeze(-1), self.mask_token, tokens)
        for position, block in enumerate(self.blocks):
            tokens = block(tokens.flip(1)).flip(1) if position % 2 else block(tokens)
        return self.norm(tokens)


class XECGNetwork(nn.Module):
    """
    Encoder plus the two parts used only during pretraining: attention pooling and a projection head.

    Parameters
    ----------
    width : int
        Features per token and size of the ECG embedding.
    heads : int
        Heads of every xLSTM block.
    pattern : str
        Block types in order; the published model uses ``"ssmmssmms"``.
    feature_size : int
        Size of the normalized features compared by the loss.
    """

    def __init__(self, width: int = 96, heads: int = 4, pattern: str = "ssmm",
                 feature_size: int = 64) -> None:
        super().__init__()
        self.config = {"width": width, "heads": heads, "pattern": pattern, "feature_size": feature_size}
        self.encoder = XLSTMEncoder(width, heads, pattern)
        self.attention = nn.Sequential(nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))
        self.head = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, feature_size))

    def forward(self, signals: torch.Tensor, mask: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Describe each view and each of its patches with unit-length features.

        Parameters
        ----------
        signals : torch.Tensor
            Shape ``(batch, 12, samples)``.
        mask : torch.Tensor or None
            Boolean ``(batch, patches)`` of patches hidden from the encoder.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            View features ``(batch, feature_size)`` and patch features ``(batch, patches, feature_size)``.
        """
        tokens = self.encoder(signals, mask)
        pooled = (self.attention(tokens).softmax(dim=1) * tokens).sum(dim=1)
        return (nn.functional.normalize(self.head(pooled), dim=-1),
                nn.functional.normalize(self.head(tokens), dim=-1))

    def embed(self, signals: torch.Tensor) -> torch.Tensor:
        """
        Embed whole ECGs as the mean of their tokens, without the pretraining parts.

        Parameters
        ----------
        signals : torch.Tensor
            Shape ``(batch, 12, samples)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, width)``.
        """
        return self.encoder(signals).mean(dim=1)


def make_teacher(student: XECGNetwork) -> XECGNetwork:
    """
    Copy the student into a teacher that is never trained by gradients.

    Parameters
    ----------
    student : XECGNetwork
        The network being trained.

    Returns
    -------
    XECGNetwork
        A copy with the same weights and gradients disabled.
    """
    return copy.deepcopy(student).requires_grad_(False)


@torch.no_grad()
def update_teacher(teacher: XECGNetwork, student: XECGNetwork, momentum: float) -> None:
    """
    Move every teacher weight a small step towards the student weight.

    Parameters
    ----------
    teacher, student : XECGNetwork
        Networks with the same layout.
    momentum : float
        Share of the old teacher weight that is kept.
    """
    for old, new in zip(teacher.parameters(), student.parameters(), strict=True):
        old.mul_(momentum).add_(new, alpha=1 - momentum)


def random_crops(signals: torch.Tensor, samples: int) -> torch.Tensor:
    """
    Cut one window per ECG at a random start.

    Parameters
    ----------
    signals : torch.Tensor
        Shape ``(batch, 12, 2500)``.
    samples : int
        Window length.

    Returns
    -------
    torch.Tensor
        Shape ``(batch, 12, samples)``.
    """
    batch = signals.shape[0]
    start = torch.randint(0, SAMPLES - samples + 1, (batch, 1, 1), device=signals.device)
    index = start + torch.arange(samples, device=signals.device)
    return signals.gather(2, index.expand(batch, LEADS, samples))


def augment(crops: torch.Tensor) -> torch.Tensor:
    """
    Apply the xECG augmentations that need one ECG only.

    Each lead except II is zeroed with probability 0.2. With probability 0.1 the crop is multiplied by
    ``1 + 0.6 * noise``, and with probability 0.1 it is scaled by a factor between 0.9 and 1.1.

    Parameters
    ----------
    crops : torch.Tensor
        Shape ``(batch, 12, samples)``.

    Returns
    -------
    torch.Tensor
        Same shape.
    """
    batch, device = crops.shape[0], crops.device
    kept = torch.rand(batch, LEADS, 1, device=device) >= LEAD_DROPOUT
    kept[:, KEPT_LEAD] = True
    crops = crops * kept
    jittered = torch.rand(batch, 1, 1, device=device) < JITTER_PROBABILITY
    crops = torch.where(jittered, crops * (1 + JITTER_AMPLITUDE * torch.randn_like(crops)), crops)
    scaled = torch.rand(batch, 1, 1, device=device) < SCALING_PROBABILITY
    factor = 1 + SCALING_RANGE * (torch.rand(batch, 1, 1, device=device) - 0.5)
    return torch.where(scaled, crops * factor, crops)


def make_views(signals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build two global and four local augmented views of every ECG.

    Parameters
    ----------
    signals : torch.Tensor
        Shape ``(batch, 12, 2500)``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Global views ``(2 * batch, 12, 2000)`` and local views ``(4 * batch, 12, 1000)``; view ``v`` of
        ECG ``b`` is row ``v * batch + b``.
    """
    global_views = [augment(random_crops(signals, GLOBAL_SAMPLES)) for _ in range(GLOBAL_VIEWS)]
    local_views = [augment(random_crops(signals, LOCAL_SAMPLES)) for _ in range(LOCAL_VIEWS)]
    return torch.cat(global_views), torch.cat(local_views)


def coding_rate(features: torch.Tensor, epsilon: float = EPSILON) -> torch.Tensor:
    """
    Measure how much of the feature space a batch occupies.

    ``R = 1/2 logdet(I + d / epsilon^2 * Cov)``. It is 0 when every ECG gets the same feature and largest
    when the features spread equally in all ``d`` directions. The determinant is computed on the CPU in
    float64, because the Apple GPU supports neither.

    Parameters
    ----------
    features : torch.Tensor
        Unit-length features of shape ``(batch, d)``.
    epsilon : float
        Precision of the rate; smaller values reward spreading more.

    Returns
    -------
    torch.Tensor
        Float64 scalar on the CPU.
    """
    batch, size = features.shape
    centered = features - features.mean(dim=0, keepdim=True)
    covariance = (centered.T @ centered / batch).cpu().double()
    matrix = torch.eye(size, dtype=torch.float64) + size / epsilon ** 2 * covariance
    return torch.linalg.cholesky(matrix).diagonal().log().sum()


def simdino_loss(student_global: torch.Tensor, student_patches: torch.Tensor, student_local: torch.Tensor,
                 teacher_global: torch.Tensor, teacher_patches: torch.Tensor, mask: torch.Tensor
                 ) -> dict[str, torch.Tensor]:
    """
    SimDINOv2 loss: agree with the teacher on views and on hidden patches, and do not collapse.

    Distances are ``1/2 |a - b|^2 = 1 - cosine`` between unit-length features.

    Parameters
    ----------
    student_global : torch.Tensor
        Student features of the masked global views, ``(2 * batch, d)``.
    student_patches : torch.Tensor
        Student patch features of the masked global views, ``(2 * batch, patches, d)``.
    student_local : torch.Tensor
        Student features of the local views, ``(4 * batch, d)``.
    teacher_global : torch.Tensor
        Teacher features of the same global views without masking, ``(2 * batch, d)``.
    teacher_patches : torch.Tensor
        Teacher patch features, ``(2 * batch, patches, d)``.
    mask : torch.Tensor
        Boolean ``(2 * batch, patches)`` of the patches hidden from the student.

    Returns
    -------
    dict[str, torch.Tensor]
        ``view``: mean distance between each student view and each teacher global view of the same ECG,
        except a view with itself. ``patch``: mean distance on hidden patches. ``coding_rate``: mean rate
        of the student global views. ``loss``: ``view + patch - gamma * coding_rate``.
    """
    batch, size = teacher_global.shape[0] // GLOBAL_VIEWS, teacher_global.shape[1]
    teacher_views = teacher_global.view(GLOBAL_VIEWS, batch, size)
    student_views = torch.cat([student_global, student_local]).view(GLOBAL_VIEWS + LOCAL_VIEWS, batch, size)
    distance = 1 - torch.einsum("sbd,tbd->st", student_views, teacher_views) / batch
    different = 1 - torch.eye(GLOBAL_VIEWS + LOCAL_VIEWS, GLOBAL_VIEWS, device=distance.device)
    view = (distance * different).sum() / different.sum()
    patch_distance = 1 - (student_patches * teacher_patches).sum(dim=-1)
    patch = (patch_distance * mask).sum() / mask.sum().clamp(min=1)
    rate = torch.stack([coding_rate(features) for features in student_views[:GLOBAL_VIEWS]]).mean()
    gamma = EPSILON * math.sqrt(batch / (size * min(size, batch)))
    loss = view + patch - gamma * rate.float().to(view.device)
    return {"loss": loss, "view": view, "patch": patch, "coding_rate": rate}


@torch.no_grad()
def xecg_embeddings(network: XECGNetwork, dataset: SignalDataset, *, batch_size: int,
                    device: torch.device) -> np.ndarray:
    """
    Embed every whole ECG of a dataset, in dataset order.

    Parameters
    ----------
    network : XECGNetwork
        Usually the teacher.
    dataset : SignalDataset
        ECGs to embed.
    batch_size : int
        ECGs per batch.
    device : torch.device
        Where the network is.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(len(dataset), width)``.
    """
    network.eval()
    parts = []
    for start in range(0, len(dataset), batch_size):
        signals = dataset.batch(np.arange(start, min(start + batch_size, len(dataset))), device)
        parts.append(network.embed(signals).cpu().numpy())
    return np.concatenate(parts).astype(np.float32)


def effective_rank(embeddings: np.ndarray) -> float:
    """
    Count the directions that embeddings really use.

    It is the exponential of the entropy of the covariance eigenvalues: 1 when every ECG gets the same
    vector up to one direction, and the embedding size when all directions carry equal variance.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape ``(records, width)``.

    Returns
    -------
    float
        Between 1 and ``width``.
    """
    values = np.linalg.eigvalsh(np.cov(embeddings.astype(np.float64), rowvar=False)).clip(min=0)
    share = values / values.sum()
    share = share[share > 0]
    return float(np.exp(-(share * np.log(share)).sum()))


def train_epoch(student: XECGNetwork, teacher: XECGNetwork, dataset: SignalDataset, order: np.ndarray,
                batch_size: int, optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler.LRScheduler, first_step: int, total_steps: int,
                device: torch.device, label: str) -> dict[str, float]:
    """
    Run one pass of SimDINOv2 over the training ECGs.

    Parameters
    ----------
    student, teacher : XECGNetwork
        The trained network and its moving average.
    dataset : SignalDataset
        Training ECGs.
    order : np.ndarray
        Shuffled record positions; the last incomplete batch is dropped.
    batch_size : int
        ECGs per batch.
    optimizer : torch.optim.Optimizer
        Optimizer of the student.
    scheduler : torch.optim.lr_scheduler.LRScheduler
        Learning-rate schedule, stepped once per batch.
    first_step : int
        Updates done before this epoch.
    total_steps : int
        Updates of the whole run; the teacher momentum grows linearly from 0.99 to 1 over them.
    device : torch.device
        Where the networks are.
    label : str
        Prefix of the progress line.

    Returns
    -------
    dict[str, float]
        Epoch means of ``loss``, ``view``, ``patch`` and ``coding_rate``.

    Raises
    ------
    FloatingPointError
        If the loss stops being finite.
    """
    student.train()
    steps = len(order) // batch_size
    names = ("loss", "view", "patch", "coding_rate")
    totals, started = torch.zeros(len(names), device=device), time.perf_counter()
    for step in range(steps):
        signals = dataset.batch(order[step * batch_size:(step + 1) * batch_size], device)
        global_views, local_views = make_views(signals)
        mask = torch.rand(global_views.shape[0], GLOBAL_SAMPLES // PATCH, device=device) < MASK_PROBABILITY
        with torch.no_grad():
            teacher_global, teacher_patches = teacher(global_views)
        student_global, student_patches = student(global_views, mask)
        terms = simdino_loss(student_global, student_patches, student(local_views)[0], teacher_global,
                             teacher_patches, mask)
        optimizer.zero_grad(set_to_none=True)
        terms["loss"].backward()
        nn.utils.clip_grad_norm_(student.parameters(), GRADIENT_NORM)
        optimizer.step()
        scheduler.step()
        progress = (first_step + step + 1) / total_steps
        update_teacher(teacher, student, MOMENTUM_START + progress * (1 - MOMENTUM_START))
        totals += torch.stack([terms[name].detach().float().to(device) for name in names])
        if (step + 1) % max(1, steps // 20) == 0 or step + 1 == steps:
            means = (totals / (step + 1)).tolist()
            if not np.isfinite(means[0]):
                raise FloatingPointError(f"Training loss is not finite at step {step + 1}")
            rate = (step + 1) * batch_size / (time.perf_counter() - started)
            print(f"\r{label} paso {step + 1}/{steps} pérdida={means[0]:.3f} vistas={means[1]:.3f} "
                  f"parches={means[2]:.3f} tasa={means[3]:.1f} {rate:,.0f} ECG/s", end="", flush=True)
    return dict(zip(names, (totals / steps).tolist(), strict=True))


def train_xecg(student: XECGNetwork, teacher: XECGNetwork, train_dataset: SignalDataset,
               val_dataset: SignalDataset, *, epochs: int, batch_size: int, seed: int, device: torch.device,
               checkpoint_path: Path, resume: bool = True, learning_rate: float = 5e-4,
               eval_batch_size: int = 512) -> pd.DataFrame:
    """
    Pretrain the student with SimDINOv2 and checkpoint both networks after every epoch.

    AdamW (weight decay 0.04) follows a one-cycle schedule with 10% warm-up; gradients are clipped at
    norm 3. The random generator is reseeded from the seed and the epoch number at the start of each
    epoch, so the views of an epoch do not depend on whether the run was resumed.

    Parameters
    ----------
    student, teacher : XECGNetwork
        The network to train and its copy from ``make_teacher``, both on ``device``.
    train_dataset, val_dataset : SignalDataset
        Training and validation ECGs.
    epochs : int
        Passes over the training ECGs.
    batch_size : int
        ECGs per batch; each one becomes six views.
    seed : int
        Seed of the batch order and of the views.
    device : torch.device
        Where the networks are.
    checkpoint_path : Path
        File rewritten after every epoch.
    resume : bool
        Continue from ``checkpoint_path`` when it exists.
    learning_rate : float
        Peak learning rate.
    eval_batch_size : int
        ECGs per batch when embedding the validation split.

    Returns
    -------
    pd.DataFrame
        One row per epoch: ``epoch``, the four training means, ``val_effective_rank`` of the teacher
        embeddings and ``seconds``.

    Raises
    ------
    ValueError
        If the checkpoint was written with another configuration.
    """
    steps = len(train_dataset) // batch_size
    settings = {"epochs": epochs, "batch_size": batch_size, "seed": seed, "learning_rate": learning_rate,
                "train_records": len(train_dataset), "model": student.config}
    optimizer = torch.optim.AdamW(student.parameters(), lr=learning_rate, weight_decay=0.04)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate,
                                                    total_steps=epochs * steps, pct_start=0.1)
    history: list[dict[str, float]] = []
    if resume and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if saved["settings"] != settings:
            raise ValueError(f"Checkpoint settings {saved['settings']} differ from {settings}")
        student.load_state_dict(saved["student"])
        teacher.load_state_dict(saved["teacher"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        history = saved["history"]
        print(f"Reanudando después de la época {len(history)}")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(len(history) + 1, epochs + 1):
        started = time.perf_counter()
        torch.manual_seed(seed * 1000 + epoch)
        means = train_epoch(student, teacher, train_dataset, epoch_order(len(train_dataset), seed, epoch),
                            batch_size, optimizer, scheduler, (epoch - 1) * steps, epochs * steps, device,
                            f"Época {epoch}/{epochs}")
        rank = effective_rank(xecg_embeddings(teacher, val_dataset, batch_size=eval_batch_size,
                                              device=device))
        seconds = time.perf_counter() - started
        history.append({"epoch": epoch, **means, "val_effective_rank": rank, "seconds": seconds})
        print(f"\rÉpoca {epoch}/{epochs}: pérdida={means['loss']:.3f} vistas={means['view']:.3f} "
              f"parches={means['patch']:.3f} tasa={means['coding_rate']:.1f} rango_val={rank:.1f} "
              f"{seconds:.0f} s" + " " * 20)
        partial = checkpoint_path.with_suffix(".partial")
        torch.save({"settings": settings, "student": student.state_dict(), "teacher": teacher.state_dict(),
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "history": history}, partial)
        os.replace(partial, checkpoint_path)
    return pd.DataFrame(history)


def save_xecg_artifacts(output_dir: Path, teacher: XECGNetwork, history: pd.DataFrame,
                        train_embeddings: np.ndarray, val_embeddings: np.ndarray, train_cluster: np.ndarray,
                        val_cluster: np.ndarray, cluster_fit: dict[str, object]) -> None:
    """
    Save the teacher, history, embeddings and cluster transformation; no patient keys or labels.

    The files match those of Experiment 042, except that the weights are in ``xecg_teacher.pt``.

    Parameters
    ----------
    output_dir : Path
        Folder to write.
    teacher : XECGNetwork
        The pretrained teacher.
    history : pd.DataFrame
        Output of ``train_xecg``.
    train_embeddings, val_embeddings : np.ndarray
        Outputs of ``xecg_embeddings``.
    train_cluster, val_cluster : np.ndarray
        Cluster numbers from ``cluster_embeddings``.
    cluster_fit : dict[str, object]
        Fitted ``scaler``, ``pca`` and ``kmeans`` from ``cluster_embeddings``.
    """
    save_artifacts(output_dir, teacher, history, train_embeddings, val_embeddings, train_cluster,
                   val_cluster, cluster_fit)
    os.replace(output_dir / "autoencoder.pt", output_dir / "xecg_teacher.pt")
