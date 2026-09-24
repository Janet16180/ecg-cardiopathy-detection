"""Compact, strictly causal local CPC for ten-second, twelve-lead ECGs.

This is an experiment-specific CNN/GRU CPC implementation, not a reproduction of
the published S4 ECG-CPC architecture. Each five-second half is encoded alone.
"""

import torch
from torch import nn
from torch.nn import functional as F


HORIZONS = (4, 8, 12)


class CausalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.left_pad = kernel - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel, stride=2)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = self.conv(F.pad(x, (self.left_pad, 0))).transpose(1, 2)
        return F.gelu(self.norm(x)).transpose(1, 2)


class CPCEncoder(nn.Module):
    """Maps each independent 1250-sample half to 79 causal tokens and contexts."""

    def __init__(self):
        super().__init__()
        widths = (12, 64, 128, 192, 256)
        kernels = (5, 3, 3, 3)
        self.convs = nn.Sequential(*(CausalConvBlock(widths[i], widths[i + 1], kernels[i])
                                     for i in range(4)))
        self.context = nn.GRU(256, 256, num_layers=2, batch_first=True,
                              dropout=0.1)

    def forward(self, signal):
        if signal.ndim != 3 or signal.shape[1:] != (12, 2500):
            raise ValueError("Expected [batch, 12, 2500] at 250 Hz")
        batch = len(signal)
        halves = signal.reshape(batch, 12, 2, 1250).permute(0, 2, 1, 3)
        flat = halves.reshape(batch * 2, 12, 1250)
        tokens = self.convs(flat).transpose(1, 2)
        contexts, _ = self.context(tokens)
        return tokens.reshape(batch, 2, -1, 256), contexts.reshape(batch, 2, -1, 256)

    @staticmethod
    def pooled(contexts):
        halves = torch.cat((contexts.mean(dim=2), contexts.amax(dim=2)), dim=-1)
        return halves.mean(dim=1)


def temporal_candidate_mask(length, horizon, device):
    """Rows are query positions; only the target and distant same-half tokens qualify."""
    position = torch.arange(length, device=device)
    target = position + horizon
    separation = (position[None, :] - target[:, None]).abs()
    mask = separation > 3
    valid = (position >= 3) & (target < length)
    mask[valid, target[valid]] = True
    return mask, valid


def cpc_loss(tokens, contexts, heads, temperature=0.1):
    batch, halves, length, width = tokens.shape
    if halves != 2 or width != 256:
        raise ValueError("Expected [batch, 2, tokens, 256] representations")
    targets = F.normalize(tokens.reshape(batch * halves, length, width), dim=-1)
    queries = contexts.reshape(batch * halves, length, width)
    losses = []
    for horizon, head in zip(HORIZONS, heads):
        candidate, valid = temporal_candidate_mask(length, horizon, tokens.device)
        if not valid.any():
            raise ValueError("Sequence too short for a CPC horizon")
        prediction = F.normalize(head(queries[:, valid]), dim=-1)
        logits = torch.bmm(prediction, targets.transpose(1, 2)) / temperature
        logits = logits.masked_fill(~candidate[valid].unsqueeze(0), torch.finfo(logits.dtype).min)
        positive = torch.arange(length, device=tokens.device)[valid] + horizon
        labels = positive.expand(batch * halves, -1)
        losses.append(F.cross_entropy(logits.reshape(-1, length), labels.reshape(-1)))
    return torch.stack(losses).mean()


def cmsc_loss(contexts, patient_ids, temperature=0.1):
    """Match halves; exclude other records from the same patient as negatives."""
    if len(patient_ids) != len(contexts):
        raise ValueError("patient_ids must match batch size")
    pooled = F.normalize(contexts.mean(dim=2), dim=-1)
    logits = pooled[:, 0] @ pooled[:, 1].T / temperature
    same = torch.as_tensor([a == b for a in patient_ids for b in patient_ids],
                           device=logits.device).reshape(len(patient_ids), -1)
    diagonal = torch.eye(len(patient_ids), dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(same & ~diagonal, torch.finfo(logits.dtype).min)
    labels = torch.arange(len(patient_ids), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


class CPCPretrainer(nn.Module):
    def __init__(self, hybrid=False):
        super().__init__()
        self.encoder = CPCEncoder()
        self.heads = nn.ModuleList(nn.Linear(256, 256, bias=False) for _ in HORIZONS)
        self.hybrid = hybrid

    def forward(self, signal, patient_ids=None):
        tokens, contexts = self.encoder(signal)
        ordinary = cpc_loss(tokens, contexts, self.heads)
        cross_half = cmsc_loss(contexts, patient_ids) if self.hybrid else ordinary.new_zeros(())
        loss = ordinary + 0.1 * cross_half
        return loss, {"cpc": float(ordinary.detach()), "cmsc": float(cross_half.detach())}


class CPCClassifier(nn.Module):
    def __init__(self, encoder=None):
        super().__init__()
        self.encoder = encoder if encoder is not None else CPCEncoder()
        self.head = nn.Linear(512, 1)

    def forward(self, signal):
        _, contexts = self.encoder(signal)
        return self.head(self.encoder.pooled(contexts)).squeeze(-1)
