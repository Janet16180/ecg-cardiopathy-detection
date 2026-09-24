"""Compact, strictly causal local CPC for ten-second, twelve-lead ECGs.

This is an experiment-specific CNN/GRU CPC implementation, not a reproduction of
the published S4 ECG-CPC architecture. Each five-second half is encoded alone.
"""

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812 - conventional PyTorch alias

LEADS = 12
SIGNAL_SAMPLES = 2500
HALF_SAMPLES = 1250
TOKEN_COUNT = 79
WIDTH = 256
HORIZONS = (4, 8, 12)
FIRST_QUERY = 3
NEAR_RADIUS = 3
TEMPERATURE = 0.1
CMSC_WEIGHT = 0.1
CONV_WIDTHS = (LEADS, 64, 128, 192, WIDTH)
CONV_KERNELS = (5, 3, 3, 3)


def split_halves(signal: torch.Tensor) -> torch.Tensor:
    """
    Stack the two independent five-second halves of each record along the batch.

    Parameters
    ----------
    signal : torch.Tensor
        Signals of shape [batch, 12, 2500]; the caller validates the shape.

    Returns
    -------
    torch.Tensor
        Halves of shape [batch * 2, 12, 1250], record-major.
    """
    batch = len(signal)
    halves = signal.reshape(batch, LEADS, 2, HALF_SAMPLES).permute(0, 2, 1, 3)
    return halves.reshape(batch * 2, LEADS, HALF_SAMPLES)


class CausalConvBlock(nn.Module):
    """Left-padded stride-two convolution, layer norm and GELU."""

    def __init__(self, in_channels: int, out_channels: int, kernel: int) -> None:
        """
        Build the block.

        Parameters
        ----------
        in_channels : int
            Input channel count.
        out_channels : int
            Output channel count.
        kernel : int
            Convolution kernel size.
        """
        super().__init__()
        self.left_pad = kernel - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel, stride=2)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the causal block.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape [batch, channels, time].

        Returns
        -------
        torch.Tensor
            Output of shape [batch, out_channels, ceil(time / 2)].
        """
        x = self.conv(F.pad(x, (self.left_pad, 0))).transpose(1, 2)
        return F.gelu(self.norm(x)).transpose(1, 2)


class CPCEncoder(nn.Module):
    """Maps each independent 1250-sample half to 79 causal tokens and contexts."""

    def __init__(self) -> None:
        """Build the causal CNN and the two-layer GRU context network."""
        super().__init__()
        self.convs = nn.Sequential(*(CausalConvBlock(CONV_WIDTHS[i], CONV_WIDTHS[i + 1], CONV_KERNELS[i])
                                     for i in range(len(CONV_KERNELS))))
        self.context = nn.GRU(WIDTH, WIDTH, num_layers=2, batch_first=True, dropout=0.1)

    def forward(self, signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode both halves of each record.

        Parameters
        ----------
        signal : torch.Tensor
            Normalized signals of shape [batch, 12, 2500] at 250 Hz.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            CNN tokens and GRU contexts, each of shape [batch, 2, 79, 256].

        Raises
        ------
        ValueError
            If the signal shape is not [batch, 12, 2500].
        """
        if signal.ndim != 3 or signal.shape[1:] != (LEADS, SIGNAL_SAMPLES):
            raise ValueError("Expected [batch, 12, 2500] at 250 Hz")
        batch = len(signal)
        tokens = self.convs(split_halves(signal)).transpose(1, 2)
        contexts, _ = self.context(tokens)
        return tokens.reshape(batch, 2, -1, WIDTH), contexts.reshape(batch, 2, -1, WIDTH)

    @staticmethod
    def pooled(contexts: torch.Tensor) -> torch.Tensor:
        """
        Mean/max pool within each half, then average the two halves.

        Parameters
        ----------
        contexts : torch.Tensor
            Representations of shape [batch, 2, time, width].

        Returns
        -------
        torch.Tensor
            Pooled features of shape [batch, 2 * width].
        """
        halves = torch.cat((contexts.mean(dim=2), contexts.amax(dim=2)), dim=-1)
        return halves.mean(dim=1)


def temporal_candidate_mask(length: int, horizon: int,
                            device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build the same-half candidate mask for one prediction horizon.

    Rows are query positions; only the target and distant same-half tokens qualify.

    Parameters
    ----------
    length : int
        Number of tokens per half.
    horizon : int
        Prediction offset in tokens.
    device : torch.device | str
        Device of the returned tensors.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Boolean candidate mask [length, length] and valid-query mask [length].
    """
    position = torch.arange(length, device=device)
    target = position + horizon
    separation = (position[None, :] - target[:, None]).abs()
    mask = separation > NEAR_RADIUS
    valid = (position >= FIRST_QUERY) & (target < length)
    mask[valid, target[valid]] = True
    return mask, valid


def cpc_loss(tokens: torch.Tensor, contexts: torch.Tensor, heads: Sequence[nn.Module],
             temperature: float = TEMPERATURE, *, first_query: int = FIRST_QUERY) -> torch.Tensor:
    """
    Average same-half InfoNCE over all prediction horizons.

    Parameters
    ----------
    tokens : torch.Tensor
        Target tokens of shape [batch, 2, time, 256].
    contexts : torch.Tensor
        Query contexts with the same shape as ``tokens``.
    heads : Sequence[nn.Module]
        One prediction head per entry of ``HORIZONS``.
    temperature : float
        Softmax temperature applied to cosine scores.
    first_query : int
        Earliest query position; earlier positions are never queries.

    Returns
    -------
    torch.Tensor
        Scalar loss.

    Raises
    ------
    ValueError
        If the representation shape is wrong or a horizon has no valid query.
    """
    batch, halves, length, width = tokens.shape
    if halves != 2 or width != WIDTH:
        raise ValueError("Expected [batch, 2, tokens, 256] representations")
    targets = F.normalize(tokens.reshape(batch * halves, length, width), dim=-1)
    queries = contexts.reshape(batch * halves, length, width)
    losses = []
    for horizon, head in zip(HORIZONS, heads, strict=True):
        candidate, valid = temporal_candidate_mask(length, horizon, tokens.device)
        valid[:first_query] = False
        if not valid.any():
            raise ValueError("Sequence too short for a CPC horizon")
        prediction = F.normalize(head(queries[:, valid]), dim=-1)
        logits = torch.bmm(prediction, targets.transpose(1, 2)) / temperature
        logits = logits.masked_fill(~candidate[valid].unsqueeze(0), torch.finfo(logits.dtype).min)
        positive = torch.arange(length, device=tokens.device)[valid] + horizon
        labels = positive.expand(batch * halves, -1)
        losses.append(F.cross_entropy(logits.reshape(-1, length), labels.reshape(-1)))
    return torch.stack(losses).mean()


def cmsc_loss(contexts: torch.Tensor, patient_ids: Sequence[str],
              temperature: float = TEMPERATURE) -> torch.Tensor:
    """
    Match the two halves of each record contrastively.

    Other records from the same patient are excluded as negatives.

    Parameters
    ----------
    contexts : torch.Tensor
        Contexts of shape [batch, 2, time, width].
    patient_ids : Sequence[str]
        Patient identity of each record.
    temperature : float
        Softmax temperature applied to cosine scores.

    Returns
    -------
    torch.Tensor
        Symmetric scalar cross-entropy.

    Raises
    ------
    ValueError
        If ``patient_ids`` does not match the batch size.
    """
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


def mean_pair_cosine(features: torch.Tensor) -> float:
    """
    Average cosine similarity between distinct records, as a collapse diagnostic.

    Parameters
    ----------
    features : torch.Tensor
        Pooled record features of shape [batch, width].

    Returns
    -------
    float
        Mean off-diagonal cosine, or 0.0 for a single record.
    """
    pooled = F.normalize(features.detach(), dim=-1)
    similarity = pooled @ pooled.T
    return float((similarity.sum() - similarity.diag().sum()) / (len(pooled) * (len(pooled) - 1))
                 if len(pooled) > 1 else 0.0)


def token_variance(tokens: torch.Tensor) -> float:
    """
    Mean per-dimension variance of unit-normalized tokens, as a collapse diagnostic.

    Parameters
    ----------
    tokens : torch.Tensor
        Tokens whose last dimension is 256 wide.

    Returns
    -------
    float
        Mean population variance across all tokens.
    """
    return float(F.normalize(tokens.detach().reshape(-1, WIDTH), dim=-1).var(dim=0, unbiased=False).mean())


def prediction_heads() -> nn.ModuleList:
    """
    Build one bias-free linear prediction head per CPC horizon.

    Returns
    -------
    nn.ModuleList
        Heads in ``HORIZONS`` order.
    """
    return nn.ModuleList(nn.Linear(WIDTH, WIDTH, bias=False) for _ in HORIZONS)


class CPCPretrainer(nn.Module):
    """Ordinary CPC, optionally with an auxiliary cross-half CMSC term."""

    def __init__(self, hybrid: bool = False) -> None:
        """
        Build the encoder and prediction heads.

        Parameters
        ----------
        hybrid : bool
            Whether to add the weighted CMSC term to the loss.
        """
        super().__init__()
        self.encoder = CPCEncoder()
        self.heads = prediction_heads()
        self.hybrid = hybrid

    def forward(self, signal: torch.Tensor,
                patient_ids: Sequence[str] | None = None) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute the pretraining loss.

        Parameters
        ----------
        signal : torch.Tensor
            Normalized signals of shape [batch, 12, 2500].
        patient_ids : Sequence[str] | None
            Patient identities; required only for the hybrid objective.

        Returns
        -------
        tuple[torch.Tensor, dict[str, float]]
            Scalar loss and its detached ``cpc``/``cmsc`` components.
        """
        tokens, contexts = self.encoder(signal)
        ordinary = cpc_loss(tokens, contexts, self.heads)
        cross_half = cmsc_loss(contexts, patient_ids) if self.hybrid else ordinary.new_zeros(())
        loss = ordinary + CMSC_WEIGHT * cross_half
        return loss, {"cpc": float(ordinary.detach()), "cmsc": float(cross_half.detach())}


class CPCClassifier(nn.Module):
    """Linear binary head on pooled CPC contexts."""

    def __init__(self, encoder: CPCEncoder | None = None) -> None:
        """
        Build the classifier.

        Parameters
        ----------
        encoder : CPCEncoder | None
            Pretrained encoder; a new one is created when omitted.
        """
        super().__init__()
        self.encoder = encoder if encoder is not None else CPCEncoder()
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
