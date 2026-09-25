"""Matched sampled InfoNCE and word2vec-style SGNS heads for local ECG CPC.

Both arms use identical positive and sampled negative cosine scores. SGNS uses
the conventional sum of sixteen negative softplus terms, with no /17 scaling.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812 - conventional PyTorch alias

from .cpc import (
    HORIZONS,
    TEMPERATURE,
    WIDTH,
    CPCEncoder,
    mean_pair_cosine,
    prediction_heads,
    temporal_candidate_mask,
    token_variance,
)

NEGATIVES = 16
OBJECTIVES = ("sampled_info", "sgns")


def sampled_negative_indices(length: int, horizon: int, batch: int, generator: torch.Generator,
                             device: torch.device | str,
                             count: int = NEGATIVES) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Draw negatives uniformly with replacement among distant same-half nonpositive positions.

    Parameters
    ----------
    length : int
        Tokens per half.
    horizon : int
        Prediction offset in tokens.
    batch : int
        Number of records; each contributes two halves.
    generator : torch.Generator
        Private CPU generator for the negative draw.
    device : torch.device | str
        Device of the returned tensors.
    count : int
        Negatives per query.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Negative positions [batch * 2, queries, count], the valid-query mask
        [length] and the positive target of each query.

    Raises
    ------
    ValueError
        If a query has no admissible negative.
    """
    mask, valid = temporal_candidate_mask(length, horizon, "cpu")
    query_positions = torch.arange(length)[valid]
    positives = query_positions + horizon
    allowed = mask[valid].clone()
    allowed[torch.arange(len(query_positions)), positives] = False
    counts = allowed.sum(dim=1)
    if not bool(torch.all(counts > 0)):
        raise ValueError("No valid temporal negatives for a query")
    # One vectorized draw per horizon; the private CPU generator never changes
    # dropout or DataLoader RNG state. Padding is unreachable after scaling.
    choices = torch.full((len(query_positions), length), 0, dtype=torch.long)
    for row in range(len(query_positions)):
        candidates = torch.nonzero(allowed[row], as_tuple=True)[0]
        choices[row, :len(candidates)] = candidates
    draw = torch.rand((batch * 2, len(query_positions), count), generator=generator)
    slots = (draw * counts.reshape(1, -1, 1)).long()
    sampled = choices.unsqueeze(0).expand(batch * 2, -1, -1).gather(2, slots)
    return sampled.to(device), valid.to(device), positives.to(device)


def _horizon_loss(positive_scores: torch.Tensor, negative_scores: torch.Tensor, variant: str) -> torch.Tensor:
    """Score sampled InfoNCE with the positive in class zero, or sum the SGNS softplus terms."""
    if variant == "sampled_info":
        logits = torch.cat((positive_scores.unsqueeze(-1), negative_scores), dim=-1)
        loss = F.cross_entropy(logits.reshape(-1, NEGATIVES + 1),
                               torch.zeros(logits.numel() // (NEGATIVES + 1),
                                           dtype=torch.long, device=logits.device))
    else:
        loss = (F.softplus(-positive_scores) + F.softplus(negative_scores).sum(dim=-1)).mean()
    return loss


def sampled_objective(tokens: torch.Tensor, contexts: torch.Tensor, heads: Sequence[nn.Module], variant: str,
                      generator: torch.Generator) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Calculate either objective from the same sampled cosine logits.

    Parameters
    ----------
    tokens : torch.Tensor
        Target tokens of shape [batch, 2, time, 256].
    contexts : torch.Tensor
        Query contexts with the same shape.
    heads : Sequence[nn.Module]
        One prediction head per horizon.
    variant : str
        ``"sampled_info"`` or ``"sgns"``.
    generator : torch.Generator
        Private CPU generator for negative sampling.

    Returns
    -------
    tuple[torch.Tensor, dict[str, float]]
        Scalar loss and mean positive/negative scores.

    Raises
    ------
    ValueError
        If the objective is unknown or the tensors are malformed.
    """
    if variant not in OBJECTIVES:
        raise ValueError(f"Unknown sampled CPC objective {variant}")
    if (tokens.shape != contexts.shape or tokens.ndim != 4 or tokens.shape[1] != 2
            or tokens.shape[-1] != WIDTH):
        raise ValueError("Expected matching [batch,2,time,256] token and context tensors")
    batch, halves, length, width = tokens.shape
    target = F.normalize(tokens.reshape(batch * halves, length, width), dim=-1)
    queries = contexts.reshape(batch * halves, length, width)
    losses, positive_means, negative_means = [], [], []
    for horizon, head in zip(HORIZONS, heads, strict=True):
        negative_indices, valid, positives = sampled_negative_indices(
            length, horizon, batch, generator, tokens.device)
        prediction = F.normalize(head(queries[:, valid]), dim=-1)
        scores = torch.bmm(prediction, target.transpose(1, 2)) / TEMPERATURE
        positive_scores = scores[:, torch.arange(len(positives), device=tokens.device), positives]
        negative_scores = scores.gather(2, negative_indices)
        losses.append(_horizon_loss(positive_scores, negative_scores, variant))
        positive_means.append(positive_scores.detach().mean())
        negative_means.append(negative_scores.detach().mean())
    return torch.stack(losses).mean(), {
        "positive_score": float(torch.stack(positive_means).mean()),
        "negative_score": float(torch.stack(negative_means).mean()),
    }


class SampledCPCPretrainer(nn.Module):
    """Native CPC encoder trained with a sampled InfoNCE or SGNS objective."""

    def __init__(self, variant: str) -> None:
        """
        Build the encoder and prediction heads.

        Parameters
        ----------
        variant : str
            ``"sampled_info"`` or ``"sgns"``.

        Raises
        ------
        ValueError
            If the variant is unknown.
        """
        super().__init__()
        if variant not in OBJECTIVES:
            raise ValueError(f"Unknown variant {variant}")
        self.encoder = CPCEncoder()
        self.heads = prediction_heads()
        self.variant = variant

    def forward(self, signal: torch.Tensor,
                sampler_generator: torch.Generator) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute the sampled objective and collapse diagnostics.

        Parameters
        ----------
        signal : torch.Tensor
            Normalized signals of shape [batch, 12, 2500].
        sampler_generator : torch.Generator
            Private CPU generator for negative sampling.

        Returns
        -------
        tuple[torch.Tensor, dict[str, float]]
            Scalar loss and detached diagnostics.
        """
        tokens, contexts = self.encoder(signal)
        loss, details = sampled_objective(tokens, contexts, self.heads,
                                          self.variant, sampler_generator)
        details["mean_pair_cosine"] = mean_pair_cosine(self.encoder.pooled(contexts))
        details["token_variance"] = token_variance(tokens)
        return loss, details
