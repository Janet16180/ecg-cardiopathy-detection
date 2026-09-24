"""Matched sampled InfoNCE and word2vec-style SGNS heads for local ECG CPC.

Both arms use identical positive and sampled negative cosine scores. SGNS uses
the conventional sum of sixteen negative softplus terms, with no /17 scaling.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .cpc import CPCEncoder, HORIZONS, temporal_candidate_mask


NEGATIVES = 16
TEMPERATURE = 0.1


def sampled_negative_indices(length, horizon, batch, generator, device, count=NEGATIVES):
    """Uniform with replacement among distant same-half nonpositive positions."""
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


def sampled_objective(tokens, contexts, heads, variant, generator):
    """Calculate either objective from the same sampled cosine logits."""
    if variant not in ("sampled_info", "sgns"):
        raise ValueError(f"Unknown sampled CPC objective {variant}")
    if tokens.shape != contexts.shape or tokens.ndim != 4 or tokens.shape[1] != 2 or tokens.shape[-1] != 256:
        raise ValueError("Expected matching [batch,2,time,256] token and context tensors")
    batch, halves, length, width = tokens.shape
    target = F.normalize(tokens.reshape(batch * halves, length, width), dim=-1)
    queries = contexts.reshape(batch * halves, length, width)
    losses, positive_means, negative_means = [], [], []
    for horizon, head in zip(HORIZONS, heads):
        negative_indices, valid, positives = sampled_negative_indices(
            length, horizon, batch, generator, tokens.device)
        prediction = F.normalize(head(queries[:, valid]), dim=-1)
        scores = torch.bmm(prediction, target.transpose(1, 2)) / TEMPERATURE
        positive_scores = scores[:, torch.arange(len(positives), device=tokens.device), positives]
        negative_scores = scores.gather(2, negative_indices)
        if variant == "sampled_info":
            logits = torch.cat((positive_scores.unsqueeze(-1), negative_scores), dim=-1)
            loss = F.cross_entropy(logits.reshape(-1, NEGATIVES + 1),
                                   torch.zeros(logits.numel() // (NEGATIVES + 1),
                                               dtype=torch.long, device=logits.device))
        else:
            loss = (F.softplus(-positive_scores) + F.softplus(negative_scores).sum(dim=-1)).mean()
        losses.append(loss)
        positive_means.append(positive_scores.detach().mean())
        negative_means.append(negative_scores.detach().mean())
    return torch.stack(losses).mean(), {
        "positive_score": float(torch.stack(positive_means).mean()),
        "negative_score": float(torch.stack(negative_means).mean()),
    }


class SampledCPCPretrainer(nn.Module):
    def __init__(self, variant):
        super().__init__()
        if variant not in ("sampled_info", "sgns"):
            raise ValueError(f"Unknown variant {variant}")
        self.encoder = CPCEncoder()
        self.heads = nn.ModuleList(nn.Linear(256, 256, bias=False) for _ in HORIZONS)
        self.variant = variant

    def forward(self, signal, sampler_generator):
        tokens, contexts = self.encoder(signal)
        loss, details = sampled_objective(tokens, contexts, self.heads,
                                          self.variant, sampler_generator)
        pooled = F.normalize(self.encoder.pooled(contexts).detach(), dim=-1)
        similarity = pooled @ pooled.T
        details["mean_pair_cosine"] = float(
            (similarity.sum() - similarity.diag().sum()) / (len(pooled) * (len(pooled) - 1))
            if len(pooled) > 1 else 0.0)
        details["token_variance"] = float(F.normalize(tokens.detach().reshape(-1, 256), dim=-1)
                                          .var(dim=0, unbiased=False).mean())
        return loss, details
