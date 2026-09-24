"""Continued CPC with frozen-round cluster targets or causal chunk contexts."""

import copy
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from torch import nn
from torch.nn import functional as F

from .cpc import CPCEncoder, HORIZONS, temporal_candidate_mask


CLUSTERS = 64
CLUSTER_WEIGHT = 0.1
FIT_RECORDS = 2000
TOKENS_PER_RECORD = 100


def cnn_tokens(convs, signal):
    if signal.ndim != 3 or signal.shape[1:] != (12, 2500):
        raise ValueError("Expected [batch,12,2500] input")
    batch = len(signal)
    halves = signal.reshape(batch, 12, 2, 1250).permute(0, 2, 1, 3)
    flat = halves.reshape(batch * 2, 12, 1250)
    return convs(flat).transpose(1, 2).reshape(batch, 2, 79, 256)


def snapshot_teacher_convs(encoder):
    core = encoder.core if hasattr(encoder, "core") else encoder
    teacher = copy.deepcopy(core.convs).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def temporal_cpc_loss(tokens, contexts, heads, first_query=3, temperature=0.1):
    """All same-half negatives except positions within three of the positive."""
    batch, halves, length, width = tokens.shape
    targets = F.normalize(tokens.reshape(batch * halves, length, width), dim=-1)
    queries = contexts.reshape(batch * halves, length, width)
    losses = []
    for horizon, head in zip(HORIZONS, heads):
        candidate, valid = temporal_candidate_mask(length, horizon, tokens.device)
        valid[:first_query] = False
        prediction = F.normalize(head(queries[:, valid]), dim=-1)
        logits = torch.bmm(prediction, targets.transpose(1, 2)) / temperature
        logits = logits.masked_fill(~candidate[valid].unsqueeze(0), torch.finfo(logits.dtype).min)
        positive = torch.arange(length, device=tokens.device)[valid] + horizon
        labels = positive.expand(batch * halves, -1)
        losses.append(F.cross_entropy(logits.reshape(-1, length), labels.reshape(-1)))
    return torch.stack(losses).mean()


def nearest_cluster(tokens, centers):
    """Assign normalized CNN tokens to Euclidean K-means centers without gradients."""
    if tokens.shape[-1] != 256 or centers.shape != (CLUSTERS, 256):
        raise ValueError("Expected 256-wide tokens and 64 K-means centers")
    normalized = F.normalize(tokens.detach(), dim=-1)
    scores = 2 * (normalized @ centers.T) - centers.square().sum(dim=-1)
    return scores.argmax(dim=-1)


def future_cluster_loss(teacher_tokens, contexts, centers, heads, first_query=24):
    """Predict each future CNN token's fixed code ID from past GRU context."""
    if teacher_tokens.shape != contexts.shape or teacher_tokens.ndim != 4 or teacher_tokens.shape[1] != 2:
        raise ValueError("Expected matching [batch,2,time,256] tokens and contexts")
    length = teacher_tokens.shape[2]
    labels = nearest_cluster(teacher_tokens, centers)
    positions = torch.arange(length, device=teacher_tokens.device)
    losses = []
    for horizon, head in zip(HORIZONS, heads):
        valid = (positions >= first_query) & (positions + horizon < length)
        logits = head(contexts[:, :, valid])
        future = labels[:, :, positions[valid] + horizon]
        losses.append(F.cross_entropy(logits.reshape(-1, CLUSTERS), future.reshape(-1)))
    return torch.stack(losses).mean()


def fit_kmeans(features, seed=42):
    """MiniBatchKMeans on bounded, training-only normalized CNN embeddings."""
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != 256 or len(features) < CLUSTERS:
        raise ValueError("Expected at least 64 CNN embeddings of width 256")
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite K-means fit embeddings")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.maximum(norms, 1e-12)
    fitted = MiniBatchKMeans(n_clusters=CLUSTERS, batch_size=4096, max_iter=100,
                             n_init=3, random_state=seed).fit(features)
    counts = np.bincount(fitted.labels_, minlength=CLUSTERS)
    probability = counts[counts > 0] / len(features)
    return np.asarray(fitted.cluster_centers_, dtype=np.float32), {
        "inertia": float(fitted.inertia_),
        "iterations": int(fitted.n_iter_),
        "clusters": CLUSTERS, "samples": len(features), "seed": seed,
        "occupied_clusters": int(np.count_nonzero(counts)),
        "normalized_entropy": float(-(probability * np.log(probability)).sum() / np.log(CLUSTERS)),
        "largest_cluster_fraction": float(counts.max() / len(features)),
        "fit_label_histogram": counts.astype(int).tolist(),
    }


class TokenizationPretrainer(nn.Module):
    """Same causal CNN and CPC heads; target or context tokenization varies."""

    def __init__(self, variant):
        super().__init__()
        if variant in ("continuation", "clusteraux"):
            self.encoder = CPCEncoder()
        elif variant in ("fixedchunk", "beatchunk", "learnedchunk"):
            from .ecg_tokenizers import CausalChunkEncoder
            self.encoder = CausalChunkEncoder(variant)
        else:
            raise ValueError(f"Unknown tokenization variant {variant}")
        self.variant = variant
        self.first_query = 24
        self.heads = nn.ModuleList(nn.Linear(256, 256, bias=False) for _ in HORIZONS)
        self.cluster_heads = (nn.ModuleList(nn.Linear(256, CLUSTERS) for _ in HORIZONS)
                              if variant == "clusteraux" else None)

    def forward(self, signal, beat_boundaries=None, centers=None, teacher_convs=None):
        if self.variant in ("continuation", "clusteraux"):
            tokens, contexts = self.encoder(signal)
            rate_penalty = tokens.new_zeros(())
        else:
            tokens, contexts = self.encoder(signal, beat_boundaries)
            rate_penalty = self.encoder.rate_penalty
        ordinary = temporal_cpc_loss(tokens, contexts, self.heads, self.first_query)
        auxiliary = ordinary.new_zeros(())
        if self.cluster_heads is not None:
            if centers is None or teacher_convs is None:
                raise ValueError("Fixed teacher CNN and centers required for cluster auxiliary")
            with torch.no_grad():
                teacher_tokens = cnn_tokens(teacher_convs, signal)
            auxiliary = future_cluster_loss(teacher_tokens, contexts, centers,
                                            self.cluster_heads, self.first_query)
        loss = ordinary + CLUSTER_WEIGHT * auxiliary + rate_penalty
        pooled = F.normalize(self.encoder.pooled(contexts).detach(), dim=-1)
        similarity = pooled @ pooled.T
        details = {"cpc": float(ordinary.detach()),
                      "cluster_ce": float(auxiliary.detach()),
                      "rate_penalty": float(rate_penalty.detach()),
                      "token_variance": float(F.normalize(tokens.detach().reshape(-1, 256), dim=-1)
                                              .var(dim=0, unbiased=False).mean()),
                      "mean_pair_cosine": float((similarity.sum() - similarity.diag().sum()) /
                                                (len(pooled) * (len(pooled) - 1))
                                                if len(pooled) > 1 else 0.0)}
        if self.variant in ("fixedchunk", "beatchunk", "learnedchunk"):
            details.update(self.encoder.diagnostics)
        return loss, details


class TokenizationClassifier(nn.Module):
    def __init__(self, variant):
        super().__init__()
        if variant in ("continuation", "clusteraux"):
            self.encoder = CPCEncoder()
        elif variant in ("fixedchunk", "beatchunk", "learnedchunk"):
            from .ecg_tokenizers import CausalChunkEncoder
            self.encoder = CausalChunkEncoder(variant)
        else:
            raise ValueError(f"Unknown variant {variant}")
        self.variant = variant
        self.head = nn.Linear(512, 1)

    def forward(self, signal, beat_boundaries=None):
        if self.variant in ("continuation", "clusteraux"):
            _, contexts = self.encoder(signal)
        else:
            _, contexts = self.encoder(signal, beat_boundaries)
        return self.head(self.encoder.pooled(contexts)).squeeze(-1)
