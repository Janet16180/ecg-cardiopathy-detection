"""Continued CPC with frozen-round cluster targets or causal chunk contexts."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from torch import nn
from torch.nn import functional as F  # noqa: N812 - conventional PyTorch alias

from .cpc import (
    HORIZONS,
    TOKEN_COUNT,
    WIDTH,
    CPCEncoder,
    check_signal_batch,
    cpc_loss,
    mean_pair_cosine,
    prediction_heads,
    split_halves,
    token_variance,
)
from .ecg_tokenizers import CausalChunkEncoder

CLUSTERS = 64
CLUSTER_WEIGHT = 0.1
FIT_RECORDS = 2000
TOKENS_PER_RECORD = 100
FIRST_QUERY = 24
KMEANS_BATCH_SIZE = 4096
KMEANS_MAX_ITER = 100
KMEANS_INITS = 3
NATIVE_VARIANTS = ("continuation", "clusteraux")
CHUNK_VARIANTS = CausalChunkEncoder.VARIANTS


def cnn_tokens(convs: nn.Module, signal: torch.Tensor) -> torch.Tensor:
    """
    Apply a CPC CNN to both independent halves of each record.

    Parameters
    ----------
    convs : nn.Module
        The ``convs`` stack of a ``CPCEncoder``.
    signal : torch.Tensor
        Normalized signals of shape [batch, 12, 2500].

    Returns
    -------
    torch.Tensor
        Tokens of shape [batch, 2, 79, 256].

    Raises
    ------
    ValueError
        If the signal shape is wrong.
    """
    check_signal_batch(signal)
    return convs(split_halves(signal)).transpose(1, 2).reshape(len(signal), 2, TOKEN_COUNT, WIDTH)


def snapshot_teacher_convs(encoder: nn.Module) -> nn.Module:
    """
    Copy an encoder's CNN as a frozen, eval-mode teacher.

    Parameters
    ----------
    encoder : nn.Module
        A ``CPCEncoder`` or ``CausalChunkEncoder``.

    Returns
    -------
    nn.Module
        Frozen copy of the CNN stack.
    """
    core = encoder.core if hasattr(encoder, "core") else encoder
    teacher = copy.deepcopy(core.convs).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def nearest_cluster(tokens: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """
    Assign normalized CNN tokens to Euclidean K-means centers without gradients.

    Parameters
    ----------
    tokens : torch.Tensor
        Tokens whose last dimension is 256.
    centers : torch.Tensor
        K-means centers of shape [64, 256].

    Returns
    -------
    torch.Tensor
        Cluster index per token.

    Raises
    ------
    ValueError
        If the token width or center shape is wrong.
    """
    if tokens.shape[-1] != WIDTH or centers.shape != (CLUSTERS, WIDTH):
        raise ValueError("Expected 256-wide tokens and 64 K-means centers")
    normalized = F.normalize(tokens.detach(), dim=-1)
    scores = 2 * (normalized @ centers.T) - centers.square().sum(dim=-1)
    return scores.argmax(dim=-1)


def future_cluster_loss(teacher_tokens: torch.Tensor, contexts: torch.Tensor, centers: torch.Tensor,
                        heads: Sequence[nn.Module], first_query: int = FIRST_QUERY) -> torch.Tensor:
    """
    Predict each future CNN token's fixed code ID from past GRU context.

    Parameters
    ----------
    teacher_tokens : torch.Tensor
        Frozen-teacher tokens of shape [batch, 2, time, 256].
    contexts : torch.Tensor
        Student contexts with the same shape.
    centers : torch.Tensor
        K-means centers of shape [64, 256].
    heads : Sequence[nn.Module]
        One cluster classifier per horizon.
    first_query : int
        Earliest query position.

    Returns
    -------
    torch.Tensor
        Scalar cross-entropy averaged over horizons.

    Raises
    ------
    ValueError
        If tokens and contexts do not match the expected layout.
    """
    if teacher_tokens.shape != contexts.shape or teacher_tokens.ndim != 4 or teacher_tokens.shape[1] != 2:
        raise ValueError("Expected matching [batch,2,time,256] tokens and contexts")
    length = teacher_tokens.shape[2]
    labels = nearest_cluster(teacher_tokens, centers)
    positions = torch.arange(length, device=teacher_tokens.device)
    losses = []
    for horizon, head in zip(HORIZONS, heads, strict=True):
        valid = (positions >= first_query) & (positions + horizon < length)
        logits = head(contexts[:, :, valid])
        future = labels[:, :, positions[valid] + horizon]
        losses.append(F.cross_entropy(logits.reshape(-1, CLUSTERS), future.reshape(-1)))
    return torch.stack(losses).mean()


def fit_kmeans(features: np.ndarray, seed: int = 42) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Fit MiniBatchKMeans on bounded, training-only normalized CNN embeddings.

    Parameters
    ----------
    features : np.ndarray
        Embeddings of shape [samples, 256]; at least 64 rows.
    seed : int
        K-means random state.

    Returns
    -------
    tuple[np.ndarray, dict[str, Any]]
        Float32 centers [64, 256] and a fit summary.

    Raises
    ------
    ValueError
        If the embeddings are malformed, too few or nonfinite.
    """
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != WIDTH or len(features) < CLUSTERS:
        raise ValueError("Expected at least 64 CNN embeddings of width 256")
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite K-means fit embeddings")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.maximum(norms, 1e-12)
    fitted = MiniBatchKMeans(n_clusters=CLUSTERS, batch_size=KMEANS_BATCH_SIZE, max_iter=KMEANS_MAX_ITER,
                             n_init=KMEANS_INITS, random_state=seed).fit(features)
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


def _build_encoder(variant: str) -> nn.Module:
    """Create the native or chunked encoder for a tokenization variant."""
    if variant in NATIVE_VARIANTS:
        encoder = CPCEncoder()
    elif variant in CHUNK_VARIANTS:
        encoder = CausalChunkEncoder(variant)
    else:
        raise ValueError(f"Unknown tokenization variant {variant}")
    return encoder


def _encode(encoder: nn.Module, variant: str, signal: torch.Tensor,
            beat_boundaries: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a variant's encoder, passing beat boundaries only to chunked encoders."""
    if variant in NATIVE_VARIANTS:
        return encoder(signal)
    return encoder(signal, beat_boundaries)


class TokenizationPretrainer(nn.Module):
    """Same causal CNN and CPC heads; target or context tokenization varies."""

    def __init__(self, variant: str) -> None:
        """
        Build the encoder and CPC heads, plus cluster heads for ``clusteraux``.

        Parameters
        ----------
        variant : str
            One of ``NATIVE_VARIANTS`` or ``CHUNK_VARIANTS``.

        Raises
        ------
        ValueError
            If the variant is unknown.
        """
        super().__init__()
        self.encoder = _build_encoder(variant)
        self.variant = variant
        self.first_query = FIRST_QUERY
        self.heads = prediction_heads()
        self.cluster_heads = (nn.ModuleList(nn.Linear(WIDTH, CLUSTERS) for _ in HORIZONS)
                              if variant == "clusteraux" else None)

    def forward(self, signal: torch.Tensor, beat_boundaries: torch.Tensor | None = None,
                centers: torch.Tensor | None = None,
                teacher_convs: nn.Module | None = None) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute the continued-pretraining loss and diagnostics.

        Parameters
        ----------
        signal : torch.Tensor
            Normalized signals of shape [batch, 12, 2500].
        beat_boundaries : torch.Tensor | None
            Beat boundaries for the ``beatchunk`` variant.
        centers : torch.Tensor | None
            Fixed K-means centers; required for ``clusteraux``.
        teacher_convs : nn.Module | None
            Frozen teacher CNN; required for ``clusteraux``.

        Returns
        -------
        tuple[torch.Tensor, dict[str, float]]
            Scalar loss and detached diagnostics.

        Raises
        ------
        ValueError
            If ``clusteraux`` lacks centers or the teacher CNN.
        """
        tokens, contexts = _encode(self.encoder, self.variant, signal, beat_boundaries)
        if self.variant in NATIVE_VARIANTS:
            rate_penalty = tokens.new_zeros(())
        else:
            rate_penalty = self.encoder.rate_penalty
        ordinary = cpc_loss(tokens, contexts, self.heads, first_query=self.first_query)
        auxiliary = ordinary.new_zeros(())
        if self.cluster_heads is not None:
            if centers is None or teacher_convs is None:
                raise ValueError("Fixed teacher CNN and centers required for cluster auxiliary")
            with torch.no_grad():
                teacher_tokens = cnn_tokens(teacher_convs, signal)
            auxiliary = future_cluster_loss(teacher_tokens, contexts, centers,
                                            self.cluster_heads, self.first_query)
        loss = ordinary + CLUSTER_WEIGHT * auxiliary + rate_penalty
        details = {"cpc": float(ordinary.detach()),
                   "cluster_ce": float(auxiliary.detach()),
                   "rate_penalty": float(rate_penalty.detach()),
                   "token_variance": token_variance(tokens),
                   "mean_pair_cosine": mean_pair_cosine(self.encoder.pooled(contexts))}
        if self.variant in CHUNK_VARIANTS:
            details.update(self.encoder.diagnostics)
        return loss, details


class TokenizationClassifier(nn.Module):
    """Linear binary head on pooled tokenization-variant contexts."""

    def __init__(self, variant: str) -> None:
        """
        Build the classifier.

        Parameters
        ----------
        variant : str
            One of ``NATIVE_VARIANTS`` or ``CHUNK_VARIANTS``.

        Raises
        ------
        ValueError
            If the variant is unknown.
        """
        super().__init__()
        self.encoder = _build_encoder(variant)
        self.variant = variant
        self.head = nn.Linear(2 * WIDTH, 1)

    def forward(self, signal: torch.Tensor, beat_boundaries: torch.Tensor | None = None) -> torch.Tensor:
        """
        Predict one logit per record.

        Parameters
        ----------
        signal : torch.Tensor
            Normalized signals of shape [batch, 12, 2500].
        beat_boundaries : torch.Tensor | None
            Beat boundaries for the ``beatchunk`` variant.

        Returns
        -------
        torch.Tensor
            Logits of shape [batch].
        """
        _, contexts = _encode(self.encoder, self.variant, signal, beat_boundaries)
        return self.head(self.encoder.pooled(contexts)).squeeze(-1)
