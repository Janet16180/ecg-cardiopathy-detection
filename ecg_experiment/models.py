"""Compact encoders and SSL objectives, not published model reproductions."""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812 - conventional PyTorch alias

LEADS = 12
PATCHES = 40
PATCHES_PER_MASK_BLOCK = 2
MASK_BLOCKS = PATCHES // PATCHES_PER_MASK_BLOCK
VARIANCE_WEIGHT = 0.1


class CNN(nn.Module):
    """Six-layer strided CNN with global mean/max pooling."""

    feature_dim = 256

    def __init__(self) -> None:
        """Build three stages of strided and same-length convolutions."""
        super().__init__()
        layers = []
        previous = LEADS
        for channels in (32, 64, 128):
            layers.extend([nn.Conv1d(previous, channels, 7, stride=2, padding=3),
                           nn.GroupNorm(8, channels), nn.GELU(),
                           nn.Conv1d(channels, channels, 5, padding=2),
                           nn.GroupNorm(8, channels), nn.GELU()])
            previous = channels
        self.layers = nn.Sequential(*layers)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Encode and pool a batch of ECGs.

        Parameters
        ----------
        signal : torch.Tensor
            Signals of shape [batch, 12, samples].

        Returns
        -------
        torch.Tensor
            Features of shape [batch, 256].
        """
        tokens = self.layers(signal)
        return torch.cat([tokens.mean(-1), tokens.amax(-1)], dim=1)


class PatchTransformer(nn.Module):
    """Joint 12-lead patches of 250 ms; 40 tokens per 10-second ECG."""

    width = 96
    patch_size = 25
    feature_dim = 192

    def __init__(self) -> None:
        """Build the patch embedding, learned positions and a pre-norm Transformer."""
        super().__init__()
        self.patch = nn.Conv1d(LEADS, self.width, self.patch_size, stride=self.patch_size)
        self.position = nn.Parameter(torch.randn(1, PATCHES, self.width) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.width) * 0.02)
        layer = nn.TransformerEncoderLayer(self.width, 4, dim_feedforward=2 * self.width,
                                           dropout=0.1, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 3, norm=nn.LayerNorm(self.width),
                                             enable_nested_tensor=False)

    def tokens(self, signal: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Encode patch tokens, replacing masked patches with the mask token.

        Parameters
        ----------
        signal : torch.Tensor
            Signals of shape [batch, 12, 1000].
        mask : torch.Tensor | None
            Boolean patch mask of shape [batch, 40].

        Returns
        -------
        torch.Tensor
            Tokens of shape [batch, 40, 96].
        """
        tokens = self.patch(signal).transpose(1, 2)
        if mask is not None:
            tokens = torch.where(mask.unsqueeze(-1), self.mask_token, tokens)
        return self.encoder(tokens + self.position)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Encode and pool a batch of ECGs.

        Parameters
        ----------
        signal : torch.Tensor
            Signals of shape [batch, 12, 1000].

        Returns
        -------
        torch.Tensor
            Features of shape [batch, 192].
        """
        tokens = self.tokens(signal)
        return torch.cat([tokens.mean(1), tokens.amax(1)], dim=1)


class Classifier(nn.Module):
    """Layer-normalized, dropout-regularized linear binary head on an encoder."""

    def __init__(self, encoder: nn.Module) -> None:
        """
        Build the classifier.

        Parameters
        ----------
        encoder : nn.Module
            Encoder with a ``feature_dim`` attribute.
        """
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.LayerNorm(encoder.feature_dim), nn.Dropout(0.2),
                                  nn.Linear(encoder.feature_dim, 1))

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Predict one logit per record.

        Parameters
        ----------
        signal : torch.Tensor
            Encoder input batch.

        Returns
        -------
        torch.Tensor
            Logits of shape [batch].
        """
        return self.head(self.encoder(signal)).squeeze(-1)


def random_mask(batch: int, device: torch.device | str, ratio: float = 0.5) -> torch.Tensor:
    """
    Mask contiguous 500-ms spans of two patches each.

    Exactly half the patches are masked for ``ratio=0.5``.

    Parameters
    ----------
    batch : int
        Number of records.
    device : torch.device | str
        Device of the mask and of the random draw.
    ratio : float
        Fraction of blocks to mask.

    Returns
    -------
    torch.Tensor
        Boolean patch mask of shape [batch, 40].
    """
    noise = torch.rand(batch, MASK_BLOCKS, device=device)
    masked_blocks = noise.argsort(dim=1).argsort(dim=1) < round(MASK_BLOCKS * ratio)
    return masked_blocks.repeat_interleave(PATCHES_PER_MASK_BLOCK, dim=1)


class MaskedAutoencoder(nn.Module):
    """Masked patch reconstruction with an MLP decoder."""

    def __init__(self) -> None:
        """Build the patch encoder and the reconstruction decoder."""
        super().__init__()
        self.encoder = PatchTransformer()
        width, patch_size = PatchTransformer.width, PatchTransformer.patch_size
        self.decoder = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(),
                                     nn.Linear(2 * width, LEADS * patch_size))

    def forward(self, signal: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Reconstruct masked raw patches.

        Parameters
        ----------
        signal : torch.Tensor
            Signals of shape [batch, 12, 1000].

        Returns
        -------
        tuple[torch.Tensor, dict[str, float]]
            Mean squared error on masked patches and its detached value.
        """
        patch_size = PatchTransformer.patch_size
        mask = random_mask(len(signal), signal.device)
        prediction = self.decoder(self.encoder.tokens(signal, mask))
        target = signal.unfold(2, patch_size, patch_size).permute(0, 2, 1, 3).flatten(2)
        loss = F.mse_loss(prediction[mask], target[mask])
        return loss, {"reconstruction": float(loss.detach())}


class JEPA(nn.Module):
    """
    Masked latent prediction with an EMA teacher and a variance penalty.

    This compact JEPA-inspired experiment is not the published ECG-JEPA/CroPA architecture.
    """

    def __init__(self) -> None:
        """Build the student, its frozen EMA teacher copy and the predictor."""
        super().__init__()
        width = PatchTransformer.width
        self.encoder = PatchTransformer()
        self.teacher = copy.deepcopy(self.encoder)
        self.teacher.requires_grad_(False)
        self.predictor = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))

    def forward(self, signal: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Predict teacher latents of masked patches.

        Parameters
        ----------
        signal : torch.Tensor
            Signals of shape [batch, 12, 1000].

        Returns
        -------
        tuple[torch.Tensor, dict[str, float]]
            Scalar loss and detached diagnostics.
        """
        mask = random_mask(len(signal), signal.device)
        context = self.encoder.tokens(signal, mask)
        prediction = self.predictor(context)
        self.teacher.eval()
        with torch.no_grad():
            target = F.layer_norm(self.teacher.tokens(signal), (PatchTransformer.width,))
        regression = F.smooth_l1_loss(prediction[mask], target[mask])
        # Across-record variance guards against a representation consisting only of position.
        record_features = context.mean(1)
        variance = F.relu(1.0 - torch.sqrt(record_features.var(0, unbiased=False) + 1e-4)).mean()
        loss = regression + VARIANCE_WEIGHT * variance
        return loss, {"latent_prediction": float(regression.detach()),
                      "variance_penalty": float(variance.detach()),
                      "feature_std": float(record_features.std(0, unbiased=False).mean().detach())}

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        """
        Move teacher parameters toward the student by exponential averaging.

        Parameters
        ----------
        momentum : float
            Weight kept on the current teacher parameters.
        """
        for target, online in zip(self.teacher.parameters(), self.encoder.parameters(), strict=True):
            target.mul_(momentum).add_(online, alpha=1 - momentum)
