"""Compact encoders and SSL objectives, not published model reproductions."""

import copy

import torch
from torch import nn
from torch.nn import functional as F


class CNN(nn.Module):
    feature_dim = 256

    def __init__(self):
        super().__init__()
        layers = []
        previous = 12
        for channels in (32, 64, 128):
            layers.extend([nn.Conv1d(previous, channels, 7, stride=2, padding=3),
                           nn.GroupNorm(8, channels), nn.GELU(),
                           nn.Conv1d(channels, channels, 5, padding=2),
                           nn.GroupNorm(8, channels), nn.GELU()])
            previous = channels
        self.layers = nn.Sequential(*layers)

    def forward(self, signal):
        tokens = self.layers(signal)
        return torch.cat([tokens.mean(-1), tokens.amax(-1)], dim=1)


class PatchTransformer(nn.Module):
    """Joint 12-lead patches of 250 ms; 40 tokens per 10-second ECG."""
    width = 96
    patch_size = 25
    feature_dim = 192

    def __init__(self):
        super().__init__()
        self.patch = nn.Conv1d(12, self.width, self.patch_size, stride=self.patch_size)
        self.position = nn.Parameter(torch.randn(1, 40, self.width) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.width) * 0.02)
        layer = nn.TransformerEncoderLayer(self.width, 4, dim_feedforward=192,
                                           dropout=0.1, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 3, norm=nn.LayerNorm(self.width),
                                             enable_nested_tensor=False)

    def tokens(self, signal, mask=None):
        tokens = self.patch(signal).transpose(1, 2)
        if mask is not None:
            tokens = torch.where(mask.unsqueeze(-1), self.mask_token, tokens)
        return self.encoder(tokens + self.position)

    def forward(self, signal):
        tokens = self.tokens(signal)
        return torch.cat([tokens.mean(1), tokens.amax(1)], dim=1)


class Classifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.LayerNorm(encoder.feature_dim), nn.Dropout(0.2),
                                  nn.Linear(encoder.feature_dim, 1))

    def forward(self, signal):
        return self.head(self.encoder(signal)).squeeze(-1)


def random_mask(batch, device, ratio=0.5):
    # Mask contiguous 500-ms spans. Exactly half the patches are masked for ratio=0.5.
    noise = torch.rand(batch, 20, device=device)
    masked_blocks = noise.argsort(dim=1).argsort(dim=1) < round(20 * ratio)
    return masked_blocks.repeat_interleave(2, dim=1)


class MaskedAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PatchTransformer()
        self.decoder = nn.Sequential(nn.Linear(96, 192), nn.GELU(), nn.Linear(192, 12 * 25))

    def forward(self, signal):
        mask = random_mask(len(signal), signal.device)
        prediction = self.decoder(self.encoder.tokens(signal, mask))
        target = signal.unfold(2, 25, 25).permute(0, 2, 1, 3).flatten(2)
        loss = F.mse_loss(prediction[mask], target[mask])
        return loss, {"reconstruction": float(loss.detach())}


class JEPA(nn.Module):
    """Masked latent prediction with an EMA teacher and a variance penalty.

    This compact JEPA-inspired experiment is not the published ECG-JEPA/CroPA architecture.
    """
    def __init__(self):
        super().__init__()
        self.encoder = PatchTransformer()
        self.teacher = copy.deepcopy(self.encoder)
        self.teacher.requires_grad_(False)
        self.predictor = nn.Sequential(nn.Linear(96, 192), nn.GELU(), nn.Linear(192, 96))

    def forward(self, signal):
        mask = random_mask(len(signal), signal.device)
        context = self.encoder.tokens(signal, mask)
        prediction = self.predictor(context)
        self.teacher.eval()
        with torch.no_grad():
            target = F.layer_norm(self.teacher.tokens(signal), (96,))
        regression = F.smooth_l1_loss(prediction[mask], target[mask])
        # Across-record variance guards against a representation consisting only of position.
        record_features = context.mean(1)
        variance = F.relu(1.0 - torch.sqrt(record_features.var(0, unbiased=False) + 1e-4)).mean()
        loss = regression + 0.1 * variance
        return loss, {"latent_prediction": float(regression.detach()),
                      "variance_penalty": float(variance.detach()),
                      "feature_std": float(record_features.std(0, unbiased=False).mean().detach())}

    @torch.no_grad()
    def update_teacher(self, momentum):
        for target, online in zip(self.teacher.parameters(), self.encoder.parameters()):
            target.mul_(momentum).add_(online, alpha=1 - momentum)
