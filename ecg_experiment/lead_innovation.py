"""Eight-lead morphology/rhythm encoder and matched latent SSL objectives.

This is an experimental architecture, not a published ECG-JEPA reproduction.
"""

import copy

import torch
from torch import nn
from torch.nn import functional as F


LEAD_INDICES = (0, 1, 6, 7, 8, 9, 10, 11)  # I, II, V1–V6
LEADS = 8
PATCHES = 25
PATCH_SAMPLES = 40  # 400 ms at 100 Hz
WIDTH = 96


def lead_patch_mask(batch, device):
    """Hide one full lead and two distinct two-second lead spans per record."""
    lead_order = torch.rand(batch, LEADS, device=device).argsort(dim=1)
    mask = torch.zeros(batch, LEADS, PATCHES, dtype=torch.bool, device=device)
    record_index = torch.arange(batch, device=device)
    mask[record_index, lead_order[:, 0], :] = True
    for slot in (1, 2):
        starts = torch.randint(0, PATCHES - 5 + 1, (batch,), device=device)
        positions = starts[:, None] + torch.arange(5, device=device)[None]
        mask[record_index[:, None], lead_order[:, slot, None], positions] = True
    return mask


def content_targets(clean):
    """Remove lead/time constants before constructing EMA latent targets."""
    centered = clean - clean.mean(dim=0, keepdim=True)
    ordinary = F.layer_norm(centered, (WIDTH,))
    other_leads = (centered.sum(dim=1, keepdim=True) - centered) / (LEADS - 1)
    residual = centered - other_leads
    innovation = F.layer_norm(residual, (WIDTH,))
    return ordinary, innovation, centered, residual


class LeadMultiscaleEncoder(nn.Module):
    """Shared per-lead morphology, shared rhythm context, and temporal mixing."""

    feature_dim = 2 * WIDTH

    def __init__(self):
        super().__init__()
        self.morphology = nn.Linear(PATCH_SAMPLES, WIDTH)
        self.rhythm = nn.Linear(100, WIDTH)
        self.lead_embedding = nn.Parameter(torch.randn(1, LEADS, 1, WIDTH) * 0.02)
        self.time_embedding = nn.Parameter(torch.randn(1, 1, PATCHES, WIDTH) * 0.02)
        self.rhythm_time_embedding = nn.Parameter(torch.randn(1, 10, WIDTH) * 0.02)
        self.mask_embedding = nn.Parameter(torch.randn(1, 1, 1, WIDTH) * 0.02)
        self.cross_attention = nn.MultiheadAttention(WIDTH, 4, batch_first=True)
        layer = nn.TransformerEncoderLayer(WIDTH, 4, dim_feedforward=192,
                                           dropout=0.1, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, num_layers=2,
                                              norm=nn.LayerNorm(WIDTH),
                                              enable_nested_tensor=False)

    def tokens(self, signal, mask=None):
        if signal.ndim != 3 or signal.shape[1:] != (LEADS, 1000):
            raise ValueError(f"Expected (batch, 8, 1000), got {tuple(signal.shape)}")
        if mask is not None:
            if mask.shape != (len(signal), LEADS, PATCHES):
                raise ValueError(f"Unexpected mask shape: {tuple(mask.shape)}")
            # Mask raw samples before either branch can see them. Patch boundaries
            # align exactly with the morphology projection's 40-sample windows.
            signal = signal.masked_fill(mask.repeat_interleave(PATCH_SAMPLES, dim=-1), 0)
        batch = len(signal)
        patches = signal.reshape(batch, LEADS, PATCHES, PATCH_SAMPLES)
        morphology = self.morphology(patches) + self.lead_embedding + self.time_embedding
        if mask is not None:
            morphology = morphology + mask[..., None] * self.mask_embedding
        # One-second rhythm windows are computed from the same masked raw signal.
        rhythm = signal.reshape(batch, LEADS, 10, 100).mean(dim=1)
        rhythm = self.rhythm(rhythm) + self.rhythm_time_embedding
        queries = morphology.reshape(batch, LEADS * PATCHES, WIDTH)
        attended, _ = self.cross_attention(queries, rhythm, rhythm, need_weights=False)
        mixed = (queries + attended).reshape(batch * LEADS, PATCHES, WIDTH)
        return self.temporal(mixed).reshape(batch, LEADS, PATCHES, WIDTH)

    def forward(self, signal):
        tokens = self.tokens(signal).flatten(1, 2)
        return torch.cat((tokens.mean(dim=1), tokens.amax(dim=1)), dim=-1)


class LeadSSL(nn.Module):
    """EMA teacher with ordinary latent and optional lead-residual targets."""

    def __init__(self, innovation_weight):
        super().__init__()
        if innovation_weight not in (0.0, 0.5):
            raise ValueError("Innovation weight must be 0 or 0.5 for this comparison")
        self.innovation_weight = innovation_weight
        self.encoder = LeadMultiscaleEncoder()
        self.teacher = copy.deepcopy(self.encoder).requires_grad_(False)
        self.ordinary_predictor = nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.GELU(),
                                                nn.Linear(WIDTH, WIDTH))
        self.innovation_predictor = nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.GELU(),
                                                  nn.Linear(WIDTH, WIDTH))

    def forward(self, signal):
        if len(signal) < 2:
            raise ValueError("EMA content targets require at least two records per batch")
        mask = lead_patch_mask(len(signal), signal.device)
        student = self.encoder.tokens(signal, mask)
        self.teacher.eval()
        with torch.no_grad():
            clean = self.teacher.tokens(signal)
            ordinary_target, innovation_target, centered, residual = content_targets(clean)
        ordinary = F.smooth_l1_loss(self.ordinary_predictor(student)[mask],
                                    ordinary_target[mask])
        if self.innovation_weight:
            innovation = F.smooth_l1_loss(self.innovation_predictor(student)[mask],
                                          innovation_target[mask])
        else:
            innovation = ordinary.new_zeros(())
        # A record-level variance floor prevents identical outputs across ECGs.
        record_features = student.mean(dim=(1, 2))
        feature_std = torch.sqrt(record_features.var(dim=0, unbiased=False) + 1e-4)
        variance = F.relu(1.0 - feature_std).mean()
        loss = ordinary + self.innovation_weight * innovation + 0.1 * variance
        details = {"ordinary_loss": float(ordinary.detach()),
                   "innovation_loss": float(innovation.detach()),
                   "variance_penalty": float(variance.detach()),
                   "student_record_std": float(feature_std.mean().detach()),
                   "teacher_content_std_across_records_before_norm": float(centered.std(dim=0, unbiased=False).mean().detach()),
                   "innovation_residual_std_across_records_before_norm": float(residual.std(dim=0, unbiased=False).mean().detach())}
        return loss, details

    @torch.no_grad()
    def update_teacher(self, momentum):
        for target, online in zip(self.teacher.parameters(), self.encoder.parameters()):
            target.lerp_(online, 1.0 - momentum)


class LeadClassifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.LayerNorm(encoder.feature_dim), nn.Dropout(0.2),
                                  nn.Linear(encoder.feature_dim, 1))

    def forward(self, signal):
        return self.head(self.encoder(signal)).squeeze(-1)
