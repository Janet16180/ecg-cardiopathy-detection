"""Eight-lead morphology/rhythm encoder and matched latent SSL objectives.

This is an experimental architecture, not a published ECG-JEPA reproduction.
"""

import copy

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812 - conventional PyTorch alias

from ecg_experiment.models import Classifier

LEAD_INDICES = (0, 1, 6, 7, 8, 9, 10, 11)  # I, II, V1–V6
LEADS = 8
SAMPLES = 1000
PATCHES = 25
PATCH_SAMPLES = 40  # 400 ms at 100 Hz
RHYTHM_WINDOWS = 10
RHYTHM_SAMPLES = SAMPLES // RHYTHM_WINDOWS
SPAN_PATCHES = 5
WIDTH = 96
HEADS = 4
VARIANCE_WEIGHT = 0.1
INNOVATION_WEIGHTS = (0.0, 0.5)

# The lead classifier is the shared head; its modules and state_dict keys are identical.
LeadClassifier = Classifier


def lead_patch_mask(batch: int, device: torch.device | str) -> torch.Tensor:
    """
    Hide one full lead and two distinct two-second lead spans per record.

    Parameters
    ----------
    batch : int
        Number of records.
    device : torch.device | str
        Device of the mask and of the random draws.

    Returns
    -------
    torch.Tensor
        Boolean mask of shape [batch, 8, 25].
    """
    lead_order = torch.rand(batch, LEADS, device=device).argsort(dim=1)
    mask = torch.zeros(batch, LEADS, PATCHES, dtype=torch.bool, device=device)
    record_index = torch.arange(batch, device=device)
    mask[record_index, lead_order[:, 0], :] = True
    for slot in (1, 2):
        starts = torch.randint(0, PATCHES - SPAN_PATCHES + 1, (batch,), device=device)
        positions = starts[:, None] + torch.arange(SPAN_PATCHES, device=device)[None]
        mask[record_index[:, None], lead_order[:, slot, None], positions] = True
    return mask


def content_targets(clean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Remove lead/time constants before constructing EMA latent targets.

    Parameters
    ----------
    clean : torch.Tensor
        Teacher tokens of shape [batch, 8, 25, 96].

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Ordinary and innovation targets, and the centered tokens and
        other-lead residuals they were normalized from.
    """
    centered = clean - clean.mean(dim=0, keepdim=True)
    ordinary = F.layer_norm(centered, (WIDTH,))
    other_leads = (centered.sum(dim=1, keepdim=True) - centered) / (LEADS - 1)
    residual = centered - other_leads
    innovation = F.layer_norm(residual, (WIDTH,))
    return ordinary, innovation, centered, residual


class LeadMultiscaleEncoder(nn.Module):
    """Shared per-lead morphology, shared rhythm context, and temporal mixing."""

    feature_dim = 2 * WIDTH

    def __init__(self) -> None:
        """Build the morphology/rhythm projections, embeddings and temporal Transformer."""
        super().__init__()
        self.morphology = nn.Linear(PATCH_SAMPLES, WIDTH)
        self.rhythm = nn.Linear(RHYTHM_SAMPLES, WIDTH)
        self.lead_embedding = nn.Parameter(torch.randn(1, LEADS, 1, WIDTH) * 0.02)
        self.time_embedding = nn.Parameter(torch.randn(1, 1, PATCHES, WIDTH) * 0.02)
        self.rhythm_time_embedding = nn.Parameter(torch.randn(1, RHYTHM_WINDOWS, WIDTH) * 0.02)
        self.mask_embedding = nn.Parameter(torch.randn(1, 1, 1, WIDTH) * 0.02)
        self.cross_attention = nn.MultiheadAttention(WIDTH, HEADS, batch_first=True)
        layer = nn.TransformerEncoderLayer(WIDTH, HEADS, dim_feedforward=2 * WIDTH,
                                           dropout=0.1, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, num_layers=2,
                                              norm=nn.LayerNorm(WIDTH),
                                              enable_nested_tensor=False)

    def tokens(self, signal: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Encode per-lead patch tokens, hiding masked raw samples first.

        Parameters
        ----------
        signal : torch.Tensor
            Eight-lead signals of shape [batch, 8, 1000] at 100 Hz.
        mask : torch.Tensor | None
            Boolean patch mask of shape [batch, 8, 25].

        Returns
        -------
        torch.Tensor
            Tokens of shape [batch, 8, 25, 96].

        Raises
        ------
        ValueError
            If the signal or mask shape is wrong.
        """
        if signal.ndim != 3 or signal.shape[1:] != (LEADS, SAMPLES):
            raise ValueError(f"Expected (batch, 8, 1000), got {tuple(signal.shape)}")
        if mask is not None and mask.shape != (len(signal), LEADS, PATCHES):
            raise ValueError(f"Unexpected mask shape: {tuple(mask.shape)}")
        if mask is not None:
            # Mask raw samples before either branch can see them. Patch boundaries
            # align exactly with the morphology projection's 40-sample windows.
            signal = signal.masked_fill(mask.repeat_interleave(PATCH_SAMPLES, dim=-1), 0)
        batch = len(signal)
        patches = signal.reshape(batch, LEADS, PATCHES, PATCH_SAMPLES)
        morphology = self.morphology(patches) + self.lead_embedding + self.time_embedding
        if mask is not None:
            morphology = morphology + mask[..., None] * self.mask_embedding
        # One-second rhythm windows are computed from the same masked raw signal.
        rhythm = signal.reshape(batch, LEADS, RHYTHM_WINDOWS, RHYTHM_SAMPLES).mean(dim=1)
        rhythm = self.rhythm(rhythm) + self.rhythm_time_embedding
        queries = morphology.reshape(batch, LEADS * PATCHES, WIDTH)
        attended, _ = self.cross_attention(queries, rhythm, rhythm, need_weights=False)
        mixed = (queries + attended).reshape(batch * LEADS, PATCHES, WIDTH)
        return self.temporal(mixed).reshape(batch, LEADS, PATCHES, WIDTH)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Encode and mean/max pool all lead-patch tokens.

        Parameters
        ----------
        signal : torch.Tensor
            Eight-lead signals of shape [batch, 8, 1000].

        Returns
        -------
        torch.Tensor
            Features of shape [batch, 192].
        """
        tokens = self.tokens(signal).flatten(1, 2)
        return torch.cat((tokens.mean(dim=1), tokens.amax(dim=1)), dim=-1)


def _predictor() -> nn.Sequential:
    """Two-layer GELU MLP on token width."""
    return nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.GELU(), nn.Linear(WIDTH, WIDTH))


class LeadSSL(nn.Module):
    """EMA teacher with ordinary latent and optional lead-residual targets."""

    def __init__(self, innovation_weight: float) -> None:
        """
        Build the student, frozen EMA teacher and both predictors.

        Parameters
        ----------
        innovation_weight : float
            Weight of the lead-residual objective; 0 or 0.5.

        Raises
        ------
        ValueError
            If the weight is not one of the compared values.
        """
        super().__init__()
        if innovation_weight not in INNOVATION_WEIGHTS:
            raise ValueError("Innovation weight must be 0 or 0.5 for this comparison")
        self.innovation_weight = innovation_weight
        self.encoder = LeadMultiscaleEncoder()
        self.teacher = copy.deepcopy(self.encoder).requires_grad_(False)
        self.ordinary_predictor = _predictor()
        self.innovation_predictor = _predictor()

    def forward(self, signal: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute the masked latent objectives and the record variance floor.

        Parameters
        ----------
        signal : torch.Tensor
            Eight-lead signals of shape [batch, 8, 1000]; at least two records.

        Returns
        -------
        tuple[torch.Tensor, dict[str, float]]
            Scalar loss and detached diagnostics.

        Raises
        ------
        ValueError
            If the batch has fewer than two records.
        """
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
        loss = ordinary + self.innovation_weight * innovation + VARIANCE_WEIGHT * variance
        content_std = centered.std(dim=0, unbiased=False).mean().detach()
        residual_std = residual.std(dim=0, unbiased=False).mean().detach()
        details = {"ordinary_loss": float(ordinary.detach()),
                   "innovation_loss": float(innovation.detach()),
                   "variance_penalty": float(variance.detach()),
                   "student_record_std": float(feature_std.mean().detach()),
                   "teacher_content_std_across_records_before_norm": float(content_std),
                   "innovation_residual_std_across_records_before_norm": float(residual_std)}
        return loss, details

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
            target.lerp_(online, 1.0 - momentum)
