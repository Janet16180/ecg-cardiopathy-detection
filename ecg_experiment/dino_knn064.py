"""
Experiment 064: beat-level lead x 40 ms tokens, SimDINOv2 + iBOT pretraining, phase-matched patch kNN.

Each 12-lead beat is a window from R - 400 ms to R + 560 ms at 250 Hz, cut into 24 patches of 40 ms per
lead (288 tokens). A small ViT is pretrained self-supervised on unlabeled beats; each patch of a test beat
is scored by its cosine distance to the nearest patches of clean normal beats with the same lead and a
neighbouring R-relative phase, then mapped back to the 500 Hz record. See
``docs/experiment-064-dino-patch-knn.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt, resample_poly
from torch import nn
from torch.nn import functional as F  # noqa: N812

from ecg_experiment import ROOT
from ecg_experiment.clean_cohorts import read_signal
from ecg_experiment.clean_cohorts_v2 import CHALLENGE_ROOT
from ecg_experiment.cpc_simdino040 import upstream_loss
from ecg_experiment.ecgad import load_split
from ecg_experiment.fragment_localization import r_peaks
from ecg_experiment.locbench_data import load_partition
from ecg_experiment.waveforms import read_record

FS_RAW = 500
FS = 250
LEADS = 12
PATCH = 10
PHASES = 24
WINDOW = PATCH * PHASES
PRE = 100
MARGIN = 10
STORED = WINDOW + 2 * MARGIN
CLIP_MV = 8.0
BAND_HZ = (0.5, 40.0)
TOKENS = LEADS * PHASES
LOCAL_LEADS = 6
LOCAL_PHASES = 12
GLOBAL_VIEWS = 2
LOCAL_VIEWS = 4
CODING_WEIGHT = 0.1
PATCH_RATE_TOKENS = 2048
EMA_START = 0.996
FLAT_LEAD_STD_MV = 1e-3
POOL_MANIFEST = "data/processed/clean_100k_v4/train_manifest.csv"
CHALLENGE_SPLITS = "data/processed/challenge_splits_v1/rows.csv"
CURATED_SOURCES = frozenset(
    {"ptbxl", "ningbo", "chapman_shaoxing", "georgia", "cpsc_2018", "cpsc_2018_extra"}
)


@dataclass(frozen=True)
class Beats:
    """Beat windows of one record at 250 Hz, aligned at ``start`` (R - 400 ms) for ``WINDOW`` samples."""

    windows: np.ndarray
    valid: np.ndarray
    starts: np.ndarray
    prematurity: np.ndarray


def preprocess(signal: np.ndarray) -> np.ndarray:
    """
    Band-pass, downsample to 250 Hz and clip one 500 Hz record, keeping millivolts.

    Parameters
    ----------
    signal : np.ndarray
        ``[12, 5000]`` waveform in mV at 500 Hz.

    Returns
    -------
    np.ndarray
        Float32 ``[12, 2500]`` waveform in mV at 250 Hz.
    """
    b, a = butter(2, BAND_HZ, btype="band", fs=FS_RAW)
    filtered = filtfilt(b, a, np.nan_to_num(signal.astype(np.float64)), axis=1)
    return np.clip(resample_poly(filtered, 1, 2, axis=1), -CLIP_MV, CLIP_MV).astype(np.float32)


def peaks_250(signal: np.ndarray) -> np.ndarray:
    """
    Frozen R peaks of a raw 500 Hz record, in 250 Hz samples.

    Parameters
    ----------
    signal : np.ndarray
        ``[12, 5000]`` raw waveform in mV.

    Returns
    -------
    np.ndarray
        Ascending integer peak positions at 250 Hz.
    """
    return np.rint(r_peaks(signal, FS_RAW) / 2).astype(np.int64)


def prematurity(peaks: np.ndarray) -> np.ndarray:
    """
    How early each beat comes: ``max(0, 1 - preceding RR / median RR)``, 0 for the first beat.

    Parameters
    ----------
    peaks : np.ndarray
        R peaks of one record.

    Returns
    -------
    np.ndarray
        One value per peak.
    """
    if len(peaks) < 3:
        return np.zeros(len(peaks))
    rr = np.diff(peaks).astype(float)
    return np.concatenate([[0.0], np.maximum(0.0, 1 - rr / np.median(rr))])


def pool_windows(filtered: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """
    Complete pretraining windows with a 40 ms margin on each side for crop jitter.

    Parameters
    ----------
    filtered : np.ndarray
        ``preprocess`` output, ``[12, 2500]``.
    peaks : np.ndarray
        R peaks at 250 Hz.

    Returns
    -------
    np.ndarray
        Float32 ``[beats, 12, STORED]``; beats whose stored window leaves the record are dropped.
    """
    starts = peaks - PRE - MARGIN
    keep = (starts >= 0) & (starts + STORED <= filtered.shape[1])
    return (
        np.stack([filtered[:, s : s + STORED] for s in starts[keep]])
        if keep.any()
        else np.zeros((0, LEADS, STORED), np.float32)
    )


def score_beats(raw: np.ndarray) -> Beats:
    """
    Every detected beat of a record as a scoring window, including beats cut by the record edges.

    Parameters
    ----------
    raw : np.ndarray
        ``[12, 5000]`` raw waveform in mV at 500 Hz.

    Returns
    -------
    Beats
        ``windows`` ``[beats, 12, WINDOW]`` (zero outside the record), ``valid`` ``[beats, PHASES]``
        (patch lies entirely inside the record), ``starts`` at 250 Hz and ``prematurity``.
    """
    filtered = preprocess(raw)
    peaks = peaks_250(raw)
    starts = peaks - PRE
    length = filtered.shape[1]
    phase_start = starts[:, None] + PATCH * np.arange(PHASES)[None]
    valid = (phase_start >= 0) & (phase_start + PATCH <= length)
    keep = valid.any(axis=1)
    padded = np.pad(filtered, ((0, 0), (WINDOW, WINDOW)))
    windows = (
        np.stack([padded[:, s + WINDOW : s + 2 * WINDOW] for s in starts[keep]])
        if keep.any()
        else (np.zeros((0, LEADS, WINDOW), np.float32))
    )
    return Beats(windows, valid[keep], starts[keep], prematurity(peaks)[keep])


def has_flat_lead(signal: np.ndarray) -> bool:
    """
    Whether any lead of a raw record is flat (standard deviation below 1 microvolt).

    Parameters
    ----------
    signal : np.ndarray
        ``[12, samples]`` waveform in mV.

    Returns
    -------
    bool
        True if a lead is flat.
    """
    return bool((np.nanstd(signal, axis=1) < FLAT_LEAD_STD_MV).any())


def to_patches(windows: torch.Tensor) -> torch.Tensor:
    """
    Cut ``[batch, 12, 240]`` windows into lead-major ``[batch, 288, 10]`` patches.

    Parameters
    ----------
    windows : torch.Tensor
        Beat windows.

    Returns
    -------
    torch.Tensor
        Token ``lead * 24 + phase`` holds that lead's 40 ms patch.
    """
    return windows.reshape(len(windows), LEADS, PHASES, PATCH).reshape(len(windows), TOKENS, PATCH)


def token_grid(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Lead and phase index of each of the 288 tokens.

    Parameters
    ----------
    device : torch.device
        Target device.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Two long tensors of length 288.
    """
    tokens = torch.arange(TOKENS, device=device)
    return tokens // PHASES, tokens % PHASES


class Block(nn.Module):
    """Pre-normalized self-attention and MLP block."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        """
        Apply the block.

        Parameters
        ----------
        x : torch.Tensor
            ``[batch, tokens, width]``.
        attn_mask : torch.Tensor or None
            Boolean ``[batch, 1, 1, tokens]``, True where a key may be attended.

        Returns
        -------
        torch.Tensor
            Same shape as ``x``.
        """
        batch, tokens, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(batch, tokens, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x + self.proj(attended.transpose(1, 2).reshape(batch, tokens, width))
        return x + self.mlp(self.norm2(x))


class BeatViT(nn.Module):
    """ViT over a set of lead x phase patch tokens with a CLS token."""

    def __init__(self, width: int = 256, depth: int = 6, heads: int = 8) -> None:
        super().__init__()
        self.embed = nn.Linear(PATCH, width)
        self.lead = nn.Parameter(torch.randn(LEADS, width) * 0.02)
        self.phase = nn.Parameter(torch.randn(PHASES, width) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.mask_token = nn.Parameter(torch.zeros(width))
        self.blocks = nn.ModuleList(Block(width, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(width)
        self.apply(_init)

    def forward(
        self,
        patches: torch.Tensor,
        leads: torch.Tensor,
        phases: torch.Tensor,
        masked: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a token set.

        Parameters
        ----------
        patches : torch.Tensor
            ``[batch, tokens, 10]`` patches in mV.
        leads, phases : torch.Tensor
            Long ``[tokens]`` or ``[batch, tokens]`` indices.
        masked : torch.Tensor or None
            Boolean ``[batch, tokens]``; masked patches are replaced by the mask token (iBOT).
        valid : torch.Tensor or None
            Boolean ``[batch, tokens]``; invalid tokens are never attended to.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            CLS ``[batch, width]`` and patch tokens ``[batch, tokens, width]`` after the final norm.
        """
        x = self.embed(patches)
        if masked is not None:
            x = torch.where(masked[..., None], self.mask_token.to(x.dtype), x)
        x = x + self.lead[leads] + self.phase[phases]
        x = torch.cat([self.cls.expand(len(x), -1, -1).to(x.dtype), x], dim=1)
        attn_mask = None
        if valid is not None:
            keys = torch.cat([torch.ones_like(valid[:, :1]), valid], dim=1)
            attn_mask = keys[:, None, None, :]
        for block in self.blocks:
            x = block(x, attn_mask)
        x = self.norm(x)
        return x[:, 0], x[:, 1:]


class Head(nn.Module):
    """Three-layer projection head without prototypes (SimDINO)."""

    def __init__(self, width: int = 256, hidden: int = 1024, out: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, out)
        )
        self.apply(_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project features.

        Parameters
        ----------
        x : torch.Tensor
            ``[..., width]``.

        Returns
        -------
        torch.Tensor
            ``[..., out]``.
        """
        return self.net(x)


class Network(nn.Module):
    """Backbone plus CLS and patch heads; the student and the EMA teacher are two instances."""

    def __init__(self, width: int = 256, depth: int = 6, heads: int = 8) -> None:
        super().__init__()
        self.backbone = BeatViT(width, depth, heads)
        self.cls_head = Head(width)
        self.patch_head = Head(width)


def _init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        nn.init.zeros_(module.bias)


def crop(windows: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """
    Per-beat crop of ``WINDOW`` samples from stored windows.

    Parameters
    ----------
    windows : torch.Tensor
        ``[batch, 12, STORED]``.
    offsets : torch.Tensor
        Long ``[batch]`` in ``[0, 2 * MARGIN]``.

    Returns
    -------
    torch.Tensor
        ``[batch, 12, WINDOW]``.
    """
    index = offsets[:, None] + torch.arange(WINDOW, device=windows.device)[None]
    return windows.gather(2, index[:, None, :].expand(-1, LEADS, -1))


def augment(windows: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """
    Crop jitter of up to 40 ms, a global gain of up to 10 percent, a per-lead offset and white noise.

    Parameters
    ----------
    windows : torch.Tensor
        Stored windows ``[batch, 12, STORED]`` in mV.
    generator : torch.Generator
        Generator on the windows' device.

    Returns
    -------
    torch.Tensor
        Augmented ``[batch, 12, WINDOW]`` float32 windows.
    """
    batch, device = len(windows), windows.device
    offsets = torch.randint(0, 2 * MARGIN + 1, (batch,), device=device, generator=generator)
    x = crop(windows.float(), offsets)
    gain = torch.exp((torch.rand(batch, 1, 1, device=device, generator=generator) * 2 - 1) * math.log(1.1))
    offset = torch.randn(batch, LEADS, 1, device=device, generator=generator) * 0.02
    sigma = torch.rand(batch, 1, 1, device=device, generator=generator) * 0.02
    return x * gain + offset + sigma * torch.randn(x.shape, device=device, generator=generator)


def block_masks(batch: int, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    """
    Draw iBOT masks: one to three blocks, each a random lead subset over 2 to 8 contiguous phases.

    Parameters
    ----------
    batch : int
        Number of beats.
    generator : torch.Generator
        Generator on ``device``.
    device : torch.device
        Target device.

    Returns
    -------
    torch.Tensor
        Boolean ``[batch, 288]`` in token order.
    """
    blocks = 3
    lead_share = 0.2 + 0.8 * torch.rand(batch, blocks, 1, device=device, generator=generator)
    leads = torch.rand(batch, blocks, LEADS, device=device, generator=generator) < lead_share
    anchor = torch.randint(0, LEADS, (batch, blocks, 1), device=device, generator=generator)
    leads = leads | (torch.arange(LEADS, device=device) == anchor)
    width = torch.randint(2, 9, (batch, blocks, 1), device=device, generator=generator)
    start = (torch.rand(batch, blocks, 1, device=device, generator=generator) * (PHASES - width + 1)).long()
    grid = torch.arange(PHASES, device=device)
    phases = (grid >= start) & (grid < start + width)
    active = torch.arange(blocks, device=device)[None] < torch.randint(
        1, blocks + 1, (batch, 1), device=device, generator=generator
    )
    mask = (leads[..., :, None] & phases[..., None, :] & active[..., None, None]).any(dim=1)
    return mask.reshape(batch, TOKENS)


def local_tokens(batch: int, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    """
    Token indices of a local view: 6 random leads x 12 contiguous phases per beat.

    Parameters
    ----------
    batch : int
        Number of beats.
    generator : torch.Generator
        Generator on ``device``.
    device : torch.device
        Target device.

    Returns
    -------
    torch.Tensor
        Long ``[batch, 72]``.
    """
    leads = torch.rand(batch, LEADS, device=device, generator=generator).argsort(dim=1)[:, :LOCAL_LEADS]
    start = torch.randint(0, PHASES - LOCAL_PHASES + 1, (batch, 1), device=device, generator=generator)
    phases = start + torch.arange(LOCAL_PHASES, device=device)[None]
    return (leads[:, :, None] * PHASES + phases[:, None, :]).reshape(batch, -1)


def simdino_loss(
    student: Network,
    teacher: Network,
    stored: torch.Tensor,
    generator: torch.Generator,
    loss_module: nn.Module,
) -> dict[str, torch.Tensor]:
    """
    SimDINOv2 CLS loss (compression + coding-rate expansion) and iBOT masked-patch cosine loss.

    Parameters
    ----------
    student, teacher : Network
        Student and frozen EMA teacher.
    stored : torch.Tensor
        ``[batch, 12, STORED]`` beat windows.
    generator : torch.Generator
        Augmentation and mask generator on the windows' device.
    loss_module : nn.Module
        Upstream SimDINOv2 loss (``cpc_simdino040.upstream_loss``).

    Returns
    -------
    dict[str, torch.Tensor]
        ``cls_compression``, ``cls_expansion``, ``patch``, ``patch_expansion`` and ``total``.
    """
    batch, device = len(stored), stored.device
    leads, phases = token_grid(device)
    globals_ = [to_patches(augment(stored, generator)) for _ in range(GLOBAL_VIEWS)]
    masks = [block_masks(batch, generator, device) for _ in range(GLOBAL_VIEWS)]
    with torch.no_grad():
        teacher_out = [teacher.backbone(view, leads, phases) for view in globals_]
        teacher_cls = torch.stack([teacher.cls_head(cls) for cls, _ in teacher_out]).float()
        teacher_patch = [teacher.patch_head(patch).float() for _, patch in teacher_out]
    student_cls, patch_terms, patch_features = [], [], []
    for view, mask, target in zip(globals_, masks, teacher_patch, strict=True):
        cls, patch = student.backbone(view, leads, phases, masked=mask)
        student_cls.append(student.cls_head(cls))
        projected = student.patch_head(patch).float()
        distance = 1 - F.cosine_similarity(projected, target, dim=-1)
        patch_terms.append((distance * mask).sum() / mask.sum().clamp(min=1))
        patch_features.append(projected.reshape(-1, projected.shape[-1]))
    for _ in range(LOCAL_VIEWS):
        index = local_tokens(batch, generator, device)
        view = to_patches(augment(stored, generator))
        patches = view.gather(1, index[..., None].expand(-1, -1, PATCH))
        cls, _ = student.backbone(patches, leads[index], phases[index])
        student_cls.append(student.cls_head(cls))
    student_stack = torch.stack(student_cls).float()
    features = torch.cat(patch_features)
    pick = torch.randperm(len(features), device=device, generator=generator)[:PATCH_RATE_TOKENS]
    with torch.autocast(device.type, enabled=False):
        compression, expansion = loss_module(student_stack, teacher_cls)
        patch_expansion = loss_module.calc_expansion(F.normalize(features[pick], dim=-1)[None])
    patch = torch.stack(patch_terms).mean()
    total = compression + CODING_WEIGHT * expansion + patch + CODING_WEIGHT * patch_expansion
    return {
        "cls_compression": compression,
        "cls_expansion": expansion,
        "patch": patch,
        "patch_expansion": patch_expansion,
        "total": total,
    }


def new_loss_module() -> nn.Module:
    """
    Load the upstream SimDINOv2 loss with Experiment 040's coefficients (eps 0.05).

    Returns
    -------
    nn.Module
        Loss exposing ``forward`` (compression, expansion) and ``calc_expansion``.
    """
    return upstream_loss()


def ema_momentum(step: int, total: int) -> float:
    """
    Cosine EMA schedule from 0.996 to 1.

    Parameters
    ----------
    step : int
        Completed optimizer steps.
    total : int
        Planned steps.

    Returns
    -------
    float
        Teacher momentum.
    """
    return 1 - (1 - EMA_START) * (math.cos(math.pi * min(step, total) / total) + 1) / 2


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, momentum: float) -> None:
    """
    ``teacher = momentum * teacher + (1 - momentum) * student``.

    Parameters
    ----------
    teacher, student : nn.Module
        Matching modules.
    momentum : float
        EMA momentum.
    """
    for t, s in zip(teacher.parameters(), student.parameters(), strict=True):
        t.mul_(momentum).add_(s.detach(), alpha=1 - momentum)


def learning_rate(step: int, total: int, peak: float, warmup: int) -> float:
    """
    Linear warmup then cosine decay to 1 percent of the peak.

    Parameters
    ----------
    step : int
        Current step.
    total : int
        Planned steps.
    peak : float
        Peak learning rate.
    warmup : int
        Warmup steps.

    Returns
    -------
    float
        Learning rate.
    """
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.01 + 0.99 * (1 + math.cos(math.pi * min(progress, 1.0))) / 2)


@torch.no_grad()
def patch_features(backbone: BeatViT | None, beats: Beats, device: torch.device) -> torch.Tensor:
    """
    Patch descriptors of every token of a set of beats.

    Parameters
    ----------
    backbone : BeatViT or None
        Teacher backbone (L2-normalized final patch tokens), or None for raw 40 ms waveform patches.
    beats : Beats
        Scoring windows.
    device : torch.device
        Compute device.

    Returns
    -------
    torch.Tensor
        ``[beats, 12, 24, dim]`` (float16 for learned features, float32 raw patches in mV).
    """
    patches = to_patches(torch.from_numpy(beats.windows).to(device))
    if backbone is None:
        return patches.reshape(len(patches), LEADS, PHASES, PATCH).float()
    valid = torch.from_numpy(beats.valid).to(device)[:, None, :].expand(-1, LEADS, -1)
    valid = valid.reshape(len(patches), -1)
    leads, phases = token_grid(device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        _, tokens = backbone(patches, leads, phases, valid=valid)
    tokens = F.normalize(tokens.float(), dim=-1)
    return tokens.reshape(len(patches), LEADS, PHASES, -1).half()


def phase_candidates(phase: int, reach: int) -> slice:
    """
    Bank phase bins compared with a query phase.

    Parameters
    ----------
    phase : int
        Query phase bin.
    reach : int
        Neighbouring bins on each side.

    Returns
    -------
    slice
        Bins ``phase - reach`` to ``phase + reach`` clipped to the window.
    """
    return slice(max(0, phase - reach), min(PHASES, phase + reach + 1))


@torch.no_grad()
def knn_distance(queries: torch.Tensor, bank: torch.Tensor, k: int, reach: int, cosine: bool) -> torch.Tensor:
    """
    Mean distance of each query patch to its ``k`` nearest bank patches of the same lead and nearby phase.

    Parameters
    ----------
    queries : torch.Tensor
        ``[beats, 12, 24, dim]``.
    bank : torch.Tensor
        ``[12, 24, size, dim]`` on the same device.
    k : int
        Neighbours.
    reach : int
        Neighbouring phase bins on each side.
    cosine : bool
        Cosine distance of normalized features if True, else Euclidean distance.

    Returns
    -------
    torch.Tensor
        Float32 ``[beats, 12, 24]``.
    """
    out = torch.empty(queries.shape[:3], device=queries.device)
    for lead in range(LEADS):
        for phase in range(PHASES):
            candidates = bank[lead, phase_candidates(phase, reach)].reshape(-1, bank.shape[-1])
            query = queries[:, lead, phase]
            if cosine:
                nearest = (query @ candidates.T).float().topk(k, dim=1).values
                out[:, lead, phase] = 1 - nearest.mean(dim=1)
            else:
                nearest = torch.cdist(query.float(), candidates.float()).topk(k, dim=1, largest=False).values
                out[:, lead, phase] = nearest.mean(dim=1)
    return out


def beat_map(scores: np.ndarray, beats: Beats, samples: int = 5000) -> np.ndarray:
    """
    Broadcast patch scores to the 500 Hz record; overlapping beats take the maximum.

    Parameters
    ----------
    scores : np.ndarray
        ``[beats, 12, 24]``; invalid patches are ignored.
    beats : Beats
        The record's scoring windows.
    samples : int
        Record length at 500 Hz.

    Returns
    -------
    np.ndarray
        Float32 ``[12, samples]``; samples outside every beat get the map's minimum.
    """
    out = np.full((LEADS, samples), -np.inf, np.float32)
    for beat in range(len(scores)):
        for phase in np.flatnonzero(beats.valid[beat]):
            begin = 2 * (beats.starts[beat] + PATCH * phase)
            end = min(samples, begin + 2 * PATCH)
            begin = max(0, begin)
            out[:, begin:end] = np.maximum(out[:, begin:end], scores[beat, :, phase][:, None])
    finite = np.isfinite(out)
    if not finite.any():
        return np.zeros((LEADS, samples), np.float32)
    out[~finite] = out[finite].min()
    return out


def pool_rows() -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Select pretraining records: curated 500 Hz mV rows of ``clean_100k_v4`` minus non-``fit`` PTB-XL.

    EchoNext (no physical unit, filtered at 250 Hz) and CODE-15 (unresolved unit) are left out. The checks
    raise if an ECGAD record, a Challenge test or calibration record (by ID or waveform hash) or Ningbo
    JS13118 is present.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, int]]
        Manifest rows kept, and counts of the exclusion steps.

    Raises
    ------
    ValueError
        If a closed or evaluation record is found after filtering.
    """
    manifest = pd.read_csv(ROOT / POOL_MANIFEST, dtype=str, keep_default_na=False)
    curated = manifest[manifest.block == "curated"]
    partition = load_partition()
    fit_ids = set(partition.ecg_id[partition.partition == "fit"].astype(int))
    evaluation_ids = set(partition.ecg_id[partition.partition == "evaluation"].astype(int))
    ptb = curated.source == "ptbxl"
    ptb_ids = curated.record_id.where(ptb, "ptbxl:-1").str.split(":").str[1].astype(int)
    kept = curated[~ptb | ptb_ids.isin(fit_ids)].reset_index(drop=True)
    kept_ptb = set(kept.record_id[kept.source == "ptbxl"].str.split(":").str[1].astype(int))
    closed = pd.read_csv(ROOT / CHALLENGE_SPLITS, usecols=["source", "record", "split", "signal_sha256"])
    closed = closed[closed.split.isin(["test", "calibration"])]
    closed_ids = set(closed.source + ":" + closed.record)
    if kept_ptb & (set(load_split().ecg_id) | evaluation_ids):
        raise ValueError("ECGAD or evaluation record in the pool")
    if set(kept.record_id) & closed_ids or set(kept.signal_sha256) & set(closed.signal_sha256):
        raise ValueError("Challenge test or calibration record in the pool")
    if "ningbo:JS13118" in set(kept.record_id) or not set(kept.source) <= CURATED_SOURCES:
        raise ValueError("Excluded record or source in the pool")
    counts = {
        "curated_rows": len(curated),
        "ptbxl_curated": int(ptb.sum()),
        "ptbxl_dropped_not_fit": int((ptb & ~ptb_ids.isin(fit_ids)).sum()),
        "ptbxl_dropped_evaluation": int((ptb & ptb_ids.isin(evaluation_ids)).sum()),
        "kept_rows": len(kept),
    }
    return kept, counts


def read_pool_signal(backend: str, path: str, index: str) -> np.ndarray:
    """
    Raw ``[12, 5000]`` mV waveform of one manifest row.

    Parameters
    ----------
    backend : str
        ``shard`` or ``challenge_wfdb``.
    path : str
        Repository-relative path.
    index : str
        Row inside a shard.

    Returns
    -------
    np.ndarray
        Float32 waveform.
    """
    if backend == "challenge_wfdb":
        return read_record(ROOT / CHALLENGE_ROOT, str(Path(path).relative_to(CHALLENGE_ROOT)))
    return read_signal(backend, path, index)
