"""
Finding-specific multiple-instance heads for Experiment 065.

A shared per-lead 1D convolutional encoder turns each lead of a 10-second 12-lead ECG into a feature
sequence at 32 ms resolution. One instance is a (lead, time step) pair. Each PTB-XL statement head scores
every instance, and the record-level statement logit pools the instance logits, so the instance scores
are themselves the evidence map. See ``docs/experiment-065-finding-mil.md``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, sosfiltfilt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional

from ecg_experiment.waveforms import LEADS

FS = 500
SAMPLES = 5000
STRIDE = 16
CROP = 4480
SPLIT_SEED = 65065
HELDOUT_SHARE = 0.15
MIN_TRAIN_POSITIVES = 30
MIN_HELDOUT_POSITIVES = 10
EXCLUDED_HEADS = frozenset({"SR", "SARRH", "SBRAD", "STACH", "VCLVH"})
REFERENCE_HEAD = "NORM"
CLIP_MV = 10.0


@dataclass
class Recipe:
    """Optimizer and stopping settings."""

    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 64
    max_epochs: int = 40
    patience: int = 6


@dataclass
class Fit:
    """Best weights and training history of one network."""

    state: dict[str, torch.Tensor]
    best_epoch: int
    best_score: float
    history: list[dict[str, float]] = field(default_factory=list)


def statement_labels(
    meta: pd.DataFrame, statements: pd.DataFrame, ecg_ids: np.ndarray, codes: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Binary statement labels and their validity mask.

    A diagnostic statement is positive at likelihood at least 50 and undefined (masked) when present
    below 50; a form or rhythm statement is positive when present. Absent statements are negative.

    Parameters
    ----------
    meta : pd.DataFrame
        ``eda.ptbxl.load_metadata`` output indexed by ``ecg_id``.
    statements : pd.DataFrame
        SCP statement table.
    ecg_ids : np.ndarray
        Records in row order.
    codes : list[str]
        Statement codes in head order.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``y`` (n, k) float32 labels and ``mask`` (n, k) float32, 1 where the label is defined.
    """
    diagnostic = {code: statements.loc[code, "diagnostic"] == 1 for code in codes}
    y = np.zeros((len(ecg_ids), len(codes)), np.float32)
    mask = np.ones_like(y)
    for row, ecg_id in enumerate(ecg_ids):
        scp = meta.loc[ecg_id, "scp_codes"]
        for col, code in enumerate(codes):
            if code not in scp:
                continue
            if not diagnostic[code] or scp[code] >= 50:
                y[row, col] = 1.0
            else:
                mask[row, col] = 0.0
    return y, mask


def candidate_codes(meta: pd.DataFrame, ecg_ids: np.ndarray) -> list[str]:
    """
    Every statement present in the given records, minus the excluded heads, sorted.

    Parameters
    ----------
    meta : pd.DataFrame
        PTB-XL metadata.
    ecg_ids : np.ndarray
        Records.

    Returns
    -------
    list[str]
        Statement codes.
    """
    present = set().union(*(meta.loc[ecg_id, "scp_codes"].keys() for ecg_id in ecg_ids))
    return sorted(present - EXCLUDED_HEADS)


def select_heads(y: np.ndarray, mask: np.ndarray, codes: list[str], minimum: int) -> list[str]:
    """
    Statements with at least ``minimum`` defined positives.

    Parameters
    ----------
    y, mask : np.ndarray
        ``statement_labels`` output on the training rows.
    codes : list[str]
        Codes in column order.
    minimum : int
        Minimum count of positives.

    Returns
    -------
    list[str]
        Selected codes in column order.
    """
    counts = (y * mask).sum(axis=0)
    return [code for code, count in zip(codes, counts, strict=True) if count >= minimum]


def heldout_mask(patient_ids: np.ndarray, share: float = HELDOUT_SHARE, seed: int = SPLIT_SEED) -> np.ndarray:
    """
    Mark every record of a random share of patients as held out.

    Parameters
    ----------
    patient_ids : np.ndarray
        Patient of each record.
    share : float
        Share of patients held out.
    seed : int
        Seed of ``numpy.random.default_rng`` applied to the sorted unique patients.

    Returns
    -------
    np.ndarray
        Boolean mask, True for held-out records.
    """
    patients = np.unique(patient_ids)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(patients, size=int(round(share * len(patients))), replace=False)
    return np.isin(patient_ids, chosen)


def preprocess(signals: np.ndarray) -> np.ndarray:
    """
    Zero-phase 0.5-40 Hz band-pass, clipped to +-10 mV.

    Parameters
    ----------
    signals : np.ndarray
        (n, 12, 5000) raw signals in mV.

    Returns
    -------
    np.ndarray
        Filtered float32 signals of the same shape, aligned sample by sample.
    """
    sos = butter(2, [0.5, 40.0], btype="band", fs=FS, output="sos")
    filtered = sosfiltfilt(sos, np.nan_to_num(np.asarray(signals, np.float64)), axis=-1)
    return np.clip(filtered, -CLIP_MV, CLIP_MV).astype(np.float32)


def conv_block(c_in: int, c_out: int, kernel: int, stride: int = 1, dilation: int = 1) -> nn.Sequential:
    """Convolution, batch norm and GELU with centred padding."""
    padding = dilation * (kernel // 2)
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, kernel, stride, padding, dilation=dilation, bias=False),
        nn.BatchNorm1d(c_out),
        nn.GELU(),
    )


class Residual(nn.Module):
    """Dilated residual block with optional cross-lead context."""

    def __init__(self, width: int, dilation: int, cross_lead: bool) -> None:
        super().__init__()
        self.body = nn.Sequential(conv_block(width, width, 5, dilation=dilation), conv_block(width, width, 5))
        self.context = nn.Linear(width, width) if cross_lead else None

    def forward(self, h: torch.Tensor, leads: int) -> torch.Tensor:
        """
        Apply the block.

        Parameters
        ----------
        h : torch.Tensor
            (batch * leads, width, steps).
        leads : int
            Leads per record.

        Returns
        -------
        torch.Tensor
            Output of the same shape.
        """
        out = h + self.body(h)
        if self.context is not None:
            grouped = out.view(-1, leads, out.shape[1], out.shape[2])
            mixed = self.context(grouped.mean(dim=1).transpose(1, 2)).transpose(1, 2)
            out = (grouped + mixed.unsqueeze(1)).view_as(out)
        return out


class LeadMIL(nn.Module):
    """
    Per-lead encoder with lead embedding and per-finding instance logits.

    Parameters
    ----------
    heads : int
        Number of statement heads.
    width : int
        Channels of the encoder.
    pooling : str
        ``"lse"`` (log-mean-exp of instance logits) or ``"attention"`` (MILLET-style conjunctive pooling).
    cross_lead : bool
        Add a lead-mean context to every residual block.
    lse_r : float
        Sharpness of the log-mean-exp pooling.
    """

    def __init__(
        self, heads: int, width: int = 128, pooling: str = "lse", cross_lead: bool = False, lse_r: float = 1.0
    ) -> None:
        super().__init__()
        if pooling not in ("lse", "attention"):
            raise ValueError(f"Unknown pooling {pooling}")
        self.pooling = pooling
        self.lse_r = lse_r
        self.stem = nn.Sequential(
            conv_block(1, 32, 15, 2),
            conv_block(32, 64, 9, 2),
            conv_block(64, width, 9, 2),
            conv_block(width, width, 9, 2),
        )
        self.lead_embedding = nn.Embedding(len(LEADS), width)
        self.blocks = nn.ModuleList(Residual(width, d, cross_lead) for d in (1, 2, 4))
        self.instance = nn.Conv1d(width, heads, 1)
        self.attention = nn.Conv1d(width, heads, 1) if pooling == "attention" else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Score every instance and pool per head.

        Parameters
        ----------
        x : torch.Tensor
            (batch, 12, samples) preprocessed signals.

        Returns
        -------
        tuple
            Bag logits (batch, heads); instance logits (batch, heads, 12, steps); attention weights of the
            same shape (softmax over leads and steps) or None for log-mean-exp pooling.
        """
        batch, leads, samples = x.shape
        h = self.stem(x.reshape(batch * leads, 1, samples))
        lead_ids = torch.arange(leads, device=x.device).repeat(batch)
        h = h + self.lead_embedding(lead_ids).unsqueeze(-1)
        for block in self.blocks:
            h = block(h, leads)
        steps = h.shape[-1]
        z = self.instance(h).float().view(batch, leads, -1, steps).transpose(1, 2)
        flat = z.reshape(batch, z.shape[1], -1)
        if self.attention is None:
            bag = (torch.logsumexp(self.lse_r * flat, dim=-1) - math.log(flat.shape[-1])) / self.lse_r
            return bag, z, None
        alpha = self.attention(h).float().view(batch, leads, -1, steps).transpose(1, 2)
        weights = torch.softmax(alpha.reshape(batch, alpha.shape[1], -1), dim=-1)
        bag = (weights * flat).sum(dim=-1)
        return bag, z, weights.view_as(z)


def masked_bce(bag: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean binary cross-entropy over defined labels."""
    loss = functional.binary_cross_entropy_with_logits(bag, y, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def macro_auroc(
    y: np.ndarray, mask: np.ndarray, scores: np.ndarray, minimum: int
) -> tuple[float, np.ndarray]:
    """
    Per-head AUROC on defined labels, and their mean over heads with at least ``minimum`` positives.

    Returns
    -------
    tuple[float, np.ndarray]
        Mean AUROC and per-head AUROCs (NaN where the head has too few positives or negatives).
    """
    aurocs = np.full(y.shape[1], np.nan)
    for k in range(y.shape[1]):
        keep = mask[:, k] > 0
        positives = y[keep, k].sum()
        if positives >= minimum and keep.sum() - positives >= minimum:
            aurocs[k] = roc_auc_score(y[keep, k], scores[keep, k])
    return float(np.nanmean(aurocs)), aurocs


def batch_tensor(
    signals: np.ndarray, rows: np.ndarray, device: str, crop: int | None, rng: np.random.Generator | None
) -> torch.Tensor:
    """Gather rows, optionally take one random crop per record, and move to the device."""
    if crop is None:
        x = signals[rows]
    else:
        starts = rng.integers(0, signals.shape[-1] - crop + 1, len(rows))
        x = np.stack([signals[r, :, s : s + crop] for r, s in zip(rows, starts, strict=True)])
    return torch.from_numpy(np.ascontiguousarray(x, np.float32)).to(device, non_blocking=True)


@torch.no_grad()
def predict(
    model: LeadMIL,
    signals: np.ndarray,
    rows: np.ndarray,
    device: str,
    batch_size: int = 64,
    keep_instances: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Bag logits, and optionally instance logits and attention weights, on full-length records.

    Returns
    -------
    tuple
        Bag logits (n, heads) float32; instance logits (n, heads, 12, steps) float32 or None; attention
        weights of the same shape or None.
    """
    model.eval()
    bags, instances, weights = [], [], []
    for start in range(0, len(rows), batch_size):
        x = batch_tensor(signals, rows[start : start + batch_size], device, None, None)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            bag, z, w = model(x)
        bags.append(bag.float().cpu().numpy())
        if keep_instances:
            instances.append(z.float().cpu().numpy())
            if w is not None:
                weights.append(w.float().cpu().numpy())
    bag_out = np.concatenate(bags)
    z_out = np.concatenate(instances) if keep_instances else None
    w_out = np.concatenate(weights) if weights else None
    return bag_out, z_out, w_out


def train(
    model: LeadMIL,
    signals: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    train_rows: np.ndarray,
    valid_rows: np.ndarray,
    recipe: Recipe,
    device: str,
    seed: int,
    max_steps: int | None = None,
) -> Fit:
    """
    Train with AdamW on random crops and keep the epoch with the best held-out macro AUROC.

    Parameters
    ----------
    model : LeadMIL
        Network, already on ``device``.
    signals : np.ndarray
        Preprocessed signals of every row.
    y, mask : np.ndarray
        Labels and validity mask of every row.
    train_rows, valid_rows : np.ndarray
        Row positions for training and early stopping.
    recipe : Recipe
        Optimizer settings.
    device : str
        Torch device.
    seed : int
        Seed of the batch order and crops.
    max_steps : int or None
        Stop each epoch after this many steps (smoke tests only).

    Returns
    -------
    Fit
        Best state, its epoch and score, and the per-epoch history.
    """
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=recipe.learning_rate, weight_decay=recipe.weight_decay
    )
    y_t = torch.from_numpy(y).to(device)
    m_t = torch.from_numpy(mask).to(device)
    best = Fit({}, -1, -np.inf)
    for epoch in range(recipe.max_epochs):
        model.train()
        start = time.time()
        order = rng.permutation(train_rows)
        losses = []
        for step, begin in enumerate(range(0, len(order), recipe.batch_size)):
            if max_steps is not None and step >= max_steps:
                break
            rows = np.sort(order[begin : begin + recipe.batch_size])
            x = batch_tensor(signals, rows, device, CROP, rng)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                bag, _, _ = model(x)
            loss = masked_bce(bag.float(), y_t[rows], m_t[rows])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        bag_valid, _, _ = predict(model, signals, valid_rows, device)
        score, _ = macro_auroc(y[valid_rows], mask[valid_rows], bag_valid, MIN_HELDOUT_POSITIVES)
        best.history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "heldout_macro_auroc": score,
                "seconds": time.time() - start,
            }
        )
        print(
            f"epoch {epoch} loss {np.mean(losses):.4f} heldout AUROC {score:.4f} {time.time() - start:.0f}s",
            flush=True,
        )
        if score > best.best_score:
            best.state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best.best_epoch, best.best_score = epoch, score
        elif epoch - best.best_epoch >= recipe.patience:
            break
    return best


def step_centres(steps: int) -> np.ndarray:
    """Sample position of the centre of each 32 ms instance."""
    return np.arange(steps) * STRIDE


def upsample(native: np.ndarray, samples: int = SAMPLES) -> np.ndarray:
    """
    Linear interpolation of instance-resolution maps onto the 500 Hz sample grid.

    Parameters
    ----------
    native : np.ndarray
        (..., steps) map.
    samples : int
        Output length.

    Returns
    -------
    np.ndarray
        (..., samples) float32 map.
    """
    centres = step_centres(native.shape[-1])
    grid = np.arange(samples)
    flat = native.reshape(-1, native.shape[-1])
    out = np.empty((flat.shape[0], samples), np.float32)
    for row in range(flat.shape[0]):
        out[row] = np.interp(grid, centres, flat[row])
    return out.reshape(*native.shape[:-1], samples)


def finding_evidence(z: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    """
    Per-finding instance evidence in [0, 1].

    Log-mean-exp models: sigmoid of the instance logit. Attention models: attention weight times sigmoid of
    the instance logit, divided by its maximum per record and head.

    Parameters
    ----------
    z : np.ndarray
        (n, heads, 12, steps) instance logits.
    weights : np.ndarray or None
        Attention weights of the same shape, or None.

    Returns
    -------
    np.ndarray
        (n, heads, 12, steps) float32 evidence.
    """
    evidence = 1.0 / (1.0 + np.exp(-z.astype(np.float64)))
    if weights is not None:
        evidence = evidence * weights
        evidence = evidence / np.maximum(evidence.max(axis=(2, 3), keepdims=True), 1e-30)
    return evidence.astype(np.float32)


def combined_map(bag: np.ndarray, evidence: np.ndarray, heads: np.ndarray, mode: str = "sum") -> np.ndarray:
    """
    Combine per-finding evidence weighted by each finding's record probability.

    Parameters
    ----------
    bag : np.ndarray
        (n, heads) bag logits.
    evidence : np.ndarray
        (n, heads, 12, steps) evidence.
    heads : np.ndarray
        Indices of the heads that enter the map (every head except NORM).
    mode : str
        ``"sum"``: sum over heads of probability times evidence; ``"max"``: maximum over heads.

    Returns
    -------
    np.ndarray
        (n, 12, steps) float32 map.
    """
    prob = 1.0 / (1.0 + np.exp(-bag[:, heads].astype(np.float64)))
    weighted = prob[:, :, None, None] * evidence[:, heads]
    if mode == "sum":
        return weighted.sum(axis=1).astype(np.float32)
    if mode == "max":
        return weighted.max(axis=1).astype(np.float32)
    raise ValueError(f"Unknown mode {mode}")


def amplitude_map(raw: np.ndarray) -> np.ndarray:
    """
    Absolute signal after subtracting a 0.6 s running median per lead (061's amplitude baseline rule).

    Parameters
    ----------
    raw : np.ndarray
        (n, 12, 5000) raw signals in mV.

    Returns
    -------
    np.ndarray
        Map of the same shape.
    """
    raw = np.asarray(raw, np.float64)
    columns = raw.reshape(-1, raw.shape[-1]).T
    window = int(0.6 * FS) + 1
    median = pd.DataFrame(columns).rolling(window, center=True, min_periods=1).median().to_numpy()
    return np.abs(columns - median).T.reshape(raw.shape).astype(np.float32)


def native_pool(full: np.ndarray, steps: int) -> np.ndarray:
    """Average a 500 Hz map over each instance's 32 ms cell, centred like the instances."""
    padded = np.pad(full, [(0, 0)] * (full.ndim - 1) + [(STRIDE // 2, STRIDE * steps)], mode="edge")
    cells = padded[..., : STRIDE * steps].reshape(*full.shape[:-1], steps, STRIDE)
    return cells.mean(axis=-1)


def record_spearman(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Spearman correlation between two maps of each record over all leads and steps.

    Parameters
    ----------
    a, b : np.ndarray
        (n, ...) maps of the same shape.

    Returns
    -------
    np.ndarray
        (n,) correlations.
    """
    return np.array([spearmanr(x.ravel(), w.ravel()).statistic for x, w in zip(a, b, strict=True)])


def lead_auroc(native: np.ndarray, expected: np.ndarray) -> float:
    """
    AUROC of a record's lead scores (95th percentile over steps) for its expected leads.

    Parameters
    ----------
    native : np.ndarray
        (12, steps) map.
    expected : np.ndarray
        (12,) boolean expected-lead mask with both classes present.

    Returns
    -------
    float
        Mann-Whitney AUROC, ties as half.
    """
    scores = np.percentile(native, 95, axis=-1)
    return float(roc_auc_score(expected.astype(int), scores))
