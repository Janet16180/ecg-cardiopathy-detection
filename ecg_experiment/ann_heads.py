"""Neural heads for Experiment 043: an MLP, an additive attention head and the tutor's CNN + transformer."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from .lead_wave_maps import UnitMap

SEEDS = (43043, 43044, 43045)
VALIDATION_SHARE = 0.10
GRID_PATCHES = 50
GRID_PATCH_SECONDS = 0.2
Batch = Callable[[np.ndarray], torch.Tensor]


@dataclass
class Recipe:
    """Optimizer and stopping settings of one network."""

    learning_rate: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    patience: int
    clip: float | None = None


@dataclass
class Fit:
    """Best weights and the training history of one network."""

    state: dict[str, torch.Tensor]
    best_epoch: int
    best_auroc: float
    history: list[dict[str, float]] = field(default_factory=list)


def validation_mask(groups: np.ndarray, share: float, seed: int) -> np.ndarray:
    """
    Mark every row of a random ``share`` of the groups as validation.

    Parameters
    ----------
    groups : np.ndarray
        Group (patient or record) of each row.
    share : float
        Share of groups to hold out.
    seed : int
        Seed of ``numpy.random.default_rng``.

    Returns
    -------
    np.ndarray
        Boolean mask of the validation rows.
    """
    unique = np.unique(groups)
    chosen = np.random.default_rng(seed).choice(len(unique), round(share * len(unique)), replace=False)
    return np.isin(groups, unique[chosen])


def weighted_standardizer(x: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Weighted mean and standard deviation of each column (zero deviations become 1).

    Parameters
    ----------
    x : np.ndarray
        ``[rows, columns]`` features.
    weights : np.ndarray
        Row weights.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Mean and scale.
    """
    w = weights / weights.sum()
    mean = w @ x
    scale = np.sqrt(w @ (x - mean) ** 2)
    scale[scale == 0] = 1.0
    return mean, scale


class MLPHead(nn.Module):
    """One hidden ReLU layer with dropout, then a single logit."""

    def __init__(self, width: int, hidden: int = 256, dropout: float = 0.2) -> None:
        """
        Build the layers.

        Parameters
        ----------
        width : int
            Input features.
        hidden : int
            Hidden units.
        dropout : float
            Dropout after the hidden layer.
        """
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout),
                                    nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Return the logit of each row and no unit contributions."""
        return self.layers(x).squeeze(-1), None


class AttentionHead(nn.Module):
    """Additive multiple-instance head: the logit is the attention-weighted sum of per-token logits."""

    def __init__(self, width: int, hidden: int = 128, scorer: int = 64, dropout: float = 0.1) -> None:
        """
        Build the layers.

        Parameters
        ----------
        width : int
            Token width.
        hidden : int
            Projected token width.
        scorer : int
            Hidden width of the attention scorer.
        dropout : float
            Dropout after the projection.
        """
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden), nn.GELU(),
                                     nn.Dropout(dropout))
        self.score = nn.Sequential(nn.Linear(hidden, scorer), nn.Tanh(), nn.Linear(scorer, 1))
        self.token_logit = nn.Linear(hidden, 1)
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pool the tokens of each record.

        Parameters
        ----------
        tokens : torch.Tensor
            ``[records, tokens, width]``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Logit ``[records]`` and per-token contributions ``[records, tokens]`` that sum to the logit minus
            the bias.
        """
        hidden = self.project(tokens.float())
        weights = torch.softmax(self.score(hidden).squeeze(-1), dim=1)
        contributions = weights * self.token_logit(hidden).squeeze(-1)
        return contributions.sum(dim=1) + self.bias, contributions


class CNNTransformer(nn.Module):
    """The tutor's architecture: a CNN shared by the leads makes 0.2 s tokens, a transformer mixes them."""

    def __init__(self, leads: int = 12, patches: int = GRID_PATCHES, width: int = 128, layers: int = 4,
                 heads: int = 4, feedforward: int = 256, dropout: float = 0.1) -> None:
        """
        Build the CNN tokenizer, the embeddings, the transformer and the attention head.

        Parameters
        ----------
        leads : int
            Leads of the input.
        patches : int
            Tokens per lead produced by the CNN (50 for 2,500 samples).
        width : int
            Token width.
        layers : int
            Transformer layers.
        heads : int
            Attention heads per layer.
        feedforward : int
            Feed-forward width.
        dropout : float
            Dropout in the transformer.
        """
        super().__init__()
        self.leads, self.patches = leads, patches
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.GELU(),
            nn.Conv1d(32, 64, 5, stride=5), nn.GELU(),
            nn.Conv1d(64, width, 5, stride=5), nn.GELU(),
            nn.Conv1d(width, width, 3, stride=2, padding=1),
        )
        self.lead_embedding = nn.Parameter(torch.zeros(leads, 1, width))
        self.time_embedding = nn.Parameter(torch.zeros(1, patches, width))
        nn.init.normal_(self.lead_embedding, std=0.02)
        nn.init.normal_(self.time_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(width, heads, feedforward, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.head = AttentionHead(width)

    def forward(self, signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Classify a batch of ECGs.

        Parameters
        ----------
        signal : torch.Tensor
            ``[records, leads, samples]`` standardized signals.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Logit and per-token contributions, token ``patches * lead + patch``.
        """
        records = len(signal)
        tokens = self.cnn(signal.reshape(records * self.leads, 1, -1))
        if tokens.shape[-1] != self.patches:
            raise ValueError(f"Expected {self.patches} tokens per lead, got {tokens.shape[-1]}")
        tokens = tokens.transpose(1, 2).reshape(records, self.leads, self.patches, -1)
        tokens = tokens + self.lead_embedding + self.time_embedding
        tokens = tokens.reshape(records, self.leads * self.patches, -1)
        return self.head(self.norm(self.transformer(tokens)))


def augment(signal: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """
    Scale each record by U(0.9, 1.1), add noise with SD 0.01 and zero each lead with probability 0.1.

    Parameters
    ----------
    signal : torch.Tensor
        ``[records, leads, samples]`` standardized signals.
    generator : torch.Generator
        Generator on the signal's device.

    Returns
    -------
    torch.Tensor
        The augmented signals.
    """
    records, leads, _ = signal.shape
    options = {"generator": generator, "device": signal.device}
    scale = 0.9 + 0.2 * torch.rand(records, 1, 1, **options)
    keep = (torch.rand(records, leads, 1, **options) >= 0.1).float()
    noise = 0.01 * torch.randn(signal.shape, **options)
    return (signal * scale + noise) * keep


@torch.no_grad()
def predict(model: nn.Module, batch: Batch, rows: np.ndarray, batch_size: int,
            keep_contributions: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Logits (and optionally unit contributions) of the given rows, in order.

    Parameters
    ----------
    model : nn.Module
        Network returning ``(logit, contributions)``.
    batch : Batch
        Maps row positions to an input tensor on the model's device.
    rows : np.ndarray
        Row positions.
    batch_size : int
        Rows per forward pass.
    keep_contributions : bool
        Also return the contributions.

    Returns
    -------
    tuple[np.ndarray, np.ndarray | None]
        Float64 logits and float32 contributions (or ``None``).
    """
    model.eval()
    logits, contributions = [], []
    for start in range(0, len(rows), batch_size):
        logit, units = model(batch(rows[start:start + batch_size]))
        logits.append(logit.double().cpu().numpy())
        if keep_contributions:
            contributions.append(units.float().cpu().numpy())
    return np.concatenate(logits), np.concatenate(contributions) if keep_contributions else None


def train(model: nn.Module, batch: Batch, train_rows: np.ndarray, y: np.ndarray, weights: np.ndarray,
          validation_rows: np.ndarray, validation_y: np.ndarray, recipe: Recipe, seed: int,
          augmentation: Callable[[torch.Tensor, torch.Generator], torch.Tensor] | None = None,
          log: Callable[[str], None] = print) -> Fit:
    """
    Train with weighted binary cross-entropy and keep the epoch with the best validation AUROC.

    Parameters
    ----------
    model : nn.Module
        Network returning ``(logit, contributions)``, already on its device.
    batch : Batch
        Maps row positions to an input tensor on the model's device.
    train_rows : np.ndarray
        Training row positions.
    y : np.ndarray
        Binary target of each training row.
    weights : np.ndarray
        Sample weight of each training row.
    validation_rows : np.ndarray
        Validation row positions.
    validation_y : np.ndarray
        Their targets.
    recipe : Recipe
        Optimizer and stopping settings.
    seed : int
        Seed of the shuffling and augmentation.
    augmentation : Callable | None
        Applied to each training batch.
    log : Callable[[str], None]
        Receives one line per epoch.

    Returns
    -------
    Fit
        The best weights, epoch, validation AUROC and history.
    """
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe.learning_rate,
                                  weight_decay=recipe.weight_decay)
    order_rng = np.random.default_rng(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    targets = torch.as_tensor(y, dtype=torch.float32, device=device)
    sample_weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
    best = Fit({}, -1, -np.inf)
    for epoch in range(recipe.max_epochs):
        began = time.perf_counter()
        model.train()
        order = order_rng.permutation(len(train_rows))
        total = 0.0
        for start in range(0, len(order), recipe.batch_size):
            chosen = order[start:start + recipe.batch_size]
            inputs = batch(train_rows[chosen])
            if augmentation is not None:
                inputs = augmentation(inputs, generator)
            logit, _ = model(inputs)
            loss = (nn.functional.binary_cross_entropy_with_logits(logit, targets[chosen], reduction="none")
                    * sample_weights[chosen]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if recipe.clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), recipe.clip)
            optimizer.step()
            total += float(loss) * len(chosen)
        logits, _ = predict(model, batch, validation_rows, recipe.batch_size * 2)
        auroc = float(roc_auc_score(validation_y, logits))
        best.history.append({"epoch": epoch, "loss": total / len(order), "validation_auroc": auroc,
                             "seconds": time.perf_counter() - began})
        log(f"seed {seed} epoch {epoch} loss {total / len(order):.4f} validation AUROC {auroc:.4f}")
        if auroc > best.best_auroc:
            best.state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best.best_epoch, best.best_auroc = epoch, auroc
        elif epoch - best.best_epoch >= recipe.patience:
            break
    model.load_state_dict(best.state)
    return best


def grid_unit_map(scores: np.ndarray, leads: tuple[int, ...], patches: int = GRID_PATCHES,
                  seconds: float = GRID_PATCH_SECONDS) -> UnitMap:
    """
    Units of a lead-by-patch grid of scores, token ``patches * lead + patch``.

    Parameters
    ----------
    scores : np.ndarray
        ``[len(leads) * patches]`` scores.
    leads : tuple[int, ...]
        Canonical lead index of each grid row.
    patches : int
        Patches per lead.
    seconds : float
        Patch length.

    Returns
    -------
    UnitMap
        The units.
    """
    lead_index = np.repeat(np.array(leads), patches)
    starts = np.tile(np.arange(patches) * seconds, len(leads))
    return UnitMap(np.asarray(scores, dtype=np.float64), lead_index, starts, starts + seconds)


def ragged_unit_maps(arrays: Mapping[str, np.ndarray], name: str) -> list[UnitMap | None]:
    """
    Split ragged unit arrays saved with per-ECG offsets (Experiment 042 ``save_maps``) into maps.

    Parameters
    ----------
    arrays : Mapping[str, np.ndarray]
        Arrays with ``{name}_offsets``, ``{name}_scores``, ``{name}_leads``, ``{name}_starts`` and
        ``{name}_ends``.
    name : str
        Map name.

    Returns
    -------
    list[UnitMap | None]
        One map per ECG, ``None`` where the ECG has no unit.
    """
    offsets = arrays[f"{name}_offsets"]
    fields = [np.asarray(arrays[f"{name}_{part}"]) for part in ("scores", "leads", "starts", "ends")]
    maps: list[UnitMap | None] = []
    for low, high in zip(offsets[:-1], offsets[1:], strict=True):
        maps.append(None if high == low else UnitMap(*(values[low:high] for values in fields)))
    return maps


def gate_beat_units(beat_scores: np.ndarray, shares: np.ndarray) -> np.ndarray:
    """
    Keep each beat unit's score where its lead-and-wave readout share is above 0, and set it to 0 elsewhere.

    Parameters
    ----------
    beat_scores : np.ndarray
        Flat unit scores of one ECG, unit ``(beat * leads + lead) * waves + wave``.
    shares : np.ndarray
        Flat lead-and-wave shares of the same ECG, unit ``lead * waves + wave``.

    Returns
    -------
    np.ndarray
        Gated flat unit scores.
    """
    blocks = np.asarray(beat_scores, dtype=np.float64).reshape(-1, shares.size)
    return np.where(np.asarray(shares)[None, :] > 0, blocks, 0.0).ravel()


def detection_reading(sph_ci_low: float, full_difference: float) -> str:
    """
    Read a detection arm against the comparator by the Experiment 043 rule.

    Parameters
    ----------
    sph_ci_low : float
        Lower bound of the SPH AUROC difference.
    full_difference : float
        AUROC difference on full PTB-XL development.

    Returns
    -------
    str
        ``beats``, ``matches`` or ``below``.
    """
    if sph_ci_low > 0 and full_difference >= -0.005:
        return "beats"
    if sph_ci_low > -0.01 and full_difference >= -0.01:
        return "matches"
    return "below"


def map_reading(localization_ci_low: float, lead_ci_low: float, benign_ci_high: float,
                auroc_ci_low: float, margin: float = -0.10) -> dict[str, Any]:
    """
    Read a map against Experiment 042's ``U_B`` by 042's rule.

    Parameters
    ----------
    localization_ci_low : float
        Lower bound of the map's hit - chance minus ``U_B``'s.
    lead_ci_low : float
        Lower bound of the map's anterior lead contrast.
    benign_ci_high : float
        Upper bound of the map's benign any-red share minus ``U_B``'s.
    auroc_ci_low : float
        Lower bound of the map's worst-unit AUROC minus ``U_B``'s.
    margin : float
        Localization margin.

    Returns
    -------
    dict[str, Any]
        ``keeps_premature_localization``, ``gains`` and ``improves_on_U_B``.
    """
    keeps = localization_ci_low > margin
    gains = {"lead": lead_ci_low > 0, "benign": benign_ci_high < 0, "detection": auroc_ci_low > 0}
    return {"keeps_premature_localization": bool(keeps),
            "gains": {key: bool(value) for key, value in gains.items()},
            "improves_on_U_B": bool(keeps and any(gains.values()))}


@dataclass
class RowFile:
    """An open ``.npy`` array read and written by whole rows with positional I/O, without a memory map."""

    path: Path
    descriptor: int
    offset: int
    shape: tuple[int, ...]
    dtype: np.dtype

    @property
    def row_bytes(self) -> int:
        """Bytes of one row."""
        return int(np.prod(self.shape[1:])) * self.dtype.itemsize


def open_row_file(path: Path, writable: bool = False) -> RowFile:
    """
    Open an existing ``.npy`` file for row reads (and writes).

    Parameters
    ----------
    path : Path
        C-ordered ``.npy`` file.
    writable : bool
        Also allow ``write_rows``.

    Returns
    -------
    RowFile
        The open file.
    """
    mapped = np.load(path, mmap_mode="r")
    if not mapped.flags.c_contiguous:
        raise ValueError(f"{path} is not C-ordered")
    found = RowFile(Path(path), os.open(path, os.O_RDWR if writable else os.O_RDONLY), int(mapped.offset),
                    tuple(mapped.shape), mapped.dtype)
    del mapped
    return found


def create_row_file(path: Path, shape: tuple[int, ...], dtype: np.dtype | str) -> RowFile:
    """
    Create a sparse ``.npy`` file of the given shape and open it for row writes.

    Parameters
    ----------
    path : Path
        New file.
    shape : tuple[int, ...]
        Array shape, rows first.
    dtype : np.dtype | str
        Element type.

    Returns
    -------
    RowFile
        The open, writable file.
    """
    created = np.lib.format.open_memmap(path, mode="w+", dtype=np.dtype(dtype), shape=shape)
    del created
    return open_row_file(path, writable=True)


def write_rows(store: RowFile, start: int, values: np.ndarray) -> None:
    """
    Write consecutive rows starting at ``start``.

    Parameters
    ----------
    store : RowFile
        Writable file.
    start : int
        First row.
    values : np.ndarray
        ``[rows, *shape[1:]]`` values, cast to the file's dtype.
    """
    data = np.ascontiguousarray(values, dtype=store.dtype)
    if data.shape[1:] != store.shape[1:] or start + len(data) > store.shape[0]:
        raise ValueError(f"Rows {start}..{start + len(data)} of shape {data.shape} do not fit {store.shape}")
    written = os.pwrite(store.descriptor, data.tobytes(), store.offset + start * store.row_bytes)
    if written != data.nbytes:
        raise OSError(f"Short write to {store.path}")


def read_rows(store: RowFile, rows: np.ndarray) -> np.ndarray:
    """
    Read rows in the given order, reading in sorted order and merging consecutive rows into one read.

    Parameters
    ----------
    store : RowFile
        Open file.
    rows : np.ndarray
        Row positions.

    Returns
    -------
    np.ndarray
        ``[len(rows), *shape[1:]]`` values.
    """
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows, kind="stable")
    ordered = rows[order]
    if len(rows) and (ordered[0] < 0 or ordered[-1] >= store.shape[0]):
        raise IndexError(f"Rows outside {store.shape[0]}")
    values = np.empty((len(rows), *store.shape[1:]), dtype=store.dtype)
    flat = values.reshape(len(rows), -1)
    breaks = np.flatnonzero(np.diff(ordered) != 1) + 1
    for low, high in zip(np.r_[0, breaks], np.r_[breaks, len(ordered)], strict=True):
        if high == low:
            continue
        size = (high - low) * store.row_bytes
        data = os.pread(store.descriptor, size, store.offset + int(ordered[low]) * store.row_bytes)
        if len(data) != size:
            raise OSError(f"Short read from {store.path}")
        flat[low:high] = np.frombuffer(data, dtype=store.dtype).reshape(high - low, -1)
    restored = np.empty_like(values)
    restored[order] = values
    return restored
