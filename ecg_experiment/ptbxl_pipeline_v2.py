"""Experiment 049: the PTB-XL pipeline of Experiments 047 and 048 with every planned improvement.

Changes against Experiment 047, each one a setting:

- the encoder reads a higher sampling rate with shorter tokens, has register tokens and drop path, and can
  give each lead its own tokens;
- SimDINOv2 pretraining uses more local views, a slower teacher and more epochs, can use bfloat16 on CUDA,
  refuses to run twice on the same checkpoint and checks that a resumed run has the same settings;
- every tensor keeps the same shape from step to step (a fixed number of hidden tokens, padded with zero
  weight, and a lead dropout without variable-size indexing): the Apple GPU keeps a compiled graph per shape,
  and the variable shapes of Experiment 047 made its memory grow until the machine swapped;
- interval features measured on the signal (heart rate, RR variability, prematurity, PR, P wave, QRS width)
  are added to the embeddings;
- the whole encoder is fine-tuned with the PTB-XL labels instead of only querying a bank.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from .echonext_xecg import effective_rank, make_teacher, update_teacher
from .ptbxl_simdino_memory import (
    GLOBAL_VIEWS, LEADS, class_aurocs, coding_rate, cosine_schedule, crop, detect_beats, span_mask,
)

INTERVAL_NAMES = ("frecuencia_lpm", "rr_desviacion_s", "rmssd_s", "rr_variacion", "prematuridad", "pausa",
                  "pr_s", "p_amplitud_mv", "qrs_ancho_s", "latidos")


@dataclass(frozen=True)
class EncoderSettings:
    """
    Shape of the encoder.

    ``token_layout`` is ``"time"`` (one token per window of all twelve leads) or ``"lead"`` (one token per
    lead and window, twelve times more tokens, each knowing its lead).
    """

    sampling_rate: int = 250
    patch_seconds: float = 0.1
    token_layout: str = "time"
    width: int = 192
    depth: int = 6
    heads: int = 6
    registers: int = 4
    drop_path: float = 0.1

    @property
    def samples(self) -> int:
        """Samples in a 10 s record."""
        return 10 * self.sampling_rate

    @property
    def patch(self) -> int:
        """Samples per token."""
        return round(self.patch_seconds * self.sampling_rate)

    @property
    def steps(self) -> int:
        """Time steps in a 10 s record."""
        return self.samples // self.patch


@dataclass(frozen=True)
class PretrainSettings:
    """
    SimDINOv2 pretraining; the loss and schedules follow the official configuration.
    """

    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 5e-4
    warmup_epochs: int = 10
    global_seconds: float = 8.0
    local_seconds: float = 2.0
    local_views: int = 10
    mask_share: float = 0.5
    mask_ratio: tuple[float, float] = (0.1, 0.5)
    mask_span_seconds: tuple[float, float] = (0.2, 0.8)
    head_hidden: int = 2048
    head_out: int = 256
    epsilon: float = 0.05
    momentum_start: float = 0.996
    weight_decay: tuple[float, float] = (0.04, 0.4)
    gradient_norm: float = 3.0
    mixed_precision: bool = False
    seed: int = 49


@dataclass(frozen=True)
class FinetuneSettings:
    """
    Supervised fine-tuning of the pretrained encoder with the PTB-XL labels.
    """

    epochs: int = 30
    batch_size: int = 128
    backbone_lr: float = 1e-4
    head_lr: float = 1e-3
    weight_decay: float = 0.05
    patience: int = 6
    labeled_size: int | None = None
    interval_features: bool = True
    augment: bool = True
    seed: int = 49


def check_settings(encoder: EncoderSettings, pretrain: PretrainSettings) -> None:
    """
    Stop early when the settings cannot work together.

    Parameters
    ----------
    encoder : EncoderSettings
        Encoder shape.
    pretrain : PretrainSettings
        Pretraining settings.

    Raises
    ------
    ValueError
        If a token does not fit a whole number of samples, the record or a view is not a whole number of
        tokens, the width is not divisible by the heads, or the token layout is unknown.
    """
    problems = []
    if not math.isclose(encoder.patch, encoder.patch_seconds * encoder.sampling_rate):
        problems.append("patch_seconds × sampling_rate debe ser un número entero de muestras")
    for name, seconds in (("record", 10.0), ("global_seconds", pretrain.global_seconds),
                          ("local_seconds", pretrain.local_seconds)):
        if round(seconds * encoder.sampling_rate) % encoder.patch:
            problems.append(f"{name} debe ser un múltiplo de patch_seconds")
    if encoder.width % encoder.heads:
        problems.append("width debe ser divisible entre heads")
    if encoder.token_layout not in ("time", "lead"):
        problems.append("token_layout debe ser 'time' o 'lead'")
    if problems:
        raise ValueError("; ".join(problems))


class Block(nn.Module):
    """
    Pre-norm Transformer block with drop path (stochastic depth), as in DINOv2.

    Parameters
    ----------
    width : int
        Token size.
    heads : int
        Attention heads.
    drop_path : float
        Probability of skipping each residual branch for a whole ECG during training.
    """

    def __init__(self, width: int, heads: int, drop_path: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))
        self.drop_path = drop_path

    def skip(self, branch: torch.Tensor) -> torch.Tensor:
        """
        Drop a residual branch for random ECGs and rescale the rest.

        Parameters
        ----------
        branch : torch.Tensor
            Shape ``(batch, tokens, width)``.

        Returns
        -------
        torch.Tensor
            Same shape.
        """
        if not self.training or self.drop_path == 0:
            return branch
        keep = torch.rand(branch.shape[0], 1, 1, device=branch.device) >= self.drop_path
        return branch * keep / (1 - self.drop_path)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Attention then MLP, each added to its input.

        Parameters
        ----------
        hidden : torch.Tensor
            Shape ``(batch, tokens, width)``.

        Returns
        -------
        torch.Tensor
            Same shape.
        """
        normed = self.norm1(hidden)
        hidden = hidden + self.skip(self.attention(normed, normed, normed, need_weights=False)[0])
        return hidden + self.skip(self.mlp(self.norm2(hidden)))


class ECGEncoder(nn.Module):
    """
    Vision Transformer over ECG time windows, with a class token and register tokens.

    Parameters
    ----------
    settings : EncoderSettings
        Shape of the encoder.
    """

    def __init__(self, settings: EncoderSettings) -> None:
        super().__init__()
        self.settings = settings
        inputs = settings.patch * (LEADS if settings.token_layout == "time" else 1)
        self.patch = nn.Linear(inputs, settings.width)
        self.position = nn.Parameter(torch.zeros(1, settings.steps, settings.width))
        self.lead = nn.Parameter(torch.zeros(1, LEADS, 1, settings.width)) if settings.token_layout == "lead" else None
        self.cls = nn.Parameter(torch.zeros(1, 1, settings.width))
        self.register_tokens = nn.Parameter(torch.zeros(1, settings.registers, settings.width))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, settings.width))
        self.blocks = nn.ModuleList([Block(settings.width, settings.heads, settings.drop_path)
                                     for _ in range(settings.depth)])
        self.norm = nn.LayerNorm(settings.width)
        for parameter in (self.position, self.cls, self.register_tokens, self.mask_token, self.lead):
            if parameter is not None:
                nn.init.trunc_normal_(parameter, std=0.02)

    def token_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Expand a mask over time steps to the patch tokens.

        Parameters
        ----------
        mask : torch.Tensor
            Boolean ``(batch, steps)``.

        Returns
        -------
        torch.Tensor
            Boolean ``(batch, tokens)``: the same mask, repeated for each lead in the ``"lead"`` layout.
        """
        if self.settings.token_layout == "time":
            return mask
        return mask[:, None, :].expand(-1, LEADS, -1).flatten(1)

    def forward(self, signals: torch.Tensor, start: torch.Tensor, mask: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Describe a window of ECG.

        Parameters
        ----------
        signals : torch.Tensor
            Shape ``(batch, 12, samples)``, ``samples`` a multiple of the token length.
        start : torch.Tensor
            Time step where each window starts inside its 10 s record, ``(batch,)``.
        mask : torch.Tensor or None
            Boolean ``(batch, steps)``; hidden time steps are replaced by the mask token in every lead.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Class token ``(batch, width)`` and patch tokens ``(batch, tokens, width)``, where ``tokens`` is
            ``steps`` or ``12 * steps`` (lead-major).
        """
        batch, _, samples = signals.shape
        steps = samples // self.settings.patch
        windows = signals.unfold(2, self.settings.patch, self.settings.patch)
        positions = self.position[0, start[:, None] + torch.arange(steps, device=signals.device)]
        if self.settings.token_layout == "time":
            hidden = self.patch(windows.permute(0, 2, 1, 3).flatten(2))
            if mask is not None:
                hidden = torch.where(mask[..., None], self.mask_token.expand_as(hidden), hidden)
            hidden = hidden + positions
        else:
            hidden = self.patch(windows)
            if mask is not None:
                hidden = torch.where(mask[:, None, :, None], self.mask_token[:, None].expand_as(hidden), hidden)
            hidden = (hidden + positions[:, None] + self.lead).flatten(1, 2)
        extra = torch.cat([self.cls.expand(batch, -1, -1), self.register_tokens.expand(batch, -1, -1)], dim=1)
        hidden = torch.cat([extra.to(hidden.dtype), hidden], dim=1)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.norm(hidden)
        return hidden[:, 0], hidden[:, 1 + self.settings.registers:]


class ProjectionHead(nn.Module):
    """
    SimDINOv2 head without its last layer: three linear layers and unit length.

    Parameters
    ----------
    width : int
        Token size of the encoder.
    hidden : int
        Size of the two hidden layers.
    out : int
        Size of the output.
    """

    def __init__(self, width: int, hidden: int, out: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                    nn.Linear(hidden, out))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Project tokens onto the unit sphere.

        Parameters
        ----------
        tokens : torch.Tensor
            Shape ``(..., width)``.

        Returns
        -------
        torch.Tensor
            Shape ``(..., out)``, unit length, float32.
        """
        return F.normalize(self.layers(tokens).float(), dim=-1)


class PretrainNetwork(nn.Module):
    """
    Encoder plus projection head, used for both student and teacher.

    Parameters
    ----------
    encoder : EncoderSettings
        Shape of the encoder.
    pretrain : PretrainSettings
        Size of the head.
    """

    def __init__(self, encoder: EncoderSettings, pretrain: PretrainSettings) -> None:
        super().__init__()
        self.backbone = ECGEncoder(encoder)
        self.head = ProjectionHead(encoder.width, pretrain.head_hidden, pretrain.head_out)

    def forward(self, signals: torch.Tensor, start: torch.Tensor, mask: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Projected class token and raw patch tokens.

        Parameters
        ----------
        signals, start, mask
            As in ``ECGEncoder.forward``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Unit-length class feature and encoder patch tokens.
        """
        cls, patches = self.backbone(signals, start, mask)
        return self.head(cls), patches


def augment_static(crops: torch.Tensor, rate: int) -> torch.Tensor:
    """
    The augmentations of Experiment 047 with tensors of fixed shape.

    Every window is scaled by 0.8 to 1.2, gets white noise of up to 0.05 mV and a slow baseline wave of up to
    0.1 mV per lead (0.05 to 0.5 Hz), and with probability 0.3 loses one random lead.

    Parameters
    ----------
    crops : torch.Tensor
        Shape ``(batch, 12, samples)``.
    rate : int
        Sampling rate in Hz.

    Returns
    -------
    torch.Tensor
        Same shape.
    """
    batch, leads, samples = crops.shape
    device = crops.device
    scale = 0.8 + 0.4 * torch.rand(batch, 1, 1, device=device)
    noise = 0.05 * torch.rand(batch, 1, 1, device=device) * torch.randn_like(crops)
    seconds = torch.arange(samples, device=device) / rate
    frequency = 0.05 + 0.45 * torch.rand(batch, leads, 1, device=device)
    phase = 2 * math.pi * torch.rand(batch, leads, 1, device=device)
    wander = 0.1 * torch.rand(batch, leads, 1, device=device) * torch.sin(2 * math.pi * frequency * seconds + phase)
    chosen = torch.randint(0, leads, (batch, 1, 1), device=device)
    dropped = (torch.rand(batch, 1, 1, device=device) < 0.3) & (torch.arange(leads, device=device)[None, :, None] == chosen)
    return (crops * scale + noise + wander) * ~dropped


def masked_positions(mask: torch.Tensor, capacity: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Flat positions of the hidden tokens, padded to a fixed number, with their weights in the patch loss.

    Each masked view weighs the same; padding positions point at token 0 with weight 0. If more than
    ``capacity`` tokens are hidden, a random subset is kept.

    Parameters
    ----------
    mask : torch.Tensor
        Boolean CPU tensor ``(views, tokens)``.
    capacity : int
        Number of positions returned.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Positions into ``(views * tokens)`` and weights summing to 1, both of length ``capacity``.
    """
    hidden = mask.flatten().nonzero().flatten()
    if len(hidden) > capacity:
        hidden = hidden[torch.randperm(len(hidden))[:capacity]].sort().values
    views = hidden // mask.shape[1]
    counts = torch.bincount(views, minlength=mask.shape[0])
    index, weight = torch.zeros(capacity, dtype=torch.long), torch.zeros(capacity)
    index[:len(hidden)] = hidden
    weight[:len(hidden)] = 1 / counts[views] / (counts > 0).sum().clamp(min=1)
    return index, weight


def pretrain_loss(student_views: torch.Tensor, teacher_views: torch.Tensor, student_patches: torch.Tensor,
                  teacher_patches: torch.Tensor, weight: torch.Tensor, epsilon: float) -> dict[str, torch.Tensor]:
    """
    The SimDINOv2 loss of Experiment 047 with the patch term weighted by ``masked_positions``.

    Parameters
    ----------
    student_views : torch.Tensor
        Unit-length student class features, ``(2 + local views, batch, d)``, global views first.
    teacher_views : torch.Tensor
        Unit-length teacher class features of the two global views, ``(2, batch, d)``.
    student_patches, teacher_patches : torch.Tensor
        Unit-length features at the positions of ``masked_positions``, ``(capacity, d)``.
    weight : torch.Tensor
        Weights of those positions, ``(capacity,)``.
    epsilon : float
        Precision of the coding rate.

    Returns
    -------
    dict[str, torch.Tensor]
        ``view``, ``patch``, ``coding_rate`` and ``loss = view + patch - gamma * coding_rate``.
    """
    views, batch, size = student_views.shape
    similarity = torch.einsum("tbd,sbd->ts", teacher_views, student_views) / batch
    keep = torch.ones_like(similarity)
    keep[:, :GLOBAL_VIEWS] -= torch.eye(GLOBAL_VIEWS, device=keep.device)
    view = 1 - (similarity * keep).sum() / keep.sum()
    patch = 1 - (weight * (student_patches * teacher_patches).sum(dim=-1)).sum()
    expansion = (student_views[:GLOBAL_VIEWS] + teacher_views) / 2
    rate = torch.stack([coding_rate(features, epsilon) for features in expansion]).mean()
    gamma = (size + batch) / (size * batch)
    loss = view + patch - gamma * rate.float().to(view.device)
    return {"loss": loss, "view": view, "patch": patch, "coding_rate": rate}


def pretrain_step(student: PretrainNetwork, teacher: PretrainNetwork, signals: torch.Tensor,
                  encoder: EncoderSettings, pretrain: PretrainSettings) -> dict[str, torch.Tensor]:
    """
    Build two global and ``local_views`` local views of one batch and compute the SimDINOv2 loss.

    Parameters
    ----------
    student, teacher : PretrainNetwork
        The trained network and its moving average.
    signals : torch.Tensor
        Shape ``(batch, 12, samples)`` on the training device.
    encoder, pretrain
        Settings.

    Returns
    -------
    dict[str, torch.Tensor]
        Output of ``pretrain_loss``.
    """
    batch, patch, rate = signals.shape[0], encoder.patch, encoder.sampling_rate
    global_samples, local_samples = round(pretrain.global_seconds * rate), round(pretrain.local_seconds * rate)
    global_parts = [crop(signals, global_samples, patch) for _ in range(GLOBAL_VIEWS)]
    local_parts = [crop(signals, local_samples, patch) for _ in range(pretrain.local_views)]
    global_views = augment_static(torch.cat([part[0] for part in global_parts]), rate)
    global_start = torch.cat([part[1] for part in global_parts])
    local_views = augment_static(torch.cat([part[0] for part in local_parts]), rate)
    local_start = torch.cat([part[1] for part in local_parts])
    span = tuple(max(1, round(seconds / encoder.patch_seconds)) for seconds in pretrain.mask_span_seconds)
    mask = span_mask(len(global_views), global_samples // patch, pretrain.mask_share, pretrain.mask_ratio, span)
    tokens = student.backbone.token_mask(mask)
    index, weight = masked_positions(tokens, math.ceil(pretrain.mask_share * pretrain.mask_ratio[1] * tokens.numel()))
    mask, index, weight = mask.to(signals.device), index.to(signals.device), weight.to(signals.device)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16,
                              enabled=pretrain.mixed_precision and signals.device.type == "cuda")
    with autocast:
        with torch.no_grad():
            teacher_cls, teacher_tokens = teacher(global_views, global_start)
            teacher_patches = teacher.head(teacher_tokens.flatten(0, 1)[index])
        student_cls, student_tokens = student(global_views, global_start, mask)
        student_local, _ = student(local_views, local_start)
        student_patches = student.head(student_tokens.flatten(0, 1)[index])
    views = torch.cat([student_cls, student_local]).view(GLOBAL_VIEWS + pretrain.local_views, batch, -1)
    return pretrain_loss(views, teacher_cls.view(GLOBAL_VIEWS, batch, -1), student_patches, teacher_patches, weight,
                         pretrain.epsilon)


def claim_lock(lock: Path) -> None:
    """
    Make sure no other process is training on the same checkpoint.

    Parameters
    ----------
    lock : Path
        Lock file holding the process ID of the trainer.

    Raises
    ------
    RuntimeError
        If the lock belongs to another process that is still running.
    """
    if lock.exists():
        owner = int(lock.read_text().strip() or 0)
        alive = owner > 0 and owner != os.getpid()
        if alive:
            try:
                os.kill(owner, 0)
            except OSError:
                alive = False
        if alive:
            raise RuntimeError(f"El proceso {owner} ya está entrenando con {lock.with_suffix('.pt')}. Espera a que "
                               "termine; si ya no existe, borra el archivo .lock.")
    lock.write_text(str(os.getpid()))


def settings_record(*settings: object) -> dict[str, dict]:
    """
    Settings as plain dictionaries, to store in checkpoints and JSON files.

    Parameters
    ----------
    *settings : object
        Dataclass instances.

    Returns
    -------
    dict[str, dict]
        One dictionary per settings class, keyed by its name.
    """
    return {type(item).__name__: json.loads(json.dumps(asdict(item))) for item in settings}


@torch.no_grad()
def encode(backbone: ECGEncoder, signals: np.ndarray, *, device: torch.device, batch_size: int = 128,
           patches: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Describe whole 10 s ECGs with a frozen encoder.

    Parameters
    ----------
    backbone : ECGEncoder
        Encoder; left in evaluation mode.
    signals : np.ndarray
        Shape ``(records, 12, samples)``.
    device : torch.device
        Where the encoder is.
    batch_size : int
        ECGs per forward pass.
    patches : bool
        Also return the mean patch token of each ECG.

    Returns
    -------
    np.ndarray or tuple[np.ndarray, np.ndarray]
        Class tokens ``(records, width)`` and, if requested, mean patch tokens ``(records, width)``.
    """
    backbone.eval()
    classes, means = [], []
    for first in range(0, len(signals), batch_size):
        batch = torch.as_tensor(np.asarray(signals[first:first + batch_size], dtype=np.float32), device=device)
        cls, tokens = backbone(batch, torch.zeros(len(batch), dtype=torch.long, device=device))
        classes.append(cls.float().cpu().numpy())
        means.append(tokens.float().mean(dim=1).cpu().numpy())
    if patches:
        return np.concatenate(classes), np.concatenate(means)
    return np.concatenate(classes)


def pretrain_simdino(student: PretrainNetwork, teacher: PretrainNetwork, train_signals: np.ndarray,
                     monitor_signals: np.ndarray, encoder: EncoderSettings, pretrain: PretrainSettings, *,
                     device: torch.device, checkpoint_path: Path, resume: bool = True) -> pd.DataFrame:
    """
    Pretrain with SimDINOv2, one checkpoint per epoch, refusing a second trainer on the same file.

    The learning rate rises for ``warmup_epochs`` and follows a cosine to 1e-6; weight decay and teacher
    momentum follow cosines to their final values. A resumed run must have the same settings, and a finished
    run is only reloaded.

    Parameters
    ----------
    student, teacher : PretrainNetwork
        Networks on ``device``; the teacher starts as a copy of the student.
    train_signals : np.ndarray
        ECGs used for pretraining; labels are never read. They are kept on the device in float16, the format of
        the caches, and each batch is cast to float32.
    monitor_signals : np.ndarray
        ECGs whose teacher class tokens give the effective rank after every epoch.
    encoder, pretrain
        Settings.
    device : torch.device
        Training device.
    checkpoint_path : Path
        Checkpoint, replaced atomically after every epoch; ``.lock`` next to it marks the trainer.
    resume : bool
        Continue from ``checkpoint_path`` when it exists.

    Returns
    -------
    pd.DataFrame
        One row per epoch: ``loss``, ``view``, ``patch``, ``coding_rate``, ``rank``, ``learning_rate``,
        ``momentum`` and ``seconds``.

    Raises
    ------
    ValueError
        If the checkpoint was made with other settings.
    RuntimeError
        If another process is training on the same checkpoint.
    FloatingPointError
        If the loss stops being finite.
    """
    check_settings(encoder, pretrain)
    record = settings_record(encoder, pretrain)
    named = list(student.named_parameters())
    exempt = ("position", "cls", "register_tokens", "mask_token", "lead")
    decay = [p for n, p in named if p.ndim > 1 and n.split(".")[-1] not in exempt]
    no_decay = [p for n, p in named if not (p.ndim > 1 and n.split(".")[-1] not in exempt)]
    optimizer = torch.optim.AdamW([{"params": decay}, {"params": no_decay, "weight_decay": 0.0}],
                                  lr=pretrain.learning_rate)
    rows, first_epoch = [], 0
    if resume and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if state["settings"] != record:
            raise ValueError(f"{checkpoint_path} se creó con otra configuración; usa otro OUTPUT_DIR o bórralo.")
        rows, first_epoch = state["history"], state["epoch"]
        student.load_state_dict(state["student"])
        teacher.load_state_dict(state["teacher"])
        if first_epoch >= pretrain.epochs:
            print(f"Preentrenamiento terminado ({first_epoch} épocas); solo se carga.")
            return pd.DataFrame(rows)
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["cpu_rng"].cpu())
        np.random.set_state(state["numpy_rng"])
        print(f"Reanudado desde la época {first_epoch} de {pretrain.epochs}.")
    else:
        torch.manual_seed(pretrain.seed)
        np.random.seed(pretrain.seed)
    lock = checkpoint_path.with_suffix(".lock")
    claim_lock(lock)
    try:
        train = torch.as_tensor(np.asarray(train_signals, dtype=np.float16), device=device)
        steps = len(train) // pretrain.batch_size
        total, warmup = pretrain.epochs * steps, pretrain.warmup_epochs * steps
        for epoch in range(first_epoch, pretrain.epochs):
            student.train()
            order = torch.as_tensor(np.random.permutation(len(train)), device=device)
            totals, started = torch.zeros(4, dtype=torch.float64), time.perf_counter()
            for step in range(steps):
                done = epoch * steps + step
                rate = (pretrain.learning_rate * (done + 1) / warmup if done < warmup else
                        cosine_schedule(pretrain.learning_rate, 1e-6, (done - warmup) / max(1, total - warmup)))
                optimizer.param_groups[0]["lr"] = optimizer.param_groups[1]["lr"] = rate
                optimizer.param_groups[0]["weight_decay"] = cosine_schedule(*pretrain.weight_decay, done / total)
                batch = train[order[step * pretrain.batch_size:(step + 1) * pretrain.batch_size]].float()
                terms = pretrain_step(student, teacher, batch, encoder, pretrain)
                optimizer.zero_grad(set_to_none=True)
                terms["loss"].backward()
                nn.utils.clip_grad_norm_(student.parameters(), pretrain.gradient_norm)
                optimizer.step()
                momentum = cosine_schedule(pretrain.momentum_start, 1.0, (done + 1) / total)
                update_teacher(teacher, student, momentum)
                if device.type == "mps":
                    torch.mps.empty_cache()
                totals += torch.stack([terms[name].detach().cpu().double()
                                       for name in ("loss", "view", "patch", "coding_rate")])
                if not math.isfinite(totals[0].item()):
                    raise FloatingPointError(f"Loss is not finite at epoch {epoch + 1}, step {step + 1}")
            means = (totals / steps).tolist()
            rank = effective_rank(encode(teacher.backbone, monitor_signals, device=device))
            rows.append({"epoch": epoch + 1, "loss": means[0], "view": means[1], "patch": means[2],
                         "coding_rate": means[3], "rank": rank, "learning_rate": rate, "momentum": momentum,
                         "seconds": time.perf_counter() - started})
            partial = checkpoint_path.with_suffix(".partial")
            torch.save({"student": student.state_dict(), "teacher": teacher.state_dict(),
                        "optimizer": optimizer.state_dict(), "history": rows, "epoch": epoch + 1,
                        "settings": record, "cpu_rng": torch.get_rng_state(),
                        "numpy_rng": np.random.get_state()}, partial)
            os.replace(partial, checkpoint_path)
            print(f"Época {epoch + 1}/{pretrain.epochs} pérdida={means[0]:.4f} vistas={means[1]:.4f} "
                  f"parches={means[2]:.4f} tasa={means[3]:.1f} rango={rank:.1f} {rows[-1]['seconds']:.0f} s",
                  flush=True)
    finally:
        lock.unlink(missing_ok=True)
    return pd.DataFrame(rows)


def interval_features(signal: np.ndarray, rate: int) -> np.ndarray:
    """
    Measure rhythm and interval features of one ECG, the timing a cardiologist reads directly.

    Beats come from ``detect_beats``. RR features use every interval. The P wave, PR and QRS width are read on
    the median beat (median over beats of the window 0.4 s before to 0.5 s after each QRS): the P peak is the
    highest point of lead II between 0.30 and 0.08 s before the QRS, measured from the level 0.40 to 0.32 s
    before it; the QRS width is the time the summed absolute slope of the twelve leads stays above 30% of its
    maximum around the QRS.

    Parameters
    ----------
    signal : np.ndarray
        One ECG, ``(12, samples)`` in mV.
    rate : int
        Sampling rate in Hz.

    Returns
    -------
    np.ndarray
        One value per name in ``INTERVAL_NAMES``; NaN when fewer than three beats are found.
    """
    signal = np.asarray(signal, dtype=np.float64)
    beats = detect_beats(signal, rate)
    values = np.full(len(INTERVAL_NAMES), np.nan)
    values[-1] = len(beats)
    if len(beats) < 3:
        return values
    rr = np.diff(beats) / rate
    median_rr = np.median(rr)
    successive = np.diff(rr)
    values[:6] = (60 / median_rr, rr.std(), np.sqrt(np.mean(successive ** 2)), rr.std() / rr.mean(),
                  rr.min() / median_rr, rr.max() / median_rr)
    before, after = round(0.4 * rate), round(0.5 * rate)
    kept = beats[(beats >= before) & (beats + after <= signal.shape[1])]
    if len(kept) < 2:
        return values
    median_beat = np.median(np.stack([signal[:, beat - before:beat + after] for beat in kept]), axis=0)
    lead_ii = median_beat[1]
    baseline = np.median(lead_ii[:round(0.08 * rate)])
    search = slice(before - round(0.30 * rate), before - round(0.08 * rate))
    p_index = search.start + int(np.argmax(lead_ii[search]))
    slope = np.abs(np.diff(median_beat, axis=1)).sum(axis=0)
    around = slice(before - round(0.1 * rate), before + round(0.1 * rate))
    above = slope[around] >= 0.3 * slope[around].max()
    values[6:9] = ((before - p_index) / rate, lead_ii[p_index] - baseline, above.sum() / rate)
    return values


def all_interval_features(signals: np.ndarray, rate: int) -> np.ndarray:
    """
    ``interval_features`` for many ECGs.

    Parameters
    ----------
    signals : np.ndarray
        Shape ``(records, 12, samples)``.
    rate : int
        Sampling rate in Hz.

    Returns
    -------
    np.ndarray
        ``(records, len(INTERVAL_NAMES))``.
    """
    return np.stack([interval_features(signal, rate) for signal in signals])


class DiagnosisModel(nn.Module):
    """
    Pretrained encoder, the mean of its class token and patch tokens, optional interval features, and one
    output per diagnosis.

    Parameters
    ----------
    backbone : ECGEncoder
        Encoder initialized from the SimDINOv2 teacher.
    classes : int
        Diagnoses predicted.
    extra : int
        Interval features added; 0 for none.
    """

    def __init__(self, backbone: ECGEncoder, classes: int, extra: int) -> None:
        super().__init__()
        self.backbone = backbone
        width = backbone.settings.width
        self.norm = nn.LayerNorm(2 * width)
        self.output = nn.Sequential(nn.Dropout(0.2), nn.Linear(2 * width + extra, classes))

    def forward(self, signals: torch.Tensor, extra: torch.Tensor | None) -> torch.Tensor:
        """
        Logit of every diagnosis.

        Parameters
        ----------
        signals : torch.Tensor
            Whole 10 s ECGs, ``(batch, 12, samples)``.
        extra : torch.Tensor or None
            Standardized interval features, ``(batch, extra)``.

        Returns
        -------
        torch.Tensor
            ``(batch, classes)``.
        """
        cls, patches = self.backbone(signals, torch.zeros(len(signals), dtype=torch.long, device=signals.device))
        features = self.norm(torch.cat([cls, patches.mean(dim=1)], dim=-1))
        if extra is not None:
            features = torch.cat([features, extra], dim=-1)
        return self.output(features)


def standardize_features(train: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    """
    Standardize interval features with the training mean and deviation; missing values become 0 (the mean).

    Parameters
    ----------
    train : np.ndarray
        Features of the training ECGs.
    *others : np.ndarray
        Features of other ECGs.

    Returns
    -------
    list[np.ndarray]
        Float32 arrays in the order given, training first.
    """
    mean, deviation = np.nanmean(train, axis=0), np.nanstd(train, axis=0)
    deviation[deviation == 0] = 1
    return [np.nan_to_num((values - mean) / deviation).astype(np.float32) for values in (train, *others)]


@torch.no_grad()
def predict(model: DiagnosisModel, signals: np.ndarray, extra: np.ndarray | None, *, device: torch.device,
            batch_size: int = 256) -> np.ndarray:
    """
    Probability of every diagnosis.

    Parameters
    ----------
    model : DiagnosisModel
        Trained model; left in evaluation mode.
    signals : np.ndarray
        ``(records, 12, samples)``.
    extra : np.ndarray or None
        Standardized interval features of the same records.
    device : torch.device
        Where the model is.
    batch_size : int
        ECGs per forward pass.

    Returns
    -------
    np.ndarray
        ``(records, classes)`` between 0 and 1.
    """
    model.eval()
    outputs = []
    for first in range(0, len(signals), batch_size):
        batch = torch.as_tensor(np.asarray(signals[first:first + batch_size], dtype=np.float32), device=device)
        features = None if extra is None else torch.as_tensor(extra[first:first + batch_size], device=device)
        outputs.append(torch.sigmoid(model(batch, features)).float().cpu().numpy())
    return np.concatenate(outputs)


def finetune(model: DiagnosisModel, train_signals: np.ndarray, train_labels: np.ndarray,
             train_extra: np.ndarray | None, val_signals: np.ndarray | None, val_labels: np.ndarray | None,
             val_extra: np.ndarray | None, settings: FinetuneSettings, *, rate: int, device: torch.device,
             checkpoint_path: Path) -> pd.DataFrame:
    """
    Train the whole model with binary cross-entropy and keep the epoch with the best ``val`` macro AUROC.

    AdamW with a small learning rate for the pretrained encoder and a larger one for the new output; one
    warm-up epoch then a cosine. Training stops after ``patience`` epochs without improvement. Without
    validation ECGs (``val_signals=None``) every epoch runs and the last one is kept, so no labels outside the
    training set are used. A finished run is only reloaded from ``checkpoint_path``.

    Parameters
    ----------
    model : DiagnosisModel
        Model on ``device``.
    train_signals, train_labels, train_extra
        Training ECGs, their boolean labels and standardized interval features (or None).
    val_signals, val_labels, val_extra
        Validation ECGs, used to pick the epoch; all None to train a fixed number of epochs.
    settings : FinetuneSettings
        Fine-tuning settings.
    rate : int
        Sampling rate, for the augmentations.
    device : torch.device
        Training device.
    checkpoint_path : Path
        Weights of the best epoch and the history.

    Returns
    -------
    pd.DataFrame
        One row per epoch: ``train_loss``, ``val_auroc`` and ``seconds``; the model holds the best weights.
    """
    record = settings_record(settings, model.backbone.settings)
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if state["settings"] != record:
            raise ValueError(f"{checkpoint_path} se creó con otra configuración; usa otro OUTPUT_DIR o bórralo.")
        model.load_state_dict(state["model"])
        print(f"Afinado terminado; se carga la mejor época ({state['best_epoch']}).")
        return pd.DataFrame(state["history"])
    torch.manual_seed(settings.seed)
    rng = np.random.default_rng(settings.seed)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": settings.backbone_lr},
        {"params": [*model.norm.parameters(), *model.output.parameters()], "lr": settings.head_lr},
    ], weight_decay=settings.weight_decay)
    peaks = [settings.backbone_lr, settings.head_lr]
    labels = torch.as_tensor(train_labels.astype(np.float32))
    steps = len(train_signals) // settings.batch_size
    total = settings.epochs * steps
    rows, best, best_state, waited = [], -1.0, None, 0
    for epoch in range(settings.epochs):
        model.train()
        order, started, losses = rng.permutation(len(train_signals)), time.perf_counter(), []
        for step in range(steps):
            done = epoch * steps + step
            scale = ((done + 1) / steps if done < steps else
                     cosine_schedule(1.0, 0.0, (done - steps) / max(1, total - steps)))
            for group, peak in zip(optimizer.param_groups, peaks, strict=True):
                group["lr"] = peak * scale
            chosen = np.sort(order[step * settings.batch_size:(step + 1) * settings.batch_size])
            batch = torch.as_tensor(np.asarray(train_signals[chosen], dtype=np.float32), device=device)
            if settings.augment:
                batch = augment_static(batch, rate)
            extra = None if train_extra is None else torch.as_tensor(train_extra[chosen], device=device)
            loss = F.binary_cross_entropy_with_logits(model(batch, extra), labels[chosen].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if device.type == "mps":
                torch.mps.empty_cache()
            losses.append(loss.item())
        auroc = (float("nan") if val_signals is None else
                 class_aurocs(val_labels, predict(model, val_signals, val_extra, device=device)).mean())
        rows.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_auroc": auroc,
                     "seconds": time.perf_counter() - started})
        if val_signals is None:
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_epoch = epoch + 1
            continue
        print(f"Época {epoch + 1}/{settings.epochs} pérdida={rows[-1]['train_loss']:.4f} AUROC val={auroc:.4f} "
              f"{rows[-1]['seconds']:.0f} s", flush=True)
        if auroc > best:
            best, waited = auroc, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_epoch = epoch + 1
        else:
            waited += 1
            if waited >= settings.patience:
                break
    model.load_state_dict(best_state)
    torch.save({"model": best_state, "history": rows, "best_epoch": best_epoch, "settings": record}, checkpoint_path)
    return pd.DataFrame(rows)


def encoder_from_experiment047(state: dict[str, torch.Tensor]) -> ECGEncoder:
    """
    Rebuild the Experiment 047 backbone as an ``ECGEncoder`` with the same weights and outputs.

    The 047 encoder (100 Hz, tokens of 0.2 s of all leads, width 192, 6 pre-norm blocks of 6 heads, no
    registers, no drop path) computes exactly what an ``ECGEncoder`` with those settings computes; only the
    parameter names differ (``nn.TransformerEncoderLayer`` against ``Block``).

    Parameters
    ----------
    state : dict[str, torch.Tensor]
        ``state_dict`` of the 047 ``ECGViT``, for example ``outputs/experiment047_*/teacher_encoder.pt``.

    Returns
    -------
    ECGEncoder
        Encoder in evaluation mode on the CPU.
    """
    encoder = ECGEncoder(EncoderSettings(sampling_rate=100, patch_seconds=0.2, token_layout="time", width=192,
                                         depth=6, heads=6, registers=0, drop_path=0.0))
    renamed = {}
    for name, value in state.items():
        if name.startswith("blocks.layers."):
            _, _, index, rest = name.split(".", 3)
            rest = (rest.replace("self_attn.", "attention.").replace("linear1.", "mlp.0.")
                    .replace("linear2.", "mlp.2."))
            renamed[f"blocks.{index}.{rest}"] = value
        else:
            renamed[name] = value
    renamed["register_tokens"] = encoder.register_tokens.detach().clone()
    encoder.load_state_dict(renamed)
    return encoder.eval()
