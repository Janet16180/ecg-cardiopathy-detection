"""Experiment 047: a SimDINOv2 ECG encoder on PTB-XL and a memory bank of named diagnoses.

"No time to train!" (arXiv:2507.02798) labels objects in images without training: a frozen DINOv2 describes
every patch, SAM proposes where objects are, each proposal gets the mean description of its patches, and the
nearest class prototype in a bank built from a few labelled references names it. This module carries the
idea to 12-lead ECGs:

- the frozen encoder is a small Vision Transformer over 0.2 s time patches, pretrained here without labels
  with SimDINOv2 (arXiv:2502.10385), because no DINOv2 for ECGs exists;
- whole ECGs are named by comparing their class token with the labelled references of PTB-XL folds 1 to 8;
- beats play the role of SAM masks: a classical R-peak detector finds them and each beat gets the mean of
  the patch tokens it covers, compared with a bank of normal beats.

The loss follows the official SimDINOv2 code (``MCRLoss`` and ``CosinePatchLoss``, coding rate on the mean
of student and teacher global features, ``eps = 0.05``). The protocol lists every difference from it.
"""

from __future__ import annotations

import ast
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.signal import butter, find_peaks, resample_poly, sosfiltfilt
from sklearn.metrics import roc_auc_score
from torch import nn

from .echonext_xecg import effective_rank, make_teacher, update_teacher
from .intervals import patient_groups, patient_resample

SOURCE_RATE = 500
RATE = 100
LEADS = 12
SAMPLES = 1000
LEAD_NAMES = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
PATCH = 20
TOKENS = SAMPLES // PATCH
GLOBAL_SAMPLES = 800
LOCAL_SAMPLES = 200
GLOBAL_VIEWS = 2
LOCAL_VIEWS = 6
WIDTH = 192
DEPTH = 6
HEADS = 6
HEAD_HIDDEN = 2048
HEAD_OUT = 256
MASK_SHARE = 0.5
MASK_RATIO = (0.1, 0.5)
MASK_SPAN = (1, 4)
EPSILON = 0.05
MOMENTUM_START = 0.992
WEIGHT_DECAY = (0.04, 0.4)
GRADIENT_NORM = 3.0
TEMPERATURE = 0.07
BEAT_WINDOW = (-0.25, 0.45)
DIAGNOSES = {
    "NORM": ("ECG normal", "referencia"),
    "AFIB": ("Fibrilación auricular", "ritmo"),
    "STACH": ("Taquicardia sinusal", "ritmo"),
    "SBRAD": ("Bradicardia sinusal", "ritmo"),
    "SARRH": ("Arritmia sinusal", "ritmo"),
    "PVC": ("Extrasístole ventricular", "latido aislado"),
    "PAC": ("Extrasístole auricular", "latido aislado"),
    "CLBBB": ("Bloqueo completo de rama izquierda", "conducción"),
    "CRBBB": ("Bloqueo completo de rama derecha", "conducción"),
    "LAFB": ("Hemibloqueo anterior izquierdo", "conducción"),
    "1AVB": ("Bloqueo AV de primer grado", "conducción"),
    "LVH": ("Hipertrofia ventricular izquierda", "estructura"),
    "IMI": ("Infarto inferior", "infarto"),
    "ASMI": ("Infarto anteroseptal", "infarto"),
}
CODES = tuple(DIAGNOSES)


def read_record(header: Path) -> np.ndarray:
    """
    Read one PTB-XL WFDB record at 500 Hz.

    Parameters
    ----------
    header : Path
        The ``.hea`` file; its ``.dat`` file must sit next to it.

    Returns
    -------
    np.ndarray
        Float32 millivolts, shape ``(12, 5000)``.

    Raises
    ------
    ValueError
        If the header is not 12 leads of format 16 at 1000 units per mV with baseline 0 and 5,000 samples.
    """
    lines = header.read_text().split("\n")
    name, leads, rate, samples = lines[0].split()[:4]
    signals = [line.split() for line in lines[1:1 + LEADS]]
    if (int(leads), int(rate), int(samples)) != (LEADS, SOURCE_RATE, SOURCE_RATE * 10) or any(
            fields[1] != "16" or fields[2] != "1000.0(0)/mV" or fields[4] != "0" for fields in signals):
        raise ValueError(f"Unexpected header layout in {header}")
    data = np.fromfile(header.with_suffix(".dat"), dtype="<i2").reshape(int(samples), LEADS)
    return data.T.astype(np.float32) / 1000


def preprocess(signals: np.ndarray, rate: int = RATE, high_pass: float = 0.5) -> np.ndarray:
    """
    Bring 500 Hz ECGs to ``rate`` and remove baseline wander.

    ``resample_poly`` applies an anti-aliasing filter before resampling (at 100 Hz it keeps one sample in
    five); a zero-phase second-order Butterworth high-pass then removes the slow drift without shifting waves.

    Parameters
    ----------
    signals : np.ndarray
        Shape ``(records, 12, 5000)`` in mV.
    rate : int
        Target sampling rate in Hz.
    high_pass : float
        Cut-off of the high-pass filter in Hz.

    Returns
    -------
    np.ndarray
        Float32, shape ``(records, 12, 10 * rate)``.
    """
    divisor = math.gcd(rate, SOURCE_RATE)
    resampled = resample_poly(signals, rate // divisor, SOURCE_RATE // divisor, axis=-1)
    high_pass_filter = butter(2, high_pass, btype="highpass", fs=rate, output="sos")
    return sosfiltfilt(high_pass_filter, resampled, axis=-1).astype(np.float32)


def load_ptbxl(root: Path, cache: Path, chunk: int = 500, rate: int = RATE, high_pass: float = 0.5
               ) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load PTB-XL metadata, the 14 named diagnoses and every signal at ``rate`` (100 Hz by default).

    The first call reads the 21,799 records at 500 Hz, preprocesses them and stores a float16 copy in
    ``cache``; later calls only read that copy.

    Parameters
    ----------
    root : Path
        PTB-XL release folder with ``ptbxl_database.csv`` and ``records500``.
    cache : Path
        ``.npy`` file of the preprocessed signals, in ``ecg_id`` order.
    chunk : int
        Records preprocessed at once while building the cache.
    rate, high_pass
        Passed to ``preprocess``; a cache must be rebuilt (another file) when they change.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        Metadata indexed by ``ecg_id`` with ``split`` (``train`` folds 1 to 8, ``val`` fold 9, ``test``
        fold 10), ``only_normal`` (statements are NORM and SR only) and one boolean column per code in
        ``CODES``; and float16 signals of shape ``(records, 12, 10 * rate)`` aligned with it.
    """
    metadata = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id").sort_index()
    statements = metadata.scp_codes.apply(ast.literal_eval)
    metadata["split"] = np.select([metadata.strat_fold <= 8, metadata.strat_fold == 9], ["train", "val"], "test")
    metadata["only_normal"] = statements.apply(lambda codes: set(codes) <= {"NORM", "SR"} and "NORM" in codes)
    for code in CODES:
        metadata[code] = statements.apply(lambda codes: code in codes)
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        signals = np.lib.format.open_memmap(cache, mode="w+", dtype=np.float16,
                                            shape=(len(metadata), LEADS, 10 * rate))
        files = [root / f"{name}.hea" for name in metadata.filename_hr]
        for start in range(0, len(files), chunk):
            raw = np.stack([read_record(header) for header in files[start:start + chunk]])
            signals[start:start + chunk] = preprocess(raw, rate, high_pass)
            print(f"\rCaché: {min(start + chunk, len(files)):,}/{len(files):,} ECG", end="", flush=True)
        signals.flush()
        print()
    signals = np.load(cache, mmap_mode="r")
    if signals.shape != (len(metadata), LEADS, 10 * rate):
        raise ValueError(f"Cache {cache} has shape {signals.shape}")
    return metadata, signals


class ECGViT(nn.Module):
    """
    Vision Transformer for 12-lead ECGs: one token per 0.2 s of all twelve leads.

    Parameters
    ----------
    width : int
        Size of every token.
    depth : int
        Number of Transformer blocks.
    heads : int
        Attention heads per block.
    """

    def __init__(self, width: int = WIDTH, depth: int = DEPTH, heads: int = HEADS) -> None:
        super().__init__()
        self.patch = nn.Linear(LEADS * PATCH, width)
        self.position = nn.Parameter(torch.zeros(1, TOKENS, width))
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, width))
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        for parameter in (self.position, self.cls, self.mask_token):
            nn.init.trunc_normal_(parameter, std=0.02)

    def forward(self, signals: torch.Tensor, start: torch.Tensor, mask: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Describe a window of ECG.

        Parameters
        ----------
        signals : torch.Tensor
            Shape ``(batch, 12, samples)``, ``samples`` a multiple of 20.
        start : torch.Tensor
            Token where each window starts inside its 10 s ECG, shape ``(batch,)``; it selects the
            position embeddings.
        mask : torch.Tensor or None
            Boolean ``(batch, samples // 20)``; masked tokens are replaced by the learned mask token.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Class token ``(batch, width)`` and patch tokens ``(batch, samples // 20, width)``.
        """
        batch, _, samples = signals.shape
        tokens = signals.unfold(2, PATCH, PATCH).permute(0, 2, 1, 3).flatten(2)
        hidden = self.patch(tokens)
        if mask is not None:
            hidden = torch.where(mask[..., None], self.mask_token.expand_as(hidden), hidden)
        positions = start[:, None] + torch.arange(samples // PATCH, device=signals.device)
        hidden = hidden + self.position[0, positions]
        hidden = self.norm(self.blocks(torch.cat([self.cls.expand(batch, -1, -1), hidden], dim=1)))
        return hidden[:, 0], hidden[:, 1:]


class ProjectionHead(nn.Module):
    """
    SimDINOv2 head without its last layer: three linear layers and unit length.

    Parameters
    ----------
    width : int
        Token size of the backbone.
    """

    def __init__(self, width: int = WIDTH) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(width, HEAD_HIDDEN), nn.GELU(), nn.Linear(HEAD_HIDDEN, HEAD_HIDDEN),
                                    nn.GELU(), nn.Linear(HEAD_HIDDEN, HEAD_OUT))

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
            Shape ``(..., 256)``, unit length.
        """
        return F.normalize(self.layers(tokens), dim=-1)


class SimDINONetwork(nn.Module):
    """
    Backbone plus projection head, used for both student and teacher.
    """

    def __init__(self) -> None:
        super().__init__()
        self.backbone = ECGViT()
        self.head = ProjectionHead()

    def forward(self, signals: torch.Tensor, start: torch.Tensor, mask: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Projected class token and raw patch tokens.

        Parameters
        ----------
        signals, start, mask
            As in ``ECGViT.forward``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Unit-length class feature ``(batch, 256)`` and backbone patch tokens ``(batch, tokens, width)``.
        """
        cls, patches = self.backbone(signals, start, mask)
        return self.head(cls), patches


def crop(signals: torch.Tensor, samples: int, patch: int = PATCH) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Cut one window per ECG at a random token boundary.

    Parameters
    ----------
    signals : torch.Tensor
        Shape ``(batch, 12, record samples)``.
    samples : int
        Window length, a multiple of ``patch``.
    patch : int
        Samples per token.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Windows ``(batch, 12, samples)`` and their first token ``(batch,)``.
    """
    batch, _, total = signals.shape
    start = torch.randint(0, (total - samples) // patch + 1, (batch,), device=signals.device)
    index = (start * patch)[:, None, None] + torch.arange(samples, device=signals.device)
    return signals.gather(2, index.expand(batch, LEADS, samples)), start


def augment(crops: torch.Tensor, rate: int = RATE) -> torch.Tensor:
    """
    Perturb windows without changing what a cardiologist would read.

    Every window is scaled by 0.8 to 1.2, gets white noise of up to 0.05 mV and a slow baseline wave of up to
    0.1 mV per lead (0.05 to 0.5 Hz), and with probability 0.3 loses one random lead. Time is never
    reversed, stretched or flipped in sign, and leads are never swapped, because those change the diagnosis.

    Parameters
    ----------
    crops : torch.Tensor
        Shape ``(batch, 12, samples)``.
    rate : int
        Sampling rate in Hz, for the frequency of the baseline wave.

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
    keep = torch.ones(batch, leads, 1, device=device)
    dropped = (torch.rand(batch, device=device) < 0.3).nonzero().flatten()
    keep[dropped, torch.randint(0, leads, (len(dropped),), device=device)] = 0
    return (crops * scale + noise + wander) * keep


def span_mask(views: int, tokens: int, share: float = MASK_SHARE, ratio: tuple[float, float] = MASK_RATIO,
              span: tuple[int, int] = MASK_SPAN) -> torch.Tensor:
    """
    Hide contiguous runs of tokens from the student, as DINOv2 hides blocks of image patches.

    Half of the views are masked; a masked view hides 10% to 50% of its tokens in runs of 1 to 4 tokens
    (0.2 to 0.8 s), long enough to cover a whole wave.

    Parameters
    ----------
    views : int
        Number of global views in the batch.
    tokens : int
        Tokens per view.
    share : float
        Share of views that are masked.
    ratio : tuple[float, float]
        Smallest and largest share of tokens hidden in a masked view.
    span : tuple[int, int]
        Shortest and longest run of hidden tokens.

    Returns
    -------
    torch.Tensor
        Boolean CPU tensor ``(views, tokens)``.
    """
    mask = torch.zeros(views, tokens, dtype=torch.bool)
    for view in (torch.rand(views) < share).nonzero().flatten().tolist():
        target = round((ratio[0] + (ratio[1] - ratio[0]) * torch.rand(1).item()) * tokens)
        while mask[view].sum() < target:
            length = int(torch.randint(span[0], span[1] + 1, (1,)))
            start = int(torch.randint(0, tokens - length + 1, (1,)))
            mask[view, start:start + length] = True
    return mask


def coding_rate(features: torch.Tensor, epsilon: float = EPSILON) -> torch.Tensor:
    """
    Half the log-determinant that measures how much space a batch of features fills.

    As in the official ``MCRLoss.calc_expansion``: ``1/2 logdet(I + d / (n * eps) * Z^T Z)``. It is 0 when all
    features are 0 and grows as they spread over more directions. Computed on the CPU in float64 because the
    Apple GPU has no Cholesky factorization.

    Parameters
    ----------
    features : torch.Tensor
        Shape ``(n, d)``.
    epsilon : float
        Precision of the rate.

    Returns
    -------
    torch.Tensor
        Float64 scalar on the CPU.
    """
    count, size = features.shape
    gram = (features.T @ features).cpu().double()
    matrix = torch.eye(size, dtype=torch.float64) + size / (count * epsilon) * gram
    return torch.linalg.cholesky(matrix).diagonal().log().sum()


def simdinov2_loss(student_views: torch.Tensor, teacher_views: torch.Tensor, student_patches: torch.Tensor,
                   teacher_patches: torch.Tensor, mask: torch.Tensor, epsilon: float = EPSILON
                   ) -> dict[str, torch.Tensor]:
    """
    SimDINOv2 loss: agree with the teacher across views and on hidden tokens, without collapsing.

    Parameters
    ----------
    student_views : torch.Tensor
        Unit-length student class features, ``(2 + local views, batch, 256)``, global views first.
    teacher_views : torch.Tensor
        Unit-length teacher class features of the two unmasked global views, ``(2, batch, 256)``.
    student_patches, teacher_patches : torch.Tensor
        Unit-length features of the hidden tokens, ``(hidden tokens, 256)``, in ``mask.nonzero()`` order.
    mask : torch.Tensor
        Boolean ``(2 * batch, tokens)`` of the hidden tokens.
    epsilon : float
        Precision of the coding rate.

    Returns
    -------
    dict[str, torch.Tensor]
        ``view``: 1 minus the mean cosine between each teacher global view and every other student view.
        ``patch``: 1 minus the mean cosine on hidden tokens, each masked view weighted equally.
        ``coding_rate``: mean rate of the averaged student and teacher global features.
        ``loss``: ``view + patch - gamma * coding_rate`` with ``gamma = (d + n) / (d * n)``.
    """
    views, batch, size = student_views.shape
    similarity = torch.einsum("tbd,sbd->ts", teacher_views, student_views) / batch
    keep = torch.ones_like(similarity)
    keep[:, :GLOBAL_VIEWS] -= torch.eye(GLOBAL_VIEWS, device=keep.device)
    view = 1 - (similarity * keep).sum() / keep.sum()
    hidden_view = mask.nonzero()[:, 0]
    counts = mask.sum(dim=1)
    weight = 1 / counts[hidden_view] / (counts > 0).sum()
    patch = 1 - (weight * (student_patches * teacher_patches).sum(dim=-1)).sum()
    expansion = (student_views[:GLOBAL_VIEWS] + teacher_views) / 2
    rate = torch.stack([coding_rate(features, epsilon) for features in expansion]).mean()
    gamma = (size + batch) / (size * batch)
    loss = view + patch - gamma * rate.float().to(view.device)
    return {"loss": loss, "view": view, "patch": patch, "coding_rate": rate}


def training_step(student: SimDINONetwork, teacher: SimDINONetwork, signals: torch.Tensor
                  ) -> dict[str, torch.Tensor]:
    """
    Build the views of one batch and compute the SimDINOv2 loss.

    Parameters
    ----------
    student, teacher : SimDINONetwork
        The trained network and its moving average.
    signals : torch.Tensor
        Shape ``(batch, 12, 1000)`` on the training device.

    Returns
    -------
    dict[str, torch.Tensor]
        Output of ``simdinov2_loss``.
    """
    batch = signals.shape[0]
    global_parts = [crop(signals, GLOBAL_SAMPLES) for _ in range(GLOBAL_VIEWS)]
    local_parts = [crop(signals, LOCAL_SAMPLES) for _ in range(LOCAL_VIEWS)]
    global_views = augment(torch.cat([part[0] for part in global_parts]))
    global_start = torch.cat([part[1] for part in global_parts])
    local_views = augment(torch.cat([part[0] for part in local_parts]))
    local_start = torch.cat([part[1] for part in local_parts])
    mask = span_mask(len(global_views), GLOBAL_SAMPLES // PATCH).to(signals.device)
    with torch.no_grad():
        teacher_cls, teacher_tokens = teacher(global_views, global_start)
        teacher_patches = teacher.head(teacher_tokens[mask])
    student_cls, student_tokens = student(global_views, global_start, mask)
    student_local, _ = student(local_views, local_start)
    student_patches = student.head(student_tokens[mask])
    student_views = torch.cat([student_cls, student_local]).view(GLOBAL_VIEWS + LOCAL_VIEWS, batch, HEAD_OUT)
    return simdinov2_loss(student_views, teacher_cls.view(GLOBAL_VIEWS, batch, HEAD_OUT), student_patches,
                          teacher_patches, mask)


def cosine_schedule(start: float, end: float, progress: float) -> float:
    """
    Move from ``start`` to ``end`` along half a cosine.

    Parameters
    ----------
    start, end : float
        Values at progress 0 and 1.
    progress : float
        Between 0 and 1.

    Returns
    -------
    float
        Interpolated value.
    """
    return end + (start - end) * (1 + math.cos(math.pi * progress)) / 2


@torch.no_grad()
def encode(backbone: ECGViT, signals: np.ndarray, *, device: torch.device, batch_size: int = 256,
           patches: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Describe whole 10 s ECGs with a frozen backbone.

    Parameters
    ----------
    backbone : ECGViT
        Encoder in evaluation mode after the call.
    signals : np.ndarray
        Shape ``(records, 12, 1000)``.
    device : torch.device
        Where the backbone is.
    batch_size : int
        ECGs per forward pass.
    patches : bool
        Also return the patch tokens, in float16.

    Returns
    -------
    np.ndarray or tuple[np.ndarray, np.ndarray]
        Class tokens ``(records, width)`` and, if requested, patch tokens ``(records, 50, width)``.
    """
    backbone.eval()
    classes, tokens = [], []
    for first in range(0, len(signals), batch_size):
        batch = torch.as_tensor(np.asarray(signals[first:first + batch_size], dtype=np.float32), device=device)
        cls, patch_tokens = backbone(batch, torch.zeros(len(batch), dtype=torch.long, device=device))
        classes.append(cls.float().cpu().numpy())
        if patches:
            tokens.append(patch_tokens.half().cpu().numpy())
    if patches:
        return np.concatenate(classes), np.concatenate(tokens)
    return np.concatenate(classes)


def train_simdino(student: SimDINONetwork, teacher: SimDINONetwork, train_signals: np.ndarray,
                  monitor_signals: np.ndarray, *, epochs: int, batch_size: int, learning_rate: float,
                  warmup_epochs: int, seed: int, device: torch.device, checkpoint_path: Path,
                  resume: bool = True) -> pd.DataFrame:
    """
    Pretrain the student with SimDINOv2 and keep its moving-average teacher.

    The learning rate rises linearly for ``warmup_epochs`` and then follows a cosine to 1e-6; weight decay
    follows a cosine from 0.04 to 0.4 and the teacher momentum from 0.992 to 1, as in the official
    configuration. A checkpoint is written after every epoch, and a finished run is only reloaded. On the
    Apple GPU the allocator cache is emptied after every update: it otherwise keeps freed blocks of the
    variable-size masked batches and grew past the 16 GB of the machine in the first run.

    Parameters
    ----------
    student, teacher : SimDINONetwork
        Networks on ``device``; the teacher starts as a copy of the student.
    train_signals : np.ndarray
        ECGs used for pretraining, ``(records, 12, 1000)``; their labels are never read.
    monitor_signals : np.ndarray
        ECGs whose teacher class tokens measure collapse after every epoch (effective rank).
    epochs : int
        Passes over ``train_signals``; the last incomplete batch is dropped.
    batch_size : int
        ECGs per update.
    learning_rate : float
        Peak learning rate of AdamW.
    warmup_epochs : int
        Epochs of linear warm-up.
    seed : int
        Seed of the weights' updates, crops, augmentations and masks.
    device : torch.device
        Training device.
    checkpoint_path : Path
        File replaced after every epoch; it is written to a ``.partial`` file first and then renamed, so a
        reader never sees half a checkpoint.
    resume : bool
        Continue from ``checkpoint_path`` when it exists.

    Returns
    -------
    pd.DataFrame
        One row per epoch: ``loss``, ``view``, ``patch``, ``coding_rate``, ``rank`` (effective rank of the
        monitor class tokens), ``learning_rate``, ``momentum`` and ``seconds``.

    Raises
    ------
    FloatingPointError
        If the loss stops being finite.
    """
    decay = [p for n, p in student.named_parameters() if p.ndim > 1 and n.split(".")[-1] not in
             ("position", "cls", "mask_token")]
    no_decay = [p for n, p in student.named_parameters() if not (p.ndim > 1 and n.split(".")[-1] not in
                ("position", "cls", "mask_token"))]
    optimizer = torch.optim.AdamW([{"params": decay}, {"params": no_decay, "weight_decay": 0.0}],
                                  lr=learning_rate)
    rows, first_epoch = [], 0
    if resume and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        student.load_state_dict(state["student"])
        teacher.load_state_dict(state["teacher"])
        optimizer.load_state_dict(state["optimizer"])
        rows, first_epoch = state["history"], state["epoch"]
        torch.set_rng_state(state["cpu_rng"].cpu())
        np.random.set_state(state["numpy_rng"])
        print(f"Reanudado desde la época {first_epoch} de {epochs}.")
        if first_epoch >= epochs:
            return pd.DataFrame(rows)
    else:
        torch.manual_seed(seed)
        np.random.seed(seed)
    train = torch.as_tensor(np.asarray(train_signals, dtype=np.float32), device=device)
    steps = len(train) // batch_size
    total, warmup = epochs * steps, warmup_epochs * steps
    for epoch in range(first_epoch, epochs):
        student.train()
        order = torch.as_tensor(np.random.permutation(len(train)), device=device)
        totals, started = torch.zeros(4), time.perf_counter()
        for step in range(steps):
            done = epoch * steps + step
            rate = (learning_rate * (done + 1) / warmup if done < warmup else
                    cosine_schedule(learning_rate, 1e-6, (done - warmup) / max(1, total - warmup)))
            optimizer.param_groups[0]["lr"] = optimizer.param_groups[1]["lr"] = rate
            optimizer.param_groups[0]["weight_decay"] = cosine_schedule(*WEIGHT_DECAY, done / total)
            terms = training_step(student, teacher, train[order[step * batch_size:(step + 1) * batch_size]])
            optimizer.zero_grad(set_to_none=True)
            terms["loss"].backward()
            nn.utils.clip_grad_norm_(student.parameters(), GRADIENT_NORM)
            optimizer.step()
            momentum = cosine_schedule(MOMENTUM_START, 1.0, (done + 1) / total)
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
                    "cpu_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state()}, partial)
        os.replace(partial, checkpoint_path)
        print(f"Época {epoch + 1}/{epochs} pérdida={means[0]:.4f} vistas={means[1]:.4f} "
              f"parches={means[2]:.4f} tasa={means[3]:.1f} rango={rank:.1f} "
              f"{rows[-1]['seconds']:.0f} s", flush=True)
    return pd.DataFrame(rows)


def normalize(features: np.ndarray) -> np.ndarray:
    """
    Scale every row to unit length, so dot products are cosines.

    Parameters
    ----------
    features : np.ndarray
        Shape ``(n, d)``.

    Returns
    -------
    np.ndarray
        Float32, same shape.
    """
    features = np.asarray(features, dtype=np.float32)
    return features / np.linalg.norm(features, axis=1, keepdims=True).clip(min=1e-12)


def knn_scores(query: np.ndarray, bank: np.ndarray, labels: np.ndarray, k: int,
               temperature: float = TEMPERATURE) -> np.ndarray:
    """
    Score every class by the votes of the ``k`` most similar references.

    Votes are weighted by ``exp(cosine / temperature)``, the weighting of the DINO k-NN evaluation, so a
    reference that is much more similar counts more. An ECG can collect votes for several classes.

    Parameters
    ----------
    query : np.ndarray
        Embeddings to label, ``(n, d)``.
    bank : np.ndarray
        Reference embeddings, ``(m, d)``.
    labels : np.ndarray
        Boolean or 0/1 labels of the references, ``(m, classes)``.
    k : int
        Neighbours that vote.
    temperature : float
        Softness of the weights.

    Returns
    -------
    np.ndarray
        Share of weighted votes per class, ``(n, classes)``, between 0 and 1.
    """
    query, bank = normalize(query), normalize(bank)
    labels = np.asarray(labels, dtype=np.float32)
    scores = []
    for first in range(0, len(query), 1024):
        similarity = query[first:first + 1024] @ bank.T
        nearest = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
        top = np.take_along_axis(similarity, nearest, axis=1)
        weight = np.exp((top - top.max(axis=1, keepdims=True)) / temperature)
        scores.append((weight[..., None] * labels[nearest]).sum(axis=1) / weight.sum(axis=1, keepdims=True))
    return np.concatenate(scores)


def prototype_scores(query: np.ndarray, bank: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Cosine to the mean reference of every class, as the memory bank of "No time to train!".

    Parameters
    ----------
    query : np.ndarray
        Embeddings to label, ``(n, d)``.
    bank : np.ndarray
        Reference embeddings, ``(m, d)``.
    labels : np.ndarray
        Boolean labels of the references, ``(m, classes)``.

    Returns
    -------
    np.ndarray
        Cosine between each query and each class prototype, ``(n, classes)``.
    """
    labels = np.asarray(labels, dtype=np.float32)
    prototypes = labels.T @ normalize(bank) / labels.sum(axis=0)[:, None]
    return normalize(query) @ normalize(prototypes).T


def class_aurocs(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """
    AUROC of every class.

    Parameters
    ----------
    labels : np.ndarray
        Boolean ``(n, classes)``.
    scores : np.ndarray
        Scores ``(n, classes)``.

    Returns
    -------
    np.ndarray
        One AUROC per class.
    """
    return np.array([roc_auc_score(labels[:, column], scores[:, column]) for column in range(labels.shape[1])])


def auroc_intervals(labels: np.ndarray, scores: np.ndarray, patients: np.ndarray, names: list[str], *,
                    reference: np.ndarray | None = None, draws: int = 2000, seed: int = 2026) -> pd.DataFrame:
    """
    Whole-patient bootstrap of per-class and macro AUROC, optionally paired against reference scores.

    Draws come from ``intervals.patient_resample``; a draw where some class has a single value is skipped,
    and the number skipped is reported.

    Parameters
    ----------
    labels : np.ndarray
        Boolean ``(n, classes)``.
    scores : np.ndarray
        Scores ``(n, classes)``.
    patients : np.ndarray
        Patient of each ECG.
    names : list[str]
        Name of every class, used as row labels.
    reference : np.ndarray or None
        Scores of a second model on the same ECGs; then the interval is of ``scores`` minus ``reference``.
    draws : int
        Bootstrap draws, including skipped ones.
    seed : int
        Seed of ``numpy.random.default_rng``.

    Returns
    -------
    pd.DataFrame
        Rows: each class and ``macro``. Columns ``value``, ``ci_low``, ``ci_high``, ``skipped_draws``.
    """
    def statistic(rows: np.ndarray) -> np.ndarray:
        values = class_aurocs(labels[rows], scores[rows])
        if reference is not None:
            values = values - class_aurocs(labels[rows], reference[rows])
        return np.append(values, values.mean())

    labels = np.asarray(labels, dtype=bool)
    groups, rng = patient_groups(np.asarray(patients)), np.random.default_rng(seed)
    everything = np.arange(len(labels))
    samples = []
    for _ in range(draws):
        rows = patient_resample(groups, rng)
        positives = labels[rows].sum(axis=0)
        if np.all((positives > 0) & (positives < len(rows))):
            samples.append(statistic(rows))
    low, high = np.percentile(np.array(samples), [2.5, 97.5], axis=0)
    return pd.DataFrame({"value": statistic(everything), "ci_low": low, "ci_high": high,
                         "skipped_draws": draws - len(samples)}, index=[*names, "macro"])


def threshold_for_sensitivity(scores: np.ndarray, positive: np.ndarray, sensitivity: float) -> float:
    """
    Largest threshold that flags at least ``sensitivity`` of the positives.

    Parameters
    ----------
    scores : np.ndarray
        Higher means more suspicious.
    positive : np.ndarray
        Boolean target.
    sensitivity : float
        Required share of positives with ``score >= threshold``.

    Returns
    -------
    float
        The threshold.
    """
    ordered = np.sort(np.asarray(scores)[np.asarray(positive, dtype=bool)])
    return float(ordered[int(math.floor((1 - sensitivity) * len(ordered)))])


def detect_beats(signal: np.ndarray, rate: int = RATE) -> np.ndarray:
    """
    Find QRS complexes, the role SAM plays for objects in the image method.

    The twelve leads are band-passed at 5 to 15 Hz, where QRS energy concentrates; the absolute slopes are
    summed over leads and smoothed over 0.1 s. Peaks at least 0.25 s apart that reach 40% of the median of the
    ten largest peaks are beats.

    Parameters
    ----------
    signal : np.ndarray
        One ECG, ``(12, samples)``.
    rate : int
        Its sampling rate in Hz.

    Returns
    -------
    np.ndarray
        Sample index of every beat.
    """
    band = sosfiltfilt(butter(2, [5, 15], btype="bandpass", fs=rate, output="sos"),
                       np.asarray(signal, dtype=np.float64), axis=-1)
    energy = np.abs(np.diff(band, axis=-1, prepend=band[:, :1])).sum(axis=0)
    width = int(0.1 * rate)
    energy = np.convolve(energy, np.ones(width) / width, mode="same")
    peaks, properties = find_peaks(energy, height=0, distance=int(0.25 * rate))
    if not len(peaks):
        return peaks
    level = np.median(np.sort(properties["peak_heights"])[::-1][:10])
    return peaks[properties["peak_heights"] >= 0.4 * level]


def beat_embeddings(patch_tokens: np.ndarray, beats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Average the patch tokens that cover each beat, the way the image method averages a mask.

    A beat covers the tokens whose centre lies from 0.25 s before to 0.45 s after its QRS, which holds the
    P wave, the QRS and the T wave at normal rates. Beats whose window leaves the 10 s record are dropped.

    Parameters
    ----------
    patch_tokens : np.ndarray
        Tokens of one ECG, ``(50, width)``.
    beats : np.ndarray
        Output of ``detect_beats``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Unit-length beat embeddings ``(kept beats, width)`` and the kept beat positions.
    """
    before, after = (int(round(edge * RATE)) for edge in BEAT_WINDOW)
    kept = beats[(beats + before >= 0) & (beats + after <= SAMPLES)]
    centres = np.arange(TOKENS) * PATCH + PATCH / 2
    rows = [np.asarray(patch_tokens, dtype=np.float32)[(centres >= beat + before) & (centres < beat + after)]
            .mean(axis=0) for beat in kept]
    return normalize(np.array(rows).reshape(len(kept), patch_tokens.shape[1])), kept


def beat_anomaly(beats: np.ndarray, normal_bank: np.ndarray, k: int = 10) -> np.ndarray:
    """
    How unlike normal beats each beat is: 1 minus its mean cosine to the ``k`` most similar normal beats.

    Parameters
    ----------
    beats : np.ndarray
        Unit-length beat embeddings, ``(n, d)``.
    normal_bank : np.ndarray
        Unit-length embeddings of beats from normal ECGs, ``(m, d)``.
    k : int
        Normal beats compared.

    Returns
    -------
    np.ndarray
        One value per beat; 0 means identical to normal beats.
    """
    similarity = beats @ normal_bank.T
    return 1 - np.sort(similarity, axis=1)[:, -k:].mean(axis=1)


def ecg_beats(signals: np.ndarray, patch_tokens: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Detect the beats of many ECGs and describe each with its patch tokens.

    Parameters
    ----------
    signals : np.ndarray
        Shape ``(records, 12, 1000)``.
    patch_tokens : np.ndarray
        Shape ``(records, 50, width)``, from ``encode`` on the same records.

    Returns
    -------
    tuple[list[np.ndarray], list[np.ndarray]]
        Per record, the unit-length beat embeddings and the sample positions of the kept beats.
    """
    embeddings, positions = [], []
    for signal, tokens in zip(signals, patch_tokens, strict=True):
        described, kept = beat_embeddings(tokens, detect_beats(signal))
        embeddings.append(described)
        positions.append(kept)
    return embeddings, positions


def record_beat_scores(beats: list[np.ndarray], normal_bank: np.ndarray, k: int = 10) -> pd.DataFrame:
    """
    Summarize the beat anomalies of every ECG.

    Parameters
    ----------
    beats : list[np.ndarray]
        Beat embeddings per record, from ``ecg_beats``.
    normal_bank : np.ndarray
        Unit-length embeddings of normal beats.
    k : int
        Normal beats compared with each beat.

    Returns
    -------
    pd.DataFrame
        One row per record: ``beats`` (kept beats), ``maximum`` (largest anomaly) and ``contrast`` (largest
        minus median anomaly, high when one beat differs from the others). Both are NaN without beats.
    """
    rows = []
    for described in beats:
        if not len(described):
            rows.append({"beats": 0, "maximum": np.nan, "contrast": np.nan})
            continue
        anomaly = beat_anomaly(described, normal_bank, k)
        rows.append({"beats": len(described), "maximum": anomaly.max(),
                     "contrast": anomaly.max() - np.median(anomaly)})
    return pd.DataFrame(rows)
