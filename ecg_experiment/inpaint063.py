"""
Experiment 063: masked cross-lead and cross-beat reconstruction trained on clean normal ECGs.

A 1D U-Net with a transformer bottleneck sees a 12-lead ECG at 250 Hz with some lead-time positions
hidden and predicts a Gaussian (mean, log-variance) for every sample. It is trained on clean normals
only, scoring the masked positions. At inference a fixed set of masks covers every lead-sample several
times, and the map is the mean of ``|x - mu| / sigma`` over the masks that hide each sample.

Mask units: the six limb leads are one unit (they are exact linear combinations of I and II, so a
single hidden limb lead could be copied from two visible ones); each precordial lead is its own unit.
Mask families: random lead-time blocks, phase masks (the same R-relative window in every beat) and
whole-unit masks. See ``docs/experiment-063-masked-reconstruction.md``.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt
from torch import nn

from ecg_experiment.fragment_localization import r_peaks

FS_RAW = 500
FS = 250
UNITS: tuple[tuple[int, ...], ...] = ((0, 1, 2, 3, 4, 5), (6,), (7,), (8,), (9,), (10,), (11,))
FAMILIES = ("block", "phase", "whole")
CROP = 2048
PAD_MULTIPLE = 16
BLOCK = 50
BLOCK_PERIOD = 4
PHASE_CENTRES_MS = tuple(range(-280, 521, 80))
PHASE_WIDTH_MS = 160
SMOOTH = 11
LOGVAR_SCALE = 6.0
LOGVAR_SHIFT = -3.0
# Electrode -> lead projection for RA, LA, LL, V1-V6 (Wilson terminal for the chest leads).
ELECTRODES = 9


def electrode_matrix() -> np.ndarray:
    """
    Linear map from nine electrode potentials to the twelve standard leads.

    Returns
    -------
    np.ndarray
        Shape (12, 9) for electrodes RA, LA, LL, V1-V6, in canonical lead order.
    """
    m = np.zeros((12, ELECTRODES))
    ra, la, ll = np.eye(ELECTRODES)[:3]
    m[0] = la - ra
    m[1] = ll - ra
    m[2] = ll - la
    m[3] = ra - (la + ll) / 2
    m[4] = la - (ra + ll) / 2
    m[5] = ll - (ra + la) / 2
    for k in range(6):
        m[6 + k] = np.eye(ELECTRODES)[3 + k] - (ra + la + ll) / 3
    return m


def preprocess(signals: np.ndarray) -> np.ndarray:
    """
    Band-pass 0.5-40 Hz (second-order Butterworth, zero phase) and decimate 500 Hz to 250 Hz.

    Parameters
    ----------
    signals : np.ndarray
        Raw records (n, 12, 5000) in mV at 500 Hz.

    Returns
    -------
    np.ndarray
        Float32 array (n, 12, 2500) in mV at 250 Hz; sample i is raw sample 2i.
    """
    sos = butter(2, [0.5, 40.0], btype="band", fs=FS_RAW, output="sos")
    return sosfiltfilt(sos, np.nan_to_num(np.asarray(signals, dtype=np.float64)), axis=-1)[..., ::2].astype(
        np.float32
    )


def peaks_250(raw: np.ndarray) -> np.ndarray:
    """
    Frozen R peaks of one raw record, in 250 Hz samples.

    Parameters
    ----------
    raw : np.ndarray
        Raw record (12, 5000) in mV at 500 Hz.

    Returns
    -------
    np.ndarray
        Ascending peak positions at 250 Hz.
    """
    return r_peaks(np.asarray(raw), FS_RAW) // 2


def upsample(maps: np.ndarray) -> np.ndarray:
    """
    Linear interpolation of 250 Hz maps to the 500 Hz record grid.

    Parameters
    ----------
    maps : np.ndarray
        Maps (..., 2500) where sample i sits at raw sample 2i.

    Returns
    -------
    np.ndarray
        Float32 maps (..., 5000).
    """
    grid = np.arange(2 * maps.shape[-1]) / 2.0
    flat = maps.reshape(-1, maps.shape[-1])
    out = np.stack([np.interp(grid, np.arange(maps.shape[-1]), row) for row in flat])
    return out.reshape(*maps.shape[:-1], 2 * maps.shape[-1]).astype(np.float32)


def unit_leads(units: list[int] | np.ndarray) -> list[int]:
    """
    Lead indices covered by a set of mask units.

    Parameters
    ----------
    units : list of int or np.ndarray
        Unit indices into ``UNITS``.

    Returns
    -------
    list[int]
        Lead indices.
    """
    return [lead for unit in units for lead in UNITS[int(unit)]]


def phase_mask_into(mask: np.ndarray, leads: list[int], peaks: np.ndarray, start: int, stop: int) -> None:
    """
    Hide the window [peak + start, peak + stop) of every beat on the given leads, in place.

    Parameters
    ----------
    mask : np.ndarray
        Boolean (12, L), true where hidden.
    leads : list[int]
        Lead indices.
    peaks : np.ndarray
        R peaks in the mask's sample coordinates; peaks outside the mask still hide their in-range part.
    start, stop : int
        Window relative to each R peak, in samples.
    """
    length = mask.shape[1]
    for peak in peaks:
        lo, hi = max(int(peak) + start, 0), min(int(peak) + stop, length)
        if hi > lo:
            mask[leads, lo:hi] = True


def train_mask(peaks: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    """
    One random training mask from the block, phase or whole-unit family.

    Parameters
    ----------
    peaks : np.ndarray
        R peaks in the crop's 250 Hz coordinates.
    length : int
        Crop length in samples.
    rng : np.random.Generator
        Random source.

    Returns
    -------
    np.ndarray
        Boolean (12, length), true where hidden; at least one sample is hidden.
    """
    mask = np.zeros((12, length), dtype=bool)
    family = rng.choice(3, p=[0.4, 0.4, 0.2])
    if family == 0:
        for _ in range(rng.integers(2, 8)):
            width = int(rng.integers(25, 76))
            start = int(rng.integers(0, length - width))
            leads = unit_leads(rng.choice(len(UNITS), rng.integers(1, 4), replace=False))
            mask[leads, start : start + width] = True
    elif family == 1:
        for _ in range(rng.integers(1, 3)):
            centre = rng.uniform(-0.3, 0.5) * FS
            half = rng.uniform(0.08, 0.25) * FS / 2
            leads = unit_leads(rng.choice(len(UNITS), rng.integers(1, 4), replace=False))
            phase_mask_into(mask, leads, peaks, int(round(centre - half)), int(round(centre + half)))
        if not mask.any():
            mask[unit_leads([rng.integers(len(UNITS))])] = True
    else:
        mask[unit_leads(rng.choice(len(UNITS), rng.integers(1, 3), replace=False))] = True
    return mask


def inference_masks(peaks: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the fixed inference mask set of one record.

    Block family: 200 ms blocks, every fourth block, at offsets 0 and 100 ms, for each unit (56 masks,
    every lead-sample hidden twice). Phase family: 160 ms R-relative windows centred from -280 to
    +520 ms in 80 ms steps, in every beat, for each unit (77 masks). Whole family: each unit for the
    whole record (7 masks).

    Parameters
    ----------
    peaks : np.ndarray
        R peaks at 250 Hz.
    length : int
        Record length at 250 Hz.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Boolean masks (M, 12, length) and the family index (M,) into ``FAMILIES``.
    """
    masks, families = [], []
    t = np.arange(length)
    for unit in range(len(UNITS)):
        leads = unit_leads([unit])
        for offset in (0, BLOCK // 2):
            for phase in range(BLOCK_PERIOD):
                mask = np.zeros((12, length), dtype=bool)
                mask[np.ix_(leads, ((t + offset) // BLOCK) % BLOCK_PERIOD == phase)] = True
                masks.append(mask)
                families.append(0)
        half = PHASE_WIDTH_MS * FS // 2000
        for centre_ms in PHASE_CENTRES_MS:
            centre = centre_ms * FS // 1000
            mask = np.zeros((12, length), dtype=bool)
            phase_mask_into(mask, leads, peaks, centre - half, centre + half)
            masks.append(mask)
            families.append(1)
        mask = np.zeros((12, length), dtype=bool)
        mask[leads] = True
        masks.append(mask)
        families.append(2)
    return np.stack(masks), np.asarray(families)


def electrode_noise(batch: int, length: int, rng: np.random.Generator) -> np.ndarray:
    """
    Random electrode-level noise projected to the twelve leads.

    Each of the nine electrodes is noisy with probability 0.3 (at least one is). A noisy electrode gets
    white noise (standard deviation log-uniform 3-50 uV), a slow sinusoid (0.5-1.5 Hz, up to 0.15 mV)
    or muscle-like noise (white noise differenced, up to 50 uV), chosen at random.

    Parameters
    ----------
    batch : int
        Number of records.
    length : int
        Samples at 250 Hz.
    rng : np.random.Generator
        Random source.

    Returns
    -------
    np.ndarray
        Float32 noise (batch, 12, length) in mV.
    """
    noise = np.zeros((batch, ELECTRODES, length))
    t = np.arange(length) / FS
    for b in range(batch):
        active = rng.random(ELECTRODES) < 0.3
        if not active.any():
            active[rng.integers(ELECTRODES)] = True
        for e in np.flatnonzero(active):
            kind = rng.integers(3)
            if kind == 0:
                noise[b, e] = rng.normal(0, math.exp(rng.uniform(math.log(0.003), math.log(0.05))), length)
            elif kind == 1:
                phase = rng.uniform(0, 2 * np.pi)
                noise[b, e] = rng.uniform(0, 0.15) * np.sin(2 * np.pi * rng.uniform(0.5, 1.5) * t + phase)
            else:
                white = rng.normal(0, 1, length + 1)
                noise[b, e] = rng.uniform(0, 0.05) * np.diff(white) / np.sqrt(2)
    return np.einsum("le,bet->blt", electrode_matrix(), noise).astype(np.float32)


class ConvBlock(nn.Module):
    """Two convolutions with group norm and GELU, plus a 1x1 residual."""

    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(cin, cout, 7, padding=3),
            nn.GroupNorm(8, cout),
            nn.GELU(),
            nn.Conv1d(cout, cout, 7, padding=3),
            nn.GroupNorm(8, cout),
            nn.GELU(),
        )
        self.skip = nn.Conv1d(cin, cout, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block."""
        return self.body(x) + self.skip(x)


class InpaintNet(nn.Module):
    """
    U-Net over time (leads and mask indicators as channels) with a transformer bottleneck.

    Input (B, 24, L): the masked signal (hidden samples set to zero) and the mask indicator.
    Output: mean (B, 12, L) in mV and log-variance (B, 12, L) in (-9, 3). L must be a multiple of 16.
    """

    def __init__(
        self, widths: tuple[int, ...] = (64, 96, 128, 192, 256), layers: int = 4, heads: int = 4
    ) -> None:
        super().__init__()
        self.stem = ConvBlock(24, widths[0])
        self.down = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(widths[k], widths[k + 1], 4, stride=2, padding=1),
                ConvBlock(widths[k + 1], widths[k + 1]),
            )
            for k in range(len(widths) - 1)
        )
        dim = widths[-1]
        self.position = nn.Conv1d(dim, dim, 9, padding=4, groups=dim)
        layer = nn.TransformerEncoderLayer(
            dim, heads, 2 * dim, dropout=0.1, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, layers)
        self.up = nn.ModuleList(
            nn.ConvTranspose1d(widths[k + 1], widths[k], 4, stride=2, padding=1)
            for k in reversed(range(len(widths) - 1))
        )
        self.merge = nn.ModuleList(
            ConvBlock(2 * widths[k], widths[k]) for k in reversed(range(len(widths) - 1))
        )
        self.head = nn.Conv1d(widths[0], 24, 1)

    def forward(self, signal: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict mean and log-variance from a signal (B, 12, L) and a boolean mask (B, 12, L)."""
        x = self.stem(torch.cat([signal * (~mask), mask.to(signal.dtype)], dim=1))
        skips = [x]
        for down in self.down:
            x = down(x)
            skips.append(x)
        x = x + self.position(x)
        x = self.transformer(x.transpose(1, 2)).transpose(1, 2)
        for up, merge, skip in zip(self.up, self.merge, reversed(skips[:-1]), strict=True):
            x = merge(torch.cat([up(x), skip], dim=1))
        out = self.head(x)
        mean, raw = out[:, :12], out[:, 12:]
        return mean, LOGVAR_SCALE * torch.tanh(raw / LOGVAR_SCALE) + LOGVAR_SHIFT


def beta_nll(
    target: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor, mask: torch.Tensor, beta: float = 0.5
) -> torch.Tensor:
    """
    Beta-weighted Gaussian negative log-likelihood over hidden samples (Seitzer et al., 2022).

    Parameters
    ----------
    target, mean, logvar : torch.Tensor
        Shape (B, 12, L).
    mask : torch.Tensor
        Boolean (B, 12, L), true where hidden; only these samples count.
    beta : float
        Weight exponent on the detached variance; 0 is the plain Gaussian NLL.

    Returns
    -------
    torch.Tensor
        Scalar mean loss over hidden samples.
    """
    nll = 0.5 * (logvar + (target - mean) ** 2 * torch.exp(-logvar))
    if beta:
        nll = nll * torch.exp(beta * logvar.detach())
    return nll[mask].mean()


def gaussian_nll(
    target: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Plain Gaussian negative log-likelihood (without the constant) over hidden samples.

    Parameters
    ----------
    target, mean, logvar : torch.Tensor
        Shape (B, 12, L).
    mask : torch.Tensor
        Boolean (B, 12, L), true where hidden.

    Returns
    -------
    torch.Tensor
        Scalar mean over hidden samples.
    """
    return beta_nll(target, mean, logvar, mask, beta=0.0)


def pad_length(length: int) -> int:
    """
    Smallest multiple of ``PAD_MULTIPLE`` at least ``length``.

    Parameters
    ----------
    length : int
        Samples.

    Returns
    -------
    int
        Padded length.
    """
    return -(-length // PAD_MULTIPLE) * PAD_MULTIPLE


@torch.no_grad()
def record_errors(
    model: nn.Module, signal: np.ndarray, peaks: np.ndarray, device: str, chunk: int = 70
) -> dict[str, np.ndarray]:
    """
    Per-family mean z-scores and absolute errors of one record over the fixed inference masks.

    Parameters
    ----------
    model : nn.Module
        Trained ``InpaintNet`` in eval mode.
    signal : np.ndarray
        Preprocessed record (12, 2500) at 250 Hz.
    peaks : np.ndarray
        R peaks at 250 Hz.
    device : str
        Torch device.
    chunk : int
        Masks per forward pass.

    Returns
    -------
    dict[str, np.ndarray]
        ``z_<family>`` and ``abs_<family>`` arrays (12, 2500): the mean over the masks of that family
        that hide each sample, NaN where no mask of the family hides it.
    """
    length = signal.shape[1]
    padded = pad_length(length)
    masks, families = inference_masks(peaks, length)
    x = torch.from_numpy(np.pad(signal, ((0, 0), (0, padded - length)), mode="reflect")).to(device)
    full = torch.zeros((len(masks), 12, padded), dtype=torch.bool)
    full[:, :, :length] = torch.from_numpy(masks)
    sums = {
        f"{kind}_{family}": torch.zeros((12, length), device=device)
        for kind in ("z", "abs")
        for family in FAMILIES
    }
    counts = {family: torch.zeros((12, length), device=device) for family in FAMILIES}
    for start in range(0, len(masks), chunk):
        m = full[start : start + chunk].to(device)
        batch = x.expand(len(m), -1, -1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            mean, logvar = model(batch, m)
        err = (batch - mean.float()).abs()[:, :, :length]
        z = err * torch.exp(-0.5 * logvar.float()[:, :, :length])
        m = m[:, :, :length].float()
        chunk_families = torch.from_numpy(families[start : start + chunk]).to(device)
        for index, family in enumerate(FAMILIES):
            weight = m * (chunk_families == index).float()[:, None, None]
            sums[f"z_{family}"] += (z * weight).sum(0)
            sums[f"abs_{family}"] += (err * weight).sum(0)
            counts[family] += weight.sum(0)
    out = {}
    for key, total in sums.items():
        count = counts[key.split("_", 1)[1]]
        out[key] = torch.where(count > 0, total / count.clamp(min=1), torch.nan).cpu().numpy()
    return out


def finish_map(parts: list[np.ndarray]) -> np.ndarray:
    """
    Combine family maps by their NaN-aware mean, fill uncovered samples with the minimum, smooth 44 ms.

    Parameters
    ----------
    parts : list of np.ndarray
        Family maps (12, 2500), NaN where uncovered.

    Returns
    -------
    np.ndarray
        Finite float32 map (12, 5000) at 500 Hz.
    """
    stack = np.stack(parts)
    finite = np.isfinite(stack)
    count = finite.sum(axis=0)
    combined = np.where(finite, stack, 0.0).sum(axis=0) / np.maximum(count, 1)
    fill = combined[count > 0].min() if (count > 0).any() else 0.0
    combined = np.where(count > 0, combined, fill)
    return upsample(uniform_filter1d(combined, SMOOTH, axis=-1, mode="nearest"))


VARIANTS: dict[str, tuple[str, ...]] = {
    "primary": ("z_block", "z_phase", "z_whole"),
    "phase": ("z_phase",),
    "block": ("z_block",),
    "whole": ("z_whole",),
    "raw": ("abs_block", "abs_phase", "abs_whole"),
}


def variant_maps(errors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Compute the declared map variants of one record.

    Parameters
    ----------
    errors : dict[str, np.ndarray]
        Output of ``record_errors``.

    Returns
    -------
    dict[str, np.ndarray]
        Variant name to finite float32 map (12, 5000).
    """
    return {name: finish_map([errors[key] for key in keys]) for name, keys in VARIANTS.items()}
