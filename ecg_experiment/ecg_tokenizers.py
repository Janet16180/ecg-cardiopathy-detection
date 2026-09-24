"""Causal ECG beat boundaries and true variable-duration CPC chunks.

Beat events are confirmed after a fixed lookahead within the *past* of the
emission time. The detector never ranks an entire half, adapts a threshold
from future samples, or moves a boundary back to the R-peak timestamp.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import maximum_filter1d
from scipy.signal import butter, lfilter, sosfilt
from torch import nn

from ecg_experiment.cpc import CPCEncoder


SAMPLE_RATE = 250
HALF_SAMPLES = 1250
GRID_STEPS = 79
GRID_STRIDE = 16
PEAK_LEADS = (1, 7)  # II and V2 in canonical twelve-lead order.
CONFIRM_DELAY = 16  # 64 ms: no boundary is backdated to the candidate peak.
REFRACTORY_SAMPLES = 75  # 300 ms.
MIN_CHUNK = 4
MAX_CHUNK = 24
FIXED_CHUNK = 16
MAX_EVENTS = 24
_SOS = butter(2, (5, 18), btype="bandpass", fs=SAMPLE_RATE, output="sos")


def detect_confirmed_beats(half: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return R-like peak samples and their later causal confirmation samples.

    A second-order causal 5--18 Hz filter is applied to leads II and V2. The
    threshold at candidate time uses only the earlier running envelope. A
    candidate at sample i is accepted only after samples through i+16 are
    available; the next 300 ms are refractory to subsequent confirmations.
    """
    half = np.asarray(half, dtype=np.float32)
    if half.shape != (12, HALF_SAMPLES) or not np.isfinite(half).all():
        raise ValueError("Expected finite canonical 12 x 1250 half at 250 Hz")
    filtered = sosfilt(_SOS, half[list(PEAK_LEADS)].astype(np.float64), axis=1)
    envelope = np.max(np.abs(filtered), axis=0)
    # This EWMA uses current and earlier samples; the candidate threshold uses
    # its previous value, so the candidate cannot change its own baseline.
    baseline = lfilter([0.01], [1.0, -0.99], envelope)
    prior = np.concatenate(([0.0], baseline[:-1]))
    threshold = np.maximum(0.045, 2.0 * prior)
    local_max = maximum_filter1d(envelope, size=2 * CONFIRM_DELAY + 1,
                                 mode="constant", cval=-np.inf)
    candidates = np.flatnonzero((envelope == local_max) & (envelope >= threshold))
    peaks = []
    confirmations = []
    last_peak = -REFRACTORY_SAMPLES
    for peak in candidates:
        confirmation = int(peak) + CONFIRM_DELAY
        # The final CNN token can only see through raw sample 1248.
        if confirmation > (GRID_STEPS - 1) * GRID_STRIDE:
            break
        if int(peak) - last_peak < REFRACTORY_SAMPLES:
            continue
        peaks.append(int(peak))
        confirmations.append(confirmation)
        last_peak = int(peak)
    if len(peaks) > MAX_EVENTS:
        raise ValueError("Beat detector exceeded capacity despite refractory period")
    return np.asarray(peaks, dtype=np.int16), np.asarray(confirmations, dtype=np.int16)


def causal_beat_boundaries(confirmations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map confirmed events to the 79-token clock with online max-gap fallback.

    The returned forced mask marks max-24 and terminal emissions. A low count
    of detected beats never changes previous or future gates retroactively.
    """
    confirmations = np.asarray(confirmations)
    if confirmations.ndim != 1 or np.any(np.diff(confirmations) <= 0):
        raise ValueError("Confirmations must be strictly increasing")
    if np.any(confirmations < 0) or np.any(confirmations > 1248):
        raise ValueError("Confirmation sample outside this half")
    event_grid = np.zeros(GRID_STEPS, dtype=np.bool_)
    for sample in confirmations:
        event_grid[(int(sample) + GRID_STRIDE - 1) // GRID_STRIDE] = True
    boundaries = np.zeros(GRID_STEPS, dtype=np.bool_)
    forced = np.zeros(GRID_STEPS, dtype=np.bool_)
    elapsed = 0
    for step in range(GRID_STEPS):
        elapsed += 1
        terminal = step == GRID_STEPS - 1
        cap = elapsed >= MAX_CHUNK
        event = event_grid[step] and elapsed >= MIN_CHUNK
        if terminal or cap or event:
            boundaries[step] = True
            forced[step] = terminal or cap
            elapsed = 0
    return boundaries, forced


def beat_metadata(signal: np.ndarray):
    """Compute two independent half-grid beat boundary records from raw mV."""
    signal = np.asarray(signal, dtype=np.float32)
    if signal.shape != (12, 2500) or not np.isfinite(signal).all():
        raise ValueError("Expected finite canonical 12 x 2500 signal at 250 Hz")
    boundaries = np.zeros((2, GRID_STEPS), dtype=np.bool_)
    forced = np.zeros_like(boundaries)
    peaks = np.full((2, MAX_EVENTS), -1, dtype=np.int16)
    confirmations = np.full_like(peaks, -1)
    counts = np.zeros(2, dtype=np.uint8)
    for half_index in range(2):
        half = signal[:, half_index * HALF_SAMPLES:(half_index + 1) * HALF_SAMPLES]
        found, confirmed = detect_confirmed_beats(half)
        count = len(found)
        counts[half_index] = count
        peaks[half_index, :count] = found
        confirmations[half_index, :count] = confirmed
        boundaries[half_index], forced[half_index] = causal_beat_boundaries(confirmed)
    return boundaries, forced, peaks, confirmations, counts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_beat_metadata(path: str | Path, cache_ids: np.ndarray) -> np.ndarray:
    """Validate completed beat metadata and return row-aligned boundary mmap."""
    path = Path(path)
    info = json.loads((path / "complete.json").read_text())
    ids = np.load(path / "ecg_ids.npy", allow_pickle=False)
    if ids.shape != cache_ids.shape or not np.array_equal(ids, cache_ids):
        raise ValueError("Beat metadata ECG IDs differ from waveform cache")
    boundaries = np.load(path / "boundaries.npy", mmap_mode="r", allow_pickle=False)
    if boundaries.shape != (len(ids), 2, GRID_STEPS) or boundaries.dtype != np.bool_:
        raise ValueError("Beat boundary array has wrong shape or dtype")
    if info.get("record_count") != len(ids) or not np.all(boundaries[:, :, -1]):
        raise ValueError("Beat metadata is incomplete")
    for name in ("ecg_ids.npy", "boundaries.npy"):
        if _sha256(path / name) != info["sha256"][name]:
            raise ValueError(f"Beat metadata hash mismatch: {name}")
    return boundaries


class CausalChunkEncoder(nn.Module):
    """Keep CNN target tokens; update context only at emitted variable chunks."""

    VARIANTS = ("fixedchunk", "beatchunk", "learnedchunk")

    def __init__(self, variant: str):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown chunk variant: {variant}")
        self.variant = variant
        self.core = CPCEncoder()
        # Only CNN targets are reused directly. Native GRU weights are copied
        # into the cells by load_bootstrap; no unused GRU remains in the model.
        self.core.context = nn.Identity()
        self.cells = nn.ModuleList((nn.GRUCell(256, 256), nn.GRUCell(256, 256)))
        self.duration = nn.Linear(1, 256, bias=False)
        nn.init.zeros_(self.duration.weight)
        self.gate = nn.Linear(256, 1) if variant == "learnedchunk" else None
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
        self.rate_penalty = torch.zeros(())
        self.diagnostics: dict[str, float] = {}
        self.last_gates: torch.Tensor | None = None

    def load_bootstrap(self, state: dict) -> None:
        """Load native 004 encoder, copying its two GRU layers to the cells."""
        conv_state = {key.removeprefix("convs."): value for key, value in state.items()
                      if key.startswith("convs.")}
        self.core.convs.load_state_dict(conv_state)
        with torch.no_grad():
            for layer, cell in enumerate(self.cells):
                for field in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                    getattr(cell, field).copy_(state[f"context.{field}_l{layer}"])

    load_from_cpc_state_dict = load_bootstrap

    def chunk_context(self, tokens: torch.Tensor,
                      beat_boundaries: torch.Tensor | None = None) -> torch.Tensor:
        if tokens.ndim != 4 or tokens.shape[1:] != (2, GRID_STEPS, 256):
            raise ValueError("Expected CNN tokens [batch,2,79,256]")
        batch = len(tokens)
        flat = tokens.reshape(batch * 2, GRID_STEPS, 256)
        if self.variant == "beatchunk":
            if beat_boundaries is None or beat_boundaries.shape != (batch, 2, GRID_STEPS):
                raise ValueError("Beat boundaries [batch,2,79] required")
            boundaries = beat_boundaries.reshape(batch * 2, GRID_STEPS).to(
                device=tokens.device, dtype=torch.bool)
        else:
            boundaries = None
        state1 = flat.new_zeros((batch * 2, 256))
        state2 = flat.new_zeros((batch * 2, 256))
        accumulation = flat.new_zeros((batch * 2, 256))
        elapsed = flat.new_zeros((batch * 2, 1))
        previous_gate = flat.new_zeros((batch * 2, 1))
        contexts = []
        gates = []
        forced_count = flat.new_zeros((batch * 2, 1))
        durations = flat.new_zeros((batch * 2, 1))
        for step in range(GRID_STEPS):
            keep = 1 - previous_gate
            accumulation = keep * accumulation + flat[:, step]
            elapsed = keep * elapsed + 1
            summary = accumulation / elapsed
            duration = self.duration(torch.log(elapsed / FIXED_CHUNK))
            candidate1 = self.cells[0](summary + duration, state1)
            # Match the native two-layer GRU's dropout location. All chunk
            # arms evaluate candidates on every grid step, including holds.
            candidate2 = self.cells[1](nn.functional.dropout(
                candidate1, p=0.1, training=self.training), state2)
            if self.variant == "fixedchunk":
                gate = ((elapsed >= FIXED_CHUNK) | (step == GRID_STEPS - 1)).to(flat.dtype)
            elif self.variant == "beatchunk":
                gate = ((boundaries[:, step:step + 1] & (elapsed >= MIN_CHUNK)) |
                        (elapsed >= MAX_CHUNK) | (step == GRID_STEPS - 1)).to(flat.dtype)
            else:
                assert self.gate is not None
                probability = torch.sigmoid(self.gate(flat[:, step]) + (elapsed - FIXED_CHUNK) / 4)
                hard = (probability >= 0.5).to(flat.dtype)
                gate = hard + (probability - probability.detach())
                gate = torch.where(elapsed < MIN_CHUNK, torch.zeros_like(gate), gate)
                forced = (elapsed >= MAX_CHUNK) | (step == GRID_STEPS - 1)
                gate = torch.where(forced, torch.ones_like(gate), gate)
            forced_count = forced_count + (((elapsed >= MAX_CHUNK) |
                                           (step == GRID_STEPS - 1)).to(flat.dtype) * gate)
            durations = durations + elapsed * gate
            state1 = gate * candidate1 + (1 - gate) * state1
            state2 = gate * candidate2 + (1 - gate) * state2
            contexts.append(state2)
            gates.append(gate)
            previous_gate = gate
        gate_tensor = torch.stack(gates, dim=1).squeeze(-1)
        self.last_gates = gate_tensor.reshape(batch, 2, GRID_STEPS)
        gate_count = gate_tensor.sum(dim=1)
        if self.variant == "learnedchunk":
            self.rate_penalty = 0.1 * (((gate_count - 5) / 5) ** 2).mean()
        else:
            self.rate_penalty = flat.new_zeros(())
        self.diagnostics = {
            "chunk_count": float(gate_count.detach().mean()),
            "mean_chunk_tokens": float((durations / gate_count[:, None]).detach().mean()),
            "forced_ends": float(forced_count.detach().mean()),
        }
        return torch.stack(contexts, dim=1).reshape(batch, 2, GRID_STEPS, 256)

    def forward(self, signal: torch.Tensor,
                beat_boundaries: torch.Tensor | None = None):
        if signal.ndim != 3 or signal.shape[1:] != (12, 2500):
            raise ValueError("Expected ECG signal [batch,12,2500]")
        batch = len(signal)
        flat = signal.reshape(batch, 12, 2, HALF_SAMPLES).permute(0, 2, 1, 3)
        flat = flat.reshape(batch * 2, 12, HALF_SAMPLES)
        tokens = self.core.convs(flat).transpose(1, 2).reshape(batch, 2, GRID_STEPS, 256)
        return tokens, self.chunk_context(tokens, beat_boundaries)

    def pooled(self, contexts: torch.Tensor) -> torch.Tensor:
        """Average and max over actual emitted chunks, then over the halves."""
        gates = self.last_gates
        if gates is None or contexts.shape != (*gates.shape, 256):
            raise ValueError("Call forward before pooling matching chunk contexts")
        mean = (contexts * gates.unsqueeze(-1)).sum(dim=2) / gates.sum(dim=2).clamp_min(1).unsqueeze(-1)
        emitted = gates.detach() >= 0.5
        maximum = contexts.masked_fill(~emitted.unsqueeze(-1), -torch.inf).amax(dim=2)
        return torch.cat((mean, maximum), dim=-1).mean(dim=1)
