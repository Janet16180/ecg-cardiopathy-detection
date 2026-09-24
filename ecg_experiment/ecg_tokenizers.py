"""Causal ECG beat boundaries and true variable-duration CPC chunks.

Beat events are confirmed after a fixed lookahead within the *past* of the
emission time. The detector never ranks an entire half, adapts a threshold
from future samples, or moves a boundary back to the R-peak timestamp.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import maximum_filter1d
from scipy.signal import butter, lfilter, sosfilt
from torch import nn

from ecg_experiment.cpc import (
    HALF_SAMPLES,
    LEADS,
    SIGNAL_SAMPLES,
    TOKEN_COUNT,
    WIDTH,
    CPCEncoder,
    split_halves,
)
from ecg_experiment.files import sha256_file

SAMPLE_RATE = 250
GRID_STEPS = TOKEN_COUNT
GRID_STRIDE = 16
# The final CNN token can only see through raw sample 1248.
LAST_VISIBLE_SAMPLE = (GRID_STEPS - 1) * GRID_STRIDE
PEAK_LEADS = (1, 7)  # II and V2 in canonical twelve-lead order.
CONFIRM_DELAY = 16  # 64 ms: no boundary is backdated to the candidate peak.
REFRACTORY_SAMPLES = 75  # 300 ms.
MIN_CHUNK = 4
MAX_CHUNK = 24
FIXED_CHUNK = 16
MAX_EVENTS = 24
MIN_THRESHOLD = 0.045
TARGET_CHUNKS = 5
RATE_PENALTY_WEIGHT = 0.1
_SOS = butter(2, (5, 18), btype="bandpass", fs=SAMPLE_RATE, output="sos")

# Retained name: scripts.data.prepare_beat_tokens imports it.
_sha256 = sha256_file


def detect_confirmed_beats(half: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return R-like peak samples and their later causal confirmation samples.

    A second-order causal 5--18 Hz filter is applied to leads II and V2. The
    threshold at candidate time uses only the earlier running envelope. A
    candidate at sample i is accepted only after samples through i+16 are
    available; the next 300 ms are refractory to subsequent confirmations.

    Parameters
    ----------
    half : np.ndarray
        One finite canonical half of shape [12, 1250] at 250 Hz.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Accepted peak samples and confirmation samples, both int16.

    Raises
    ------
    ValueError
        If the half is malformed or more than ``MAX_EVENTS`` beats are found.
    """
    half = np.asarray(half, dtype=np.float32)
    if half.shape != (LEADS, HALF_SAMPLES) or not np.isfinite(half).all():
        raise ValueError("Expected finite canonical 12 x 1250 half at 250 Hz")
    filtered = sosfilt(_SOS, half[list(PEAK_LEADS)].astype(np.float64), axis=1)
    envelope = np.max(np.abs(filtered), axis=0)
    # This EWMA uses current and earlier samples; the candidate threshold uses
    # its previous value, so the candidate cannot change its own baseline.
    baseline = lfilter([0.01], [1.0, -0.99], envelope)
    prior = np.concatenate(([0.0], baseline[:-1]))
    threshold = np.maximum(MIN_THRESHOLD, 2.0 * prior)
    local_max = maximum_filter1d(envelope, size=2 * CONFIRM_DELAY + 1,
                                 mode="constant", cval=-np.inf)
    candidates = np.flatnonzero((envelope == local_max) & (envelope >= threshold))
    peaks = []
    confirmations = []
    last_peak = -REFRACTORY_SAMPLES
    for peak in candidates:
        confirmation = int(peak) + CONFIRM_DELAY
        if confirmation > LAST_VISIBLE_SAMPLE:
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
    """
    Map confirmed events to the 79-token clock with online max-gap fallback.

    The returned forced mask marks max-24 and terminal emissions. A low count
    of detected beats never changes previous or future gates retroactively.

    Parameters
    ----------
    confirmations : np.ndarray
        Strictly increasing confirmation samples within one half.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Boolean boundary and forced-emission masks, each of length 79.

    Raises
    ------
    ValueError
        If confirmations are not increasing or fall outside the visible half.
    """
    confirmations = np.asarray(confirmations)
    if confirmations.ndim != 1 or np.any(np.diff(confirmations) <= 0):
        raise ValueError("Confirmations must be strictly increasing")
    if np.any(confirmations < 0) or np.any(confirmations > LAST_VISIBLE_SAMPLE):
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


def beat_metadata(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute two independent half-grid beat boundary records from raw mV.

    Parameters
    ----------
    signal : np.ndarray
        One finite canonical record of shape [12, 2500] at 250 Hz.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Boundaries [2, 79], forced ends [2, 79], peaks [2, 24] and
        confirmations [2, 24] padded with -1, and beat counts [2].

    Raises
    ------
    ValueError
        If the signal is malformed.
    """
    signal = np.asarray(signal, dtype=np.float32)
    if signal.shape != (LEADS, SIGNAL_SAMPLES) or not np.isfinite(signal).all():
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


def load_beat_metadata(path: str | Path, cache_ids: np.ndarray) -> np.ndarray:
    """
    Validate completed beat metadata and return the row-aligned boundary mmap.

    Parameters
    ----------
    path : str | Path
        Beat metadata directory written by ``prepare_beat_tokens``.
    cache_ids : np.ndarray
        ECG IDs of the waveform cache, in row order.

    Returns
    -------
    np.ndarray
        Read-only memory-mapped boundaries of shape [records, 2, 79].

    Raises
    ------
    ValueError
        If IDs, shape, completeness or file hashes disagree.
    """
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
        if sha256_file(path / name) != info["sha256"][name]:
            raise ValueError(f"Beat metadata hash mismatch: {name}")
    return boundaries


def _forced_end(elapsed: torch.Tensor, step: int) -> torch.Tensor:
    """Chunks that must close now: maximum length reached or last grid step."""
    return (elapsed >= MAX_CHUNK) | (step == GRID_STEPS - 1)


def _fixed_gate(elapsed: torch.Tensor, step: int) -> torch.Tensor:
    """Close every sixteen tokens and at the final grid step."""
    return (elapsed >= FIXED_CHUNK) | (step == GRID_STEPS - 1)


def _beat_gate(boundary: torch.Tensor, elapsed: torch.Tensor, forced_end: torch.Tensor) -> torch.Tensor:
    """Close at a beat boundary after the minimum length, or when forced."""
    return (boundary & (elapsed >= MIN_CHUNK)) | forced_end


def _learned_gate(gate_layer: nn.Module, token: torch.Tensor, elapsed: torch.Tensor,
                  forced_end: torch.Tensor) -> torch.Tensor:
    """Straight-through hard router, blocked below the minimum and forced above the maximum."""
    probability = torch.sigmoid(gate_layer(token) + (elapsed - FIXED_CHUNK) / 4)
    hard = (probability >= 0.5).to(token.dtype)
    gate = hard + (probability - probability.detach())
    gate = torch.where(elapsed < MIN_CHUNK, torch.zeros_like(gate), gate)
    return torch.where(forced_end, torch.ones_like(gate), gate)


class CausalChunkEncoder(nn.Module):
    """Keep CNN target tokens; update context only at emitted variable chunks."""

    VARIANTS = ("fixedchunk", "beatchunk", "learnedchunk")

    def __init__(self, variant: str) -> None:
        """
        Build the chunk encoder.

        Parameters
        ----------
        variant : str
            One of ``VARIANTS``.

        Raises
        ------
        ValueError
            If the variant is unknown.
        """
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown chunk variant: {variant}")
        self.variant = variant
        self.core = CPCEncoder()
        # Only CNN targets are reused directly. Native GRU weights are copied
        # into the cells by load_from_cpc_state_dict; no unused GRU remains in the model.
        self.core.context = nn.Identity()
        self.cells = nn.ModuleList((nn.GRUCell(WIDTH, WIDTH), nn.GRUCell(WIDTH, WIDTH)))
        self.duration = nn.Linear(1, WIDTH, bias=False)
        nn.init.zeros_(self.duration.weight)
        self.gate = nn.Linear(WIDTH, 1) if variant == "learnedchunk" else None
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
        self.rate_penalty = torch.zeros(())
        self.diagnostics: dict[str, float] = {}
        self.last_gates: torch.Tensor | None = None

    def load_from_cpc_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """
        Load a native 004 encoder, copying its two GRU layers to the cells.

        Parameters
        ----------
        state : dict[str, torch.Tensor]
            ``CPCEncoder`` state dict.
        """
        conv_state = {key.removeprefix("convs."): value for key, value in state.items()
                      if key.startswith("convs.")}
        self.core.convs.load_state_dict(conv_state)
        with torch.no_grad():
            for layer, cell in enumerate(self.cells):
                for field in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                    getattr(cell, field).copy_(state[f"context.{field}_l{layer}"])

    def _flat_boundaries(self, beat_boundaries: torch.Tensor | None, batch: int,
                         device: torch.device) -> torch.Tensor | None:
        """Validate beat boundaries for the beat arm and flatten them per half."""
        if self.variant != "beatchunk":
            return None
        if beat_boundaries is None or beat_boundaries.shape != (batch, 2, GRID_STEPS):
            raise ValueError("Beat boundaries [batch,2,79] required")
        return beat_boundaries.reshape(batch * 2, GRID_STEPS).to(device=device, dtype=torch.bool)

    def _step_gate(self, step: int, token: torch.Tensor, elapsed: torch.Tensor,
                   boundaries: torch.Tensor | None, forced_end: torch.Tensor) -> torch.Tensor:
        """Emission gate of the configured variant at one grid step."""
        if self.variant == "fixedchunk":
            gate = _fixed_gate(elapsed, step).to(token.dtype)
        elif self.variant == "beatchunk":
            gate = _beat_gate(boundaries[:, step:step + 1], elapsed, forced_end).to(token.dtype)
        else:
            if self.gate is None:
                raise RuntimeError("The learned chunk encoder lacks its gate layer")
            gate = _learned_gate(self.gate, token, elapsed, forced_end)
        return gate

    def _summarize_gates(self, gates: list[torch.Tensor], durations: torch.Tensor,
                         forced_count: torch.Tensor, batch: int) -> None:
        """Store emitted gates, the learned rate penalty and chunk diagnostics."""
        gate_tensor = torch.stack(gates, dim=1).squeeze(-1)
        self.last_gates = gate_tensor.reshape(batch, 2, GRID_STEPS)
        gate_count = gate_tensor.sum(dim=1)
        if self.variant == "learnedchunk":
            relative_error = (gate_count - TARGET_CHUNKS) / TARGET_CHUNKS
            self.rate_penalty = RATE_PENALTY_WEIGHT * (relative_error ** 2).mean()
        else:
            self.rate_penalty = durations.new_zeros(())
        self.diagnostics = {
            "chunk_count": float(gate_count.detach().mean()),
            "mean_chunk_tokens": float((durations / gate_count[:, None]).detach().mean()),
            "forced_ends": float(forced_count.detach().mean()),
        }

    def chunk_context(self, tokens: torch.Tensor,
                      beat_boundaries: torch.Tensor | None = None) -> torch.Tensor:
        """
        Run the two GRU cells over running chunk means, committing state only at emissions.

        Parameters
        ----------
        tokens : torch.Tensor
            CNN tokens of shape [batch, 2, 79, 256].
        beat_boundaries : torch.Tensor | None
            Causal beat boundaries [batch, 2, 79]; required for ``beatchunk``.

        Returns
        -------
        torch.Tensor
            Held contexts of shape [batch, 2, 79, 256].

        Raises
        ------
        ValueError
            If tokens or required beat boundaries are malformed.
        """
        if tokens.ndim != 4 or tokens.shape[1:] != (2, GRID_STEPS, WIDTH):
            raise ValueError("Expected CNN tokens [batch,2,79,256]")
        batch = len(tokens)
        flat = tokens.reshape(batch * 2, GRID_STEPS, WIDTH)
        boundaries = self._flat_boundaries(beat_boundaries, batch, tokens.device)
        state1 = flat.new_zeros((batch * 2, WIDTH))
        state2 = flat.new_zeros((batch * 2, WIDTH))
        accumulation = flat.new_zeros((batch * 2, WIDTH))
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
            forced_end = _forced_end(elapsed, step)
            gate = self._step_gate(step, flat[:, step], elapsed, boundaries, forced_end)
            forced_count = forced_count + forced_end.to(flat.dtype) * gate
            durations = durations + elapsed * gate
            state1 = gate * candidate1 + (1 - gate) * state1
            state2 = gate * candidate2 + (1 - gate) * state2
            contexts.append(state2)
            gates.append(gate)
            previous_gate = gate
        self._summarize_gates(gates, durations, forced_count, batch)
        return torch.stack(contexts, dim=1).reshape(batch, 2, GRID_STEPS, WIDTH)

    def forward(self, signal: torch.Tensor,
                beat_boundaries: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode CNN tokens and chunked contexts for both halves.

        Parameters
        ----------
        signal : torch.Tensor
            Normalized signals of shape [batch, 12, 2500].
        beat_boundaries : torch.Tensor | None
            Causal beat boundaries [batch, 2, 79]; required for ``beatchunk``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tokens and contexts, each of shape [batch, 2, 79, 256].

        Raises
        ------
        ValueError
            If the signal shape is wrong.
        """
        if signal.ndim != 3 or signal.shape[1:] != (LEADS, SIGNAL_SAMPLES):
            raise ValueError("Expected ECG signal [batch,12,2500]")
        batch = len(signal)
        tokens = self.core.convs(split_halves(signal)).transpose(1, 2).reshape(batch, 2, GRID_STEPS, WIDTH)
        return tokens, self.chunk_context(tokens, beat_boundaries)

    def pooled(self, contexts: torch.Tensor) -> torch.Tensor:
        """
        Average and max over actual emitted chunks, then over the halves.

        Parameters
        ----------
        contexts : torch.Tensor
            Contexts from the latest ``chunk_context`` call, [batch, 2, 79, 256].

        Returns
        -------
        torch.Tensor
            Pooled features of shape [batch, 512].

        Raises
        ------
        ValueError
            If no gates are stored or their shape does not match ``contexts``.
        """
        gates = self.last_gates
        if gates is None or contexts.shape != (*gates.shape, WIDTH):
            raise ValueError("Call forward before pooling matching chunk contexts")
        mean = (contexts * gates.unsqueeze(-1)).sum(dim=2) / gates.sum(dim=2).clamp_min(1).unsqueeze(-1)
        emitted = gates.detach() >= 0.5
        maximum = contexts.masked_fill(~emitted.unsqueeze(-1), -torch.inf).amax(dim=2)
        return torch.cat((mean, maximum), dim=-1).mean(dim=1)
