"""Independent paired simulation for engineering intervention localization, in mV."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ecg_experiment.resample import resample_full
from ecg_experiment.waveforms import LEADS

RAW_RATE = 500
SECONDS = 10
SUPPORT_FLOOR_MV = 0.005


@dataclass(frozen=True)
class InterventionPair:
    """Paired observations and engineering truth; only ``edited`` is model input."""

    baseline: np.ndarray
    edited: np.ndarray
    support: np.ndarray
    candidate: np.ndarray
    metadata: dict[str, Any]
    raw_baseline: np.ndarray
    raw_edited: np.ndarray
    nominal: np.ndarray

    @property
    def difference(self) -> np.ndarray:
        """Return the exact final observation difference in float64 mV."""
        return self.edited.astype(np.float64) - self.baseline.astype(np.float64)


def _rng(seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, stream]))


def couple_limb_leads(independent: np.ndarray) -> np.ndarray:
    """Expand I, II, V1–V6 to canonical twelve-lead coordinates.

    Parameters
    ----------
    independent : np.ndarray
        Shape (8, samples), in mV, with I and II first.

    Returns
    -------
    np.ndarray
        Shape (12, samples), retaining all algebraically coupled changes.
    """
    if independent.ndim != 2 or independent.shape[0] != 8:
        raise ValueError("Expected I, II, V1–V6 with shape (8, samples)")
    first, second = independent[:2]
    return np.vstack(
        (
            first,
            second,
            second - first,
            -(first + second) / 2,
            first - second / 2,
            second - first / 2,
            independent[2:],
        )
    )


def edit_profile(time_fraction: np.ndarray, profile: str) -> np.ndarray:
    """Evaluate one of two independent compact zero-slope edit mechanisms.

    Parameters
    ----------
    time_fraction : np.ndarray
        Elapsed fraction of the nominal edit window.
    profile : str
        A is a cosine-ramp plateau; B is a quartic beta pulse, held out from learning.

    Returns
    -------
    np.ndarray
        Unit-peak envelope, zero outside the open unit interval.
    """
    inside = (time_fraction > 0) & (time_fraction < 1)
    result = np.zeros_like(time_fraction, dtype=np.float64)
    u = time_fraction[inside]
    if profile == "A":
        ramp = np.minimum(np.minimum(u, 1 - u) / 0.25, 1)
        result[inside] = (1 - np.cos(np.pi * ramp)) / 2
    elif profile == "B":
        result[inside] = 16 * u**2 * (1 - u) ** 2
    else:
        raise ValueError("Profile must be A or held-out B")
    return result


def _background(seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    rng = _rng(seed, 101)
    time = np.arange(RAW_RATE * SECONDS) / RAW_RATE
    rate = float(rng.uniform(45, 110))
    beats = np.arange(float(rng.uniform(0.4, 1.0)), 9.5, 60 / rate)
    scales = rng.uniform(0.45, 1.4, size=8)
    polarities = np.array([1, 1, -0.5, -0.1, 0.5, 1.0, 1.1, 0.9])
    morph = rng.uniform(0.8, 1.2, size=(8, 5))
    widths = rng.uniform(0.85, 1.15, size=5) * np.array([0.035, 0.009, 0.012, 0.014, 0.060])
    offsets = np.array([-0.18, -0.026, 0, 0.030, 0.31])
    amplitude = np.array([0.12, -0.12, 1.0, -0.25, 0.27])
    clean = np.zeros((8, time.size))
    for beat in beats:
        components = np.exp(-0.5 * ((time[None, :] - beat - offsets[:, None]) / widths[:, None]) ** 2)
        clean += (morph * amplitude) @ components * (scales * polarities)[:, None]
    latent = {
        "seed": seed,
        "rate_bpm": rate,
        "beat_centers_seconds": beats.tolist(),
        "scales": scales.tolist(),
        "morphology": morph.tolist(),
        "widths": widths.tolist(),
        "beat_landmarks": [{"r": float(r), "j": float(r + 0.10), "st_end": float(r + 0.32)} for r in beats],
        "permissible_leads": list(LEADS),
    }
    return clean, latent


def _nuisance(seed: int, enabled: bool) -> tuple[np.ndarray, dict[str, Any]]:
    rng = _rng(seed, 303)
    gain = float(rng.uniform(0.8, 1.2)) if enabled else 1.0
    noise = float(rng.uniform(0, 0.025)) if enabled else 0.0
    drift = float(rng.uniform(0, 0.12)) if enabled else 0.0
    frequency = float(rng.uniform(0.05, 0.40))
    phases = rng.uniform(0, 2 * np.pi, size=(8, 1))
    time = np.arange(RAW_RATE * SECONDS) / RAW_RATE
    values = rng.normal(0, noise, size=(8, time.size))
    values += drift * np.sin(2 * np.pi * frequency * time[None, :] + phases)
    return values, {
        "seed": seed,
        "enabled": enabled,
        "gain": gain,
        "noise_sd_mv": noise,
        "drift_amplitude_mv": drift,
        "drift_hz": frequency,
        "phases": phases[:, 0].tolist(),
    }


def _intervention(seed: int, beats: list[float], mode: str) -> tuple[np.ndarray, dict[str, Any]]:
    rng = _rng(seed, 202)
    coefficients = np.zeros(8)
    limb_pattern = int(rng.integers(0, 4))
    coefficients[:2] = [(0, 0), (1, 0), (0, 1), (1, 1)][limb_pattern]
    chest = rng.choice(np.arange(2, 8), size=int(rng.integers(1, 4)), replace=False)
    coefficients[chest] = 1
    amplitude = float(rng.uniform(0.04, 0.30) * rng.choice([-1, 1]))
    width = float(rng.uniform(0.14, 0.22))
    chosen = [int(rng.integers(len(beats)))] if mode == "focal" else list(range(len(beats)))
    params = {
        "seed": seed,
        "mode": mode,
        "amplitude_mv": amplitude,
        "width_seconds": width,
        "primitive_coefficients": coefficients.tolist(),
        "selected_beats": chosen,
    }
    return coefficients * amplitude, params


def _domains(
    beats: list[float], params: dict[str, Any], coefficients: np.ndarray, sample_rate: int
) -> tuple[np.ndarray, np.ndarray]:
    time = np.arange(sample_rate * SECONDS) / sample_rate
    candidate_time = np.zeros(time.size, dtype=bool)
    nominal_time = np.zeros(time.size, dtype=bool)
    for index, beat in enumerate(beats):
        candidate_time |= (time >= beat + 0.06) & (time <= beat + 0.39)
        if index in params["selected_beats"]:
            nominal_time |= (time > beat + 0.10) & (time < beat + 0.10 + params["width_seconds"])
    changed_leads = couple_limb_leads(coefficients[:, None])[:, 0] != 0
    return np.broadcast_to(candidate_time, (12, time.size)).copy(), changed_leads[:, None] & nominal_time


def generate_pair(
    seed: int,
    profile: str = "A",
    mode: str = "focal",
    sample_rate: int = 250,
    target: bool = True,
    nuisance: bool = True,
    edit_seed: int | None = None,
    nuisance_seed: int | None = None,
) -> InterventionPair:
    """Render a deterministic paired engineering intervention on a simulated ECG.

    Parameters
    ----------
    seed : int
        Nonnegative latent-background identity. Related renders must share their data role.
    profile : str
        A for training/validation; B is held-out engineering evaluation only.
    mode : str
        Focal changes one beat; persistent changes every eligible beat.
    sample_rate : int
        Either 500 raw Hz or 250 whole-record polyphase Hz.
    target, nuisance : bool
        Enable intervention and independently shared noise/drift/gain, respectively.
    edit_seed, nuisance_seed : int or None
        Optional independent substream identities; default to the latent identity.

    Returns
    -------
    InterventionPair
        Lead-major float32 observations in mV and actual postprocessing support. Baseline,
        support, metadata and candidates are label/audit information, never inference inputs.
    """
    if sample_rate not in (250, 500) or mode not in ("focal", "persistent"):
        raise ValueError("Expected 250/500 Hz and focal/persistent mode")
    if (
        min(seed, seed if edit_seed is None else edit_seed, seed if nuisance_seed is None else nuisance_seed)
        < 0
    ):
        raise ValueError("Seed identities must be nonnegative")
    clean, latent = _background(seed)
    disturbance, noise_params = _nuisance(seed if nuisance_seed is None else nuisance_seed, nuisance)
    coefficients, params = _intervention(
        seed if edit_seed is None else edit_seed, latent["beat_centers_seconds"], mode
    )
    time = np.arange(RAW_RATE * SECONDS) / RAW_RATE
    envelope = np.zeros(time.size)
    for index in params["selected_beats"]:
        start = latent["beat_centers_seconds"][index] + 0.10
        envelope += edit_profile((time - start) / params["width_seconds"], profile)
    delta = coefficients[:, None] * envelope * float(target)
    primitive = clean * noise_params["gain"] + disturbance
    raw_baseline = couple_limb_leads(primitive).astype(np.float32)
    raw_edited = couple_limb_leads(primitive + delta * noise_params["gain"]).astype(np.float32)
    baseline = raw_baseline.copy() if sample_rate == 500 else resample_full(raw_baseline, 1, 2)
    edited = raw_edited.copy() if sample_rate == 500 else resample_full(raw_edited, 1, 2)
    difference = edited.astype(np.float64) - baseline.astype(np.float64)
    candidate, nominal = _domains(latent["beat_centers_seconds"], params, coefficients, sample_rate)
    metadata = {
        "latent": latent,
        "intervention": params,
        "nuisance": noise_params,
        "profile": profile,
        "target": target,
        "sample_rate": sample_rate,
        "units": "mV",
        "support_floor_mv": SUPPORT_FLOOR_MV,
        "candidate_rule": "all leads, every beat R+[0.06,0.39] seconds",
        "st_prior_rule": "all leads, every beat R+[0.10,0.32] seconds; independent of target",
    }
    return InterventionPair(
        baseline,
        edited,
        np.abs(difference) > SUPPORT_FLOOR_MV,
        candidate,
        metadata,
        raw_baseline,
        raw_edited,
        nominal & target,
    )
