"""Forward narrowed-domain ST interventions retaining frozen v1 backgrounds and profiles."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from ecg_experiment.resample import resample_full
from ecg_experiment.ssl_synthetic_generator import (
    RAW_RATE,
    SECONDS,
    SUPPORT_FLOOR_MV,
    InterventionPair,
    _domains,
    couple_limb_leads,
    edit_profile,
)
from ecg_experiment.ssl_synthetic_generator import generate_pair as generate_pair_v1


def _narrow_parameters(params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    narrowed = deepcopy(params)
    fraction = (params["width_seconds"] - 0.14) / 0.08
    narrowed["original_width_seconds"] = params["width_seconds"]
    narrowed["width_seconds"] = float(0.14 + 0.05 * fraction)
    coefficients = np.array(params["primitive_coefficients"], dtype=np.float64)
    if coefficients[:2].any():
        choices = np.flatnonzero(coefficients[2:]) + 2
        rng = np.random.default_rng(np.random.SeedSequence([params["seed"], 404]))
        chosen = int(rng.choice(choices))
        coefficients[2:] = 0
        coefficients[chosen] = 1
    narrowed["primitive_coefficients"] = coefficients.tolist()
    narrowed["domain_version"] = "v2"
    return coefficients * params["amplitude_mv"], narrowed


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
    """Render the predeclared v2 intervention on exactly the same v1 background.

    Parameters
    ----------
    seed : int
        Nonnegative latent identity; all related realizations retain the same data role.
    profile : str
        A for learning; B held out from training and model selection.
    mode : str
        Focal or persistent, with the identical narrowed parameter rule.
    sample_rate : int
        250 or 500 Hz observations, ten seconds, float32 mV.
    target, nuisance : bool
        Intervention and shared nuisance switches, independent of the parameter narrowing.
    edit_seed, nuisance_seed : int or None
        Independent deterministic substream identities, defaulting to seed.

    Returns
    -------
    InterventionPair
        V1-compatible observations and exact actual postprocessing support. Only edited is input.
    """
    original = generate_pair_v1(
        seed,
        profile=profile,
        mode=mode,
        sample_rate=sample_rate,
        target=target,
        nuisance=nuisance,
        edit_seed=edit_seed,
        nuisance_seed=nuisance_seed,
    )
    metadata = deepcopy(original.metadata)
    coefficients, params = _narrow_parameters(metadata["intervention"])
    metadata["intervention"] = params
    metadata["generator_version"] = "v2"
    time = np.arange(RAW_RATE * SECONDS) / RAW_RATE
    beats = metadata["latent"]["beat_centers_seconds"]
    envelope = np.zeros(time.size)
    for index in params["selected_beats"]:
        envelope += edit_profile((time - beats[index] - 0.10) / params["width_seconds"], profile)
    delta = couple_limb_leads(coefficients[:, None] * envelope * metadata["nuisance"]["gain"] * target)
    raw_edited = (original.raw_baseline.astype(np.float64) + delta).astype(np.float32)
    edited = raw_edited.copy() if sample_rate == 500 else resample_full(raw_edited, 1, 2)
    difference = edited.astype(np.float64) - original.baseline.astype(np.float64)
    candidate, nominal = _domains(beats, params, coefficients, sample_rate)
    return InterventionPair(
        original.baseline,
        edited,
        np.abs(difference) > SUPPORT_FLOOR_MV,
        candidate,
        metadata,
        original.raw_baseline,
        raw_edited,
        nominal & target,
    )
