"""Published-equation ECGSYN and explicit engineering projections for Experiment 058.

The ECGSYN equations are independently implemented from McSharry et al. (2003).
Official reference sources remain local under their original license. The numerical
integrator and multilead projection here are separate engineering choices, not an
anatomical heart/torso model or a validated disease simulator.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import lfilter, resample_poly
from scipy.spatial.transform import Rotation

FS = 500
SEED = 58058
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
KINDS = ("qrs", "st", "t", "persistent_st", "persistent_t", "sham", "drift", "gain", "noise")
ANGLES = np.deg2rad([-70.0, -15.0, 0.0, 15.0, 100.0])
AMPLITUDES = np.array([1.2, -5.0, 30.0, -7.5, 0.75])
WIDTHS = np.array([0.25, 0.1, 0.1, 0.1, 0.4])
VECTORS = np.array(
    [[0.35, 1.0, 0.05], [0.25, 1.0, -0.15], [0.3, 1.0, 0.15], [0.1, 1.0, 0.4], [0.45, 1.0, 0.1]]
)
ELECTRODES = np.array(
    [
        [-0.6, -0.45, 0.15],
        [0.6, -0.45, 0.15],
        [0.0, 0.85, 0.1],
        [-0.25, 0.05, 0.6],
        [-0.1, 0.05, 0.63],
        [0.05, 0.1, 0.64],
        [0.2, 0.18, 0.6],
        [0.4, 0.2, 0.5],
        [0.55, 0.2, 0.35],
    ]
)


def limb_projection(electrode_potentials: np.ndarray) -> np.ndarray:
    """Derive standard lead voltages from nine virtual electrode potentials.

    Parameters
    ----------
    electrode_potentials : np.ndarray
        RA, LA, LL and V1 through V6 electrode rows by samples or latent dimensions.

    Returns
    -------
    np.ndarray
        Twelve leads satisfying the exact limb-lead algebra.
    """
    ra, la, ll = electrode_potentials[:3]
    wilson = (ra + la + ll) / 3
    return np.vstack(
        (
            la - ra,
            ll - ra,
            ll - la,
            ra - (la + ll) / 2,
            la - (ra + ll) / 2,
            ll - (ra + la) / 2,
            electrode_potentials[3:] - wilson,
        )
    )


def rr_process(rng: np.random.Generator, heart_rate: float, deviation: float) -> np.ndarray:
    """Generate ECGSYN's two-band RR spectrum with independently seeded phases.

    Parameters
    ----------
    rng : np.random.Generator
        Subject-specific random state.
    heart_rate : float
        Mean heart rate in beats per minute.
    deviation : float
        Heart-rate standard deviation in beats per minute.

    Returns
    -------
    np.ndarray
        One-hertz RR values in seconds, length 128.
    """
    frequency = np.fft.rfftfreq(128, d=1.0)
    density = 0.5 * np.exp(-0.5 * ((frequency - 0.1) / 0.01) ** 2) + np.exp(
        -0.5 * ((frequency - 0.25) / 0.01) ** 2
    )
    phase = rng.uniform(0, 2 * np.pi, len(frequency))
    phase[[0, -1]] = 0
    fluctuation = np.fft.irfft(np.sqrt(density) * np.exp(1j * phase), n=128)
    return 60 / heart_rate + fluctuation / fluctuation.std() * 60 * deviation / heart_rate**2


def source_components(
    rng: np.random.Generator, parameters: dict[str, Any], family: str
) -> dict[str, np.ndarray]:
    """Render clean source components with no localization method or intervention.

    Parameters
    ----------
    rng : np.random.Generator
        Subject-specific state shared by both paired counterfactual traces.
    parameters : dict
        Heart rate, heart-rate deviation and width/amplitude multipliers.
    family : str
        ``ecgsyn`` uses published ODE equations; ``compact`` is an engineering control.

    Returns
    -------
    dict
        Five component rows, original-time latent R anchors and instantaneous phase.
    """
    rate = float(parameters["heart_rate"])
    rr = rr_process(rng, rate, float(parameters["hr_std"]))
    interpolated = resample_poly(rr, FS, 1)
    samples = 16 * FS
    beat_rr = np.empty(samples)
    position = 0
    while position < samples:
        interval = float(interpolated[position])
        stop = min(samples, position + round(interval * FS))
        beat_rr[position:stop] = interval
        position = stop
    phase = np.cumsum(2 * np.pi / (FS * beat_rr))
    wrapped = np.arctan2(np.sin(phase), np.cos(phase))
    factor = np.sqrt(rate / 60)
    angles = ANGLES * np.array([np.sqrt(factor), factor, 1, factor, np.sqrt(factor)])
    widths = WIDTHS * factor * float(parameters["width_scale"])
    delta = np.fmod(wrapped[None] - angles[:, None], 2 * np.pi)
    amplitudes = AMPLITUDES * np.asarray(parameters["amplitude_scale"])
    if family == "ecgsyn":
        forcing = -amplitudes[:, None] * delta * np.exp(-0.5 * (delta / widths[:, None]) ** 2)
        decay = np.exp(-1 / FS)
        average = (forcing + np.column_stack((forcing[:, :1], forcing[:, :-1]))) / 2
        components = lfilter([1 - decay], [1, -decay], average, axis=1)
        respiratory = 0.005 * np.sin(2 * np.pi * 0.25 * np.arange(samples) / FS)
        background = lfilter(
            [1 - decay], [1, -decay], (respiratory + np.r_[respiratory[0], respiratory[:-1]]) / 2
        )
        background += 0.04 * decay ** np.arange(samples)
    elif family == "compact":
        widths = widths[:, None]
        normalized = delta / (3 * widths)
        taper = np.where(np.abs(normalized) < 1, 0.5 * (1 + np.cos(np.pi * normalized)), 0)
        components = (
            np.array([0.12, -0.13, 1, -0.2, 0.24])[:, None] * np.exp(-0.5 * (delta / widths) ** 2) * taper
        )
        components *= np.asarray(parameters["amplitude_scale"])[:, None]
        background = np.zeros(samples)
    else:
        raise ValueError(family)
    burn = 4 * FS
    anchors = np.flatnonzero(np.diff(np.floor(phase / (2 * np.pi)), prepend=0) > 0) - burn
    anchors = anchors[(anchors > 0) & (anchors < 12 * FS)]
    return {
        "components": components[:, burn:],
        "background": background[burn:],
        "anchors": anchors,
        "phase": phase[burn:],
    }


def subject_parameters(subject: int, cohort: str) -> dict[str, Any]:
    """Draw fixed subject parameters without inspecting localization outcomes.

    Parameters
    ----------
    subject : int
        Globally unique simulator subject identifier.
    cohort : str
        ``calibration``, ``iid`` or ``shifted``.

    Returns
    -------
    dict
        JSON-serializable seed, geometry, timing and morphology parameters.
    """
    rng = np.random.default_rng(np.random.SeedSequence([SEED, subject, 1]))
    shifted = cohort == "shifted"
    if cohort not in ("calibration", "iid", "shifted"):
        raise ValueError(cohort)
    heart_rate = float(rng.uniform(55, 85))
    width = float(rng.uniform(0.9, 1.1))
    rotation = rng.uniform(-20, 20, 3)
    if shifted:
        heart_rate = float(rng.uniform(45, 55) if rng.integers(2) else rng.uniform(85, 105))
        width = float(rng.uniform(0.8, 0.9) if rng.integers(2) else rng.uniform(1.1, 1.2))
        rotation = rng.choice([-1, 1], 3) * rng.uniform(20, 35, 3)
    return {
        "subject": subject,
        "cohort": cohort,
        "heart_rate": heart_rate,
        "hr_std": float(rng.uniform(1, 3)),
        "width_scale": width,
        "rotation_degrees": rotation.tolist(),
        "amplitude_scale": rng.uniform(0.85, 1.15, 5).tolist(),
        "electrode_jitter": rng.uniform(-0.025, 0.025, (9, 3)).tolist(),
    }


def generate_subject(subject: int, cohort: str, family: str) -> dict[str, Any]:
    """Generate a paired-ready twelve-lead engineering ECG and its provenance.

    Parameters
    ----------
    subject : int
        Unique simulator subject, shared between generator families.
    cohort : str
        Prospective parameter-distribution family.
    family : str
        Published-equation ``ecgsyn`` or distinct ``compact`` control.

    Returns
    -------
    dict
        Clean and noisy waveforms, source components, lead projection, scale and parameters.
    """
    parameters = subject_parameters(subject, cohort)
    rng = np.random.default_rng(np.random.SeedSequence([SEED, subject, 2]))
    source = source_components(rng, parameters, family)
    electrodes = ELECTRODES + np.asarray(parameters["electrode_jitter"])
    electrode_vectors = electrodes / (np.linalg.norm(electrodes, axis=1) ** 3)[:, None]
    rotation = Rotation.from_euler("xyz", parameters["rotation_degrees"], degrees=True).as_matrix()
    projection = limb_projection(electrode_vectors) @ rotation
    vector_sources = (
        VECTORS.T @ source["components"] + VECTORS.mean(axis=0)[:, None] * source["background"][None]
    )
    unscaled = projection @ vector_sources
    scale = 1.2 / np.ptp(unscaled[1])
    clean = unscaled * scale
    noise_rng = np.random.default_rng(np.random.SeedSequence([SEED, subject, 3]))
    noise = limb_projection(noise_rng.normal(0, 0.003, (9, clean.shape[1])))
    return {
        **source,
        "parameters": parameters,
        "projection": projection,
        "scale": scale,
        "clean": clean,
        "signal": clean + noise,
        "noise": noise,
        "family": family,
    }


def _pulse(samples: np.ndarray, left: int, right: int) -> np.ndarray:
    """Return a compact raised-cosine pulse on a specified sample interval.

    Parameters
    ----------
    samples : np.ndarray
        Original sample coordinates.
    left, right : int
        Finite support boundaries.

    Returns
    -------
    np.ndarray
        Zero outside the interval, unit height at its midpoint.
    """
    phase = (samples - left) / (right - left)
    return np.where((phase > 0) & (phase < 1), 0.5 * (1 - np.cos(2 * np.pi * phase)), 0)


def intervene(subject: dict[str, Any], kind: str) -> dict[str, Any]:
    """Apply one frozen latent intervention or paired negative control.

    Parameters
    ----------
    subject : dict
        One generator output. Its clean-derived scale is reused unchanged.
    kind : str
        Frozen intervention or nuisance name from ``KINDS``.

    Returns
    -------
    dict
        Altered inference signal, noise-free difference energy, observable mask and provenance.
        These targets are evaluation-only and never enter a localization method.
    """
    if kind not in KINDS:
        raise ValueError(kind)
    samples = np.arange(subject["clean"].shape[1])
    anchors = subject["anchors"]
    complete = anchors[(anchors >= FS) & (anchors < 11 * FS)]
    if len(complete) < 3:
        raise ValueError("Generator has fewer than three interior latent beats")
    rng = np.random.default_rng(np.random.SeedSequence([SEED, subject["parameters"]["subject"], 4]))
    selected = int(complete[rng.integers(len(complete))])
    vectors = VECTORS.T @ subject["components"] + VECTORS.mean(axis=0)[:, None] * subject["background"][None]
    changed = vectors.copy()
    pulse = _pulse(samples, selected + round(0.1 * FS), selected + round(0.24 * FS))
    if kind in ("st", "persistent_st"):
        if kind == "persistent_st":
            pulse = sum(
                (_pulse(samples, int(anchor) + 50, int(anchor) + 120) for anchor in anchors),
                start=np.zeros(len(samples)),
            )
        direction = np.array([0.3, 1, 0.25])
        amplitude = 0.15 / (subject["scale"] * np.max(np.abs(subject["projection"] @ direction)))
        changed += amplitude * direction[:, None] * pulse[None]
    elif kind in ("t", "persistent_t"):
        envelope = _pulse(samples, selected + round(0.08 * FS), selected + round(0.48 * FS))
        if kind == "persistent_t":
            envelope = np.ones(len(samples))
        changed -= 2 * VECTORS[4, :, None] * subject["components"][4, None] * envelope[None]
    elif kind == "qrs":
        envelope = _pulse(samples, selected - round(0.12 * FS), selected + round(0.12 * FS))
        query = selected + (samples - selected) / 1.5
        original = subject["components"][1:4]
        stretched = np.stack([np.interp(query, samples, row) for row in original])
        changed += VECTORS[1:4].T @ ((stretched - original) * envelope[None])
    noiseless = (subject["projection"] @ changed) * subject["scale"]
    signal = noiseless + subject["noise"]
    signal = nuisance_signal(subject, kind, signal)
    difference = noiseless - subject["clean"]
    mask = np.abs(difference) >= 0.01
    affected = mask.sum(axis=1) >= round(0.02 * FS)
    mask &= affected[:, None]
    return {
        "signal": signal,
        "noiseless": noiseless,
        "difference": difference,
        "energy": difference**2,
        "mask": mask,
        "affected_leads": affected,
        "kind": kind,
        "selected_latent_sample": selected,
        "informative": bool(mask.any()),
    }


def nuisance_signal(subject: dict[str, Any], kind: str, signal: np.ndarray) -> np.ndarray:
    """Apply a fixed negative-control corruption without changing causal targets.

    Parameters
    ----------
    subject : dict
        Subject provenance used for deterministic noise.
    kind : str
        Frozen control name; disease-component interventions pass through unchanged.
    signal : np.ndarray
        Physical lead signal to corrupt.

    Returns
    -------
    np.ndarray
        Corrupted trace with exact limb-lead algebra retained.
    """
    samples = np.arange(signal.shape[1])
    if kind == "drift":
        electrode_drift = np.tile(0.15 + 0.05 * np.sin(2 * np.pi * 0.2 * samples / FS), (9, 1))
        electrode_drift[3:] *= np.linspace(0.8, 1.2, 6)[:, None]
        signal += limb_projection(electrode_drift)
    elif kind == "gain":
        signal *= 1.1
    elif kind == "noise":
        noise_rng = np.random.default_rng(np.random.SeedSequence([SEED, subject["parameters"]["subject"], 5]))
        signal += limb_projection(noise_rng.normal(0, 0.02, (9, len(samples))))
    return signal
