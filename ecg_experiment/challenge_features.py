"""Read verified PhysioNet Challenge records as the inputs of the frozen CPC, ECG-JEPA and xECG encoders.

Each record is read from WFDB in float64 mV and canonical lead order. CPC and ECG-JEPA receive the
float32 window, as ``waveforms.read_record`` gave the PTB-XL caches and SPH gave Experiment 022. xECG
receives the float64 window, as the PTB-XL xECG cache was resampled from WFDB's float64 samples.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import wfdb

from . import ROOT
from .cpc_input_audit import historical_resample
from .eda.signals import canonical_order
from .external_encoders import jepa_input, xecg_input
from .wfdb_records import first_unverified_file

RAW_ROOTS = {
    "ningbo": ROOT / "data/raw/challenge-2021/1.0.3",
    "chapman_shaoxing": ROOT / "data/raw/challenge-2021/1.0.3",
    "georgia": ROOT / "data/raw/challenge-2020/1.0.2",
    "cpsc_2018": ROOT / "data/raw/challenge-2020/1.0.2",
    "cpsc_2018_extra": ROOT / "data/raw/challenge-2020/1.0.2",
}
SAMPLE_RATE = 500
SAMPLES = 5000
LEAD_COUNT = 12
RECORD_SUFFIXES = (".hea", ".mat")


def record_stems(checksums: dict[str, str], source: str) -> list[str]:
    """
    Sorted record paths of one source listed in an official checksum manifest.

    Parameters
    ----------
    checksums : dict[str, str]
        Digests keyed by release-relative file name, as from ``downloads.parse_checksums``.
    source : str
        Directory name under ``training/``, such as ``georgia``.

    Returns
    -------
    list[str]
        Release-relative record paths without extension.
    """
    prefix = f"training/{source}/"
    return sorted(name.removesuffix(".hea") for name in checksums
                  if name.startswith(prefix) and name.endswith(".hea"))


def files_digest(stems: list[str], checksums: dict[str, str]) -> str:
    """
    One SHA-256 over the official digests of the record files, in record order.

    Parameters
    ----------
    stems : list[str]
        Record paths without extension.
    checksums : dict[str, str]
        Official digests keyed by release-relative file name.

    Returns
    -------
    str
        Hexadecimal digest of the ``<digest>  <name>`` lines of every ``.hea`` and ``.mat`` file.
    """
    lines = "".join(f"{checksums[stem + suffix]}  {stem + suffix}\n"
                    for stem in stems for suffix in RECORD_SUFFIXES)
    return hashlib.sha256(lines.encode()).hexdigest()


def skip_reasons(sampling_rate: int, signal: np.ndarray) -> list[str]:
    """
    Reasons a record is not a finite ten-second twelve-lead 500 Hz ECG.

    Parameters
    ----------
    sampling_rate : int
        Sampling rate in Hz.
    signal : np.ndarray
        Time-major array of shape ``(samples, leads)``.

    Returns
    -------
    list[str]
        Empty when the record can be featurized.
    """
    samples, leads = signal.shape
    reasons = []
    if sampling_rate != SAMPLE_RATE:
        reasons.append(f"sampling_rate_{sampling_rate}_hz")
    if leads != LEAD_COUNT:
        reasons.append(f"leads_{leads}")
    if samples != SAMPLES:
        reasons.append(f"samples_{samples}")
    if not np.isfinite(signal).all():
        reasons.append("nonfinite")
    return reasons


def read_verified(root: Path, stem: str, checksums: dict[str, str]) -> tuple[np.ndarray, int, list[str]]:
    """
    Read one WFDB record after checking its files against the official checksums.

    Parameters
    ----------
    root : Path
        Release root the checksum names are relative to.
    stem : str
        Record path relative to ``root``, without extension.
    checksums : dict[str, str]
        Official digests keyed by release-relative file name.

    Returns
    -------
    tuple[np.ndarray, int, list[str]]
        Float64 time-major samples in mV, the sampling rate and the stored lead names.

    Raises
    ------
    ValueError
        If a file is missing or differs from its checksum, or a lead is not in mV.
    """
    unverified = first_unverified_file(root, stem, RECORD_SUFFIXES, checksums)
    if unverified is not None:
        raise ValueError(f"Record file missing or differs from the official checksum: {unverified}")
    signal, fields = wfdb.rdsamp(str(root / stem))
    if set(fields["units"]) != {"mV"}:
        raise ValueError(f"Expected mV in every lead of {stem}: {fields['units']}")
    return signal, int(fields["fs"]), list(fields["sig_name"])


def canonical_window(signal: np.ndarray, names: list[str]) -> np.ndarray:
    """
    Lead-major float64 window in canonical lead order.

    Parameters
    ----------
    signal : np.ndarray
        Time-major ``(5000, 12)`` samples.
    names : list[str]
        Stored lead name of each column.

    Returns
    -------
    np.ndarray
        Float64 ``[12, 5000]`` window.
    """
    return np.ascontiguousarray(canonical_order(signal, names).T, dtype=np.float64)


def encoder_inputs(window: np.ndarray) -> dict[str, np.ndarray]:
    """
    Build the inputs of the three encoders from one float64 window.

    Parameters
    ----------
    window : np.ndarray
        Float64 ``[12, 5000]`` 500 Hz window in mV, canonical lead order.

    Returns
    -------
    dict[str, np.ndarray]
        ``cpc`` ``[12, 2500]``, ``jepa`` ``[8, 2500]`` and ``xecg`` ``[1000, 12]``, all float32.
    """
    narrow = window.astype(np.float32)
    return {"cpc": historical_resample(narrow), "jepa": jepa_input(narrow), "xecg": xecg_input(window)}
