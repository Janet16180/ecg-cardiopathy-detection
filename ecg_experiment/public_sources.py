"""Public Challenge ECG releases and their canonical ten-second views."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import wfdb

from .waveforms import LEADS


SOURCE = {
    "georgia": ("challenge-2020", "1.0.2", "training/georgia"),
    "cpsc_2018": ("challenge-2020", "1.0.2", "training/cpsc_2018"),
    "cpsc_2018_extra": ("challenge-2020", "1.0.2", "training/cpsc_2018_extra"),
    "chapman_shaoxing": ("challenge-2021", "1.0.3", "training/chapman_shaoxing"),
}


# signal_sha256 is hashed by inspect.getsource() into cached leakage-reference
# identities (scripts/data/prepare_public_ecg.py). Keep its text byte-identical.
def signal_sha256(signal: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(signal, dtype="<f4").tobytes()).hexdigest()


def load_view(raw_dir: Path, stem: str, policy: str) -> tuple[np.ndarray, int, int, str]:
    """
    Read one public WFDB record as a canonical ten-second view.

    The raw file is never modified. Leads are reordered to ``LEADS``.

    Parameters
    ----------
    raw_dir : Path
        Release root that ``stem`` must stay inside.
    stem : str
        Record path relative to ``raw_dir``, without extension.
    policy : str
        ``"strict_10s"`` requires exactly 5000 samples; ``"ssl_center_crop"``
        takes the centered 5000 samples of a longer record.

    Returns
    -------
    tuple[np.ndarray, int, int, str]
        Float32 ``[12, 5000]`` physical-mV signal, window start sample, full
        record length, and semicolon-separated QC review flags.

    Raises
    ------
    ValueError
        With a short reason code when the record breaks the input contract.
    """
    if policy not in {"strict_10s", "ssl_center_crop"}:
        raise ValueError("unknown_policy")
    path = (raw_dir / stem).resolve()
    if not path.is_relative_to(raw_dir.resolve()):
        raise ValueError("unsafe_path")
    record = wfdb.rdrecord(str(path))
    names = [name.upper() for name in record.sig_name]
    if record.fs != 500 or len(names) != 12 or set(names) != {x.upper() for x in LEADS}:
        raise ValueError("lead_or_rate_contract")
    if record.units != ["mV"] * 12:
        raise ValueError("units_contract")
    waveform = np.asarray(record.p_signal, dtype=np.float32)
    if waveform.ndim != 2 or waveform.shape[1] != 12:
        raise ValueError("nonfinite_or_shape")
    samples = waveform.shape[0]
    if samples < 5000 or (policy == "strict_10s" and samples != 5000):
        raise ValueError("duration_contract")
    start = (samples - 5000) // 2 if policy == "ssl_center_crop" else 0
    signal = waveform[start:start + 5000, [names.index(x.upper()) for x in LEADS]].T.copy()
    if signal.shape != (12, 5000) or not np.isfinite(signal).all():
        raise ValueError("nonfinite_or_shape")
    spread = np.ptp(signal, axis=1)
    if np.any(spread == 0):
        raise ValueError("constant_lead")
    flags = []
    if np.any(signal.std(axis=1) < 0.01):
        flags.append("near_flat_lead_review")
    if np.max(np.abs(signal)) > 10:
        flags.append("amplitude_over_10mV_review")
    return signal, start, samples, ";".join(flags)
