"""Experiment 053: the five PhysioNet Challenge sources as unlabelled pretraining ECGs and external test sets.

The team's frozen Challenge split (``docs/challenge-splits-v1.md``) cannot be rebuilt on this machine, so this
module makes a local one with the same shares and the same spirit: record-level 60% / 20% / 20%
(``train`` / ``calibracion`` / ``prueba``) per source, exact duplicates kept together across sources. It is
not the team's split.

Every record is converted to the format of the PTB-XL pipeline: the centred 10 s window, 100 Hz, millivolts,
0.5 Hz high-pass, standard lead order. Diagnoses are read from the SNOMED codes of each header and mapped to
the 12 PTB-XL statements that have an equivalent; the two infarction locations have none. "Normal ECG" maps to
sinus rhythm as the only code, the closest Challenge statement.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score, roc_curve

from .ptbxl_simdino_memory import LEAD_NAMES, LEADS, preprocess

SOURCES = {
    "ningbo": ("challenge-2021/1.0.3", "chapman_ningbo"),
    "chapman_shaoxing": ("challenge-2021/1.0.3", "chapman_ningbo"),
    "georgia": ("challenge-2020/1.0.2", "georgia"),
    "cpsc_2018": ("challenge-2020/1.0.2", "cpsc"),
    "cpsc_2018_extra": ("challenge-2020/1.0.2", "cpsc"),
}
SNOMED = {
    "AFIB": {"164889003"},
    "STACH": {"427084000"},
    "SBRAD": {"426177001"},
    "SARRH": {"427393009"},
    "PVC": {"427172004", "17338001"},
    "PAC": {"284470004", "63593006"},
    "CLBBB": {"733534002", "164909002"},
    "CRBBB": {"713427006", "59118001"},
    "LAFB": {"445118002"},
    "1AVB": {"270492004"},
    "LVH": {"164873001"},
}
SINUS_RHYTHM = "426783006"
EXTERNAL_CODES = ("NORM", *SNOMED)
SPLIT_SHARES = (("train", 0.6), ("calibracion", 0.2), ("prueba", 0.2))
SOURCE_RATE = 500


def read_header(path: Path) -> dict:
    """
    Read one Challenge WFDB header.

    Parameters
    ----------
    path : Path
        The ``.hea`` file.

    Returns
    -------
    dict
        ``record``, ``rate``, ``samples``, ``leads`` (names), ``gains``, ``baselines``, ``codes`` (set of SNOMED
        codes), ``age`` and ``sex``.
    """
    lines = path.read_text().strip().split("\n")
    record, count, rate, samples = lines[0].split()[:4]
    fields = [line.split() for line in lines[1:1 + int(count)]]
    comments = {line[1:].split(":", 1)[0].strip(): line.split(":", 1)[1].strip()
                for line in lines[1 + int(count):] if line.startswith("#") and ":" in line}
    return {"record": record, "rate": int(float(rate)), "samples": int(samples),
            "leads": [field[-1] for field in fields],
            "gains": [float(field[2].split("/")[0].split("(")[0]) for field in fields],
            "baselines": [int(field[4]) for field in fields],
            "codes": {code.strip() for code in comments.get("Dx", "").split(",") if code.strip()},
            "age": comments.get("Age", ""), "sex": comments.get("Sex", "")}


def read_signal(path: Path, header: dict) -> np.ndarray:
    """
    Read a Challenge ``.mat`` signal in millivolts, leads in the standard order.

    Parameters
    ----------
    path : Path
        The ``.mat`` file next to the header.
    header : dict
        Output of ``read_header``.

    Returns
    -------
    np.ndarray
        Float32 ``(12, samples)``.

    Raises
    ------
    ValueError
        If the leads are not the twelve standard ones.
    """
    raw = loadmat(path)["val"].astype(np.float32)
    gains = np.asarray(header["gains"], dtype=np.float32)[:, None]
    baselines = np.asarray(header["baselines"], dtype=np.float32)[:, None]
    signal = (raw - baselines) / gains
    names = [name.upper() for name in header["leads"]]
    wanted = [name.upper() for name in LEAD_NAMES]
    if sorted(names) != sorted(wanted):
        raise ValueError(f"Unexpected leads {header['leads']} in {path}")
    return signal[[names.index(name) for name in wanted]]


def centred_window(signal: np.ndarray, rate: int, seconds: int = 10) -> np.ndarray | None:
    """
    The centred ``seconds`` window of a signal, or None if it is shorter.

    Parameters
    ----------
    signal : np.ndarray
        ``(12, samples)``.
    rate : int
        Sampling rate in Hz.
    seconds : int
        Window length.

    Returns
    -------
    np.ndarray or None
        ``(12, seconds * rate)``.
    """
    length = seconds * rate
    if signal.shape[1] < length:
        return None
    start = (signal.shape[1] - length) // 2
    return signal[:, start:start + length]


def external_labels(codes: set[str]) -> dict[str, bool]:
    """
    PTB-XL statements implied by a set of Challenge SNOMED codes.

    Parameters
    ----------
    codes : set[str]
        SNOMED codes of one record.

    Returns
    -------
    dict[str, bool]
        One value per code in ``EXTERNAL_CODES``; ``NORM`` is sinus rhythm as the only code.
    """
    return {"NORM": codes == {SINUS_RHYTHM}, **{name: bool(codes & snomed) for name, snomed in SNOMED.items()}}


def duplicate_groups(hashes: Iterable[str], keys: Iterable[str]) -> pd.Series:
    """
    Name every record by the lowest key among the records with an identical signal.

    Parameters
    ----------
    hashes : Iterable[str]
        Signal hash of each record.
    keys : Iterable[str]
        Unique key of each record, for example ``source/record``.

    Returns
    -------
    pd.Series
        Group key of each record, indexed by record key.
    """
    frame = pd.DataFrame({"hash": list(hashes), "key": list(keys)})
    lowest = frame.groupby("hash")["key"].transform("min")
    return pd.Series(lowest.to_numpy(), index=frame.key.to_numpy())


def assign_splits(manifest: pd.DataFrame, seed: int) -> pd.Series:
    """
    Record-level 60/20/20 split per source, with every duplicate group in one split.

    A group belongs to the source of its lowest key; within each source the groups are sorted, shuffled with
    one ``numpy`` generator (sources visited in sorted order) and cut by ``SPLIT_SHARES``.

    Parameters
    ----------
    manifest : pd.DataFrame
        One row per record with ``key``, ``group`` and ``source``.
    seed : int
        Seed of the generator.

    Returns
    -------
    pd.Series
        Split of each record, indexed by ``key``.
    """
    rng = np.random.default_rng(seed)
    owner = manifest.set_index("key").source
    groups = pd.Series(sorted(manifest.group.unique()))
    split_of_group = {}
    for source in sorted(manifest.source.unique()):
        members = groups[groups.map(owner).eq(source)].to_numpy()
        members = members[rng.permutation(len(members))]
        edges = np.cumsum([round(share * len(members)) for _, share in SPLIT_SHARES[:-1]])
        for part, (name, _) in zip(np.split(members, edges), SPLIT_SHARES):
            split_of_group.update(dict.fromkeys(part, name))
    return pd.Series(manifest.group.map(split_of_group).to_numpy(), index=manifest.key.to_numpy())


def build_source(raw_root: Path, source: str, cache: Path) -> pd.DataFrame:
    """
    Convert every record of one source to a 100 Hz float16 cache and describe it.

    Records that cannot be read, are shorter than 10 s, are not sampled at 500 Hz or have non-finite samples are
    listed with ``usable = False`` and left out of the cache.

    Parameters
    ----------
    raw_root : Path
        ``data/raw``.
    source : str
        Key of ``SOURCES``.
    cache : Path
        ``.npy`` file written with the usable records, in manifest order; an existing file is reused.

    Returns
    -------
    pd.DataFrame
        One row per record: ``source``, ``family``, ``record``, ``key``, ``seconds``, ``age``, ``sex``,
        ``usable``, ``reason``, ``signal_sha256``, ``cache_row`` and one boolean column per ``EXTERNAL_CODES``.
    """
    folder, family = SOURCES[source]
    headers = sorted((raw_root / folder / "training" / source).rglob("*.hea"))
    rows, pending, converted, usable = [], [], [], 0

    def flush() -> None:
        if pending:
            converted.append(preprocess(np.stack(pending), rate=100).astype(np.float16))
            pending.clear()

    for header_path in headers:
        header = read_header(header_path)
        row = {"source": source, "family": family, "record": header["record"],
               "key": f"{source}/{header['record']}", "seconds": header["samples"] / max(header["rate"], 1),
               "age": header["age"], "sex": header["sex"], "usable": False, "reason": "", "signal_sha256": "",
               "cache_row": -1, **external_labels(header["codes"])}
        try:
            signal = read_signal(header_path.with_suffix(".mat"), header)
            row["signal_sha256"] = hashlib.sha256(np.ascontiguousarray(signal, dtype=np.float32).tobytes()).hexdigest()
            window = centred_window(signal, header["rate"])
            if header["rate"] != SOURCE_RATE:
                row["reason"] = f"frecuencia {header['rate']} Hz"
            elif window is None:
                row["reason"] = "menos de 10 s"
            elif not np.isfinite(window).all():
                row["reason"] = "muestras no finitas"
            else:
                row["usable"], row["cache_row"] = True, usable
                usable += 1
                if not cache.exists():
                    pending.append(window)
                    if len(pending) == 1000:
                        flush()
        except (ValueError, KeyError, OSError) as error:
            row["reason"] = f"no se pudo leer: {type(error).__name__}"
        rows.append(row)
    if not cache.exists():
        flush()
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, np.concatenate(converted) if converted else np.zeros((0, LEADS, 1000), dtype=np.float16))
    return pd.DataFrame(rows)


def evaluable_pairs(sources: np.ndarray, labels: np.ndarray, codes: tuple[str, ...], minimum: int
                    ) -> list[tuple[str, int]]:
    """
    Source–statement pairs with at least ``minimum`` positives and ``minimum`` negatives.

    Parameters
    ----------
    sources : np.ndarray
        Source of each ECG.
    labels : np.ndarray
        Boolean ``(n, len(codes))``.
    codes : tuple[str, ...]
        Statement names, for the order of the columns.
    minimum : int
        Smallest count of each class.

    Returns
    -------
    list[tuple[str, int]]
        ``(source, column)`` pairs, sorted.
    """
    pairs = []
    for source in sorted(set(sources)):
        part = labels[sources == source]
        for column in range(len(codes)):
            positives = int(part[:, column].sum())
            if positives >= minimum and len(part) - positives >= minimum:
                pairs.append((source, column))
    return pairs


def pair_aurocs(sources: np.ndarray, labels: np.ndarray, scores: np.ndarray, pairs: list[tuple[str, int]]
                ) -> np.ndarray:
    """
    AUROC of every source–statement pair.

    Parameters
    ----------
    sources, labels, scores
        Source, boolean labels ``(n, codes)`` and scores ``(n, codes)`` of each ECG.
    pairs : list[tuple[str, int]]
        Output of ``evaluable_pairs``.

    Returns
    -------
    np.ndarray
        One AUROC per pair.
    """
    return np.array([roc_auc_score(labels[sources == source, column], scores[sources == source, column])
                     for source, column in pairs])


def paired_pair_mean_bootstrap(sources: np.ndarray, labels: np.ndarray, first: np.ndarray, second: np.ndarray,
                               pairs: list[tuple[str, int]], draws: int, seed: int) -> dict[str, float]:
    """
    Paired bootstrap of the mean pair AUROC, first minus second, resampling ECGs within each source.

    Parameters
    ----------
    sources, labels
        Source and boolean labels ``(n, codes)`` of each ECG.
    first, second : np.ndarray
        Scores ``(n, codes)`` of the two models on the same ECGs.
    pairs : list[tuple[str, int]]
        Output of ``evaluable_pairs``.
    draws : int
        Bootstrap draws; a draw where a pair loses one class is skipped and counted.
    seed : int
        Seed of ``numpy.random.default_rng``.

    Returns
    -------
    dict[str, float]
        ``difference``, ``ci_low``, ``ci_high`` and ``skipped_draws``.
    """
    rng = np.random.default_rng(seed)
    positions = {source: np.flatnonzero(sources == source) for source in sorted(set(sources))}
    observed = pair_aurocs(sources, labels, first, pairs).mean() - pair_aurocs(sources, labels, second, pairs).mean()
    differences = []
    for _ in range(draws):
        rows = np.concatenate([rng.choice(index, len(index)) for index in positions.values()])
        part_sources, part_labels = sources[rows], labels[rows]
        try:
            differences.append(pair_aurocs(part_sources, part_labels, first[rows], pairs).mean()
                               - pair_aurocs(part_sources, part_labels, second[rows], pairs).mean())
        except ValueError:
            continue
    low, high = np.percentile(differences, [2.5, 97.5])
    return {"difference": float(observed), "ci_low": float(low), "ci_high": float(high),
            "skipped_draws": draws - len(differences)}


def youden_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Threshold that maximizes sensitivity plus specificity (Youden's J).

    Parameters
    ----------
    labels : np.ndarray
        Boolean target.
    scores : np.ndarray
        Higher means more likely positive.

    Returns
    -------
    float
        The threshold; ECGs with ``score >= threshold`` are called positive.
    """
    false_positive, true_positive, thresholds = roc_curve(labels, scores)
    return float(thresholds[np.argmax(true_positive - false_positive)])


def sensitivity_specificity(labels: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[float, float]:
    """
    Sensitivity and specificity of calling ``score >= threshold`` positive.

    Parameters
    ----------
    labels : np.ndarray
        Boolean target with both classes.
    scores : np.ndarray
        Scores.
    threshold : float
        Decision threshold.

    Returns
    -------
    tuple[float, float]
        Sensitivity and specificity.
    """
    called = scores >= threshold
    return float(called[labels].mean()), float((~called[~labels]).mean())


def threshold_table(sources: np.ndarray, labels: np.ndarray, scores: np.ndarray, pairs: list[tuple[str, int]],
                    thresholds: dict[tuple[str, int], float]) -> pd.DataFrame:
    """
    Sensitivity, specificity and balanced accuracy of every pair at given thresholds.

    Parameters
    ----------
    sources, labels, scores
        Source, boolean labels ``(n, codes)`` and scores ``(n, codes)`` of each ECG.
    pairs : list[tuple[str, int]]
        Source–statement pairs.
    thresholds : dict[tuple[str, int], float]
        Threshold of each pair.

    Returns
    -------
    pd.DataFrame
        One row per pair with ``source``, ``column``, ``sensitivity``, ``specificity`` and ``balanced``.
    """
    rows = []
    for source, column in pairs:
        chosen = sources == source
        sensitivity, specificity = sensitivity_specificity(labels[chosen, column], scores[chosen, column],
                                                           thresholds[(source, column)])
        rows.append({"source": source, "column": column, "sensitivity": sensitivity, "specificity": specificity,
                     "balanced": (sensitivity + specificity) / 2})
    return pd.DataFrame(rows)


def paired_balanced_bootstrap(sources: np.ndarray, labels: np.ndarray, scores: np.ndarray,
                              pairs: list[tuple[str, int]], first: dict[tuple[str, int], float],
                              second: dict[tuple[str, int], float], draws: int, seed: int) -> dict[str, float]:
    """
    Paired bootstrap of the mean balanced accuracy over pairs, thresholds ``first`` minus ``second``.

    The thresholds stay fixed; ECGs are resampled within each source.

    Parameters
    ----------
    sources, labels, scores
        Source, boolean labels and scores of the evaluated ECGs.
    pairs : list[tuple[str, int]]
        Source–statement pairs.
    first, second : dict[tuple[str, int], float]
        Two sets of thresholds.
    draws : int
        Bootstrap draws; draws where a pair loses one class are skipped and counted.
    seed : int
        Seed of ``numpy.random.default_rng``.

    Returns
    -------
    dict[str, float]
        ``difference``, ``ci_low``, ``ci_high`` and ``skipped_draws``.
    """
    def mean_difference(rows: np.ndarray) -> float:
        part = (sources[rows], labels[rows], scores[rows])
        return float(threshold_table(*part, pairs, first).balanced.mean()
                     - threshold_table(*part, pairs, second).balanced.mean())

    rng = np.random.default_rng(seed)
    positions = [np.flatnonzero(sources == source) for source in sorted(set(sources))]
    observed = mean_difference(np.arange(len(sources)))
    differences = []
    for _ in range(draws):
        rows = np.concatenate([rng.choice(index, len(index)) for index in positions])
        value = mean_difference(rows)
        if np.isfinite(value):
            differences.append(value)
    low, high = np.percentile(differences, [2.5, 97.5])
    return {"difference": observed, "ci_low": float(low), "ci_high": float(high),
            "skipped_draws": draws - len(differences)}
