"""
Experiment 067: the frozen localization maps on new ECGAD records, and the holdout metric helpers.

The 061-068 runners refuse ECGAD holdout records and keep their per-record steps inside the scripts. This
module repeats those steps by calling the frozen module functions with the frozen fitted quantities (065
network weights, 066 lead-quality model and combiner weights, 062 normative model, 061 U_B references, 068
combiner weights). Nothing is fitted. See ``docs/experiment-067-holdout.md``.
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ecg_experiment import ROOT
from ecg_experiment import combination066 as cb
from ecg_experiment import combiner068 as cv
from ecg_experiment import mil065 as mil
from ecg_experiment import normative_map062 as nm
from ecg_experiment.ecgad import ECGAD_DIR, load_split
from ecg_experiment.eda.ptbxl import load_metadata, load_statements
from ecg_experiment.files import sha256_file
from ecg_experiment.locbench import (
    DRAWS,
    ECGAD_COLUMNS,
    FAMILIES,
    SEED,
    START,
    STOP,
    _bootstrap,
    _mean,
    a1_histograms,
    auroc,
    ecgad_record_metrics,
    record_score,
)
from ecg_experiment.locbench_baselines061 import (
    amplitude_map,
    exact_signal,
    qrs_prior_map,
    random_map,
    residual_map,
    ub_map,
)
from ecg_experiment.locbench_data import REGION_TABLE
from ecg_experiment.waveforms import LEADS

MIL065 = ROOT / "outputs/experiment065_mil_v1"
LOCBENCH061 = ROOT / "outputs/experiment061_locbench_v1"
NORMATIVE062 = ROOT / "outputs/experiment062_normative_v1"
OUT066 = ROOT / "outputs/experiment066_combination_v1"
OUT068 = ROOT / "outputs/experiment068_combiner_v2"
MIL_WEIGHTS = ("lse_65065", "lse_65066", "lse_65067")
BAR = "qrs_prior"
PRIMARY = "units_v2"
CANDIDATES = (BAR, "qrs_lexicographic", "factor", "units_learned", PRIMARY)
REFERENCES = ("random", "amplitude", "U_B", "residual056", "textbook", "textbook_true")
MAPS = (*CANDIDATES, *REFERENCES)
SAMPLES = 5000
MIL_CHUNK = 128
SHIFT_SEED = 61062
MIN_GROUP = 5
STRATA = ("A2_qrs", "A2_nonqrs")
DEVELOPMENT_TABLES = {
    **{name: LOCBENCH061 / "tables" / name for name in ("random", "amplitude", BAR, "U_B", "residual056")},
    "qrs_lexicographic": NORMATIVE062 / "locbench/qrs_lexicographic/tables",
    **{name: OUT066 / "locbench" / name / "tables" for name in ("factor", "textbook", "textbook_true")},
    "units_learned": OUT066 / "locbench/units_learned/tables",
    PRIMARY: OUT068 / "locbench" / PRIMARY / "tables",
}
DEVELOPMENT_A2 = {
    BAR: 0.7419383358406554,
    "qrs_lexicographic": 0.7530198339166797,
    "factor": 0.7246870157102797,
    "units_learned": 0.7603613741425232,
    PRIMARY: 0.7671251079351179,
    "textbook": 0.7493730086608462,
}
FROZEN_FILES = {
    **{f"weights_{name}": MIL065 / f"weights_{name}.pt" for name in MIL_WEIGHTS},
    "mil065_result": MIL065 / "result.json",
    "normative062": NORMATIVE062 / "normative.pkl",
    "ub_references061": LOCBENCH061 / "ub_references.pkl",
    "quality066": OUT066 / "quality/model.pkl",
    "result066": OUT066 / "result.json",
    "result068": OUT068 / "result.json",
}


def split_records(partition: str) -> pd.DataFrame:
    """
    ECGAD records of one split partition in benchmark-index order.

    Parameters
    ----------
    partition : str
        ``development`` or ``holdout``.

    Returns
    -------
    pd.DataFrame
        ``ecg_id``, ``patient_id``, ``bench_index``.
    """
    split = load_split()
    rows = split[split["partition"] == partition].sort_values("bench_index").reset_index(drop=True)
    return rows[["ecg_id", "patient_id", "bench_index"]].astype(int)


def frozen_hashes() -> dict[str, str]:
    """
    SHA-256 of every frozen fitted input.

    Returns
    -------
    dict[str, str]
        Name to SHA-256.
    """
    return {name: sha256_file(path) for name, path in FROZEN_FILES.items()}


def receipt_hash_check(hashes: dict[str, str]) -> dict[str, Any]:
    """
    Compare the frozen input hashes with those recorded in 066's ``result.json``.

    Parameters
    ----------
    hashes : dict[str, str]
        Output of ``frozen_hashes``.

    Returns
    -------
    dict[str, Any]
        Per input: recorded and current hash; ``pass``.
    """
    inputs = json.loads((OUT066 / "result.json").read_text())["inputs"]
    recorded = {
        **{f"weights_{name}": inputs["mil065_weights_sha256"][name] for name in MIL_WEIGHTS},
        "mil065_result": inputs["mil065_result_sha256"],
        "normative062": inputs["normative062_model_sha256"],
        "ub_references061": inputs["ub_references_sha256"],
    }
    rows = {name: {"recorded": value, "current": hashes[name]} for name, value in recorded.items()}
    return {"inputs": rows, "pass": all(row["recorded"] == row["current"] for row in rows.values())}


def heads065() -> list[str]:
    """Return the 49 065 heads in network order."""
    return json.loads((MIL065 / "result.json").read_text())["heads"]


def combiner_weights() -> dict[str, tuple[tuple[str, ...], np.ndarray]]:
    """
    Frozen unit-combiner weights of 066 ``units_learned`` and 068 ``units_v2``.

    Returns
    -------
    dict[str, tuple[tuple[str, ...], np.ndarray]]
        Map name to (feature names, weights).
    """
    saved066 = json.loads((OUT066 / "result.json").read_text())["maps"]["combiner_weights"]
    model068 = json.loads((OUT068 / "result.json").read_text())["fit"]["models"][PRIMARY]
    names068 = tuple(model068["features"])
    return {
        "units_learned": (cb.UNIT_FEATURES, np.array([saved066[name] for name in cb.UNIT_FEATURES])),
        PRIMARY: (names068, np.array([model068["weights"][name] for name in names068])),
    }


def load_frozen() -> dict[str, Any]:
    """
    Load every frozen fitted quantity used by ``record_maps``.

    Returns
    -------
    dict[str, Any]
        ``normative``, ``references``, ``quality``, ``weights`` and ``heads``.
    """
    with (NORMATIVE062 / "normative.pkl").open("rb") as handle:
        normative = pickle.load(handle)  # noqa: S301
    with (LOCBENCH061 / "ub_references.pkl").open("rb") as handle:
        references = pickle.load(handle)  # noqa: S301
    with (OUT066 / "quality/model.pkl").open("rb") as handle:
        quality = pickle.load(handle)  # noqa: S301
    return {
        "normative": normative,
        "references": references,
        "quality": quality,
        "weights": combiner_weights(),
        "heads": heads065(),
    }


def device_setup() -> str:
    """CUDA when visible, capped at 6 GB, deterministic cuDNN, as in 065 and 066."""
    if not torch.cuda.is_available():
        return "cpu"
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, 6e9 / total))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return "cuda"


def load_networks(heads: int, device: str) -> list[mil.LeadMIL]:
    """
    Load the three 065 ``mil_lse`` networks from their saved weights.

    Parameters
    ----------
    heads : int
        Number of heads (49).
    device : str
        Torch device.

    Returns
    -------
    list[mil.LeadMIL]
        Networks in evaluation mode.
    """
    networks = []
    for name in MIL_WEIGHTS:
        model = mil.LeadMIL(heads, pooling="lse", cross_lead=False)
        model.load_state_dict(torch.load(MIL065 / f"weights_{name}.pt", map_location="cpu"))
        networks.append(model.to(device).eval())
    return networks


def mil_summaries(
    signals: np.ndarray, networks: list[mil.LeadMIL], heads: list[str], device: str
) -> dict[str, np.ndarray]:
    """
    065 three-seed bag logits and evidence summaries, computed as Experiment 066 computed them.

    Parameters
    ----------
    signals : np.ndarray
        (n, 12, 5000) raw cache signals in mV.
    networks : list[mil.LeadMIL]
        Output of ``load_networks``.
    heads : list[str]
        Head names in network order.
    device : str
        Torch device.

    Returns
    -------
    dict[str, np.ndarray]
        ``bags`` (n, heads) float32, ``lead_means`` (n, heads, 12) float32, ``rhythm`` (n, 7, 12, steps)
        float16 and ``combined`` (n, 12, steps) float32.
    """
    abnormal = np.array([k for k, head in enumerate(heads) if head != mil.REFERENCE_HEAD])
    rhythm_index = [heads.index(head) for head in cb.RHYTHM_HEADS]
    n, steps = len(signals), mil.SAMPLES // mil.STRIDE + 1
    out = {
        "bags": np.zeros((n, len(heads)), np.float32),
        "lead_means": np.zeros((n, len(heads), len(LEADS)), np.float32),
        "rhythm": np.zeros((n, len(rhythm_index), len(LEADS), steps), np.float16),
        "combined": np.zeros((n, len(LEADS), steps), np.float32),
    }
    for start in range(0, n, MIL_CHUNK):
        chunk = np.arange(start, min(start + MIL_CHUNK, n))
        batch = mil.preprocess(signals[chunk])
        local = np.arange(len(chunk))
        outputs = [mil.predict(net, batch, local, device, keep_instances=True) for net in networks]
        bag = np.mean([o[0] for o in outputs], axis=0)
        evidence = mil.finding_evidence(np.mean([o[1] for o in outputs], axis=0), None)
        out["bags"][chunk] = bag
        out["lead_means"][chunk] = cb.lead_means(evidence)
        out["rhythm"][chunk] = evidence[:, rhythm_index].astype(np.float16)
        out["combined"][chunk] = mil.combined_map(bag, evidence, abnormal, "sum")
    return out


def true_labels(ecg_ids: np.ndarray, heads: list[str]) -> np.ndarray:
    """
    Binary 065-rule PTB-XL statement labels (labels times validity mask), as 066 built for ``textbook_true``.

    Parameters
    ----------
    ecg_ids : np.ndarray
        Records.
    heads : list[str]
        Head names.

    Returns
    -------
    np.ndarray
        (n, heads) float32.
    """
    y, mask = mil.statement_labels(load_metadata(), load_statements(), ecg_ids, heads)
    return y * mask


def lead_quality(features: np.ndarray, model: Any) -> np.ndarray:
    """
    Lead-quality probabilities from the frozen 066 model.

    Parameters
    ----------
    features : np.ndarray
        (n, 12, 9) ``combination066.quality_features`` of the records.
    model : Any
        Saved 066 quality pipeline.

    Returns
    -------
    np.ndarray
        (n, 12) probabilities of a noisy lead.
    """
    flat = features.reshape(-1, features.shape[-1])
    return model.predict_proba(flat)[:, 1].reshape(-1, len(LEADS))


def unit_raw(units: np.ndarray, fields: dict[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray:
    """Return the unit means of the named fields, shape (units, len(names))."""
    if not len(units):
        return np.zeros((0, len(names)))
    return np.column_stack([cb.unit_means(units, fields[name]) for name in names])


def record_maps(
    stored: np.ndarray,
    ecg_id: int,
    age: float,
    sex: float,
    mil_row: dict[str, np.ndarray],
    q: np.ndarray,
    truth: np.ndarray,
    frozen: dict[str, Any],
) -> dict[str, np.ndarray]:
    """
    Every declared map of one record, built as 061, 062, 066 and 068 built it.

    Parameters
    ----------
    stored : np.ndarray
        (12, 5000) float32 cache signal in mV.
    ecg_id : int
        PTB-XL record ID (seeds ``random``).
    age, sex : float
        Cache covariates (age NaN above 89).
    mil_row : dict[str, np.ndarray]
        This record's row of each ``mil_summaries`` array.
    q : np.ndarray
        (12,) lead-quality probabilities.
    truth : np.ndarray
        (heads,) ``true_labels`` row.
    frozen : dict[str, Any]
        Output of ``load_frozen``.

    Returns
    -------
    dict[str, np.ndarray]
        Map name to (12, 5000) float32 map, for every name in ``MAPS``.
    """
    raw = np.asarray(stored, np.float64)
    exact = exact_signal(stored)
    amplitude = amplitude_map(exact)
    prior, _ = qrs_prior_map(exact)
    ub = ub_map(exact, frozen["references"])
    residual = residual_map(exact, frozen["references"])
    residual = np.zeros_like(exact) if residual is None else residual
    normative, _ = nm.record_maps(raw, age, sex, frozen["normative"])

    heads = frozen["heads"]
    abnormal = [k for k, head in enumerate(heads) if head != mil.REFERENCE_HEAD]
    names = [heads[k] for k in abnormal]
    prob = 1.0 / (1.0 + np.exp(-np.asarray(mil_row["bags"], np.float64)[abnormal]))
    means = np.asarray(mil_row["lead_means"])[abnormal]
    rhythm = {head: np.asarray(mil_row["rhythm"][j], np.float32) for j, head in enumerate(cb.RHYTHM_HEADS)}
    intervals = cb.delineated_components(raw)
    masks = cb.component_masks(intervals)
    fields = cv.raw_fields(names, prob, means, rhythm, masks)
    fields["mil"] = mil.upsample(np.asarray(mil_row["combined"], np.float32)).astype(np.float64)
    fields["normative"] = normative[nm.PRIMARY].astype(np.float16).astype(np.float64)
    fields["residual"] = residual
    fields["amplitude"] = amplitude
    factor = cb.factor_sum(names, prob, means, rhythm, masks)
    textbook = cb.textbook_sum(names, prob, masks)
    truth = np.asarray(truth, np.float64)[abnormal]

    units = cb.unit_table(intervals)
    q = np.asarray(q, np.float64)
    learned_names, learned_weights = frozen["weights"]["units_learned"]
    v2_names, v2_weights = frozen["weights"][PRIMARY]
    if len(units):
        learned = cb.unit_design(unit_raw(units, fields, cb.UNIT_MAPS), q, units) @ learned_weights
        design = cv.unit_design(unit_raw(units, fields, cv.RAW_MAPS), q, units)
        v2 = design[:, cv.columns(v2_names)] @ v2_weights
    else:
        learned, v2 = np.zeros(0), np.zeros(0)
    maps = {
        "random": random_map(ecg_id),
        "amplitude": amplitude,
        BAR: prior,
        "U_B": np.zeros_like(exact) if ub is None else ub,
        "residual056": residual,
        "qrs_lexicographic": normative[nm.LEXICOGRAPHIC].astype(np.float16),
        "factor": cb.finish(factor, amplitude),
        "textbook": cb.finish(textbook, amplitude),
        "textbook_true": cb.finish(cb.textbook_sum(names, truth, masks), amplitude),
        "units_learned": cb.units_map(learned, units, amplitude),
        PRIMARY: cb.units_map(v2, units, amplitude),
    }
    out = {name: np.asarray(maps[name], np.float32) for name in MAPS}
    for name, values in out.items():
        if values.shape != (len(LEADS), SAMPLES) or not np.isfinite(values).all():
            raise ValueError(f"Map {name} of {ecg_id} is not a finite (12, 5000) array")
    return out


def saved_development_maps(name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Scored development maps of one map and their record order.

    Parameters
    ----------
    name : str
        Map name in ``MAPS``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``ecg_id`` order and memory-mapped float32 maps (198, 12, 5000).
    """
    folder = DEVELOPMENT_TABLES[name]
    order = np.array(json.loads((folder / "meta.json").read_text())["ecgad_ecg_ids"], np.int64)
    return order, np.load(folder / "ecgad_maps.npy", mmap_mode="r")


def map_difference(new: np.ndarray, saved: np.ndarray) -> dict[str, Any]:
    """
    Largest absolute difference between two map stacks and whether they are identical.

    Parameters
    ----------
    new, saved : np.ndarray
        Map stacks of equal shape.

    Returns
    -------
    dict[str, Any]
        ``max_abs_difference``, ``identical`` and ``records_differing``.
    """
    new = np.asarray(new, np.float32)
    saved = np.asarray(saved, np.float32)
    if new.shape != saved.shape:
        raise ValueError(f"Shapes differ: {new.shape} against {saved.shape}")
    difference = np.abs(new.astype(np.float64) - saved.astype(np.float64))
    per_record = difference.reshape(len(new), -1).max(axis=1)
    return {
        "max_abs_difference": float(difference.max()),
        "identical": bool(np.array_equal(new, saved)),
        "records_differing": int((per_record > 0).sum()),
    }


def label_rows(labels: np.ndarray, bench_index: np.ndarray) -> np.ndarray:
    """
    Boolean marks of chosen benchmark rows in canonical (12, 5000) orientation.

    Parameters
    ----------
    labels : np.ndarray
        ``ecgad.load_labels()`` (400, 5000, 12).
    bench_index : np.ndarray
        Benchmark rows to read.

    Returns
    -------
    np.ndarray
        (n, 12, 5000) boolean.
    """
    return np.stack([np.asarray(labels[int(index)]).T > 0.5 for index in bench_index])


def ecgad_tables(maps: np.ndarray, marks: np.ndarray, records: pd.DataFrame, name: str) -> dict[str, Any]:
    """
    Per-record ECGAD metrics and A1 histograms of one map on a set of records, in LocBench table form.

    The result works with ``locbench.metric_interval`` and ``paired_difference`` for A1-A6.

    Parameters
    ----------
    maps : np.ndarray
        (n, 12, 5000) maps.
    marks : np.ndarray
        (n, 12, 5000) boolean marks.
    records : pd.DataFrame
        ``ecg_id`` and ``patient_id`` of each row.
    name : str
        Map name.

    Returns
    -------
    dict[str, Any]
        ``name``, ``ecgad`` (``ecg_id``, ``patient_id``, ``marked``, A2, A3, A4, A6, ``marked_share``,
        ``record_score``), ``a1_positive``, ``a1_negative``, ``a1_exact``.
    """
    values = np.asarray(maps, np.float32)
    if values.shape != marks.shape or not np.isfinite(values).all():
        raise ValueError(f"Maps of {name} are not finite or not aligned with the marks")
    frame = records[["ecg_id", "patient_id"]].reset_index(drop=True).copy()
    frame["marked"] = marks[:, :, START:STOP].any(axis=(1, 2))
    rows = [ecgad_record_metrics(m, y) for m, y in zip(values, marks, strict=True)]
    for column in ECGAD_COLUMNS:
        frame[column] = [row[column] for row in rows]
    frame["record_score"] = [record_score(m) for m in values]
    positive, negative, exact = a1_histograms(
        values[:, :, START:STOP].reshape(len(values), -1), marks[:, :, START:STOP].reshape(len(values), -1)
    )
    return {"name": name, "ecgad": frame, "a1_positive": positive, "a1_negative": negative, "a1_exact": exact}


def strata_records(
    maps: np.ndarray, marks: np.ndarray, qrs: np.ndarray, records: pd.DataFrame
) -> pd.DataFrame:
    """
    Per-record ``A2_qrs`` and ``A2_nonqrs`` (061 Amendment 2) on a set of records.

    Parameters
    ----------
    maps : np.ndarray
        (n, 12, 5000) maps.
    marks : np.ndarray
        (n, 12, 5000) boolean marks.
    qrs : np.ndarray
        (n, 12, 5000) boolean QRS mask (``qrs_prior >= 0.5``).
    records : pd.DataFrame
        ``ecg_id`` and ``patient_id`` of each row.

    Returns
    -------
    pd.DataFrame
        ``ecg_id``, ``patient_id``, ``A2_qrs``, ``A2_nonqrs`` (NaN where a stratum lacks a class).
    """
    rows = []
    for k, record in enumerate(records.itertuples(index=False)):
        y_all = marks[k][:, START:STOP]
        values = np.asarray(maps[k], np.float64)[:, START:STOP]
        inside = qrs[k][:, START:STOP]
        row = {"ecg_id": int(record.ecg_id), "patient_id": int(record.patient_id)}
        for name, mask in zip(STRATA, (inside, ~inside), strict=True):
            y = y_all[mask]
            row[name] = auroc(y, values[mask]) if y.any() and not y.all() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def statement_groups(ecg_ids: np.ndarray, meta: pd.DataFrame) -> dict[int, list[str]]:
    """
    ``REGION_TABLE`` families of each record's statements at likelihood at least 50, or ``other``.

    Parameters
    ----------
    ecg_ids : np.ndarray
        Records.
    meta : pd.DataFrame
        ``eda.ptbxl.load_metadata`` output.

    Returns
    -------
    dict[int, list[str]]
        Record to sorted family names (``["other"]`` when it has none).
    """
    groups = {}
    for ecg_id in ecg_ids:
        scp = meta.at[int(ecg_id), "scp_codes"]
        families = sorted({FAMILIES[code] for code in REGION_TABLE if scp.get(code, 0) >= 50})
        groups[int(ecg_id)] = families or ["other"]
    return groups


def group_means(
    frames: dict[str, pd.DataFrame],
    groups: dict[int, list[str]],
    column: str = "A2",
    minimum: int = MIN_GROUP,
) -> pd.DataFrame:
    """
    Mean of a per-record metric per statement group and map, for groups with enough records.

    Parameters
    ----------
    frames : dict[str, pd.DataFrame]
        Map name to per-record table with ``ecg_id`` and ``column``.
    groups : dict[int, list[str]]
        Output of ``statement_groups``.
    column : str
        Metric column.
    minimum : int
        Fewest records with a defined metric.

    Returns
    -------
    pd.DataFrame
        One row per group: ``group``, ``records`` and one column per map.
    """
    first = next(iter(frames.values()))
    defined = first.loc[first[column].notna(), "ecg_id"].astype(int).tolist()
    names = sorted({group for ecg_id in defined for group in groups[ecg_id]})
    rows = []
    for group in names:
        members = [ecg_id for ecg_id in defined if group in groups[ecg_id]]
        if len(members) < minimum:
            continue
        row: dict[str, Any] = {"group": group, "records": len(members)}
        for name, frame in frames.items():
            row[name] = float(frame.set_index("ecg_id").loc[members, column].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def independent_shift(
    holdout: pd.DataFrame,
    development: pd.DataFrame,
    column: str = "A2",
    draws: int = DRAWS,
    seeds: tuple[int, int] = (SEED, SHIFT_SEED),
) -> dict[str, Any]:
    """
    Holdout mean minus development mean of a per-record metric, with independent patient bootstraps.

    Parameters
    ----------
    holdout, development : pd.DataFrame
        Per-record tables with ``patient_id`` and ``column``.
    column : str
        Metric column.
    draws : int
        Bootstrap draws per half.
    seeds : tuple[int, int]
        Seeds of the holdout and development draws.

    Returns
    -------
    dict[str, Any]
        ``value``, ``ci_low``, ``ci_high``, ``holdout``, ``development`` and record counts.
    """
    halves = []
    for frame, seed in zip((holdout, development), seeds, strict=True):
        kept = frame[frame[column].notna()]
        values, _ = _bootstrap(kept["patient_id"].to_numpy(), [_mean(kept[column].to_numpy())], draws, seed)
        halves.append((kept, values[:, 0]))
    (held, held_draws), (dev, dev_draws) = halves
    low, high = np.percentile(held_draws - dev_draws, [2.5, 97.5])
    return {
        "value": float(held[column].mean() - dev[column].mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "holdout": float(held[column].mean()),
        "development": float(dev[column].mean()),
        "holdout_records": len(held),
        "development_records": len(dev),
    }


def decision(interval: dict[str, Any]) -> str:
    """
    Primary decision words from a paired interval.

    Parameters
    ----------
    interval : dict[str, Any]
        ``ci_low`` and ``ci_high`` of a paired difference.

    Returns
    -------
    str
        ``confirmed``, ``matches`` or ``below``.
    """
    if interval["ci_low"] > 0:
        return "confirmed"
    if interval["ci_high"] < 0:
        return "below"
    return "matches"


def label_hash_check() -> dict[str, Any]:
    """
    SHA-256 of the ECGAD label file against the hash recorded in ``docs/ecgad-benchmark-v1.md``.

    Returns
    -------
    dict[str, Any]
        ``recorded``, ``current`` and ``pass``.
    """
    text = (ROOT / "docs/ecgad-benchmark-v1.md").read_text()
    recorded = re.search(r"`benchmark_label\.npy` \| `([0-9a-f]{64})`", text).group(1)
    current = sha256_file(ECGAD_DIR / "benchmark_label.npy")
    return {"recorded": recorded, "current": current, "pass": recorded == current}


def manifest_hashes(folder: Path, names: tuple[str, ...] = MAPS) -> dict[str, str]:
    """
    SHA-256 of ``maps_<name>.npy`` in a folder.

    Parameters
    ----------
    folder : Path
        Map folder.
    names : tuple[str, ...]
        Map names.

    Returns
    -------
    dict[str, str]
        Name to SHA-256.
    """
    return {name: sha256_file(folder / f"maps_{name}.npy") for name in names}
