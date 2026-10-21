"""Independent frozen-localizer ranks, input geometry and CPU convolution replay."""

from typing import Any

import numpy as np

from ecg_experiment.intervals import patient_groups, patient_resample
from ecg_experiment.ssl_synthetic_ludb_coherence_audit import standardize
from ecg_experiment.ssl_synthetic_ludb_landmark_audit import midranks

ARMS = ("B", "P", "K", "PK")
FITS = (1, 2, 3)
NATIVE = (0, 1, 6, 7, 8, 9, 10, 11)
LEADS = ("i", "ii", "v1", "v2", "v3", "v4", "v5", "v6")
FRACTIONS = (0.17, 0.31, 0.43, 0.59, 0.71, 0.83)


def prepare(physical: np.ndarray) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    """Reconstruct physical-to-training input and full-record native derivative energy.

    Parameters
    ----------
    physical : np.ndarray
        Finite physical canonical twelve-lead array of shape (12, 5000).

    Returns
    -------
    tuple
        Float32 model input, robust transformation and float64 native energy.
    """
    if physical.shape != (12, 5000) or not np.isfinite(physical).all():
        raise ValueError("Require canonical physical twelve-lead geometry")
    native, transformation = standardize(physical[list(NATIVE)].astype(np.float32))
    first, second = native[0], native[1]
    derived = np.stack(
        (
            first,
            second,
            second - first,
            -(first + second) / 2,
            first - second / 2,
            second - first / 2,
            *native[2:],
        )
    )
    values = native.astype(np.float64)
    differences = np.zeros_like(values)
    differences[:, 1:] = values[:, 1:] - values[:, :-1]
    energy = np.stack([np.convolve(row * row, np.full(20, 0.05), mode="same") for row in differences])
    maxima = energy.max(axis=1)
    np.divide(energy, maxima[:, None], out=energy, where=maxima[:, None] > 0)
    return derived, transformation, energy


def association(values: np.ndarray, energy: np.ndarray) -> dict[str, Any]:
    """Reconstruct tied-rank correlation and strict linear-q90 set overlap.

    Parameters
    ----------
    values : np.ndarray
        Probability values within the fixed unlabeled span.
    energy : np.ndarray
        Full-record derivative energy from standardized native input.

    Returns
    -------
    dict[str, Any]
        Independently reconstructed values or explicitly verified metadata.
    """
    x, y = np.asarray(values, dtype=np.float64), np.asarray(energy, dtype=np.float64)
    if x.ndim != 1 or x.shape != y.shape or not x.size or not np.isfinite([x, y]).all():
        raise ValueError("Require paired finite one-dimensional spans")
    flat_x, flat_y = np.ptp(x) == 0, np.ptp(y) == 0
    reason = (
        "constant_map_and_energy"
        if flat_x and flat_y
        else "constant_map"
        if flat_x
        else "constant_energy"
        if flat_y
        else None
    )
    a, b = midranks(x), midranks(y)
    a, b = a - a.mean(), b - b.mean()
    rho = None if reason else float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))
    xmask, ymask = x > np.quantile(x, 0.9, method="linear"), y > np.quantile(y, 0.9, method="linear")
    union = int(np.count_nonzero(xmask | ymask))
    return {
        "spearman": rho,
        "spearman_undefined_reason": reason,
        "jaccard": float(np.count_nonzero(xmask & ymask) / union) if union else 0.0,
        "empty_jaccard_union": union == 0,
    }


def lead_row(probability: np.ndarray, energy: np.ndarray, structural: dict[str, Any]) -> dict[str, Any]:
    """Independently reconstruct every retained native-lead landmark and energy endpoint.

    Parameters
    ----------
    probability : np.ndarray
        One native full-record probability vector.
    energy : np.ndarray
        Full-record derivative energy from standardized native input.
    structural : dict[str, Any]
        Previously verified saved N timestamps and reference span.

    Returns
    -------
    dict[str, Any]
        Independently reconstructed values or explicitly verified metadata.
    """
    start, end = structural["start"], structural["end_exclusive"]
    values = np.asarray(probability[start:end], dtype=np.float64)
    activity = np.asarray(energy[start:end], dtype=np.float64)
    if values.size == 0 or values.shape != activity.shape or not np.isfinite([values, activity]).all():
        raise ValueError("Invalid finite reference geometry")
    if np.any(values < 0) or np.any(values > 1):
        raise ValueError("Probability outside zero-to-one range")
    positions = np.asarray(structural["N_samples"], dtype=np.int64) - start
    shifts = [int(values.size * fraction) for fraction in FRACTIONS]
    if (
        not positions.size
        or positions.min() < 0
        or positions.max() >= values.size
        or len(set(shifts)) != 6
        or min(shifts) <= 0
        or max(shifts) >= values.size
    ):
        raise ValueError("Invalid retained landmark or six-control geometry")
    ranks = midranks(values)
    original = ranks[positions]
    controls = [np.roll(ranks, offset)[positions] for offset in shifts]
    control_means = [float(value.mean()) for value in controls]
    associations = [
        association(values, activity),
        *[association(np.roll(values, offset), activity) for offset in shifts],
    ]
    defined = [row["spearman"] for row in associations[1:] if row["spearman"] is not None]
    return {
        "lead": structural["lead"],
        "start": start,
        "end_exclusive": end,
        "N_samples": structural["N_samples"],
        "shift_samples": shifts,
        "N_midrank_values": original.tolist(),
        "N_mean_midrank": float(original.mean()),
        "shift_N_midrank_values": [value.tolist() for value in controls],
        "shift_N_mean_midrank": control_means,
        "mean_shift_N_midrank": float(np.mean(control_means)),
        "gain": float(original.mean()) - float(np.mean(control_means)),
        "constant_map": bool(np.ptp(values) == 0),
        "probability_mean": float(values.mean()),
        "probability_q95": float(np.quantile(values, 0.95, method="linear")),
        "probability_max": float(values.max()),
        "energy_association": associations[0],
        "shift_energy_association": associations[1:],
        "mean_shift_energy_spearman": float(np.mean(defined)) if defined else None,
        "defined_shift_spearman_count": len(defined),
        "mean_shift_energy_jaccard": float(np.mean([row["jaccard"] for row in associations[1:]])),
    }


def record_row(
    probabilities: np.ndarray, energy: np.ndarray, structure: dict[str, Any], fit: int, arm: str
) -> dict[str, Any]:
    """Reduce eight independently reconstructed native rows with fixed equal weights.

    Parameters
    ----------
    probabilities : np.ndarray
        Complete twelve-output native-rate probability maps.
    energy : np.ndarray
        Full-record derivative energy from standardized native input.
    structure : dict[str, Any]
        Previously verified twelve-lead saved annotation structure.
    fit : int
        One of the three fixed checkpoint fit identifiers.
    arm : str
        One of the four fixed editor-training arms.

    Returns
    -------
    dict[str, Any]
        Independently reconstructed values or explicitly verified metadata.
    """
    if probabilities.shape != (12, 5000) or energy.shape != (8, 5000):
        raise ValueError("Require complete maps and native input energy")
    leads = [
        lead_row(probabilities[channel], energy[position], structure["leads"][channel])
        for position, channel in enumerate(NATIVE)
    ]
    controls = np.asarray([row["shift_N_mean_midrank"] for row in leads]).mean(axis=0)
    native = float(np.mean([row["N_mean_midrank"] for row in leads]))
    return {
        "record_id": structure["record_id"],
        "fit_id": fit,
        "arm": arm,
        "leads": leads,
        "N_mean_midrank": native,
        "shift_N_mean_midrank": controls.tolist(),
        "mean_shift_N_midrank": float(controls.mean()),
        "gain": native - float(controls.mean()),
    }


def defined_summary(values: list[float | None]) -> dict[str, Any]:
    """Retain every defined and undefined correlation denominator.

    Parameters
    ----------
    values : list[float | None]
        Probability values within the fixed unlabeled span.

    Returns
    -------
    dict[str, Any]
        Independently reconstructed values or explicitly verified metadata.
    """
    defined = [value for value in values if value is not None]
    return {
        "mean": float(np.mean(defined)) if defined else None,
        "defined_count": len(defined),
        "undefined_count": len(values) - len(defined),
        "total_count": len(values),
    }


def aggregate(rows: list[dict[str, Any]], config: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    """Independently reproduce fixed-fit reductions and shared record-only bootstrap draws.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        Complete saved model-record rows in their preserved order.
    config : dict[str, Any]
        Frozen prospective scientific and resource declarations.

    Returns
    -------
    tuple[dict[str, Any], np.ndarray]
        Independently reconstructed values or explicitly verified metadata.
    """
    identifiers = [record for record in range(1, 201) if record % 5]
    indexed = {(row["record_id"], row["fit_id"], row["arm"]): row for row in rows}
    expected = {(record, fit, arm) for record in identifiers for fit in FITS for arm in ARMS}
    if set(indexed) != expected or len(rows) != len(expected):
        raise ValueError("Require all160 records, three fixed fits and four arms exactly once")
    paired = np.asarray(
        [[[indexed[record, fit, arm]["gain"] for arm in ARMS] for fit in FITS] for record in identifiers]
    ).mean(axis=1)
    rng, groups = np.random.default_rng(config["bootstrap"]["seed"]), patient_groups(np.arange(160))
    bootstrap = np.asarray(
        [paired[patient_resample(groups, rng)].mean(axis=0) for _ in range(config["bootstrap"]["draws"])]
    )
    draws = np.column_stack((bootstrap, bootstrap[:, 3] - bootstrap[:, 0]))
    points = np.r_[paired.mean(axis=0), (paired[:, 3] - paired[:, 0]).mean()]
    bounds = np.percentile(draws, [2.5, 97.5], axis=0, method="linear")
    estimates = {
        name: {
            "mean": float(points[position]),
            "ci_low": float(bounds[0, position]),
            "ci_high": float(bounds[1, position]),
        }
        for position, name in enumerate((*ARMS, "PK_minus_B"))
    }
    arms = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        leads = [lead for row in selected for lead in row["leads"]]
        arms[arm] = {
            "gain": estimates[arm],
            "N_mean_midrank": float(np.mean([row["N_mean_midrank"] for row in selected])),
            "shift_N_mean_midrank": np.asarray([row["shift_N_mean_midrank"] for row in selected])
            .mean(0)
            .tolist(),
            "energy_spearman": defined_summary([row["energy_association"]["spearman"] for row in leads]),
            "shift_energy_spearman": defined_summary([row["mean_shift_energy_spearman"] for row in leads]),
            "energy_jaccard": float(np.mean([row["energy_association"]["jaccard"] for row in leads])),
            "shift_energy_jaccard": float(np.mean([row["mean_shift_energy_jaccard"] for row in leads])),
            "constant_map_leads": sum(row["constant_map"] for row in leads),
            "empty_jaccard_unions": sum(row["energy_association"]["empty_jaccard_union"] for row in leads),
            "probability_mean": float(np.mean([row["probability_mean"] for row in leads])),
            "probability_q95_mean": float(np.mean([row["probability_q95"] for row in leads])),
            "probability_max": float(max(row["probability_max"] for row in leads)),
        }
    per_lead = {
        str(fit): {
            arm: {
                lead: float(
                    np.mean([indexed[record, fit, arm]["leads"][position]["gain"] for record in identifiers])
                )
                for position, lead in enumerate(LEADS)
            }
            for arm in ARMS
        }
        for fit in FITS
    }
    return {
        "primary_descriptive": estimates["B"],
        "per_arm": arms,
        "PK_minus_B": estimates["PK_minus_B"],
        "per_fit_arm_lead_gain": per_lead,
        "bootstrap_seed": config["bootstrap"]["seed"],
        "bootstrap_draws": len(draws),
        "skipped_draws": 0,
        "bootstrap_columns": [*ARMS, "PK_minus_B"],
        "records": 160,
        "model_record_rows": 1920,
        "native_lead_rows": 15360,
        "fixed_fit_count": 3,
    }, draws


def functional_replay(state: dict[str, Any], observed: np.ndarray) -> np.ndarray:
    """Run six strictly checked frozen tensors through independent CPU functional convolutions.

    Parameters
    ----------
    state : dict[str, Any]
        Plain six-tensor CPU float32 frozen state dictionary.
    observed : np.ndarray
        Float32 batched twelve-lead model inputs.

    Returns
    -------
    np.ndarray
        Independently reconstructed values or explicitly verified metadata.
    """
    import torch
    import torch.nn.functional as functional

    shapes = {
        "0.weight": (32, 12, 25),
        "0.bias": (32,),
        "2.weight": (32, 32, 25),
        "2.bias": (32,),
        "4.weight": (12, 32, 1),
        "4.bias": (12,),
    }
    if set(state) != set(shapes) or any(
        tuple(state[key].shape) != shape
        or state[key].dtype != torch.float32
        or state[key].device.type != "cpu"
        or not torch.isfinite(state[key]).all()
        for key, shape in shapes.items()
    ):
        raise ValueError("Checkpoint does not match the actual six-tensor frozen architecture")
    if observed.dtype != np.float32 or observed.ndim != 3 or observed.shape[1:] != (12, 5000):
        raise ValueError("Require float32 batched actual native-rate model inputs")
    with torch.inference_mode():
        values = torch.from_numpy(np.array(observed, copy=True))
        values = functional.relu(functional.conv1d(values, state["0.weight"], state["0.bias"], padding=12))
        values = functional.relu(
            functional.conv1d(values, state["2.weight"], state["2.bias"], padding=48, dilation=4)
        )
        values = functional.conv1d(values, state["4.weight"], state["4.bias"])
        return values.sigmoid().numpy()


CONFIG = "configs/ssl_synthetic_ludb_localizer_v1.json"
RUNNER = "ecg_experiment/ssl_synthetic_ludb_localizer_audit.py"
CHECKER = "scripts/validation/audit_ssl_synthetic_ludb_localizer.py"
AUDIT_PROTOCOL = "docs/ssl-synthetic-localization-ludb-localizer-independent-audit-v1.md"
TESTS = "tests/test_ssl_synthetic_ludb_localizer_audit.py"
COHERENCE = "outputs/ssl_synthetic_localization/ludb_coherence_bridge_v2"
TIMING = "outputs/ssl_synthetic_localization/ludb_peak_timing_v1"
MODELS = "outputs/ssl_synthetic_localization/generated_paired_suppression_v1"


def read_rows(path: Any) -> list[dict[str, Any]]:
    """Read a complete local JSONL table without reopening annotation streams.

    Parameters
    ----------
    path : Any
        Local input table path.

    Returns
    -------
    list[dict[str, Any]]
        Independently reconstructed values or explicitly verified metadata.
    """
    import json

    return [json.loads(line) for line in path.read_text().splitlines()]


def metadata_setup(root: Any, source: Any, admission_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate source and predecessor receipt schemas without reading maps or waveforms.

    Parameters
    ----------
    root : Any
        Artifact checkout containing local outputs and permitted raw aliases.
    source : Any
        Isolated checkout containing the committed executable sources.
    admission_name : str
        Repository-relative committed identity admission.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        Independently reconstructed values or explicitly verified metadata.
    """
    import os

    from ecg_experiment import ssl_synthetic_ludb_coherence_audit_v2 as previous
    from ecg_experiment.ssl_synthetic_ludb_landmark_audit_recovery import next_stage
    from ecg_experiment.ssl_synthetic_pretrained_saved_audit import require
    from ecg_experiment.synthetic_audit_io import file_sha256, read_json

    config, admission = read_json(source / CONFIG), read_json(source / admission_name)
    required = {
        RUNNER,
        CHECKER,
        AUDIT_PROTOCOL,
        TESTS,
        CONFIG,
        "ecg_experiment/intervals.py",
        "ecg_experiment/synthetic_audit_io.py",
        "ecg_experiment/ssl_synthetic_ludb_audit.py",
        "ecg_experiment/ssl_synthetic_ludb_landmark_audit.py",
        "ecg_experiment/ssl_synthetic_ludb_coherence_audit.py",
        "ecg_experiment/ssl_synthetic_ludb_coherence_audit_v2.py",
        "ecg_experiment/ssl_synthetic_ludb_landmark_audit_recovery.py",
        "ecg_experiment/ssl_synthetic_replication_saved_audit.py",
        "ecg_experiment/ssl_synthetic_suppression_saved_audit.py",
        "ecg_experiment/ssl_synthetic_pretrained_saved_audit.py",
    }
    require(required.issubset(admission["source_sha256"]), "complete independent source roster")
    for name, digest in admission["source_sha256"].items():
        require(file_sha256(source / name) == digest, "unchanged complete executable source: " + name)
    require(admission["prior_ledger"] == config["prior_ledger"], "frozen actual30 accounting anchor")
    old_admission, _ = previous.metadata_setup(
        root, source, "configs/ssl_synthetic_ludb_coherence_admission_v2.json"
    )
    old_audit = read_json(root / COHERENCE / "independent_audit_v1/summary.json")
    require(old_audit["all_passed"] and old_audit["mismatch_count"] == 0, "actual audit30 technical PASS")
    prior = read_json(root / old_admission["prior_ledger"]["path"])
    digest = old_admission["prior_ledger"]["sha256"]
    for filename in (
        COHERENCE + "/prospective_ledger.json",
        COHERENCE + "/independent_audit_v1/prospective_ledger.json",
    ):
        digest_next = file_sha256(root / filename)
        prior = next_stage(root, root / filename, digest_next, prior, digest)
        digest = digest_next
    require(
        prior["current_machine_stages"] == 30 and digest == config["prior_ledger"]["sha256"],
        "genuine immediate successful descriptor audit30",
    )
    for name in (
        "current_machine_stages",
        "current_machine_cumulative_cpu_seconds",
        "current_machine_cumulative_gpu_wall_seconds",
    ):
        require(prior[name] == config["prior_ledger"][name], "exact actual30 cumulative costs")
    for parent in (MODELS, TIMING, COHERENCE):
        summary = read_json(root / parent / "summary.json")
        for name, digest in admission["input_sha256"].items():
            prefix = parent + "/"
            if name.startswith(prefix):
                relative = name[len(prefix) :]
                if relative not in {"summary.json", "prospective_ledger.json"} and not relative.startswith(
                    "independent_audit"
                ):
                    require(
                        summary["output_sha256"].get(relative) == digest,
                        "original immutable parent catalog: " + name,
                    )
    require(TIMING + "/record_rows.jsonl" in admission["input_sha256"], "original N table admitted")
    assets = admission["assets"]
    require(
        assets["logical_root"] == config["raw_alias"]
        and assets["actual_root"] == config["actual_raw_target"]
        and os.path.samefile(root / assets["logical_root"], root / assets["actual_root"]),
        "same read-only-use local data alias",
    )
    qualified = read_json(root / TIMING / "release_file_identities.json")
    expected = {f"{identifier}.{extension}" for identifier in range(1, 201) for extension in ("dat", "hea")}
    require(
        set(assets["sha256"]) == expected
        and all(qualified[name] == digest for name, digest in assets["sha256"].items()),
        "same400 previously qualified physical files",
    )
    return admission, config


def input_identities(root: Any, admission: dict[str, Any], guard: Any) -> int:
    """Verify admitted local bytes and aliases before and after numerical reconstruction.

    Parameters
    ----------
    root : Any
        Artifact checkout containing local outputs and permitted raw aliases.
    admission : dict[str, Any]
        Declared frozen input values.
    guard : Any
        Callable enforcing the live measured resource gates.

    Returns
    -------
    int
        Independently reconstructed values or explicitly verified metadata.
    """
    import os

    from ecg_experiment.ssl_synthetic_pretrained_saved_audit import require
    from ecg_experiment.synthetic_audit_io import file_sha256

    for name, digest in admission["input_sha256"].items():
        guard()
        require(file_sha256(root / name, guard) == digest, "unchanged admitted predecessor: " + name)
    assets = admission["assets"]
    for name, digest in assets["sha256"].items():
        guard()
        actual, alias = root / assets["actual_root"] / name, root / assets["logical_root"] / name
        require(
            os.path.samefile(actual, alias) and file_sha256(actual, guard) == digest,
            "unchanged permitted physical asset",
        )
    return len(admission["input_sha256"]) + len(assets["sha256"])


def predecessor_closures(root: Any, study: Any, guard: Any) -> dict[str, Any]:
    """Reconstruct all old synthetic metrics and both real-study bootstrap arrays independently.

    Parameters
    ----------
    root : Any
        Artifact checkout containing local outputs and permitted raw aliases.
    study : Any
        Local main-output directory.
    guard : Any
        Callable enforcing the live measured resource gates.

    Returns
    -------
    dict[str, Any]
        Independently reconstructed values or explicitly verified metadata.
    """
    from ecg_experiment.ssl_synthetic_ludb_coherence_audit import summarize
    from ecg_experiment.ssl_synthetic_ludb_landmark_audit import aggregate as point_aggregate
    from ecg_experiment.ssl_synthetic_pretrained_saved_audit import require
    from ecg_experiment.ssl_synthetic_replication_saved_audit import (
        hotspot_rows,
        panel_controls,
        panel_metrics,
    )
    from ecg_experiment.ssl_synthetic_suppression_saved_audit import control_top_values, suppression_summary
    from ecg_experiment.synthetic_audit_io import compare_values, read_json

    parent = root / MODELS
    observed = np.load(parent / "fresh_observed.npy", mmap_mode="r", allow_pickle=False)
    support = np.load(parent / "fresh_support.npy", mmap_mode="r", allow_pickle=False)
    prior = np.load(
        root / "outputs/ssl_synthetic_localization/randomized_localization_v1/spatial_prior.npy",
        allow_pickle=False,
    )
    fits, checks = [], 0
    for fit in FITS:
        record = {"fit_id": fit}
        for arm in ARMS:
            guard()
            directory = parent / f"fit_{fit}" / arm
            probability = np.load(directory / "fresh_probabilities.npy", mmap_mode="r", allow_pickle=False)
            cases = panel_metrics(probability, observed, support, prior, list(range(2000, 2064)))
            hotspots = hotspot_rows(probability, support, list(range(2000, 2064)))
            means = {
                "sham_nuisance_persistence": float(
                    np.mean([row["sham_nuisance_persistence"] for row in hotspots])
                )
            }
            for mode in ("focal", "repeated"):
                means[mode] = {
                    key: float(np.mean([row[mode][key] for row in hotspots])) for key in hotspots[0][mode]
                }
            metrics = {
                "cases": cases,
                "controls": panel_controls(probability),
                "hotspots": {"cases": hotspots, "means": means},
                "top600_control": control_top_values(probability),
            }
            checks += compare_values(
                metrics, read_json(directory / "fresh_metrics.json"), "independent complete synthetic panel"
            )
            record[arm] = metrics
        fits.append(record)
    synthetic = suppression_summary(fits)
    checks += compare_values(synthetic, read_json(parent / "summary.json")["aggregate"], "old full aggregate")
    checks += compare_values(
        synthetic,
        read_json(study / "synthetic_predecessor_closure.json")["aggregate"],
        "new saved complete synthetic closure",
    )
    timing = point_aggregate(read_rows(root / TIMING / "record_rows.jsonl"))
    point_draws = timing.pop("bootstrap_gains")
    checks += compare_values(
        timing, read_json(root / TIMING / "summary.json")["statistics"], "independent exact timing closure", 0
    )
    checks += compare_values(
        timing, read_json(study / "timing_predecessor_closure.json")["statistics"], "new timing closure", 0
    )
    require(
        np.array_equal(point_draws, np.load(root / TIMING / "bootstrap_gains.npy", allow_pickle=False)),
        "all2000 original released-N draws",
    )
    coherence, coherence_draws = summarize(
        read_rows(root / COHERENCE / "generated_rows.jsonl"), read_rows(root / COHERENCE / "ludb_rows.jsonl")
    )
    checks += compare_values(
        coherence,
        read_json(root / COHERENCE / "summary.json")["statistics"],
        "independent coherence closure",
        0,
    )
    checks += compare_values(
        coherence,
        read_json(study / "coherence_predecessor_closure.json")["statistics"],
        "new coherence closure",
        0,
    )
    require(
        np.array_equal(
            coherence_draws,
            np.load(root / COHERENCE / "bootstrap_cohort_means_and_contrasts.npy", allow_pickle=False),
        ),
        "all2000 original coherence draws",
    )
    return {
        "scalar_checks": checks,
        "synthetic_model_panels": 12,
        "synthetic_case_rows": 1536,
        "timing_bootstrap_draws": 2000,
        "coherence_bootstrap_draws": 2000,
        "original_synthetic_scientific_failure_preserved": True,
    }


def numerical(root: Any, study: Any, config: dict[str, Any], guard: Any) -> dict[str, Any]:
    """Reconstruct all native scores and24 independent functional CPU forwards without GPU access.

    Parameters
    ----------
    root : Any
        Artifact checkout containing local outputs and permitted raw aliases.
    study : Any
        Local main-output directory.
    config : dict[str, Any]
        Frozen prospective scientific and resource declarations.
    guard : Any
        Callable enforcing the live measured resource gates.

    Returns
    -------
    dict[str, Any]
        Independently reconstructed values or explicitly verified metadata.
    """
    import torch

    from ecg_experiment.ssl_synthetic_ludb_audit import waveform
    from ecg_experiment.ssl_synthetic_pretrained_saved_audit import checkpoint_digest, require
    from ecg_experiment.synthetic_audit_io import compare_values, read_json

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    identifiers = [identifier for identifier in range(1, 201) if identifier % 5]
    require(read_json(study / "record_ids.json") == identifiers, "complete fixed160 record order")
    inputs = np.load(study / "inputs.npy", mmap_mode="r", allow_pickle=False)
    energy = np.load(study / "energy.npy", mmap_mode="r", allow_pickle=False)
    require(inputs.dtype == np.float32 and inputs.shape == (160, 12, 5000), "full float32 observed inputs")
    require(energy.dtype == np.float64 and energy.shape == (160, 8, 5000), "full float64 native energy")
    metadata = read_json(study / "input_metadata.json")
    require(len(metadata) == 160, "complete160 input metadata rows")
    scalar_checks = 0
    for position, identifier in enumerate(identifiers):
        guard()
        physical, signal = waveform(root / config["raw_alias"] / str(identifier))
        expected, transformation, expected_energy = prepare(physical)
        require(np.array_equal(expected, inputs[position]), "exact physical-to-training input")
        require(np.array_equal(expected_energy, energy[position]), "exact standardized native energy")
        scalar_checks += compare_values(
            {"record_id": identifier, "transformation": transformation, "signal_metadata": signal},
            metadata[position],
            "independent nonclinical input metadata",
            0,
        )
    structure = {row["record_id"]: row for row in read_rows(root / TIMING / "record_rows.jsonl")}
    saved = read_rows(study / "model_rows.jsonl")
    indexed = {(row["record_id"], row["fit_id"], row["arm"]): row for row in saved}
    require(len(saved) == len(indexed) == 1920, "complete1920 model-record rows")
    states = read_json(study / "model_states.json")
    parent = read_json(root / MODELS / "summary.json")
    reconstructed, replay_rows = [], []
    for fit in FITS:
        for arm in ARMS:
            guard()
            name = f"fit_{fit}_{arm}"
            probabilities = np.load(
                study / "probabilities" / (name + ".npy"), mmap_mode="r", allow_pickle=False
            )
            require(
                probabilities.shape == (160, 12, 5000) and probabilities.dtype == np.float32,
                "complete full-resolution float32 probability map",
            )
            require(
                np.isfinite(probabilities).all() and probabilities.min() >= 0 and probabilities.max() <= 1,
                "all twelve full output channels admitted without clipping",
            )
            state = torch.load(
                root / MODELS / f"fit_{fit}" / arm / "final_state.pt", map_location="cpu", weights_only=True
            )
            digest, count = checkpoint_digest(state)
            require(
                count == 35660 and digest == parent["final_state_sha256"][str(fit)][arm],
                "actual frozen checkpoint named-tensor identity",
            )
            require(
                states[name]["initial_tensor_state_sha256"] == digest
                and states[name]["final_tensor_state_sha256"] == digest,
                "saved unchanged model tensor state",
            )
            positions = [identifiers.index(identifier) for identifier in (1, 199)]
            replay = functional_replay(state, inputs[positions])
            difference = float(
                np.max(np.abs(replay.astype(np.float64) - probabilities[positions].astype(np.float64)))
            )
            require(difference <= 1e-5, "fixed independent CPU real replay absolute tolerance")
            replay_rows.append(
                {
                    "model": name,
                    "record_ids": [1, 199],
                    "maximum_absolute_difference": difference,
                    "tensor_state_sha256": digest,
                }
            )
            require(checkpoint_digest(state)[0] == digest, "CPU replay leaves frozen state unchanged")
            for position, identifier in enumerate(identifiers):
                guard()
                row = record_row(probabilities[position], energy[position], structure[identifier], fit, arm)
                scalar_checks += compare_values(
                    row, indexed[identifier, fit, arm], "independent native endpoint"
                )
                reconstructed.append(row)
    statistics, draws = aggregate(reconstructed, config)
    scalar_checks += compare_values(
        statistics, read_json(study / "summary.json")["statistics"], "independent descriptive reductions"
    )
    require(
        np.array_equal(draws, np.load(study / "bootstrap.npy", allow_pickle=False)),
        "exact all2000 shared record draws across four arms and paired contrast",
    )
    return {
        "scalar_checks": scalar_checks,
        "records": 160,
        "model_record_rows": 1920,
        "native_lead_rows": 15360,
        "bootstrap_draws": 2000,
        "cpu_functional_real_forwards": 24,
        "cpu_replays": replay_rows,
        "statistics": statistics,
    }


def profile_checks(root: Any, study: Any, config: dict[str, Any], guard: Any) -> dict[str, Any]:
    """Reconstruct profile forecasts, all saved old-map replays and exact real first batches.

    Parameters
    ----------
    root : Any
        Artifact checkout containing local outputs and permitted raw aliases.
    study : Any
        Local main-output directory.
    config : dict[str, Any]
        Frozen prospective scientific and resource declarations.
    guard : Any
        Callable enforcing the live measured resource gates.

    Returns
    -------
    dict[str, Any]
        Independently reconstructed values or explicitly verified metadata.
    """
    from ecg_experiment.ssl_synthetic_pretrained_saved_audit import require
    from ecg_experiment.synthetic_audit_io import compare_values, file_sha256, read_json

    profile = read_json(study / "profile/summary.json")
    require(
        profile["profile_scores_computed"] == 0 and profile["record_ids"] == config["profile"]["record_ids"],
        "unscored fixed actual-path profile",
    )
    require(
        profile["operations"]
        == {
            "checkpoint_loads": 12,
            "synthetic_forwards": 12,
            "real_batch8_forwards": 12,
            "real_record_reads": 8,
            "profile_figures": 1,
        },
        "complete prospective actual-path operations",
    )
    fixed, repeated, actual = profile["fixed_phase"], profile["repeated_phase"], profile["actual_at_forecast"]
    setup = {name: max(0.0, actual[name] - fixed[name] - repeated[name]) for name in fixed}
    checks = compare_values(setup, profile["setup_outside_phases"], "profile setup arithmetic", 0)
    forecast = {name: setup[name] + 2 * fixed[name] + 40 * repeated[name] + 180 for name in fixed}
    forecast["gpu_lock_wall_seconds"] = 2 * fixed["wall_seconds"] + 40 * repeated["wall_seconds"] + 30
    closures = {
        "synthetic_predecessor_closure.json",
        "timing_predecessor_closure.json",
        "coherence_predecessor_closure.json",
        "admission_snapshot.json",
    }
    before = sum(
        path.stat().st_size
        for path in study.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and (
            path.relative_to(study).as_posix() in closures
            or path.relative_to(study).parts[0] == "source_snapshot"
            or path.relative_to(study).parts[0] == "profile"
            and path.name != "summary.json"
        )
    )
    forecast["output_bytes"] = (
        before + 460800000 + 38400000 + 51200000 + 72 * 1024**2 + config["profile"]["output_reserve_bytes"]
    )
    checks += compare_values(forecast, profile["forecast"], "complete fixed/repeated forecast", 0)
    limits = config["resources"]
    gates = {
        name: forecast[name] < limits[name]
        for name in ("cpu_seconds", "wall_seconds", "gpu_lock_wall_seconds", "output_bytes")
    }
    gates["cuda_allocated_memory"] = (
        profile["peak_allocated_bytes_at_profile"] < limits["maximum_gpu_allocated_bytes"]
    )
    require(
        gates == profile["forecast_gates"] and all(gates.values()), "actual-path admission gate conjunction"
    )
    require(
        profile["gpu"]["free_bytes_at_admission"] >= limits["minimum_free_gpu_bytes"],
        "actual CUDA free memory admission",
    )
    states = read_json(study / "model_states.json")
    replays = {row["model"]: row for row in profile["synthetic_replays"]}
    inputs = np.load(study / "inputs.npy", mmap_mode="r", allow_pickle=False)
    require(
        np.array_equal(inputs[:8], np.load(study / "profile/inputs.npy", allow_pickle=False)),
        "exact retained input profile/full first batch",
    )
    require(
        read_json(study / "input_metadata.json")[:8] == read_json(study / "profile/input_metadata.json"),
        "exact profile/full structural metadata",
    )
    differences = {}
    for fit in FITS:
        for arm in ARMS:
            guard()
            name = f"fit_{fit}_{arm}"
            replay = np.load(study / "profile" / (name + "_synthetic_replay.npy"), allow_pickle=False)
            reference = np.load(
                root / MODELS / f"fit_{fit}" / arm / "fresh_probabilities.npy",
                mmap_mode="r",
                allow_pickle=False,
            )[0]
            require(
                replay.dtype == np.float32 and replay.shape == (4, 12, 5000) and np.isfinite(replay).all(),
                "complete synthetic replay evidence",
            )
            difference = float(np.max(np.abs(replay.astype(np.float64) - reference.astype(np.float64))))
            row = replays[name]
            require(
                difference <= 1e-6
                and row["maximum_absolute_difference"] == difference
                and row["initial_tensor_state_sha256"] == states[name]["initial_tensor_state_sha256"]
                and row["all_passed"]
                and row["atol"] == 1e-6
                and row["rtol"] == 0,
                "prospective old CUDA map replay tolerance and original state",
            )
            require(
                read_json(study / "profile" / (name + "_synthetic_replay.json")) == row,
                "retained per-model replay receipt",
            )
            ppath = study / "profile" / (name + ".npy")
            fullpath = study / "probabilities" / (name + ".npy")
            require(
                file_sha256(ppath) == profile["profile_map_sha256"][name]
                and file_sha256(fullpath) == states[name]["probabilities_sha256"],
                "saved profile/full map hashes",
            )
            require(
                np.array_equal(
                    np.load(ppath, mmap_mode="r", allow_pickle=False),
                    np.load(fullpath, mmap_mode="r", allow_pickle=False)[:8],
                )
                and states[name]["profile_first_batch_exact"]
                and states[name]["full_batches"] == 20
                and states[name]["unchanged"],
                "exact fixed profile versus full first batch",
            )
            differences[name] = difference
    require(
        file_sha256(study / "profile/profile_fixed_maps.png") == profile["figure_sha256"],
        "profile fixed figure identity",
    )
    return {
        "scalar_checks": checks,
        "synthetic_replays": 12,
        "exact_profile_batches": 12,
        "old_map_maximum_absolute_differences": differences,
        "forecast": forecast,
        "profile_retained_bytes_before_summary": before,
    }


def main_accounting(
    root: Any, study: Any, admission: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Validate actual GPU-main additive costs and the immutable output catalog before own scoring.

    Parameters
    ----------
    root : Any
        Artifact checkout containing local outputs and permitted raw aliases.
    study : Any
        Local main-output directory.
    admission : dict[str, Any]
        Declared frozen input values.
    config : dict[str, Any]
        Frozen prospective scientific and resource declarations.

    Returns
    -------
    dict[str, Any]
        Independently reconstructed values or explicitly verified metadata.
    """
    from ecg_experiment.ssl_synthetic_pretrained_saved_audit import require
    from ecg_experiment.synthetic_audit_io import file_sha256, read_json

    main = read_json(study / "summary.json")
    path = study / "prospective_ledger.json"
    ledger = read_json(path)
    prior = admission["prior_ledger"]
    require(
        ledger["receipt_sha256"] == file_sha256(study / "summary.json")
        and ledger["prior_ledger_sha256"] == prior["sha256"]
        and ledger["prior_accounting_verified"]
        and ledger["current_machine_stages"] == 31,
        "actual main31 receipt and immediately verified prior30",
    )
    for suffix in ("cpu_seconds", "gpu_wall_seconds"):
        key = "current_machine_cumulative_" + suffix
        require(
            ledger[key] == prior[key] + ledger[suffix] and ledger[suffix] >= 0,
            "exact once-only GPU-main actual costs",
        )
    require(
        ledger["technical_passed"]
        and ledger["status"] == "PASS"
        and all(ledger["resource_gates"].values())
        and main["status"] == "PASS"
        and main["technical_passed_before_independent_audit"],
        "completed main31 technical resource conjunction",
    )
    limits = config["resources"]
    require(
        ledger["cpu_seconds"] < limits["cpu_seconds"]
        and ledger["wall_seconds"] < limits["wall_seconds"]
        and ledger["gpu_wall_seconds"] < limits["gpu_lock_wall_seconds"]
        and ledger["peak_rss_bytes"] < limits["peak_rss_bytes"],
        "main measured resource ceilings",
    )
    files = {
        path.relative_to(study).as_posix(): path
        for path in study.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and "independent_audit_v1" not in path.relative_to(study).parts
    }
    require(
        set(files) == set(main["output_sha256"]) | {"summary.json", "prospective_ledger.json"},
        "complete main retained file catalog",
    )
    require(
        sum(path.stat().st_size for name, path in files.items() if name != "prospective_ledger.json")
        == ledger["output_bytes_before_terminal_ledger"],
        "main byte arithmetic excludes only its terminal ledger",
    )
    require(
        ledger["output_bytes_before_terminal_ledger"] + 16384 < limits["output_bytes"],
        "main retained byte reserve",
    )
    return {
        "prior_ledger_sha256": file_sha256(path),
        "cpu_through_main": ledger["current_machine_cumulative_cpu_seconds"],
        "gpu_through_main": ledger["current_machine_cumulative_gpu_wall_seconds"],
        "stages_through_main": 31,
    }


def output_identities(
    study: Any, source: Any, admission: dict[str, Any], admission_name: str, guard: Any
) -> int:
    """Verify every main output, retained source snapshot and final admission identity.

    Parameters
    ----------
    study : Any
        Local main-output directory.
    source : Any
        Isolated checkout containing the committed executable sources.
    admission : dict[str, Any]
        Declared frozen input values.
    admission_name : str
        Repository-relative committed identity admission.
    guard : Any
        Callable enforcing the live measured resource gates.

    Returns
    -------
    int
        Independently reconstructed values or explicitly verified metadata.
    """
    from ecg_experiment.ssl_synthetic_pretrained_saved_audit import require
    from ecg_experiment.synthetic_audit_io import file_sha256, read_json

    main = read_json(study / "summary.json")
    expected_sources = {**admission["source_sha256"], admission_name: file_sha256(source / admission_name)}
    require(main["source_sha256"] == expected_sources, "actual complete admitted executable roster")
    for name, digest in admission["source_sha256"].items():
        guard()
        require(
            file_sha256(study / "source_snapshot" / name) == digest and file_sha256(source / name) == digest,
            "same committed and snapshotted executable source",
        )
    require(
        file_sha256(study / "admission_snapshot.json") == expected_sources[admission_name],
        "actual main identity admission snapshot",
    )
    for name, digest in main["output_sha256"].items():
        guard()
        require(file_sha256(study / name, guard) == digest, "immutable generated diagnostic output: " + name)
    return len(main["output_sha256"]) + len(expected_sources)


def run(
    root: Any,
    source: Any,
    admission_name: str,
    main_sha: str,
    entry_sha: str,
    runner_sha: str,
    clock: dict[str, float],
) -> dict[str, Any]:
    """Run one fully admitted independent CPU audit and preserve actual failure and measured cost.

    Parameters
    ----------
    root : Any
        Artifact checkout containing local outputs and permitted raw aliases.
    source : Any
        Isolated checkout containing the committed executable sources.
    admission_name : str
        Repository-relative committed identity admission.
    main_sha : str
        Literal SHA256 of the completed main summary.
    entry_sha : str
        Literal SHA256 of the independent CLI.
    runner_sha : str
        Literal SHA256 of this independent module.
    clock : dict[str, float]
        Whole-audit CPU and monotonic wall start values.

    Returns
    -------
    dict[str, Any]
        Independently reconstructed values or explicitly verified metadata.
    """
    import os
    import resource
    import signal
    import time
    from pathlib import Path

    from ecg_experiment.ssl_synthetic_pretrained_saved_audit import require
    from ecg_experiment.synthetic_audit_io import file_sha256, read_json

    def write(path: Any, value: dict[str, Any]) -> None:
        import json

        with path.open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")

    config = read_json(source / CONFIG)
    study = root / config["output_root"]
    destination = study / "independent_audit_v1"
    destination.mkdir(exist_ok=False)
    costs, timers = {}, {}

    def guard() -> None:
        require(
            time.process_time() - clock["cpu"] < 898 and time.perf_counter() - clock["wall"] < 1198,
            "independent localizer receipt time reserve",
        )
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 < 2 * 1024**3, "independent RSS")

    def interrupted(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Independent localizer timer {signum}")

    try:
        for timer, signum, ceiling, kind in (
            (signal.ITIMER_REAL, signal.SIGALRM, 1198, "wall"),
            (signal.ITIMER_PROF, signal.SIGPROF, 898, "cpu"),
        ):
            timers[timer] = signum, signal.getsignal(signum)
            signal.signal(signum, interrupted)
            elapsed = (time.process_time() if kind == "cpu" else time.perf_counter()) - clock[kind]
            signal.setitimer(timer, max(0.01, ceiling - elapsed))
        require(
            os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(
                os.environ.get(name) == "1"
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            ),
            "hidden CUDA and four single-thread runtime limits",
        )
        require(
            file_sha256(Path(__file__)) == runner_sha and file_sha256(source / CHECKER) == entry_sha,
            "literal independent executable identities",
        )
        admission, config = metadata_setup(root, source, admission_name)
        require(file_sha256(study / "summary.json") == main_sha, "literal actual main31 summary")
        costs = main_accounting(root, study, admission, config)
        own_sources = {**admission["source_sha256"], admission_name: file_sha256(source / admission_name)}
        for name in own_sources:
            target = destination / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((source / name).read_bytes())
        identity_checks = input_identities(root, admission, guard)
        identity_checks += output_identities(study, source, admission, admission_name, guard)
        predecessors = predecessor_closures(root, study, guard)
        profile = profile_checks(root, study, config, guard)
        values = numerical(root, study, config, guard)
        main = read_json(study / "summary.json")
        require(
            main["generator_forwards"]
            == main["optimizer_updates"]
            == main["original_annotation_stream_reads"]
            == main["parser_records_scored"]
            == 0
            and main["real_record_reads"] == 168
            and main["clinical_claim"] is False
            and main["no_training"] is True,
            "fixed nonclinical inference-only operation scope",
        )
        require(
            "scientific_passed" not in main and "scientific_passed" not in main["statistics"],
            "descriptive localizer diagnostic has no post-hoc scientific-success threshold",
        )
        from PIL import Image

        for name, dimensions in (
            ("profile/profile_fixed_maps.png", (1920, 960)),
            ("all_model_landmark_and_activity.png", (2160, 720)),
            ("fixed_input_map_N_overlays.png", (2160, 2400)),
        ):
            with Image.open(study / name) as figure:
                require(figure.format == "PNG" and figure.size == dimensions, "fixed figure signature/layout")
        identity_checks += input_identities(root, admission, guard)
        identity_checks += output_identities(study, source, admission, admission_name, guard)
        guard()
        result = {
            "all_passed": True,
            "mismatch_count": 0,
            "main_summary_sha256": main_sha,
            "source_sha256": own_sources,
            "predecessors": predecessors,
            "profile": profile,
            "numerical": values,
            "identity_checks": identity_checks,
            "clinical_claim": False,
            "gpu_used": False,
            "new_annotation_stream_reads": 0,
            "figure_scope": "hashes, PNG signatures and fixed dimensions; no pixel rerender",
        }
    except Exception as error:
        result = {
            "all_passed": False,
            "mismatch_count": 1,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    try:
        write(destination / "summary.json", result)
        digest = file_sha256(destination / "summary.json")
        retained = sum(
            path.stat().st_size for path in destination.rglob("*") if path.is_file() and not path.is_symlink()
        )
        cpu, wall = time.process_time() - clock["cpu"], time.perf_counter() - clock["wall"]
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        gates = {
            "cpu": cpu < 900,
            "wall": wall < 1200,
            "rss": rss < 2 * 1024**3,
            "output": retained + 16384 < 128 * 1024**2,
        }
        verified = bool(costs)
        ledger = {
            "stage": "ludb_localizer_independent_audit_v1",
            "receipt": "summary.json",
            "sha256": digest,
            "cpu_seconds": cpu,
            "wall_seconds": wall,
            "gpu_wall_seconds": 0.0,
            "peak_rss_bytes": rss,
            "resource_gates": gates,
            "prior_accounting_verified": verified,
            "prior_ledger_sha256": costs.get("prior_ledger_sha256"),
            "current_machine_stages": costs["stages_through_main"] + 1 if verified else None,
            "current_machine_cumulative_cpu_seconds": costs["cpu_through_main"] + cpu if verified else None,
            "current_machine_cumulative_gpu_wall_seconds": costs.get("gpu_through_main"),
            "output_bytes_before_terminal_ledger": retained,
            "historical_costs_verified": False,
            "measurement_endpoint": "after receipt hash and byte enumeration, before ledger serialization",
            "terminal_bookkeeping_reserve_seconds": 2,
            "terminal_bookkeeping_reserve_bytes": 16384,
            "technical_passed": result["all_passed"] and all(gates.values()),
        }
        for timer in timers:
            signal.setitimer(timer, 0)
        write(destination / "prospective_ledger.json", ledger)
    finally:
        for timer, (signum, handler) in timers.items():
            signal.setitimer(timer, 0)
            signal.signal(signum, handler)
    require(result["all_passed"] and all(gates.values()), "retained independent localizer failure")
    return result
