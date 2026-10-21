"""Describe lead ranking and oracle-conditioned temporal ranking using saved maps only."""

import json
import os
import resource
import signal
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from ecg_experiment import ROOT
from ecg_experiment import ssl_synthetic_paired_suppression as parent_metrics
from ecg_experiment import ssl_synthetic_pretrained_generator as shared
from ecg_experiment.intervals import patient_groups, patient_resample
from ecg_experiment.paths import from_stored, to_stored
from ecg_experiment.synthetic_audit_io import compare_values, file_sha256, read_json, relative_file

CONFIG = "configs/ssl_synthetic_temporal_diagnosis_v1.json"
ADMISSION = "configs/ssl_synthetic_temporal_diagnosis_admission_v1.json"
PROTOCOL = "docs/ssl-synthetic-localization-temporal-diagnosis-v1.md"
ENDPOINTS = ("active_lead_temporal_AP", "lead_presence_AP")
MODES = {"focal": 2, "repeated": 3}
ARMS = ("B", "P", "K", "PK")
SOURCES = (
    CONFIG,
    ADMISSION,
    PROTOCOL,
    "docs/ssl-synthetic-localization-temporal-diagnosis-admission-v1.md",
    "ecg_experiment/ssl_synthetic_lead_time_diagnostic.py",
    "scripts/reports/diagnose_ssl_synthetic_lead_time.py",
    "tests/test_ssl_synthetic_lead_time_diagnostic.py",
    "ecg_experiment/ssl_synthetic_paired_suppression.py",
    "ecg_experiment/ssl_synthetic_pretrained_generator.py",
    "ecg_experiment/intervals.py",
    "ecg_experiment/paths.py",
    "ecg_experiment/synthetic_audit_io.py",
    "ecg_experiment/ssl_synthetic_counterfactual_factorial_training.py",
    "ecg_experiment/ssl_synthetic_paired_background_training.py",
    "ecg_experiment/ssl_synthetic_model.py",
    "ecg_experiment/ssl_synthetic_model_v2.py",
    "ecg_experiment/ssl_synthetic_preparation.py",
    "ecg_experiment/ssl_synthetic_training.py",
    "ecg_experiment/ssl_synthetic_real_transfer.py",
    "ecg_experiment/gpu.py",
    "ecg_experiment/files.py",
)


def score_case(probability: np.ndarray, support: np.ndarray) -> dict[str, Any]:
    """Score temporal ranking separately within active leads and presence ranking over twelve leads.

    Parameters
    ----------
    probability : np.ndarray
        One positive variant's native float32 map with shape(12,5000).
    support : np.ndarray
        Its final Boolean numerical intervention support, with identical geometry.

    Returns
    -------
    dict
        Active canonical lead indices, every active-lead AP, their equal mean and lead-presence AP.
    """
    if (
        probability.shape != (12, 5000)
        or support.shape != probability.shape
        or probability.dtype != np.float32
        or support.dtype != bool
    ):
        raise ValueError("Expected one native float32 map and Boolean support with shape(12,5000)")
    if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
        raise ValueError("Saved probability values must be finite and within0–1")
    active = support.any(axis=1)
    if not 0 < int(active.sum()) < 12:
        raise ValueError("Lead-presence target must contain both classes")
    leads = np.flatnonzero(active)
    values = []
    for lead in leads:
        if not 0 < int(support[lead].sum()) < 5000:
            raise ValueError("Every active temporal lead must contain both target classes")
        values.append(float(average_precision_score(support[lead], probability[lead].astype(np.float64))))
    presence = float(average_precision_score(active, probability.astype(np.float64).max(axis=1)))
    return {
        "active_leads": leads.tolist(),
        "active_lead_AP": values,
        "active_lead_temporal_AP": float(np.mean(values)),
        "lead_presence_AP": presence,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Use shared crossed resampling for six descriptive PK-minus-B contrasts.

    Parameters
    ----------
    rows : list
        All1536 fixed fit/arm/background/positive-mode records in declared order.

    Returns
    -------
    dict
        Six exploratory intervals, all arm/fit means and each contrast's three fit estimates/range.
    """
    keys = [
        (fit, arm, index, mode)
        for fit in [1, 2, 3]
        for arm in ARMS
        for index in range(2000, 2064)
        for mode in MODES
    ]
    actual = [(row["fit_id"], row["arm"], row["index"], row["mode"]) for row in rows]
    if actual != keys or any(row["seed"] != 2026103000 + row["index"] for row in rows):
        raise ValueError("Diagnostic rows must preserve all fixed identities and their exact order")
    lookup = {(row["fit_id"], row["arm"], row["index"], row["mode"]): row for row in rows}
    endpoints = {arm: {} for arm in ARMS}
    means = {str(fit): {arm: {} for arm in ARMS} for fit in [1, 2, 3]}
    for arm in ARMS:
        for endpoint in ENDPOINTS:
            for mode in MODES:
                value = np.array(
                    [
                        [lookup[(fit, arm, index, mode)][endpoint] for index in range(2000, 2064)]
                        for fit in [1, 2, 3]
                    ],
                    dtype=np.float64,
                )
                if (
                    value.shape != (3, 64)
                    or not np.isfinite(value).all()
                    or not ((value >= 0) & (value <= 1)).all()
                ):
                    raise ValueError("Endpoint values must be complete finite APs")
                endpoints[arm][(endpoint, mode)] = value
            endpoints[arm][(endpoint, "pooled")] = (
                endpoints[arm][(endpoint, "focal")] + endpoints[arm][(endpoint, "repeated")]
            ) / 2
        for fit in [1, 2, 3]:
            means[str(fit)][arm] = {
                mode: {
                    endpoint: float(endpoints[arm][(endpoint, mode)][fit - 1].mean())
                    for endpoint in ENDPOINTS
                }
                for mode in [*MODES, "pooled"]
            }
    vectors = {
        f"PK_minus_B__{endpoint}__{mode}": endpoints["PK"][(endpoint, mode)]
        - endpoints["B"][(endpoint, mode)]
        for endpoint in ENDPOINTS
        for mode in [*MODES, "pooled"]
    }
    samples = {key: np.empty(2000, dtype=np.float64) for key in vectors}
    groups = patient_groups(np.arange(64))
    rng = np.random.default_rng(2026105601)
    for draw in range(2000):
        fits = rng.integers(0, 3, 3)
        backgrounds = patient_resample(groups, rng)
        for key, value in vectors.items():
            samples[key][draw] = value[fits[:, None], backgrounds[None, :]].mean()
    contrasts = {
        key: {
            "mean": float(value.mean()),
            "ci_low": float(np.quantile(samples[key], 0.025, method="linear")),
            "ci_high": float(np.quantile(samples[key], 0.975, method="linear")),
            "draws": 2000,
            "seed": 2026105601,
            "skipped_draws": 0,
            "fit_count": 3,
            "background_count": 64,
        }
        for key, value in vectors.items()
    }
    ranges = {
        key: {
            "fit_means": value.mean(axis=1).tolist(),
            "minimum": float(value.mean(axis=1).min()),
            "maximum": float(value.mean(axis=1).max()),
        }
        for key, value in vectors.items()
    }
    return {"descriptive_contrasts": contrasts, "arm_fit_means": means, "fit_contrast_ranges": ranges}


def load_inputs(parent: Path) -> tuple[np.ndarray, dict[int, Any], dict[int, Any]]:
    """Open only the29 permitted parent artifacts, mapping prediction/support arrays read-only."""
    support = np.load(parent / "fresh_support.npy", mmap_mode="r", allow_pickle=False)
    if support.shape != (64, 4, 12, 5000) or support.dtype != bool:
        raise ValueError("Saved parent support geometry differs")
    maps, metrics = {}, {}
    for fit in [1, 2, 3]:
        maps[fit], metrics[fit] = {}, {}
        for arm in ARMS:
            directory = parent / f"fit_{fit}" / arm
            value = np.load(directory / "fresh_probabilities.npy", mmap_mode="r", allow_pickle=False)
            if value.shape != (64, 4, 12, 5000) or value.dtype != np.float32:
                raise ValueError("Saved parent map geometry differs")
            maps[fit][arm] = value
            metrics[fit][arm] = read_json(directory / "fresh_metrics.json")
    return support, maps, metrics


def predecessor_closure(
    parent: Path,
    support: np.ndarray,
    maps: dict[int, Any],
    metrics: dict[int, Any],
    destination: Path,
    guard: Any,
) -> dict[str, Any]:
    """Exactly close every full-grid positive AP and the complete frozen parent aggregate.

    Parameters
    ----------
    parent : Path
        Explicit audited four-arm study root.
    support, maps, metrics : objects
        Read-only admitted support/maps and their bound metric records.
    destination : Path
        Fresh diagnostic output directory.
    guard : callable
        Resource guard between case calculations.

    Returns
    -------
    dict
        Exact1536-case closure, full parent aggregate and preserved scientific decision.
    """
    count = 0
    for fit in [1, 2, 3]:
        for arm in ARMS:
            saved = metrics[fit][arm]["cases"]
            if len(saved) != 64:
                raise ValueError("Parent case metric rows incomplete")
            for offset, index in enumerate(range(2000, 2064)):
                guard()
                if saved[offset]["index"] != index or saved[offset]["seed"] != 2026103000 + index:
                    raise ValueError("Parent metric identities differ")
                for mode, position in MODES.items():
                    actual = float(
                        average_precision_score(
                            support[offset, position].ravel(),
                            maps[fit][arm][offset, position].astype(np.float64).ravel(),
                        )
                    )
                    if actual != saved[offset][mode]["edited_ap"]:
                        raise ValueError("Exact parent full-grid edited AP closure differs")
                    count += 1
    fits = [{"fit_id": fit, **metrics[fit]} for fit in [1, 2, 3]]
    aggregate = parent_metrics.aggregate(fits)
    main = read_json(parent / "summary.json")
    compare_values(aggregate, main["aggregate"], "full parent aggregate closure", tolerance=0)
    receipt = {
        "all_exact": True,
        "full_grid_edited_AP_checks": count,
        "aggregate": aggregate,
        "parent_scientific_passed": aggregate["scientific_passed"],
    }
    shared.write_new_json(destination / "predecessor_closure.json", receipt)
    return receipt


def make_figure(destination: Path, means: dict[str, Any]) -> list[str]:
    """Plot every arm and fit on the fixed two-endpoint by two-mode AP grid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    name = "lead_and_time_diagnostics.png"
    if (destination / name).exists():
        raise FileExistsError(name)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fits = [1, 2, 3]
    for row, endpoint in enumerate(ENDPOINTS):
        for column, mode in enumerate(MODES):
            axis = axes[row, column]
            for arm in ARMS:
                axis.plot(fits, [means[str(fit)][arm][mode][endpoint] for fit in fits], "o-", label=arm)
            axis.set(
                title=f"{mode}: {endpoint}", xlabel="Fixed fit ID", ylabel="AP", ylim=(0, 1), xticks=fits
            )
            axis.legend()
    fig.suptitle("Saved-map diagnosis; temporal AP is conditional on known active leads")
    fig.tight_layout()
    fig.savefig(destination / name, dpi=120)
    plt.close(fig)
    return [name]


def snapshot(destination: Path, guard: Any) -> dict[str, str]:
    """Snapshot all declared source bytes before parent admission or diagnostic calculations."""
    catalog = {}
    for name in SOURCES:
        guard()
        target = destination / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(from_stored(name).read_bytes())
        catalog[name] = file_sha256(target, guard)
    return catalog


def admit(
    root: Path, config: dict[str, Any], admission: dict[str, Any], inputs: dict[str, str], guard: Any
) -> Path:
    """Admit exactly the declared parent allowlist and its audited terminal accounting."""
    if admission["frozen_config_sha256"] != file_sha256(from_stored(CONFIG), guard) or admission[
        "frozen_protocol_sha256"
    ] != file_sha256(from_stored(PROTOCOL), guard):
        raise ValueError("Frozen temporal diagnostic design identity differs")
    parent = admission["parent"]
    if parent["root"] != config["parent_root"] or set(parent["sha256"]) != set(
        config["allowed_parent_files"]
    ):
        raise ValueError("Diagnostic input allowlist differs")
    path = relative_file(root, parent["root"])
    for name, expected in parent["sha256"].items():
        if file_sha256(relative_file(path, name), guard) != expected:
            raise ValueError("Parent diagnostic input identity mismatch")
        inputs[name] = expected
    main = read_json(path / "summary.json")
    audit = read_json(path / "independent_audit_v1/summary.json")
    design = config["parent_scientific_design"]
    if any(
        main["source_sha256"][design[kind]] != design[kind + "_sha256"] for kind in ["config", "protocol"]
    ):
        raise ValueError("Parent frozen scientific design identity differs")
    if not all(main["integrity_gates"].values()) or not audit["all_passed"] or audit["mismatch_count"] != 0:
        raise ValueError("Parent technical integrity/audit did not pass")
    prior = admission["prior_ledger"]
    if (
        relative_file(root, prior["path"]) != path / "independent_audit_v1/prospective_ledger.json"
        or file_sha256(relative_file(root, prior["path"]), guard) != prior["sha256"]
    ):
        raise ValueError("Immediate parent terminal ledger identity differs")
    ledger = read_json(relative_file(root, prior["path"]))
    for key in ["current_machine_cumulative_cpu_seconds", "current_machine_cumulative_gpu_wall_seconds"]:
        if ledger[key] != prior[key]:
            raise ValueError("Immediate audited cumulative cost differs")
    return path


def verify(
    destination: Path, parent: Path, sources: dict[str, str], inputs: dict[str, str], guard: Any
) -> None:
    """Rehash sources, snapshots and all29 admitted parent files without reading another payload."""
    for name, expected in sources.items():
        if (
            file_sha256(from_stored(name), guard) != expected
            or file_sha256(destination / "source_snapshot" / name, guard) != expected
        ):
            raise ValueError("Diagnostic source or snapshot changed")
    for name, expected in inputs.items():
        if file_sha256(relative_file(parent, name), guard) != expected:
            raise ValueError("Diagnostic admitted parent input changed")


def finish_ledger(
    destination: Path,
    root: Path,
    admission: dict[str, Any],
    receipt: str,
    clock: dict[str, float],
    prior_verified: bool,
) -> None:
    """Charge success or failure once after the actual receipt hash, without rounding prior costs."""
    digest = file_sha256(destination / receipt)
    prior = admission["prior_ledger"]
    if prior_verified and file_sha256(relative_file(root, prior["path"])) != prior["sha256"]:
        raise ValueError("Audited parent ledger changed before terminal accounting")
    size = shared.directory_bytes(destination)
    cpu, wall = time.process_time() - clock["cpu"], time.perf_counter() - clock["wall"]
    row = {
        "stage": "generated_temporal_diagnosis_v1",
        "receipt": receipt,
        "receipt_sha256": digest,
        "cpu_seconds": cpu,
        "wall_seconds": wall,
        "gpu_wall_seconds": 0.0,
        "prior_ledger_sha256": prior["sha256"] if prior_verified else None,
        "prior_current_machine_cpu_seconds": prior["current_machine_cumulative_cpu_seconds"]
        if prior_verified
        else None,
        "prior_current_machine_gpu_wall_seconds": prior["current_machine_cumulative_gpu_wall_seconds"]
        if prior_verified
        else None,
        "current_machine_cumulative_cpu_seconds": prior["current_machine_cumulative_cpu_seconds"] + cpu
        if prior_verified
        else None,
        "current_machine_cumulative_gpu_wall_seconds": prior["current_machine_cumulative_gpu_wall_seconds"]
        if prior_verified
        else None,
        "prior_accounting_verified": prior_verified,
        "cpu_budget_passed": cpu < 180,
        "wall_budget_passed": wall < 240,
        "output_bytes_before_terminal_ledger": size,
        "historical_costs_verified": False,
        "measurement_endpoint": "after receipt/prior hashes and byte count, before ledger serialization",
    }
    shared.write_new_json(destination / "prospective_ledger.json", row)
    if (
        not row["cpu_budget_passed"]
        or not row["wall_budget_passed"]
        or shared.directory_bytes(destination) >= 16 * 1024**2
    ):
        raise ValueError("Terminal diagnostic resource ceiling exceeded")


def score_rows(
    support: np.ndarray,
    maps: dict[int, Any],
    destination: Path,
    rows: list[dict[str, Any]],
    context: dict[str, Any],
    guard: Any,
) -> None:
    """Persist every fixed positive-case endpoint row, retaining the offending identity on failure."""
    with (destination / "cases.jsonl").open("x") as stream:
        for fit in [1, 2, 3]:
            for arm in ARMS:
                for offset, index in enumerate(range(2000, 2064)):
                    for mode, position in MODES.items():
                        guard()
                        context["current_case"] = {
                            "fit_id": fit,
                            "arm": arm,
                            "index": index,
                            "seed": 2026103000 + index,
                            "mode": mode,
                        }
                        row = {
                            **context["current_case"],
                            **score_case(maps[fit][arm][offset, position], support[offset, position]),
                        }
                        rows.append(row)
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                        stream.flush()


def setup_runtime(timers: dict[int, Any], interrupted: Any) -> None:
    """Require hidden CUDA, one numerical thread and exclusive finite CPU/wall timers."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or any(
        os.environ.get(key) != "1"
        for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]
    ):
        raise ValueError("Diagnostic requires hidden CUDA and one numerical thread")
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    for timer, signum, seconds in [
        (signal.ITIMER_REAL, signal.SIGALRM, 238),
        (signal.ITIMER_PROF, signal.SIGPROF, 178),
    ]:
        if signal.getitimer(timer) != (0.0, 0.0):
            raise ValueError("Diagnostic requires exclusive process timers")
        timers[timer] = (signum, signal.getsignal(signum))
        signal.signal(signum, interrupted)
        signal.setitimer(timer, seconds)


def run_study(output_root: Path) -> dict[str, Any]:
    """Execute one source-frozen saved-map diagnostic with no neural operations.

    Parameters
    ----------
    output_root : Path
        Explicit local host root for the parent and fresh sibling output directory.

    Returns
    -------
    dict
        Descriptive endpoint/contrast results, exact closure and technical integrity evidence.
    """
    clock = {"cpu": time.process_time(), "wall": time.perf_counter()}
    config = read_json(from_stored(CONFIG))
    admission = read_json(from_stored(ADMISSION))
    destination = output_root / config["output_root"]
    destination.mkdir(exist_ok=False)
    sources, inputs, rows = {}, {}, []
    context: dict[str, Any] = {"current_case": None}
    parent = output_root / config["parent_root"]
    prior_verified = False
    timers = {}

    def guard() -> None:
        if time.process_time() - clock["cpu"] + 2 >= 180 or time.perf_counter() - clock["wall"] + 2 >= 240:
            raise TimeoutError("Temporal diagnostic CPU/wall reserve reached")
        if shared.directory_bytes(destination) + 1024**2 >= 16 * 1024**2:
            raise ValueError("Temporal diagnostic output reserve exhausted")

    def interrupted(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Temporal diagnostic timer {signum}")

    try:
        setup_runtime(timers, interrupted)
        sources = snapshot(destination, guard)
        parent = admit(output_root, config, admission, inputs, guard)
        prior_verified = True
        verify(destination, parent, sources, inputs, guard)
        support, maps, metrics = load_inputs(parent)
        closure = predecessor_closure(parent, support, maps, metrics, destination, guard)
        verify(destination, parent, sources, inputs, guard)
        score_rows(support, maps, destination, rows, context, guard)
        statistics = summarize(rows)
        figures = make_figure(destination, statistics["arm_fit_means"])
        verify(destination, parent, sources, inputs, guard)
        guard()
        result = {
            "study_id": config["study_id"],
            "created_at_utc": datetime.now(UTC).isoformat(),
            "output_relative": to_stored(ROOT / config["output_root"]),
            **statistics,
            "predecessor_closure": closure,
            "integrity_gates": {
                "parent_full_grid_AP_exact": True,
                "parent_full_aggregate_exact": True,
                "fixed_complete_rows": len(rows) == 1536,
                "sources_and_inputs_unchanged": True,
            },
            "technical_passed": True,
            "source_sha256": sources,
            "input_sha256": inputs,
            "figures": figures,
            "case_rows": len(rows),
            "operations": config["allowed_operations"],
            "gpu_used": False,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "runtime_environment": {
                key: os.environ.get(key)
                for key in [
                    "CUDA_VISIBLE_DEVICES",
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                ]
            },
            "oracle_conditioning": "active-lead oracle at scoring only; not deployable performance",
            "derived_lead_dependence": "equal canonical-lead weights do not imply physical independence",
            "clinical_claim": False,
            "historical_costs_verified": False,
            "cpu_seconds": time.process_time() - clock["cpu"],
            "wall_seconds": time.perf_counter() - clock["wall"],
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "output_bytes_before_receipt": shared.directory_bytes(destination),
            "output_sha256": {
                path.relative_to(destination).as_posix(): file_sha256(path, guard)
                for path in destination.rglob("*")
                if path.is_file()
            },
        }
        guard()
        shared.write_new_json(destination / "summary.json", result)
        guard()
        finish_ledger(destination, output_root, admission, "summary.json", clock, prior_verified)
        return result
    except Exception as error:
        shared.write_new_json(
            destination / "failure.json",
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "current_or_offending_case": context["current_case"],
                "completed_case_rows": len(rows),
                "source_sha256": sources,
                "input_sha256": inputs,
                "cpu_seconds": time.process_time() - clock["cpu"],
                "wall_seconds": time.perf_counter() - clock["wall"],
                "operations": config["allowed_operations"],
                "gpu_used": False,
                "historical_costs_verified": False,
            },
        )
        finish_ledger(destination, output_root, admission, "failure.json", clock, prior_verified)
        raise
    finally:
        for timer, (signum, handler) in timers.items():
            signal.setitimer(timer, 0)
            signal.signal(signum, handler)
