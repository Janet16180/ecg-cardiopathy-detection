"""Measure only the separately admitted fixed saved-output illustration step."""

import resource
import struct
from pathlib import Path
from typing import Any

from ecg_experiment import ssl_synthetic_background_editor_transfer_figures as figures
from ecg_experiment import ssl_synthetic_ludb_localizer_run as runtime
from ecg_experiment import ssl_synthetic_ludb_run as shared
from ecg_experiment.files import sha256_file
from ecg_experiment.paths import from_stored

CONFIG = "configs/ssl_synthetic_background_editor_transfer_figures_v1.json"
PROTOCOL = "docs/ssl-synthetic-localization-background-editor-transfer-figures-v1.md"
MAIN_ROOT = "outputs/ssl_synthetic_localization/background_editor_transfer_v1"
SOURCES: tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            *runtime.SOURCES,
            *figures.SOURCE_FILES,
            CONFIG,
            PROTOCOL,
            "ecg_experiment/ssl_synthetic_background_editor_transfer_figures_run.py",
            "scripts/reports/run_ssl_synthetic_background_editor_transfer_figures.py",
        )
    )
)
REQUIRED_INPUTS: tuple[str, ...] = tuple(
    f"{MAIN_ROOT}/{name}"
    for name in [
        "inputs.npy",
        "support.npy",
        "geometry.jsonl",
        "summary.json",
        *[f"probabilities/fit_1_{arm}.npy" for arm in figures.ARMS],
    ]
)


def preflight(name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the complete identity contract before array reads or output creation.

    Parameters
    ----------
    name : str
        Committed repository-relative identity admission.

    Returns
    -------
    tuple
        Frozen plotting settings and complete source/input/accounting identities.
    """
    config, admission = shared.read_json(from_stored(CONFIG)), shared.read_json(from_stored(name))
    if not set(SOURCES) <= set(admission["source_sha256"]) or not set(REQUIRED_INPUTS) <= set(
        admission["input_sha256"]
    ):
        raise ValueError("Fixed-figure source/input identity roster incomplete")
    if (
        admission["source_sha256"][CONFIG] != admission["frozen_config_sha256"]
        or admission["source_sha256"][PROTOCOL] != admission["frozen_protocol_sha256"]
        or config["main_root"] != MAIN_ROOT
        or config["slot"] != 0
        or config["fit"] != 1
        or config["sources"] != list(figures.COHORTS)
        or config["variants"] != [2, 3, 4, 5]
        or config["arms"] != list(figures.ARMS)
        or config["probability_limits"] != [0.0, 1.0]
    ):
        raise ValueError("Prospective fixed illustration identity or selection differs")
    prior = admission["prior_ledger"]
    audit_summary = str(Path(prior["path"]).with_name("summary.json"))
    if (
        not prior["path"].startswith(MAIN_ROOT + "/independent_audit_")
        or prior["current_machine_stages"] != 36
        or admission["input_sha256"].get(prior["path"]) != prior["sha256"]
        or audit_summary not in admission["input_sha256"]
    ):
        raise ValueError("Require the actual successful independent36 prior and its receipt")
    for path, digest in admission["source_sha256"].items():
        if sha256_file(from_stored(path)) != digest:
            raise ValueError("Fixed illustration executable source changed: " + path)
    return config, admission


def verify_inputs(admission: dict[str, Any], guard: Any) -> None:
    """Verify every admitted byte and its original main/audit receipt linkage.

    Parameters
    ----------
    admission : dict
        Explicit complete saved-input/source identities and actual accounting base.
    guard : Any
        Live CPU/wall/RSS/output guard.

    Returns
    -------
    None
        Raises before rendering or terminal success if a binding differs.
    """
    for path, digest in admission["input_sha256"].items():
        guard()
        if sha256_file(from_stored(path)) != digest:
            raise ValueError("Saved illustration input changed: " + path)
    main = shared.read_json(from_stored(MAIN_ROOT) / "summary.json")
    for name in REQUIRED_INPUTS:
        relative = name[len(MAIN_ROOT) + 1 :]
        if (
            relative != "summary.json"
            and main["output_sha256"].get(relative) != admission["input_sha256"][name]
        ):
            raise ValueError("Saved illustration input differs from original main manifest")
    prior = admission["prior_ledger"]
    ledger = shared.read_json(from_stored(prior["path"]))
    audit_path = str(Path(prior["path"]).with_name("summary.json"))
    audit = shared.read_json(from_stored(audit_path))
    if (
        not audit["all_passed"]
        or audit["mismatch_count"] != 0
        or audit["main_summary_sha256"] != admission["input_sha256"][MAIN_ROOT + "/summary.json"]
        or not ledger["technical_passed"]
        or not all(ledger["resource_gates"].values())
        or ledger["sha256"] != admission["input_sha256"][audit_path]
    ):
        raise ValueError("Required independent audit or actual resource receipt did not pass")


def run(root: Path, name: str, clock: dict[str, float]) -> dict[str, Any]:
    """Render only the fixed admitted local figures and charge their actual CPU cost once.

    Parameters
    ----------
    root : Path
        Source checkout with the baseline-owned local output alias.
    name : str
        Committed illustration identity admission.
    clock : dict
        Actual whole-stage clocks captured before scientific imports.

    Returns
    -------
    dict
        Output identities/dimensions and explicit zero-forward/zero-score receipt.
    """
    config, admission = preflight(name)
    destination = shared.confined(root, config["output_root"])
    destination.mkdir(parents=True, exist_ok=False)
    limits = {
        "cpu_seconds": config["cpu_seconds"],
        "wall_seconds": config["wall_seconds"],
        "peak_rss_bytes": config["rss_bytes"],
        "output_bytes": config["output_bytes"],
        "gpu_lock_wall_seconds": 1,
    }
    compatible = {"study_id": config["study_id"], "resources": limits}
    timers = {}
    verified = False
    result = {
        "study_id": config["study_id"],
        "status": "started",
        "new_scores": 0,
        "new_forwards": 0,
        "gpu_used": False,
    }

    def guard() -> None:
        """Keep the existing measured CPU/wall/RSS limits and retained bytes live."""
        runtime.resource_guard(compatible, clock)

    def interrupted(signum: int, frame: Any) -> None:
        """Preserve the attempt when its bounded CPU/wall timer fires."""
        raise TimeoutError(f"Fixed illustration process timer:{signum}")

    try:
        shared.setup_timers(limits, clock, interrupted, timers)
        environment = shared.runtime_environment()
        result["runtime"] = environment
        sources = runtime.bridge_runtime.peak_runtime.snapshot(destination, admission, name)
        result["source_sha256"] = sources
        if shared.directory_bytes(destination) + 16384 >= config["output_bytes"]:
            raise ValueError("Source snapshot exhausts fixed illustration output budget")
        shared.verify_prior(root, admission)
        verified = True
        verify_inputs(admission, guard)
        guard()
        names = figures.plot(shared.confined(root, config["main_root"]), destination)
        guard()
        if shared.directory_bytes(destination) + 16384 >= config["output_bytes"]:
            raise ValueError("Fixed illustration output budget exhausted")
        outputs = {}
        for filename in names:
            with (destination / filename).open("rb") as stream:
                header = stream.read(24)
            if header[:8] != b"\x89PNG\r\n\x1a\n":
                raise ValueError("Saved fixed figure is not a PNG")
            outputs[filename] = {
                "sha256": sha256_file(destination / filename),
                "bytes": (destination / filename).stat().st_size,
                "dimensions": list(struct.unpack("!II", header[16:24])),
            }
        verify_inputs(admission, guard)
        for path, digest in sources.items():
            guard()
            if sha256_file(from_stored(path)) != digest:
                raise ValueError("Fixed illustration source changed during execution")
        result.update(
            status="PASS",
            outputs=outputs,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            model_forwards=0,
            generator_forwards=0,
            raw_record_reads=0,
            clinical_claim=False,
        )
    except Exception as error:
        result.update(status="FAIL", error_type=type(error).__name__, error=str(error))
    finally:
        runtime.finish(destination, compatible, admission, result, clock, verified, timers)
    return result
