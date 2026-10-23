"""Execute source-admitted EDB annotation qualification without signals or model scores."""

import json
import sys
import threading
from importlib import metadata, util
from pathlib import Path
from typing import Any

import numpy as np
import wfdb
from wfdb.io import annotation as wfdb_annotation
from wfdb.io import record as wfdb_record

from ecg_experiment import ssl_synthetic_edb_acquisition as shared
from ecg_experiment import ssl_synthetic_edb_qualification as core
from ecg_experiment.files import sha256_file
from ecg_experiment.paths import to_stored

CONFIG = shared.CONFIG
CONTRACT = core.CONTRACT
EXECUTION = "configs/ssl_synthetic_edb_qualification_execution_v2.json"
GROUPING = "configs/ssl_synthetic_edb_acquisition_recovery_v3.json"
RUNNER = "ecg_experiment/ssl_synthetic_edb_qualification_run.py"
CLI = "scripts/data/qualify_ssl_synthetic_edb_annotations.py"
TESTS = "tests/test_ssl_synthetic_edb_qualification_run.py"
SOURCES = tuple(
    dict.fromkeys(
        (
            *shared.SOURCES,
            *core.SOURCES,
            EXECUTION,
            GROUPING,
            RUNNER,
            CLI,
            TESTS,
            "docs/ssl-synthetic-localization-edb-qualification-execution-v2.md",
            "docs/ssl-synthetic-localization-edb-acquisition-recovery-v3.md",
            "scripts/__init__.py",
            "scripts/data/__init__.py",
        )
    )
)
ROOT_OUTPUT = "outputs/ssl_synthetic_localization/edb_annotation_qualification_v1/qualification_v1"


def package_identity(root: Path) -> dict[str, Any]:
    """Pin installed annotation-library sources and numerical extension without copying binaries.

    Parameters
    ----------
    root : Path
        Source checkout with the shared environment alias.

    Returns
    -------
    dict
        Version strings and repository-relative file identities.
    """
    site = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    files = [Path(m.__file__) for m in (np, wfdb, wfdb_annotation, wfdb_record)]
    for name in ("wfdb.io._header", "wfdb.io._signal", "numpy._core._multiarray_umath"):
        spec = util.find_spec(name)
        if spec is None or spec.origin is None:
            raise ValueError("Required installed numerical/annotation implementation is absent")
        files.append(Path(spec.origin))
    identities = {}
    for path in files:
        relative = Path(".venv") / "lib" / site.parent.name / "site-packages" / path.relative_to(site)
        identities[relative.as_posix()] = sha256_file(root / relative)
    return {"numpy": np.__version__, "wfdb": metadata.version("wfdb"), "file_sha256": identities}


def operating_config(root: Path) -> dict[str, Any]:
    """Combine immutable original truth, corrected roster, grid and execution declarations.

    Parameters
    ----------
    root : Path
        Source checkout containing all separately frozen declarations.

    Returns
    -------
    dict
        Explicit operating configuration, without editing historical base files.
    """
    base = json.loads((root / CONFIG).read_text())
    execution = json.loads((root / EXECUTION).read_text())
    grouping = json.loads((root / GROUPING).read_text())
    result = json.loads(json.dumps(base))
    result["grouping"] = grouping["grouping"]
    result["contract"] = json.loads((root / CONTRACT).read_text())
    result["resources"]["qualification"] = execution["resources"]["qualification"]
    if result["grouping"]["expected_document_components"] != 79:
        raise ValueError("Require released-roster corrected79-component contract")
    return result


def verified_acquisition_snapshot(root: Path, acquired: Path, admission: dict[str, Any]) -> None:
    """Require the executed admission, source snapshot and successful summary byte identities."""
    declared = admission["acquisition_admission"]
    if sha256_file(root / declared["path"]) != declared["sha256"]:
        raise ValueError("Executed acquisition admission changed")
    original = json.loads((root / declared["path"]).read_text())
    shared.verify_sources(root, original)
    for name, expected in original["source_sha256"].items():
        if sha256_file(acquired / "source_snapshot" / name) != expected:
            raise ValueError("Actual acquisition source snapshot changed")
    if sha256_file(acquired / "admission_snapshot.json") != declared["sha256"]:
        raise ValueError("Actual acquisition admission snapshot changed")
    if sha256_file(acquired / "summary.json") != admission["acquisition_summary_sha256"]:
        raise ValueError("Actual acquisition receipt changed")


def verified_acquisition_ledger(acquired: Path, admission: dict[str, Any]) -> dict[str, Any]:
    """Require literal immediate prior stage and cumulative costs without resetting history."""
    ledger_path = acquired / "prospective_ledger.json"
    expected = admission["prior_ledger"]
    if to_stored(ledger_path) != expected["path"] or sha256_file(ledger_path) != expected["sha256"]:
        raise ValueError("Qualification must use the literal immediate acquisition ledger")
    ledger = json.loads(ledger_path.read_text())
    for key in (
        "current_machine_stages",
        "current_machine_cumulative_cpu_seconds",
        "current_machine_cumulative_gpu_wall_seconds",
    ):
        if ledger[key] != expected[key]:
            raise ValueError("Immediate acquisition cumulative costs changed")
    if not (
        ledger["technical_passed"]
        and ledger["prior_accounting_verified"]
        and all(ledger["resource_gates"].values())
        and ledger["sha256"] == admission["acquisition_summary_sha256"]
    ):
        raise ValueError("Qualification requires completed acquisition resource/source admission")
    return ledger


def verified_inputs(root: Path, admission: dict[str, Any], config: dict[str, Any]) -> tuple[dict, list]:
    """Hash-check every acquired asset and the actual successful immediate prior before decoding.

    Parameters
    ----------
    root : Path
        Source checkout whose output alias targets baseline local artifacts.
    admission, config : dict
        Source-bound literal acquired identities and operating annotation contract.

    Returns
    -------
    tuple
        Verified acquisition ledger and complete original-order record-role roster.
    """
    acquired = root / admission["acquisition_root"]
    verified_acquisition_snapshot(root, acquired, admission)
    ledger = verified_acquisition_ledger(acquired, admission)
    roles = shared.record_roles((acquired / "assets/RECORDS").read_bytes(), config)
    if sha256_file(acquired / "record_roles.json") != admission["roles_sha256"]:
        raise ValueError("Prospective acquired roles changed")
    if json.loads((acquired / "record_roles.json").read_text()) != roles:
        raise ValueError("Corrected role allocation must reproduce exactly")
    names = {
        *config["source"]["metadata_files"],
        *(r["record"] + "." + s for r in roles for s in ("atr", "hea")),
    }
    actual = {p.name for p in (acquired / "assets").iterdir() if p.is_file()}
    if names != actual or names != set(admission["asset_sha256"]):
        raise ValueError("Require complete184-asset catalog and no waveform")
    table = shared.checksum_table((acquired / "assets/SHA256SUMS.txt").read_bytes())
    for name, digest in admission["asset_sha256"].items():
        shared.safe_name(name)
        if sha256_file(acquired / "assets" / name) != digest:
            raise ValueError("Acquired asset bytes changed")
        if name != "SHA256SUMS.txt" and table[name] != digest:
            raise ValueError("Acquired asset differs from official source declaration")
    return ledger, roles


def preflight(root: Path, admission_path: Path) -> tuple[dict, dict, dict, list]:
    """Verify source/package/runtime and raw byte identities without parsing annotations.

    Parameters
    ----------
    root, admission_path : Path
        Explicit source checkout and committed literal asset/source admission.

    Returns
    -------
    tuple
        Configuration, admission, immediate accounting prior and record-role roster.
    """
    admission = json.loads(admission_path.read_text())
    if not set(SOURCES) <= set(admission["source_sha256"]):
        raise ValueError("Incomplete qualifier source/import roster")
    shared.verify_sources(root, admission)
    if package_identity(root) != admission["package_runtime"]:
        raise ValueError("Installed annotation/numerical implementation changed")
    config = operating_config(root)
    prior, roles = verified_inputs(root, admission, config)
    return config, admission, prior, roles


def finalized_record(decoded: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Fold decoder issues into record-wide unknown grids and recount saved labels.

    Parameters
    ----------
    decoded, result : dict
        Transient primary decode and annotation-grammar/grid result.

    Returns
    -------
    dict
        Complete persistable result and exact final arrays; no usable decoder-issue cells.
    """
    issues = [*result["issues"], *decoded["metadata"]["decoder_issues"]]
    labels, reasons = result["labels"], result["reasons"]
    if labels.shape != (2, 720) or labels.dtype != np.uint8:
        raise ValueError("Qualification label geometry/dtype differs")
    if reasons.shape != labels.shape or reasons.dtype != np.uint16:
        raise ValueError("Qualification reason geometry/dtype differs")
    if np.any(labels > 2) or np.any(reasons > 255):
        raise ValueError("Qualification label or reason code outside frozen domain")
    if np.any(reasons[labels != 0]) or np.any((labels == 0) & (reasons == 0)):
        raise ValueError("Final valid/unknown reason contract violated")
    if not issues and np.any(reasons & 128):
        raise ValueError("Record-issue reason requires an explicit recorded issue")
    if issues:
        labels = np.zeros((2, 720), dtype=np.uint8)
        reasons = np.full((2, 720), 128, dtype=np.uint16)
    counts = {
        **result["counts"],
        "issues": len(issues),
        "ST_positive_windows": int(np.count_nonzero(labels == 1)),
        "reference_windows": int(np.count_nonzero(labels == 2)),
        "unknown_windows": int(np.count_nonzero(labels == 0)),
    }
    return {
        **result,
        "metadata": decoded["metadata"],
        "issues": issues,
        "counts": counts,
        "labels": labels,
        "reasons": reasons,
        "engineering_passed": not issues,
    }


def record_descriptives(result: dict[str, Any], fs: int) -> dict[str, Any]:
    """Count saved unknown reasons, signed episode classes and raw channel-quality durations.

    Parameters
    ----------
    result : dict
        Final record output, including unknown-overridden label and reason arrays.
    fs : int
        Prospectively fixed sample rate.

    Returns
    -------
    dict
        Descriptive counts and seconds; reason bits may overlap and are not a partition.
    """
    reasons = result["reasons"]
    signed = {}
    for episode in result["episodes"]:
        key = ":".join(str(episode[k]) for k in ("category", "family", "channel", "sign", "extreme"))
        signed[key] = signed.get(key, 0) + 1
    quality = [dict.fromkeys(("unknown", "clean", "noisy", "unreadable"), 0) for _ in range(2)]
    for span in result["quality_spans"]:
        for channel, state in enumerate(span["states"]):
            name = {"c": "clean", "n": "noisy", "u": "unreadable", "unknown": "unknown"}[state]
            quality[channel][name] += (span["end"] - span["start"]) / fs
    return {
        "unknown_reason_bit_cells": {
            str(bit): int(np.count_nonzero(reasons & bit)) for bit in (1, 2, 4, 8, 16, 32, 64, 128)
        },
        "episode_class_counts": signed,
        "raw_quality_seconds_by_channel": quality,
        "quality_seconds_are_not_qualified_label_coverage": True,
    }


def record_summary(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Describe complete census availability without disease prevalence or model metrics.

    Parameters
    ----------
    rows : list
        Original-order completed record index including final counts and role.
    config : dict
        Frozen prospective role names.

    Returns
    -------
    dict
        Whole-record/component/role counts, with no exclusion or favorable selection.
    """
    groups = {r["component"] for r in rows}
    result = {"records": len(rows), "components": len(groups), "channels": 2 * len(rows), "roles": {}}
    for role in config["roles"]["names"]:
        selected = [r for r in rows if r["role"] == role]
        result["roles"][role] = {
            "records": len(selected),
            "components": len({r["component"] for r in selected}),
            **{
                key: sum(r["counts"][key] for r in selected)
                for key in (
                    "events",
                    "selected_events",
                    "issues",
                    "episodes",
                    "base_episodes",
                    "extreme_episodes",
                    "axis_episodes",
                    "ST_positive_windows",
                    "reference_windows",
                    "unknown_windows",
                )
            },
            "ST_capable_components": len(
                {r["component"] for r in selected if r["counts"]["ST_positive_windows"]}
            ),
            "reference_capable_components": len(
                {r["component"] for r in selected if r["counts"]["reference_windows"]}
            ),
        }
    return result


def run(root: Path, admission_path: Path, started: tuple[float, float]) -> dict[str, Any]:
    """Execute one bounded annotation-only qualification with immutable partial evidence.

    Parameters
    ----------
    root, admission_path : Path
        Explicit source checkout and prospectively committed input/source admission.
    started : tuple
        Whole-process monotonic wall and CPU starts from the thin CLI.

    Returns
    -------
    dict
        Actual terminal ledger; a complete engineering failure remains auditable.
    """
    wall, cpu = started
    config = operating_config(root)
    output = root / ROOT_OUTPUT
    output.mkdir(parents=True, exist_ok=False)
    context = {
        "config": config,
        "output": output,
        "limits": config["resources"]["qualification"],
        "wall_start": wall,
        "cpu_start": cpu,
        "requests": 0,
        "received_bytes": 0,
        "retained_bytes": 0,
        "stop": threading.Event(),
        "annotation_decodes": 0,
        "stage": "edb_annotation_qualification_v1",
    }
    prior, rows = None, []
    summary: dict[str, Any] = {
        "study_id": config["study_id"],
        "phase": "qualification",
        "status": "started",
        "record_decode_attempts": 0,
        "record_decodes_completed": 0,
        "independent_decoder_verified": False,
        "independent_audit_required": True,
        "annotation_decode_counter_unit": "completed_record_decode_calls",
    }
    try:
        config, admission, prior, roles = preflight(root, admission_path)
        context["prior_ledger_sha256"] = admission["prior_ledger"]["sha256"]
        shared.guard(context)
        for name in admission["source_sha256"]:
            target = output / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write((root / name).read_bytes())
        shared.write_new(output / "admission_snapshot.json", admission)
        labels = np.lib.format.open_memmap(
            output / "labels.npy", mode="w+", dtype=np.uint8, shape=(90, 2, 720)
        )
        reasons = np.lib.format.open_memmap(
            output / "reasons.npy", mode="w+", dtype=np.uint16, shape=labels.shape
        )
        labels[:] = 0
        reasons[:] = 128
        acquired = root / admission["acquisition_root"] / "assets"
        for slot, role in enumerate(roles):
            shared.guard(context)
            record = role["record"]
            shared.write_new(
                output / "record_receipts" / f"{record}.started.json", {**role, "status": "started"}
            )
            summary["record_decode_attempts"] += 1
            try:
                decoded = core.decode_record(acquired / record, output / "sanitized_headers" / record, config)
                context["annotation_decodes"] += 1
                summary["record_decodes_completed"] += 1
                result = finalized_record(decoded, core.qualify_events(decoded["events"], config))
                result["descriptives"] = record_descriptives(result, config["geometry"]["fs"])
                labels[slot], reasons[slot] = result.pop("labels"), result.pop("reasons")
                labels.flush()
                reasons.flush()
                row = {**role, **result}
                name = f"records/{record}.json"
                shared.write_new(output / name, row)
                index = {
                    **role,
                    "path": name,
                    "sha256": sha256_file(output / name),
                    "counts": row["counts"],
                    "descriptives": row["descriptives"],
                    "engineering_passed": row["engineering_passed"],
                    "wfdb_effective_view_verified": row["metadata"]["wfdb_effective_view_verified"],
                }
                rows.append(index)
                shared.write_new(output / "record_receipts" / f"{record}.completed.json", index)
                del decoded
            except Exception as error:
                shared.write_new(
                    output / "record_receipts" / f"{record}.failed.json",
                    {**role, "error_type": type(error).__name__, "error": str(error)},
                )
                raise
            context["retained_bytes"] = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
        shared.write_new(output / "record_index.json", rows)
        shared.verify_sources(root, admission)
        verified_inputs(root, admission, config)
        if package_identity(root) != admission["package_runtime"]:
            raise ValueError("Installed package implementation changed during qualification")
        gates = {
            "complete90records": len(rows) == 90,
            "all_record_contracts": all(r["engineering_passed"] for r in rows),
            "all_WFDB_effective_views": all(r["wfdb_effective_view_verified"] for r in rows),
            "published_grouping79": len({r["component"] for r in rows}) == 79,
            "all_sources_assets_reverified": True,
        }
        catalog = {
            p.relative_to(output).as_posix(): sha256_file(p)
            for p in output.rglob("*")
            if p.is_file() and not p.is_symlink()
        }
        summary.update(
            status="passed",
            source_sha256=admission["source_sha256"],
            package_runtime=admission["package_runtime"],
            source_admission_sha256=sha256_file(admission_path),
            engineering_gates=gates,
            engineering_passed_before_independent_audit=all(gates.values()),
            availability=record_summary(rows, config),
            output_sha256=catalog,
            acquired_assets_reverified=True,
            effective_resources=context["limits"],
            corrected_published_components=79,
            earlier78_assumption_resolved=True,
            resource_byte_accounting="addressed_files_including_readonly_ATR_alias_targets",
            engineering_failure_blocks_downstream=True,
            clinical_claim=False,
            all_roles_spent_development=True,
        )
    except Exception as error:
        summary.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            completed_records=len(rows),
            engineering_passed_before_independent_audit=False,
        )
    return shared.finish(context, summary, prior)
