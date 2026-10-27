"""Recover annotation acquisition with a prospective RSS cap and preserved failed prior."""

import json
import threading
from pathlib import Path
from typing import Any

from ecg_experiment import ssl_synthetic_edb_acquisition as frozen
from ecg_experiment.files import sha256_file

RECOVERY = "configs/ssl_synthetic_edb_acquisition_recovery_v2.json"
PROTOCOL = "docs/ssl-synthetic-localization-edb-acquisition-recovery-v2.md"
RUNNER = "ecg_experiment/ssl_synthetic_edb_acquisition_v2.py"
CLI = "scripts/data/acquire_ssl_synthetic_edb_annotations_v2.py"
TESTS = "tests/test_ssl_synthetic_edb_acquisition_v2.py"
SOURCES = (*frozen.SOURCES, RECOVERY, PROTOCOL, RUNNER, CLI, TESTS,
           "configs/ssl_synthetic_edb_acquisition_admission_v1.json",
           "docs/ssl-synthetic-localization-edb-acquisition-v1-failure-results.md")


def operating_config(base: dict[str, Any], recovery: dict[str, Any]) -> dict[str, Any]:
    """Apply only the prospectively declared acquisition RSS resource overlay.

    Parameters
    ----------
    base, recovery : dict
        Immutable original protocol and separate recovery declaration.

    Returns
    -------
    dict
        Independent in-memory configuration; original input remains unchanged.
    """
    old, new = base["resources"]["acquisition"], recovery["resources"]
    if old["rss_bytes"] != 1024**3 or new["rss_bytes"] != 2 * 1024**3:
        raise ValueError("Require the exact prospective 1-to-2 GiB RSS amendment")
    if {k: v for k, v in old.items() if k != "rss_bytes"} != {
        k: v for k, v in new.items() if k != "rss_bytes"
    }:
        raise ValueError("Recovery changes a resource other than RSS")
    result = json.loads(json.dumps(base))
    result["resources"]["acquisition"] = dict(new)
    return result


def latest_prior(root: Path, recovery: dict[str, Any]) -> dict[str, Any]:
    """Verify actual40 to failed41 and preserve its cost and absent source snapshot.

    Parameters
    ----------
    root : Path
        Source checkout with the local baseline output alias.
    recovery : dict
        Literal failed-attempt identities and prospective successor declaration.

    Returns
    -------
    dict
        Verified failed41 ledger as the successor's direct accounting prior.
    """
    original_path = root / recovery["original_admission"]
    if sha256_file(original_path) != recovery["original_admission_sha256"]:
        raise ValueError("Original source admission identity changed")
    original = json.loads(original_path.read_text())
    frozen.verify_sources(root, original)
    base = json.loads((root / frozen.CONFIG).read_text())
    earlier = frozen.verify_prior(root, original, base)
    directory = root / recovery["failed_directory"]
    expected = {"summary.json": recovery["failed_summary_sha256"],
                "prospective_ledger.json": recovery["failed_ledger_sha256"]}
    for name, digest in expected.items():
        if sha256_file(directory / name) != digest:
            raise ValueError("Failed41 receipt identity changed")
    names = {p.relative_to(directory).as_posix() for p in directory.rglob("*")}
    if names != set(expected):
        raise ValueError("Failed41 pre-snapshot directory membership changed")
    summary = json.loads((directory / "summary.json").read_text())
    ledger = json.loads((directory / "prospective_ledger.json").read_text())
    if not (summary["status"] == "failed" and summary["error_type"] == "MemoryError"
            and summary["error"] == "Acquisition RSS ceiling exceeded"
            and all(summary[k] == 0 for k in (
                "requests", "received_payload_bytes", "new_waveform_decodes",
                "new_annotation_decodes", "gpu_operations", "new_model_scores", "new_training_updates"))):
        raise ValueError("Expected exact pre-HTTP resource failure")
    if not (ledger["sha256"] == expected["summary.json"]
            and ledger["prior_accounting_verified"] and not ledger["technical_passed"]
            and ledger["resource_gates"] == {
                "cpu": True, "wall": True, "rss": False, "output": True, "network": True}
            and ledger["prior_ledger_sha256"] == base["accounting"]["initial_ledger_sha256"]):
        raise ValueError("Failed41 predecessor/resource lineage changed")
    if not (ledger["cpu_seconds"] == recovery["failed_cpu_seconds"]
            and ledger["wall_seconds"] == recovery["failed_wall_seconds"]
            and ledger["peak_rss_bytes"] == recovery["failed_peak_rss_bytes"]
            and ledger["gpu_wall_seconds"] == 0
            and ledger["current_machine_stages"] == earlier["current_machine_stages"] + 1
            == recovery["failed_stage"]
            and ledger["current_machine_cumulative_cpu_seconds"]
            == earlier["current_machine_cumulative_cpu_seconds"] + ledger["cpu_seconds"]
            == recovery["prior_cpu_seconds"]
            and ledger["current_machine_cumulative_gpu_wall_seconds"]
            == earlier["current_machine_cumulative_gpu_wall_seconds"]
            == recovery["prior_gpu_wall_seconds"]):
        raise ValueError("Failed41 cost must be counted exactly once")
    return ledger


def preflight(root: Path, admission_path: Path) -> tuple[dict, dict, dict, dict]:
    """Verify frozen source/runtime and failed history without any HTTP or annotation access.

    Parameters
    ----------
    root : Path
        Source checkout and local artifact aliases.
    admission_path : Path
        Separately committed successor source admission.

    Returns
    -------
    tuple
        Operating configuration, recovery declaration, admission and actual prior.
    """
    admission = json.loads(admission_path.read_text())
    if not set(SOURCES) <= set(admission["source_sha256"]):
        raise ValueError("Incomplete recovery source roster")
    frozen.verify_sources(root, admission)
    recovery = json.loads((root / RECOVERY).read_text())
    base = json.loads((root / frozen.CONFIG).read_text())
    config = operating_config(base, recovery)
    prior = latest_prior(root, recovery)
    expected = {
        "path": recovery["failed_directory"] + "/prospective_ledger.json",
        "sha256": recovery["failed_ledger_sha256"],
        "current_machine_stages": prior["current_machine_stages"],
        "current_machine_cumulative_cpu_seconds": prior["current_machine_cumulative_cpu_seconds"],
        "current_machine_cumulative_gpu_wall_seconds": prior["current_machine_cumulative_gpu_wall_seconds"],
    }
    if admission["prior_ledger"] != expected:
        raise ValueError("Recovery admission must start directly from actual41")
    return config, recovery, admission, prior


def collect(context: dict[str, Any], summary: dict[str, Any]) -> None:
    """Reuse the frozen exact asset/role/hash/HTTP recipe without interpreting annotations.

    Parameters
    ----------
    context : dict
        Resource-amended state for the new exclusive output directory.
    summary : dict
        Terminal result populated with original acquisition facts.
    """
    output, config = context["output"], context["config"]
    table_receipt = frozen.receive(context, "SHA256SUMS.txt", None)
    table = frozen.checksum_table((output / "assets/SHA256SUMS.txt").read_bytes())
    metadata = [n for n in config["source"]["metadata_files"] if n != "SHA256SUMS.txt"]
    if any(name not in table for name in metadata):
        raise ValueError("Official table omits a required metadata asset")
    receipts = [table_receipt, *frozen.receive_group(context, [(n, table[n]) for n in metadata])]
    roles = frozen.record_roles((output / "assets/RECORDS").read_bytes(), config)
    frozen.write_new(output / "record_roles.json", roles)
    names = [r["record"] + "." + s for r in roles for s in config["source"]["record_extensions"]]
    if len(names) != 180 or any(name not in table for name in names):
        raise ValueError("Official table omits a required annotation/header asset")
    context["allowlist"].update(names)
    frozen.write_new(output / "prospective_asset_manifest.json",
                     {name: table[name] for name in [*metadata, *names]})
    receipts.extend(frozen.receive_group(context, [(n, table[n]) for n in names]))
    frozen.guard(context)
    if any(sha256_file(output / "assets" / r["name"]) != r["sha256"] for r in receipts):
        raise ValueError("An acquired asset changed after transfer")
    if context["requests"] != 184:
        raise ValueError("Final request count differs from184")
    summary.update(status="passed", assets=receipts, records=90, document_components=78,
                   biological_identity_verified=False,
                   asset_manifest_sha256=sha256_file(output / "prospective_asset_manifest.json"),
                   record_roles_sha256=sha256_file(output / "record_roles.json"),
                   received_table_first_use_sha256=table_receipt["sha256"])


def acquire(root: Path, admission_path: Path, started: tuple[float, float]) -> dict:
    """Execute one separately admitted recovery, retaining failed41 and every new failure.

    Parameters
    ----------
    root, admission_path : Path
        Explicit source root and committed recovery source admission.
    started : tuple
        Monotonic wall and process CPU starts captured before imports.

    Returns
    -------
    dict
        Actual successor terminal ledger; no failed cost is erased or reset.
    """
    wall_start, cpu_start = started
    recovery = json.loads((root / RECOVERY).read_text())
    base = json.loads((root / frozen.CONFIG).read_text())
    config = operating_config(base, recovery)
    output = root / recovery["output_directory"]
    output.mkdir(parents=True, exist_ok=False)
    context = {
        "config": config, "output": output, "limits": config["resources"]["acquisition"],
        "wall_start": wall_start, "cpu_start": cpu_start, "requests": 0,
        "received_bytes": 0, "retained_bytes": 0, "mutex": threading.Lock(),
        "allowlist": set(config["source"]["metadata_files"]), "stop": threading.Event(),
        "deadline": wall_start + config["resources"]["acquisition"]["wall_seconds"] - 2,
        "stage": "edb_annotation_acquisition_v2",
        "prior_ledger_sha256": recovery["failed_ledger_sha256"], "annotation_decodes": 0,
    }
    prior = None
    summary = {"study_id": recovery["study_id"], "phase": "acquisition", "status": "started"}
    try:
        config, recovery, admission, prior = preflight(root, admission_path)
        frozen.guard(context)
        for name in admission["source_sha256"]:
            target = output / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write((root / name).read_bytes())
        frozen.write_new(output / "admission_snapshot.json", admission)
        context["retained_bytes"] = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
        summary.update(source_sha256=admission["source_sha256"], runtime=admission["runtime"],
                       resource_amendment=RECOVERY, effective_resources=context["limits"])
        collect(context, summary)
        frozen.verify_sources(root, admission)
        latest_prior(root, recovery)
    except Exception as error:
        context["stop"].set()
        summary.update(status="failed", error_type=type(error).__name__, error=str(error))
    return frozen.finish(context, summary, prior)
