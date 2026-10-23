"""Reuse verified metadata and acquire only the released EDB annotation/header roster."""

import json
import threading
from pathlib import Path
from typing import Any

from ecg_experiment import ssl_synthetic_edb_acquisition as frozen
from ecg_experiment import ssl_synthetic_edb_acquisition_v2 as previous
from ecg_experiment.files import sha256_file

RECOVERY = "configs/ssl_synthetic_edb_acquisition_recovery_v3.json"
PROTOCOL = "docs/ssl-synthetic-localization-edb-acquisition-recovery-v3.md"
RUNNER = "ecg_experiment/ssl_synthetic_edb_acquisition_v3.py"
CLI = "scripts/data/acquire_ssl_synthetic_edb_annotations_v3.py"
TESTS = "tests/test_ssl_synthetic_edb_acquisition_v3.py"
SOURCES = (
    *previous.SOURCES,
    RECOVERY,
    PROTOCOL,
    RUNNER,
    CLI,
    TESTS,
    "configs/ssl_synthetic_edb_acquisition_admission_v2.json",
    "docs/ssl-synthetic-localization-edb-acquisition-v2-failure-results.md",
)


def operating_config(base: dict[str, Any], recovery: dict[str, Any]) -> dict[str, Any]:
    """Apply only the prospective released-roster, request-count and RSS amendments.

    Parameters
    ----------
    base, recovery : dict
        Immutable original configuration and separately committed v3 declaration.

    Returns
    -------
    dict
        Independent operating copy with corrected79-component grouping and180 new GETs.
    """
    result = previous.operating_config(base, recovery)
    expected = json.loads(json.dumps(base["grouping"]))
    expected["explicit_groups"][0].remove("e0120")
    expected["expected_document_components"] = 79
    if recovery["grouping"] != expected or recovery["unreleased_document_range_members"] != ["e0120"]:
        raise ValueError("Require exactly the released-roster grouping correction")
    if not (
        recovery["maximum_new_requests"] == 180
        and recovery["maximum_concurrent_requests"] == 2
        and recovery["new_metadata_requests"] == 0
    ):
        raise ValueError("Require180 record-asset requests and zero metadata requests")
    result["grouping"] = json.loads(json.dumps(recovery["grouping"]))
    result["source"]["maximum_requests"] = recovery["maximum_new_requests"]
    return result


def _checked_json(root: Path, name: str, digest: str) -> dict[str, Any]:
    """Require an immutable local JSON identity before interpreting its accounting fields."""
    path = root / name
    if sha256_file(path) != digest:
        raise ValueError(f"Prior artifact identity changed: {name}")
    return json.loads(path.read_text())


def _snapshot(root: Path, recovery: dict, admission: dict) -> None:
    """Verify exact failed42 file membership and its original complete source snapshot."""
    directory = root / recovery["failed_directory"]
    expected = {"summary.json", "prospective_ledger.json", "admission_snapshot.json"}
    expected.update("source_snapshot/" + name for name in admission["source_sha256"])
    for name, row in recovery["reused_metadata"].items():
        expected.add("assets/" + name)
        for kind in ("started", "terminal"):
            expected.add((root / row[kind + "_receipt_path"]).relative_to(directory).as_posix())
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if actual != expected:
        raise ValueError("Failed42 file membership must contain exactly four metadata assets and no records")
    if json.loads((directory / "admission_snapshot.json").read_text()) != admission:
        raise ValueError("Failed42 admission snapshot differs")
    for name, digest in admission["source_sha256"].items():
        if sha256_file(directory / "source_snapshot" / name) != digest:
            raise ValueError("Failed42 source snapshot differs")


def latest_prior(root: Path, recovery: dict[str, Any]) -> tuple[dict, dict]:
    """Verify immutable40-to41-to42 lineage and retain failed42's actual cost exactly once.

    Parameters
    ----------
    root : Path
        Source checkout and local output alias.
    recovery : dict
        Committed source/receipt identities and exact failure/cost facts.

    Returns
    -------
    tuple
        Direct failed42 ledger and its source/runtime admission.
    """
    admission = _checked_json(root, recovery["failed_admission"], recovery["failed_admission_sha256"])
    _, prior_recovery, checked_admission, prior = previous.preflight(
        root, root / recovery["failed_admission"]
    )
    if admission != checked_admission or recovery["previous_recovery"] != previous.RECOVERY:
        raise ValueError("Failed42 source admission or preceding recovery differs")
    prefix = recovery["failed_directory"] + "/"
    summary = _checked_json(root, prefix + "summary.json", recovery["failed_summary_sha256"])
    ledger = _checked_json(root, prefix + "prospective_ledger.json", recovery["failed_ledger_sha256"])
    expected = {
        "status": "failed",
        "error_type": "ValueError",
        "error": "Missing or overlapping published linkage constraint",
        "requests": 4,
        "received_payload_bytes": 35385,
        "new_waveform_decodes": 0,
        "new_annotation_decodes": 0,
        "gpu_operations": 0,
        "new_model_scores": 0,
        "new_training_updates": 0,
        "source_sha256": admission["source_sha256"],
        "runtime": admission["runtime"],
        "effective_resources": prior_recovery["resources"],
    }
    if any(summary[key] != value for key, value in expected.items()):
        raise ValueError("Expected exact metadata-only failed42 membership guard")
    _verify_cost(ledger, prior, recovery, prior_recovery["failed_ledger_sha256"])
    _snapshot(root, recovery, admission)
    return ledger, admission


def _verify_cost(ledger: dict, prior: dict, recovery: dict, prior_digest: str) -> None:
    """Reject unverified, reset or twice-counted failed42 cumulative accounting."""
    expected = {
        "stage": "edb_annotation_acquisition_v2",
        "receipt": "summary.json",
        "sha256": recovery["failed_summary_sha256"],
        "prior_accounting_verified": True,
        "technical_passed": False,
        "prior_ledger_sha256": prior_digest,
        "cpu_seconds": recovery["failed_cpu_seconds"],
        "wall_seconds": recovery["failed_wall_seconds"],
        "peak_rss_bytes": recovery["failed_peak_rss_bytes"],
        "gpu_wall_seconds": 0,
        "current_machine_stages": recovery["failed_stage"],
        "current_machine_cumulative_cpu_seconds": recovery["prior_cpu_seconds"],
        "current_machine_cumulative_gpu_wall_seconds": recovery["prior_gpu_wall_seconds"],
        "resource_gates": {"cpu": True, "wall": True, "rss": True, "output": True, "network": True},
    }
    if any(ledger[key] != value for key, value in expected.items()):
        raise ValueError("Failed42 accounting/resource identity differs")
    if not (
        ledger["current_machine_stages"] == prior["current_machine_stages"] + 1
        and ledger["current_machine_cumulative_cpu_seconds"]
        == prior["current_machine_cumulative_cpu_seconds"] + ledger["cpu_seconds"]
        and ledger["current_machine_cumulative_gpu_wall_seconds"]
        == prior["current_machine_cumulative_gpu_wall_seconds"]
    ):
        raise ValueError("Failed42 cost must be counted exactly once")


def _metadata_asset(root: Path, recovery: dict, config: dict, name: str, row: dict) -> bytes:
    """Verify one metadata payload and its original started/completed direct-GET receipts."""
    frozen.safe_name(name)
    path = root / recovery["metadata_directory"] / name
    if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
        raise ValueError("Reused metadata identity differs")
    expected = {
        "name": name,
        "url": config["source"]["file_base_url"] + name,
        "declared_sha256": row["declared_sha256"],
    }
    for kind in ("started", "terminal"):
        receipt = _checked_json(root, row[kind + "_receipt_path"], row[kind + "_receipt_sha256"])
        details = {
            **expected,
            "status": "started" if kind == "started" else "passed",
            "received_bytes": 0 if kind == "started" else row["bytes"],
        }
        if kind == "terminal":
            details.update(http_status=200, sha256=row["sha256"], received_sha256=row["sha256"])
        if any(receipt[key] != value for key, value in details.items()):
            raise ValueError("Reused metadata transfer receipt differs")
    return path.read_bytes()


def verified_metadata(root: Path, recovery: dict, config: dict) -> dict[str, bytes]:
    """Verify exactly four previously completed metadata assets without HTTP or annotation access.

    Parameters
    ----------
    root : Path
        Local checkout with the immutable prior output alias.
    recovery, config : dict
        Committed reuse identities and operative configuration.

    Returns
    -------
    dict
        Only the four verified local metadata byte strings.
    """
    if set(recovery["reused_metadata"]) != set(config["source"]["metadata_files"]):
        raise ValueError("Require exactly four declared metadata reuse assets")
    payloads = {
        name: _metadata_asset(root, recovery, config, name, row)
        for name, row in recovery["reused_metadata"].items()
    }
    table = frozen.checksum_table(payloads["SHA256SUMS.txt"])
    for name, row in recovery["reused_metadata"].items():
        if name == "SHA256SUMS.txt":
            if row["declared_sha256"] is not None:
                raise ValueError("First-use SHA table has no official self-declaration")
            continue
        if row["declared_sha256"] != table.get(name) or row["sha256"] != table.get(name):
            raise ValueError("Metadata official declaration differs")
    frozen.record_roles(payloads["RECORDS"], config)
    return payloads


def preflight(root: Path, admission_path: Path) -> tuple[dict, dict, dict, dict, dict]:
    """Verify complete current/prior source, history and metadata reuse before any new GET.

    Parameters
    ----------
    root, admission_path : Path
        Explicit checkout and separately committed successor source admission.

    Returns
    -------
    tuple
        Operating config, recovery, admission, direct42 prior and metadata bytes.
    """
    admission = json.loads(admission_path.read_text())
    if not set(SOURCES) <= set(admission["source_sha256"]):
        raise ValueError("Incomplete v3 source roster")
    frozen.verify_sources(root, admission)
    recovery = json.loads((root / RECOVERY).read_text())
    config = operating_config(json.loads((root / frozen.CONFIG).read_text()), recovery)
    prior, old_admission = latest_prior(root, recovery)
    if any(
        admission["source_sha256"].get(name) != digest
        for name, digest in old_admission["source_sha256"].items()
    ):
        raise ValueError("Successor must preserve every failed42 source obligation")
    expected = {
        "path": recovery["failed_directory"] + "/prospective_ledger.json",
        "sha256": recovery["failed_ledger_sha256"],
        "current_machine_stages": prior["current_machine_stages"],
        "current_machine_cumulative_cpu_seconds": prior["current_machine_cumulative_cpu_seconds"],
        "current_machine_cumulative_gpu_wall_seconds": prior["current_machine_cumulative_gpu_wall_seconds"],
    }
    if admission["prior_ledger"] != expected:
        raise ValueError("V3 admission must start directly from failed42")
    return config, recovery, admission, prior, verified_metadata(root, recovery, config)


def collect(context: dict[str, Any], summary: dict[str, Any], recovery: dict, payloads: dict) -> None:
    """Retain four verified local assets and perform only180 new allowlisted record GETs.

    Parameters
    ----------
    context, summary : dict
        Measured phase state and terminal facts.
    recovery, payloads : dict
        Verified provenance and four local metadata payloads.
    """
    output, config = context["output"], context["config"]
    (output / "assets").mkdir()
    reused = []
    for name, payload in payloads.items():
        frozen.guard(context)
        with (output / "assets" / name).open("xb") as stream:
            stream.write(payload)
        context["retained_bytes"] += len(payload)
        reused.append(
            {
                "name": name,
                **recovery["reused_metadata"][name],
                "source_path": recovery["metadata_directory"] + "/" + name,
                "acquisition_mode": "verified_local_reuse",
                "new_requests": 0,
                "new_received_bytes": 0,
            }
        )
    frozen.write_new(output / "metadata_reuse.json", reused)
    table = frozen.checksum_table(payloads["SHA256SUMS.txt"])
    roles = frozen.record_roles(payloads["RECORDS"], config)
    frozen.write_new(output / "record_roles.json", roles)
    names = [r["record"] + "." + s for r in roles for s in config["source"]["record_extensions"]]
    if len(names) != 180 or len(set(names)) != 180 or any(name not in table for name in names):
        raise ValueError("Official table omits a required record asset")
    context["allowlist"] = set(names)
    frozen.write_new(
        output / "prospective_asset_manifest.json",
        {
            **{name: row["sha256"] for name, row in recovery["reused_metadata"].items()},
            **{name: table[name] for name in names},
        },
    )
    context["retained_bytes"] = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    receipts = frozen.receive_group(context, [(name, table[name]) for name in names])
    frozen.guard(context)
    if context["requests"] != 180:
        raise ValueError("Final request count differs from180")
    if any(
        sha256_file(output / "assets" / name) != row["sha256"]
        for name, row in recovery["reused_metadata"].items()
    ):
        raise ValueError("Reused metadata changed after local retention")
    if any(sha256_file(output / "assets" / r["name"]) != r["sha256"] for r in receipts):
        raise ValueError("A new record asset changed after transfer")
    summary.update(
        status="passed",
        assets=receipts,
        reused_metadata=reused,
        metadata_reuse_sha256=sha256_file(output / "metadata_reuse.json"),
        records=90,
        document_components=79,
        biological_identity_verified=False,
        released_roster_grouping_correction=RECOVERY,
        asset_manifest_sha256=sha256_file(output / "prospective_asset_manifest.json"),
        record_roles_sha256=sha256_file(output / "record_roles.json"),
        received_table_first_use_sha256=recovery["reused_metadata"]["SHA256SUMS.txt"]["sha256"],
        reused_metadata_bytes=sum(len(value) for value in payloads.values()),
        new_metadata_requests=0,
    )


def acquire(root: Path, admission_path: Path, started: tuple[float, float]) -> dict:
    """Execute one separately admitted v3 phase and retain every failure and actual cost.

    Parameters
    ----------
    root, admission_path : Path
        Explicit checkout and committed source admission.
    started : tuple
        Wall and CPU starts captured before controller imports.

    Returns
    -------
    dict
        Terminal successor ledger with direct verified42 cost lineage.
    """
    wall_start, cpu_start = started
    recovery = json.loads((root / RECOVERY).read_text())
    config = operating_config(json.loads((root / frozen.CONFIG).read_text()), recovery)
    output = root / recovery["output_directory"]
    output.mkdir(parents=True, exist_ok=False)
    context = {
        "config": config,
        "output": output,
        "limits": config["resources"]["acquisition"],
        "wall_start": wall_start,
        "cpu_start": cpu_start,
        "requests": 0,
        "received_bytes": 0,
        "retained_bytes": 0,
        "mutex": threading.Lock(),
        "allowlist": set(),
        "stop": threading.Event(),
        "deadline": wall_start + config["resources"]["acquisition"]["wall_seconds"] - 2,
        "stage": "edb_annotation_acquisition_v3",
        "prior_ledger_sha256": recovery["failed_ledger_sha256"],
        "annotation_decodes": 0,
    }
    prior = None
    summary = {"study_id": recovery["study_id"], "phase": "acquisition", "status": "started"}
    try:
        prior, _ = latest_prior(root, recovery)
        config, recovery, admission, prior, payloads = preflight(root, admission_path)
        frozen.guard(context)
        for name in admission["source_sha256"]:
            target = output / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write((root / name).read_bytes())
        frozen.write_new(output / "admission_snapshot.json", admission)
        context["retained_bytes"] = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
        summary.update(
            source_sha256=admission["source_sha256"],
            runtime=admission["runtime"],
            operating_amendment=RECOVERY,
            effective_resources=context["limits"],
        )
        collect(context, summary, recovery, payloads)
        frozen.verify_sources(root, admission)
        latest_prior(root, recovery)
        verified_metadata(root, recovery, config)
    except Exception as error:
        context["stop"].set()
        summary.update(status="failed", error_type=type(error).__name__, error=str(error))
    return frozen.finish(context, summary, prior)
