"""Acquire only prospectively admitted public EDB annotations and structural headers."""

import concurrent.futures
import hashlib
import http.client
import json
import os
import platform
import re
import resource
import socket
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any, Never

from ecg_experiment.files import sha256_file

CONFIG = "configs/ssl_synthetic_edb_annotation_qualification_v1.json"
SOURCES = (
    CONFIG,
    "docs/ssl-synthetic-localization-edb-annotation-qualification-v1.md",
    "docs/ssl-synthetic-localization-public-annotation-scientific-review-v1.md",
    "ecg_experiment/__init__.py",
    "ecg_experiment/files.py",
    "ecg_experiment/paths.py",
    "ecg_experiment/ssl_synthetic_edb_acquisition.py",
    "scripts/data/acquire_ssl_synthetic_edb_annotations.py",
    "tests/test_ssl_synthetic_edb_acquisition.py",
    "pyproject.toml",
    "uv.lock",
)


def runtime_identity() -> dict[str, Any]:
    """Identify the installed Python and standard-library HTTP implementation.

    Returns
    -------
    dict
        Version strings and file hashes, without machine-specific source paths.
    """
    modules = (urllib.request, urllib.error, http.client, ssl, socket)
    return {
        "python": platform.python_version(),
        "executable_sha256": sha256_file(sys.executable),
        "openssl": ssl.OPENSSL_VERSION,
        "stdlib_sha256": {module.__name__: sha256_file(module.__file__) for module in modules},
    }


def write_new(path: Path, value: Any) -> None:
    """Serialize one immutable, exclusively created JSON artifact.

    Parameters
    ----------
    path : Path
        New local output path.
    value : Any
        Finite JSON-compatible value.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def safe_name(value: str) -> str:
    """Reject any asset that can escape its single, canonical file name.

    Parameters
    ----------
    value : str
        Official manifest file name.

    Returns
    -------
    str
        Unchanged safe name.
    """
    if (
        not value
        or value != PurePosixPath(value).name
        or "\\" in value
        or value in {".", ".."}
        or not re.fullmatch(r"[A-Za-z0-9_.-]+", value)
    ):
        raise ValueError("Noncanonical asset filename")
    return value


def checksum_table(payload: bytes) -> dict[str, str]:
    """Read the official checksum table without accepting duplicate declarations.

    Parameters
    ----------
    payload : bytes
        Admitted checksum-table bytes.

    Returns
    -------
    dict
        Literal file names mapped to lowercase SHA-256 declarations.
    """
    result: dict[str, str] = {}
    for line in payload.decode("ascii").splitlines():
        if not line:
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64})[ \t]+\*?([^\r\n]+)", line)
        if match is None:
            raise ValueError("Malformed official SHA declaration")
        name = match[2]
        if name in result:
            raise ValueError("Duplicate official SHA declaration")
        result[name] = match[1].lower()
    return result


def record_roles(payload: bytes, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Freeze official record linkage and hash roles before annotation access.

    Parameters
    ----------
    payload : bytes
        Official RECORDS bytes; no header or annotation contents.
    config : dict
        Committed source, grouping and role contract.

    Returns
    -------
    list
        Original record order, conservative component identities and roles.
    """
    records = payload.decode("ascii").splitlines()
    if (
        len(records) != config["source"]["records"]
        or len(set(records)) != len(records)
        or any(re.fullmatch(r"e[0-9]{4}", name) is None for name in records)
    ):
        raise ValueError("Official RECORDS does not contain exactly90 unique canonical names")
    canonical = dict.fromkeys(records)
    grouped: set[str] = set()
    for group in config["grouping"]["explicit_groups"]:
        if not set(group) <= set(records) or grouped.intersection(group):
            raise ValueError("Missing or overlapping published linkage constraint")
        grouped.update(group)
        for name in group:
            canonical[name] = min(group)
    canonical = {name: component or name for name, component in canonical.items()}
    if len(set(canonical.values())) != config["grouping"]["expected_document_components"]:
        raise ValueError("Published linkage component count differs")
    policy = config["roles"]
    result = []
    for name in records:
        text = f"{policy['namespace']}:{policy['seed']}:{canonical[name]}"
        bucket = int(hashlib.sha256(text.encode()).hexdigest(), 16) % policy["bucket_count"]
        role = 0 if bucket < policy["train_stop"] else 1 if bucket < policy["validation_stop"] else 2
        result.append(
            {"record": name, "component": canonical[name], "bucket": bucket, "role": policy["names"][role]}
        )
    return result


def guard(context: dict[str, Any]) -> None:
    """Enforce active phase limits with the declared terminal reserves.

    Parameters
    ----------
    context : dict
        Measured acquisition state shared by two HTTP workers.
    """
    limits = context["limits"]
    if context["stop"].is_set():
        raise RuntimeError("Acquisition cancelled after another asset failure")
    if time.process_time() - context["cpu_start"] >= limits["cpu_seconds"] - 2:
        raise TimeoutError("Acquisition CPU reserve reached")
    if time.perf_counter() - context["wall_start"] >= limits["wall_seconds"] - 2:
        raise TimeoutError("Acquisition wall reserve reached")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > limits["rss_bytes"]:
        raise MemoryError("Acquisition RSS ceiling exceeded")
    if context["retained_bytes"] >= limits["output_bytes"] - 1048576:
        raise ValueError("Acquisition retained-byte reserve reached")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Prevent an allowlisted request from causing a second, unadmitted request."""

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Never:
        """Refuse redirection without following the destination."""
        raise urllib.error.HTTPError(req.full_url, code, "Redirect prohibited", headers, fp)


def _body(context: dict[str, Any], response: Any, partial: Path, receipt: dict[str, Any]) -> None:
    """Stream bounded bytes with socket deadlines and synchronized actual network counters."""
    source = context["config"]["source"]
    if response.status != 200 or response.geturl() != receipt["url"]:
        raise ValueError("Require direct200 response from exact admitted URL")
    length = response.headers.get("Content-Length")
    if length is not None and (not length.isdigit() or int(length) > source["per_asset_bytes"]):
        raise ValueError("Invalid or oversized Content-Length")
    digest = hashlib.sha256()
    with partial.open("xb") as stream:
        while True:
            guard(context)
            if response.fp is None:
                break
            timeout = min(
                source["connect_read_idle_timeout_seconds"], context["deadline"] - time.perf_counter()
            )
            response.fp.raw._sock.settimeout(max(0.001, timeout))
            chunk = response.read1(65536)
            if not chunk:
                break
            with context["mutex"]:
                context["received_bytes"] += len(chunk)
                receipt["received_bytes"] += len(chunk)
                digest.update(chunk)
                receipt["received_sha256"] = digest.hexdigest()
                receipt["last_received_chunk_sha256"] = hashlib.sha256(chunk).hexdigest()
            stream.write(chunk)
            with context["mutex"]:
                context["retained_bytes"] += len(chunk)
                if context["received_bytes"] > source["received_payload_bytes"]:
                    raise ValueError("Received-payload ceiling exceeded")
                if receipt["received_bytes"] > source["per_asset_bytes"]:
                    raise ValueError("Per-asset payload ceiling exceeded")
                guard(context)
    receipt["sha256"] = digest.hexdigest()
    if length is not None and receipt["received_bytes"] != int(length):
        raise ValueError("Response body differs from Content-Length")


def receive(context: dict[str, Any], name: str, declared_sha256: str | None) -> dict[str, Any]:
    """Perform one allowlisted GET, retaining partial bytes and a terminal receipt.

    Parameters
    ----------
    context : dict
        Immutable source contract and measured, synchronized counters.
    name : str
        One explicitly authorized filename.
    declared_sha256 : str or None
        Official identity; absent only for the first-use checksum table itself.

    Returns
    -------
    dict
        HTTP status, received bytes, own identity and declaration comparison.
    """
    safe_name(name)
    if name not in context["allowlist"]:
        raise ValueError("Asset is outside the admitted metadata/annotation/header allowlist")
    guard(context)
    root, source = context["output"], context["config"]["source"]
    url = source["file_base_url"] + name
    receipt: dict[str, Any] = {
        "name": name,
        "url": url,
        "status": "started",
        "declared_sha256": declared_sha256,
        "received_bytes": 0,
    }
    write_new(root / "transfers" / (name + ".started.json"), receipt)
    partial = root / "assets" / (name + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    response = None
    start = time.perf_counter()
    try:
        with context["mutex"]:
            guard(context)
            context["requests"] += 1
            if context["requests"] > source["maximum_requests"]:
                raise ValueError("Request-count ceiling exceeded")
        remaining = source["connect_read_idle_timeout_seconds"]
        remaining = min(remaining, context["deadline"] - time.perf_counter())
        opener = urllib.request.build_opener(NoRedirect())
        request = urllib.request.Request(url, headers={"User-Agent": "ECG-local-annotation-research/1"})
        response = opener.open(request, timeout=max(0.001, remaining))
        receipt["http_status"] = response.status
        _body(context, response, partial, receipt)
        if declared_sha256 is not None and receipt["sha256"] != declared_sha256:
            raise ValueError("Official asset SHA-256 mismatch")
        os.link(partial, root / "assets" / name)
        partial.unlink()
        receipt["status"] = "passed"
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error))
        if isinstance(error, urllib.error.HTTPError):
            receipt["http_status"] = error.code
            error.close()
        if partial.exists():
            receipt["partial_bytes"] = partial.stat().st_size
            receipt["partial_sha256"] = sha256_file(partial)
            receipt["unretained_received_bytes"] = receipt["received_bytes"] - receipt["partial_bytes"]
        context["stop"].set()
        raise
    finally:
        if response is not None:
            response.close()
        receipt["wall_seconds"] = time.perf_counter() - start
        write_new(root / "transfers" / (name + ".terminal.json"), receipt)
    return receipt


def receive_group(context: dict[str, Any], assets: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """Drain at most two HTTP workers completely inside the measured phase.

    Parameters
    ----------
    context : dict
        Measured acquisition state.
    assets : list
        Explicit filename/official-hash pairs, without waveform assets.

    Returns
    -------
    list
        Successful transfer receipts in original allowlist order.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(receive, context, name, expected) for name, expected in assets]
        return [future.result() for future in futures]


def verify_prior(root: Path, admission: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Bind the exact successful stage40 ledger and its referenced receipt.

    Parameters
    ----------
    root : Path
        Source/artifact checkout with the permitted baseline output alias.
    admission : dict
        Committed acquisition identity admission.
    config : dict
        Frozen scientific accounting policy.

    Returns
    -------
    dict
        Verified predecessor ledger, without summing historical cumulative totals.
    """
    policy, prior = config["accounting"], admission["prior_ledger"]
    expected = {
        "path": policy["initial_ledger"],
        "sha256": policy["initial_ledger_sha256"],
        "current_machine_stages": policy["initial_stage"],
        "current_machine_cumulative_cpu_seconds": policy["initial_cpu_seconds"],
        "current_machine_cumulative_gpu_wall_seconds": policy["initial_gpu_wall_seconds"],
    }
    if prior != expected:
        raise ValueError("Acquisition admission does not use exact configured actual40 prior")
    path = root / prior["path"]
    if sha256_file(path) != prior["sha256"]:
        raise ValueError("Prior ledger identity mismatch")
    ledger = json.loads(path.read_text())
    for key in expected.keys() - {"path", "sha256"}:
        if ledger[key] != expected[key]:
            raise ValueError("Prior ledger accounting differs")
    receipt_name = safe_name(ledger["receipt"])
    if sha256_file(path.parent / receipt_name) != ledger["sha256"]:
        raise ValueError("Prior referenced receipt identity differs")
    if not ledger["technical_passed"] or not all(ledger["resource_gates"].values()):
        raise ValueError("Prior technical/resource gates did not pass")
    return ledger


def verify_sources(root: Path, admission: dict[str, Any]) -> None:
    """Check the complete declared source/environment catalog before HTTP.

    Parameters
    ----------
    root : Path
        Source checkout.
    admission : dict
        Source and installed-runtime identities.
    """
    sources = admission["source_sha256"]
    if not set(SOURCES) <= set(sources):
        raise ValueError("Acquisition source roster is incomplete")
    for name, expected in sources.items():
        if Path(name).is_absolute() or ".." in PurePosixPath(name).parts:
            raise ValueError("Source binding escapes checkout")
        if sha256_file(root / name) != expected:
            raise ValueError(f"Source identity mismatch: {name}")
    if runtime_identity() != admission["runtime"]:
        raise ValueError("Installed Python/HTTP runtime identity differs")


def finish(context: dict[str, Any], summary: dict[str, Any], prior: dict[str, Any] | None) -> dict:
    """Preserve a terminal receipt and charge the actual phase cost once.

    Parameters
    ----------
    context : dict
        Drained acquisition state and live phase clocks.
    summary : dict
        Successful manifest or retained failure description.
    prior : dict or None
        Verified immediate predecessor, if verification completed.

    Returns
    -------
    dict
        Final terminal ledger; no unknown historical cumulative costs are invented.
    """
    root, limits = context["output"], context["limits"]
    summary.update(
        requests=context["requests"],
        received_payload_bytes=context["received_bytes"],
        new_waveform_decodes=0,
        new_annotation_decodes=context.get("annotation_decodes", 0),
        gpu_operations=0,
        new_model_scores=0,
        new_training_updates=0,
    )
    write_new(root / "summary.json", summary)
    receipt_hash = sha256_file(root / "summary.json")
    retained = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    cpu = time.process_time() - context["cpu_start"]
    wall = time.perf_counter() - context["wall_start"]
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    gates = {
        "cpu": cpu <= limits["cpu_seconds"] - 2,
        "wall": wall <= limits["wall_seconds"] - 2,
        "rss": rss <= limits["rss_bytes"],
        "output": retained <= limits["output_bytes"] - 1048576,
        "network": context["received_bytes"] <= context["config"]["source"]["received_payload_bytes"],
    }
    verified = prior is not None
    ledger = {
        "stage": context.get("stage", "edb_annotation_acquisition_v1"),
        "receipt": "summary.json",
        "sha256": receipt_hash,
        "cpu_seconds": cpu,
        "wall_seconds": wall,
        "gpu_wall_seconds": 0.0,
        "peak_rss_bytes": rss,
        "resource_gates": gates,
        "technical_passed": summary["status"] == "passed" and all(gates.values()),
        "prior_accounting_verified": verified,
        "prior_ledger_sha256": context.get(
            "prior_ledger_sha256", context["config"]["accounting"]["initial_ledger_sha256"]
        )
        if verified
        else None,
        "current_machine_stages": prior["current_machine_stages"] + 1 if verified else None,
        "current_machine_cumulative_cpu_seconds": prior["current_machine_cumulative_cpu_seconds"] + cpu
        if verified
        else None,
        "current_machine_cumulative_gpu_wall_seconds": prior["current_machine_cumulative_gpu_wall_seconds"]
        if verified
        else None,
        "historical_costs_verified": False,
        "output_bytes_before_terminal_ledger": retained,
        "measurement_endpoint": (
            "after summary hash and final byte enumeration; before terminal ledger serialization"
        ),
        "terminal_bookkeeping_reserve_seconds": 2,
        "terminal_bookkeeping_reserve_bytes": 1048576,
    }
    write_new(root / "prospective_ledger.json", ledger)
    return ledger


def acquire(root: Path, admission_path: Path, started: tuple[float, float] | None = None) -> dict:
    """Run the frozen annotation-only acquisition after committed source admission.

    Parameters
    ----------
    root : Path
        Source checkout whose outputs alias targets baseline outputs.
    admission_path : Path
        Committed executable source/runtime/prior identity admission.
    started : tuple or None
        CLI wall/CPU clocks captured before importing this module.

    Returns
    -------
    dict
        Terminal acquisition ledger; failures remain locally saved and charged.
    """
    wall_start, cpu_start = started or (time.perf_counter(), time.process_time())
    config = json.loads((root / CONFIG).read_text())
    output = root / config["output_root"] / config["acquisition_subdirectory"]
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
        "allowlist": set(config["source"]["metadata_files"]),
        "stop": threading.Event(),
        "deadline": wall_start + config["resources"]["acquisition"]["wall_seconds"] - 2,
    }
    prior = None
    summary: dict[str, Any] = {"study_id": config["study_id"], "phase": "acquisition", "status": "started"}
    try:
        admission = json.loads(admission_path.read_text())
        verify_sources(root, admission)
        prior = verify_prior(root, admission, config)
        guard(context)
        for name in admission["source_sha256"]:
            target = output / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write((root / name).read_bytes())
        write_new(output / "admission_snapshot.json", admission)
        context["retained_bytes"] = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
        summary["source_sha256"] = admission["source_sha256"]
        summary["runtime"] = admission["runtime"]
        table_receipt = receive(context, "SHA256SUMS.txt", None)
        table = checksum_table((output / "assets/SHA256SUMS.txt").read_bytes())
        metadata = [name for name in config["source"]["metadata_files"] if name != "SHA256SUMS.txt"]
        if any(name not in table for name in metadata):
            raise ValueError("Official table omits a required metadata asset")
        receipts = [table_receipt, *receive_group(context, [(n, table[n]) for n in metadata])]
        roles = record_roles((output / "assets/RECORDS").read_bytes(), config)
        write_new(output / "record_roles.json", roles)
        names = [
            row["record"] + "." + suffix for row in roles for suffix in config["source"]["record_extensions"]
        ]
        if len(names) != 180 or any(name not in table for name in names):
            raise ValueError("Official table omits a required annotation/header asset")
        context["allowlist"].update(names)
        write_new(
            output / "prospective_asset_manifest.json", {name: table[name] for name in [*metadata, *names]}
        )
        receipts.extend(receive_group(context, [(name, table[name]) for name in names]))
        guard(context)
        verify_sources(root, admission)
        if any(sha256_file(output / "assets" / row["name"]) != row["sha256"] for row in receipts):
            raise ValueError("An acquired asset changed after transfer")
        if context["requests"] != 184:
            raise ValueError("Final request count differs from184")
        summary.update(
            status="passed",
            assets=receipts,
            asset_manifest_sha256=sha256_file(output / "prospective_asset_manifest.json"),
            record_roles_sha256=sha256_file(output / "record_roles.json"),
            records=90,
            document_components=78,
            biological_identity_verified=False,
            received_table_first_use_sha256=table_receipt["sha256"],
        )
    except Exception as error:
        context["stop"].set()
        summary.update(status="failed", error_type=type(error).__name__, error=str(error))
    return finish(context, summary, prior)
