"""Acquire only public release metadata before declaring any ECGAD binary assets."""

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ecg_experiment import ssl_synthetic_edb_acquisition as shared
from ecg_experiment.files import sha256_file

CONFIG = "configs/ssl_synthetic_ecgad_asset_discovery_v1.json"
PROTOCOL = "docs/ssl-synthetic-localization-ecgad-asset-discovery-v1.md"
SOURCES = (
    CONFIG,
    PROTOCOL,
    "ecg_experiment/__init__.py",
    "ecg_experiment/files.py",
    "ecg_experiment/ssl_synthetic_edb_acquisition.py",
    "ecg_experiment/ssl_synthetic_ecgad_asset_discovery.py",
    "scripts/__init__.py",
    "scripts/data/__init__.py",
    "scripts/data/discover_ssl_synthetic_ecgad_assets.py",
    "tests/test_ssl_synthetic_ecgad_asset_discovery.py",
    "pyproject.toml",
    "uv.lock",
)


def verified_prior(root: Path, declaration: dict[str, Any]) -> dict[str, Any]:
    """Verify the literal immediate terminal ledger and referenced receipt before any GET.

    Parameters
    ----------
    root : Path
        Checkout with locally owned output aliases.
    declaration : dict
        Committed ledger path, identity and cumulative counters.

    Returns
    -------
    dict
        Hash-verified actual prior, never an inferred historical cost.
    """
    path = root / declaration["path"]
    if sha256_file(path) != declaration["sha256"]:
        raise ValueError("Immediate prior ledger identity differs")
    ledger = json.loads(path.read_text())
    for key in (
        "current_machine_stages",
        "current_machine_cumulative_cpu_seconds",
        "current_machine_cumulative_gpu_wall_seconds",
    ):
        if ledger[key] != declaration[key]:
            raise ValueError("Immediate prior cumulative accounting differs")
    receipt = path.parent / shared.safe_name(ledger["receipt"])
    if sha256_file(receipt) != ledger["sha256"] or not ledger["prior_accounting_verified"]:
        raise ValueError("Immediate prior receipt or verified accounting differs")
    if not ledger["technical_passed"] or not all(ledger["resource_gates"].values()):
        raise ValueError("Discovery requires successful immediate source/resource accounting")
    return ledger


def preflight(root: Path, admission_path: Path) -> tuple[dict, dict, dict]:
    """Check committed source/runtime/prior identities without HTTP or payload parsing.

    Parameters
    ----------
    root, admission_path : Path
        Explicit checkout and prospective admission.

    Returns
    -------
    tuple
        Literal configuration, source admission and verified immediate ledger.
    """
    admission = json.loads(admission_path.read_text())
    if not set(SOURCES) <= set(admission["source_sha256"]):
        raise ValueError("Incomplete discovery source/import admission")
    shared.verify_sources(root, admission)
    config = json.loads((root / CONFIG).read_text())
    if config["source"]["maximum_requests"] != 2 + len(config["source_files"]):
        raise ValueError("Exact metadata request roster differs")
    return config, admission, verified_prior(root, admission["prior_ledger"])


def receive_text(context: dict, name: str, url: str) -> dict:
    """Retain one direct text GET, including a partial payload and terminal error receipt.

    Parameters
    ----------
    context : dict
        Source-admitted phase state with shared byte/CPU/wall guards.
    name, url : str
        Explicit safe local name and previously allowlisted HTTPS URL.

    Returns
    -------
    dict
        Original transfer identity, actual bytes and final status.
    """
    shared.safe_name(name)
    if context["allowed_urls"].get(name) != url:
        raise ValueError("Unadmitted metadata URL")
    shared.guard(context)
    output = context["output"]
    receipt = {
        "name": name,
        "url": url,
        "status": "started",
        "received_bytes": 0,
        "declared_sha256": None,
        "identity_kind": "first_use_own_hash",
    }
    shared.write_new(output / "transfers" / f"{name}.started.json", receipt)
    partial = output / "assets" / f"{name}.partial"
    partial.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    response = None
    try:
        context["requests"] += 1
        if context["requests"] > context["config"]["source"]["maximum_requests"]:
            raise ValueError("Metadata request ceiling exceeded")
        request = urllib.request.Request(url, headers={"User-Agent": "ECG-local-research/1"})
        timeout = min(60, context["deadline"] - time.perf_counter())
        response = urllib.request.build_opener(shared.NoRedirect()).open(request, timeout=max(0.001, timeout))
        receipt["http_status"] = response.status
        shared._body(context, response, partial, receipt)
        os.link(partial, output / "assets" / name)
        partial.unlink()
        receipt["status"] = "passed"
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error))
        if isinstance(error, urllib.error.HTTPError):
            receipt["http_status"] = error.code
            error.close()
        if partial.exists():
            receipt.update(partial_bytes=partial.stat().st_size, partial_sha256=sha256_file(partial))
        context["stop"].set()
        raise
    finally:
        if response is not None:
            response.close()
        receipt["wall_seconds"] = time.perf_counter() - started
        shared.write_new(output / "transfers" / f"{name}.terminal.json", receipt)
    return receipt


def source_urls(config: dict, revision: str) -> dict[str, str]:
    """Construct only declared text paths at one validated immutable public source revision."""
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("Require immutable40hex source revision")
    prefix = f"https://raw.githubusercontent.com/{config['repository']}/{revision}/"
    return {"source_" + name.replace("/", "_"): prefix + name for name in config["source_files"]}


def verify_transfers(context: dict, transfers: list[dict]) -> None:
    """Require the complete admitted transfer roster and rehash exact locally retained bytes."""
    expected = set(context["allowed_urls"])
    if context["requests"] != 8 or len(transfers) != 8 or {r["name"] for r in transfers} != expected:
        raise ValueError("Require exactly eight admitted metadata transfers")
    for receipt in transfers:
        path = context["output"] / "assets" / receipt["name"]
        if (
            receipt["status"] != "passed"
            or sha256_file(path) != receipt["sha256"]
            or path.stat().st_size != receipt["received_bytes"]
        ):
            raise ValueError("Retained metadata differs from completed transfer identity")


def discover(root: Path, admission_path: Path, started: tuple[float, float]) -> dict:
    """Run one exclusive metadata discovery and append measured success or failure exactly once.

    Parameters
    ----------
    root, admission_path : Path
        Local checkout and committed source/runtime/prior admission.
    started : tuple
        Monotonic wall and CPU starts captured before controller imports.

    Returns
    -------
    dict
        Actual terminal accounting ledger, with no waveform/annotation/model operation.
    """
    config = json.loads((root / CONFIG).read_text())
    admission = json.loads(admission_path.read_text())
    config["accounting"] = {"initial_ledger_sha256": admission["prior_ledger"]["sha256"]}
    output = root / config["output_directory"]
    output.mkdir(parents=True, exist_ok=False)
    context = {
        "config": config,
        "limits": config["resources"],
        "output": output,
        "wall_start": started[0],
        "cpu_start": started[1],
        "retained_bytes": 0,
        "received_bytes": 0,
        "requests": 0,
        "stop": threading.Event(),
        "mutex": threading.Lock(),
        "deadline": started[0] + config["resources"]["wall_seconds"] - 2,
        "prior_ledger_sha256": admission["prior_ledger"]["sha256"],
        "stage": "ecgad_asset_discovery_v1",
        "allowed_urls": {},
    }
    prior = None
    summary = {"study_id": config["study_id"], "phase": "asset_discovery", "status": "started"}
    try:
        prior = verified_prior(root, admission["prior_ledger"])
        preflight(root, admission_path)
        shared.guard(context)
        for name in admission["source_sha256"]:
            destination = output / "source_snapshot" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write((root / name).read_bytes())
        shared.write_new(output / "admission_snapshot.json", admission)
        context["retained_bytes"] = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
        context["allowed_urls"] = {
            "github_tip.json": config["tip_url"],
            "drive_folder.html": config["folder_url"],
        }
        transfers = [receive_text(context, "github_tip.json", config["tip_url"])]
        revision = json.loads((output / "assets/github_tip.json").read_text())["sha"]
        urls = source_urls(config, revision)
        context["allowed_urls"].update(urls)
        transfers.append(receive_text(context, "drive_folder.html", config["folder_url"]))
        transfers.extend(receive_text(context, name, url) for name, url in urls.items())
        verify_transfers(context, transfers)
        shared.verify_sources(root, admission)
        verified_prior(root, admission["prior_ledger"])
        summary.update(
            status="passed",
            source_revision=revision,
            assets=transfers,
            source_sha256=admission["source_sha256"],
            runtime=admission["runtime"],
            scientific_payload_downloads=0,
            binary_assets_present=False,
            clinical_claim=False,
            all_roles_spent_development=True,
            output_sha256={
                p.relative_to(output).as_posix(): sha256_file(p) for p in output.rglob("*") if p.is_file()
            },
        )
    except Exception as error:
        summary.update(status="failed", error_type=type(error).__name__, error=str(error))
    return shared.finish(context, summary, prior)
