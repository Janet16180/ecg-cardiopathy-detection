"""Artificial metadata access, failure-retention and literal accounting checks."""

import json
import threading
import time
import urllib.error
from pathlib import Path

import pytest

from ecg_experiment import ssl_synthetic_ecgad_asset_discovery as discovery
from ecg_experiment import ssl_synthetic_edb_acquisition as shared
from ecg_experiment.files import sha256_file


def context(tmp_path: Path) -> dict:
    """Create an artificial bounded phase with no real URL or payload."""
    return {
        "config": {"source": {"maximum_requests": 1}},
        "output": tmp_path,
        "limits": {"cpu_seconds": 60, "wall_seconds": 60, "rss_bytes": 2**40, "output_bytes": 8 * 2**20},
        "wall_start": time.perf_counter(),
        "cpu_start": time.process_time(),
        "retained_bytes": 0,
        "received_bytes": 0,
        "requests": 0,
        "stop": threading.Event(),
        "deadline": time.perf_counter() + 58,
        "allowed_urls": {"metadata.txt": "https://example.invalid/metadata.txt"},
    }


def test_unadmitted_url_causes_no_request_or_receipt(tmp_path, monkeypatch):
    """Reject URL substitution before a network call or creation of scientific receipts."""
    calls = []
    monkeypatch.setattr(discovery.urllib.request, "build_opener", lambda *args: calls.append(args))
    phase = context(tmp_path)
    with pytest.raises(ValueError, match="Unadmitted"):
        discovery.receive_text(phase, "metadata.txt", "https://example.invalid/other")
    assert phase["requests"] == 0
    assert calls == []
    assert list(tmp_path.iterdir()) == []


def test_http_failure_is_retained_and_stops_phase(tmp_path, monkeypatch):
    """A failed official metadata request cannot disappear or trigger an implicit retry."""

    class Opener:
        def open(self, request, timeout):
            raise urllib.error.HTTPError(request.full_url, 403, "denied", {}, None)

    monkeypatch.setattr(discovery.urllib.request, "build_opener", lambda *args: Opener())
    phase = context(tmp_path)
    with pytest.raises(urllib.error.HTTPError):
        discovery.receive_text(phase, "metadata.txt", phase["allowed_urls"]["metadata.txt"])
    receipt = json.loads((tmp_path / "transfers/metadata.txt.terminal.json").read_text())
    assert phase["requests"] == 1
    assert phase["stop"].is_set()
    assert receipt["status"] == "failed"
    assert receipt["http_status"] == 403
    assert receipt["received_bytes"] == 0
    assert receipt["declared_sha256"] is None


def test_partial_failure_retains_exact_payload_and_hash(tmp_path, monkeypatch):
    """Bytes received before a bounded-body failure remain auditable without a success asset."""

    class Response:
        status = 200
        closed = False

        def close(self):
            self.closed = True

    response = Response()

    class Opener:
        def open(self, request, timeout):
            return response

    def body(phase, response, partial, receipt):
        partial.write_bytes(b"truncated")
        phase["received_bytes"] += 9
        receipt["received_bytes"] += 9
        raise ValueError("bounded test failure")

    monkeypatch.setattr(discovery.urllib.request, "build_opener", lambda *args: Opener())
    monkeypatch.setattr(shared, "_body", body)
    phase = context(tmp_path)
    with pytest.raises(ValueError, match="bounded"):
        discovery.receive_text(phase, "metadata.txt", phase["allowed_urls"]["metadata.txt"])
    path = tmp_path / "assets/metadata.txt.partial"
    receipt = json.loads((tmp_path / "transfers/metadata.txt.terminal.json").read_text())
    assert path.read_bytes() == b"truncated"
    assert receipt["partial_sha256"] == sha256_file(path)
    assert receipt["partial_bytes"] == receipt["received_bytes"] == 9
    assert response.closed
    assert not (tmp_path / "assets/metadata.txt").exists()


@pytest.mark.parametrize("revision", ["main", "a" * 39, "A" * 40, "../bad", "a" * 40 + "?raw=1"])
def test_source_urls_require_immutable_revision(revision):
    """Mutable refs and query/path injection cannot become admitted text source URLs."""
    with pytest.raises(ValueError, match="immutable"):
        discovery.source_urls({"repository": "authors/project", "source_files": ["LICENSE"]}, revision)


def test_exact_source_roster_uses_one_revision():
    """Only the declared source text list is constructed from the recovered revision."""
    config = {"repository": "authors/project", "source_files": ["LICENSE", "data/preprocess.py"]}
    urls = discovery.source_urls(config, "a" * 40)
    assert set(urls) == {"source_LICENSE", "source_data_preprocess.py"}
    assert all("/" + "a" * 40 + "/" in value for value in urls.values())


def test_prior_requires_exact_receipt_and_counters(tmp_path):
    """A ledger with changed receipt or reset cumulative cost cannot authorize discovery."""
    receipt = tmp_path / "summary.json"
    shared.write_new(receipt, {"status": "passed"})
    path = tmp_path / "ledger.json"
    ledger = {
        "receipt": "summary.json",
        "sha256": sha256_file(receipt),
        "prior_accounting_verified": True,
        "technical_passed": True,
        "resource_gates": {"cpu": True},
        "current_machine_stages": 44,
        "current_machine_cumulative_cpu_seconds": 1234.5,
        "current_machine_cumulative_gpu_wall_seconds": 500.0,
    }
    shared.write_new(path, ledger)
    declaration = {
        "path": "ledger.json",
        "sha256": sha256_file(path),
        **{k: v for k, v in ledger.items() if k.startswith("current_machine_")},
    }
    assert discovery.verified_prior(tmp_path, declaration) == ledger
    with pytest.raises(ValueError, match="cumulative"):
        discovery.verified_prior(tmp_path, {**declaration, "current_machine_cumulative_cpu_seconds": 0})
    receipt.write_text("{}\n")
    with pytest.raises(ValueError, match="receipt"):
        discovery.verified_prior(tmp_path, declaration)


def test_final_gate_rejects_asset_mutation_and_incomplete_roster(tmp_path):
    """Previously completed metadata must still match its receipt at the final PASS gate."""
    assets = tmp_path / "assets"
    assets.mkdir()
    transfers = []
    urls = {}
    for index in range(8):
        name = f"metadata{index}.txt"
        path = assets / name
        path.write_bytes(b"source text")
        urls[name] = "https://example.invalid/" + name
        transfers.append(
            {
                "name": name,
                "status": "passed",
                "sha256": sha256_file(path),
                "received_bytes": path.stat().st_size,
            }
        )
    phase = {"requests": 8, "allowed_urls": urls, "output": tmp_path}
    discovery.verify_transfers(phase, transfers)
    with pytest.raises(ValueError, match="exactly eight"):
        discovery.verify_transfers(phase, transfers[:-1])
    (assets / "metadata0.txt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Retained metadata"):
        discovery.verify_transfers(phase, transfers)
