"""Artificial-only contracts for prospective annotation-only HTTP acquisition."""

import hashlib
import io
import json
import threading
import time
import urllib.error
from types import SimpleNamespace

import pytest

from ecg_experiment import ROOT
from ecg_experiment import ssl_synthetic_edb_acquisition as acq


def config():
    """Read only the committed protocol configuration used by artificial fixtures."""
    return json.loads((ROOT / acq.CONFIG).read_text())


def context(tmp_path):
    """Create an artificial, zero-network phase context."""
    policy = config()
    return {
        "config": policy,
        "output": tmp_path,
        "limits": policy["resources"]["acquisition"],
        "wall_start": time.perf_counter(),
        "cpu_start": time.process_time(),
        "requests": 0,
        "received_bytes": 0,
        "retained_bytes": 0,
        "mutex": threading.Lock(),
        "stop": threading.Event(),
        "allowlist": {"RECORDS"},
        "deadline": time.perf_counter() + 10,
    }


class Response:
    """A fake HTTP body reproducing response.fp closure on the last read."""

    def __init__(self, body, declared_length=None, status=200):
        self.body = io.BytesIO(body)
        self.status = status
        self.headers = {"Content-Length": str(len(body) if declared_length is None else declared_length)}
        self.fp = SimpleNamespace(raw=SimpleNamespace(_sock=SimpleNamespace(settimeout=lambda value: None)))

    def geturl(self):
        return config()["source"]["file_base_url"] + "RECORDS"

    def read1(self, size):
        result = self.body.read(size)
        if self.body.tell() == len(self.body.getvalue()):
            self.fp = None
        return result

    def close(self):
        self.body.close()


def test_official_table_rejects_duplicate_and_unsafe_names():
    """Do not silently overwrite declarations or accept asset path traversal."""
    line = "a" * 64 + "  RECORDS\n"
    assert acq.checksum_table(line.encode()) == {"RECORDS": "a" * 64}
    with pytest.raises(ValueError, match="Duplicate"):
        acq.checksum_table((line + line).encode())
    for name in ["../e0103.atr", "/e0103.atr", "x/e0103.atr", "x\\e0103.atr"]:
        with pytest.raises(ValueError, match="Noncanonical"):
            acq.safe_name(name)


def test_document_components_have_one_role_before_any_annotation():
    """Published linkage, not record count, determines prospective role allocation."""
    policy = config()
    linked = [n for group in policy["grouping"]["explicit_groups"] for n in group]
    names = linked + [f"e{index:04d}" for index in range(1000, 1000 + 90 - len(linked))]
    rows = acq.record_roles(("\n".join(names) + "\n").encode(), policy)
    assert len(rows) == 90
    assert len({r["component"] for r in rows}) == 78
    for group in policy["grouping"]["explicit_groups"]:
        assert len({r["role"] for r in rows if r["record"] in group}) == 1
    with pytest.raises(ValueError, match="unique"):
        acq.record_roles(("\n".join(names[:-1] + names[:1])).encode(), policy)


def test_exact_asset_success_is_exclusive_and_counts_one_get(tmp_path, monkeypatch):
    """Hash-check and retain a complete body; repeated use cannot overwrite it."""
    body = b"artificial official record names\n"
    opens = []

    def open_response(request, timeout):
        opens.append((request.full_url, timeout))
        return Response(body)

    monkeypatch.setattr(
        acq.urllib.request, "build_opener", lambda handler: SimpleNamespace(open=open_response)
    )
    state = context(tmp_path)
    row = acq.receive(state, "RECORDS", hashlib.sha256(body).hexdigest())
    assert row["status"] == "passed"
    assert state["requests"] == 1
    assert state["received_bytes"] == len(body)
    assert (tmp_path / "assets/RECORDS").read_bytes() == body
    assert len(opens) == 1
    with pytest.raises(FileExistsError):
        acq.receive(state, "RECORDS", row["sha256"])
    assert len(opens) == 1


@pytest.mark.parametrize(("wrong_hash", "declared_length"), [(True, None), (False, 100)])
def test_failed_asset_retains_partial_and_terminal_identity(
    tmp_path, monkeypatch, wrong_hash, declared_length
):
    """Wrong hashes and truncated bodies cannot be silently admitted or retried."""
    body = b"partial artificial annotation-free bytes"
    response = Response(body, declared_length)
    monkeypatch.setattr(
        acq.urllib.request, "build_opener", lambda handler: SimpleNamespace(open=lambda *a, **k: response)
    )
    expected = "f" * 64 if wrong_hash else hashlib.sha256(body).hexdigest()
    state = context(tmp_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch|Content-Length"):
        acq.receive(state, "RECORDS", expected)
    receipt = json.loads((tmp_path / "transfers/RECORDS.terminal.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["partial_bytes"] == len(body)
    assert receipt["partial_sha256"] == hashlib.sha256(body).hexdigest()
    assert state["requests"] == 1
    assert state["stop"].is_set()


def test_waveform_and_redirect_never_trigger_a_followup_get(tmp_path, monkeypatch):
    """Reject prohibited DAT before HTTP and redirects without following them."""
    state = context(tmp_path)
    with pytest.raises(ValueError, match="allowlist"):
        acq.receive(state, "e0103.dat", None)
    assert state["requests"] == 0
    with pytest.raises(urllib.error.HTTPError) as error:
        acq.NoRedirect().redirect_request(
            SimpleNamespace(full_url="https://physionet.org/files/edb/1.0.0/RECORDS"),
            None,
            302,
            "Moved",
            {},
            "https://example.org/RECORDS",
        )
    assert error.value.code == 302


def test_source_roster_preflight_fails_before_any_asset_io(tmp_path):
    """Omitting required source obligations is caught before opening an HTTP request."""
    with pytest.raises(ValueError, match="incomplete"):
        acq.verify_sources(tmp_path, {"source_sha256": {}})


def test_prior_requires_literal_stage40_not_an_older_good_ledger(tmp_path):
    """A valid-looking older cumulative ledger cannot reset prospective cost history."""
    policy = config()
    with pytest.raises(ValueError, match="actual40"):
        acq.verify_prior(tmp_path, {"prior_ledger": {}}, policy)


def test_received_cap_failure_preserves_and_counts_the_crossing_chunk(tmp_path, monkeypatch):
    """A cap-crossing received chunk remains counted, hashed and locally retained."""
    body = b"too-large artificial body"
    response = Response(body)
    monkeypatch.setattr(
        acq.urllib.request, "build_opener", lambda handler: SimpleNamespace(open=lambda *a, **k: response)
    )
    state = context(tmp_path)
    state["config"]["source"]["received_payload_bytes"] = len(body) - 1
    with pytest.raises(ValueError, match="Received-payload"):
        acq.receive(state, "RECORDS", hashlib.sha256(body).hexdigest())
    row = json.loads((tmp_path / "transfers/RECORDS.terminal.json").read_text())
    assert row["received_bytes"] == len(body)
    assert state["received_bytes"] == len(body)
    assert row["partial_bytes"] == len(body)
    assert row["received_sha256"] == hashlib.sha256(body).hexdigest()
    assert row["unretained_received_bytes"] == 0


def test_existing_final_asset_is_never_replaced(tmp_path, monkeypatch):
    """Exclusive publication cannot replace even an unexpected existing final asset."""
    body = b"artificial new bytes"
    response = Response(body)
    monkeypatch.setattr(
        acq.urllib.request, "build_opener", lambda handler: SimpleNamespace(open=lambda *a, **k: response)
    )
    asset = tmp_path / "assets/RECORDS"
    asset.parent.mkdir()
    asset.write_bytes(b"existing immutable bytes")
    with pytest.raises(FileExistsError):
        acq.receive(context(tmp_path), "RECORDS", hashlib.sha256(body).hexdigest())
    assert asset.read_bytes() == b"existing immutable bytes"
    assert (tmp_path / "assets/RECORDS.partial").read_bytes() == body


@pytest.mark.parametrize("explicit_phase", [False, True])
def test_terminal_phase_counters_and_prior_are_explicit(tmp_path, explicit_phase):
    """The separately charged parser cannot inherit acquisition's zero-decode claim."""
    state = context(tmp_path)
    if explicit_phase:
        state.update(
            stage="edb_annotation_qualification_v1", prior_ledger_sha256="b" * 64, annotation_decodes=90
        )
    prior = {
        "current_machine_stages": 41,
        "current_machine_cumulative_cpu_seconds": 10.0,
        "current_machine_cumulative_gpu_wall_seconds": 20.0,
    }
    ledger = acq.finish(state, {"status": "passed"}, prior)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["new_annotation_decodes"] == (90 if explicit_phase else 0)
    assert ledger["stage"] == (
        "edb_annotation_qualification_v1" if explicit_phase else "edb_annotation_acquisition_v1"
    )
    assert ledger["prior_ledger_sha256"] == (
        "b" * 64 if explicit_phase else state["config"]["accounting"]["initial_ledger_sha256"]
    )
    assert ledger["current_machine_stages"] == 42
    assert ledger["current_machine_cumulative_cpu_seconds"] == 10.0 + ledger["cpu_seconds"]
    assert ledger["current_machine_cumulative_gpu_wall_seconds"] == 20.0


def test_terminal_enumeration_failure_preserves_failed_ledger(tmp_path, monkeypatch):
    """A failed final byte enumeration retains measured costs and cannot pass admission."""
    state = context(tmp_path)
    prior = {
        "current_machine_stages": 40,
        "current_machine_cumulative_cpu_seconds": 10.0,
        "current_machine_cumulative_gpu_wall_seconds": 20.0,
    }

    def broken_enumeration(self, pattern):
        raise OSError("Artificial final byte enumeration failure")

    monkeypatch.setattr(type(tmp_path), "rglob", broken_enumeration)
    ledger = acq.finish(state, {"status": "passed"}, prior)
    assert ledger["receipt"] == "finalization_failure.json"
    assert not ledger["technical_passed"]
    assert not ledger["resource_gates"]["output"]
    assert ledger["output_bytes_before_terminal_ledger"] is None
    assert ledger["current_machine_stages"] == 41
    assert ledger["current_machine_cumulative_cpu_seconds"] == 10.0 + ledger["cpu_seconds"]
    assert json.loads((tmp_path / ledger["receipt"]).read_text())["finalization_failure"]
