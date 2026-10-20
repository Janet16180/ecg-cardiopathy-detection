"""Artificial transport checks; no HTTP requests or actual LUDB reads."""

import io
from types import SimpleNamespace

import pytest

from ecg_experiment import ssl_synthetic_ludb_acquisition_v2 as acquisition


@pytest.fixture
def config():
    return {"resume": {"expected_complete_bytes": 14474484, "prefix_bytes": 6291456,
                       "required_content_range": "bytes 6291456-14474483/14474484"}}


@pytest.mark.parametrize("length", [None, "8183028"])
def test_exact_partial_response_admitted(config, length):
    headers = {"Content-Range": config["resume"]["required_content_range"]}
    if length is not None:
        headers["Content-Length"] = length
    assert acquisition.verify_response(SimpleNamespace(status=206, headers=headers), config) == 8183028


@pytest.mark.parametrize(("status", "content_range", "length"), [
    (200, "bytes 6291456-14474483/14474484", "8183028"),
    (206, "bytes 0-14474483/14474484", "8183028"),
    (206, "bytes 6291456-14474482/14474484", "8183028"),
    (206, "bytes 6291456-14474483/14474485", "8183028"),
    (206, "bytes 6291456-14474483/14474484", "8183029"),
    (206, None, None),
])
def test_wrong_response_rejected_before_body(config, status, content_range, length):
    headers = {"Content-Range": content_range, "Content-Length": length}
    with pytest.raises(ValueError, match="HTTP206|Content-Range|Content-Length"):
        acquisition.verify_response(SimpleNamespace(status=status, headers=headers), config)


def test_redirect_forbidden():
    with pytest.raises(ValueError, match="single-GET"):
        acquisition.NoRedirect().redirect_request(None, None, 302, "Found", {}, "https://example.com")


def test_member_stream_bounded_and_failure_bytes_retained():
    target = io.BytesIO()
    with pytest.raises(ValueError, match="exceeds"):
        acquisition.copy_member(io.BytesIO(b"abcdef"), target, 5, lambda: None)
    assert target.getvalue() == b""
    with pytest.raises(ValueError, match="shorter"):
        acquisition.copy_member(io.BytesIO(b"abc"), target, 5, lambda: None)
    assert target.getvalue() == b"abc"


def test_write_new_refuses_overwrite(tmp_path):
    target = tmp_path / "summary.json"
    acquisition.write_new(target, {"status": "FAIL"})
    with pytest.raises(FileExistsError):
        acquisition.write_new(target, {"status": "PASS"})
    assert target.read_text().find("FAIL") > 0


def test_manifest_rejects_duplicate_or_invalid_identity(tmp_path):
    path = tmp_path / "SHA256SUMS.txt"
    path.write_text("a" * 64 + " data/1.dat\n" + "a" * 64 + " data/1.dat\n")
    with pytest.raises(ValueError, match="duplicate"):
        acquisition.checksum_manifest(path)


def test_lineage_cannot_skip_failed_or_audit_cost():
    source = {"current_machine_stages": 19, "sha256": "old", "current_machine_cumulative_cpu_seconds": 10.,
              "current_machine_cumulative_gpu_wall_seconds": 20.}
    ledger = {"current_machine_stages": 20, "prior_ledger_sha256": "old", "receipt": "summary.json",
              "sha256": "audit", "current_machine_cumulative_cpu_seconds": 11., "cpu_seconds": 1.,
              "current_machine_cumulative_gpu_wall_seconds": 20., "gpu_wall_seconds": 0,
              "resource_gates": {"cpu": True}}
    config = {"source_predecessor_ledger": source}
    admission = {"failure_audit_summary": {"sha256": "audit"}}
    audit = {"input_sha256": {"prospective_ledger.json": "old"}, "acquisition_failure_preserved": True}
    acquisition.verify_lineage(config, admission, ledger, audit)
    ledger["current_machine_cumulative_cpu_seconds"] = 10.
    with pytest.raises(ValueError, match="exactly once"):
        acquisition.verify_lineage(config, admission, ledger, audit)


@pytest.mark.parametrize(("body", "passed", "retained"), [
    (b"def", True, b"abcdef"), (b"de", False, b"abcde"), (b"defg", False, b"abc")])
def test_one_range_body_exact_or_retained_failure(tmp_path, monkeypatch, body, passed, retained):
    config = {"request_url": "https://example.invalid/release", "resources": {"received_payload_bytes": 64},
              "resume": {"expected_complete_bytes": 6, "prefix_bytes": 3, "prefix_file": "prefix.partial",
                         "required_content_range": "bytes 3-5/6", "range_header": "bytes=3-",
                         "if_range_etag": "fixed"}}
    calls = []

    class Response(io.BytesIO):
        status = 206
        url = config["request_url"]
        headers = {"Content-Range": "bytes 3-5/6"}

    class Opener:
        def open(self, request, timeout):
            calls.append(request)
            assert request.get_header("Range") == "bytes=3-"
            assert request.get_header("If-range") == '"fixed"'
            assert timeout == 30
            return Response(body)

    monkeypatch.setattr(acquisition.urllib.request, "build_opener", lambda handler: Opener())
    path = tmp_path / "prefix.partial"
    path.write_bytes(b"abc")
    summary = {"received_payload_bytes": 0}
    if passed:
        final = acquisition.receive(tmp_path, config, summary, lambda: None)
        assert final.read_bytes() == retained
        assert not path.exists()
    else:
        with pytest.raises(ValueError, match="shorter|exceeds"):
            acquisition.receive(tmp_path, config, summary, lambda: None)
        assert path.read_bytes() == retained
    assert len(calls) == 1
    assert summary["requests"] == 1
