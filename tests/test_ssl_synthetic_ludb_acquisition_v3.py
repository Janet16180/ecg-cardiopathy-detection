"""Artificial checks of socket-idle and retained-byte accounting amendments."""

import io
import json
import time

import pytest

from ecg_experiment import ssl_synthetic_ludb_acquisition_v3 as acquisition


@pytest.mark.parametrize(("body", "passed", "retained"), [
    (b"def", True, b"abcdef"), (b"de", False, b"abcde"), (b"defg", False, b"abc")])
def test_only_one_exact_suffix_request_uses_180_seconds(tmp_path, monkeypatch, body, passed, retained):
    config = {"request_url": "https://example.invalid/release", "socket_timeout_seconds": 180,
              "resources": {"received_payload_bytes": 64},
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
            assert timeout == 180
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
    assert len(calls) == summary["requests"] == 1


def test_storage_forecast_uses_one_scan_and_enforces_maximum(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(acquisition.transport, "directory_bytes", lambda root: calls.append(root) or 100)
    config = {"resume": {"expected_complete_bytes": 1000, "prefix_bytes": 300},
              "resources": {"selective_extract_bytes": 1024, "new_output_bytes": 2 * 1024**2}}
    result = acquisition.storage_bound(tmp_path, config)
    assert result["conservative_final_bytes"] == 100 + 700 + 1024 + 1024**2
    assert len(calls) == 1
    config["resources"]["new_output_bytes"] = 1024
    with pytest.raises(ValueError, match="bound"):
        acquisition.storage_bound(tmp_path, config)


def test_exact_adjacent_stage_accounting_cannot_reset_cost():
    previous = {"current_machine_stages": 20, "current_machine_cumulative_cpu_seconds": 10.,
                "current_machine_cumulative_gpu_wall_seconds": 20.}
    current = {"current_machine_stages": 21, "current_machine_cumulative_cpu_seconds": 11.,
               "current_machine_cumulative_gpu_wall_seconds": 20., "prior_ledger_sha256": "old",
               "cpu_seconds": 1., "gpu_wall_seconds": 0}
    acquisition.check_additive(previous, current, "old")
    current["current_machine_cumulative_cpu_seconds"] = 10.
    with pytest.raises(ValueError, match="exactly once"):
        acquisition.check_additive(previous, current, "old")


def test_terminal_accounting_does_not_repeat_directory_walk(tmp_path, monkeypatch):
    def forbidden_scan(root):
        raise AssertionError("Terminal tail cannot walk the full data tree again")

    monkeypatch.setattr(acquisition.transport, "directory_bytes", forbidden_scan)
    monkeypatch.setattr(acquisition, "to_stored", lambda path: path.name)
    config = {"resources": {"cpu_seconds": 60, "wall_seconds": 3600, "rss_bytes": 2**40,
                            "new_output_bytes": 128 * 1024**2, "received_payload_bytes": 64 * 1024**2}}
    admission = {"prior_ledger": {"current_machine_stages": 21, "sha256": "old",
                                 "current_machine_cumulative_cpu_seconds": 10.,
                                 "current_machine_cumulative_gpu_wall_seconds": 20.}}
    summary = {"status": "PASS", "received_payload_bytes": 3}
    clock = {"cpu": time.process_time(), "wall": time.monotonic()}
    acquisition.finish(tmp_path, config, admission, summary, clock, True, 100)
    ledger = json.loads((tmp_path / "prospective_ledger.json").read_text())
    assert ledger["current_machine_stages"] == 22
    assert ledger["current_machine_cumulative_cpu_seconds"] == 10 + ledger["cpu_seconds"]
    assert all(ledger["resource_gates"].values())


def test_interrupted_final_enumeration_retains_actual_cost_ledger(tmp_path, monkeypatch):
    def interrupted_count(root):
        raise TimeoutError("Artificial final-enumeration deadline")

    monkeypatch.setattr(acquisition.transport, "directory_bytes", interrupted_count)
    monkeypatch.setattr(acquisition, "to_stored", lambda path: path.name)
    summary = {"status": "PASS", "received_payload_bytes": 3,
               "storage_bound": {"conservative_final_bytes": 1000000}}
    retained = acquisition.final_enumeration(tmp_path, summary, {})
    assert retained is None
    assert summary["status"] == "FAIL"
    assert not summary["retained_byte_count_exact"]
    assert summary["retained_output_upper_bound_bytes"] == 1000000
    config = {"resources": {"cpu_seconds": 60, "wall_seconds": 3600, "rss_bytes": 2**40,
                            "new_output_bytes": 128 * 1024**2, "received_payload_bytes": 64 * 1024**2}}
    admission = {"prior_ledger": {"current_machine_stages": 21, "sha256": "old",
                                 "current_machine_cumulative_cpu_seconds": 10.,
                                 "current_machine_cumulative_gpu_wall_seconds": 20.}}
    clock = {"cpu": time.process_time(), "wall": time.monotonic()}
    with pytest.raises(RuntimeError, match="failure"):
        acquisition.finish(tmp_path, config, admission, summary, clock, True, retained)
    ledger = json.loads((tmp_path / "prospective_ledger.json").read_text())
    assert ledger["current_machine_stages"] == 22
    assert ledger["current_machine_cumulative_cpu_seconds"] == 10 + ledger["cpu_seconds"]
    assert not ledger["resource_gates"]["output"]
    assert ledger["output_bytes_before_terminal_ledger"] is None
