"""Artificial HTTP, child-resource and honest split-accounting checks without network access."""

import json
import time

import pytest

from ecg_experiment import ssl_synthetic_ludb_acquisition_v4 as acquisition


def time_log(tmp_path, exit_status=0):
    path = tmp_path / "time.txt"
    path.write_text('Command being timed: "curl public-release"\n'
                    'User time (seconds): 0.12\nSystem time (seconds): 0.03\n'
                    'Elapsed (wall clock) time (h:mm:ss or m:ss): 1:02.50\n'
                    'Maximum resident set size (kbytes): 1024\n'
                    f'Exit status: {exit_status}\n')
    return path


def test_verbose_child_metrics_preserve_logged_precision(tmp_path):
    result = acquisition.transfer_metrics(time_log(tmp_path))
    assert result["cpu_seconds"] == .15
    assert result["wall_seconds"] == 62.5
    assert result["peak_rss_bytes"] == 1024**2
    assert result["exit_status"] == 0
    assert acquisition.elapsed_seconds("1:02:03.25") == 3723.25


def test_signalled_child_is_not_mistaken_for_success(tmp_path):
    path = time_log(tmp_path)
    with path.open("a") as stream:
        stream.write("Command terminated by signal 15\n")
    assert acquisition.transfer_metrics(path)["terminated_by_signal"]


def test_connect_and_informational_headers_keep_only_final_response(tmp_path):
    path = tmp_path / "headers.txt"
    path.write_bytes(b'HTTP/1.1 200 Connection established\r\n\r\nHTTP/1.1 100 Continue\r\n\r\n'
                     b'HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 3-5/6\r\nETag: "fixed"\r\n\r\n')
    result = acquisition.response_headers(path)
    assert result["status"] == 206
    assert result["headers"]["content-range"] == "bytes 3-5/6"


def test_redirect_history_is_rejected(tmp_path):
    path = tmp_path / "headers.txt"
    path.write_text("HTTP/1.1 302 Found\nLocation: elsewhere\n\nHTTP/1.1 206 Partial Content\n\n")
    with pytest.raises(ValueError, match="redirect"):
        acquisition.response_headers(path)


@pytest.mark.parametrize(("status", "etag", "body"), [(206, '"fixed"', b"def"),
                                                         (200, '"fixed"', b"def"),
                                                         (206, '"changed"', b"def"),
                                                         (206, '"fixed"', b"de")])
def test_offline_suffix_verification_before_assembly(tmp_path, status, etag, body):
    time_log(tmp_path)
    (tmp_path / "headers.txt").write_text(f"HTTP/1.1 {status} Status\nContent-Range: bytes 3-5/6\n"
                                          f"Content-Length: 3\nETag: {etag}\n\n")
    (tmp_path / "suffix").write_bytes(body)
    config = {"curl": {"resource_log": "time.txt", "response_headers": "headers.txt",
                       "suffix_file": "suffix"},
              "resume": {"expected_complete_bytes": 6, "prefix_bytes": 3,
                         "required_content_range": "bytes 3-5/6", "if_range_etag": "fixed"},
              "resources": {"external_curl": {"cpu_seconds": 60, "wall_seconds": 3600},
                            "total": {"rss_bytes": 2**30}}}
    if status == 206 and etag == '"fixed"' and body == b"def":
        metrics, response = acquisition.verify_transfer(tmp_path, config)
        assert metrics["cpu_seconds"] == .15
        assert response["status"] == 206
    else:
        with pytest.raises(ValueError, match="identity|length"):
            acquisition.verify_transfer(tmp_path, config)


def test_prestart_zero_requires_no_external_artifacts(tmp_path):
    paths = [tmp_path / "suffix", tmp_path / "headers", tmp_path / "time"]
    assert acquisition.proven_unstarted(paths)["cpu_seconds"] == 0
    paths[0].write_bytes(b"partial")
    with pytest.raises(ValueError, match="unstarted"):
        acquisition.proven_unstarted(paths)


def test_unknown_child_runtime_retains_known_cpu_without_fabricating_total(tmp_path, monkeypatch):
    monkeypatch.setattr(acquisition, "to_stored", lambda path: path.name)
    config = {"resources": {"offline_prepare_and_finish_combined": {"cpu_seconds": 60,
                              "active_wall_seconds": 600},
                            "external_curl": {"cpu_seconds": 60, "wall_seconds": 3600},
                            "total": {"cpu_seconds": 120, "summed_active_wall_seconds": 4200,
                                      "rss_bytes": 2**40, "new_output_bytes": 128 * 1024**2}}}
    admission = {"prior_ledger": {"sha256": "old", "current_machine_cumulative_cpu_seconds": 10.,
                                 "current_machine_cumulative_gpu_wall_seconds": 20.}}
    prepared = {"cpu_seconds": .2, "wall_seconds": .3, "peak_rss_bytes": 1000}
    clock = {"cpu": time.process_time(), "wall": time.monotonic()}
    with pytest.raises(RuntimeError, match="failure"):
        acquisition.completion(tmp_path, config, admission, {"status": "FAIL"}, prepared,
                               None, clock, True, 100)
    ledger = json.loads((tmp_path / "prospective_ledger.json").read_text())
    assert ledger["cpu_seconds"] is None
    assert ledger["current_machine_cumulative_cpu_seconds"] is None
    assert ledger["known_current_machine_cumulative_cpu_seconds"] >= 10.2
    assert not ledger["resource_gates"]["total_cpu"]
