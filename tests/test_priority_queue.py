import hashlib
import json
import signal

import pytest

from scripts.coordination import run_priority_queue as queue


def fixture_job(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    (tmp_path / "source.py").write_text("frozen = True\n")
    sources = {"source.py": queue.sha256(tmp_path / "source.py")}
    (tmp_path / "sources.json").write_text(json.dumps(sources))
    job = {"name": "test", "output_dir": "out", "sources": "sources.json",
           "sources_sha256": queue.sha256(tmp_path / "sources.json"),
           "stages": [{"name": "profile", "command": ["python", "-u", "-m", "scripts.fake"]},
                      {"name": "train", "command": ["python", "-u", "-m", "scripts.fake"]}],
           "required_artifacts": ["out/result.json"]}
    manifest = {"jobs": [job]}
    runner = queue.Coordinator(manifest, tmp_path / "queue", "fingerprint")
    return job, runner


def test_source_mutation_fails_before_process_launch(tmp_path, monkeypatch):
    job, runner = fixture_job(tmp_path, monkeypatch)
    (tmp_path / "source.py").write_text("changed\n")
    monkeypatch.setattr(queue.subprocess, "Popen", lambda *a, **k: pytest.fail("launched modified code"))
    with pytest.raises(RuntimeError, match="source changed"):
        runner.execute_job(job)


def test_rewriting_source_hash_map_cannot_redefine_frozen_code(tmp_path, monkeypatch):
    job, runner = fixture_job(tmp_path, monkeypatch)
    (tmp_path / "source.py").write_text("changed\n")
    (tmp_path / "sources.json").write_text(json.dumps({"source.py": queue.sha256(tmp_path / "source.py")}))
    monkeypatch.setattr(queue.subprocess, "Popen", lambda *a, **k: pytest.fail("launched changed source map"))
    with pytest.raises(RuntimeError, match="source hash map changed"):
        runner.execute_job(job)


def test_failed_stage_cannot_start_next_stage(tmp_path, monkeypatch):
    job, runner = fixture_job(tmp_path, monkeypatch)
    calls = []

    class Child:
        pid = 12345

        def __init__(self, command, **kwargs):
            calls.append(command)

        def wait(self):
            return 2

    monkeypatch.setattr(queue.subprocess, "Popen", Child)
    with pytest.raises(RuntimeError, match="profile exited 2"):
        runner.execute_job(job)
    assert len(calls) == 1
    assert not (tmp_path / "out/priority_queue_completion.json").exists()


def test_complete_reuse_checks_artifacts(tmp_path, monkeypatch):
    job, runner = fixture_job(tmp_path, monkeypatch)
    calls = []

    class Child:
        pid = 12345

        def __init__(self, command, **kwargs):
            calls.append(command)

        def wait(self):
            (tmp_path / "out/result.json").write_text('{"ok":true}')
            return 0

    monkeypatch.setattr(queue.subprocess, "Popen", Child)
    runner.execute_job(job)
    runner.execute_job(job)
    assert len(calls) == 2
    assert json.loads((tmp_path / "out/coordination.json").read_text())["state"] == "complete"
    (tmp_path / "out/result.json").write_text("changed")
    with pytest.raises(RuntimeError, match="artifact changed"):
        runner.execute_job(job)


def test_failed_predecessor_stops_queue(tmp_path, monkeypatch):
    _, runner = fixture_job(tmp_path, monkeypatch)
    (tmp_path / "previous.json").write_text('{"state":"failed","returncode":1}')
    runner.manifest["predecessor"] = {"pid": 123, "identity": {"start": "old"},
                                      "status_path": "previous.json"}
    monkeypatch.setattr(queue, "identity", lambda pid: None)
    with pytest.raises(RuntimeError, match="did not finish successfully"):
        runner.wait_predecessor()


def test_resume_legacy_only_if_owned_identity_matches(tmp_path, monkeypatch):
    _, runner = fixture_job(tmp_path, monkeypatch)
    expected = {"start": "original", "command": ["scripts.experiments.run_mimic_scale"]}
    runner.manifest["legacy"] = {"pid": 123, "identity": expected}
    runner.paused_legacy = True
    calls = []
    monkeypatch.setattr(queue.os, "kill", lambda *args: calls.append(args))
    monkeypatch.setattr(queue, "identity", lambda pid: {"start": "reused"})
    runner.cleanup()
    assert calls == []
    monkeypatch.setattr(queue, "identity", lambda pid: expected)
    runner.cleanup()
    assert calls == [(123, signal.SIGCONT)]


def test_manifest_requires_unique_explicit_jobs(tmp_path, monkeypatch):
    job, runner = fixture_job(tmp_path, monkeypatch)
    queue.validate_manifest(runner.manifest)
    with pytest.raises(ValueError, match="uniquely"):
        queue.validate_manifest({"jobs": [job, job]})
    job["stages"][0]["command"] = "python -m scripts.fake"
    with pytest.raises(ValueError, match="explicit"):
        queue.validate_manifest({"jobs": [job]})
