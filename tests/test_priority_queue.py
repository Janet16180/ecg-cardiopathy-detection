"""Frozen priority queue: source pinning, stage failures and completion reuse."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from ecg_experiment import processes
from ecg_experiment.files import sha256_file
from scripts.coordination import common
from scripts.coordination import run_priority_queue as queue


def fixture_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], queue.Coordinator]:
    """
    Build a two-stage job with a frozen source map under a fake root.

    Parameters
    ----------
    tmp_path : Path
        Fake repository root.
    monkeypatch : pytest.MonkeyPatch
        Used to point the coordinator at ``tmp_path``.

    Returns
    -------
    tuple[dict[str, Any], queue.Coordinator]
        The job and a coordinator for a manifest holding only it.
    """
    monkeypatch.setattr(common, "ROOT", tmp_path)
    (tmp_path / "source.py").write_text("frozen = True\n")
    sources = {"source.py": sha256_file(tmp_path / "source.py")}
    (tmp_path / "sources.json").write_text(json.dumps(sources))
    job = {"name": "test", "output_dir": "out", "sources": "sources.json",
           "sources_sha256": sha256_file(tmp_path / "sources.json"),
           "stages": [{"name": "profile", "command": ["python", "-u", "-m", "scripts.fake"]},
                      {"name": "train", "command": ["python", "-u", "-m", "scripts.fake"]}],
           "required_artifacts": ["out/result.json"]}
    manifest = {"jobs": [job]}
    runner = queue.Coordinator(manifest, tmp_path / "queue", "fingerprint")
    return job, runner


def test_source_mutation_fails_before_process_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job, runner = fixture_job(tmp_path, monkeypatch)
    (tmp_path / "source.py").write_text("changed\n")
    monkeypatch.setattr(queue.subprocess, "Popen", lambda *a, **k: pytest.fail("launched modified code"))
    with pytest.raises(RuntimeError, match="source changed"):
        runner.execute_job(job)


def test_rewriting_source_hash_map_cannot_redefine_frozen_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job, runner = fixture_job(tmp_path, monkeypatch)
    (tmp_path / "source.py").write_text("changed\n")
    (tmp_path / "sources.json").write_text(json.dumps({"source.py": sha256_file(tmp_path / "source.py")}))
    monkeypatch.setattr(queue.subprocess, "Popen", lambda *a, **k: pytest.fail("launched changed source map"))
    with pytest.raises(RuntimeError, match="source hash map changed"):
        runner.execute_job(job)


def test_failed_stage_cannot_start_next_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_coordinator_fingerprint_covers_shared_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    before = queue.coordinator_sha256()
    copy = tmp_path / "common.py"
    copy.write_text(Path(common.__file__).read_text() + "\n")
    monkeypatch.setattr(common, "__file__", str(copy))
    assert queue.coordinator_sha256() != before


def test_complete_reuse_checks_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_failed_predecessor_stops_queue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, runner = fixture_job(tmp_path, monkeypatch)
    (tmp_path / "previous.json").write_text('{"state":"failed","returncode":1}')
    runner.manifest["predecessor"] = {"pid": 123, "identity": {"start": "old", "command": []},
                                      "status_path": "previous.json"}
    monkeypatch.setattr(processes, "process_identity", lambda pid: None)
    with pytest.raises(RuntimeError, match="did not finish successfully"):
        runner.wait_predecessor()


def test_cleanup_resumes_only_the_paused_legacy_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, runner = fixture_job(tmp_path, monkeypatch)
    legacy = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        identity = processes.process_identity(legacy.pid)
        runner.legacy = processes.IdleOrchestrator(legacy.pid, identity, tmp_path / "status.json",
                                                   tmp_path / "ready")
        processes.stop_process(legacy.pid, identity)
        runner.cleanup()
        assert processes.is_stopped(legacy.pid)
        runner.legacy.paused = True
        runner.cleanup()
        assert not processes.is_stopped(legacy.pid)
    finally:
        processes.terminate_child(legacy, timeout=5)


def test_manifest_requires_unique_explicit_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job, runner = fixture_job(tmp_path, monkeypatch)
    queue.validate_manifest(runner.manifest)
    with pytest.raises(ValueError, match="uniquely"):
        queue.validate_manifest({"jobs": [job, job]})
    job["stages"][0]["command"] = "python -m scripts.fake"
    with pytest.raises(ValueError, match="explicit"):
        queue.validate_manifest({"jobs": [job]})
