import json
import signal

import pytest

from scripts import handoff_priority_queue as handoff
from scripts import run_priority_queue as queue


OLD_SHA = "a" * 64
IDENTITY = {"start": "123", "command": ["python", "-u", "-m", "scripts.run_priority_queue"]}


def setup_old(tmp_path, monkeypatch, state="profiling", experiment="015_cached_jepa_distillation"):
    monkeypatch.setattr(handoff, "ROOT", tmp_path)
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    old_path = tmp_path / "old/queue.json"
    old_path.parent.mkdir()
    first = {"name": "015_cached_jepa_distillation", "output_dir": "015",
             "required_artifacts": ["015/result.json"]}
    second = {"name": "017_morphology_templates", "output_dir": "017"}
    old = {"jobs": [first, second]}
    queue.atomic_json(old_path.parent / "status.json",
                      {"queue_manifest_sha256": OLD_SHA, "wrapper_pid": 41,
                       "experiment": experiment, "state": state})
    (tmp_path / "015").mkdir()
    (tmp_path / "015/result.json").write_text("verified")
    return old, old_path


def complete_015(tmp_path):
    queue.atomic_json(tmp_path / "015/coordination.json",
                      {"queue_manifest_sha256": OLD_SHA, "state": "complete", "returncode": 0})
    queue.atomic_json(tmp_path / "015/priority_queue_completion.json",
                      {"queue_manifest_sha256": OLD_SHA,
                       "artifacts": {"015/result.json": queue.sha256(tmp_path / "015/result.json")}})


def test_never_signals_015_when_015_failed(tmp_path, monkeypatch):
    old, old_path = setup_old(tmp_path, monkeypatch, state="failed")
    monkeypatch.setattr(queue, "identity", lambda pid: IDENTITY)
    monkeypatch.setattr(handoff.os, "kill", lambda *args: pytest.fail("signaled 015"))
    with pytest.raises(RuntimeError, match="015 failed"):
        handoff.wait_handoff_gate(old, old_path, OLD_SHA, 41, IDENTITY, 0.01)


def test_017_status_cannot_bypass_015_artifact_gate(tmp_path, monkeypatch):
    old, old_path = setup_old(tmp_path, monkeypatch, experiment="017_morphology_templates")
    complete_015(tmp_path)
    (tmp_path / "015/result.json").write_text("changed")
    monkeypatch.setattr(queue, "identity", lambda pid: IDENTITY)
    monkeypatch.setattr(handoff.os, "kill", lambda *args: pytest.fail("signaled before artifact gate"))
    with pytest.raises(RuntimeError, match="artifact changed"):
        handoff.wait_handoff_gate(old, old_path, OLD_SHA, 41, IDENTITY, 0.01)


def test_identity_change_refuses_signal(tmp_path, monkeypatch):
    old, old_path = setup_old(tmp_path, monkeypatch, experiment="017_morphology_templates")
    complete_015(tmp_path)
    monkeypatch.setattr(queue, "identity", lambda pid: {"start": "reused", "command": IDENTITY["command"]})
    monkeypatch.setattr(handoff.os, "kill", lambda *args: pytest.fail("signaled reused PID"))
    with pytest.raises(RuntimeError, match="identity changed"):
        handoff.wait_handoff_gate(old, old_path, OLD_SHA, 41, IDENTITY, 0.01)
    with pytest.raises(RuntimeError, match="identity changed"):
        handoff.stop_old(41, IDENTITY, 1, 0.01)


def test_verified_015_allows_017_stop_only(tmp_path, monkeypatch):
    old, old_path = setup_old(tmp_path, monkeypatch, experiment="017_morphology_templates")
    complete_015(tmp_path)
    monkeypatch.setattr(queue, "identity", lambda pid: IDENTITY)
    assert handoff.wait_handoff_gate(old, old_path, OLD_SHA, 41, IDENTITY, 0.01) == "stop_old_017"


def test_old_exited_after_017_completion_refuses_duplicate(tmp_path, monkeypatch):
    old, old_path = setup_old(tmp_path, monkeypatch, state="complete",
                              experiment="017_morphology_templates")
    complete_015(tmp_path)
    (tmp_path / "017").mkdir()
    queue.atomic_json(tmp_path / "017/priority_queue_completion.json",
                      {"queue_manifest_sha256": OLD_SHA, "artifacts": {}})
    monkeypatch.setattr(queue, "identity", lambda pid: None)
    with pytest.raises(RuntimeError, match="already completed"):
        handoff.wait_handoff_gate(old, old_path, OLD_SHA, 41, IDENTITY, 0.01)


def test_launch_intent_prevents_duplicate_after_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(handoff, "ROOT", tmp_path)
    new_path = tmp_path / "new/queue.json"
    new_path.parent.mkdir()
    calls = []

    class Child:
        pid = 314

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return Child()

    monkeypatch.setattr(handoff.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(queue, "identity", lambda pid: IDENTITY)
    assert handoff.launch_successor(new_path, "b" * 64, OLD_SHA, 41, "c" * 64) == 314
    with pytest.raises(RuntimeError, match="duplicate launch"):
        handoff.launch_successor(new_path, "b" * 64, OLD_SHA, 41, "c" * 64)
    assert len(calls) == 1
    assert calls[0][1]["start_new_session"] is True
    assert json.loads((new_path.parent / "launch.json").read_text())["pid"] == 314
