"""Handoff watcher: 015 evidence gates, PID identity checks and single launch."""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from ecg_experiment.files import sha256_file, write_json_atomic
from scripts.coordination import common
from scripts.coordination import handoff_priority_queue as handoff

OLD_SHA = "a" * 64
IDENTITY = {"start": "123", "command": ["python", "-u", "-m", "scripts.coordination.run_priority_queue"]}


def setup_old(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str = "profiling",
              experiment: str = "015_cached_jepa_distillation") -> tuple[dict[str, Any], Path]:
    """
    Write an old queue with the given status and a 015 result artifact.

    Parameters
    ----------
    tmp_path : Path
        Fake repository root.
    monkeypatch : pytest.MonkeyPatch
        Used to point the coordinators at ``tmp_path``.
    state : str
        Old queue state.
    experiment : str
        Experiment named in the old queue status.

    Returns
    -------
    tuple[dict[str, Any], Path]
        Old manifest and its path.
    """
    monkeypatch.setattr(common, "ROOT", tmp_path)
    old_path = tmp_path / "old/queue.json"
    old_path.parent.mkdir()
    first = {"name": "015_cached_jepa_distillation", "output_dir": "015",
             "required_artifacts": ["015/result.json"]}
    second = {"name": "017_morphology_templates", "output_dir": "017"}
    old = {"jobs": [first, second]}
    write_json_atomic(old_path.parent / "status.json",
                      {"queue_manifest_sha256": OLD_SHA, "wrapper_pid": 41,
                       "experiment": experiment, "state": state})
    (tmp_path / "015").mkdir()
    (tmp_path / "015/result.json").write_text("verified")
    return old, old_path


def complete_015(tmp_path: Path) -> None:
    """
    Mark 015 complete with a verified completion marker.

    Parameters
    ----------
    tmp_path : Path
        Fake repository root.
    """
    write_json_atomic(tmp_path / "015/coordination.json",
                      {"queue_manifest_sha256": OLD_SHA, "state": "complete", "returncode": 0})
    write_json_atomic(tmp_path / "015/priority_queue_completion.json",
                      {"queue_manifest_sha256": OLD_SHA,
                       "artifacts": {"015/result.json": sha256_file(tmp_path / "015/result.json")}})


def test_never_signals_015_when_015_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    old, old_path = setup_old(tmp_path, monkeypatch, state="failed")
    monkeypatch.setattr(handoff, "process_identity", lambda pid: IDENTITY)
    monkeypatch.setattr(handoff.os, "kill", lambda *args: pytest.fail("signaled 015"))
    with pytest.raises(RuntimeError, match="015 failed"):
        handoff.wait_handoff_gate(old, old_path, OLD_SHA, 41, IDENTITY, 0.01)


def test_017_status_cannot_bypass_015_artifact_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    old, old_path = setup_old(tmp_path, monkeypatch, experiment="017_morphology_templates")
    complete_015(tmp_path)
    (tmp_path / "015/result.json").write_text("changed")
    monkeypatch.setattr(handoff, "process_identity", lambda pid: IDENTITY)
    monkeypatch.setattr(handoff.os, "kill", lambda *args: pytest.fail("signaled before artifact gate"))
    with pytest.raises(RuntimeError, match="artifact changed"):
        handoff.wait_handoff_gate(old, old_path, OLD_SHA, 41, IDENTITY, 0.01)


def test_identity_change_refuses_signal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    old, old_path = setup_old(tmp_path, monkeypatch, experiment="017_morphology_templates")
    complete_015(tmp_path)
    monkeypatch.setattr(handoff, "process_identity", lambda pid: {"start": "reused", "command": IDENTITY["command"]})
    monkeypatch.setattr(handoff.os, "kill", lambda *args: pytest.fail("signaled reused PID"))
    with pytest.raises(RuntimeError, match="identity changed"):
        handoff.wait_handoff_gate(old, old_path, OLD_SHA, 41, IDENTITY, 0.01)
    with pytest.raises(RuntimeError, match="identity changed"):
        handoff.stop_old(41, IDENTITY, 1, 0.01)


def test_verified_015_allows_017_stop_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    old, old_path = setup_old(tmp_path, monkeypatch, experiment="017_morphology_templates")
    complete_015(tmp_path)
    monkeypatch.setattr(handoff, "process_identity", lambda pid: IDENTITY)
    assert handoff.wait_handoff_gate(old, old_path, OLD_SHA, 41, IDENTITY, 0.01) == "stop_old_017"


def test_old_exited_after_017_completion_refuses_duplicate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    old, old_path = setup_old(tmp_path, monkeypatch, state="complete",
                              experiment="017_morphology_templates")
    complete_015(tmp_path)
    (tmp_path / "017").mkdir()
    write_json_atomic(tmp_path / "017/priority_queue_completion.json",
                      {"queue_manifest_sha256": OLD_SHA, "artifacts": {}})
    monkeypatch.setattr(handoff, "process_identity", lambda pid: None)
    with pytest.raises(RuntimeError, match="already completed"):
        handoff.wait_handoff_gate(old, old_path, OLD_SHA, 41, IDENTITY, 0.01)


def test_launch_intent_prevents_duplicate_after_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(common, "ROOT", tmp_path)
    new_path = tmp_path / "new/queue.json"
    new_path.parent.mkdir()
    calls = []

    class Child:
        pid = 314

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return Child()

    monkeypatch.setattr(handoff.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(handoff, "process_identity", lambda pid: IDENTITY)
    assert handoff.launch_successor(new_path, "b" * 64, OLD_SHA, 41, "c" * 64) == 314
    with pytest.raises(RuntimeError, match="duplicate launch"):
        handoff.launch_successor(new_path, "b" * 64, OLD_SHA, 41, "c" * 64)
    assert len(calls) == 1
    assert calls[0][1]["start_new_session"] is True
    assert calls[0][0][0] == sys.executable
    assert json.loads((new_path.parent / "launch.json").read_text())["pid"] == 314


@pytest.mark.parametrize(("status", "alive", "first", "second", "expected"), [
    ({"experiment": "015_cached_jepa_distillation", "state": "running"}, True, False, False, "wait"),
    ({"experiment": "017_morphology_templates", "state": "running"}, True, True, False, "stop_old_017"),
    ({"experiment": "017_morphology_templates", "state": "failed"}, True, True, False, "wait"),
    ({"experiment": "017_morphology_templates", "state": "failed"}, False, True, False, "old_exited"),
    ({"experiment": "015_cached_jepa_distillation", "state": "running"}, False, True, False, "old_exited"),
])
def test_handoff_action_allowed_steps(status: dict[str, str], alive: bool, first: bool, second: bool,
                                      expected: str) -> None:
    assert handoff.handoff_action(status, alive, first, second) == expected


@pytest.mark.parametrize(("status", "alive", "first", "second", "message"), [
    ({"experiment": "015_cached_jepa_distillation", "state": "interrupted"}, True, False, False, "015 failed"),
    ({"experiment": "017_morphology_templates", "state": "running"}, True, False, False, "without verified"),
    ({"experiment": "015_cached_jepa_distillation", "state": "running"}, False, False, False, "exited before"),
    ({"experiment": "017_morphology_templates", "state": "running"}, True, True, True, "already completed"),
    ({"experiment": "017_morphology_templates", "state": "complete"}, True, True, False, "already completed"),
    ({"experiment": "015_cached_jepa_distillation", "state": "complete"}, True, True, False, "unexpected state"),
])
def test_handoff_action_refusals(status: dict[str, str], alive: bool, first: bool, second: bool,
                                 message: str) -> None:
    with pytest.raises(RuntimeError, match=message):
        handoff.handoff_action(status, alive, first, second)
