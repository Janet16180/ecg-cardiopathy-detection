"""Idle-only pausing of the legacy orchestrator and coordinator cleanup."""

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from ecg_experiment import processes
from ecg_experiment.processes import IdleOrchestrator
from scripts.coordination import common

SLEEP = [sys.executable, "-c", "import time; time.sleep(30)"]
# The parent forks a sleeping child, then waits for it like the orchestrator.
WITH_CHILD = [sys.executable, "-c",
              "import subprocess, sys, time; "
              "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); time.sleep(30)"]


@pytest.fixture
def restore_signals() -> Iterator[None]:
    """Restore the SIGINT and SIGTERM handlers changed by a test."""
    original = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
    yield
    for number, handler in original.items():
        signal.signal(number, handler)


def start_orchestrator(tmp_path: Path, command: list[str] = SLEEP,
                       state: str = "waiting") -> tuple[subprocess.Popen, IdleOrchestrator]:
    """
    Start a fake orchestrator process with a MIMIC-style status file.

    Parameters
    ----------
    tmp_path : Path
        Directory for the status and ready files.
    command : list[str]
        Command of the fake orchestrator.
    state : str
        State of its ``mimic_preparation`` stage.

    Returns
    -------
    tuple[subprocess.Popen, IdleOrchestrator]
        The process and its orchestrator handle.
    """
    status_path = tmp_path / "status.json"
    status_path.write_text(json.dumps({"stages": {"mimic_preparation": {"state": state}}}))
    process = subprocess.Popen(command, start_new_session=True)
    identity = processes.process_identity(process.pid)
    return process, IdleOrchestrator(process.pid, identity, status_path, tmp_path / "ready.json")


def stop_group(process: subprocess.Popen) -> None:
    """
    Kill a fake orchestrator and any children it started.

    Parameters
    ----------
    process : subprocess.Popen
        Process started with its own session.
    """
    os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def wait_for_children(pid: int) -> None:
    """
    Wait until a process has started a child.

    Parameters
    ----------
    pid : int
        Parent process ID.
    """
    for _ in range(500):
        if processes.has_children(pid):
            return
        time.sleep(0.01)
    pytest.fail("fake orchestrator never started its child")


def test_idle_orchestrator_is_paused_and_resumed(tmp_path: Path) -> None:
    process, orchestrator = start_orchestrator(tmp_path)
    try:
        assert orchestrator.pause()
        assert processes.is_stopped(process.pid)
        assert orchestrator.resume()
        assert not processes.is_stopped(process.pid)
        assert not orchestrator.paused
        assert not orchestrator.resume()
    finally:
        stop_group(process)


def test_no_pause_when_data_is_ready(tmp_path: Path) -> None:
    process, orchestrator = start_orchestrator(tmp_path)
    try:
        orchestrator.ready_path.write_text("{}")
        assert not orchestrator.pause()
        assert not processes.is_stopped(process.pid)
    finally:
        stop_group(process)


def test_no_pause_when_not_waiting(tmp_path: Path) -> None:
    process, orchestrator = start_orchestrator(tmp_path, state="running")
    try:
        assert not orchestrator.pause()
        assert not processes.is_stopped(process.pid)
    finally:
        stop_group(process)


def test_no_pause_when_orchestrator_has_children(tmp_path: Path) -> None:
    process, orchestrator = start_orchestrator(tmp_path, command=WITH_CHILD)
    try:
        wait_for_children(process.pid)
        assert not orchestrator.pause()
        assert not processes.is_stopped(process.pid)
    finally:
        stop_group(process)


def test_race_after_stop_resumes_orchestrator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    process, orchestrator = start_orchestrator(tmp_path)
    original_stop = processes.stop_process

    def stop_then_data_arrives(pid: int, identity: processes.ProcessIdentity) -> None:
        original_stop(pid, identity)
        orchestrator.ready_path.write_text("{}")

    monkeypatch.setattr(processes, "stop_process", stop_then_data_arrives)
    try:
        assert not orchestrator.pause()
        assert not orchestrator.paused
        assert not processes.is_stopped(process.pid)
    finally:
        stop_group(process)


def test_pause_when_idle_waits_until_orchestrator_exits(tmp_path: Path) -> None:
    process, orchestrator = start_orchestrator(tmp_path, state="running")
    waits = []

    def on_wait() -> None:
        waits.append(True)
        process.kill()
        process.wait()

    assert not orchestrator.pause_when_idle(on_wait, interval=0.01)
    assert waits == [True]


def test_missing_orchestrator_is_never_paused(tmp_path: Path) -> None:
    orchestrator = IdleOrchestrator(1, None, tmp_path / "status.json", tmp_path / "ready.json")
    assert not orchestrator.pause_when_idle(lambda: pytest.fail("waited"), interval=0.01)
    assert not orchestrator.resume()


def test_cleanup_resumes_orchestrator_despite_signal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                      restore_signals: None) -> None:
    process, orchestrator = start_orchestrator(tmp_path)

    def interrupted_terminate(child: subprocess.Popen | None) -> None:
        os.kill(os.getpid(), signal.SIGINT)
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(0.05)
        raise OSError("terminate failed")

    monkeypatch.setattr(common, "terminate_child", interrupted_terminate)
    processes.interrupt_on_termination()
    try:
        assert orchestrator.pause()
        with pytest.raises(OSError, match="terminate failed"):
            common.stop_child_and_resume(None, orchestrator)
        assert not processes.is_stopped(process.pid)
        assert signal.getsignal(signal.SIGINT) is processes._raise_interrupt
    finally:
        stop_group(process)


def test_first_signal_interrupts_and_later_ones_are_ignored(restore_signals: None) -> None:
    processes.interrupt_on_termination()
    with pytest.raises(KeyboardInterrupt, match="Received signal"):
        signal.raise_signal(signal.SIGTERM)
    os.kill(os.getpid(), signal.SIGINT)
    time.sleep(0.01)
    assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN


def test_expect_module_rejects_unexpected_process() -> None:
    process = subprocess.Popen(SLEEP)
    try:
        assert common.expect_module(process.pid, "-c") is not None
        with pytest.raises(RuntimeError, match="not the expected"):
            common.expect_module(process.pid, common.MIMIC_MODULE)
        assert common.expect_module(None, common.MIMIC_MODULE) is None
    finally:
        processes.terminate_child(process, timeout=5)


def test_write_status_records_wrapper_pid(tmp_path: Path) -> None:
    record = common.write_status(tmp_path / "coordination.json", "queued", reason="test")
    assert json.loads((tmp_path / "coordination.json").read_text()) == record
    assert record["state"] == "queued"
    assert record["wrapper_pid"] == os.getpid()
