"""Coordinator entry points record their outcome and always resume the orchestrator."""

import json
import signal
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from ecg_experiment import processes
from scripts.coordination import common, run_cpc_coordinated

FAKE_MIMIC = [sys.executable, "-c", "import time; time.sleep(30)", common.MIMIC_MODULE]


class FinishedChild:
    """Stand-in for an experiment process that exits with a fixed code."""

    pid = 4242
    returncode = 0

    def __init__(self, command: list[str], **kwargs: object) -> None:
        self.command = command

    def wait(self) -> int:
        """Return the exit code at once."""
        return self.returncode

    def poll(self) -> int:
        """Report that the child has exited."""
        return self.returncode


@pytest.fixture
def fake_mimic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[subprocess.Popen]:
    """Idle fake MIMIC orchestrator under a fake repository root."""
    monkeypatch.setattr(common, "ROOT", tmp_path)
    status = tmp_path / common.MIMIC_STATUS
    status.parent.mkdir(parents=True)
    status.write_text(json.dumps({"stages": {"mimic_preparation": {"state": "waiting"}}}))
    original = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
    process = subprocess.Popen(FAKE_MIMIC)
    yield process
    processes.terminate_child(process, timeout=5)
    for number, handler in original.items():
        signal.signal(number, handler)


def run_cpc(monkeypatch: pytest.MonkeyPatch, output_dir: Path, pid: int) -> None:
    """
    Run the CPC coordinator entry point with the given arguments.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to set ``sys.argv``.
    output_dir : Path
        Coordinator output directory.
    pid : int
        Fake MIMIC orchestrator PID.
    """
    monkeypatch.setattr(sys, "argv", ["run_cpc_coordinated", "--waiting-runner-pid", str(pid),
                                      "--output-dir", str(output_dir)])
    run_cpc_coordinated.main()


def test_cpc_success_resumes_orchestrator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                          fake_mimic: subprocess.Popen) -> None:
    stopped_during_run = []

    class Child(FinishedChild):
        def wait(self) -> int:
            stopped_during_run.append(processes.is_stopped(fake_mimic.pid))
            return 0

    monkeypatch.setattr(run_cpc_coordinated.subprocess, "Popen", Child)
    run_cpc(monkeypatch, tmp_path / "out", fake_mimic.pid)
    assert stopped_during_run == [True, True]
    assert not processes.is_stopped(fake_mimic.pid)
    assert json.loads((tmp_path / "out/coordination.json").read_text())["state"] == "complete"


def test_cpc_exception_records_failure_and_resumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                   fake_mimic: subprocess.Popen) -> None:
    def broken_popen(command: list[str], **kwargs: object) -> None:
        raise OSError("cannot start")

    monkeypatch.setattr(run_cpc_coordinated.subprocess, "Popen", broken_popen)
    with pytest.raises(OSError, match="cannot start"):
        run_cpc(monkeypatch, tmp_path / "out", fake_mimic.pid)
    record = json.loads((tmp_path / "out/coordination.json").read_text())
    assert record["state"] == "failed" and record["reason"] == "cannot start"
    assert not processes.is_stopped(fake_mimic.pid)


def test_cpc_refuses_busy_orchestrator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                       fake_mimic: subprocess.Popen) -> None:
    (tmp_path / common.MIMIC_READY).parent.mkdir(parents=True)
    (tmp_path / common.MIMIC_READY).write_text("{}")
    monkeypatch.setattr(run_cpc_coordinated.subprocess, "Popen", lambda *a, **k: pytest.fail("launched"))
    with pytest.raises(RuntimeError, match="not idle"):
        run_cpc(monkeypatch, tmp_path / "out", fake_mimic.pid)
    assert json.loads((tmp_path / "out/coordination.json").read_text())["state"] == "failed"
    assert not processes.is_stopped(fake_mimic.pid)
