"""Linux process identity and the pause/resume of the legacy MIMIC orchestrator.

A PID alone can be reused, so a process is identified by its PID, kernel start
time and command line together.  The dictionary form matches the identities
recorded in queue manifests.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TypedDict

WAITING_STAGE = "mimic_preparation"
EXEC_SETTLE_SECONDS = 1.0


class ProcessIdentity(TypedDict):
    start: str
    command: list[str]


def _stat_fields(pid: int) -> list[str]:
    # The command name in field 2 may contain spaces and parentheses.
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()


def _command_line(pid: int) -> list[str]:
    # Just after a spawn returns, the child can still be inside execve with
    # an empty command line; wait briefly for the new arguments.
    path = Path(f"/proc/{pid}/cmdline")
    deadline = time.monotonic() + EXEC_SETTLE_SECONDS
    raw = path.read_bytes()
    while not raw and time.monotonic() < deadline:
        time.sleep(0.001)
        raw = path.read_bytes()
    return raw.decode().rstrip("\0").split("\0")


def process_identity(pid: int) -> ProcessIdentity | None:
    """
    Identify a live process by start time and command line.

    Parameters
    ----------
    pid : int
        Process ID.

    Returns
    -------
    ProcessIdentity | None
        The identity, or None if the process is gone or a zombie.
    """
    try:
        fields = _stat_fields(pid)
        alive = fields[0] != "Z"
        command = _command_line(pid) if alive else []
    except FileNotFoundError:
        return None
    identity = None
    if alive:
        # starttime is field 22 of /proc/<pid>/stat, index 19 after the name.
        identity = ProcessIdentity(start=fields[19], command=command)
    return identity


def runs_module(identity: ProcessIdentity | None, module: str) -> bool:
    """
    Check whether a process runs a given Python module.

    Parameters
    ----------
    identity : ProcessIdentity | None
        Process identity from ``process_identity``.
    module : str
        Dotted module name passed after ``-m``.

    Returns
    -------
    bool
        True if ``module`` is one of the command-line arguments.
    """
    return identity is not None and module in identity["command"]


def is_stopped(pid: int) -> bool:
    """
    Check whether the kernel reports a process as stopped.

    Parameters
    ----------
    pid : int
        Process ID of a live process.

    Returns
    -------
    bool
        True for the stopped or traced states.
    """
    return _stat_fields(pid)[0] in ("T", "t")


def has_children(pid: int) -> bool:
    """
    Check whether a process has child processes.

    Parameters
    ----------
    pid : int
        Process ID.

    Returns
    -------
    bool
        True if the main thread has at least one child.
    """
    return bool(Path(f"/proc/{pid}/task/{pid}/children").read_text().strip())


def wait_while_alive(pid: int, identity: ProcessIdentity | None, on_wait: Callable[[], None],
                     interval: float = 30.0) -> None:
    """
    Block while the identified process is still running.

    Parameters
    ----------
    pid : int
        Process ID.
    identity : ProcessIdentity | None
        Identity captured earlier; None returns immediately.
    on_wait : Callable[[], None]
        Called before each sleep, typically to record a status.
    interval : float
        Seconds between checks.
    """
    while identity is not None and process_identity(pid) == identity:
        on_wait()
        time.sleep(interval)


def stop_process(pid: int, identity: ProcessIdentity, timeout: float = 5.0) -> None:
    """
    Send SIGSTOP and wait until the kernel reports the process stopped.

    Signal delivery is asynchronous, so checks made immediately after
    ``kill`` could race with a fork in the target.

    Parameters
    ----------
    pid : int
        Process ID.
    identity : ProcessIdentity
        Expected identity of the process.
    timeout : float
        Seconds to wait for the stopped state.

    Raises
    ------
    RuntimeError
        If the identity changes or the process does not stop in time.
    """
    os.kill(pid, signal.SIGSTOP)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process_identity(pid) != identity:
            raise RuntimeError("Orchestrator identity changed while stopping it")
        if is_stopped(pid):
            return
        time.sleep(0.01)
    raise RuntimeError("Timed out confirming the orchestrator stopped")


def resume_process(pid: int, identity: ProcessIdentity) -> bool:
    """
    Send SIGCONT if the identified process still exists.

    Parameters
    ----------
    pid : int
        Process ID.
    identity : ProcessIdentity
        Identity captured before the process was stopped.

    Returns
    -------
    bool
        True if the process was resumed, False if it had already exited.
    """
    if process_identity(pid) != identity:
        return False
    try:
        os.kill(pid, signal.SIGCONT)
    except ProcessLookupError:
        return False
    return True


def terminate_child(child: subprocess.Popen | None, timeout: float = 30.0) -> None:
    """
    Terminate a running child process, killing it if it does not exit.

    Parameters
    ----------
    child : subprocess.Popen | None
        Child to stop; None or an exited child is ignored.
    timeout : float
        Seconds to wait after SIGTERM before SIGKILL.
    """
    if child is None or child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()


class IdleOrchestrator:
    """
    Pause the legacy MIMIC orchestrator only while it idles waiting for data.

    The orchestrator predates the shared GPU lock.  It is idle when its status
    file reports the preparation stage as ``waiting``, its data marker is
    absent and it has no children.  ``paused`` records that this coordinator
    owns the stop, so cleanup knows to send SIGCONT.

    Parameters
    ----------
    pid : int
        Process ID of the orchestrator.
    identity : ProcessIdentity | None
        Identity captured earlier; None means there is nothing to pause.
    status_path : Path
        The orchestrator's ``status.json``.
    ready_path : Path
        File whose existence means the orchestrator's data is ready.
    """

    def __init__(self, pid: int, identity: ProcessIdentity | None, status_path: Path,
                 ready_path: Path) -> None:
        self.pid = pid
        self.identity = identity
        self.status_path = status_path
        self.ready_path = ready_path
        self.paused = False

    def is_alive(self) -> bool:
        """
        Check whether the identified orchestrator still runs.

        Returns
        -------
        bool
            True if the PID still has the captured identity.
        """
        return self.identity is not None and process_identity(self.pid) == self.identity

    def is_idle(self) -> bool:
        """
        Check whether the orchestrator only waits for its data.

        Returns
        -------
        bool
            True if it is alive, waiting, without ready data and childless.
        """
        if not self.is_alive():
            return False
        stages = json.loads(self.status_path.read_text())["stages"]
        waiting = stages.get(WAITING_STAGE, {}).get("state") == "waiting"
        return waiting and not self.ready_path.exists() and not has_children(self.pid)

    def pause(self) -> bool:
        """
        Stop the orchestrator if it is idle.

        Returns
        -------
        bool
            True if the orchestrator is now stopped by this coordinator.
        """
        if not self.is_idle():
            return False
        # Claim the stop before sending it so a signal arriving during the
        # stop still leads cleanup to send SIGCONT.
        self.paused = True
        stop_process(self.pid, self.identity)
        # A child or the data may have appeared before the stop took effect.
        if not self.is_idle():
            self.resume()
        return self.paused

    def pause_when_idle(self, on_wait: Callable[[], None], interval: float = 30.0) -> bool:
        """
        Wait until the orchestrator can be paused or has exited.

        Parameters
        ----------
        on_wait : Callable[[], None]
            Called before each sleep, typically to record a status.
        interval : float
            Seconds between checks.

        Returns
        -------
        bool
            True if the orchestrator was paused, False if it is gone.
        """
        while not self.pause() and self.is_alive():
            on_wait()
            time.sleep(interval)
        return self.paused

    def resume(self) -> bool:
        """
        Send SIGCONT if this coordinator stopped the orchestrator.

        Returns
        -------
        bool
            True if a stopped orchestrator was resumed.
        """
        if not self.paused:
            return False
        resumed = resume_process(self.pid, self.identity)
        self.paused = False
        return resumed


def _raise_interrupt(signum: int, _frame: object) -> None:
    # Later signals are ignored so they cannot interrupt the cleanup that
    # this first interruption starts.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    raise KeyboardInterrupt(f"Received signal {signum}")


def interrupt_on_termination() -> None:
    """Raise ``KeyboardInterrupt`` once on the first SIGINT or SIGTERM."""
    signal.signal(signal.SIGTERM, _raise_interrupt)
    signal.signal(signal.SIGINT, _raise_interrupt)


@contextmanager
def termination_deferred() -> Iterator[None]:
    """
    Ignore SIGINT and SIGTERM while cleanup runs.

    A second signal during cleanup would otherwise raise inside ``finally``
    and could leave a stopped orchestrator without its SIGCONT.

    Yields
    ------
    None
        Control while the signals are ignored.
    """
    previous = {number: signal.signal(number, signal.SIG_IGN) for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
