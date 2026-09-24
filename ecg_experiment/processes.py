"""Linux process identity and the pause/resume of the legacy MIMIC orchestrator.

A PID alone can be reused, so a process is identified by its PID, kernel start
time and command line together.  The dictionary form matches the identities
recorded in queue manifests.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TypedDict


class ProcessIdentity(TypedDict):
    start: str
    command: list[str]


def _stat_fields(pid: int) -> list[str]:
    # The command name in field 2 may contain spaces and parentheses.
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()


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
        command = Path(f"/proc/{pid}/cmdline").read_bytes().decode().rstrip("\0").split("\0")
    except FileNotFoundError:
        return None
    if fields[0] == "Z":
        return None
    # starttime is field 22 of /proc/<pid>/stat, index 19 after the name.
    return {"start": fields[19], "command": command}


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
        if _stat_fields(pid)[0] in ("T", "t"):
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


def _raise_interrupt(signum: int, _frame: object) -> None:
    raise KeyboardInterrupt(f"Received signal {signum}")


def interrupt_on_termination() -> None:
    """Raise ``KeyboardInterrupt`` on SIGTERM as well as SIGINT."""
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
