"""Paths, status records and cleanup shared by the GPU queue coordinators."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.processes import (
    IdleOrchestrator,
    ProcessIdentity,
    process_identity,
    runs_module,
    terminate_child,
    termination_deferred,
)
from ecg_experiment.provenance import utc_now

ROOT = Path(__file__).resolve().parents[2]
MIMIC_MODULE = "scripts.experiments.run_mimic_scale"
MIMIC_STATUS = Path("outputs/experiment003_mimic/status.json")
MIMIC_READY = Path("data/processed/mimic_ssl_200k/metadata.json")
POLL_SECONDS = 30.0



def write_status(path: Path, state: str, **details: Any) -> dict[str, Any]:
    """
    Record and print a coordinator state.

    Parameters
    ----------
    path : Path
        JSON file to replace atomically.
    state : str
        Coordinator state, such as ``queued`` or ``running``.
    **details : Any
        Extra JSON-serializable fields.

    Returns
    -------
    dict[str, Any]
        The written record.
    """
    record = {"state": state, "updated_at": utc_now(), "wrapper_pid": os.getpid(), **details}
    write_json_atomic(path, record)
    print(json.dumps(record), flush=True)
    return record


def expect_module(pid: int | None, module: str) -> ProcessIdentity | None:
    """
    Identify a process that must run a given module while it is alive.

    Parameters
    ----------
    pid : int | None
        Process ID; None means no process is expected.
    module : str
        Dotted module name the process must run.

    Returns
    -------
    ProcessIdentity | None
        The identity, or None if there is no such live process.

    Raises
    ------
    RuntimeError
        If the PID belongs to a process running something else.
    """
    identity = process_identity(pid) if pid is not None else None
    if identity is not None and not runs_module(identity, module):
        raise RuntimeError(f"PID {pid} is not the expected {module} process")
    return identity


def mimic_orchestrator(pid: int) -> IdleOrchestrator:
    """
    Describe the legacy MIMIC orchestrator for idle-only pausing.

    Parameters
    ----------
    pid : int
        Process ID of the orchestrator.

    Returns
    -------
    IdleOrchestrator
        Orchestrator handle; its identity is None if the PID is gone.
    """
    identity = expect_module(pid, MIMIC_MODULE)
    return IdleOrchestrator(pid, identity, ROOT / MIMIC_STATUS, ROOT / MIMIC_READY)


def verify_sources(path: Path, expected_map_sha256: str | None = None) -> None:
    """
    Check frozen sources against a map of repository paths to SHA-256 digests.

    Parameters
    ----------
    path : Path
        JSON source map.
    expected_map_sha256 : str | None
        Pinned digest of the map itself, if any.

    Raises
    ------
    RuntimeError
        If the map or any listed source changed.
    """
    if expected_map_sha256 is not None and sha256_file(path) != expected_map_sha256:
        raise RuntimeError(f"Frozen source hash map changed: {path}")
    for name, expected in json.loads(path.read_text()).items():
        if sha256_file(ROOT / name) != expected:
            raise RuntimeError(f"Queued source changed: {name}")


def stop_child_and_resume(child: subprocess.Popen | None,
                          orchestrator: IdleOrchestrator | None) -> None:
    """
    Stop the GPU child, then resume the orchestrator if it was paused.

    Signals are ignored meanwhile, and the orchestrator is resumed even if
    stopping the child fails.

    Parameters
    ----------
    child : subprocess.Popen | None
        Running experiment process, if any.
    orchestrator : IdleOrchestrator | None
        Orchestrator this coordinator may have paused.
    """
    with termination_deferred():
        try:
            terminate_child(child)
        finally:
            if orchestrator is not None and orchestrator.resume():
                print(f"Resumed MIMIC orchestrator {orchestrator.pid}", flush=True)
