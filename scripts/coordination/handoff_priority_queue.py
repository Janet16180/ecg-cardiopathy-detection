"""Wait for experiment 015, then hand off the frozen queue to a verified successor.

Run this watcher in the host process namespace. It never signals a 015 worker.
The old coordinator handles its own child and legacy-runner cleanup on SIGTERM.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ecg_experiment.files import read_json, sha256_file, write_json_atomic
from ecg_experiment.processes import ProcessIdentity, is_stopped, process_identity
from ecg_experiment.provenance import utc_now
from scripts.coordination import common
from scripts.coordination import run_priority_queue as queue

OLD_ORDER = ["015_cached_jepa_distillation", "017_morphology_templates"]
NEW_ORDER = ["017_morphology_templates_v2"]
NEW_017_MODULE = "scripts.experiments.run_cpc_morphology017_v2"
ENDED_BADLY = ("failed", "interrupted")
FINISHED = ("failed", "interrupted", "complete")
ACTIVE = ("profiling", "running")
WAIT, OLD_EXITED, STOP_OLD_017 = "wait", "old_exited", "stop_old_017"
MAX_STOP_POLL_SECONDS = 1.0


def root_path(value: str | Path) -> Path:
    """
    Resolve a path relative to the repository root.

    Parameters
    ----------
    value : str | Path
        Absolute or repository-relative path.

    Returns
    -------
    Path
        Absolute path.
    """
    path = Path(value)
    return path if path.is_absolute() else common.ROOT / path


def pinned_json(path: Path, digest: str) -> Any:
    """
    Read a JSON file whose SHA-256 is pinned.

    Parameters
    ----------
    path : Path
        File to read.
    digest : str
        Expected SHA-256.

    Returns
    -------
    Any
        Parsed value.

    Raises
    ------
    RuntimeError
        If the file changed.
    """
    if sha256_file(path) != digest:
        raise RuntimeError(f"Pinned file changed: {path}")
    return read_json(path)


def check_queue_pair(old: dict[str, Any], new: dict[str, Any]) -> None:
    """
    Check that the old and successor manifests form the pinned 017 handoff.

    Parameters
    ----------
    old : dict[str, Any]
        Old queue manifest (015 then original 017).
    new : dict[str, Any]
        Successor manifest (corrected 017 only).

    Raises
    ------
    RuntimeError
        If job order, coordinator source, runner or sources do not match.
    """
    queue.validate_manifest(old)
    queue.validate_manifest(new)
    if [job["name"] for job in old["jobs"]] != OLD_ORDER:
        raise RuntimeError("Old queue order is not the pinned 015 then 017 sequence")
    if [job["name"] for job in new["jobs"]] != NEW_ORDER:
        raise RuntimeError("Successor must contain only the corrected 017 job")
    coordinator_hash = queue.coordinator_sha256()
    if coordinator_hash != old["coordinator_sha256"] or coordinator_hash != new["coordinator_sha256"]:
        raise RuntimeError("Pinned coordinator source changed")
    if any(stage["command"][3] != NEW_017_MODULE for stage in new["jobs"][0]["stages"]):
        raise RuntimeError("Successor does not use the versioned 017 runner")
    for job in new["jobs"]:
        common.verify_sources(root_path(job["sources"]), job["sources_sha256"])
    if new.get("legacy") != old.get("legacy"):
        raise RuntimeError("Successor legacy-runner identity differs from old queue")


def check_old_launch(launch: dict[str, Any], old_path: Path, old_sha: str, old_pid: int) -> None:
    """
    Check that the old launch receipt pins the old coordinator process.

    Parameters
    ----------
    launch : dict[str, Any]
        Old launch receipt.
    old_path : Path
        Old queue manifest.
    old_sha : str
        SHA-256 of the old manifest.
    old_pid : int
        PID of the old coordinator.

    Raises
    ------
    RuntimeError
        If the receipt does not pin this PID, identity and command.
    """
    identity = launch.get("identity")
    pinned = (launch.get("pid") == old_pid and isinstance(identity, dict)
              and identity.get("start") and identity.get("command")
              and launch.get("queue_manifest_sha256") == old_sha
              and launch.get("command") == identity["command"])
    if not pinned:
        raise RuntimeError("Old launch receipt does not pin the coordinator PID and identity")
    command = identity["command"]
    expected_command = (len(command) == 8
                        and command[1:4] == ["-u", "-m", "scripts.coordination.run_priority_queue"]
                        and command[4] == "--manifest" and Path(command[5]).resolve() == old_path.resolve()
                        and command[6:] == ["--manifest-sha256", old_sha])
    if not expected_command:
        raise RuntimeError("Old launch command does not match the pinned manifest")


def check_successor_pins(old: dict[str, Any], new: dict[str, Any], identity: ProcessIdentity,
                         old_pid: int) -> None:
    """
    Check that the successor waits for and supersedes the old coordinator.

    Parameters
    ----------
    old : dict[str, Any]
        Old queue manifest.
    new : dict[str, Any]
        Successor manifest.
    identity : ProcessIdentity
        Pinned identity of the old coordinator.
    old_pid : int
        PID of the old coordinator.

    Raises
    ------
    RuntimeError
        If the successor predecessor or superseded entries differ.
    """
    predecessor = new.get("predecessor", {})
    expected_status = root_path(old["jobs"][0]["output_dir"]) / "coordination.json"
    pinned = (predecessor.get("identity") == identity and predecessor.get("pid") == old_pid
              and root_path(predecessor.get("status_path", "")).resolve() == expected_status.resolve())
    if not pinned:
        raise RuntimeError("Successor predecessor must be completed 015 coordination")
    if {"pid": old_pid, "identity": identity} not in new.get("superseded_coordinators", []):
        raise RuntimeError("Successor must pin the superseded coordinator identity")


def check_manifests(old_path: Path, old_sha: str, old_launch_path: Path, old_launch_sha: str,
                    old_pid: int, new_path: Path,
                    new_sha: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """
    Load and cross-check the pinned old manifest, its launch receipt and the successor.

    Parameters
    ----------
    old_path, new_path : Path
        Old and successor queue manifests.
    old_sha, new_sha : str
        Their pinned SHA-256 digests.
    old_launch_path : Path
        Launch receipt of the old coordinator.
    old_launch_sha : str
        Pinned SHA-256 of the receipt.
    old_pid : int
        PID of the old coordinator.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
        Old manifest, successor manifest and old launch receipt.
    """
    old = pinned_json(old_path, old_sha)
    new = pinned_json(new_path, new_sha)
    launch = pinned_json(old_launch_path, old_launch_sha)
    check_queue_pair(old, new)
    check_old_launch(launch, old_path, old_sha, old_pid)
    check_successor_pins(old, new, launch["identity"], old_pid)
    return old, new, launch


def verified_015(old: dict[str, Any], old_sha: str) -> bool:
    """
    Check whether 015 completed with verified artifacts in the old queue.

    Parameters
    ----------
    old : dict[str, Any]
        Old queue manifest.
    old_sha : str
        SHA-256 of the old manifest.

    Returns
    -------
    bool
        True once 015 is complete and every required artifact matches.

    Raises
    ------
    RuntimeError
        If the 015 records belong to another queue, 015 failed, or an
        artifact changed.
    """
    job = old["jobs"][0]
    directory = root_path(job["output_dir"])
    status_path = directory / "coordination.json"
    marker_path = directory / queue.COMPLETION_MARKER
    if not status_path.exists() or not marker_path.exists():
        return False
    status = read_json(status_path)
    if status.get("queue_manifest_sha256") != old_sha:
        raise RuntimeError("015 coordination belongs to a different queue")
    if status.get("state") in ENDED_BADLY:
        raise RuntimeError("015 failed or was interrupted")
    if status.get("state") != "complete" or status.get("returncode") != 0:
        return False
    marker = read_json(marker_path)
    if marker.get("queue_manifest_sha256") != old_sha:
        raise RuntimeError("015 completion marker belongs to a different queue")
    if set(marker.get("artifacts", {})) != set(job["required_artifacts"]):
        raise RuntimeError("015 completion marker does not cover required artifacts")
    for name, digest in marker["artifacts"].items():
        if sha256_file(root_path(name)) != digest:
            raise RuntimeError(f"015 completed artifact changed: {name}")
    return True


def old_state(old_path: Path, old_sha: str, old_pid: int,
              expected_identity: ProcessIdentity) -> tuple[dict[str, Any], ProcessIdentity | None]:
    """
    Read the old queue status and the current identity of its coordinator.

    Parameters
    ----------
    old_path : Path
        Old queue manifest.
    old_sha : str
        SHA-256 of the old manifest.
    old_pid : int
        PID of the old coordinator.
    expected_identity : ProcessIdentity
        Pinned identity of the old coordinator.

    Returns
    -------
    tuple[dict[str, Any], ProcessIdentity | None]
        Old queue status and the identity, or None once it has exited.

    Raises
    ------
    RuntimeError
        If the status is missing or foreign, or the PID was reused.
    """
    status_path = old_path.parent / "status.json"
    if not status_path.exists():
        raise RuntimeError("Old queue status is missing")
    status = read_json(status_path)
    if status.get("queue_manifest_sha256") != old_sha or status.get("wrapper_pid") != old_pid:
        raise RuntimeError("Old queue status does not match pinned coordinator")
    current = process_identity(old_pid)
    if current is not None and current != expected_identity:
        raise RuntimeError("Old coordinator PID identity changed")
    return status, current


def original_017_complete(old: dict[str, Any], old_sha: str) -> bool:
    """
    Check whether the original 017 output directory already holds a result.

    Parameters
    ----------
    old : dict[str, Any]
        Old queue manifest.
    old_sha : str
        SHA-256 of the old manifest.

    Returns
    -------
    bool
        True if a completion marker exists or the old queue completed 017.
    """
    directory = root_path(old["jobs"][1]["output_dir"])
    status_path = directory / "coordination.json"
    # Even a marker from another queue means this output path has already been used.
    if (directory / queue.COMPLETION_MARKER).exists():
        return True
    if not status_path.exists():
        return False
    record = read_json(status_path)
    return record.get("queue_manifest_sha256") == old_sha and record.get("state") == "complete"


def handoff_action(status: dict[str, Any], old_alive: bool, first_verified: bool,
                   second_complete: bool) -> str:
    """
    Decide the next handoff step from the observed old-queue evidence.

    Parameters
    ----------
    status : dict[str, Any]
        Old queue status.
    old_alive : bool
        Whether the pinned old coordinator still runs.
    first_verified : bool
        Whether 015 completed with verified artifacts.
    second_complete : bool
        Whether the original 017 output already holds a result.

    Returns
    -------
    str
        ``WAIT``, ``OLD_EXITED`` or ``STOP_OLD_017``.

    Raises
    ------
    RuntimeError
        If the evidence rules out a safe handoff.
    """
    experiment, state = status.get("experiment"), status.get("state")
    in_017 = experiment == OLD_ORDER[1]
    if experiment == OLD_ORDER[0] and state in ENDED_BADLY:
        raise RuntimeError("015 failed; refusing handoff")
    if not first_verified and in_017:
        raise RuntimeError("Old queue entered 017 without verified 015 artifacts")
    if not first_verified and not old_alive:
        raise RuntimeError("Old coordinator exited before verified 015 completion")
    if first_verified and (second_complete or (in_017 and state == "complete")):
        raise RuntimeError("Original 017 already completed; refusing duplicate run")
    # A failed or interrupted original 017 is left to its coordinator's cleanup.
    if first_verified and old_alive and not in_017 and state in FINISHED:
        raise RuntimeError("Old queue ended in an unexpected state before 017 handoff")

    action = WAIT
    if first_verified and not old_alive:
        action = OLD_EXITED
    elif first_verified and in_017 and state in ACTIVE:
        action = STOP_OLD_017
    return action


def observe_handoff(old: dict[str, Any], old_path: Path, old_sha: str, old_pid: int,
                    expected_identity: ProcessIdentity) -> str:
    """
    Gather the old-queue evidence once and classify it.

    Parameters
    ----------
    old : dict[str, Any]
        Old queue manifest.
    old_path : Path
        Old queue manifest path.
    old_sha : str
        SHA-256 of the old manifest.
    old_pid : int
        PID of the old coordinator.
    expected_identity : ProcessIdentity
        Pinned identity of the old coordinator.

    Returns
    -------
    str
        Handoff action from ``handoff_action``.
    """
    status, current = old_state(old_path, old_sha, old_pid, expected_identity)
    return handoff_action(status, current is not None, verified_015(old, old_sha),
                          original_017_complete(old, old_sha))


def wait_handoff_gate(old: dict[str, Any], old_path: Path, old_sha: str, old_pid: int,
                      expected_identity: ProcessIdentity, poll_seconds: float) -> str:
    """
    Poll the old queue until the handoff can proceed.

    Parameters
    ----------
    old : dict[str, Any]
        Old queue manifest.
    old_path : Path
        Old queue manifest path.
    old_sha : str
        SHA-256 of the old manifest.
    old_pid : int
        PID of the old coordinator.
    expected_identity : ProcessIdentity
        Pinned identity of the old coordinator.
    poll_seconds : float
        Seconds between observations.

    Returns
    -------
    str
        ``OLD_EXITED`` or ``STOP_OLD_017``.
    """
    action = observe_handoff(old, old_path, old_sha, old_pid, expected_identity)
    while action == WAIT:
        time.sleep(poll_seconds)
        action = observe_handoff(old, old_path, old_sha, old_pid, expected_identity)
    return action


def stop_old(old_pid: int, expected_identity: ProcessIdentity, timeout_seconds: float,
             poll_seconds: float) -> None:
    """
    Send SIGTERM to the old coordinator and wait for it to exit.

    Parameters
    ----------
    old_pid : int
        PID of the old coordinator.
    expected_identity : ProcessIdentity
        Pinned identity of the old coordinator.
    timeout_seconds : float
        Seconds to wait for the exit.
    poll_seconds : float
        Upper bound on the time between checks.

    Raises
    ------
    RuntimeError
        If the PID was reused or the coordinator did not exit in time.
    """
    current = process_identity(old_pid)
    if current is not None and current != expected_identity:
        raise RuntimeError("Old coordinator PID identity changed before SIGTERM")
    if current is not None:
        os.kill(old_pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout_seconds
    while current is not None and time.monotonic() < deadline:
        time.sleep(min(poll_seconds, MAX_STOP_POLL_SECONDS))
        current = process_identity(old_pid)
        if current is not None and current != expected_identity:
            raise RuntimeError("Old coordinator PID was reused while stopping")
    if current is not None:
        raise RuntimeError("Old coordinator did not exit after SIGTERM; successor was not launched")


def verify_legacy(old: dict[str, Any]) -> None:
    """
    Require the pinned legacy runner to be alive and not left stopped.

    Parameters
    ----------
    old : dict[str, Any]
        Old queue manifest.

    Raises
    ------
    RuntimeError
        If the legacy runner changed, disappeared or is still stopped.
    """
    legacy = old.get("legacy")
    if not legacy or legacy.get("identity") is None:
        return
    if process_identity(legacy["pid"]) != legacy["identity"]:
        raise RuntimeError("Pinned legacy-runner identity changed or disappeared")
    if is_stopped(legacy["pid"]):
        raise RuntimeError("Legacy runner remained stopped after old coordinator exited")


def launch_successor(new_path: Path, new_sha: str, old_sha: str, old_pid: int,
                     old_launch_sha: str) -> int:
    """
    Launch the successor coordinator in its own session and record receipts.

    Parameters
    ----------
    new_path : Path
        Successor queue manifest.
    new_sha : str
        SHA-256 of the successor manifest.
    old_sha : str
        SHA-256 of the old manifest.
    old_pid : int
        PID of the old coordinator.
    old_launch_sha : str
        SHA-256 of the old launch receipt.

    Returns
    -------
    int
        PID of the successor coordinator.

    Raises
    ------
    RuntimeError
        If the successor directory already has launch state.
    """
    directory = new_path.parent
    intent_path = directory / "handoff_launch_intent.json"
    receipt_path = directory / "launch.json"
    if intent_path.exists() or receipt_path.exists() or (directory / "status.json").exists():
        raise RuntimeError("Successor already has launch state; refusing duplicate launch")
    command = [sys.executable, "-u", "-m", "scripts.coordination.run_priority_queue",
               "--manifest", str(new_path.resolve()), "--manifest-sha256", new_sha]
    write_json_atomic(intent_path, {"state": "launching", "at_utc": utc_now(),
                                    "new_manifest_sha256": new_sha, "old_manifest_sha256": old_sha,
                                    "old_launch_sha256": old_launch_sha, "old_pid": old_pid,
                                    "command": command})
    with (directory / "runner.log").open("a", buffering=1) as log, open(os.devnull, "rb") as devnull:
        child = subprocess.Popen(command, cwd=common.ROOT, stdin=devnull, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    write_json_atomic(receipt_path, {"pid": child.pid, "identity": process_identity(child.pid),
                                     "command": command, "launched_at": utc_now(),
                                     "queue_manifest_sha256": new_sha,
                                     "predecessor_manifest_sha256": old_sha,
                                     "predecessor_launch_sha256": old_launch_sha,
                                     "handoff_intent": str(intent_path)})
    return child.pid


def check_still_running_017(old: dict[str, Any], old_path: Path, old_sha: str, old_pid: int,
                            expected_identity: ProcessIdentity) -> None:
    """
    Recheck 015 evidence and 017 status immediately before signaling.

    Parameters
    ----------
    old : dict[str, Any]
        Old queue manifest.
    old_path : Path
        Old queue manifest path.
    old_sha : str
        SHA-256 of the old manifest.
    old_pid : int
        PID of the old coordinator.
    expected_identity : ProcessIdentity
        Pinned identity of the old coordinator.

    Raises
    ------
    RuntimeError
        If 015 evidence vanished or a live old coordinator left 017.
    """
    if not verified_015(old, old_sha):
        raise RuntimeError("015 completion evidence disappeared")
    status, current = old_state(old_path, old_sha, old_pid, expected_identity)
    running_017 = status.get("experiment") == OLD_ORDER[1] and status.get("state") in ACTIVE
    if current is not None and not running_017:
        raise RuntimeError("Old coordinator is no longer running 017")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse the command line.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or ``None`` for ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Validated arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-manifest", type=Path, required=True)
    parser.add_argument("--old-manifest-sha256", required=True)
    parser.add_argument("--old-launch", type=Path, required=True)
    parser.add_argument("--old-launch-sha256", required=True)
    parser.add_argument("--old-pid", type=int, required=True)
    parser.add_argument("--new-manifest", type=Path, required=True)
    parser.add_argument("--new-manifest-sha256", required=True)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--stop-timeout-seconds", type=float, default=90.0)
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0 or args.stop_timeout_seconds <= 0:
        parser.error("poll and stop timeout must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    """
    Verify the pinned handoff, wait for 015, stop the old 017 and launch the successor.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or ``None`` for ``sys.argv``.
    """
    args = parse_args(argv)
    old_path = root_path(args.old_manifest)
    new_path = root_path(args.new_manifest)
    if old_path.resolve() == new_path.resolve() or old_path.parent.resolve() == new_path.parent.resolve():
        raise RuntimeError("Successor must use a new manifest directory")
    old, _, launch = check_manifests(old_path, args.old_manifest_sha256, root_path(args.old_launch),
                                     args.old_launch_sha256, args.old_pid, new_path,
                                     args.new_manifest_sha256)
    identity = launch["identity"]
    with (new_path.parent / "handoff.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        launch_files = ("handoff_launch_intent.json", "launch.json")
        if any((new_path.parent / name).exists() for name in launch_files):
            raise RuntimeError("Successor handoff was already launched or attempted")
        action = wait_handoff_gate(old, old_path, args.old_manifest_sha256, args.old_pid, identity,
                                   args.poll_seconds)
        if action == STOP_OLD_017:
            check_still_running_017(old, old_path, args.old_manifest_sha256, args.old_pid, identity)
            stop_old(args.old_pid, identity, args.stop_timeout_seconds, args.poll_seconds)
        if not verified_015(old, args.old_manifest_sha256):
            raise RuntimeError("015 completion evidence changed before successor launch")
        if original_017_complete(old, args.old_manifest_sha256):
            raise RuntimeError("Original 017 completed before successor launch")
        verify_legacy(old)
        pid = launch_successor(new_path, args.new_manifest_sha256, args.old_manifest_sha256,
                               args.old_pid, args.old_launch_sha256)
        print(json.dumps({"state": "launched", "pid": pid,
                          "new_manifest_sha256": args.new_manifest_sha256}), flush=True)


if __name__ == "__main__":
    main()
