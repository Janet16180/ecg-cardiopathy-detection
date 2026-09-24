"""Wait for experiment 015, then hand off the frozen queue to a verified successor.

Run this watcher in the host process namespace. It never signals a 015 worker.
The old coordinator handles its own child and legacy-runner cleanup on SIGTERM.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from scripts import run_priority_queue as queue


ROOT = queue.ROOT


def now():
    return datetime.now(timezone.utc).isoformat()


def root_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path):
    return json.loads(Path(path).read_text())


def pinned_json(path, digest):
    if queue.sha256(path) != digest:
        raise RuntimeError(f"Pinned file changed: {path}")
    return read_json(path)


def check_manifests(old_path, old_sha, old_launch_path, old_launch_sha,
                    old_pid, new_path, new_sha):
    old = pinned_json(old_path, old_sha)
    new = pinned_json(new_path, new_sha)
    launch = pinned_json(old_launch_path, old_launch_sha)
    queue.validate_manifest(old)
    queue.validate_manifest(new)
    if [j["name"] for j in old["jobs"]] != ["015_cached_jepa_distillation", "017_morphology_templates"]:
        raise RuntimeError("Old queue order is not the pinned 015 then 017 sequence")
    if [j["name"] for j in new["jobs"]] != ["017_morphology_templates_v2"]:
        raise RuntimeError("Successor must contain only the corrected 017 job")
    coordinator_hash = queue.sha256(Path(queue.__file__))
    if coordinator_hash != old["coordinator_sha256"] or coordinator_hash != new["coordinator_sha256"]:
        raise RuntimeError("Pinned coordinator source changed")
    if any(stage["command"][3] != "scripts.run_cpc_morphology017_v2"
           for stage in new["jobs"][0]["stages"]):
        raise RuntimeError("Successor does not use the versioned 017 runner")
    for job in new["jobs"]:
        queue.verify_sources(root_path(job["sources"]), job["sources_sha256"])
    expected_identity = launch.get("identity")
    if (launch.get("pid") != old_pid or not isinstance(expected_identity, dict)
            or not expected_identity.get("start") or not expected_identity.get("command")
            or launch.get("queue_manifest_sha256") != old_sha
            or launch.get("command") != expected_identity["command"]):
        raise RuntimeError("Old launch receipt does not pin the coordinator PID and identity")
    command = expected_identity["command"]
    if (len(command) != 8 or command[1:4] != ["-u", "-m", "scripts.run_priority_queue"]
            or command[4] != "--manifest" or Path(command[5]).resolve() != old_path.resolve()
            or command[6:] != ["--manifest-sha256", old_sha]):
        raise RuntimeError("Old launch command does not match the pinned manifest")
    predecessor = new.get("predecessor", {})
    expected_status = root_path(old["jobs"][0]["output_dir"]) / "coordination.json"
    if (predecessor.get("identity") != expected_identity or predecessor.get("pid") != old_pid
            or root_path(predecessor.get("status_path", "")).resolve() != expected_status.resolve()):
        raise RuntimeError("Successor predecessor must be completed 015 coordination")
    superseded = new.get("superseded_coordinators", [])
    if {"pid": old_pid, "identity": expected_identity} not in superseded:
        raise RuntimeError("Successor must pin the superseded coordinator identity")
    if new.get("legacy") != old.get("legacy"):
        raise RuntimeError("Successor legacy-runner identity differs from old queue")
    return old, new, launch


def verified_015(old, old_sha):
    job = old["jobs"][0]
    directory = root_path(job["output_dir"])
    status_path = directory / "coordination.json"
    marker_path = directory / "priority_queue_completion.json"
    if not status_path.exists() or not marker_path.exists():
        return False
    status = read_json(status_path)
    if status.get("queue_manifest_sha256") != old_sha:
        raise RuntimeError("015 coordination belongs to a different queue")
    if status.get("state") in ("failed", "interrupted"):
        raise RuntimeError("015 failed or was interrupted")
    if status.get("state") != "complete" or status.get("returncode") != 0:
        return False
    marker = read_json(marker_path)
    if marker.get("queue_manifest_sha256") != old_sha:
        raise RuntimeError("015 completion marker belongs to a different queue")
    required = set(job["required_artifacts"])
    if set(marker.get("artifacts", {})) != required:
        raise RuntimeError("015 completion marker does not cover required artifacts")
    for name, digest in marker["artifacts"].items():
        if queue.sha256(root_path(name)) != digest:
            raise RuntimeError(f"015 completed artifact changed: {name}")
    return True


def old_state(old_path, old_sha, old_pid, expected_identity):
    status_path = old_path.parent / "status.json"
    if not status_path.exists():
        raise RuntimeError("Old queue status is missing")
    status = read_json(status_path)
    if (status.get("queue_manifest_sha256") != old_sha
            or status.get("wrapper_pid") != old_pid):
        raise RuntimeError("Old queue status does not match pinned coordinator")
    current = queue.identity(old_pid)
    if current is not None and current != expected_identity:
        raise RuntimeError("Old coordinator PID identity changed")
    return status, current


def original_017_complete(old, old_sha):
    directory = root_path(old["jobs"][1]["output_dir"])
    marker = directory / "priority_queue_completion.json"
    status = directory / "coordination.json"
    if marker.exists():
        # Even a marker from another queue means this output path has already been used.
        return True
    if status.exists():
        record = read_json(status)
        return record.get("queue_manifest_sha256") == old_sha and record.get("state") == "complete"
    return False


def wait_handoff_gate(old, old_path, old_sha, old_pid, expected_identity, poll_seconds):
    while True:
        status, current = old_state(old_path, old_sha, old_pid, expected_identity)
        if status.get("experiment") == old["jobs"][0]["name"] and status.get("state") in ("failed", "interrupted"):
            raise RuntimeError("015 failed; refusing handoff")
        complete = verified_015(old, old_sha)
        if complete:
            if original_017_complete(old, old_sha) or (status.get("experiment") == old["jobs"][1]["name"] and status.get("state") == "complete"):
                raise RuntimeError("Original 017 already completed; refusing duplicate run")
            if current is None:
                return "old_exited"
            if status.get("experiment") == old["jobs"][1]["name"]:
                if status.get("state") in ("profiling", "running"):
                    return "stop_old_017"
                if status.get("state") in ("failed", "interrupted"):
                    time.sleep(poll_seconds)
                    continue
            if status.get("state") in ("failed", "interrupted", "complete"):
                raise RuntimeError("Old queue ended in an unexpected state before 017 handoff")
        elif status.get("experiment") == old["jobs"][1]["name"]:
            raise RuntimeError("Old queue entered 017 without verified 015 artifacts")
        elif current is None:
            raise RuntimeError("Old coordinator exited before verified 015 completion")
        time.sleep(poll_seconds)


def stop_old(old_pid, expected_identity, timeout_seconds, poll_seconds):
    current = queue.identity(old_pid)
    if current is not None and current != expected_identity:
        raise RuntimeError("Old coordinator PID identity changed before SIGTERM")
    if current == expected_identity:
        os.kill(old_pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        current = queue.identity(old_pid)
        if current is None:
            return
        if current != expected_identity:
            raise RuntimeError("Old coordinator PID was reused while stopping")
        time.sleep(min(poll_seconds, 1.0))
    raise RuntimeError("Old coordinator did not exit after SIGTERM; successor was not launched")


def verify_legacy(old):
    legacy = old.get("legacy")
    if legacy and legacy.get("identity") is not None:
        if queue.identity(legacy["pid"]) != legacy["identity"]:
            raise RuntimeError("Pinned legacy-runner identity changed or disappeared")
        fields = Path(f"/proc/{legacy['pid']}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] in ("T", "t"):
            raise RuntimeError("Legacy runner remained stopped after old coordinator exited")


def launch_successor(new_path, new_sha, old_sha, old_pid, old_launch_sha):
    directory = new_path.parent
    intent_path = directory / "handoff_launch_intent.json"
    receipt_path = directory / "launch.json"
    if intent_path.exists() or receipt_path.exists() or (directory / "status.json").exists():
        raise RuntimeError("Successor already has launch state; refusing duplicate launch")
    command = [str(ROOT / ".venv-pretrained/bin/python"), "-u", "-m",
               "scripts.run_priority_queue", "--manifest", str(new_path.resolve()),
               "--manifest-sha256", new_sha]
    queue.atomic_json(intent_path, {"state": "launching", "at_utc": now(),
                                    "new_manifest_sha256": new_sha,
                                    "old_manifest_sha256": old_sha,
                                    "old_launch_sha256": old_launch_sha,
                                    "old_pid": old_pid, "command": command})
    with (directory / "runner.log").open("a", buffering=1) as log:
        with open(os.devnull, "rb") as devnull:
            child = subprocess.Popen(command, cwd=ROOT, stdin=devnull, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
    identity = queue.identity(child.pid)
    queue.atomic_json(receipt_path, {"pid": child.pid, "identity": identity,
                                     "command": command, "launched_at": now(),
                                     "queue_manifest_sha256": new_sha,
                                     "predecessor_manifest_sha256": old_sha,
                                     "predecessor_launch_sha256": old_launch_sha,
                                     "handoff_intent": str(intent_path)})
    return child.pid


def main():
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
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.stop_timeout_seconds <= 0:
        parser.error("poll and stop timeout must be positive")
    old_path = root_path(args.old_manifest)
    new_path = root_path(args.new_manifest)
    old_launch_path = root_path(args.old_launch)
    if old_path.resolve() == new_path.resolve() or old_path.parent.resolve() == new_path.parent.resolve():
        raise RuntimeError("Successor must use a new manifest directory")
    old, _, launch = check_manifests(old_path, args.old_manifest_sha256,
                                     old_launch_path, args.old_launch_sha256,
                                     args.old_pid, new_path, args.new_manifest_sha256)
    lock_path = new_path.parent / "handoff.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (new_path.parent / "handoff_launch_intent.json").exists() or (new_path.parent / "launch.json").exists():
            raise RuntimeError("Successor handoff was already launched or attempted")
        action = wait_handoff_gate(old, old_path, args.old_manifest_sha256,
                                   args.old_pid, launch["identity"], args.poll_seconds)
        if action == "stop_old_017":
            # Recheck the 015 evidence and 017 status immediately before the signal.
            if not verified_015(old, args.old_manifest_sha256):
                raise RuntimeError("015 completion evidence disappeared")
            status, current = old_state(old_path, args.old_manifest_sha256,
                                        args.old_pid, launch["identity"])
            if current == launch["identity"]:
                if status.get("experiment") != old["jobs"][1]["name"] or status.get("state") not in ("profiling", "running"):
                    raise RuntimeError("Old coordinator is no longer running 017")
            stop_old(args.old_pid, launch["identity"], args.stop_timeout_seconds,
                     args.poll_seconds)
        if not verified_015(old, args.old_manifest_sha256):
            raise RuntimeError("015 completion evidence changed before successor launch")
        if original_017_complete(old, args.old_manifest_sha256):
            raise RuntimeError("Original 017 completed before successor launch")
        verify_legacy(old)
        pid = launch_successor(new_path, args.new_manifest_sha256,
                               args.old_manifest_sha256, args.old_pid,
                               args.old_launch_sha256)
        print(json.dumps({"state": "launched", "pid": pid,
                          "new_manifest_sha256": args.new_manifest_sha256}), flush=True)


if __name__ == "__main__":
    main()
