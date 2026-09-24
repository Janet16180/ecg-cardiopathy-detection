"""Execute the frozen ECG experiment queue sequentially after the active suite.

This coordinator owns no CUDA context. Individual runners keep their existing
GPU locks. The older data-waiting MIMIC orchestrator is held during the queue,
and resumed on completion, failure or a handled interruption.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def identity(pid):
    try:
        path = Path(f"/proc/{pid}")
        fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"start": fields[19], "command":
                (path / "cmdline").read_bytes().decode().rstrip("\0").split("\0")}
    except FileNotFoundError:
        return None


def verify_sources(path, expected_map_sha256=None):
    if expected_map_sha256 is not None and sha256(path) != expected_map_sha256:
        raise RuntimeError(f"Frozen source hash map changed: {path}")
    entries = json.loads(Path(path).read_text())
    for name, expected in entries.items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError(f"Queued source changed: {name}")


def wait_stopped(pid, expected, timeout=5.0):
    """SIGSTOP delivery is asynchronous; observe a stopped parent before racing forks."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if identity(pid) != expected:
            raise RuntimeError("Orchestrator identity changed while stopping it")
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        if state in ("T", "t"):
            return
        time.sleep(0.01)
    raise RuntimeError("Timed out confirming the orchestrator stopped")


def validate_manifest(manifest):
    jobs = manifest["jobs"]
    if not jobs or len({job["name"] for job in jobs}) != len(jobs):
        raise ValueError("Queue needs uniquely named jobs")
    for job in jobs:
        if not job["stages"] or not job["required_artifacts"]:
            raise ValueError("Each job needs stages and completion artifacts")
        for stage in job["stages"]:
            command = stage["command"]
            if (not isinstance(command, list) or len(command) < 4
                    or not all(isinstance(part, str) for part in command)
                    or command[1:3] != ["-u", "-m"]
                    or not command[3].startswith("scripts.")):
                raise ValueError("Expected an explicit Python module command")


class Coordinator:
    def __init__(self, manifest, directory, fingerprint):
        self.manifest = manifest
        self.directory = directory
        self.fingerprint = fingerprint
        self.child = None
        self.paused_legacy = False
        self.current_job = None

    def status(self, state, **kwargs):
        record = {"state": state, "updated_at": datetime.now(timezone.utc).isoformat(),
                  "wrapper_pid": os.getpid(), "queue_manifest_sha256": self.fingerprint,
                  "queue_order": [j["name"] for j in self.manifest["jobs"]], **kwargs}
        atomic_json(self.directory / "status.json", record)
        print(json.dumps(record), flush=True)
        return record

    def job_status(self, job, state, **kwargs):
        record = self.status(state, experiment=job["name"], **kwargs)
        atomic_json(ROOT / job["output_dir"] / "coordination.json", record)

    def wait_predecessor(self):
        previous = self.manifest["predecessor"]
        while previous["identity"] is not None and identity(previous["pid"]) == previous["identity"]:
            self.status("queued", waiting_for_pid=previous["pid"],
                        reason="Let the current tokenization suite finish")
            time.sleep(30)
        result = json.loads((ROOT / previous["status_path"]).read_text())
        if result.get("state") != "complete" or result.get("returncode") != 0:
            raise RuntimeError("Current suite did not finish successfully; queue stopped")

    def coordinate_legacy(self):
        legacy = self.manifest.get("legacy")
        if not legacy or legacy["identity"] is None:
            return
        pid = legacy["pid"]
        while identity(pid) == legacy["identity"]:
            children = Path(f"/proc/{pid}/task/{pid}/children")
            state = json.loads((ROOT / legacy["status_path"]).read_text())
            waiting = state["stages"].get("mimic_preparation", {}).get("state") == "waiting"
            if waiting and not children.read_text().strip():
                os.kill(pid, signal.SIGSTOP)
                self.paused_legacy = True
                wait_stopped(pid, legacy["identity"])
                # If a child won the race, allow that active job to finish.
                if not children.read_text().strip():
                    self.status("legacy_paused", legacy_pid=pid,
                                reason="Requested experiment priority; downloader continues")
                    return
                os.kill(pid, signal.SIGCONT)
                self.paused_legacy = False
            self.status("queued", waiting_for_pid=pid, reason="Existing active MIMIC job")
            time.sleep(30)

    def execute_job(self, job):
        self.current_job = job
        verify_sources(ROOT / job["sources"], job["sources_sha256"])
        output = ROOT / job["output_dir"]
        output.mkdir(parents=True, exist_ok=True)
        complete = output / "priority_queue_completion.json"
        if complete.exists():
            old = json.loads(complete.read_text())
            if old["queue_manifest_sha256"] != self.fingerprint:
                raise RuntimeError("Completed job belongs to a different frozen queue")
            for name, expected in old["artifacts"].items():
                if sha256(ROOT / name) != expected:
                    raise RuntimeError(f"Completed queue artifact changed: {name}")
            self.job_status(job, "complete", returncode=0, reused=True)
            return
        for stage in job["stages"]:
            verify_sources(ROOT / job["sources"], job["sources_sha256"])
            with (output / "priority_queue.log").open("a", buffering=1) as log:
                log.write(f"\n{datetime.now(timezone.utc).isoformat()} {stage['name']}\n")
                self.child = subprocess.Popen(stage["command"], cwd=ROOT,
                                              stdout=log, stderr=subprocess.STDOUT)
                self.job_status(job, "profiling" if stage["name"] == "profile" else "running",
                                stage=stage["name"], child_pid=self.child.pid,
                                command=stage["command"],
                                mimic_orchestrator_paused=self.paused_legacy)
                code = self.child.wait()
                self.child = None
            if code != 0:
                raise RuntimeError(f"{job['name']} stage {stage['name']} exited {code}")
        artifacts = {name: sha256(ROOT / name) for name in job["required_artifacts"]}
        atomic_json(complete, {"queue_manifest_sha256": self.fingerprint, "artifacts": artifacts})
        self.job_status(job, "complete", returncode=0)

    def cleanup(self):
        if self.child is not None and self.child.poll() is None:
            self.child.terminate()
            try:
                self.child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.child.kill()
                self.child.wait()
        legacy = self.manifest.get("legacy")
        if self.paused_legacy and identity(legacy["pid"]) == legacy["identity"]:
            os.kill(legacy["pid"], signal.SIGCONT)
            print(f"Resumed MIMIC orchestrator {legacy['pid']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if sha256(args.manifest) != args.manifest_sha256:
        raise ValueError("Queue manifest changed")
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    if sha256(Path(__file__)) != manifest["coordinator_sha256"]:
        raise ValueError("Queue coordinator source changed")
    for job in manifest["jobs"]:
        verify_sources(ROOT / job["sources"], job["sources_sha256"])
    if args.check:
        print(json.dumps({"status": "verified", "order": [j["name"] for j in manifest["jobs"]]}))
        return
    coordinator = Coordinator(manifest, args.manifest.parent, args.manifest_sha256)

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    with (args.manifest.parent / "runner.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for old in manifest.get("superseded_coordinators", []):
                if old["identity"] is not None and identity(old["pid"]) == old["identity"]:
                    raise RuntimeError(f"Superseded queue coordinator is still active: {old['pid']}")
            coordinator.wait_predecessor()
            coordinator.coordinate_legacy()
            for job in manifest["jobs"]:
                coordinator.execute_job(job)
            coordinator.status("complete", returncode=0)
        except (Exception, KeyboardInterrupt) as exc:
            state = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            if coordinator.current_job:
                coordinator.job_status(coordinator.current_job, state, reason=str(exc))
            else:
                coordinator.status(state, reason=str(exc))
            raise
        finally:
            coordinator.cleanup()


if __name__ == "__main__":
    main()
