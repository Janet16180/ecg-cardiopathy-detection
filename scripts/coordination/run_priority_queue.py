"""Execute the frozen ECG experiment queue sequentially after the active suite.

This coordinator owns no CUDA context. Individual runners keep their existing
GPU locks. The older data-waiting MIMIC orchestrator is held during the queue,
and resumed on completion, failure or a handled interruption.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
from functools import partial
from pathlib import Path
from typing import Any

import ecg_experiment.files
import ecg_experiment.processes
from ecg_experiment.files import sha256_file, sha256_json, write_json_atomic
from ecg_experiment.processes import (IdleOrchestrator, interrupt_on_termination, process_identity,
                                      wait_while_alive)
from scripts.coordination import common

COMPLETION_MARKER = "priority_queue_completion.json"


def coordinator_sha256() -> str:
    """
    Fingerprint this coordinator together with the local code it imports.

    Returns
    -------
    str
        SHA-256 of the canonical map from source file name to file digest.
    """
    modules = (__file__, common.__file__, ecg_experiment.files.__file__, ecg_experiment.processes.__file__)
    return sha256_json({Path(path).name: sha256_file(path) for path in modules})


def validate_manifest(manifest: dict[str, Any]) -> None:
    """
    Check the structure of a queue manifest.

    Parameters
    ----------
    manifest : dict[str, Any]
        Parsed queue manifest.

    Raises
    ------
    ValueError
        If job names repeat, a job lacks stages or artifacts, or a stage is
        not an explicit ``python -u -m scripts...`` command.
    """
    jobs = manifest["jobs"]
    if not jobs or len({job["name"] for job in jobs}) != len(jobs):
        raise ValueError("Queue needs uniquely named jobs")
    for job in jobs:
        if not job["stages"] or not job["required_artifacts"]:
            raise ValueError("Each job needs stages and completion artifacts")
        for stage in job["stages"]:
            command = stage["command"]
            explicit = (isinstance(command, list) and len(command) >= 4
                        and all(isinstance(part, str) for part in command)
                        and command[1:3] == ["-u", "-m"] and command[3].startswith("scripts."))
            if not explicit:
                raise ValueError("Expected an explicit Python module command")


def legacy_orchestrator(manifest: dict[str, Any]) -> IdleOrchestrator | None:
    """
    Describe the pinned legacy MIMIC orchestrator of a manifest.

    Parameters
    ----------
    manifest : dict[str, Any]
        Parsed queue manifest.

    Returns
    -------
    IdleOrchestrator | None
        Orchestrator handle, or None if the manifest pins none.
    """
    legacy = manifest.get("legacy")
    if not legacy:
        return None
    return IdleOrchestrator(legacy["pid"], legacy["identity"], common.ROOT / legacy["status_path"],
                            common.ROOT / common.MIMIC_READY)


class Coordinator:
    """
    Run the queue jobs in order and hold the state its cleanup needs.

    Parameters
    ----------
    manifest : dict[str, Any]
        Validated queue manifest.
    directory : Path
        Directory for the queue ``status.json``.
    fingerprint : str
        SHA-256 of the manifest file.
    """

    def __init__(self, manifest: dict[str, Any], directory: Path, fingerprint: str) -> None:
        self.manifest = manifest
        self.directory = directory
        self.fingerprint = fingerprint
        self.child: subprocess.Popen | None = None
        self.legacy = legacy_orchestrator(manifest)
        self.current_job: dict[str, Any] | None = None

    def status(self, state: str, **details: Any) -> dict[str, Any]:
        """
        Record the queue state.

        Parameters
        ----------
        state : str
            Queue state.
        **details : Any
            Extra JSON-serializable fields.

        Returns
        -------
        dict[str, Any]
            The written record.
        """
        return common.write_status(self.directory / "status.json", state,
                                   queue_manifest_sha256=self.fingerprint,
                                   queue_order=[job["name"] for job in self.manifest["jobs"]], **details)

    def job_status(self, job: dict[str, Any], state: str, **details: Any) -> None:
        """
        Record the state of one job in the queue and in its output directory.

        Parameters
        ----------
        job : dict[str, Any]
            Manifest job.
        state : str
            Job state.
        **details : Any
            Extra JSON-serializable fields.
        """
        record = self.status(state, experiment=job["name"], **details)
        write_json_atomic(common.ROOT / job["output_dir"] / "coordination.json", record)

    def wait_predecessor(self) -> None:
        """
        Wait for the predecessor suite and require its clean completion.

        Raises
        ------
        RuntimeError
            If the predecessor did not complete with return code 0.
        """
        previous = self.manifest["predecessor"]
        wait_while_alive(previous["pid"], previous["identity"],
                         partial(self.status, "queued", waiting_for_pid=previous["pid"],
                                 reason="Let the current tokenization suite finish"),
                         common.POLL_SECONDS)
        result = json.loads((common.ROOT / previous["status_path"]).read_text())
        if result.get("state") != "complete" or result.get("returncode") != 0:
            raise RuntimeError("Current suite did not finish successfully; queue stopped")

    def coordinate_legacy(self) -> None:
        """Pause the legacy orchestrator once it idles, or wait for it to exit."""
        if self.legacy is None:
            return
        on_wait = partial(self.status, "queued", waiting_for_pid=self.legacy.pid,
                          reason="Existing active MIMIC job")
        if self.legacy.pause_when_idle(on_wait, common.POLL_SECONDS):
            self.status("legacy_paused", legacy_pid=self.legacy.pid,
                        reason="Requested experiment priority; downloader continues")

    def verify_completion(self, marker: Path) -> None:
        """
        Check that a completion marker belongs to this queue and still matches.

        Parameters
        ----------
        marker : Path
            Completion marker of a job.

        Raises
        ------
        RuntimeError
            If the marker is from another queue or an artifact changed.
        """
        completion = json.loads(marker.read_text())
        if completion["queue_manifest_sha256"] != self.fingerprint:
            raise RuntimeError("Completed job belongs to a different frozen queue")
        for name, expected in completion["artifacts"].items():
            if sha256_file(common.ROOT / name) != expected:
                raise RuntimeError(f"Completed queue artifact changed: {name}")

    def run_stage(self, job: dict[str, Any], stage: dict[str, Any], output: Path) -> None:
        """
        Run one job stage as a child process with its output logged.

        Parameters
        ----------
        job : dict[str, Any]
            Manifest job.
        stage : dict[str, Any]
            Stage with a ``name`` and a ``command``.
        output : Path
            Job output directory.

        Raises
        ------
        RuntimeError
            If the stage exits with a non-zero code.
        """
        with (output / "priority_queue.log").open("a", buffering=1) as log:
            log.write(f"\n{common.utc_now()} {stage['name']}\n")
            self.child = subprocess.Popen(stage["command"], cwd=common.ROOT,
                                          stdout=log, stderr=subprocess.STDOUT)
            self.job_status(job, "profiling" if stage["name"] == "profile" else "running",
                            stage=stage["name"], child_pid=self.child.pid, command=stage["command"],
                            mimic_orchestrator_paused=self.legacy is not None and self.legacy.paused)
            code = self.child.wait()
            self.child = None
        if code != 0:
            raise RuntimeError(f"{job['name']} stage {stage['name']} exited {code}")

    def execute_job(self, job: dict[str, Any]) -> None:
        """
        Run every stage of a job, or reuse its verified earlier completion.

        Parameters
        ----------
        job : dict[str, Any]
            Manifest job.
        """
        self.current_job = job
        common.verify_sources(common.ROOT / job["sources"], job["sources_sha256"])
        output = common.ROOT / job["output_dir"]
        output.mkdir(parents=True, exist_ok=True)
        marker = output / COMPLETION_MARKER
        if marker.exists():
            self.verify_completion(marker)
            self.job_status(job, "complete", returncode=0, reused=True)
            return
        for stage in job["stages"]:
            common.verify_sources(common.ROOT / job["sources"], job["sources_sha256"])
            self.run_stage(job, stage, output)
        artifacts = {name: sha256_file(common.ROOT / name) for name in job["required_artifacts"]}
        write_json_atomic(marker, {"queue_manifest_sha256": self.fingerprint, "artifacts": artifacts})
        self.job_status(job, "complete", returncode=0)

    def cleanup(self) -> None:
        """Stop a running stage and resume the legacy orchestrator if paused."""
        common.stop_child_and_resume(self.child, self.legacy)


def check_superseded(manifest: dict[str, Any]) -> None:
    """
    Refuse to run while a superseded queue coordinator is still alive.

    Parameters
    ----------
    manifest : dict[str, Any]
        Parsed queue manifest.

    Raises
    ------
    RuntimeError
        If a pinned superseded coordinator still runs.
    """
    for old in manifest.get("superseded_coordinators", []):
        if old["identity"] is not None and process_identity(old["pid"]) == old["identity"]:
            raise RuntimeError(f"Superseded queue coordinator is still active: {old['pid']}")


def run_queue(coordinator: Coordinator) -> None:
    """
    Run the whole queue, recording failures and always cleaning up.

    Parameters
    ----------
    coordinator : Coordinator
        Coordinator of a verified manifest.
    """
    try:
        check_superseded(coordinator.manifest)
        coordinator.wait_predecessor()
        coordinator.coordinate_legacy()
        for job in coordinator.manifest["jobs"]:
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


def main() -> None:
    """Verify the frozen manifest and its sources, then run or only check it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if sha256_file(args.manifest) != args.manifest_sha256:
        raise ValueError("Queue manifest changed")
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    if coordinator_sha256() != manifest["coordinator_sha256"]:
        raise ValueError("Queue coordinator source changed")
    for job in manifest["jobs"]:
        common.verify_sources(common.ROOT / job["sources"], job["sources_sha256"])
    if args.check:
        print(json.dumps({"status": "verified", "order": [job["name"] for job in manifest["jobs"]]}))
        return
    coordinator = Coordinator(manifest, args.manifest.parent, args.manifest_sha256)
    interrupt_on_termination()
    with (args.manifest.parent / "runner.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_queue(coordinator)


if __name__ == "__main__":
    main()
