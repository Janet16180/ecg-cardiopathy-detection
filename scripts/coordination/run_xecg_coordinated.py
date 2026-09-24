"""Queue xECG fine-tuning after an existing suite, coordinating the older runner."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def identity(pid):
    try:
        directory = Path(f"/proc/{pid}")
        fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return fields[19], (directory / "cmdline").read_bytes().split(b"\0")
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--mimic-runner-pid", type=int, default=3769)
    parser.add_argument("--preparation-pid", type=int,
                        help="Existing CPU cache preparation to wait for")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/experiment007_xecg")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predecessor = identity(args.wait_pid)
    mimic = identity(args.mimic_runner_pid)
    preparation = identity(args.preparation_pid) if args.preparation_pid else None
    if predecessor and b"scripts.coordination.run_cpc_coordinated" not in predecessor[1]:
        raise RuntimeError("Predecessor is not the expected ECG suite")
    if mimic and b"scripts.experiments.run_mimic_scale" not in mimic[1]:
        raise RuntimeError("MIMIC PID belongs to an unexpected process")
    if preparation and b"scripts.data.prepare_xecg" not in preparation[1]:
        raise RuntimeError("Preparation PID belongs to an unexpected process")
    child = None
    paused = False

    def status(state, **details):
        value = {"state": state, "updated_at": datetime.now(timezone.utc).isoformat(),
                 "wrapper_pid": os.getpid(), "predecessor_pid": args.wait_pid, **details}
        target = args.output_dir / "coordination.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(target)
        print(json.dumps(value), flush=True)

    def interrupt(signum, _frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    def wait_for(pid, original, reason):
        while original is not None and identity(pid) == original:
            status("queued", waiting_for_pid=pid, reason=reason)
            time.sleep(30)

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    try:
        if preparation is not None:
            wait_for(args.preparation_pid, preparation, "xECG CPU input preparation")
        if not (ROOT / "data/processed/ptbxl/xecg_views/metadata.json").exists():
            command = [sys.executable, "-u", "-m", "scripts.experiments.run_xecg_finetune",
                       "--stage", "prep", "--budget", "all", "--device", "cpu",
                       "--output-dir", str(args.output_dir)]
            child = subprocess.Popen(command, cwd=ROOT)
            status("preparing", child_pid=child.pid, command=command)
            code = child.wait()
            if code:
                status("failed", failed_stage="preparation", returncode=code)
                raise SystemExit(code)
        wait_for(args.wait_pid, predecessor, "Earlier tokenization suite")
        # The older MIMIC runner predates the shared GPU lock. Pause it only
        # while idle waiting for data; if its data is ready, let it finish first.
        if mimic is not None and identity(args.mimic_runner_pid) == mimic:
            children_path = Path(f"/proc/{args.mimic_runner_pid}/task/{args.mimic_runner_pid}/children")
            state_path = ROOT / "outputs/experiment003_mimic/status.json"
            ready = ROOT / "data/processed/mimic_ssl_200k/metadata.json"
            state = json.loads(state_path.read_text())
            idle = (state["stages"]["mimic_preparation"]["state"] == "waiting"
                    and not ready.exists() and not children_path.read_text().strip())
            if idle:
                os.kill(args.mimic_runner_pid, signal.SIGSTOP)
                paused = True
                # Close a child-launch race after stopping the orchestrator.
                if children_path.read_text().strip() or ready.exists():
                    os.kill(args.mimic_runner_pid, signal.SIGCONT)
                    paused = False
                    wait_for(args.mimic_runner_pid, mimic, "MIMIC suite became ready")
            else:
                wait_for(args.mimic_runner_pid, mimic, "Active or ready MIMIC suite")
        provenance = args.output_dir / "provenance/sources.json"
        if provenance.exists():
            import hashlib
            for name, expected in json.loads(provenance.read_text()).items():
                actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                if actual != expected:
                    raise RuntimeError(f"Queued xECG source changed: {name}")
        for stage in ("profile", "train"):
            command = [sys.executable, "-u", "-m", "scripts.experiments.run_xecg_finetune",
                       "--stage", stage, "--budget", "all", "--device", "cuda",
                       "--output-dir", str(args.output_dir)]
            if stage == "train":
                command.append("--resume")
            child = subprocess.Popen(command, cwd=ROOT)
            status("profiling" if stage == "profile" else "running",
                   child_pid=child.pid, command=command,
                   mimic_orchestrator_paused=paused)
            code = child.wait()
            if code:
                status("failed", failed_stage=stage, returncode=code)
                raise SystemExit(code)
        status("complete", returncode=0)
    except KeyboardInterrupt as exc:
        status("interrupted", reason=str(exc))
        raise SystemExit(130)
    except Exception as exc:
        status("failed", reason=str(exc))
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if paused and identity(args.mimic_runner_pid) == mimic:
            os.kill(args.mimic_runner_pid, signal.SIGCONT)
            print(f"Resumed MIMIC orchestrator {args.mimic_runner_pid}", flush=True)


if __name__ == "__main__":
    main()
