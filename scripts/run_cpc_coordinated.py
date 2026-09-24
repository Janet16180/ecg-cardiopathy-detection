"""Run the compact CPC suite while the older MIMIC runner waits for its data.

Only the verified idle orchestrator is suspended; the downloader keeps running.
The orchestrator is resumed on success, failure, SIGINT, or SIGTERM.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def identity(pid):
    directory = Path(f"/proc/{pid}")
    # starttime is field 22; the process name may contain spaces.
    start = (directory / "stat").read_text().rsplit(")", 1)[1].split()[19]
    command = (directory / "cmdline").read_bytes().split(b"\0")
    return start, command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waiting-runner-pid", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/experiment004_cpc_40k")
    parser.add_argument("--wait-cache", action="store_true")
    parser.add_argument("--wait-pid", type=int,
                        help="Wait for an earlier coordinated GPU suite to exit before suspending the MIMIC runner")
    parser.add_argument("--experiment-module", choices=("scripts.run_cpc_experiment", "scripts.run_cpc_word2vec",
                                                      "scripts.run_cpc_tokenization"),
                        default="scripts.run_cpc_experiment")
    parser.add_argument("--ssl-epochs", type=int, default=20)
    args = parser.parse_args()
    if args.ssl_epochs < 1:
        parser.error("ssl-epochs must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.wait_pid:
        try:
            predecessor = identity(args.wait_pid)
        except FileNotFoundError:
            predecessor = None
        while predecessor is not None:
            try:
                current = identity(args.wait_pid)
                state = Path(f"/proc/{args.wait_pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
                if current != predecessor or state == "Z":
                    break
            except FileNotFoundError:
                break
            print(f"Waiting for earlier ECG suite PID {args.wait_pid}", flush=True)
            time.sleep(30)
    if args.wait_cache:
        deadline = time.monotonic() + 6 * 3600
        ready = ROOT / "data/processed/cpc_pool_40k/complete.json"
        while not ready.exists():
            if time.monotonic() > deadline:
                raise RuntimeError("Timed out after six hours waiting for audited CPC cache")
            print("Waiting for audited CPC cache; MIMIC download/runner unchanged", flush=True)
            time.sleep(30)
    pid = args.waiting_runner_pid
    original = identity(pid)
    if b"scripts.run_mimic_scale" not in original[1]:
        raise RuntimeError("PID is not the expected MIMIC experiment orchestrator")
    state = json.loads((ROOT / "outputs/experiment003_mimic/status.json").read_text())
    if state["stages"]["mimic_preparation"]["state"] != "waiting":
        raise RuntimeError("The MIMIC orchestrator is no longer waiting for preparation")
    if (ROOT / "data/processed/mimic_ssl_200k/metadata.json").exists():
        raise RuntimeError("The MIMIC data is ready; explicitly coordinate its active training first")
    children = Path(f"/proc/{pid}/task/{pid}/children").read_text().strip()
    if children:
        raise RuntimeError(f"Waiting orchestrator has active children: {children}")
    child = None
    paused = False

    def status(stage, **details):
        value = {"state": stage, "updated_at": datetime.now(timezone.utc).isoformat(),
                 "wrapper_pid": os.getpid(), "waiting_runner_pid": pid, **details}
        path = args.output_dir / "coordination.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(path)
        print(json.dumps(value), flush=True)

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        os.kill(pid, signal.SIGSTOP)
        paused = True
        # Check again after stopping to close the ready-data race.
        if Path(f"/proc/{pid}/task/{pid}/children").read_text().strip():
            raise RuntimeError("Orchestrator started a child during coordination")
        profile = [sys.executable, "-u", "-m", args.experiment_module,
                   "--stage", "profile", "--variant", "all", "--device", "cuda", "--threads", "1",
                   "--ssl-epochs", str(args.ssl_epochs)]
        with (args.output_dir / "runtime_profile.log").open("a") as log:
            child = subprocess.Popen(profile, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            status("profiling", child_pid=child.pid, command=profile)
            code = child.wait()
            if code:
                status("failed", returncode=code, failed_stage="profile")
                raise SystemExit(code)
        command = [sys.executable, "-u", "-m", args.experiment_module,
                   "--stage", "all", "--variant", "all", "--labels", "all",
                   "--device", "cuda", "--threads", "1", "--ssl-epochs", str(args.ssl_epochs),
                   "--output-dir", str(args.output_dir)]
        child = subprocess.Popen(command, cwd=ROOT)
        status("running", child_pid=child.pid, command=command,
               note="Only waiting MIMIC orchestrator suspended; downloader continues")
        code = child.wait()
        status("complete" if code == 0 else "failed", returncode=code)
        if code:
            raise SystemExit(code)
    except KeyboardInterrupt as exc:
        status("interrupted", reason=str(exc))
        raise SystemExit(130)
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if paused:
            try:
                if identity(pid) == original:
                    os.kill(pid, signal.SIGCONT)
                    print(f"Resumed MIMIC orchestrator {pid}", flush=True)
            except FileNotFoundError:
                print("MIMIC orchestrator exited before resume", flush=True)


if __name__ == "__main__":
    main()
