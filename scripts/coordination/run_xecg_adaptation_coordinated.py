"""Run Experiment 008 after successful xECG transfer and its input preparation."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.coordination.run_xecg_coordinated import identity

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--preparation-pid", type=int, required=True)
    parser.add_argument("--mimic-runner-pid", type=int, default=3769)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/experiment008_vision_ssl")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    previous, preparation, mimic = (identity(pid) for pid in
        (args.wait_pid, args.preparation_pid, args.mimic_runner_pid))
    for value, expected in ((previous, b"scripts.coordination.run_xecg_coordinated"),
                            (preparation, b"scripts.data.prepare_xecg_ssl"),
                            (mimic, b"scripts.experiments.run_mimic_scale")):
        if value is not None and expected not in value[1]:
            raise RuntimeError(f"Unexpected queue process; expected {expected.decode()}")
    child, paused = None, False

    def status(state, **kwargs):
        data = {"state": state, "updated_at": datetime.now(timezone.utc).isoformat(),
                "wrapper_pid": os.getpid(), "predecessor_pid": args.wait_pid, **kwargs}
        path = args.output_dir / "coordination.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(data, indent=2) + "\n")
        temp.replace(path)
        print(json.dumps(data), flush=True)

    def wait(pid, expected, reason):
        while expected is not None and identity(pid) == expected:
            status("queued", waiting_for_pid=pid, reason=reason)
            time.sleep(30)

    def interrupt(signum, _frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    try:
        wait(args.preparation_pid, preparation, "Train-only xECG input preparation")
        if not (ROOT / "data/processed/xecg_ssl_40k/metadata.json").is_file():
            raise RuntimeError("xECG SSL input preparation did not complete")
        wait(args.wait_pid, previous, "Experiment 007 xECG fine-tuning")
        earlier = json.loads((ROOT / "outputs/experiment007_xecg/coordination.json").read_text())
        if earlier["state"] != "complete" or earlier.get("returncode") != 0:
            raise RuntimeError("Experiment 007 did not complete successfully; resolve its failure before adaptation")
        if mimic is not None and identity(args.mimic_runner_pid) == mimic:
            children = Path(f"/proc/{args.mimic_runner_pid}/task/{args.mimic_runner_pid}/children")
            ready = ROOT / "data/processed/mimic_ssl_200k/metadata.json"
            old_status = json.loads((ROOT / "outputs/experiment003_mimic/status.json").read_text())
            idle = (old_status["stages"]["mimic_preparation"]["state"] == "waiting"
                    and not ready.exists() and not children.read_text().strip())
            if idle:
                os.kill(args.mimic_runner_pid, signal.SIGSTOP)
                paused = True
                if ready.exists() or children.read_text().strip():
                    os.kill(args.mimic_runner_pid, signal.SIGCONT)
                    paused = False
                    wait(args.mimic_runner_pid, mimic, "MIMIC training became ready")
            else:
                wait(args.mimic_runner_pid, mimic, "MIMIC training active or ready")
        for name, expected in json.loads((args.output_dir / "provenance/sources.json").read_text()).items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Queued adaptation source changed: {name}")
        for stage in ("profile", "all"):
            command = [sys.executable, "-u", "-m", "scripts.experiments.run_xecg_adaptation",
                       "--stage", stage, "--device", "cuda", "--output-dir", str(args.output_dir)]
            if stage == "all":
                command.append("--resume")
            child = subprocess.Popen(command, cwd=ROOT)
            status("profiling" if stage == "profile" else "running", child_pid=child.pid,
                   command=command, mimic_orchestrator_paused=paused)
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


if __name__ == "__main__":
    main()
