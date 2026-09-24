"""Queue xECG fine-tuning after an existing suite, coordinating the older runner."""

from __future__ import annotations

import argparse
import subprocess
import sys
from functools import partial
from pathlib import Path

from ecg_experiment.processes import interrupt_on_termination, wait_while_alive
from scripts.coordination import common

XECG_VIEWS_READY = Path("data/processed/ptbxl/xecg_views/metadata.json")


def parse_args() -> argparse.Namespace:
    """
    Parse the command line.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--mimic-runner-pid", type=int, required=True)
    parser.add_argument("--preparation-pid", type=int,
                        help="Existing CPU cache preparation to wait for")
    parser.add_argument("--output-dir", type=Path,
                        default=common.ROOT / "outputs/experiment007_xecg")
    return parser.parse_args()


def xecg_command(stage: str, device: str, output_dir: Path) -> list[str]:
    """
    Build a fine-tuning runner command.

    Parameters
    ----------
    stage : str
        Runner stage: ``prep``, ``profile`` or ``train``.
    device : str
        Torch device name.
    output_dir : Path
        Experiment output directory.

    Returns
    -------
    list[str]
        Command line; training resumes from checkpoints.
    """
    command = [sys.executable, "-u", "-m", "scripts.experiments.run_xecg_finetune",
               "--stage", stage, "--budget", "all", "--device", device,
               "--output-dir", str(output_dir)]
    if stage == "train":
        command.append("--resume")
    return command


def main() -> None:
    """Prepare inputs, wait for the predecessor and idle MIMIC runner, then train."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predecessor = common.expect_module(args.wait_pid, "scripts.coordination.run_cpc_coordinated")
    preparation = common.expect_module(args.preparation_pid, "scripts.data.prepare_xecg")
    orchestrator = common.mimic_orchestrator(args.mimic_runner_pid)
    status = partial(common.write_status, args.output_dir / "coordination.json",
                     predecessor_pid=args.wait_pid)
    provenance = args.output_dir / "provenance/sources.json"
    child = None
    interrupt_on_termination()
    try:
        wait_while_alive(args.preparation_pid, preparation,
                         partial(status, "queued", waiting_for_pid=args.preparation_pid,
                                 reason="xECG CPU input preparation"),
                         common.POLL_SECONDS)
        if not (common.ROOT / XECG_VIEWS_READY).exists():
            command = xecg_command("prep", "cpu", args.output_dir)
            child = subprocess.Popen(command, cwd=common.ROOT)
            status("preparing", child_pid=child.pid, command=command)
            code = child.wait()
            if code:
                status("failed", failed_stage="preparation", returncode=code)
                raise SystemExit(code)
        wait_while_alive(args.wait_pid, predecessor,
                         partial(status, "queued", waiting_for_pid=args.wait_pid,
                                 reason="Earlier tokenization suite"),
                         common.POLL_SECONDS)
        orchestrator.pause_when_idle(partial(status, "queued", waiting_for_pid=args.mimic_runner_pid,
                                             reason="Active or ready MIMIC suite"),
                                     common.POLL_SECONDS)
        if provenance.exists():
            common.verify_sources(provenance)
        for stage in ("profile", "train"):
            command = xecg_command(stage, "cuda", args.output_dir)
            child = subprocess.Popen(command, cwd=common.ROOT)
            status("profiling" if stage == "profile" else "running", child_pid=child.pid,
                   command=command, mimic_orchestrator_paused=orchestrator.paused)
            code = child.wait()
            if code:
                status("failed", failed_stage=stage, returncode=code)
                raise SystemExit(code)
        status("complete", returncode=0)
    except KeyboardInterrupt as exc:
        status("interrupted", reason=str(exc))
        raise SystemExit(130) from exc
    except Exception as exc:
        status("failed", reason=str(exc))
        raise
    finally:
        common.stop_child_and_resume(child, orchestrator)


if __name__ == "__main__":
    main()
