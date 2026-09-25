"""Run Experiment 008 after successful xECG transfer and its input preparation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from functools import partial
from pathlib import Path

from ecg_experiment.processes import interrupt_on_termination, wait_while_alive
from scripts.coordination import common

XECG_SSL_READY = Path("data/processed/xecg_ssl_40k/metadata.json")
XECG_COORDINATION = Path("outputs/experiment007_xecg/coordination.json")


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
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--preparation-pid", type=int, required=True)
    parser.add_argument("--mimic-runner-pid", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=common.ROOT / "outputs/experiment008_vision_ssl")
    return parser.parse_args(argv)


def check_xecg_complete() -> None:
    """
    Require Experiment 007 to have completed successfully.

    Raises
    ------
    RuntimeError
        If its coordination record is not a clean completion.
    """
    earlier = json.loads((common.ROOT / XECG_COORDINATION).read_text())
    if earlier["state"] != "complete" or earlier.get("returncode") != 0:
        raise RuntimeError("Experiment 007 did not complete successfully; "
                           "resolve its failure before adaptation")


def main(argv: list[str] | None = None) -> None:
    """
    Wait for preparation, Experiment 007 and the idle MIMIC runner, then adapt.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or ``None`` for ``sys.argv``.
    """
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    previous = common.expect_module(args.wait_pid, "scripts.coordination.run_xecg_coordinated")
    preparation = common.expect_module(args.preparation_pid, "scripts.data.prepare_xecg_ssl")
    orchestrator = common.mimic_orchestrator(args.mimic_runner_pid)
    status = partial(common.write_status, args.output_dir / "coordination.json",
                     predecessor_pid=args.wait_pid)
    child = None
    interrupt_on_termination()
    try:
        wait_while_alive(args.preparation_pid, preparation,
                         partial(status, "queued", waiting_for_pid=args.preparation_pid,
                                 reason="Train-only xECG input preparation"),
                         common.POLL_SECONDS)
        if not (common.ROOT / XECG_SSL_READY).is_file():
            raise RuntimeError("xECG SSL input preparation did not complete")
        wait_while_alive(args.wait_pid, previous,
                         partial(status, "queued", waiting_for_pid=args.wait_pid,
                                 reason="Experiment 007 xECG fine-tuning"),
                         common.POLL_SECONDS)
        check_xecg_complete()
        orchestrator.pause_when_idle(partial(status, "queued", waiting_for_pid=args.mimic_runner_pid,
                                             reason="MIMIC training active or ready"),
                                     common.POLL_SECONDS)
        common.verify_sources(args.output_dir / "provenance/sources.json")
        for stage in ("profile", "all"):
            command = [sys.executable, "-u", "-m", "scripts.experiments.run_xecg_adaptation",
                       "--stage", stage, "--device", "cuda", "--output-dir", str(args.output_dir)]
            if stage == "all":
                command.append("--resume")
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
