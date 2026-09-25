"""Run the compact CPC suite while the older MIMIC runner waits for its data.

Only the verified idle orchestrator is suspended; the downloader keeps running.
The orchestrator is resumed on success, failure, SIGINT, or SIGTERM.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

from ecg_experiment.processes import interrupt_on_termination, process_identity, wait_while_alive
from scripts.coordination import common

EXPERIMENT_MODULES = ("scripts.experiments.run_cpc_experiment", "scripts.experiments.run_cpc_word2vec",
                      "scripts.experiments.run_cpc_tokenization")
CPC_CACHE_READY = Path("data/processed/cpc_pool_40k/complete.json")
CACHE_TIMEOUT_SECONDS = 6 * 3600


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
    parser.add_argument("--waiting-runner-pid", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=common.ROOT / "outputs/experiment004_cpc_40k")
    parser.add_argument("--wait-cache", action="store_true")
    parser.add_argument("--wait-pid", type=int,
                        help=("Wait for an earlier coordinated GPU suite to exit "
                              "before suspending the MIMIC runner"))
    parser.add_argument("--experiment-module", choices=EXPERIMENT_MODULES, default=EXPERIMENT_MODULES[0])
    parser.add_argument("--ssl-epochs", type=int, default=20)
    args = parser.parse_args(argv)
    if args.ssl_epochs < 1:
        parser.error("ssl-epochs must be positive")
    return args


def wait_for_cache() -> None:
    """
    Wait for the audited CPC cache to be marked complete.

    Raises
    ------
    RuntimeError
        If the cache is not ready within ``CACHE_TIMEOUT_SECONDS``.
    """
    deadline = time.monotonic() + CACHE_TIMEOUT_SECONDS
    while not (common.ROOT / CPC_CACHE_READY).exists():
        if time.monotonic() > deadline:
            raise RuntimeError("Timed out after six hours waiting for audited CPC cache")
        print("Waiting for audited CPC cache; MIMIC download/runner unchanged", flush=True)
        time.sleep(common.POLL_SECONDS)


def main(argv: list[str] | None = None) -> None:
    """
    Wait for inputs, pause the idle MIMIC runner, then profile and run the suite.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or ``None`` for ``sys.argv``.
    """
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.wait_pid:
        wait_while_alive(args.wait_pid, process_identity(args.wait_pid),
                         lambda: print(f"Waiting for earlier ECG suite PID {args.wait_pid}", flush=True),
                         common.POLL_SECONDS)
    if args.wait_cache:
        wait_for_cache()
    status = partial(common.write_status, args.output_dir / "coordination.json",
                     waiting_runner_pid=args.waiting_runner_pid)
    orchestrator = common.mimic_orchestrator(args.waiting_runner_pid)
    profile = [sys.executable, "-u", "-m", args.experiment_module,
               "--stage", "profile", "--variant", "all", "--device", "cuda", "--threads", "1",
               "--ssl-epochs", str(args.ssl_epochs)]
    command = [sys.executable, "-u", "-m", args.experiment_module,
               "--stage", "all", "--variant", "all", "--labels", "all",
               "--device", "cuda", "--threads", "1", "--ssl-epochs", str(args.ssl_epochs),
               "--output-dir", str(args.output_dir)]
    child = None
    interrupt_on_termination()
    try:
        if not orchestrator.pause():
            raise RuntimeError("The MIMIC orchestrator is not idle waiting for its data; "
                               "explicitly coordinate its active work first")
        with (args.output_dir / "runtime_profile.log").open("a") as log:
            child = subprocess.Popen(profile, cwd=common.ROOT, stdout=log, stderr=subprocess.STDOUT)
            status("profiling", child_pid=child.pid, command=profile)
            code = child.wait()
        if code:
            status("failed", returncode=code, failed_stage="profile")
            raise SystemExit(code)
        child = subprocess.Popen(command, cwd=common.ROOT)
        status("running", child_pid=child.pid, command=command,
               note="Only waiting MIMIC orchestrator suspended; downloader continues")
        code = child.wait()
        status("complete" if code == 0 else "failed", returncode=code)
        if code:
            raise SystemExit(code)
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
