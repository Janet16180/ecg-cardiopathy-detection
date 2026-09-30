"""Measure the diagnostic process wall time, including startup and CLI overhead."""

from __future__ import annotations

import json
import subprocess
import sys
import time

from ecg_experiment import ROOT
from ecg_experiment.files import write_json_atomic


def main() -> None:
    """Run one diagnostic child and charge wall time not covered by its stage timer."""
    began = time.monotonic()
    from ecg_experiment.simdino_runtime_probe040 import NAME, charge

    destination = ROOT / "outputs" / NAME
    ledger_path = destination / "day_ledger.json"
    before = json.loads(ledger_path.read_text())["total_seconds"] if ledger_path.exists() else 0.0
    timed_out = False
    try:
        completed = subprocess.run(
            [sys.executable, "-u", "-m", "scripts.experiments.profile_simdino_runtime040"],
            cwd=ROOT, check=False, timeout=max(1.0, 1200.0 - (time.monotonic() - began)),
        )
        returncode = completed.returncode
    except subprocess.TimeoutExpired:
        returncode = -1
        timed_out = True
    elapsed = time.monotonic() - began
    after = json.loads(ledger_path.read_text())["total_seconds"] if ledger_path.exists() else before
    child_charge = after - before
    if child_charge > elapsed + 0.001:
        raise ValueError("Child stage charges exceed measured diagnostic process time")
    overhead = max(0.0, elapsed - child_charge)
    status = "complete" if returncode == 0 else "failed"
    charge(ROOT, "startup_and_cli_overhead", overhead, status)
    receipt = {
        "status": status, "returncode": returncode, "timed_out": timed_out,
        "whole_process_seconds": elapsed, "child_stage_charged_seconds": child_charge,
        "startup_and_cli_overhead_seconds": overhead, "runtime_ceiling_seconds": 1200,
    }
    write_json_atomic(destination / "process.json", receipt, sort_keys=True)
    print(json.dumps(receipt), flush=True)
    if returncode:
        raise SystemExit(returncode)


if __name__ == "__main__":
    main()
