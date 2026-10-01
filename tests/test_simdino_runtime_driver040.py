"""Verify that the external diagnostic clock charges each interval once."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from ecg_experiment import simdino_runtime_probe040 as probe
from scripts.coordination import measure_simdino_runtime040 as driver


def test_external_clock_only_charges_uncovered_startup(tmp_path, monkeypatch):
    """A four-second child within six measured seconds leaves two seconds to charge."""
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    readings = iter([0.0, 1.0, 6.0])
    monkeypatch.setattr(driver.time, "monotonic", lambda: next(readings))

    def child(*args, **kwargs):
        probe.charge(tmp_path, "probe", 4.0, "complete")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(driver.subprocess, "run", child)
    driver.main()
    destination = tmp_path / "outputs" / probe.NAME
    ledger = json.loads((destination / "day_ledger.json").read_text())
    assert ledger["total_seconds"] == 6.0
    assert ledger["attempts"][-1]["elapsed_seconds"] == 2.0
    assert json.loads((destination / "process.json").read_text())["status"] == "complete"


def test_timeout_retains_and_charges_the_whole_failed_interval(tmp_path, monkeypatch):
    """Killed children still leave a failed receipt and complete wall-time accounting."""
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    readings = iter([0.0, 1.0, 1200.0])
    monkeypatch.setattr(driver.time, "monotonic", lambda: next(readings))

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("diagnostic", 1199)

    monkeypatch.setattr(driver.subprocess, "run", timeout)
    with pytest.raises(SystemExit):
        driver.main()
    destination = tmp_path / "outputs" / probe.NAME
    ledger = json.loads((destination / "day_ledger.json").read_text())
    receipt = json.loads((destination / "process.json").read_text())
    assert ledger["total_seconds"] == 1200.0
    assert ledger["attempts"][-1]["status"] == "failed"
    assert receipt["timed_out"]
