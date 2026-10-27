"""Shared pytest fixtures with synthetic data; no private ECG data is required."""

import csv
import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch

from ecg_experiment.files import sha256_file

POOL_FIELDS = ("ecg_id", "patient_id", "source", "split", "target")


@pytest.fixture(autouse=True, scope="session")
def single_torch_thread() -> Iterator[None]:
    """Run every test with one torch thread, restoring the previous count afterwards."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def write_synthetic_cpc_pool(directory: Path, records: int = 600, seed: int = 0) -> Path:
    """
    Write a small cache with the layout expected by the CPC ``Pool``.

    Parameters
    ----------
    directory : Path
        Destination directory; it is created if missing.
    records : int
        Number of ECG records to generate.
    seed : int
        Seed for the synthetic waveforms.

    Returns
    -------
    Path
        The cache directory.
    """
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    signals = rng.standard_normal((records, 12, 2500), dtype=np.float32)
    ecg_ids = np.arange(1000, 1000 + records, dtype=np.int64)
    splits = ["train"] * (records - 40) + ["validation"] * 20 + ["test"] * 20
    np.save(directory / "signals.npy", signals)
    np.save(directory / "ecg_ids.npy", ecg_ids)
    with (directory / "rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=POOL_FIELDS)
        writer.writeheader()
        for index, (ecg_id, split) in enumerate(zip(ecg_ids, splits, strict=True)):
            writer.writerow({"ecg_id": str(ecg_id), "patient_id": f"p{index // 2}",
                             "source": "ptbxl", "split": split, "target": index % 2})
    completion = {"records": records, "signals_sha256": sha256_file(directory / "signals.npy"),
                  "rows_sha256": sha256_file(directory / "rows.csv"),
                  "ecg_ids_sha256": sha256_file(directory / "ecg_ids.npy"), "ptb_manifest_sha256": {}}
    (directory / "complete.json").write_text(json.dumps(completion))
    return directory


@pytest.fixture
def synthetic_cpc_pool(tmp_path: Path) -> Path:
    """Directory holding a synthetic CPC waveform cache."""
    return write_synthetic_cpc_pool(tmp_path / "cpc_pool")


def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    """Import the immutable audit command in a child process to isolate its thread settings.

    Parameters
    ----------
    pyfuncitem : pytest.Function
        Collected test; every other test retains pytest's normal execution path.

    Returns
    -------
    bool or None
        True after the real command import succeeds, or None for normal test execution.
    """
    module = "scripts.validation.audit_ssl_synthetic_counterfactual_factorial"
    node = f"tests/test_imports.py::test_script_module_imports[{module}]"
    if pyfuncitem.nodeid != node:
        return None
    subprocess.run(
        [sys.executable, "-c", "import importlib, sys; importlib.import_module(sys.argv[1])", module],
        cwd=Path(__file__).parents[1],
        check=True,
    )
    return True
