"""Shared file, seeding, GPU-lock and process helpers."""

import hashlib
import json
import os
import random
import signal
import subprocess
import sys

import numpy as np
import pytest
import torch

from ecg_experiment import files, gpu, processes, reproducibility


def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(os.urandom(3 * 1024 * 1024 + 7))
    assert files.sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_sha256_json_is_key_order_independent():
    assert files.sha256_json({"b": 1, "a": [1, 2]}) == files.sha256_json({"a": [1, 2], "b": 1})
    expected = hashlib.sha256(b'{"a":[1,2],"b":1}').hexdigest()
    assert files.sha256_json({"b": 1, "a": [1, 2]}) == expected


def test_write_json_atomic_matches_previous_bytes(tmp_path):
    value = {"z": 1.5, "a": [1, {"b": None}]}
    path = tmp_path / "nested" / "value.json"
    files.write_json_atomic(path, value)
    assert path.read_text() == json.dumps(value, indent=2, allow_nan=False) + "\n"
    assert not path.with_name("value.json.tmp").exists()
    files.write_json_atomic(path, value, sort_keys=True)
    assert path.read_text() == json.dumps(value, indent=2, sort_keys=True) + "\n"


def test_write_json_atomic_rejects_nan_by_default(tmp_path):
    with pytest.raises(ValueError, match="not JSON compliant"):
        files.write_json_atomic(tmp_path / "nan.json", {"value": float("nan")})
    files.write_json_atomic(tmp_path / "nan.json", {"value": float("nan")}, allow_nan=True)


def test_csv_roundtrip(tmp_path):
    rows = [{"ecg_id": "1", "target": "0"}, {"ecg_id": "2", "target": "1"}]
    files.write_csv_atomic(tmp_path / "rows.csv", rows, ("ecg_id", "target"))
    assert files.read_csv(tmp_path / "rows.csv") == rows


def test_csv_and_text_roundtrip_non_ascii_as_utf8(tmp_path):
    rows = [{"ecg_id": "1", "note": "señal ñ µV"}]
    files.write_csv_atomic(tmp_path / "rows.csv", rows, ("ecg_id", "note"))
    assert files.read_csv(tmp_path / "rows.csv") == rows
    assert "señal".encode() in (tmp_path / "rows.csv").read_bytes()
    files.write_text_atomic(tmp_path / "note.txt", "µV")
    assert (tmp_path / "note.txt").read_bytes() == "µV".encode()


def test_write_torch_atomic_roundtrip(tmp_path):
    files.write_torch_atomic(tmp_path / "state.pt", {"weight": torch.arange(3)})
    assert torch.equal(torch.load(tmp_path / "state.pt")["weight"], torch.arange(3))


def test_rng_state_roundtrip_restores_every_generator():
    reproducibility.seed_everything(7)
    generator = torch.Generator().manual_seed(3)
    state = reproducibility.capture_rng_state(generator)
    expected = (random.random(), np.random.rand(), torch.rand(1), torch.rand(1, generator=generator))
    reproducibility.restore_rng_state(state, generator)
    actual = (random.random(), np.random.rand(), torch.rand(1), torch.rand(1, generator=generator))
    assert expected[:2] == actual[:2]
    assert torch.equal(expected[2], actual[2])
    assert torch.equal(expected[3], actual[3])


def test_cpu_state_is_an_independent_copy():
    model = torch.nn.Linear(2, 1)
    state = reproducibility.cpu_state(model)
    with torch.no_grad():
        model.weight.add_(1)
    assert not torch.equal(state["weight"], model.weight)


def test_gpu_lock_skips_cpu_and_rejects_held_lock(tmp_path):
    path = tmp_path / "gpu.lock"
    with gpu.gpu_lock("cpu", path=path):
        assert not path.exists()
    with (gpu.gpu_lock("cuda", blocking=False, path=path),
          pytest.raises(RuntimeError, match="reserved"),
          gpu.gpu_lock("cuda", blocking=False, path=path)):
        pass


def test_process_identity_and_pause_resume():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        identity = processes.process_identity(child.pid)
        assert identity is not None
        assert identity["command"][0] == sys.executable
        assert processes.runs_module({"start": "1", "command": ["python", "-m", "a.b"]}, "a.b")
        assert not processes.runs_module(None, "a.b")
        processes.stop_process(child.pid, identity)
        assert processes.resume_process(child.pid, identity)
        assert processes.process_identity(child.pid) == identity
    finally:
        processes.terminate_child(child, timeout=5)
    assert child.poll() is not None
    assert processes.process_identity(child.pid) is None
    assert not processes.resume_process(child.pid, identity)


def test_termination_deferred_restores_handlers():
    original = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        processes.interrupt_on_termination()
        before = signal.getsignal(signal.SIGTERM)
        with processes.termination_deferred():
            assert signal.getsignal(signal.SIGTERM) is signal.SIG_IGN
        assert signal.getsignal(signal.SIGTERM) is before
    finally:
        for number, handler in original.items():
            signal.signal(number, handler)
