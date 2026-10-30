"""Exercise cold raw decoding and hash/role gates without archived scientific artifacts."""

import hashlib
import json

import h5py
import numpy as np
import pandas as pd
import pytest

from ecg_experiment.files import sha256_file
from ecg_experiment.pr74_fresh_baseline import align_roles, assign_roles, read_signal, stage_inputs
from ecg_experiment.public_sources import signal_sha256


def test_cold_ptb_decode_preserves_original_float64_scaling_without_any_cache(tmp_path):
    path = tmp_path / "data/raw/ptb-xl/1.0.3/records500/record"
    path.parent.mkdir(parents=True)
    digital = (np.arange(5000 * 12, dtype=np.int32) % 1700 - 850).astype("<i2").reshape(5000, 12)
    digital.tofile(str(path) + ".dat")
    header = path.with_suffix(".hea")
    header.write_text("synthetic fixture; raw byte verification only\n")
    expected = digital.T.astype(np.float64) / 1000
    row = pd.Series({
        "source": "ptbxl", "backend": "ptbxl_wfdb", "waveform_path": str(path.relative_to(tmp_path)),
        "strat_fold": "7", "samples": "5000", "window_start": "0",
        "signal_sha256": signal_sha256(expected), "raw_header_sha256": sha256_file(header),
        "raw_waveform_sha256": sha256_file(path.with_suffix(".dat")),
    })
    actual = read_signal(tmp_path, row)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, expected)
    row["strat_fold"] = "10"
    with pytest.raises(ValueError, match="Reserved PTB"):
        read_signal(tmp_path, row)


def test_cold_sph_decodes_first_window_and_rejects_changed_source_or_window(tmp_path):
    path = tmp_path / "data/raw/sph/record.h5"
    path.parent.mkdir(parents=True)
    stored = np.random.default_rng(4).normal(size=(12, 6000)).astype(np.float16)
    with h5py.File(path, "w") as handle:
        handle["ecg"] = stored
    expected = stored[:, :5000].astype(np.float32).astype(np.float64)
    row = pd.Series({
        "source": "sph", "backend": "sph_h5", "waveform_path": str(path.relative_to(tmp_path)),
        "samples": "5000", "window_start": "0", "signal_sha256": signal_sha256(expected),
        "raw_waveform_sha256": sha256_file(path),
    })
    np.testing.assert_array_equal(read_signal(tmp_path, row), expected)
    row["signal_sha256"] = hashlib.sha256(b"wrong canonical view").hexdigest()
    with pytest.raises(ValueError, match="canonical window"):
        read_signal(tmp_path, row)
    row["raw_waveform_sha256"] = hashlib.sha256(b"wrong file").hexdigest()
    with pytest.raises(ValueError, match="raw waveform changed"):
        read_signal(tmp_path, row)


def test_raw_role_alignment_is_lossless_and_does_not_invent_patient_identity():
    rows = pd.DataFrame({"ecg_id": ["1"], "patient_id": ["12"], "target": ["0"]})
    evidence = pd.DataFrame({"ecg_id": ["1.0"], "patient_id": ["12.0"], "target": ["0.0"]})
    matched = align_roles(rows.copy(), evidence.copy(), "ptb")
    assert matched["patient_id"].tolist() == ["12"]
    evidence.loc[0, "patient_id"] = "12.5"
    with pytest.raises(ValueError, match="integral|fractional|integer"):
        align_roles(rows.copy(), evidence, "ptb")


def test_cold_roles_block_fold_ten_and_crossed_patients_before_feature_loading():
    rows = pd.DataFrame({"patient_id": ["a", "a"], "strat_fold": ["7", "8"]})
    with pytest.raises(ValueError, match="patients cross"):
        assign_roles(rows, "ptb")
    rows["strat_fold"] = ["9", "10"]
    with pytest.raises(ValueError, match="fold 10"):
        assign_roles(rows, "ptb")


def test_installed_runtime_discovery_requires_no_manual_environment_fix(tmp_path, monkeypatch):
    from ecg_experiment import cuda_runtime20261028 as runtime

    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.setattr(runtime.sysconfig, "get_path", lambda name: str(tmp_path))
    include = tmp_path / "nvidia/cuda_runtime/include"
    include.mkdir(parents=True)
    (include / "cuda_runtime.h").write_text("synthetic runtime header")
    assert runtime.configure_cuda_runtime() == str(include.parent)
    assert runtime.os.environ["CUDA_HOME"] == str(include.parent)


def test_old_scientific_receipt_cannot_be_mislabeled_a_fresh_bootstrap(tmp_path):
    directory = tmp_path / "outputs/cold/profile"
    directory.mkdir(parents=True)
    receipt = {"status": "completed", "input_mode": "historical"}
    (directory / "completion.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="scientific cold-bootstrap"):
        stage_inputs(tmp_path, directory.relative_to(tmp_path).parent, "profile")
